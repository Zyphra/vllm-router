//! Optional judge-session queue feedback. Observations come from the existing census.
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

pub const SPILL_CONFIG_ENV: &str = "VLLM_ROUTER_SESSION_SPILL_CONFIG";
const MAX_AGE: f64 = 15.0;
fn default_ratio() -> f64 {
    2.0
}
fn default_minimum() -> u64 {
    32
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionSpillConfig {
    #[serde(default = "default_ratio")]
    pub waiting_ratio: f64,
    #[serde(default = "default_minimum")]
    pub minimum_waiting: u64,
}
impl Default for SessionSpillConfig {
    fn default() -> Self {
        Self {
            waiting_ratio: default_ratio(),
            minimum_waiting: default_minimum(),
        }
    }
}
impl SessionSpillConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.waiting_ratio.is_finite() || self.waiting_ratio <= 1.0 {
            return Err("spill waiting_ratio must be finite and greater than one".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueObservation {
    pub worker_url: String,
    pub observed_at: f64,
    pub waiting: u64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueSnapshot {
    pub workers: Vec<QueueObservation>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PlacementSnapshot {
    Kv(super::KvSnapshot),
    Queue(QueueSnapshot),
}

pub fn host(worker: &str) -> String {
    let origin = worker
        .rsplit_once('@')
        .filter(|(_, rank)| rank.parse::<usize>().is_ok())
        .map_or(worker, |(origin, _)| origin);
    url::Url::parse(origin)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_else(|| origin.to_owned())
}

#[derive(Debug, Default)]
pub struct QueuePressure {
    workers: HashMap<String, QueueObservation>,
}
impl QueuePressure {
    /// A complete, validated publication replaces the prior snapshot atomically.
    pub fn update(
        &mut self,
        snapshot: QueueSnapshot,
        registered: &[String],
        now: f64,
    ) -> Result<usize, String> {
        let mut next = HashMap::new();
        for row in snapshot.workers {
            if !registered.contains(&row.worker_url)
                || next.contains_key(&row.worker_url)
                || !row.observed_at.is_finite()
                || row.observed_at <= 0.0
                || row.observed_at > now + 1.0
                || now - row.observed_at > MAX_AGE
            {
                return Err("invalid, stale, duplicate or unregistered queue observation".into());
            }
            if let Some(old) = self.workers.get(&row.worker_url) {
                if row.observed_at < old.observed_at
                    || (row.observed_at == old.observed_at && row.waiting != old.waiting)
                {
                    return Err("queue observation moved backwards or conflicts".into());
                }
            }
            next.insert(row.worker_url.clone(), row);
        }
        let count = next.len();
        self.workers = next;
        Ok(count)
    }

    /// Pool median is nearest-rank over fully observed healthy HOSTS, not DP ranks.
    /// Missing/stale telemetry preserves affinity. A spill needs >=90% host coverage,
    /// the resident host and at least one different host no busier than the median.
    pub fn spill_hosts(
        &self,
        resident: &str,
        healthy: &[&str],
        config: &SessionSpillConfig,
        now: f64,
    ) -> Option<HashSet<String>> {
        let mut totals: HashMap<String, (usize, usize, u64)> = HashMap::new();
        for worker in healthy {
            let total = totals.entry(host(worker)).or_default();
            total.0 += 1;
            if let Some(row) = self
                .workers
                .get(*worker)
                .filter(|row| 0.0 <= now - row.observed_at && now - row.observed_at <= MAX_AGE)
            {
                total.1 += 1;
                total.2 = total.2.saturating_add(row.waiting);
            }
        }
        let fresh: HashMap<_, _> = totals
            .iter()
            .filter(|(_, (expected, seen, _))| expected == seen)
            .map(|(host, (_, _, waiting))| (host.clone(), *waiting))
            .collect();
        if fresh.len() < 2 || fresh.len() * 10 < totals.len() * 9 {
            return None;
        }
        let resident = host(resident);
        let waiting = *fresh.get(&resident)?;
        let mut counts: Vec<_> = fresh.values().copied().collect();
        counts.sort_unstable();
        let median =
            (counts[(counts.len() - 1) / 2] as f64 + counts[counts.len() / 2] as f64) / 2.0;
        let threshold = (config.minimum_waiting as f64).max(config.waiting_ratio * median);
        if waiting as f64 <= threshold {
            return None;
        }
        let candidates: HashSet<_> = fresh
            .into_iter()
            .filter(|(host, count)| *host != resident && *count as f64 <= median)
            .map(|(host, _)| host)
            .collect();
        if candidates.is_empty() {
            None
        } else {
            Some(candidates)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(worker: &str, waiting: u64, at: f64) -> QueueObservation {
        QueueObservation {
            worker_url: worker.into(),
            observed_at: at,
            waiting,
        }
    }
    #[test]
    fn host_median_aggregates_ranks_and_requires_fresh_coverage() {
        let registered = vec![
            "http://hot:1@0".into(),
            "http://hot:1@1".into(),
            "http://cool:2@0".into(),
            "http://middle:3@0".into(),
        ];
        let healthy: Vec<_> = registered.iter().map(String::as_str).collect();
        let mut q = QueuePressure::default();
        q.update(
            QueueSnapshot {
                workers: vec![
                    sample(&registered[0], 100, 100.0),
                    sample(&registered[1], 100, 100.0),
                    sample(&registered[2], 20, 100.0),
                    sample(&registered[3], 30, 100.0),
                ],
            },
            &registered,
            100.0,
        )
        .unwrap();
        let cfg = SessionSpillConfig::default();
        let hosts = q
            .spill_hosts(&registered[0], &healthy, &cfg, 100.0)
            .unwrap();
        assert_eq!(hosts, HashSet::from(["cool".into(), "middle".into()]));
        assert!(q
            .spill_hosts(&registered[0], &healthy, &cfg, 116.0)
            .is_none());
        q.update(
            QueueSnapshot {
                workers: vec![
                    sample(&registered[0], 100, 101.0),
                    sample(&registered[2], 20, 101.0),
                    sample(&registered[3], 30, 101.0),
                ],
            },
            &registered,
            101.0,
        )
        .unwrap();
        assert!(q
            .spill_hosts(&registered[0], &healthy, &cfg, 101.0)
            .is_none());
    }
    #[test]
    fn heterogeneous_host_census_replay_spills_only_above_twice_median() {
        // Sanitized host totals from one immutable 42-host native census.
        let waiting: [u64; 42] = [
            0, 13, 173, 1572, 36, 1043, 10, 251, 1417, 0, 663, 80, 600, 331, 158, 221, 31, 0, 472,
            673, 1059, 53, 0, 710, 6, 1170, 336, 1272, 1069, 71, 408, 1163, 358, 6, 22, 1225, 39,
            1, 25, 101, 0, 338,
        ];
        let registered: Vec<String> = (0..waiting.len())
            .map(|i| format!("http://host{i}:8000@0"))
            .collect();
        let healthy: Vec<_> = registered.iter().map(String::as_str).collect();
        let mut q = QueuePressure::default();
        q.update(
            QueueSnapshot {
                workers: registered
                    .iter()
                    .zip(waiting)
                    .map(|(w, n)| sample(w, n, 100.0))
                    .collect(),
            },
            &registered,
            100.0,
        )
        .unwrap();
        for (worker, n) in registered.iter().zip(waiting) {
            let candidates = q.spill_hosts(worker, &healthy, &SessionSpillConfig::default(), 100.0);
            assert_eq!(candidates.is_some(), n > 394);
            if let Some(candidates) = candidates {
                assert_eq!(candidates.len(), 21);
                assert!(!candidates.contains(&host(worker)));
            }
        }
    }
    #[test]
    fn validates_config_and_publications_before_mutation() {
        assert!(SessionSpillConfig {
            waiting_ratio: f64::NAN,
            minimum_waiting: 0
        }
        .validate()
        .is_err());
        assert!(SessionSpillConfig {
            waiting_ratio: 1.0,
            minimum_waiting: 0
        }
        .validate()
        .is_err());
        let mut q = QueuePressure::default();
        let registered = vec!["a".into()];
        q.update(
            QueueSnapshot {
                workers: vec![sample("a", 50, 100.0)],
            },
            &registered,
            100.0,
        )
        .unwrap();
        for rows in [
            vec![sample("a", 99, 99.0)],
            vec![sample("a", 51, 100.0)],
            vec![sample("b", 50, 101.0)],
            vec![sample("a", 50, 101.0); 2],
        ] {
            assert!(q
                .update(QueueSnapshot { workers: rows }, &registered, 101.0)
                .is_err());
            assert_eq!(q.workers["a"].waiting, 50);
        }
    }
}
