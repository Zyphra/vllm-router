//! Native engine KV observations supplied by the existing control-plane census.
//! No worker metrics are fetched from the routing path.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

pub const MAX_AGE_SECS: f64 = 15.0;
pub const MAX_USAGE: f64 = 0.80;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvObservation {
    pub worker_url: String,
    pub observed_at: f64,
    pub kv_usage: f64,
    pub waiting: u64,
    pub capacity_tokens: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvSnapshot {
    pub workers: Vec<KvObservation>,
}

#[derive(Debug, Clone)]
struct Capacity {
    observation: KvObservation,
    // Existing counted session load at this scrape. New admissions reserve in
    // that same load map; this is an observation baseline, not a second ledger.
    baseline_tokens: u64,
    present: bool,
}

#[derive(Debug, Default)]
pub struct KvCapacity {
    workers: HashMap<String, Capacity>,
}

impl KvCapacity {
    /// Validate the entire update before changing any state. Missing rows remain
    /// unknown; repeated observations must not erase admissions since that scrape.
    pub fn update(
        &mut self,
        snapshot: KvSnapshot,
        registered: &[String],
        now: f64,
        counted_tokens: impl Fn(&str) -> u64,
    ) -> Result<usize, String> {
        let mut next = HashMap::new();
        let mut seen = HashSet::new();
        for row in snapshot.workers {
            if !registered.contains(&row.worker_url)
                || !seen.insert(row.worker_url.clone())
                || !row.observed_at.is_finite()
                || row.observed_at <= 0.0
                || row.observed_at > now + 1.0
                || now - row.observed_at > MAX_AGE_SECS
                || !row.kv_usage.is_finite()
                || !(0.0..=1.0).contains(&row.kv_usage)
                || row.capacity_tokens == 0
            {
                return Err("invalid, stale, duplicate or unregistered KV observation".into());
            }
            let baseline_tokens = if let Some(old) = self.workers.get(&row.worker_url) {
                if row.observed_at < old.observation.observed_at {
                    return Err("KV observation moved backwards".into());
                }
                if row.observed_at == old.observation.observed_at {
                    if row.kv_usage != old.observation.kv_usage
                        || row.waiting != old.observation.waiting
                        || row.capacity_tokens != old.observation.capacity_tokens
                    {
                        return Err("conflicting repeated KV observation".into());
                    }
                    old.baseline_tokens
                } else {
                    counted_tokens(&row.worker_url)
                }
            } else {
                counted_tokens(&row.worker_url)
            };
            next.insert(
                row.worker_url.clone(),
                Capacity {
                    observation: row,
                    baseline_tokens,
                    present: true,
                },
            );
        }
        let count = next.len();
        // Keep the prior baseline/watermark for an omitted row so a later replay of the
        // same scrape cannot erase reservations. It remains ineligible meanwhile.
        for (worker, old) in &self.workers {
            if registered.contains(worker) && !next.contains_key(worker) {
                let mut missing = old.clone();
                missing.present = false;
                next.insert(worker.clone(), missing);
            }
        }
        self.workers = next;
        Ok(count)
    }

    /// Free tokens after reserving this prompt. Unknown/stale/at-limit rows are
    /// ineligible, including a prompt that would cross the selected limit.
    pub fn free_after(
        &self,
        worker: &str,
        prompt: u64,
        now: f64,
        counted_tokens: u64,
    ) -> Option<u64> {
        let row = self.workers.get(worker)?;
        if !row.present || row.observation.waiting > 0 {
            return None;
        }
        let observation = &row.observation;
        if now - observation.observed_at > MAX_AGE_SECS || now < observation.observed_at - 1.0 {
            return None;
        }
        let used = (observation.kv_usage * observation.capacity_tokens as f64).ceil() as u64;
        let additional = counted_tokens.saturating_sub(row.baseline_tokens);
        let projected = used.saturating_add(additional).saturating_add(prompt);
        if projected as f64 >= MAX_USAGE * observation.capacity_tokens as f64 {
            return None;
        }
        Some(observation.capacity_tokens.saturating_sub(projected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(worker: &str, usage: f64, capacity: u64, observed_at: f64) -> KvObservation {
        KvObservation {
            worker_url: worker.into(),
            observed_at,
            kv_usage: usage,
            waiting: 0,
            capacity_tokens: capacity,
        }
    }

    #[test]
    fn rejects_unknown_stale_invalid_and_duplicate_rows_atomically() {
        let mut capacity = KvCapacity::default();
        let registered = vec!["a@0".into()];
        capacity
            .update(
                KvSnapshot {
                    workers: vec![row("a@0", 0.5, 1000, 100.0)],
                },
                &registered,
                100.0,
                |_| 0,
            )
            .unwrap();
        for bad in [
            row("b@0", 0.1, 1000, 100.0),
            row("a@0", f64::NAN, 1000, 100.0),
            row("a@0", 0.1, 0, 100.0),
            row("a@0", 0.1, 1000, 80.0),
            row("a@0", 0.1, 1000, 102.0),
        ] {
            assert!(capacity
                .update(KvSnapshot { workers: vec![bad] }, &registered, 100.0, |_| 0)
                .is_err());
            assert_eq!(capacity.free_after("a@0", 0, 100.0, 0), Some(500));
        }
        assert!(capacity
            .update(
                KvSnapshot {
                    workers: vec![row("a@0", 0.1, 1000, 100.0); 2]
                },
                &registered,
                100.0,
                |_| 0
            )
            .is_err());
    }

    #[test]
    fn reservation_prevents_oversubscription_and_duplicate_reset() {
        let mut capacity = KvCapacity::default();
        let registered = vec!["a@0".into()];
        let snapshot = || KvSnapshot {
            workers: vec![row("a@0", 0.75, 1000, 100.0)],
        };
        capacity
            .update(snapshot(), &registered, 100.0, |_| 0)
            .unwrap();
        assert_eq!(capacity.free_after("a@0", 49, 100.0, 0), Some(201));
        assert_eq!(capacity.free_after("a@0", 50, 100.0, 0), None);
        capacity
            .update(snapshot(), &registered, 100.0, |_| 0)
            .unwrap();
        assert_eq!(capacity.free_after("a@0", 1, 100.0, 49), None);
        assert_eq!(capacity.free_after("a@0", 0, 116.0, 49), None);
        capacity
            .update(
                KvSnapshot {
                    workers: vec![row("a@0", 0.7, 1000, 101.0)],
                },
                &registered,
                101.0,
                |_| 49,
            )
            .unwrap();
        assert_eq!(capacity.free_after("a@0", 0, 101.0, 49), Some(300));
        let mut queued = row("a@0", 0.5, 1000, 102.0);
        queued.waiting = 1;
        capacity
            .update(
                KvSnapshot {
                    workers: vec![queued],
                },
                &registered,
                102.0,
                |_| 49,
            )
            .unwrap();
        assert_eq!(capacity.free_after("a@0", 0, 102.0, 49), None);
    }
}
