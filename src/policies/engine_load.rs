//! Shared engine-load placement. Census gauges, never token estimates, own load.
use super::{get_healthy_worker_indices, hash_key, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const MAX_AGE_SECS: f64 = 6.0;
const DEFAULT_IDLE_SECS: u64 = 900;
const DEFAULT_SPILL_FACTOR: f64 = 2.0;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineObservation {
    pub worker_url: String,
    pub observed_at: f64,
    pub running: u64,
    pub waiting: u64,
    pub kv_usage: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineSnapshot {
    pub workers: Vec<EngineObservation>,
}

#[derive(Debug)]
struct Reading {
    observation: EngineObservation,
    epoch: u64,
    // Reservations made AFTER this engine's observation. Older leases are
    // already represented by the census and must not be counted twice.
    pending: u64,
    last_selected: u64,
}

#[derive(Debug)]
struct Session {
    worker: String,
    inflight: u64,
    last_settled: Instant,
    released: bool,
}

#[derive(Debug, Default)]
struct State {
    readings: HashMap<String, Reading>,
    host_ids: HashMap<String, Arc<str>>,
    last_gc: Option<Instant>,
    watermarks: HashMap<String, EngineObservation>,
    sessions: HashMap<String, Session>,
    epoch: u64,
    sequence: u64,
}

/// The response owns this reservation until EOF, failure, cancellation or drop.
#[derive(Debug)]
pub struct SessionLease {
    state: Arc<Mutex<State>>,
    key: Option<String>,
    worker: String,
    worker_handle: Arc<dyn Worker>,
    epoch: u64,
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        self.worker_handle.decrement_load();
        let mut state = self.state.lock().unwrap();
        if let Some(reading) = state.readings.get_mut(&self.worker) {
            if reading.epoch == self.epoch {
                reading.pending = reading.pending.saturating_sub(1);
            }
        }
        if let Some(session) = self
            .key
            .as_ref()
            .and_then(|key| state.sessions.get_mut(key))
        {
            session.inflight = session.inflight.saturating_sub(1);
            if session.inflight == 0 {
                session.last_settled = Instant::now();
            }
        }
    }
}

#[derive(Debug)]
pub struct EngineLoadPolicy {
    state: Arc<Mutex<State>>,
    idle: Duration,
    spill_factor: f64,
}

fn wall_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

fn host(worker: &str) -> String {
    let origin = worker
        .rsplit_once('@')
        .filter(|(_, rank)| rank.parse::<usize>().is_ok())
        .map_or(worker, |(origin, _)| origin);
    url::Url::parse(origin)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_else(|| origin.to_owned())
}

fn abort_request(text: Option<&str>) -> bool {
    let Some(serde_json::Value::Object(body)) = text.and_then(|s| serde_json::from_str(s).ok())
    else {
        return false;
    };
    !["prompt_ids", "prompt", "messages", "input"]
        .iter()
        .any(|key| body.contains_key(*key))
        && ["request_id", "request_ids", "rid"]
            .iter()
            .any(|key| body.contains_key(*key))
}

impl Default for EngineLoadPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineLoadPolicy {
    #[cfg(test)]
    pub(crate) fn test_set_queue(&self, owner: &str, queue: u64) {
        let mut state = self.state.lock().unwrap();
        for (url, reading) in &mut state.readings {
            reading.observation.running = 0;
            reading.observation.waiting = if url == owner { queue } else { 0 };
            reading.observation.observed_at = wall_time();
        }
    }
    #[cfg(test)]
    pub(crate) fn seed_probe_sessions(&self, count: usize, workers: &[Arc<dyn Worker>]) {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        for i in 0..count {
            state.sessions.insert(
                format!("header:x-session-id:settled-{i}"),
                Session {
                    worker: workers[i % workers.len()].url().to_owned(),
                    inflight: 0,
                    last_settled: now,
                    released: false,
                },
            );
        }
    }

    pub fn new() -> Self {
        let idle = std::env::var("VLLM_ROUTER_SESSION_IDLE_SECS")
            .ok()
            .map(|s| {
                s.parse::<u64>()
                    .expect("session idle seconds must be a positive integer")
            })
            .unwrap_or(DEFAULT_IDLE_SECS);
        let spill = std::env::var("VLLM_ROUTER_SPILL_FACTOR")
            .ok()
            .map(|s| {
                s.parse::<f64>()
                    .expect("spill factor must be finite and greater than one")
            })
            .unwrap_or(DEFAULT_SPILL_FACTOR);
        Self::with_parameters(Duration::from_secs(idle), spill)
    }

    pub fn with_parameters(idle: Duration, spill_factor: f64) -> Self {
        assert!(!idle.is_zero(), "session idle seconds must be positive");
        assert!(
            spill_factor.is_finite() && spill_factor > 1.0,
            "spill factor must exceed one"
        );
        Self {
            state: Arc::new(Mutex::new(State::default())),
            idle,
            spill_factor,
        }
    }

    fn update_at(
        &self,
        snapshot: EngineSnapshot,
        registered: &[String],
        now: f64,
    ) -> Result<usize, String> {
        // Canonical host identity and membership are prepared off the placement
        // lock once per census, rather than reparsed for every generation.
        let registered_set: HashSet<&str> = registered.iter().map(String::as_str).collect();
        let host_ids: HashMap<String, Arc<str>> = registered
            .iter()
            .map(|url| (url.clone(), Arc::from(host(url))))
            .collect();
        let mut state = self.state.lock().unwrap();
        let mut seen = HashSet::new();
        for row in &snapshot.workers {
            if !registered_set.contains(row.worker_url.as_str())
                || !seen.insert(&row.worker_url)
                || !row.observed_at.is_finite()
                || row.observed_at <= 0.0
                || row.observed_at > now + 1.0
                || now - row.observed_at > MAX_AGE_SECS
                || !row.kv_usage.is_finite()
                || !(0.0..=1.0).contains(&row.kv_usage)
            {
                return Err("invalid, duplicate, stale or unregistered engine observation".into());
            }
            if let Some(old) = state.watermarks.get(&row.worker_url) {
                if row.observed_at < old.observed_at
                    || (row.observed_at == old.observed_at && *row != *old)
                {
                    return Err("engine observation moved backwards or conflicts".into());
                }
            }
        }
        state
            .watermarks
            .retain(|url, _| registered_set.contains(url.as_str()));
        state.host_ids = host_ids;
        for row in &snapshot.workers {
            state.watermarks.insert(row.worker_url.clone(), row.clone());
        }
        let mut next = HashMap::new();
        for row in snapshot.workers {
            let old = state.readings.remove(&row.worker_url);
            let reading = match old {
                Some(old) if old.observation == row => old,
                old => {
                    state.epoch = state.epoch.checked_add(1).expect("census epoch overflow");
                    Reading {
                        observation: row.clone(),
                        epoch: state.epoch,
                        pending: 0,
                        last_selected: old.map_or(0, |r| r.last_selected),
                    }
                }
            };
            next.insert(row.worker_url, reading);
        }
        let count = next.len();
        state.readings = next;
        Ok(count)
    }

    fn choose(
        &self,
        workers: &[Arc<dyn Worker>],
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
        reserve: bool,
        instant: Instant,
        wall: f64,
    ) -> Option<(usize, Option<SessionLease>)> {
        self.choose_kind(
            workers,
            text,
            headers,
            reserve,
            instant,
            wall,
            abort_request(text),
        )
    }
    fn choose_kind(
        &self,
        workers: &[Arc<dyn Worker>],
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
        reserve: bool,
        instant: Instant,
        wall: f64,
        abort: bool,
    ) -> Option<(usize, Option<SessionLease>)> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        let key = headers
            .and_then(hash_key::extract_hash_key_from_headers)
            .or_else(|| hash_key::extract_hash_key_from_body(text));
        let selected = {
            let mut state = self.state.lock().unwrap();
            self.choose_locked(
                &mut state, workers, &healthy, key, reserve, instant, wall, abort,
            )
        };
        if let Some((idx, _)) = &selected {
            workers[*idx].increment_processed();
            RouterMetrics::record_processed_request(workers[*idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[*idx].url());
        }
        selected
    }

    fn choose_locked(
        &self,
        state: &mut State,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        key: Option<String>,
        reserve: bool,
        instant: Instant,
        wall: f64,
        abort: bool,
    ) -> Option<(usize, Option<SessionLease>)> {
        // Preserve the established abort-retention period. Only idle owners can
        // expire; after one idle period a generation loses affinity, while an
        // abort can still find its exact owner for four idle periods.
        let retention = self.idle.saturating_mul(4);
        let gc_period = self.idle.min(Duration::from_secs(30));
        if state
            .last_gc
            .is_none_or(|last| instant.saturating_duration_since(last) >= gc_period)
        {
            state.sessions.retain(|_, session| {
                session.inflight > 0
                    || instant.saturating_duration_since(session.last_settled) <= retention
            });
            state.last_gc = Some(instant);
        }
        // Expire the addressed key exactly even between global sweeps. Inflight
        // owners still never expire, and a late abort cannot reach an old owner.
        if let Some(key) = &key {
            if state.sessions.get(key).is_some_and(|session| {
                session.inflight == 0
                    && instant.saturating_duration_since(session.last_settled) > retention
            }) {
                state.sessions.remove(key);
            }
        }
        let resident = key.as_ref().and_then(|key| state.sessions.get(key));
        let owner = resident.and_then(|s| {
            healthy
                .iter()
                .copied()
                .find(|&i| workers[i].url() == s.worker)
        });
        let keep_affinity = resident.is_some_and(|session| {
            !session.released
                && instant.saturating_duration_since(session.last_settled) <= self.idle
        });
        // A failed/unhealthy owner is NEVER replaced while outstanding, nor on
        // abort. Returning no route is safer than aborting an unrelated engine.
        if resident.is_some_and(|s| s.inflight > 0) || abort {
            let idx = owner?;
            return self.reserve_locked(state, workers, idx, key, reserve, !abort, false, instant);
        }

        let mut hosts: HashMap<Arc<str>, (usize, usize)> = HashMap::new();
        let mut fresh = Vec::new();
        for &idx in healthy {
            let url = workers[idx].url();
            let host_id = match state.host_ids.get(url) {
                Some(value) => Arc::clone(value),
                None => {
                    // An immediately hot-added worker may precede its first
                    // census; cache its identity once, without inventing load.
                    let value: Arc<str> = Arc::from(host(url));
                    state.host_ids.insert(url.to_owned(), Arc::clone(&value));
                    value
                }
            };
            let counts = hosts.entry(Arc::clone(&host_id)).or_default();
            counts.0 += 1;
            if let Some(reading) = state.readings.get(workers[idx].url()).filter(|r| {
                wall >= r.observation.observed_at
                    && wall - r.observation.observed_at <= MAX_AGE_SECS
            }) {
                counts.1 += 1;
                fresh.push((
                    idx,
                    reading
                        .observation
                        .running
                        .saturating_add(reading.observation.waiting)
                        .saturating_add(reading.pending),
                    reading.observation.kv_usage,
                    reading.last_selected,
                    host_id,
                ));
            }
        }
        let complete: HashSet<_> = hosts
            .iter()
            .filter(|(_, (expected, seen))| expected == seen)
            .map(|(host, _)| host.clone())
            .collect();
        if complete.is_empty() || complete.len() * 10 < hosts.len() * 9 {
            // Missing feedback cannot introduce a new blind placement. Existing
            // sessions retain their known owner until fresh feedback returns.
            let idx = owner?;
            return self.reserve_locked(state, workers, idx, key, reserve, true, false, instant);
        }
        fresh.retain(|(_, _, _, _, host_id)| complete.contains(host_id));
        if fresh.is_empty() {
            return None;
        }
        let mut loads: Vec<_> = fresh.iter().map(|(_, load, _, _, _)| *load).collect();
        loads.sort_unstable();
        let median = (loads[(loads.len() - 1) / 2] as f64 + loads[loads.len() / 2] as f64) / 2.0;
        let keep = keep_affinity
            && owner.is_some_and(|idx| {
                fresh.iter().any(|(i, load, _, _, _)| {
                    *i == idx && (*load == 0 || (*load as f64) < self.spill_factor * median)
                })
            });
        let idx = if keep {
            owner.unwrap()
        } else {
            fresh
                .iter()
                .min_by(|a, b| {
                    a.1.cmp(&b.1)
                        .then_with(|| a.2.total_cmp(&b.2))
                        .then_with(|| a.3.cmp(&b.3))
                })
                .unwrap()
                .0
        };
        self.reserve_locked(state, workers, idx, key, reserve, true, true, instant)
    }

    fn reserve_locked(
        &self,
        state: &mut State,
        workers: &[Arc<dyn Worker>],
        idx: usize,
        key: Option<String>,
        reserve: bool,
        inference: bool,
        placement_selected: bool,
        instant: Instant,
    ) -> Option<(usize, Option<SessionLease>)> {
        let worker = workers[idx].url().to_owned();
        state.sequence = state.sequence.wrapping_add(1);
        let sequence = state.sequence;
        let epoch = if let Some(reading) = state.readings.get_mut(&worker) {
            reading.last_selected = sequence;
            if reserve && inference {
                reading.pending = reading.pending.saturating_add(1);
            }
            if inference {
                reading.epoch
            } else {
                0
            }
        } else {
            0
        };
        if let Some(key) = &key {
            let session = state.sessions.entry(key.clone()).or_insert(Session {
                worker: worker.clone(),
                inflight: 0,
                last_settled: instant,
                released: false,
            });
            assert!(
                session.inflight == 0 || session.worker == worker,
                "cannot move an outstanding session"
            );
            session.worker = worker.clone();
            if placement_selected {
                session.released = false;
            }
            if reserve {
                session.inflight += 1;
            }
        }
        if reserve {
            workers[idx].increment_load();
        }
        let lease = reserve.then(|| SessionLease {
            state: Arc::clone(&self.state),
            key,
            worker,
            worker_handle: Arc::clone(&workers[idx]),
            epoch,
        });
        Some((idx, lease))
    }
}

impl LoadBalancingPolicy for EngineLoadPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.choose_kind(
            workers,
            text,
            headers,
            false,
            Instant::now(),
            wall_time(),
            false,
        )
        .map(|x| x.0)
    }
    fn select_worker_with_lease(
        &self,
        workers: &[Arc<dyn Worker>],
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, Option<SessionLease>)> {
        self.choose_kind(
            workers,
            text,
            headers,
            true,
            Instant::now(),
            wall_time(),
            false,
        )
    }
    fn select_generation_worker_with_lease(
        &self,
        workers: &[Arc<dyn Worker>],
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, Option<SessionLease>)> {
        self.choose_kind(
            workers,
            text,
            headers,
            true,
            Instant::now(),
            wall_time(),
            false,
        )
    }
    fn select_control_worker_with_lease(
        &self,
        workers: &[Arc<dyn Worker>],
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, Option<SessionLease>)> {
        self.choose_kind(
            workers,
            text,
            headers,
            true,
            Instant::now(),
            wall_time(),
            true,
        )
    }
    fn update_engine_snapshot(
        &self,
        snapshot: EngineSnapshot,
        registered: &[String],
    ) -> Result<usize, String> {
        self.update_at(snapshot, registered, wall_time())
    }
    fn unselected_is_new_request(
        &self,
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> bool {
        if abort_request(text) {
            return false;
        }
        let key = headers
            .and_then(hash_key::extract_hash_key_from_headers)
            .or_else(|| hash_key::extract_hash_key_from_body(text));
        let state = self.state.lock().unwrap();
        key.as_ref()
            .and_then(|key| state.sessions.get(key))
            .map_or(true, |s| s.inflight == 0)
    }
    fn release_session_owners(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        for session in state.sessions.values_mut() {
            session.released = true;
        }
        state.sessions.len()
    }
    fn session_placement(&self) -> &'static str {
        "engine_load"
    }
    fn name(&self) -> &'static str {
        "engine_load"
    }
    fn needs_request_text(&self) -> bool {
        true
    }
    fn needs_headers(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    fn workers() -> Vec<Arc<dyn Worker>> {
        ["http://a:8000@0", "http://b:8000@0"]
            .iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.to_string(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect()
    }
    fn headers() -> RequestHeaders {
        HashMap::from([("x-session-id".to_owned(), "episode".to_owned())])
    }
    #[test]
    fn addressed_expiry_is_exact_between_bounded_sweeps_and_inflight_never_expires() {
        let p = EngineLoadPolicy::with_parameters(Duration::from_secs(10), 2.0);
        let w = workers();
        let h = headers();
        let now = Instant::now();
        {
            let mut state = p.state.lock().unwrap();
            state.last_gc = Some(now);
            for (key, age, inflight) in [
                ("episode", 41, 0),
                ("unaddressed", 41, 0),
                ("retained", 35, 0),
                ("outstanding", 100, 1),
            ] {
                state.sessions.insert(
                    format!("header:x-session-id:{key}"),
                    Session {
                        worker: w[0].url().to_owned(),
                        inflight,
                        last_settled: now - Duration::from_secs(age),
                        released: false,
                    },
                );
            }
        }
        let abort = Some(r#"{"request_ids":["r"]}"#);
        assert!(p.choose(&w, abort, Some(&h), false, now, 100.0).is_none());
        // The global sweep did not run merely because a request arrived.
        assert!(p
            .state
            .lock()
            .unwrap()
            .sessions
            .contains_key("header:x-session-id:unaddressed"));
        for key in ["retained", "outstanding"] {
            let headers = HashMap::from([("x-session-id".into(), key.into())]);
            assert_eq!(
                p.choose(&w, abort, Some(&headers), false, now, 100.0)
                    .unwrap()
                    .0,
                0
            );
        }
        // A periodic sweep clears only settled expired owners.
        p.choose(&w, None, None, false, now + Duration::from_secs(10), 100.0);
        let state = p.state.lock().unwrap();
        assert!(!state
            .sessions
            .contains_key("header:x-session-id:unaddressed"));
        assert!(!state.sessions.contains_key("header:x-session-id:retained"));
        assert_eq!(
            state.sessions["header:x-session-id:outstanding"].inflight,
            1
        );
    }
    fn publish(p: &EngineLoadPolicy, w: &[Arc<dyn Worker>], queues: &[u64], kv: &[f64], time: f64) {
        p.update_at(
            EngineSnapshot {
                workers: w
                    .iter()
                    .enumerate()
                    .map(|(i, w)| EngineObservation {
                        worker_url: w.url().to_owned(),
                        observed_at: time,
                        running: queues[i],
                        waiting: 0,
                        kv_usage: kv[i],
                    })
                    .collect(),
            },
            &w.iter().map(|w| w.url().to_owned()).collect::<Vec<_>>(),
            time,
        )
        .unwrap();
    }
    #[test]
    fn reset_release_survives_overlap_abort_and_stale_retention() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let h = headers();
        publish(&p, &w, &[0, 0], &[0.2, 0.2], 100.0);
        let first = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                100.0,
            )
            .unwrap();
        p.release_session_owners();
        let overlap = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                100.0,
            )
            .unwrap();
        assert_eq!(overlap.0, first.0);
        drop(overlap);
        let abort = p
            .choose(
                &w,
                Some(r#"{"request_ids":["r"]}"#),
                Some(&h),
                true,
                Instant::now(),
                100.0,
            )
            .unwrap();
        drop(abort);
        assert!(p
            .state
            .lock()
            .unwrap()
            .sessions
            .values()
            .all(|s| s.released));
        let original = first.0;
        drop(first);
        let retained = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                110.0,
            )
            .unwrap();
        drop(retained);
        assert!(p
            .state
            .lock()
            .unwrap()
            .sessions
            .values()
            .all(|s| s.released));
        publish(&p, &w, &[0, 0], &[0.2, 0.2], 111.0);
        let next = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                111.0,
            )
            .unwrap();
        assert_ne!(next.0, original);
    }
    #[test]
    fn omitted_observation_cannot_erase_monotonic_watermark() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        publish(&p, &w, &[0, 0], &[0.2, 0.2], 100.0);
        let registered = w.iter().map(|w| w.url().to_owned()).collect::<Vec<_>>();
        p.update_at(EngineSnapshot { workers: vec![] }, &registered, 101.0)
            .unwrap();
        let row = EngineObservation {
            worker_url: w[0].url().to_owned(),
            observed_at: 99.0,
            running: 0,
            waiting: 0,
            kv_usage: 0.2,
        };
        assert!(p
            .update_at(EngineSnapshot { workers: vec![row] }, &registered, 101.0)
            .is_err());
    }

    #[test]
    fn measured_pool_snapshot_replay() {
        let Ok(path) = std::env::var("ROUTER_REPLAY_FILE") else {
            return;
        };
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut w: Vec<Arc<dyn Worker>> = Vec::new();
        let mut rows = Vec::new();
        for host in data["observations"].as_array().unwrap() {
            for (rank, running) in host["running"].as_object().unwrap() {
                let url = format!("{}@{}", host["origin"].as_str().unwrap(), rank);
                rows.push(EngineObservation {
                    worker_url: url.clone(),
                    observed_at: 100.0,
                    running: running.as_f64().unwrap() as u64,
                    waiting: host["waiting"][rank].as_f64().unwrap() as u64,
                    kv_usage: host["kv"][rank].as_f64().unwrap(),
                });
                w.push(Arc::new(BasicWorker::new(url, WorkerType::Regular)));
            }
        }
        let p = EngineLoadPolicy::new();
        p.update_at(
            EngineSnapshot {
                workers: rows.clone(),
            },
            &w.iter().map(|w| w.url().to_owned()).collect::<Vec<_>>(),
            100.0,
        )
        .unwrap();
        let minimum = rows.iter().map(|r| r.running + r.waiting).min().unwrap();
        let first = p
            .choose(&w, None, None, true, Instant::now(), 100.0)
            .unwrap();
        assert_eq!(rows[first.0].running + rows[first.0].waiting, minimum);
        let mut leases = vec![first.1];
        let mut assigned = vec![0u64; w.len()];
        assigned[first.0] += 1;
        for _ in 1..4096 {
            let (idx, lease) = p
                .choose(&w, None, None, true, Instant::now(), 100.0)
                .unwrap();
            assigned[idx] += 1;
            leases.push(lease);
        }
        let max_source = rows
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.running + r.waiting)
            .unwrap()
            .0;
        assert_eq!(
            assigned[max_source], 0,
            "must not add calls to hottest measured engine"
        );
        println!("MEASURED_POOL_REPLAY ranks={} calls=4096 initial_min_queue={} hottest_queue={} hottest_new_calls={} body_or_model_calls=0",w.len(),minimum,rows[max_source].running+rows[max_source].waiting,assigned[max_source]);
        drop(leases);
    }

    #[test]
    fn actual_queue_precedes_kv_and_headerless_reservations_are_atomic() {
        let p = EngineLoadPolicy::with_parameters(Duration::from_secs(900), 2.0);
        let w = workers();
        publish(&p, &w, &[100, 0], &[0.1, 0.9], 100.0);
        let (idx, lease) = p
            .choose(&w, None, None, true, Instant::now(), 100.0)
            .unwrap();
        assert_eq!(idx, 1);
        drop(lease);
        publish(&p, &w, &[0, 0], &[0.2, 0.2], 101.0);
        let a = p
            .choose(&w, None, None, true, Instant::now(), 101.0)
            .unwrap();
        let b = p
            .choose(&w, None, None, true, Instant::now(), 101.0)
            .unwrap();
        assert_ne!(a.0, b.0);
        drop(a);
        drop(b);
        publish(&p, &w, &[1, 1], &[0.9, 0.1], 102.0);
        assert_eq!(
            p.choose(&w, None, None, false, Instant::now(), 102.0)
                .unwrap()
                .0,
            1
        );
    }
    #[test]
    fn sticky_spill_waits_for_settlement_and_abort_never_moves() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let h = headers();
        publish(&p, &w, &[0, 10], &[0.2, 0.2], 100.0);
        let first = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                100.0,
            )
            .unwrap();
        assert_eq!(first.0, 0);
        publish(&p, &w, &[100, 0], &[0.2, 0.2], 101.0);
        let abort = p
            .choose(
                &w,
                Some(r#"{"request_ids":["r"]}"#),
                Some(&h),
                true,
                Instant::now(),
                101.0,
            )
            .unwrap();
        assert_eq!(abort.0, 0);
        drop(first);
        let overlap = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                101.0,
            )
            .unwrap();
        assert_eq!(overlap.0, 0);
        drop(overlap);
        drop(abort);
        assert_eq!(
            p.choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                101.0
            )
            .unwrap()
            .0,
            1
        );
    }
    #[test]
    fn health_loss_cannot_move_outstanding_session_or_abort() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let h = headers();
        publish(&p, &w, &[0, 10], &[0.1, 0.1], 100.0);
        let first = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                100.0,
            )
            .unwrap();
        let remaining = vec![Arc::clone(&w[1])];
        assert!(p
            .choose(
                &remaining,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                100.0
            )
            .is_none());
        assert!(p
            .choose(
                &remaining,
                Some(r#"{"request_id":"r"}"#),
                Some(&h),
                true,
                Instant::now(),
                100.0
            )
            .is_none());
        drop(first);
        assert!(p
            .choose(
                &remaining,
                Some(r#"{"request_id":"r"}"#),
                Some(&h),
                true,
                Instant::now(),
                100.0
            )
            .is_none());
        assert_eq!(
            p.choose(
                &remaining,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                100.0
            )
            .unwrap()
            .0,
            0
        );
    }
    #[test]
    fn expiry_starts_after_last_settlement_and_keeps_late_abort_owner() {
        let p = EngineLoadPolicy::with_parameters(Duration::from_secs(1), 2.0);
        let w = workers();
        let h = headers();
        publish(&p, &w, &[0, 0], &[0.2, 0.2], 100.0);
        let start = Instant::now();
        let first = p
            .choose(&w, Some("generation"), Some(&h), true, start, 100.0)
            .unwrap();
        p.release_session_owners();
        publish(&p, &w, &[100, 0], &[0.2, 0.2], 101.0);
        let still = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                start + Duration::from_secs(20),
                101.0,
            )
            .unwrap();
        assert_eq!(still.0, first.0);
        drop(still);
        drop(first);
        let late = p
            .choose(
                &w,
                Some(r#"{"request_id":"r"}"#),
                Some(&h),
                true,
                Instant::now() + Duration::from_secs(2),
                101.0,
            )
            .unwrap();
        assert_eq!(late.0, 0);
        drop(late);
        assert_eq!(
            p.choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now() + Duration::from_secs(2),
                101.0
            )
            .unwrap()
            .0,
            1
        );
    }
    #[test]
    fn stale_quorum_holds_new_placement_and_rejects_bad_publication_atomically() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        publish(&p, &w, &[1, 2], &[0.1, 0.2], 100.0);
        assert!(p
            .choose(&w, None, None, true, Instant::now(), 107.0)
            .is_none());
        let bad = EngineObservation {
            worker_url: w[0].url().to_owned(),
            observed_at: 99.0,
            running: 0,
            waiting: 0,
            kv_usage: 0.0,
        };
        assert!(p
            .update_at(
                EngineSnapshot { workers: vec![bad] },
                &w.iter().map(|w| w.url().to_owned()).collect::<Vec<_>>(),
                100.0
            )
            .is_err());
        assert_eq!(
            p.choose(&w, None, None, true, Instant::now(), 100.0)
                .unwrap()
                .0,
            0
        );
    }
    #[test]
    fn repeated_snapshot_does_not_erase_pending_and_old_leases_do_not_double_count() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        publish(&p, &w, &[0, 0], &[0.1, 0.1], 100.0);
        let a = p
            .choose(&w, None, None, true, Instant::now(), 100.0)
            .unwrap();
        publish(&p, &w, &[0, 0], &[0.1, 0.1], 100.0);
        let b = p
            .choose(&w, None, None, true, Instant::now(), 100.0)
            .unwrap();
        assert_ne!(a.0, b.0);
        publish(&p, &w, &[1, 1], &[0.1, 0.1], 101.0);
        drop(a);
        drop(b);
        assert!(p
            .state
            .lock()
            .unwrap()
            .readings
            .values()
            .all(|r| r.pending == 0));
    }
    #[test]
    fn heterogeneous_hosts_require_complete_fresh_host_coverage() {
        let p = EngineLoadPolicy::new();
        let w: Vec<Arc<dyn Worker>> = (0..10)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    if i < 8 {
                        format!("http://a:8000@{i}")
                    } else {
                        format!("http://b:8000@{}", i - 8)
                    },
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        publish(
            &p,
            &w,
            &[100, 100, 100, 100, 100, 100, 100, 100, 0, 0],
            &[0.1; 10],
            100.0,
        );
        assert!(
            p.choose(&w, None, None, true, Instant::now(), 100.0)
                .unwrap()
                .0
                >= 8
        );
        let rows = w
            .iter()
            .skip(1)
            .map(|w| EngineObservation {
                worker_url: w.url().into(),
                observed_at: 101.0,
                running: 0,
                waiting: 0,
                kv_usage: 0.1,
            })
            .collect();
        p.update_at(
            EngineSnapshot { workers: rows },
            &w.iter().map(|w| w.url().into()).collect::<Vec<_>>(),
            101.0,
        )
        .unwrap();
        assert!(p
            .choose(&w, None, None, true, Instant::now(), 101.0)
            .is_none());
    }
}
