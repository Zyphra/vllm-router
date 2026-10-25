//! Shared placement by router-owned bound context bytes; telemetry is optional.
use super::{get_healthy_worker_indices, hash_key, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
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
    context_bytes: u64,
    context_revision: u64,
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
    last_selected: HashMap<String, u64>,
    context_bytes: HashMap<String, u128>,
    context_expiry: BinaryHeap<Reverse<(Instant, u64, String)>>,
}

impl State {
    fn replace_context(&mut self, worker: &str, old: u64, new: u64) {
        let total = self.context_bytes.entry(worker.to_owned()).or_default();
        *total = total.saturating_sub(u128::from(old)) + u128::from(new);
    }

    fn expire_contexts(&mut self, now: Instant) {
        // A deadline queue avoids a full session scan on the request path.
        // Old entries are harmless after another request or a weight reset.
        while self
            .context_expiry
            .peek()
            .is_some_and(|Reverse((deadline, _, _))| *deadline < now)
        {
            let Reverse((_, revision, key)) = self.context_expiry.pop().unwrap();
            let expired = self.sessions.get_mut(&key).and_then(|session| {
                (session.inflight == 0 && session.context_revision == revision).then(|| {
                    let bytes = std::mem::take(&mut session.context_bytes);
                    (session.worker.clone(), bytes)
                })
            });
            if let Some((worker, bytes)) = expired {
                self.replace_context(&worker, bytes, 0);
            }
        }
    }
}

/// The response owns this reservation until EOF, failure, cancellation or drop.
#[derive(Debug)]
pub struct SessionLease {
    state: Arc<Mutex<State>>,
    key: Option<String>,
    worker: String,
    worker_handle: Arc<dyn Worker>,
    epoch: u64,
    transient_bytes: u64,
    idle: Duration,
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        self.worker_handle.decrement_load();
        let mut state = self.state.lock().unwrap();
        state.replace_context(&self.worker, self.transient_bytes, 0);
        if let Some(reading) = state.readings.get_mut(&self.worker) {
            if reading.epoch == self.epoch {
                reading.pending = reading.pending.saturating_sub(1);
            }
        }
        let mut released_context = None;
        let mut expiry = None;
        if let Some(session) = self
            .key
            .as_ref()
            .and_then(|key| state.sessions.get_mut(key))
        {
            session.inflight = session.inflight.saturating_sub(1);
            if session.inflight == 0 {
                session.last_settled = Instant::now();
                session.context_revision = session.context_revision.wrapping_add(1);
                if session.released {
                    released_context = Some((
                        session.worker.clone(),
                        std::mem::take(&mut session.context_bytes),
                    ));
                } else if session.context_bytes > 0 {
                    expiry = session.last_settled.checked_add(self.idle).map(|deadline| {
                        Reverse((
                            deadline,
                            session.context_revision,
                            self.key.clone().unwrap(),
                        ))
                    });
                }
            }
        }
        if let Some((worker, bytes)) = released_context {
            state.replace_context(&worker, bytes, 0);
        }
        if let Some(expiry) = expiry {
            state.context_expiry.push(expiry);
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
    pub(crate) fn test_hold_requests(
        &self,
        workers: &[Arc<dyn Worker>],
        owner: &str,
        count: usize,
    ) -> Vec<SessionLease> {
        let idx = workers.iter().position(|w| w.url() == owner).unwrap();
        let mut state = self.state.lock().unwrap();
        (0..count)
            .map(|_| {
                self.reserve_locked(
                    &mut state,
                    workers,
                    idx,
                    None,
                    true,
                    true,
                    false,
                    Instant::now(),
                    None,
                )
                .unwrap()
                .1
                .unwrap()
            })
            .collect()
    }
    #[cfg(test)]
    pub(crate) fn seed_probe_sessions(&self, count: usize, workers: &[Arc<dyn Worker>]) {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        let mut workers = workers.to_vec();
        workers.sort_by(|a, b| a.url().cmp(b.url()));
        for i in 0..count {
            state.sessions.insert(
                format!("header:x-session-id:settled-{i}"),
                Session {
                    worker: workers[i % workers.len()].url().to_owned(),
                    inflight: 0,
                    last_settled: now,
                    released: false,
                    context_bytes: 0,
                    context_revision: 0,
                },
            );
        }
        if let Ok(path) = std::env::var("ROUTER_FAN_IN_STATE_OUTPUT") {
            let mut normalized = Vec::new();
            let mut actual = Vec::new();
            for (key, session) in &state.sessions {
                let rank = session.worker.rsplit_once('@').unwrap().1;
                normalized.push((
                    key,
                    format!("{}@{rank}", host(&session.worker)),
                    session.inflight,
                    session.released,
                ));
                actual.push((key, &session.worker, session.inflight, session.released));
            }
            normalized.sort_by(|a, b| a.0.cmp(b.0));
            actual.sort_by(|a, b| a.0.cmp(b.0));
            std::fs::write(path, serde_json::to_vec(&serde_json::json!({
                "normalized_sessions": normalized,
                "actual_sessions": actual,
                "last_settled_age_at_seed_secs": 0,
                "normalization": "Only ephemeral backend TCP port is removed; host, DP rank, key, inflight and released values are preserved.",
                "inputs": {"sessions":count,"engines":workers.len(),"hosts":72,"seed":"deterministic sorted URLs; no RNG","requests":8192,"concurrency":512,"running":0,"waiting":0,"kv_usage":0.1,"typed_request":"even i: messages=user:preserve session history; model=m; stream=false","native_request":"odd i: request_id=probe-i; prompt_ids=[1;1024]; max_tokens=1","session_header":"X-Session-ID=settled-(i%100000)"}
            })).unwrap()).unwrap();
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
            let bytes = headers.and_then(|headers| {
                headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                    .and_then(|(_, value)| value.parse::<u64>().ok())
            });
            let mut state = self.state.lock().unwrap();
            self.choose_locked(
                &mut state, workers, &healthy, key, reserve, instant, wall, abort, bytes,
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
        _wall: f64,
        abort: bool,
        bytes: Option<u64>,
    ) -> Option<(usize, Option<SessionLease>)> {
        state.expire_contexts(instant);
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
            return self.reserve_locked(
                state, workers, idx, key, reserve, !abort, false, instant, bytes,
            );
        }

        // The latest declared body size belongs to the bound logical session,
        // including its idle affinity window, rather than just this HTTP lease.
        // Preview an owner's growth before deciding whether it should spill.
        let old_bytes = resident.map_or(0, |session| session.context_bytes);
        let latest_bytes = bytes.unwrap_or(old_bytes);
        let candidates: Vec<_> = healthy
            .iter()
            .map(|&idx| {
                (
                    idx,
                    state
                        .context_bytes
                        .get(workers[idx].url())
                        .copied()
                        .unwrap_or(0)
                        - if owner == Some(idx) {
                            u128::from(old_bytes)
                        } else {
                            0
                        }
                        + if owner == Some(idx) {
                            u128::from(latest_bytes)
                        } else {
                            0
                        },
                    state
                        .last_selected
                        .get(workers[idx].url())
                        .copied()
                        .unwrap_or(0),
                )
            })
            .collect();
        let mut loads: Vec<_> = candidates.iter().map(|(_, load, _)| *load).collect();
        loads.sort_unstable();
        let median = (loads[(loads.len() - 1) / 2] as f64 + loads[loads.len() / 2] as f64) / 2.0;
        let keep = keep_affinity
            && owner.is_some_and(|idx| {
                let load = candidates.iter().find(|(i, _, _)| *i == idx).unwrap().1;
                let best_destination = candidates
                    .iter()
                    .filter(|(i, _, _)| *i != idx)
                    .map(|(_, load, _)| *load + u128::from(latest_bytes))
                    .min();
                load == 0
                    || (load as f64) < self.spill_factor * median
                    // Moving a sole context to an idle engine cannot reduce its
                    // weight; preserve affinity when relocation cannot help.
                    || (latest_bytes > 0 && best_destination.is_none_or(|best| best >= load))
            });
        let idx = if keep {
            owner.unwrap()
        } else {
            candidates
                .iter()
                .min_by_key(|(_, load, selected)| (*load, *selected))
                .unwrap()
                .0
        };
        self.reserve_locked(
            state, workers, idx, key, reserve, true, true, instant, bytes,
        )
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
        bytes: Option<u64>,
    ) -> Option<(usize, Option<SessionLease>)> {
        let worker = workers[idx].url().to_owned();
        state.sequence = state.sequence.wrapping_add(1);
        let sequence = state.sequence;
        if reserve && inference {
            state.last_selected.insert(worker.clone(), sequence);
        }
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
        let mut context_change = None;
        if let Some(key) = &key {
            let session = state.sessions.entry(key.clone()).or_insert(Session {
                worker: worker.clone(),
                inflight: 0,
                last_settled: instant,
                released: false,
                context_bytes: 0,
                context_revision: 0,
            });
            assert!(
                session.inflight == 0 || session.worker == worker,
                "cannot move an outstanding session"
            );
            if inference && (reserve || session.worker != worker) {
                let previous = (session.worker.clone(), session.context_bytes);
                if reserve {
                    session.context_bytes = bytes.unwrap_or(session.context_bytes);
                    session.context_revision = session.context_revision.wrapping_add(1);
                }
                context_change = Some((previous, session.context_bytes));
            }
            session.worker = worker.clone();
            if placement_selected {
                session.released = false;
            }
            if reserve {
                session.inflight += 1;
            }
        }
        if let Some(((previous, old), new)) = context_change {
            state.replace_context(&previous, old, 0);
            state.replace_context(&worker, 0, new);
        }
        // Unknown Content-Length keeps the existing live-request fallback.
        // Headerless requests have no retained affinity and live only to EOF.
        let transient_bytes = if reserve && inference {
            if key.is_none() {
                bytes.unwrap_or(1).max(1)
            } else if bytes.is_none_or(|bytes| bytes == 0) {
                1
            } else {
                0
            }
        } else {
            0
        };
        state.replace_context(&worker, 0, transient_bytes);
        if reserve {
            workers[idx].increment_load();
        }
        let lease = reserve.then(|| SessionLease {
            state: Arc::clone(&self.state),
            key,
            worker,
            worker_handle: Arc::clone(&workers[idx]),
            epoch,
            transient_bytes,
            idle: self.idle,
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
    fn release_session_owners(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        let mut idle_contexts = Vec::new();
        for session in state.sessions.values_mut() {
            session.released = true;
            if session.inflight == 0 {
                idle_contexts.push((
                    session.worker.clone(),
                    std::mem::take(&mut session.context_bytes),
                ));
                session.context_revision = session.context_revision.wrapping_add(1);
            }
        }
        for (worker, bytes) in idle_contexts {
            state.replace_context(&worker, bytes, 0);
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
    fn native_load_without_any_publication_balances_outstanding_requests() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let now = Instant::now();
        let a = p.choose(&w, None, None, true, now, 100.0).unwrap();
        let b = p.choose(&w, None, None, true, now, 100.0).unwrap();
        assert_ne!(a.0, b.0);
        assert_eq!(w.iter().map(|w| w.load()).sum::<usize>(), 2);
        drop(a);
        drop(b);
        assert!(w.iter().all(|w| w.load() == 0));
    }
    #[test]
    fn native_load_ignores_stale_missing_and_conflicting_telemetry() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        publish(&p, &w, &[0, 9999], &[0.1, 0.9], 100.0);
        let a = p
            .choose(&w, None, None, true, Instant::now(), 100.0)
            .unwrap();
        let b = p
            .choose(&w, None, None, true, Instant::now(), 1000.0)
            .unwrap();
        assert_ne!(a.0, b.0);
        drop(a);
        drop(b);
    }
    #[test]
    fn native_load_counts_requests_not_retained_session_bindings() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let h = headers();
        let a = p
            .choose(&w, None, Some(&h), true, Instant::now(), 100.0)
            .unwrap();
        let owner = a.0;
        drop(a);
        assert_eq!(p.state.lock().unwrap().sessions.len(), 1);
        assert!(w.iter().all(|w| w.load() == 0));
        let b = p
            .choose(&w, None, Some(&h), true, Instant::now(), 9999.0)
            .unwrap();
        assert_eq!(b.0, owner);
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
                        context_bytes: 0,
                        context_revision: 0,
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
    fn reset_release_survives_overlap_abort_until_next_idle_selection() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let h = headers();
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
                1000.0,
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
                1000.0,
            )
            .unwrap();
        assert_eq!(abort.0, first.0);
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
        let next = p
            .choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                1000.0,
            )
            .unwrap();
        assert_ne!(next.0, original);
        assert!(p
            .state
            .lock()
            .unwrap()
            .sessions
            .values()
            .all(|s| !s.released));
        let owner = next.0;
        drop(next);
        assert_eq!(
            p.choose(
                &w,
                Some("generation"),
                Some(&h),
                true,
                Instant::now(),
                1000.0
            )
            .unwrap()
            .0,
            owner
        );
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
    fn own_outstanding_requests_precede_optional_queue_and_kv() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        publish(&p, &w, &[0, 9999], &[0.0, 1.0], 100.0);
        let held = p.test_hold_requests(&w, w[0].url(), 10);
        let (idx, lease) = p
            .choose(&w, None, None, true, Instant::now(), 1000.0)
            .unwrap();
        assert_eq!(idx, 1);
        drop(lease);
        drop(held);
        let a = p
            .choose(&w, None, None, true, Instant::now(), 1000.0)
            .unwrap();
        let b = p
            .choose(&w, None, None, true, Instant::now(), 1000.0)
            .unwrap();
        assert_ne!(a.0, b.0);
        drop(a);
        drop(b);
        assert!(w.iter().all(|w| w.load() == 0));
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
        let _held = p.test_hold_requests(&w, w[0].url(), 100);
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
    fn stale_publication_never_blocks_and_bad_updates_remain_atomic() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        publish(&p, &w, &[1, 2], &[0.1, 0.2], 100.0);
        let a = p
            .choose(&w, None, None, true, Instant::now(), 107.0)
            .unwrap();
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
        let b = p
            .choose(&w, None, None, true, Instant::now(), 1000.0)
            .unwrap();
        assert_ne!(a.0, b.0);
        drop(a);
        drop(b);
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
    fn heterogeneous_hosts_balance_without_complete_fresh_coverage() {
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
        p.update_at(
            EngineSnapshot {
                workers: vec![EngineObservation {
                    worker_url: w[0].url().to_owned(),
                    observed_at: 100.0,
                    running: 9999,
                    waiting: 9999,
                    kv_usage: 1.0,
                }],
            },
            &w.iter().map(|w| w.url().to_owned()).collect::<Vec<_>>(),
            100.0,
        )
        .unwrap();
        let leases: Vec<_> = (0..100)
            .map(|_| {
                p.choose(&w, None, None, true, Instant::now(), 1000.0)
                    .unwrap()
            })
            .collect();
        assert!(w.iter().all(|w| w.load() == 10));
        drop(leases);
        assert!(w.iter().all(|w| w.load() == 0));
    }

    fn sized_headers(key: &str, bytes: u64) -> RequestHeaders {
        HashMap::from([
            ("x-session-id".into(), key.into()),
            ("content-length".into(), bytes.to_string()),
        ])
    }

    fn bound_total(p: &EngineLoadPolicy) -> u128 {
        p.state.lock().unwrap().context_bytes.values().sum()
    }

    #[test]
    fn context_is_retained_after_eof_and_replaced_by_the_latest_request() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let first = p
            .select_generation_worker_with_lease(&w, None, Some(&sized_headers("s", 10)))
            .unwrap();
        let overlap = p
            .select_generation_worker_with_lease(&w, None, Some(&sized_headers("s", 100_000)))
            .unwrap();
        assert_eq!(first.0, overlap.0);
        assert_eq!(bound_total(&p), 100_000);
        drop(first);
        assert_eq!(bound_total(&p), 100_000);
        drop(overlap);
        assert!(w.iter().all(|w| w.load() == 0));
        assert_eq!(bound_total(&p), 100_000);
        let shorter = p
            .select_generation_worker_with_lease(&w, None, Some(&sized_headers("s", 30)))
            .unwrap();
        assert_eq!(bound_total(&p), 30);
        drop(shorter);
        assert_eq!(bound_total(&p), 30);
    }

    #[test]
    fn a_sole_context_does_not_move_when_relocation_cannot_reduce_weight() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let mut owner = None;
        for size in [16, 1024, 100_000, 500_000] {
            let request = p
                .select_generation_worker_with_lease(&w, None, Some(&sized_headers("s", size)))
                .unwrap();
            assert_eq!(*owner.get_or_insert(request.0), request.0);
            drop(request);
            assert_eq!(bound_total(&p), u128::from(size));
        }
    }

    #[test]
    fn idle_context_deadline_expires_before_the_abort_owner_tombstone() {
        let p = EngineLoadPolicy::with_parameters(Duration::from_secs(10), 2.0);
        let w = workers();
        drop(
            p.select_generation_worker_with_lease(&w, None, Some(&sized_headers("s", 1000)))
                .unwrap(),
        );
        p.choose(
            &w,
            None,
            None,
            false,
            Instant::now() + Duration::from_secs(11),
            0.0,
        );
        assert_eq!(bound_total(&p), 0);
        assert!(p
            .state
            .lock()
            .unwrap()
            .sessions
            .contains_key("header:x-session-id:s"));
        let active = p
            .select_generation_worker_with_lease(&w, None, Some(&sized_headers("active", 2000)))
            .unwrap();
        p.choose(
            &w,
            None,
            None,
            false,
            Instant::now() + Duration::from_secs(100),
            0.0,
        );
        assert_eq!(bound_total(&p), 2000);
        drop(active);
    }

    #[test]
    fn reset_drops_idle_weights_and_retains_active_weights_until_settled() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        drop(
            p.select_generation_worker_with_lease(&w, None, Some(&sized_headers("idle", 1000)))
                .unwrap(),
        );
        let active = p
            .select_generation_worker_with_lease(&w, None, Some(&sized_headers("active", 2000)))
            .unwrap();
        p.release_session_owners();
        assert_eq!(bound_total(&p), 2000);
        let overlap = p
            .select_generation_worker_with_lease(&w, None, Some(&sized_headers("active", 3000)))
            .unwrap();
        assert_eq!(active.0, overlap.0);
        drop(overlap);
        assert_eq!(bound_total(&p), 3000);
        drop(active);
        assert_eq!(bound_total(&p), 0);
    }

    #[test]
    fn abort_body_length_never_replaces_the_generation_context() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let request = p
            .select_generation_worker_with_lease(&w, None, Some(&sized_headers("s", 1000)))
            .unwrap();
        let owner = request.0;
        drop(request);
        let abort = p
            .select_control_worker_with_lease(
                &w,
                Some(r#"{"request_id":"r"}"#),
                Some(&sized_headers("s", 900_000)),
            )
            .unwrap();
        assert_eq!(abort.0, owner);
        drop(abort);
        assert_eq!(bound_total(&p), 1000);
    }

    #[test]
    fn missing_invalid_and_unkeyed_lengths_use_finite_live_request_fallback() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        for value in ["bad", "-1", "18446744073709551616", "0"] {
            let h = HashMap::from([("content-length".into(), value.into())]);
            let request = p
                .select_generation_worker_with_lease(&w, None, Some(&h))
                .unwrap();
            assert_eq!(bound_total(&p), 1);
            drop(request);
            assert_eq!(bound_total(&p), 0);
        }
        let h = HashMap::from([("Content-Length".into(), "12345".into())]);
        let request = p
            .select_generation_worker_with_lease(&w, None, Some(&h))
            .unwrap();
        assert_eq!(bound_total(&p), 12345);
        drop(request);
        assert_eq!(bound_total(&p), 0);
    }

    #[test]
    fn stale_idle_deadline_cannot_remove_a_newer_turns_context() {
        let p = EngineLoadPolicy::with_parameters(Duration::from_secs(10), 2.0);
        let w = workers();
        drop(
            p.select_generation_worker_with_lease(&w, None, Some(&sized_headers("s", 1000)))
                .unwrap(),
        );
        let first = p.state.lock().unwrap().context_expiry.peek().unwrap().0 .0;
        drop(
            p.select_generation_worker_with_lease(&w, None, Some(&sized_headers("s", 2000)))
                .unwrap(),
        );
        p.choose(&w, None, None, false, first + Duration::from_nanos(1), 0.0);
        assert_eq!(bound_total(&p), 2000);
    }

    #[test]
    fn preview_rehoming_keeps_bound_weight_and_session_owner_consistent() {
        let p = EngineLoadPolicy::new();
        let w = workers();
        let h = sized_headers("s", 1000);
        let request = p
            .select_generation_worker_with_lease(&w, None, Some(&h))
            .unwrap();
        let owner = request.0;
        drop(request);
        let held = p.test_hold_requests(&w, w[owner].url(), 10);
        let next = p.select_worker_with_headers(&w, None, Some(&h)).unwrap();
        assert_ne!(next, owner);
        assert_eq!(bound_total(&p), 1010);
        drop(
            p.select_generation_worker_with_lease(&w, None, Some(&h))
                .unwrap(),
        );
        drop(held);
        assert_eq!(bound_total(&p), 1000);
    }

    #[test]
    fn growing_native_sessions_balance_bound_context_after_each_growth_wave() {
        let p = EngineLoadPolicy::new();
        let w: Vec<Arc<dyn Worker>> = (0..8)
            .map(|rank| {
                Arc::new(BasicWorker::new(
                    format!("http://a:8000@{rank}"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        // The ledger is derived from actual selected destinations and request
        // lengths, independently of the policy's accounting implementation.
        let mut observed: HashMap<usize, (usize, u64)> = HashMap::new();
        let mut hot = Vec::new();
        for session in 0..64 {
            let body = serde_json::to_vec(&serde_json::json!({"request_id":format!("{session:03}-initial"),"prompt_ids":vec![1_u32;16],"max_tokens":1})).unwrap();
            let request = p
                .select_generation_worker_with_lease(
                    &w,
                    None,
                    Some(&sized_headers(
                        &format!("growing-{session}"),
                        body.len() as u64,
                    )),
                )
                .unwrap();
            if request.0 == 0 {
                hot.push(session);
            }
            observed.insert(session, (request.0, body.len() as u64));
            drop(request);
        }
        assert_eq!(hot, vec![0, 8, 16, 24, 32, 40, 48, 56]);
        // Only the sessions initially assigned to engine0 grow. Request-count
        // affinity leaves all those settled long contexts on that engine.
        // The selector gets Content-Length, not the native token-id payload.
        for tokens in [512, 4096, 100_001, 140_000] {
            for &session in &hot {
                let body = serde_json::to_vec(&serde_json::json!({"request_id":format!("{session:03}-{tokens}"),"prompt_ids":vec![1_u32;tokens],"max_tokens":1})).unwrap();
                let request = p
                    .select_generation_worker_with_lease(
                        &w,
                        None,
                        Some(&sized_headers(
                            &format!("growing-{session}"),
                            body.len() as u64,
                        )),
                    )
                    .unwrap();
                observed.insert(session, (request.0, body.len() as u64));
                drop(request);
            }
            let mut weights = vec![0_u128; w.len()];
            for &(engine, bytes) in observed.values() {
                weights[engine] += u128::from(bytes);
            }
            weights.sort_unstable();
            let twice_median = weights[3] + weights[4];
            assert!(
                weights[7] <= twice_median,
                "tokens={tokens}; weights={weights:?}"
            );
        }
        assert!(w.iter().all(|worker| worker.load() == 0));
        assert!(
            observed
                .values()
                .map(|(_, bytes)| u128::from(*bytes))
                .sum::<u128>()
                > hot.len() as u128 * 200_000
        );
    }
}
