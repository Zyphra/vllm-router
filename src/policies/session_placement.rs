//! Token-aware placement of new sessions for the consistent hash policy.
//!
//! Opt-in with `VLLM_ROUTER_SESSION_PLACEMENT=least_tokens`. A session key seen
//! for the first time is placed on the healthy worker whose sessions hold the
//! fewest context tokens, instead of the worker the hash ring names. Later
//! requests of that session stay on the same worker for prefix-cache reuse.
//! Every request updates its session's context length. A routed request holds a
//! [`SessionLease`] until it settles (response fully sent, failed or dropped);
//! a session with a request in flight never expires, so a long generation keeps
//! counting and its abort reaches the same worker. Once its last request has
//! settled, a session idle for `VLLM_ROUTER_SESSION_IDLE_SECS` (default 900)
//! stops counting toward its worker's load, and its stickiness is forgotten
//! after four idle periods.
//!
//! A weight update resets every engine's prefix cache, so stickiness has no
//! value right after one: [`TokenPlacement::release_owners`] marks every session
//! for re-placement by the same least-tokens rule on its next request made with
//! none of its requests in flight. A session with a request in flight keeps its
//! owner until that request settles, so abort and resume stay consistent.
//!
//! Context length is the number of `prompt_ids` in the body when present;
//! otherwise it is estimated as body bytes / 4 and only ever grows, so an
//! abort-shaped body neither moves nor shrinks a session.

use super::kv_capacity::{KvCapacity, KvSnapshot};
use super::session_spill::{
    host, QueueObservation, QueuePressure, QueueSnapshot, SessionSpillConfig, SPILL_CONFIG_ENV,
};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const PLACEMENT_ENV: &str = "VLLM_ROUTER_SESSION_PLACEMENT";
pub const IDLE_SECS_ENV: &str = "VLLM_ROUTER_SESSION_IDLE_SECS";
const DEFAULT_IDLE_SECS: u64 = 900;
const RETAIN_IDLE_PERIODS: u32 = 4;
const SWEEP_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug)]
struct Session {
    worker: String,
    tokens: u64,
    /// Last placement or settle; idle time only runs while `inflight` is zero.
    last_seen: Instant,
    counted: bool,
    inflight: u32,
    /// Set by `release_owners`: re-place on the next request made while idle.
    released: bool,
}

/// Ordered by tokens first, then by session count as the tie-break.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Load {
    tokens: u64,
    sessions: u64,
}

#[derive(Debug)]
struct State {
    sessions: HashMap<String, Session>,
    loads: HashMap<String, Load>,
    last_sweep: Instant,
    capacity: KvCapacity,
    queue: QueuePressure,
}

impl State {
    fn set_counted(&mut self, key: &str, counted: bool) {
        let Some(session) = self.sessions.get_mut(key) else {
            return;
        };
        if session.counted == counted {
            return;
        }
        session.counted = counted;
        let load = self.loads.entry(session.worker.clone()).or_default();
        if counted {
            load.tokens += session.tokens;
            load.sessions += 1;
        } else {
            load.tokens = load.tokens.saturating_sub(session.tokens);
            load.sessions = load.sessions.saturating_sub(1);
        }
    }
}

/// Keeps a session's owner while one routed request is in flight; dropping it
/// settles the request and starts the session's idle clock.
#[derive(Debug)]
pub enum SessionLease {
    Session {
        placement: Arc<TokenPlacement>,
        key: String,
    },
    Stateless(Arc<dyn crate::core::Worker>),
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        match self {
            Self::Session { placement, key } => placement.settle_at(key, Instant::now()),
            Self::Stateless(worker) => {
                worker.decrement_load();
                crate::metrics::RouterMetrics::set_running_requests(worker.url(), worker.load());
            }
        }
    }
}

/// Sticky least-tokens session placement shared by all requests of one Router.
#[derive(Debug)]
pub struct TokenPlacement {
    idle: Duration,
    kv_aware: bool,
    spill: Option<SessionSpillConfig>,
    state: Mutex<State>,
}

impl TokenPlacement {
    pub fn new(idle: Duration) -> Self {
        Self {
            idle,
            kv_aware: false,
            spill: None,
            state: Mutex::new(State {
                sessions: HashMap::new(),
                loads: HashMap::new(),
                last_sweep: Instant::now(),
                capacity: KvCapacity::default(),
                queue: QueuePressure::default(),
            }),
        }
    }

    pub fn with_kv_capacity(mut self) -> Self {
        self.kv_aware = true;
        self
    }

    pub fn with_queue_spill(mut self, config: SessionSpillConfig) -> Result<Self, String> {
        config.validate()?;
        self.spill = Some(config);
        Ok(self)
    }

    fn generation_request(&self, request_text: Option<&str>, exact: Option<u64>) -> bool {
        self.spill.is_some()
            && (exact.is_some()
                || request_text
                    .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
                    .is_some_and(|body| {
                        body.get("messages").is_some()
                            || body.get("prompt").is_some()
                            || body.get("input").is_some()
                    }))
    }

    /// Build from the environment; `None` keeps plain consistent hashing.
    pub fn from_env() -> Option<Self> {
        match std::env::var(PLACEMENT_ENV).ok().as_deref() {
            None | Some("") | Some("hash") => None,
            Some(mode @ ("least_tokens" | "kv_aware" | "least_tokens_spill")) => {
                let idle = match std::env::var(IDLE_SECS_ENV).ok().as_deref() {
                    None | Some("") => DEFAULT_IDLE_SECS,
                    Some(value) => match value.parse::<u64>() {
                        Ok(secs) if secs > 0 => secs,
                        _ => panic!("{IDLE_SECS_ENV} must be a positive integer, got {value:?}"),
                    },
                };
                let placement = Self::new(Duration::from_secs(idle));
                let placement = if mode == "kv_aware" {
                    placement.with_kv_capacity()
                } else if mode == "least_tokens_spill" {
                    let config = std::env::var(SPILL_CONFIG_ENV)
                        .ok()
                        .filter(|text| !text.is_empty())
                        .map(|text| {
                            serde_json::from_str::<SessionSpillConfig>(&text)
                                .expect("invalid session spill config")
                        })
                        .unwrap_or_default();
                    placement
                        .with_queue_spill(config)
                        .expect("invalid session spill config")
                } else {
                    placement
                };
                Some(placement)
            }
            Some(other) => {
                panic!(
                    "{PLACEMENT_ENV} must be 'hash', 'least_tokens', 'kv_aware' or 'least_tokens_spill', got {other:?}"
                )
            }
        }
    }

    /// Return the worker for session `key`. A new session, or one whose worker
    /// left `healthy`, goes to the healthy worker with the fewest counted
    /// context tokens (then fewest sessions, then the earliest in `healthy`).
    /// `healthy` must not be empty. For dispatch use [`Self::place_and_lease`]
    /// so reset cannot move the owner between selection and reservation.
    pub fn place(&self, key: &str, request_text: Option<&str>, healthy: &[&str]) -> String {
        self.place_at(key, request_text, healthy, Instant::now())
    }

    pub fn try_place(
        &self,
        key: &str,
        request_text: Option<&str>,
        healthy: &[&str],
    ) -> Option<String> {
        let tokens = prompt_token_count(request_text);
        let estimate = request_text.map_or(0, |text| text.len() as u64 / 4);
        let generation = self.generation_request(request_text, tokens);
        let mut state = self.state.lock().unwrap();
        self.place_locked(
            &mut state,
            key,
            (tokens, estimate, generation),
            healthy,
            Instant::now(),
        )
    }

    fn place_at(
        &self,
        key: &str,
        request_text: Option<&str>,
        healthy: &[&str],
        now: Instant,
    ) -> String {
        let tokens = prompt_token_count(request_text);
        let estimate = request_text.map_or(0, |text| text.len() as u64 / 4);
        let generation = self.generation_request(request_text, tokens);
        let mut state = self.state.lock().unwrap();
        self.place_locked(
            &mut state,
            key,
            (tokens, estimate, generation),
            healthy,
            now,
        )
        .expect("no fresh KV capacity for session placement")
    }

    /// Select and reserve the owner in one transaction. Hold the returned lease
    /// until the routed response completes, fails or is dropped.
    pub fn place_and_lease(
        self: &Arc<Self>,
        key: &str,
        request_text: Option<&str>,
        healthy: &[&str],
    ) -> (String, SessionLease) {
        self.try_place_and_lease(key, request_text, healthy)
            .expect("no fresh KV capacity for session placement")
    }

    pub fn try_place_and_lease(
        self: &Arc<Self>,
        key: &str,
        request_text: Option<&str>,
        healthy: &[&str],
    ) -> Option<(String, SessionLease)> {
        let now = Instant::now();
        // Request-only scanning must not serialize every placement and lease
        // behind a long prompt while occupying the async runtime's workers.
        let tokens = prompt_token_count(request_text);
        let estimate = request_text.map_or(0, |text| text.len() as u64 / 4);
        let generation = self.generation_request(request_text, tokens);
        let mut state = self.state.lock().unwrap();
        let worker = self.place_locked(
            &mut state,
            key,
            (tokens, estimate, generation),
            healthy,
            now,
        )?;
        state.sessions.get_mut(key).unwrap().inflight += 1;
        let lease = SessionLease::Session {
            placement: Arc::clone(self),
            key: key.to_string(),
        };
        Some((worker, lease))
    }

    fn place_locked(
        &self,
        state: &mut State,
        key: &str,
        context: (Option<u64>, u64, bool),
        healthy: &[&str],
        now: Instant,
    ) -> Option<String> {
        let (exact, estimate, generation) = context;
        self.sweep(state, now);

        let owner_gone = state.sessions.get(key).is_some_and(|session| {
            !healthy.contains(&session.worker.as_str())
                || (session.released && session.inflight == 0)
        });
        let spill_hosts = if generation {
            state
                .sessions
                .get(key)
                .filter(|session| session.inflight == 0)
                .and_then(|session| {
                    self.spill.as_ref().and_then(|config| {
                        state
                            .queue
                            .spill_hosts(&session.worker, healthy, config, unix_now())
                    })
                })
        } else {
            None
        };
        let new_owner = owner_gone || !state.sessions.contains_key(key) || spill_hosts.is_some();
        let previous_tokens = state.sessions.get(key).map_or(0, |session| session.tokens);
        if new_owner {
            // Re-placing an existing session keeps its in-flight count, so its
            // outstanding leases still settle against it.
            let worker = if self.kv_aware {
                let prompt = exact.unwrap_or(previous_tokens.max(estimate));
                let observed_now = unix_now();
                healthy
                    .iter()
                    .enumerate()
                    .filter_map(|(index, worker)| {
                        state
                            .capacity
                            .free_after(
                                worker,
                                prompt,
                                observed_now,
                                state.loads.get(*worker).map_or(0, |load| load.tokens),
                            )
                            .map(|free| (index, *worker, free))
                    })
                    .max_by_key(|(index, worker, free)| {
                        (
                            *free,
                            Reverse(state.loads.get(*worker).copied().unwrap_or_default()),
                            Reverse(*index),
                        )
                    })?
                    .1
                    .to_string()
            } else {
                healthy
                    .iter()
                    .filter(|worker| {
                        spill_hosts
                            .as_ref()
                            .map_or(true, |hosts| hosts.contains(&host(worker)))
                    })
                    .min_by_key(|worker| state.loads.get(**worker).copied().unwrap_or_default())?
                    .to_string()
            };
            state.set_counted(key, false);
            let session = state.sessions.entry(key.to_string()).or_insert(Session {
                worker: String::new(),
                tokens: 0,
                last_seen: now,
                counted: false,
                inflight: 0,
                released: false,
            });
            session.worker = worker;
            session.released = false;
        }

        state.set_counted(key, false);
        let session = state.sessions.get_mut(key).unwrap();
        session.tokens = exact.unwrap_or(session.tokens.max(estimate));
        session.last_seen = now;
        let worker = session.worker.clone();
        state.set_counted(key, true);
        Some(worker)
    }

    pub fn mode(&self) -> &'static str {
        if self.kv_aware {
            "kv_aware"
        } else if self.spill.is_some() {
            "least_tokens_spill"
        } else {
            "least_tokens"
        }
    }

    pub fn update_queue_snapshot(
        &self,
        snapshot: QueueSnapshot,
        registered: &[String],
    ) -> Result<usize, String> {
        if self.spill.is_none() {
            return Err("queue spill placement is disabled".into());
        }
        self.state
            .lock()
            .unwrap()
            .queue
            .update(snapshot, registered, unix_now())
    }

    pub fn update_kv_snapshot(
        &self,
        snapshot: KvSnapshot,
        registered: &[String],
    ) -> Result<usize, String> {
        if self.spill.is_some() {
            return self.update_queue_snapshot(
                QueueSnapshot {
                    workers: snapshot
                        .workers
                        .into_iter()
                        .map(|row| QueueObservation {
                            worker_url: row.worker_url,
                            observed_at: row.observed_at,
                            waiting: row.waiting,
                        })
                        .collect(),
                },
                registered,
            );
        }
        if !self.kv_aware {
            return Err("KV updates require kv_aware session placement".into());
        }
        let mut state = self.state.lock().unwrap();
        let State {
            capacity, loads, ..
        } = &mut *state;
        let count = capacity.update(snapshot, registered, unix_now(), |worker| {
            loads.get(worker).map_or(0, |load| load.tokens)
        })?;
        drop(state);
        Ok(count)
    }

    /// Release every session's owner (the prefix caches were reset): each session
    /// is re-placed on its next request made with none of its requests in flight.
    /// Returns the number of sessions released.
    pub fn release_owners(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        for session in state.sessions.values_mut() {
            session.released = true;
        }
        if self.kv_aware {
            // The explicit weight-reset operation emptied the prefix cache.
            // Idle old contexts must not become the next census load baseline:
            // re-admitting them now allocates KV again. In-flight owners keep
            // their existing reservation and abort routing until they settle.
            let idle: Vec<_> = state
                .sessions
                .iter()
                .filter(|(_, session)| session.inflight == 0)
                .map(|(key, _)| key.clone())
                .collect();
            for key in idle {
                state.set_counted(&key, false);
            }
        }
        state.sessions.len()
    }

    #[cfg(test)]
    fn lease_at(self: &Arc<Self>, key: &str, now: Instant) -> Option<SessionLease> {
        let mut state = self.state.lock().unwrap();
        let session = state.sessions.get_mut(key)?;
        session.inflight += 1;
        session.last_seen = now;
        state.set_counted(key, true);
        Some(SessionLease::Session {
            placement: Arc::clone(self),
            key: key.to_string(),
        })
    }

    fn settle_at(&self, key: &str, now: Instant) {
        let mut state = self.state.lock().unwrap();
        if let Some(session) = state.sessions.get_mut(key) {
            session.inflight = session.inflight.saturating_sub(1);
            session.last_seen = now;
        }
    }

    fn sweep(&self, state: &mut State, now: Instant) {
        if now.saturating_duration_since(state.last_sweep) < SWEEP_INTERVAL {
            return;
        }
        state.last_sweep = now;
        let retain = self.idle * RETAIN_IDLE_PERIODS;
        let expired: Vec<(String, bool)> = state
            .sessions
            .iter()
            .filter(|(_, session)| session.inflight == 0)
            .filter_map(|(key, session)| {
                let age = now.saturating_duration_since(session.last_seen);
                if age > retain {
                    Some((key.clone(), true))
                } else if age > self.idle && session.counted {
                    Some((key.clone(), false))
                } else {
                    None
                }
            })
            .collect();
        for (key, forget) in expired {
            state.set_counted(&key, false);
            if forget {
                state.sessions.remove(&key);
            }
        }
    }

    /// Counted context tokens per worker, for diagnostics and tests.
    pub fn worker_tokens(&self) -> HashMap<String, u64> {
        let state = self.state.lock().unwrap();
        state
            .loads
            .iter()
            .map(|(worker, load)| (worker.clone(), load.tokens))
            .collect()
    }
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Number of entries in the body's flat `prompt_ids` array, if it has one.
pub fn prompt_token_count(request_text: Option<&str>) -> Option<u64> {
    const FIELD: &str = "\"prompt_ids\"";
    let text = request_text?;
    let rest = &text[text.find(FIELD)? + FIELD.len()..];
    let rest = rest
        .trim_start()
        .strip_prefix(':')?
        .trim_start()
        .strip_prefix('[')?;
    let body = &rest[..rest.find(']')?];
    if body.trim().is_empty() {
        return Some(0);
    }
    Some(body.bytes().filter(|byte| *byte == b',').count() as u64 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_placement(rows: &[(&str, f64, u64)]) -> Arc<TokenPlacement> {
        let placement = TokenPlacement::new(Duration::from_secs(900)).with_kv_capacity();
        let placement = Arc::new(placement);
        publish(&placement, rows);
        placement
    }

    fn publish(placement: &TokenPlacement, rows: &[(&str, f64, u64)]) {
        let observed_at = unix_now();
        placement
            .update_kv_snapshot(
                KvSnapshot {
                    workers: rows
                        .iter()
                        .map(|(worker, usage, capacity)| super::super::KvObservation {
                            worker_url: worker.to_string(),
                            observed_at,
                            kv_usage: *usage,
                            waiting: 0,
                            capacity_tokens: *capacity,
                        })
                        .collect(),
                },
                &rows
                    .iter()
                    .map(|(worker, _, _)| worker.to_string())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }

    #[test]
    fn kv_ranks_free_tokens_skips_limit_and_preserves_existing_owners() {
        let placement = kv_placement(&[("a@0", 0.5, 1000), ("b@0", 0.7, 2000)]);
        let (owner, lease) = placement
            .try_place_and_lease("s", Some(&body(100)), &["a@0", "b@0"])
            .unwrap();
        assert_eq!(owner, "b@0"); // More free tokens despite a larger fraction used.
        publish(&placement, &[("a@0", 0.2, 1000), ("b@0", 0.99, 2000)]);
        let (same, next) = placement
            .try_place_and_lease("s", Some(&body(200)), &["a@0", "b@0"])
            .unwrap();
        assert_eq!(same, owner); // Census changes never move a live owner.
        assert_eq!(
            placement
                .try_place_and_lease("new", Some(&body(100)), &["a@0", "b@0"])
                .unwrap()
                .0,
            "a@0"
        );
        drop((lease, next));
    }

    #[test]
    fn kv_missing_full_or_oversized_new_sessions_have_no_owner_or_reservation() {
        let placement = kv_placement(&[("a@0", 0.85, 1000)]);
        assert!(placement
            .try_place_and_lease("full", Some(&body(1)), &["a@0"])
            .is_none());
        publish(&placement, &[("a@0", 0.1, 1000)]);
        assert!(placement
            .try_place_and_lease("oversized", Some(&body(750)), &["a@0"])
            .is_none());
        placement
            .update_kv_snapshot(KvSnapshot { workers: vec![] }, &["a@0".to_string()])
            .unwrap();
        assert!(placement
            .try_place_and_lease("missing", Some(&body(1)), &["a@0"])
            .is_none());
        assert!(placement.state.lock().unwrap().sessions.is_empty());
        assert_eq!(tokens_on(&placement, "a@0"), 0);
    }

    #[test]
    fn kv_concurrent_requests_reserve_existing_load_atomically() {
        let placement = kv_placement(&[("a@0", 0.5, 1000), ("b@0", 0.5, 1000)]);
        let barrier = Arc::new(std::sync::Barrier::new(32));
        let handles: Vec<_> = (0..32)
            .map(|i| {
                let placement = placement.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    placement.try_place_and_lease(
                        &format!("s{i}"),
                        Some(&body(100)),
                        &["a@0", "b@0"],
                    )
                })
            })
            .collect();
        let admitted: Vec<_> = handles
            .into_iter()
            .filter_map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(admitted.len(), 4);
        assert_eq!(tokens_on(&placement, "a@0"), 200);
        assert_eq!(tokens_on(&placement, "b@0"), 200);
        assert_eq!(placement.state.lock().unwrap().sessions.len(), 4);
        drop(admitted);
    }

    #[test]
    fn kv_membership_recovery_uses_only_an_eligible_survivor() {
        let placement = kv_placement(&[("a@0", 0.1, 1000), ("b@0", 0.5, 1000)]);
        let (owner, first) = placement
            .try_place_and_lease("s", Some(&body(100)), &["a@0", "b@0"])
            .unwrap();
        assert_eq!(owner, "a@0");
        drop(first);
        assert_eq!(
            placement
                .try_place_and_lease("s", Some(&body(110)), &["b@0"])
                .unwrap()
                .0,
            "b@0"
        );
    }

    #[test]
    fn kv_weight_reset_reserves_readmitted_idle_contexts_again() {
        let placement = kv_placement(&[("a@0", 0.0, 1000), ("b@0", 0.0, 1000)]);
        for i in 0..6 {
            let (_, lease) = placement
                .try_place_and_lease(&format!("s{i}"), Some(&body(100)), &["a@0", "b@0"])
                .unwrap();
            drop(lease);
        }
        assert_eq!(
            tokens_on(&placement, "a@0") + tokens_on(&placement, "b@0"),
            600
        );
        assert_eq!(placement.release_owners(), 6);
        assert_eq!(
            tokens_on(&placement, "a@0") + tokens_on(&placement, "b@0"),
            0
        );
        publish(&placement, &[("a@0", 0.0, 1000), ("b@0", 0.0, 1000)]);
        let admitted: Vec<_> = (0..18)
            .filter_map(|i| {
                placement.try_place_and_lease(&format!("s{i}"), Some(&body(100)), &["a@0", "b@0"])
            })
            .collect();
        assert_eq!(admitted.len(), 14);
        assert_eq!(tokens_on(&placement, "a@0"), 700);
        assert_eq!(tokens_on(&placement, "b@0"), 700);
        drop(admitted);
    }

    #[test]
    fn heterogeneous_waiting_filters_only_new_sessions() {
        let placement = kv_placement(&[("hot@0", 0.79, 1000), ("cool@0", 0.7, 1000)]);
        let update = || super::super::KvSnapshot {
            workers: vec![
                super::super::KvObservation {
                    worker_url: "hot@0".into(),
                    observed_at: unix_now(),
                    kv_usage: 0.79,
                    waiting: 22,
                    capacity_tokens: 1000,
                },
                super::super::KvObservation {
                    worker_url: "cool@0".into(),
                    observed_at: unix_now(),
                    kv_usage: 0.7,
                    waiting: 0,
                    capacity_tokens: 1000,
                },
            ],
        };
        placement
            .update_kv_snapshot(update(), &["hot@0".into(), "cool@0".into()])
            .unwrap();
        let (owner, lease) = placement
            .try_place_and_lease("resident", Some(&body(10)), &["hot@0", "cool@0"])
            .unwrap();
        assert_eq!(owner, "cool@0");
        placement
            .update_kv_snapshot(
                super::super::KvSnapshot { workers: vec![] },
                &["hot@0".into(), "cool@0".into()],
            )
            .unwrap();
        assert!(placement
            .try_place_and_lease("new", Some(&body(10)), &["hot@0", "cool@0"])
            .is_none());
        assert_eq!(
            placement
                .try_place_and_lease("resident", Some(&body(20)), &["hot@0", "cool@0"])
                .unwrap()
                .0,
            owner
        );
        drop(lease);
    }

    fn queue_publish(placement: &TokenPlacement, waiting: &[(&str, u64)]) {
        placement
            .update_queue_snapshot(
                QueueSnapshot {
                    workers: waiting
                        .iter()
                        .map(|(worker, waiting)| QueueObservation {
                            worker_url: worker.to_string(),
                            observed_at: unix_now(),
                            waiting: *waiting,
                        })
                        .collect(),
                },
                &waiting
                    .iter()
                    .map(|(worker, _)| worker.to_string())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }

    #[test]
    fn judge_spill_moves_only_idle_generation_and_conserves_counted_load() {
        let placement = Arc::new(
            TokenPlacement::new(Duration::from_secs(900))
                .with_queue_spill(SessionSpillConfig::default())
                .unwrap(),
        );
        let workers = ["http://hot:1@0", "http://cool:2@0", "http://middle:3@0"];
        let (owner, lease) = placement
            .try_place_and_lease("s", Some(&body(100)), &workers)
            .unwrap();
        assert_eq!(owner, workers[0]);
        queue_publish(
            &placement,
            &[(workers[0], 1200), (workers[1], 10), (workers[2], 100)],
        );
        // Concurrent/abort routes cannot move a physical request's owner.
        let (same, concurrent) = placement
            .try_place_and_lease("s", Some(&body(110)), &workers)
            .unwrap();
        assert_eq!(same, owner);
        drop(concurrent);
        drop(lease);
        assert_eq!(
            placement
                .try_place("s", Some("{\"request_id\":\"r\"}"), &workers)
                .unwrap(),
            owner
        );
        let (moved, lease) = placement
            .try_place_and_lease("s", Some(&body(120)), &workers)
            .unwrap();
        assert_eq!(moved, workers[1]);
        assert_eq!(tokens_on(&placement, workers[0]), 0);
        assert_eq!(tokens_on(&placement, workers[1]), 120);
        drop(lease);
        // Unqueued resident affinity survives later requests.
        let (same, _) = placement
            .try_place_and_lease("s", Some(&body(130)), &workers)
            .unwrap();
        assert_eq!(same, moved);
    }

    #[test]
    fn judge_spill_unknown_or_stale_queue_preserves_affinity_and_new_sessions_use_least_tokens() {
        let placement = Arc::new(
            TokenPlacement::new(Duration::from_secs(900))
                .with_queue_spill(SessionSpillConfig::default())
                .unwrap(),
        );
        let workers = ["a", "b", "c"];
        let (owner, lease) = placement
            .try_place_and_lease("s", Some(&body(100)), &workers)
            .unwrap();
        drop(lease);
        assert_eq!(owner, "a");
        queue_publish(&placement, &[("a", 30), ("b", 0), ("c", 0)]);
        assert_eq!(
            placement
                .try_place("s", Some(&body(101)), &workers)
                .unwrap(),
            owner
        ); // floor32 avoids jitter
        queue_publish(&placement, &[("a", 1000), ("b", 0), ("c", 0)]);
        placement.state.lock().unwrap().queue = QueuePressure::default();
        assert_eq!(
            placement
                .try_place("s", Some(&body(102)), &workers)
                .unwrap(),
            owner
        );
        let (new, _) = placement
            .try_place_and_lease("new", Some(&body(10)), &workers)
            .unwrap();
        assert_eq!(new, "b");
        assert!(TokenPlacement::new(Duration::from_secs(900))
            .update_queue_snapshot(QueueSnapshot { workers: vec![] }, &[])
            .is_err());
    }

    #[test]
    fn judge_chat_messages_are_work_but_abort_bodies_are_not() {
        let placement = Arc::new(
            TokenPlacement::new(Duration::from_secs(900))
                .with_queue_spill(SessionSpillConfig {
                    minimum_waiting: 0,
                    ..SessionSpillConfig::default()
                })
                .unwrap(),
        );
        let workers = ["a", "b", "c"];
        let chat = "{\"messages\":[{\"role\":\"user\",\"content\":\"judge\"}]}";
        let (owner, lease) = placement
            .try_place_and_lease("s", Some(chat), &workers)
            .unwrap();
        drop(lease);
        queue_publish(&placement, &[("a", 1000), ("b", 0), ("c", 0)]);
        assert_eq!(
            placement
                .try_place("s", Some("{\"request_id\":\"r\"}"), &workers)
                .unwrap(),
            owner
        );
        assert_ne!(
            placement.try_place("s", Some(chat), &workers).unwrap(),
            owner
        );
    }

    fn body(tokens: usize) -> String {
        let ids: Vec<String> = (0..tokens).map(|i| (i % 1000).to_string()).collect();
        format!(
            "{{\"prompt_ids\":[{}],\"request_id\":\"r\"}}",
            ids.join(",")
        )
    }

    fn settle(placement: &TokenPlacement, lease: SessionLease, now: Instant) {
        let SessionLease::Session { key, .. } = &lease else {
            panic!("expected session lease")
        };
        placement.settle_at(key, now);
        std::mem::forget(lease);
    }

    fn tokens_on(placement: &TokenPlacement, worker: &str) -> u64 {
        placement.worker_tokens().get(worker).copied().unwrap_or(0)
    }

    #[test]
    fn counts_prompt_ids() {
        assert_eq!(prompt_token_count(Some(&body(3))), Some(3));
        assert_eq!(prompt_token_count(Some(&body(103_000))), Some(103_000));
        assert_eq!(prompt_token_count(Some("null")), None);
        assert_eq!(prompt_token_count(Some("[7,8]")), None);
        assert_eq!(prompt_token_count(Some("{}")), None);
        assert_eq!(prompt_token_count(Some("{\"prompt_ids\":null}")), None);
        assert_eq!(prompt_token_count(Some("{\"prompt_ids\": [ ]}")), Some(0));
        assert_eq!(
            prompt_token_count(Some("{\"prompt_ids\" : [7, 8]}")),
            Some(2)
        );
        assert_eq!(prompt_token_count(Some("{\"request_id\":\"r\"}")), None);
        assert_eq!(prompt_token_count(None), None);
    }

    #[test]
    fn sessions_are_sticky_and_track_growth() {
        let placement = TokenPlacement::new(Duration::from_secs(900));
        let workers = ["a", "b"];
        let owner = placement.place("s", Some(&body(1000)), &workers);
        assert_eq!(tokens_on(&placement, &owner), 1000);
        // A new session goes to the other, empty worker.
        let other = placement.place("t", Some(&body(10)), &workers);
        assert_ne!(other, owner);
        assert_eq!(placement.place("s", Some(&body(5000)), &workers), owner);
        assert_eq!(tokens_on(&placement, &owner), 5000);
        // An abort-shaped body neither moves nor shrinks the session.
        assert_eq!(
            placement.place("s", Some("{\"request_id\":\"r\"}"), &workers),
            owner
        );
        assert_eq!(tokens_on(&placement, &owner), 5000);
    }

    #[test]
    fn idle_sessions_stop_counting_then_are_forgotten() {
        let placement = TokenPlacement::new(Duration::from_secs(10));
        let workers = ["a", "b"];
        let start = Instant::now();
        let owner = placement.place_at("s", Some(&body(80_000)), &workers, start);
        let later = start + Duration::from_secs(11);
        // The idle 80k session no longer counts, so a new session may share its worker.
        placement.place_at("t", Some(&body(10)), &workers, later);
        assert!(tokens_on(&placement, &owner) <= 10);
        // Returning within the retention window restores stickiness and load.
        assert_eq!(
            placement.place_at("s", Some(&body(80_001)), &workers, later),
            owner
        );
        assert!(tokens_on(&placement, &owner) >= 80_001);
        let forgotten = start + Duration::from_secs(60);
        placement.place_at("u", Some(&body(1)), &workers, forgotten);
        assert_eq!(tokens_on(&placement, "a") + tokens_on(&placement, "b"), 1);
    }

    #[test]
    fn a_session_in_flight_never_expires() {
        let placement = Arc::new(TokenPlacement::new(Duration::from_secs(10)));
        let workers = ["a", "b"];
        let start = Instant::now();
        let owner = placement.place_at("s", Some(&body(80_000)), &workers, start);
        let generation = placement.lease_at("s", start).unwrap();
        // A generation far longer than the idle and retention periods keeps counting.
        let during = start + Duration::from_secs(3_600);
        placement.place_at("t", Some(&body(10)), &workers, during);
        assert!(tokens_on(&placement, &owner) >= 80_000);
        // Its abort, sent after the long gap, reaches the same worker.
        assert_eq!(
            placement.place_at("s", Some("{\"request_id\":\"r\"}"), &workers, during),
            owner
        );
        let abort = placement.lease_at("s", during).unwrap();
        // Settle both at the synthetic clock (dropping a lease settles at the real one).
        settle(&placement, abort, during);
        settle(&placement, generation, during + Duration::from_secs(1));
        // Expiry starts only after the last request settles.
        let settled = during + Duration::from_secs(1);
        placement.place_at(
            "u",
            Some(&body(1)),
            &workers,
            settled + Duration::from_secs(5),
        );
        assert!(tokens_on(&placement, &owner) >= 80_000);
        placement.place_at(
            "u",
            Some(&body(1)),
            &workers,
            settled + Duration::from_secs(11),
        );
        assert!(tokens_on(&placement, &owner) < 80_000);
    }

    #[test]
    fn an_abort_after_a_long_gap_keeps_the_owner_until_retention_ends() {
        let placement = Arc::new(TokenPlacement::new(Duration::from_secs(10)));
        let workers = ["a", "b"];
        let start = Instant::now();
        let owner = placement.place_at("s", Some(&body(500)), &workers, start);
        settle(&placement, placement.lease_at("s", start).unwrap(), start);
        // An abort idle past the idle period but inside retention reaches the owner,
        // even though the owner now carries more load than its peer.
        placement.place_at("t", Some(&body(10)), &workers, start);
        let abort = "{\"request_id\":\"r\"}";
        let gap = start + Duration::from_secs(35);
        assert_eq!(placement.place_at("s", Some(abort), &workers, gap), owner);
        let lease = placement.lease_at("s", gap).unwrap();
        assert_eq!(tokens_on(&placement, &owner), 500);
        settle(&placement, lease, gap);
    }

    #[test]
    fn released_sessions_are_re_placed_by_tokens_once_idle() {
        let placement = Arc::new(TokenPlacement::new(Duration::from_secs(900)));
        let workers = ["a", "b"];
        let start = Instant::now();
        // Two 40k sessions on "a" and "b"; "s" is in flight on its owner.
        let owner = placement.place_at("s", Some(&body(40_000)), &workers, start);
        placement.place_at("t", Some(&body(40_000)), &workers, start);
        let generation = placement.lease_at("s", start).unwrap();
        // Make the owner the heavier worker so a re-placement would move "s".
        placement.place_at("u", Some(&body(50_000)), &[owner.as_str()], start);
        assert_eq!(placement.release_owners(), 3);
        // In flight: the abort and any resume keep the owner.
        assert_eq!(
            placement.place_at("s", Some("{\"request_id\":\"g\"}"), &workers, start),
            owner
        );
        settle(&placement, generation, start);
        // Idle after the release: the resumed turn goes to the least-token worker.
        let other = if owner == "a" { "b" } else { "a" };
        assert_eq!(
            placement.place_at("s", Some(&body(40_100)), &workers, start),
            other
        );
        // ...and sticks there afterwards.
        assert_eq!(
            placement.place_at("s", Some(&body(40_200)), &workers, start),
            other
        );
        assert_eq!(tokens_on(&placement, &owner), 50_000);
    }

    #[test]
    fn atomic_reservation_keeps_owner_across_concurrent_resets_and_aborts() {
        let placement = Arc::new(TokenPlacement::new(Duration::from_secs(900)));
        let workers = ["a", "b"];
        let (owner, generation) = placement.place_and_lease("s", Some(&body(100)), &workers);
        placement.place("heavy", Some(&body(1000)), &[owner.as_str()]);
        let reset = Arc::clone(&placement);
        let resetter = std::thread::spawn(move || {
            for _ in 0..1000 {
                reset.release_owners();
                std::thread::yield_now();
            }
        });
        for _ in 0..1000 {
            let (abort_owner, abort) =
                placement.place_and_lease("s", Some("{\"request_id\":\"r\"}"), &workers);
            assert_eq!(abort_owner, owner);
            drop(abort);
        }
        resetter.join().unwrap();
        placement.release_owners();
        drop(generation);
        let (next_owner, next) = placement.place_and_lease("s", Some(&body(100)), &workers);
        assert_ne!(next_owner, owner);
        drop(next);
        assert_eq!(placement.state.lock().unwrap().sessions["s"].inflight, 0);
    }

    #[test]
    fn atomic_lease_drop_allows_idle_expiry() {
        let placement = Arc::new(TokenPlacement::new(Duration::from_secs(10)));
        let (owner, request) = placement.place_and_lease("s", Some(&body(1000)), &["a", "b"]);
        drop(request);
        placement.place_at(
            "next",
            Some(&body(1)),
            &["a", "b"],
            Instant::now() + Duration::from_secs(11),
        );
        assert!(tokens_on(&placement, &owner) <= 1);
    }

    #[test]
    fn unhealthy_owner_moves_the_session() {
        let placement = TokenPlacement::new(Duration::from_secs(900));
        let owner = placement.place("s", Some(&body(100)), &["a", "b"]);
        let survivor = if owner == "a" { "b" } else { "a" };
        assert_eq!(
            placement.place("s", Some(&body(200)), &[survivor]),
            survivor
        );
        assert_eq!(tokens_on(&placement, &owner), 0);
        assert_eq!(tokens_on(&placement, survivor), 200);
    }
}
