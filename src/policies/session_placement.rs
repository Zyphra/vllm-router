//! Load-aware placement of new sessions for the consistent hash policy.
//!
//! Opt-in with `VLLM_ROUTER_SESSION_PLACEMENT`:
//! - `least_tokens`: a session key seen for the first time is placed on the
//!   healthy worker whose sessions hold the fewest context tokens, instead of the
//!   worker the hash ring names.
//! - `bounded_hash`: consistent hashing with bounded loads. A new session goes to
//!   the first healthy worker clockwise from its key on the hash ring whose open
//!   sessions are below `ceil(c * (open sessions + 1) / workers)`, with
//!   `c = VLLM_ROUTER_SESSION_LOAD_FACTOR` (default 1.25, at least 1). Most
//!   sessions keep their ring worker; a worker whose ring arc is larger than its
//!   share, or whose sessions run longer, stops taking new sessions at the bound
//!   instead of queueing them behind a full engine.
//!
//! Later requests of a session stay on the same worker for prefix-cache reuse.
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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const PLACEMENT_ENV: &str = "VLLM_ROUTER_SESSION_PLACEMENT";
pub const IDLE_SECS_ENV: &str = "VLLM_ROUTER_SESSION_IDLE_SECS";
pub const LOAD_FACTOR_ENV: &str = "VLLM_ROUTER_SESSION_LOAD_FACTOR";
const DEFAULT_LOAD_FACTOR: f64 = 1.25;

/// How a new (or released) session chooses its worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlacementRule {
    /// The healthy worker with the fewest counted context tokens.
    LeastTokens,
    /// The first worker in hash-ring order below `ceil(factor * (sessions + 1) / workers)`.
    BoundedHash { factor: f64 },
}
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
pub struct SessionLease {
    placement: Arc<TokenPlacement>,
    key: String,
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        self.placement.settle_at(&self.key, Instant::now());
    }
}

/// Sticky least-tokens session placement shared by all requests of one Router.
#[derive(Debug)]
pub struct TokenPlacement {
    idle: Duration,
    rule: PlacementRule,
    state: Mutex<State>,
}

impl TokenPlacement {
    pub fn new(idle: Duration) -> Self {
        Self::with_rule(idle, PlacementRule::LeastTokens)
    }

    pub fn with_rule(idle: Duration, rule: PlacementRule) -> Self {
        if let PlacementRule::BoundedHash { factor } = rule {
            assert!(
                factor.is_finite() && factor >= 1.0,
                "bounded_hash load factor must be finite and at least 1, got {factor}"
            );
        }
        Self {
            idle,
            rule,
            state: Mutex::new(State {
                sessions: HashMap::new(),
                loads: HashMap::new(),
                last_sweep: Instant::now(),
            }),
        }
    }

    /// Build from the environment; `None` keeps plain consistent hashing.
    pub fn from_env() -> Option<Self> {
        let rule = match std::env::var(PLACEMENT_ENV).ok().as_deref() {
            None | Some("") | Some("hash") => return None,
            Some("least_tokens") => PlacementRule::LeastTokens,
            Some("bounded_hash") => PlacementRule::BoundedHash {
                factor: match std::env::var(LOAD_FACTOR_ENV).ok().as_deref() {
                    None | Some("") => DEFAULT_LOAD_FACTOR,
                    Some(value) => match value.parse::<f64>() {
                        Ok(factor) if factor.is_finite() && factor >= 1.0 => factor,
                        _ => panic!(
                            "{LOAD_FACTOR_ENV} must be a number of at least 1, got {value:?}"
                        ),
                    },
                },
            },
            Some(other) => {
                panic!(
                    "{PLACEMENT_ENV} must be 'hash', 'least_tokens' or 'bounded_hash', got {other:?}"
                )
            }
        };
        let idle = match std::env::var(IDLE_SECS_ENV).ok().as_deref() {
            None | Some("") => DEFAULT_IDLE_SECS,
            Some(value) => match value.parse::<u64>() {
                Ok(secs) if secs > 0 => secs,
                _ => panic!("{IDLE_SECS_ENV} must be a positive integer, got {value:?}"),
            },
        };
        Some(Self::with_rule(Duration::from_secs(idle), rule))
    }

    /// Whether [`TokenPlacement::place`] expects `healthy` in hash-ring order from the key.
    pub fn wants_ring_order(&self) -> bool {
        matches!(self.rule, PlacementRule::BoundedHash { .. })
    }

    /// Return the worker for session `key`. A new session, or one whose worker
    /// left `healthy`, goes to the healthy worker with the fewest counted
    /// context tokens (then fewest sessions, then the earliest in `healthy`).
    /// `healthy` must not be empty. For dispatch use [`Self::place_and_lease`]
    /// so reset cannot move the owner between selection and reservation.
    pub fn place(&self, key: &str, request_text: Option<&str>, healthy: &[&str]) -> String {
        self.place_at(key, request_text, healthy, Instant::now())
    }

    fn place_at(
        &self,
        key: &str,
        request_text: Option<&str>,
        healthy: &[&str],
        now: Instant,
    ) -> String {
        let mut state = self.state.lock().unwrap();
        self.place_locked(&mut state, key, request_text, healthy, now)
    }

    /// Select and reserve the owner in one transaction. Hold the returned lease
    /// until the routed response completes, fails or is dropped.
    pub fn place_and_lease(
        self: &Arc<Self>,
        key: &str,
        request_text: Option<&str>,
        healthy: &[&str],
    ) -> (String, SessionLease) {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        let worker = self.place_locked(&mut state, key, request_text, healthy, now);
        state.sessions.get_mut(key).unwrap().inflight += 1;
        let lease = SessionLease {
            placement: Arc::clone(self),
            key: key.to_string(),
        };
        (worker, lease)
    }

    fn place_locked(
        &self,
        state: &mut State,
        key: &str,
        request_text: Option<&str>,
        healthy: &[&str],
        now: Instant,
    ) -> String {
        let exact = prompt_token_count(request_text);
        let estimate = request_text.map_or(0, |text| text.len() as u64 / 4);
        self.sweep(state, now);

        let owner_gone = state.sessions.get(key).is_some_and(|session| {
            !healthy.contains(&session.worker.as_str())
                || (session.released && session.inflight == 0)
        });
        if owner_gone || !state.sessions.contains_key(key) {
            // Re-placing an existing session keeps its in-flight count, so its
            // outstanding leases still settle against it.
            state.set_counted(key, false);
            let load = |worker: &str| state.loads.get(worker).copied().unwrap_or_default();
            let worker = match self.rule {
                PlacementRule::LeastTokens => healthy.iter().min_by_key(|worker| load(worker)),
                PlacementRule::BoundedHash { factor } => {
                    let open: u64 = healthy.iter().map(|worker| load(worker).sessions).sum();
                    let bound =
                        (factor * (open + 1) as f64 / healthy.len().max(1) as f64).ceil() as u64;
                    healthy
                        .iter()
                        .find(|worker| load(worker).sessions < bound)
                        .or_else(|| healthy.iter().min_by_key(|worker| load(worker).sessions))
                }
            }
            .expect("placement requires at least one healthy worker")
            .to_string();
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
        worker
    }

    /// Release every session's owner (the prefix caches were reset): each session
    /// is re-placed on its next request made with none of its requests in flight.
    /// Returns the number of sessions released.
    pub fn release_owners(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        for session in state.sessions.values_mut() {
            session.released = true;
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
        Some(SessionLease {
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

    fn body(tokens: usize) -> String {
        let ids: Vec<String> = (0..tokens).map(|i| (i % 1000).to_string()).collect();
        format!(
            "{{\"prompt_ids\":[{}],\"request_id\":\"r\"}}",
            ids.join(",")
        )
    }

    fn settle(placement: &TokenPlacement, lease: SessionLease, now: Instant) {
        placement.settle_at(&lease.key, now);
        std::mem::forget(lease);
    }

    fn tokens_on(placement: &TokenPlacement, worker: &str) -> u64 {
        placement.worker_tokens().get(worker).copied().unwrap_or(0)
    }

    #[test]
    fn counts_prompt_ids() {
        assert_eq!(prompt_token_count(Some(&body(3))), Some(3));
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

    fn sessions_on(placement: &TokenPlacement) -> HashMap<String, u64> {
        let state = placement.state.lock().unwrap();
        state
            .loads
            .iter()
            .map(|(worker, load)| (worker.clone(), load.sessions))
            .collect()
    }

    #[test]
    fn bounded_hash_keeps_ring_owner_until_the_bound() {
        let placement = TokenPlacement::with_rule(
            Duration::from_secs(900),
            PlacementRule::BoundedHash { factor: 1.0 },
        );
        assert!(placement.wants_ring_order());
        // Every key names "a" first on its ring walk; the bound moves the overflow on.
        let order = ["a", "b", "c", "d"];
        let owners: Vec<String> = (0..8)
            .map(|i| placement.place(&format!("s{i}"), Some(&body(10)), &order))
            .collect();
        assert_eq!(owners[0], "a");
        let loads = sessions_on(&placement);
        assert!(loads.values().all(|&n| n == 2), "{loads:?}");
        // Sessions stay put on later turns, whatever the loads.
        for (i, owner) in owners.iter().enumerate() {
            assert_eq!(
                &placement.place(&format!("s{i}"), Some(&body(20)), &order),
                owner
            );
        }
    }

    #[test]
    fn bounded_hash_rejects_a_factor_below_one() {
        let result = std::panic::catch_unwind(|| {
            TokenPlacement::with_rule(
                Duration::from_secs(1),
                PlacementRule::BoundedHash { factor: 0.9 },
            )
        });
        assert!(result.is_err());
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
