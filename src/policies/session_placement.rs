//! Sticky placement of new sessions for the consistent hash policy.
//!
//! Opt-in with `VLLM_ROUTER_SESSION_PLACEMENT=least_tokens` or `least_inflight`.
//! A session key seen for the first time is placed on a healthy worker chosen by
//! load instead of the worker the hash ring names. Later requests of that
//! session stay on the same worker for prefix-cache reuse.
//!
//! - `least_tokens`: the worker whose sessions hold the fewest context tokens.
//! - `least_inflight`: the worker with the fewest routed requests in flight, then
//!   the fewest counted sessions, then the fewest tokens (veRL's default rule:
//!   first turn to the least in-flight server, later turns sticky). A placement
//!   counts its request in flight under the same lock, so a burst sees its own
//!   placements; the request's lease, taken right after, settles it. Every
//!   keyed request the regular Router places is leased; a caller that places
//!   without leasing (prefill/decode routing) must not use this rule.
//!
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

/// How a new or released session picks its worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementRule {
    /// Fewest counted context tokens, then fewest counted sessions.
    LeastTokens,
    /// Fewest requests in flight, then fewest counted sessions, then tokens.
    LeastInflight,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Load {
    tokens: u64,
    sessions: u64,
    /// Routed requests not yet settled: the sum of `Session::inflight` over the
    /// sessions this worker owns.
    inflight: u64,
}

impl Load {
    fn rank(&self, rule: PlacementRule) -> (u64, u64, u64) {
        match rule {
            PlacementRule::LeastTokens => (self.tokens, self.sessions, 0),
            PlacementRule::LeastInflight => (self.inflight, self.sessions, self.tokens),
        }
    }
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

    /// Count one more request of `key` in flight on its worker.
    fn begin_request(&mut self, key: &str) {
        let Some(session) = self.sessions.get_mut(key) else {
            return;
        };
        session.inflight += 1;
        self.loads
            .entry(session.worker.clone())
            .or_default()
            .inflight += 1;
    }

    /// Settle one request of `key` (completed, failed, aborted or dropped).
    fn end_request(&mut self, key: &str) {
        let Some(session) = self.sessions.get_mut(key) else {
            return;
        };
        if session.inflight == 0 {
            return;
        }
        session.inflight -= 1;
        if let Some(load) = self.loads.get_mut(&session.worker) {
            load.inflight = load.inflight.saturating_sub(1);
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

/// Sticky session placement shared by all requests of one Router.
#[derive(Debug)]
pub struct TokenPlacement {
    rule: PlacementRule,
    idle: Duration,
    state: Mutex<State>,
}

impl TokenPlacement {
    /// Least-tokens placement.
    pub fn new(idle: Duration) -> Self {
        Self::with_rule(PlacementRule::LeastTokens, idle)
    }

    pub fn with_rule(rule: PlacementRule, idle: Duration) -> Self {
        Self {
            rule,
            idle,
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
            Some("least_inflight") => PlacementRule::LeastInflight,
            Some(other) => panic!(
                "{PLACEMENT_ENV} must be 'hash', 'least_tokens' or 'least_inflight', got {other:?}"
            ),
        };
        let idle = match std::env::var(IDLE_SECS_ENV).ok().as_deref() {
            None | Some("") => DEFAULT_IDLE_SECS,
            Some(value) => match value.parse::<u64>() {
                Ok(secs) if secs > 0 => secs,
                _ => panic!("{IDLE_SECS_ENV} must be a positive integer, got {value:?}"),
            },
        };
        Some(Self::with_rule(rule, Duration::from_secs(idle)))
    }

    /// Return the worker for session `key`. A new session, or one whose worker
    /// left `healthy`, goes to the healthy worker ranked lowest by the rule
    /// (ties to the earliest in `healthy`). `healthy` must not be empty. Call
    /// [`TokenPlacement::lease`] right after and hold the lease until the routed
    /// request settles; under `least_inflight` the placement has already
    /// counted that request in flight and the lease settles it.
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
        let exact = prompt_token_count(request_text);
        let estimate = request_text.map_or(0, |text| text.len() as u64 / 4);
        let mut state = self.state.lock().unwrap();
        self.sweep(&mut state, now);

        let owner_gone = state.sessions.get(key).is_some_and(|session| {
            !healthy.contains(&session.worker.as_str())
                || (session.released && session.inflight == 0)
        });
        if owner_gone || !state.sessions.contains_key(key) {
            // Re-placing an existing session keeps its in-flight count, so its
            // outstanding leases still settle against it; the count moves with it.
            state.set_counted(key, false);
            let worker = healthy
                .iter()
                .min_by_key(|worker| {
                    let load = state.loads.get(**worker).copied().unwrap_or_default();
                    load.rank(self.rule)
                })
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
            let previous = std::mem::replace(&mut session.worker, worker.clone());
            let inflight = u64::from(session.inflight);
            session.released = false;
            if inflight > 0 {
                let load = state.loads.entry(previous).or_default();
                load.inflight = load.inflight.saturating_sub(inflight);
                state.loads.entry(worker).or_default().inflight += inflight;
            }
        }

        state.set_counted(key, false);
        let session = state.sessions.get_mut(key).unwrap();
        session.tokens = exact.unwrap_or(session.tokens.max(estimate));
        session.last_seen = now;
        let worker = session.worker.clone();
        state.set_counted(key, true);
        if self.rule == PlacementRule::LeastInflight {
            state.begin_request(key);
        }
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

    /// Mark one request of `key` in flight; `None` when `key` was never placed.
    pub fn lease(self: &Arc<Self>, key: &str) -> Option<SessionLease> {
        self.lease_at(key, Instant::now())
    }

    fn lease_at(self: &Arc<Self>, key: &str, now: Instant) -> Option<SessionLease> {
        let mut state = self.state.lock().unwrap();
        let session = state.sessions.get_mut(key)?;
        session.last_seen = now;
        if self.rule != PlacementRule::LeastInflight {
            state.begin_request(key);
        }
        state.set_counted(key, true);
        Some(SessionLease {
            placement: Arc::clone(self),
            key: key.to_string(),
        })
    }

    fn settle_at(&self, key: &str, now: Instant) {
        let mut state = self.state.lock().unwrap();
        state.end_request(key);
        if let Some(session) = state.sessions.get_mut(key) {
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

    /// Routed requests in flight per worker, for diagnostics and tests.
    pub fn worker_inflight(&self) -> HashMap<String, u64> {
        let state = self.state.lock().unwrap();
        state
            .loads
            .iter()
            .map(|(worker, load)| (worker.clone(), load.inflight))
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

    fn inflight_on(placement: &TokenPlacement, worker: &str) -> u64 {
        placement
            .worker_inflight()
            .get(worker)
            .copied()
            .unwrap_or(0)
    }

    fn least_inflight() -> Arc<TokenPlacement> {
        Arc::new(TokenPlacement::with_rule(
            PlacementRule::LeastInflight,
            Duration::from_secs(900),
        ))
    }

    /// Place and lease one request, as the Router does for every keyed request.
    fn dispatch(
        placement: &Arc<TokenPlacement>,
        key: &str,
        text: &str,
        healthy: &[&str],
        now: Instant,
    ) -> (String, SessionLease) {
        let worker = placement.place_at(key, Some(text), healthy, now);
        (worker, placement.lease_at(key, now).unwrap())
    }

    /// veRL's rule: a new session goes to the worker with the fewest requests
    /// in flight, and every later turn sticks to it.
    #[test]
    fn least_inflight_places_new_sessions_on_the_least_busy_worker() {
        let placement = least_inflight();
        let workers = ["a", "b", "c"];
        let now = Instant::now();
        let mut leases = Vec::new();
        let (heavy, lease) = dispatch(&placement, "heavy", &body(10), &workers, now);
        leases.push(lease);
        for turn in 0..2 {
            let (worker, lease) = dispatch(&placement, "heavy", &body(20 + turn), &workers, now);
            assert_eq!(worker, heavy);
            leases.push(lease);
        }
        let (mid, lease) = dispatch(&placement, "mid", &body(10), &workers, now);
        leases.push(lease);
        assert_ne!(mid, heavy);
        assert_eq!(inflight_on(&placement, &heavy), 3);
        let idle = *workers.iter().find(|w| **w != heavy && **w != mid).unwrap();
        let (new, lease) = dispatch(&placement, "new", &body(10), &workers, now);
        assert_eq!(new, idle);
        leases.push(lease);
        drop(leases);
        assert!(placement.worker_inflight().values().all(|&n| n == 0));
    }

    /// The 103276 shape: half the workers hold only long sessions that are idle
    /// in tool calls (settled, still counted), the other half run short
    /// sessions. A burst of new sessions spreads by requests in flight to
    /// within one request per worker, reaching the idle long-session workers;
    /// least tokens sends the whole burst to the short-session workers.
    #[test]
    fn a_burst_spreads_by_inflight_past_idle_long_sessions() {
        let names: Vec<String> = (0..8).map(|i| format!("w{i}")).collect();
        let workers: Vec<&str> = names.iter().map(String::as_str).collect();
        let run = |rule| {
            let placement = Arc::new(TokenPlacement::with_rule(rule, Duration::from_secs(900)));
            let start = Instant::now();
            let mut leases = Vec::new();
            for (i, worker) in workers.iter().enumerate() {
                if i < 4 {
                    let key = format!("long{i}");
                    let (_, lease) = dispatch(&placement, &key, &body(100_000), &[*worker], start);
                    settle(&placement, lease, start);
                } else {
                    for j in 0..2 {
                        let key = format!("short{i}-{j}");
                        let (_, lease) =
                            dispatch(&placement, &key, &body(2_000), &[*worker], start);
                        leases.push(lease);
                    }
                }
            }
            let burst_at = start + Duration::from_secs(60);
            let mut burst = vec![0u64; workers.len()];
            for n in 0..24 {
                let key = format!("new{n}");
                let (worker, lease) = dispatch(&placement, &key, &body(3_000), &workers, burst_at);
                burst[workers.iter().position(|w| *w == worker).unwrap()] += 1;
                leases.push(lease);
            }
            let inflight: Vec<u64> = workers.iter().map(|w| inflight_on(&placement, w)).collect();
            for lease in leases {
                settle(&placement, lease, burst_at);
            }
            assert!(
                placement.worker_inflight().values().all(|&n| n == 0),
                "every settled request leaves the count"
            );
            (burst, inflight)
        };
        let (burst, inflight) = run(PlacementRule::LeastInflight);
        let (max, min) = (
            inflight.iter().max().unwrap(),
            inflight.iter().min().unwrap(),
        );
        assert!(max - min <= 1, "{inflight:?}");
        assert!(burst[..4].iter().all(|&n| n > 0), "{burst:?}");
        let (by_tokens, _) = run(PlacementRule::LeastTokens);
        assert_eq!(by_tokens[..4].iter().sum::<u64>(), 0, "{by_tokens:?}");
    }

    /// Placement counts the request under the placement lock, so requests
    /// placed before any of them is leased still see each other.
    #[test]
    fn least_inflight_counts_a_request_when_it_is_placed() {
        let placement = least_inflight();
        let workers = ["a", "b", "c", "d"];
        let now = Instant::now();
        let placed: Vec<String> = (0..4)
            .map(|n| placement.place_at(&format!("s{n}"), Some(&body(10)), &workers, now))
            .collect();
        let mut distinct = placed.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), 4, "{placed:?}");
        let leases: Vec<SessionLease> = (0..4)
            .map(|n| placement.lease_at(&format!("s{n}"), now).unwrap())
            .collect();
        // The lease holds the placement's count instead of adding another.
        assert!(workers.iter().all(|w| inflight_on(&placement, w) == 1));
        drop(leases);
        assert!(workers.iter().all(|w| inflight_on(&placement, w) == 0));
    }

    /// A generation and its abort both count against the owner while in
    /// flight: the abort reaches the owner although it is the busier worker,
    /// and once both settle the owner is the least busy worker again.
    #[test]
    fn least_inflight_abort_reaches_the_owner_and_both_settle() {
        let placement = least_inflight();
        let workers = ["a", "b"];
        let now = Instant::now();
        let (owner, generation) = dispatch(&placement, "s", &body(500), &workers, now);
        let other = if owner == "a" { "b" } else { "a" };
        let abort_body = "{\"request_id\":\"r\"}";
        let (target, abort) = dispatch(&placement, "s", abort_body, &workers, now);
        assert_eq!(target, owner);
        assert_eq!(inflight_on(&placement, &owner), 2);
        drop(abort);
        assert_eq!(inflight_on(&placement, &owner), 1);
        // The aborted generation returns, or its client disconnects: its lease drops.
        drop(generation);
        assert_eq!(inflight_on(&placement, &owner), 0);
        // Idle and tied on requests, the next session goes to the worker with fewer sessions.
        let (first, running) = dispatch(&placement, "t", &body(10), &workers, now);
        assert_eq!(first, other);
        let (second, _second) = dispatch(&placement, "u", &body(10), &workers, now);
        assert_eq!(second, owner);
        drop(running);
    }

    /// A weight update keeps an in-flight session on its owner and re-places it
    /// by requests in flight once idle; a session whose owner leaves the healthy
    /// set carries its in-flight count, so every lease still settles to zero.
    #[test]
    fn least_inflight_re_placement_keeps_the_count_consistent() {
        let placement = least_inflight();
        let workers = ["a", "b"];
        let now = Instant::now();
        let (owner, generation) = dispatch(&placement, "s", &body(100), &workers, now);
        let other = if owner == "a" { "b" } else { "a" };
        let (_, busy) = dispatch(&placement, "u", &body(100), &[owner.as_str()], now);
        assert_eq!(placement.release_owners(), 2);
        let abort_body = "{\"request_id\":\"r\"}";
        let (target, abort) = dispatch(&placement, "s", abort_body, &workers, now);
        assert_eq!(target, owner);
        settle(&placement, abort, now);
        settle(&placement, generation, now);
        // Idle after the release: the resumed turn goes to the worker with fewer in flight.
        let (moved, resumed) = dispatch(&placement, "s", &body(150), &workers, now);
        assert_eq!(moved, other);
        // Its new owner fails with the turn in flight: the count moves with the session.
        let (back, retry) = dispatch(&placement, "s", &body(150), &[owner.as_str()], now);
        assert_eq!(back, owner);
        assert_eq!(inflight_on(&placement, other), 0);
        assert_eq!(inflight_on(&placement, &owner), 3);
        drop((busy, resumed, retry));
        assert!(placement.worker_inflight().values().all(|&n| n == 0));
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
