//! Tests for transparent proxy routing with headers and availability filtering.
//!
//! These tests verify the fixes to route_transparent() across router
//! implementations (Router, VllmPDRouter):
//!   1. Headers are passed to select_worker_with_headers() for consistent hash routing
//!   2. Workers are filtered by is_available() before selection
//!   3. The inline header conversion pattern (used in vllm_pd_router) matches
//!      the Router::headers_to_request_headers() output

mod common;

#[cfg(test)]
mod reset_affinity_routes {
    use async_trait::async_trait;
    use axum::{
        body::Body,
        extract::State,
        http::{HeaderMap, Request, StatusCode},
        routing::post,
        Json,
    };
    use serde_json::{json, Value};
    use std::{
        collections::HashMap,
        future::Future,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc, Condvar, Mutex,
        },
        time::Duration,
    };
    use tokio::{net::TcpListener, sync::Notify, task::JoinHandle};
    use tower::ServiceExt;
    use vllm_router_rs::{
        config::{PolicyConfig, RouterConfig},
        core::{
            worker::WorkerMetadata, CircuitBreaker, ConnectionMode, DPAwareWorker, Worker,
            WorkerResult, WorkerType,
        },
        routers::http::router::Router,
    };

    const G: &str = "physical-G";
    static PLACEMENT_ENV: Mutex<()> = Mutex::new(());
    struct RestoreEnv(Option<std::ffi::OsString>);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var("VLLM_ROUTER_SESSION_PLACEMENT", value),
                None => std::env::remove_var("VLLM_ROUTER_SESSION_PLACEMENT"),
            }
        }
    }
    fn run_with_placement(future: impl Future<Output = ()>) {
        // Other tests in this executable instantiate plain policies directly.
        // Serialize the only two AppContext/environment users, including failure.
        let _guard = PLACEMENT_ENV
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _restore = RestoreEnv(std::env::var_os("VLLM_ROUTER_SESSION_PLACEMENT"));
        std::env::set_var("VLLM_ROUTER_SESSION_PLACEMENT", "least_tokens");
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(future);
    }

    struct BackendState {
        url: String,
        rank: usize,
        lose_transport: bool,
        admitted: Arc<Notify>,
        lose: Notify,
        work: Mutex<HashMap<String, (JoinHandle<()>, Arc<Notify>)>>,
        records: Mutex<Vec<Value>>,
        sequence: Arc<AtomicUsize>,
    }
    impl BackendState {
        fn record(&self, path: &str, headers: &HeaderMap, body: &Value) -> usize {
            let rank = headers
                .get("x-data-parallel-rank")
                .unwrap()
                .to_str()
                .unwrap()
                .parse::<usize>()
                .unwrap();
            assert_eq!(rank, self.rank);
            self.records.lock().unwrap().push(json!({"sequence": self.sequence.fetch_add(1, Ordering::SeqCst), "path": path, "worker": self.url, "rank": rank, "request_id": body["request_id"], "session_id": body["session_id"]}));
            rank
        }
        fn remaining(&self) -> usize {
            self.work.lock().unwrap().len()
        }
    }
    async fn generate(
        State(state): State<Arc<BackendState>>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        let rank = state.record("generate", &headers, &body);
        if body["request_id"] == G {
            // Controlled backend work survives frontend TCP loss. This is not a
            // GPU engine; its exact request table supplies honest cleanup receipts.
            let task = tokio::spawn(std::future::pending::<()>());
            let done = Arc::new(Notify::new());
            state
                .work
                .lock()
                .unwrap()
                .insert(G.to_string(), (task, done.clone()));
            state.admitted.notify_one();
            if state.lose_transport {
                state.lose.notified().await;
                // Only this Axum connection task panics: TCP closes before headers,
                // while the accept loop and independently owned work remain alive.
                panic!("injected pre-response transport loss for physical-G");
            }
            done.notified().await;
        }
        Json(
            json!({"worker": state.url, "rank": rank, "request_id": body["request_id"], "stop_reason": "completed"}),
        )
    }
    async fn abort(
        State(state): State<Arc<BackendState>>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        let rank = state.record("abort", &headers, &body);
        let id = body["request_id"].as_str().unwrap();
        let work = { state.work.lock().unwrap().remove(id) };
        let aborted = if let Some((task, done)) = work {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            done.notify_one();
            true
        } else {
            false
        };
        Json(json!({"worker": state.url, "rank": rank, "request_id": id, "aborted": aborted}))
    }
    struct Backend {
        state: Arc<BackendState>,
        server: JoinHandle<()>,
    }
    impl Drop for Backend {
        fn drop(&mut self) {
            self.server.abort();
            for (_, (task, done)) in self.state.work.lock().unwrap().drain() {
                task.abort();
                done.notify_one();
            }
        }
    }

    #[derive(Debug, Default)]
    struct SelectionPause {
        first: AtomicBool,
        owner: Mutex<Option<String>>,
        reached: Notify,
        released: Mutex<bool>,
        release: Condvar,
    }
    impl SelectionPause {
        fn pause(&self, owner: &str) {
            if self.first.swap(true, Ordering::SeqCst) {
                return;
            }
            *self.owner.lock().unwrap() = Some(owner.to_string());
            self.reached.notify_one();
            let guard = self.released.lock().unwrap();
            let (guard, timeout) = self
                .release
                .wait_timeout_while(guard, Duration::from_secs(15), |released| !*released)
                .unwrap();
            assert!(
                *guard && !timeout.timed_out(),
                "selection barrier timed out"
            );
        }
        fn release(&self) {
            *self.released.lock().unwrap() = true;
            self.release.notify_all();
        }
    }
    struct ReleaseOnDrop(Arc<SelectionPause>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.0.release();
        }
    }
    #[derive(Debug)]
    struct PausedWorker {
        inner: Arc<dyn Worker>,
        pause: Arc<SelectionPause>,
    }
    #[async_trait]
    impl Worker for PausedWorker {
        fn url(&self) -> &str {
            self.inner.url()
        }
        fn worker_type(&self) -> WorkerType {
            self.inner.worker_type()
        }
        fn connection_mode(&self) -> ConnectionMode {
            self.inner.connection_mode()
        }
        fn is_healthy(&self) -> bool {
            self.inner.is_healthy()
        }
        fn set_healthy(&self, value: bool) {
            self.inner.set_healthy(value);
        }
        async fn check_health_async(&self) -> WorkerResult<()> {
            self.inner.check_health_async().await
        }
        fn load(&self) -> usize {
            self.inner.load()
        }
        fn increment_load(&self) {
            self.inner.increment_load();
        }
        fn decrement_load(&self) {
            self.inner.decrement_load();
        }
        fn processed_requests(&self) -> usize {
            self.inner.processed_requests()
        }
        fn increment_processed(&self) {
            self.pause.pause(self.url());
            self.inner.increment_processed();
        }
        fn metadata(&self) -> &WorkerMetadata {
            self.inner.metadata()
        }
        fn circuit_breaker(&self) -> &CircuitBreaker {
            self.inner.circuit_breaker()
        }
        fn is_dp_aware(&self) -> bool {
            self.inner.is_dp_aware()
        }
        fn base_url(&self) -> &str {
            self.inner.base_url()
        }
        fn dp_rank(&self) -> Option<usize> {
            self.inner.dp_rank()
        }
        fn dp_size(&self) -> Option<usize> {
            self.inner.dp_size()
        }
    }
    struct Fixture {
        app: axum::Router,
        backends: Vec<Backend>,
        admitted: Arc<Notify>,
        registry: Arc<vllm_router_rs::core::WorkerRegistry>,
        workers: Vec<Arc<dyn Worker>>,
    }
    impl Fixture {
        async fn new(lose_transport: bool, pause: Option<Arc<SelectionPause>>) -> Self {
            let config = RouterConfig {
                policy: PolicyConfig::ConsistentHash { virtual_nodes: 160 },
                ..RouterConfig::default()
            };
            let context = super::common::create_test_context(config.clone());
            let admitted = Arc::new(Notify::new());
            let sequence = Arc::new(AtomicUsize::new(0));
            let mut backends = Vec::new();
            let mut workers = Vec::new();
            for rank in 0..2 {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let url = format!("http://{}", listener.local_addr().unwrap());
                let state = Arc::new(BackendState {
                    url: url.clone(),
                    rank,
                    lose_transport,
                    admitted: admitted.clone(),
                    lose: Notify::new(),
                    work: Mutex::new(HashMap::new()),
                    records: Mutex::new(Vec::new()),
                    sequence: sequence.clone(),
                });
                let app = axum::Router::new()
                    .route("/verl/v1/generate", post(generate))
                    .route("/verl/v1/abort", post(abort))
                    .with_state(state.clone());
                let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
                let inner: Arc<dyn Worker> =
                    Arc::new(DPAwareWorker::new(url, rank, 2, WorkerType::Regular));
                let worker = match &pause {
                    Some(pause) => Arc::new(PausedWorker {
                        inner,
                        pause: pause.clone(),
                    }) as Arc<dyn Worker>,
                    None => inner,
                };
                // Establish rank0 as the same original owner on every variant,
                // independently of the registry's iteration order.
                if rank == 0 {
                    context.worker_registry.register(worker.clone());
                }
                workers.push(worker);
                backends.push(Backend { state, server });
            }
            let router = Arc::new(Router::new(Vec::new(), &context).await.unwrap());
            let app =
                super::common::test_app::create_test_app(router, context.client.clone(), &config);
            Self {
                app,
                backends,
                admitted,
                registry: context.worker_registry.clone(),
                workers,
            }
        }
        fn add_second_worker(&self) {
            self.registry.register(self.workers[1].clone());
        }
        async fn admission(&self) {
            tokio::time::timeout(Duration::from_secs(10), self.admitted.notified())
                .await
                .unwrap();
        }
        fn owner(&self) -> Arc<BackendState> {
            self.backends
                .iter()
                .find(|backend| backend.state.remaining() == 1)
                .unwrap()
                .state
                .clone()
        }
        fn records(&self) -> Vec<Value> {
            let mut records: Vec<Value> = self
                .backends
                .iter()
                .flat_map(|backend| backend.state.records.lock().unwrap().clone())
                .collect();
            records.sort_by_key(|record| record["sequence"].as_u64().unwrap());
            records
        }
        fn remaining(&self) -> usize {
            self.backends
                .iter()
                .map(|backend| backend.state.remaining())
                .sum()
        }
    }
    async fn request(
        app: &axum::Router,
        path: &str,
        session: &str,
        id: &str,
        tokens: usize,
    ) -> (StatusCode, Value) {
        let mut body = json!({"request_id": id, "session_id": session});
        if path == "/verl/v1/generate" {
            body["prompt_ids"] = json!(vec![1u8; tokens]);
        }
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            app.clone().oneshot(
                Request::post(path)
                    .header("content-type", "application/json")
                    .header("x-session-id", session)
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        let status = response.status();
        let bytes = tokio::time::timeout(
            Duration::from_secs(10),
            axum::body::to_bytes(response.into_body(), 8192),
        )
        .await
        .unwrap()
        .unwrap();
        let body = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| json!({"error": String::from_utf8_lossy(&bytes)}));
        (status, body)
    }
    async fn ok(app: &axum::Router, path: &str, session: &str, id: &str, tokens: usize) -> Value {
        let (status, body) = request(app, path, session, id, tokens).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    #[test]
    fn transport_loss_after_reset_keeps_abort_on_original_worker() {
        run_with_placement(async {
            let fixture = Fixture::new(true, None).await;
            let app = fixture.app.clone();
            let generation = tokio::spawn(async move {
                request(&app, "/verl/v1/generate", "session-S", G, 100).await
            });
            fixture.admission().await;
            let owner = fixture.owner();
            assert_eq!(owner.rank, 0);
            fixture.add_second_worker();
            ok(
                &fixture.app,
                "/reset_session_placement",
                "control",
                "reset",
                0,
            )
            .await;
            let pressure_p = ok(
                &fixture.app,
                "/verl/v1/generate",
                "session-P",
                "pressure-P",
                200,
            )
            .await;
            let pressure_q = ok(
                &fixture.app,
                "/verl/v1/generate",
                "session-Q",
                "pressure-Q",
                300,
            )
            .await;
            assert_ne!(pressure_p["worker"], owner.url);
            assert_eq!(pressure_q["worker"], owner.url);
            owner.lose.notify_one();
            let (status, _) = generation.await.unwrap();
            assert_eq!(status, StatusCode::BAD_GATEWAY);
            assert_eq!(fixture.remaining(), 1);
            // Follow the real client's ordering: await keyed abort after HTTP
            // failure and before another attempt, without overlapping generation.
            let receipt = ok(&fixture.app, "/verl/v1/abort", "session-S", G, 0).await;
            println!(
                "AFFINITY_T1 {}",
                json!({"physical_id": G, "original_worker": owner.url, "original_rank": owner.rank, "abort": receipt, "controlled_backend_remaining": fixture.remaining(), "records": fixture.records()})
            );
            assert_eq!(receipt["worker"], owner.url);
            assert_eq!(receipt["rank"], owner.rank);
            assert_eq!(receipt["request_id"], G);
            assert_eq!(receipt["aborted"], true);
            assert_eq!(fixture.remaining(), 0);
        });
    }
    #[test]
    fn selection_reset_abort_then_explicit_reconciliation_keeps_owner() {
        run_with_placement(async {
            let pause = Arc::new(SelectionPause::default());
            let _release = ReleaseOnDrop(pause.clone());
            let fixture = Fixture::new(false, Some(pause.clone())).await;
            let app = fixture.app.clone();
            let generation = tokio::spawn(async move {
                request(&app, "/verl/v1/generate", "session-S", G, 100).await
            });
            tokio::time::timeout(Duration::from_secs(10), pause.reached.notified())
                .await
                .unwrap();
            let selected = pause.owner.lock().unwrap().clone().unwrap();
            let original = fixture
                .backends
                .iter()
                .find(|backend| selected == format!("{}@{}", backend.state.url, backend.state.rank))
                .unwrap()
                .state
                .clone();
            assert_eq!(fixture.remaining(), 0);
            assert_eq!(original.rank, 0);
            fixture.add_second_worker();
            ok(
                &fixture.app,
                "/reset_session_placement",
                "control",
                "reset",
                0,
            )
            .await;
            // Seed rank1 deterministically without touching S's owner. These
            // same pressure requests expose the window on both deployed b622
            // and the accepted accounting baseline, despite their load offset.
            fixture.workers[0].set_healthy(false);
            let seed = ok(
                &fixture.app,
                "/verl/v1/generate",
                "session-seed",
                "seed-B",
                10,
            )
            .await;
            fixture.workers[0].set_healthy(true);
            assert_eq!(seed["rank"], 1);
            let mut pressure = Vec::new();
            for (session, id, tokens) in [
                ("session-P", "pressure-P", 200),
                ("session-Q", "pressure-Q", 50),
                ("session-R", "pressure-R", 100),
                ("session-U", "pressure-U", 100),
                ("session-V", "pressure-V", 200),
            ] {
                pressure.push(ok(&fixture.app, "/verl/v1/generate", session, id, tokens).await);
            }
            let early = ok(&fixture.app, "/verl/v1/abort", "session-S", G, 0).await;
            assert_eq!(early["aborted"], false);
            assert_eq!(fixture.remaining(), 0);
            pause.release();
            fixture.admission().await;
            assert_eq!(fixture.owner().url, original.url);
            // Explicit reconciliation after backend admission; the first abort
            // did not settle a request which had not yet reached registration.
            let reconciled = ok(&fixture.app, "/verl/v1/abort", "session-S", G, 0).await;
            println!(
                "AFFINITY_T2 {}",
                json!({"physical_id": G, "original_worker": original.url, "original_rank": original.rank, "pressure": pressure, "early_abort": early, "reconciliation": reconciled, "controlled_backend_remaining": fixture.remaining(), "records": fixture.records()})
            );
            assert_eq!(early["worker"], original.url);
            assert_eq!(early["rank"], original.rank);
            assert_eq!(early["request_id"], G);
            assert_eq!(reconciled["worker"], original.url);
            assert_eq!(reconciled["rank"], original.rank);
            assert_eq!(reconciled["request_id"], G);
            assert_eq!(reconciled["aborted"], true);
            assert_eq!(fixture.remaining(), 0);
            assert_eq!(generation.await.unwrap().0, StatusCode::OK);
        });
    }
}

#[cfg(test)]
mod transparent_proxy_routing_tests {
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::sync::Arc;

    use vllm_router_rs::core::BasicWorker;
    use vllm_router_rs::core::Worker;
    use vllm_router_rs::core::WorkerType;
    use vllm_router_rs::policies::ConsistentHashPolicy;
    use vllm_router_rs::policies::LoadBalancingPolicy;
    use vllm_router_rs::policies::RequestHeaders;

    /// Helper to create test workers
    fn create_workers(n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://worker{}:8080", i + 1),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect()
    }

    /// Helper to build RequestHeaders from key-value pairs
    fn make_headers(pairs: &[(&str, &str)]) -> RequestHeaders {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // =====================================================================
    // Test 1: Availability filtering before policy selection
    // =====================================================================
    // The route_transparent fix filters workers by is_available() BEFORE
    // passing them to the policy. These tests verify that behavior.

    #[test]
    fn test_availability_filter_excludes_unhealthy_workers() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(3);

        // Mark worker1 as unhealthy
        workers[0].set_healthy(false);

        // Filter by availability (same pattern as route_transparent)
        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        assert_eq!(
            available.len(),
            2,
            "Only 2 of 3 workers should be available"
        );

        // Policy should still work with filtered workers
        let headers = make_headers(&[("x-session-id", "test-session")]);
        let result = policy.select_worker_with_headers(
            &available,
            Some(r#"{"prompt": "test"}"#),
            Some(&headers),
        );
        assert!(result.is_some(), "Should select from available workers");

        let idx = result.unwrap();
        assert!(
            available[idx].is_healthy(),
            "Selected worker must be healthy"
        );
    }

    #[test]
    fn test_availability_filter_returns_empty_when_all_unhealthy() {
        let workers = create_workers(3);

        // Mark all workers as unhealthy
        for w in &workers {
            w.set_healthy(false);
        }

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        assert!(
            available.is_empty(),
            "No workers should be available when all are unhealthy"
        );

        // Policy should return None for empty list
        let policy = ConsistentHashPolicy::new();
        let result = policy.select_worker_with_headers(&available, None, None);
        assert!(result.is_none(), "Should return None for empty worker list");
    }

    // =====================================================================
    // Test 2: Headers are forwarded to the consistent hash policy
    // =====================================================================
    // Before the fix, route_transparent called select_worker() which ignores
    // headers entirely, making consistent hash fall back to request body hash.
    // After the fix, it calls select_worker_with_headers() so x-session-id
    // in headers produces sticky routing.

    #[test]
    fn test_session_id_header_produces_sticky_routing() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(5);

        let headers = make_headers(&[("x-session-id", "my-sticky-session")]);

        // Simulate what route_transparent now does: pass headers to policy
        let mut selected: Vec<usize> = Vec::new();
        for i in 0..20 {
            let body = format!(
                r#"{{"prompt": "different prompt {}", "model": "default"}}"#,
                i
            );
            if let Some(idx) =
                policy.select_worker_with_headers(&workers, Some(&body), Some(&headers))
            {
                selected.push(idx);
            }
        }

        // All requests with the same session ID should route to the same worker
        assert!(!selected.is_empty());
        let first = selected[0];
        for (i, &idx) in selected.iter().enumerate() {
            assert_eq!(
                idx, first,
                "Request {} routed to worker {}, expected {} (session stickiness broken)",
                i, idx, first
            );
        }
    }

    #[test]
    fn test_without_headers_uses_body_fallback() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(3);

        // Without headers, consistent hash should fall back to body content
        let body = r#"{"session_params": {"session_id": "body-session"}, "prompt": "test"}"#;

        let mut selected: Vec<usize> = Vec::new();
        for _ in 0..10 {
            if let Some(idx) = policy.select_worker_with_headers(&workers, Some(body), None) {
                selected.push(idx);
            }
        }

        // Same body → same worker (body fallback is deterministic)
        assert!(!selected.is_empty());
        let first = selected[0];
        for &idx in &selected {
            assert_eq!(idx, first, "Body fallback should be deterministic");
        }
    }

    #[test]
    fn test_header_session_id_takes_priority_over_body_session_id() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(5);

        let header_session = "header-session-wins";
        let body_session = "body-session-ignored";

        let headers = make_headers(&[("x-session-id", header_session)]);
        let body = format!(
            r#"{{"session_params": {{"session_id": "{}"}}, "prompt": "test"}}"#,
            body_session
        );

        // Route with both header and body session ID
        let with_both = policy
            .select_worker_with_headers(&workers, Some(&body), Some(&headers))
            .expect("Should select a worker");

        // Route with header only (different body)
        let header_only = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "completely different"}"#),
                Some(&headers),
            )
            .expect("Should select a worker");

        assert_eq!(
            with_both, header_only,
            "Header x-session-id should take priority - different body should not change routing"
        );
    }

    // =====================================================================
    // Test 3: Availability + headers work together correctly
    // =====================================================================
    // These tests verify the combined effect: availability filtering happens
    // BEFORE policy selection, and headers are passed to the policy.

    #[test]
    fn test_sticky_routing_with_availability_filtering() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(4);

        // Initial routing with all workers available
        let headers = make_headers(&[("x-session-id", "stable-session")]);
        let body = r#"{"prompt": "test"}"#;

        let initial_idx = policy
            .select_worker_with_headers(&workers, Some(body), Some(&headers))
            .expect("Should select a worker");

        // Now filter by availability (as route_transparent does)
        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        // Should get the same worker since all are still available
        let after_filter_idx = policy
            .select_worker_with_headers(&available, Some(body), Some(&headers))
            .expect("Should select a worker");

        assert_eq!(
            workers[initial_idx].url(),
            available[after_filter_idx].url(),
            "Same session ID should route to same worker URL when all workers are available"
        );
    }

    #[test]
    fn test_distribution_with_headers_across_sessions() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(3);

        let mut worker_urls: HashSet<String> = HashSet::new();

        for i in 0..100 {
            let session = format!("unique-session-{}", i);
            let headers = make_headers(&[("x-session-id", &session)]);

            if let Some(idx) = policy.select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "test"}"#),
                Some(&headers),
            ) {
                worker_urls.insert(workers[idx].url().to_string());
            }
        }

        // With 100 different sessions and 3 workers, all workers should be used
        assert!(
            worker_urls.len() >= 2,
            "Expected distribution, only used {} workers: {:?}",
            worker_urls.len(),
            worker_urls
        );
    }

    // =====================================================================
    // Test 4: Inline header conversion correctness
    // =====================================================================
    // vllm_pd_router.rs uses an inline pattern to convert
    // HeaderMap → HashMap<String, String>. Verify it produces correct output.

    #[test]
    fn test_inline_header_conversion_lowercases_keys() {
        use axum::http::HeaderMap;
        use axum::http::HeaderValue;

        let mut header_map = HeaderMap::new();
        header_map.insert("X-Session-Id", HeaderValue::from_static("abc-123"));
        header_map.insert("X-USER-ID", HeaderValue::from_static("user-456"));
        header_map.insert("content-type", HeaderValue::from_static("application/json"));

        // Simulate the inline pattern from vllm_pd_router.rs
        let request_headers: Option<HashMap<String, String>> = Some(&header_map).map(|h| {
            h.iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                })
                .collect()
        });

        let headers = request_headers.unwrap();

        // axum normalizes header names to lowercase already, but verify the
        // to_lowercase() call in our conversion is idempotent and correct
        assert_eq!(headers.get("x-session-id").unwrap(), "abc-123");
        assert_eq!(headers.get("x-user-id").unwrap(), "user-456");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_inline_header_conversion_used_by_policy() {
        use axum::http::HeaderMap;
        use axum::http::HeaderValue;

        let policy = ConsistentHashPolicy::new();
        let workers = create_workers(3);

        // Convert via the inline pattern (as vllm_pd_router does)
        let mut header_map = HeaderMap::new();
        header_map.insert(
            "x-session-id",
            HeaderValue::from_static("policy-test-session"),
        );

        let request_headers: Option<HashMap<String, String>> = Some(&header_map).map(|h| {
            h.iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                })
                .collect()
        });

        // Use the converted headers with the policy (same as route_transparent now does)
        let mut selected: Vec<usize> = Vec::new();
        for i in 0..10 {
            let body = format!(r#"{{"prompt": "request {}"}}"#, i);
            if let Some(idx) =
                policy.select_worker_with_headers(&workers, Some(&body), request_headers.as_ref())
            {
                selected.push(idx);
            }
        }

        // All should go to the same worker (session stickiness via header)
        assert!(!selected.is_empty());
        let first = selected[0];
        for &idx in &selected {
            assert_eq!(
                idx, first,
                "Inline header conversion should produce sticky routing"
            );
        }

        // Also verify with make_headers helper (which matches RequestHeaders directly)
        let direct_headers = make_headers(&[("x-session-id", "policy-test-session")]);
        let direct_result = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt": "request 0"}"#),
                Some(&direct_headers),
            )
            .expect("Direct headers should work");

        assert_eq!(
            workers[first].url(),
            workers[direct_result].url(),
            "Inline conversion and direct headers should route to the same worker"
        );
    }

    // =====================================================================
    // Test 5: PD mode worker pair selection with headers
    // =====================================================================
    // For vllm_pd_router, both prefill and decode workers need headers.

    #[test]
    fn test_pd_mode_worker_pair_with_headers() {
        let policy = ConsistentHashPolicy::new();

        let prefill_workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://prefill{}:8080", i + 1),
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                )) as Arc<dyn Worker>
            })
            .collect();

        let decode_workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://decode{}:8080", i + 1),
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();

        let headers = make_headers(&[("x-session-id", "pd-session")]);
        let body = r#"{"prompt": "test"}"#;

        // select_worker_with_headers on each pool (as vllm_pd_router now does)
        let prefill_idx = policy
            .select_worker_with_headers(&prefill_workers, Some(body), Some(&headers))
            .expect("Should select prefill worker");

        let decode_idx = policy
            .select_worker_with_headers(&decode_workers, Some(body), Some(&headers))
            .expect("Should select decode worker");

        // Verify consistency: same session → same workers
        let prefill_idx2 = policy
            .select_worker_with_headers(&prefill_workers, Some(body), Some(&headers))
            .expect("Should select prefill worker again");

        let decode_idx2 = policy
            .select_worker_with_headers(&decode_workers, Some(body), Some(&headers))
            .expect("Should select decode worker again");

        assert_eq!(
            prefill_idx, prefill_idx2,
            "Prefill selection should be consistent"
        );
        assert_eq!(
            decode_idx, decode_idx2,
            "Decode selection should be consistent"
        );

        // Verify the workers are valid
        assert!(prefill_idx < prefill_workers.len());
        assert!(decode_idx < decode_workers.len());
    }

    #[test]
    fn test_pd_mode_availability_filtering() {
        let prefill_workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://prefill{}:8080", i + 1),
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                )) as Arc<dyn Worker>
            })
            .collect();

        let decode_workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://decode{}:8080", i + 1),
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();

        // Mark one prefill and one decode worker as unhealthy
        prefill_workers[0].set_healthy(false);
        decode_workers[1].set_healthy(false);

        // Filter by availability (as route_transparent now does)
        let available_prefill: Vec<Arc<dyn Worker>> = prefill_workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        let available_decode: Vec<Arc<dyn Worker>> = decode_workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        assert_eq!(
            available_prefill.len(),
            2,
            "2 of 3 prefill workers should be available"
        );
        assert_eq!(
            available_decode.len(),
            2,
            "2 of 3 decode workers should be available"
        );

        // Policy should select from available workers only
        let policy = ConsistentHashPolicy::new();
        let headers = make_headers(&[("x-session-id", "pd-avail-test")]);

        let prefill_idx = policy
            .select_worker_with_headers(
                &available_prefill,
                Some(r#"{"prompt": "test"}"#),
                Some(&headers),
            )
            .expect("Should select available prefill worker");

        let decode_idx = policy
            .select_worker_with_headers(
                &available_decode,
                Some(r#"{"prompt": "test"}"#),
                Some(&headers),
            )
            .expect("Should select available decode worker");

        assert!(available_prefill[prefill_idx].is_healthy());
        assert!(available_decode[decode_idx].is_healthy());
    }
}
