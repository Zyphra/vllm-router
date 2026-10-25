//! One retry for a native, non-streaming generation that provably never started.
//!
//! HTTP status alone, cancellation ACKs and body/read failures are not admission
//! evidence. A refused connection or a nonce-bound native rejection emitted
//! before generate() is called is required. Logical session placement stays put;
//! physical request ownership follows only this request onto its alternate.
use super::*;
use parking_lot::Mutex;
use tokio::time::{timeout_at, Instant as Deadline};

const GENERATE: &str = "/verl/v1/generate";
const ABORT: &str = "/verl/v1/abort";
const ATTEMPT: &str = "x-verl-generation-attempt";
const ADMISSION: &str = "x-verl-generation-admission";
const REQUEST_ID: &str = "x-verl-generation-request-id";
const REJECTION: &str = "x-verl-generation-rejection";

type Owners = Arc<Mutex<HashMap<String, Arc<Mutex<Ownership>>>>>;

#[derive(Debug)]
pub(super) struct NativeRetry {
    retries: u8,
    timeout: Duration,
    health_timeout: Duration,
    owners: Owners,
}

impl NativeRetry {
    pub(super) fn new(retries: u8, timeout: Duration, health_timeout: Duration) -> Self {
        Self {
            retries,
            timeout,
            health_timeout,
            owners: Arc::default(),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.retries != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admission {
    NotStarted,
    Unknown,
    Completed,
}

#[derive(Debug)]
struct Ownership {
    session: String,
    worker: Option<Arc<dyn Worker>>,
    admission: Admission,
    cancelled: bool,
    abort_confirmed: bool,
    forwarding_done: bool,
}

/// Keep ambiguous delivery quarantined until native cleanup confirms settlement.
/// Dropping a forwarding future must not forget the address needed by abort.
struct Forwarding {
    id: String,
    owner: Arc<Mutex<Ownership>>,
    owners: Owners,
}

impl Drop for Forwarding {
    fn drop(&mut self) {
        let remove = {
            let mut owner = self.owner.lock();
            owner.forwarding_done = true;
            owner.admission == Admission::Completed || owner.abort_confirmed
        };
        if remove {
            self.owners.lock().remove(&self.id);
        }
    }
}

fn identity(body: &serde_json::Value) -> Option<(&str, &str)> {
    let id = body.get("request_id")?.as_str()?;
    let session = body.get("session_id")?.as_str()?;
    if id.is_empty() || session.is_empty() {
        return None;
    }
    // IDs are echoed in admission proof headers. Refuse invalid HTTP values.
    HeaderValue::from_str(id).ok()?;
    Some((id, session))
}

fn error_response(status: StatusCode, reason: &str) -> Response {
    (status, Json(serde_json::json!({"error": reason}))).into_response()
}

fn abort_response(id: &str, confirmed: bool, error: Option<&str>) -> Response {
    Json(serde_json::json!({"request_id": id, "aborted": confirmed, "error": error}))
        .into_response()
}

fn rejected_before_generate(response: &reqwest::Response, id: &str, attempt: &str) -> bool {
    let headers = response.headers();
    response.status() == StatusCode::SERVICE_UNAVAILABLE
        && headers.get(ADMISSION).and_then(|v| v.to_str().ok()) == Some("not-started-v1")
        && headers.get(REQUEST_ID).and_then(|v| v.to_str().ok()) == Some(id)
        && headers.get(ATTEMPT).and_then(|v| v.to_str().ok()) == Some(attempt)
        && headers.get(REJECTION).and_then(|v| v.to_str().ok()) == Some("engine-dead")
}

fn stream_response(response: reqwest::Response, leases: Vec<SessionLease>) -> Response {
    let status = response.status();
    let headers = response.headers().clone();
    // Keep both the original logical-session lease and the alternate's physical
    // load lease through response delivery; never rewrite sibling placement.
    let stream = response.bytes_stream().map(move |chunk| {
        let _ = &leases;
        chunk
    });
    let mut builder = Response::builder().status(status.as_u16());
    for (name, value) in &headers {
        if name != "transfer-encoding" && name != "content-length" {
            builder = builder.header(name, value);
        }
    }
    builder.body(Body::from_stream(stream)).unwrap_or_else(|_| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to build native response",
        )
    })
}

impl Router {
    pub(super) async fn route_native_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: serde_json::Value,
    ) -> Response {
        let Some((id, session)) = identity(&body) else {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Native generation requires request_id and session_id",
            );
        };
        if headers
            .and_then(|h| h.get("x-session-id"))
            .and_then(|v| v.to_str().ok())
            .is_some_and(|value| value != session)
        {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Native routing session does not match body",
            );
        }
        let owner = Arc::new(Mutex::new(Ownership {
            session: session.to_owned(),
            worker: None,
            admission: Admission::NotStarted,
            cancelled: false,
            abort_confirmed: false,
            forwarding_done: false,
        }));
        {
            let mut owners = self.native_retry.owners.lock();
            if owners.contains_key(id) {
                return error_response(StatusCode::CONFLICT, "Native request already has an owner");
            }
            owners.insert(id.to_owned(), owner.clone());
        }
        let _forwarding = Forwarding {
            id: id.to_owned(),
            owner: owner.clone(),
            owners: self.native_retry.owners.clone(),
        };
        let deadline = Deadline::now() + self.native_retry.timeout;
        let workers: Vec<_> = self
            .worker_registry
            .get_all()
            .into_iter()
            .filter(|worker| worker.is_available())
            .collect();
        if workers.is_empty() {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "No available native worker",
            );
        }
        let request_text = serde_json::to_string(&body).ok();
        let request_headers = Self::headers_to_request_headers(headers);
        let policy = self.policy_registry.get_default_policy();
        let Some((index, lease)) = policy.select_worker_with_lease(
            &workers,
            request_text.as_deref(),
            request_headers.as_ref(),
        ) else {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "No available native worker",
            );
        };
        let mut leases: Vec<_> = lease.into_iter().collect();
        let mut worker = workers[index].clone();
        for attempt_number in 0..=self.native_retry.retries {
            let attempt = uuid::Uuid::new_v4().to_string();
            {
                let mut state = owner.lock();
                if state.cancelled {
                    return error_response(
                        StatusCode::CONFLICT,
                        "Native generation cancelled before dispatch",
                    );
                }
                // Publish the physical owner before dispatch. A racing abort
                // before native registration must return false, preserving
                // unknown delivery; an abort ACK never authorizes a retry.
                state.worker = Some(worker.clone());
                state.admission = Admission::Unknown;
            }
            let url = worker.endpoint_url(GENERATE);
            let mut request = self.client.post(&url).json(&body).header(ATTEMPT, &attempt);
            request = dp_utils::add_dp_rank_header(request, worker.dp_rank());
            if let Some(key) = &self.api_key {
                request = request.bearer_auth(key);
            }
            let result = timeout_at(
                deadline,
                otel_http::send_client_request(
                    request,
                    headers,
                    ClientRequestOptions {
                        method: "POST",
                        url: &url,
                        route: Some(GENERATE),
                        request_phase: Some("inference"),
                    },
                ),
            )
            .await;
            let evidence = match result {
                Ok(Ok(response)) => {
                    if rejected_before_generate(&response, id, &attempt) {
                        owner.lock().admission = Admission::NotStarted;
                        if attempt_number == self.native_retry.retries {
                            return stream_response(response, leases);
                        }
                        "native-engine-dead-before-generate"
                    } else {
                        // Native /generate sends successful response headers only
                        // after generation has finished. Other errors stay unknown.
                        if response.status() == StatusCode::OK {
                            owner.lock().admission = Admission::Completed;
                        }
                        return stream_response(response, leases);
                    }
                }
                Ok(Err(error)) if error.is_connect() && !error.is_timeout() => {
                    owner.lock().admission = Admission::NotStarted;
                    "connection-failed-before-delivery"
                }
                Ok(Err(error)) => {
                    error!(
                        request_id = id,
                        worker = worker.url(),
                        "Native delivery unknown: {error:?}"
                    );
                    return error_response(
                        StatusCode::BAD_GATEWAY,
                        "Native backend delivery is unknown; retry forbidden",
                    );
                }
                Err(_) => {
                    return error_response(
                        StatusCode::GATEWAY_TIMEOUT,
                        "Native generation deadline exceeded; retry forbidden",
                    );
                }
            };
            if attempt_number == self.native_retry.retries {
                return error_response(
                    StatusCode::BAD_GATEWAY,
                    "Native backend connection failed before delivery",
                );
            }
            if owner.lock().cancelled {
                return error_response(
                    StatusCode::CONFLICT,
                    "Native generation cancelled before retry",
                );
            }
            // A DP rank on the same endpoint is not an independent replica.
            // Probe one distinct available worker; this is not a search loop.
            let alternate = self
                .worker_registry
                .get_all()
                .into_iter()
                .filter(|candidate| {
                    candidate.is_available() && candidate.base_url() != worker.base_url()
                })
                .min_by_key(|candidate| candidate.load());
            let Some(alternate) = alternate else {
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No distinct healthy native replica",
                );
            };
            let health_url = alternate.endpoint_url("/health");
            let mut probe = self.client.get(&health_url);
            probe = dp_utils::add_dp_rank_header(probe, alternate.dp_rank());
            if let Some(key) = &self.api_key {
                probe = probe.bearer_auth(key);
            }
            let health_deadline = deadline.min(Deadline::now() + self.native_retry.health_timeout);
            let health = timeout_at(health_deadline, probe.send()).await;
            if !matches!(health, Ok(Ok(ref response)) if response.status() == StatusCode::OK) {
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Alternate native engine health was not confirmed",
                );
            }
            alternate.increment_load();
            leases.push(SessionLease::Stateless(alternate.clone()));
            info!(
                request_id = id,
                failed_worker = worker.url(),
                alternate = alternate.url(),
                evidence,
                "Native pre-admission retry candidate passed engine health"
            );
            worker = alternate;
        }
        unreachable!("native retry count is bounded")
    }

    pub(super) async fn route_native_abort(
        &self,
        headers: Option<&HeaderMap>,
        body: serde_json::Value,
    ) -> Response {
        let Some((id, session)) = identity(&body) else {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Native abort requires request_id and session_id",
            );
        };
        let owner = self.native_retry.owners.lock().get(id).cloned();
        let Some(owner) = owner else {
            return abort_response(id, false, Some("No live physical request owner"));
        };
        let worker = {
            let mut state = owner.lock();
            if state.session != session {
                return abort_response(
                    id,
                    false,
                    Some("Native abort session does not match request owner"),
                );
            }
            state.cancelled = true;
            if state.admission != Admission::Unknown {
                state.abort_confirmed = true;
                if state.forwarding_done {
                    self.native_retry.owners.lock().remove(id);
                }
                return abort_response(id, true, None);
            }
            state.worker.clone()
        };
        let Some(worker) = worker else {
            return abort_response(
                id,
                false,
                Some("Unknown native delivery has no physical owner"),
            );
        };
        let url = worker.endpoint_url(ABORT);
        let mut request = self.client.post(&url).json(&body);
        request = dp_utils::add_dp_rank_header(request, worker.dp_rank());
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = otel_http::send_client_request(
            request,
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &url,
                route: Some(ABORT),
                request_phase: Some("abort"),
            },
        )
        .await;
        match response {
            Ok(response) if response.status() == StatusCode::OK => {
                match response.json::<serde_json::Value>().await {
                    Ok(receipt)
                        if receipt.get("aborted").and_then(|v| v.as_bool()) == Some(true)
                            && receipt.get("request_id").and_then(|v| v.as_str()) == Some(id) =>
                    {
                        // This ACK settles cancellation only. It is never used
                        // as no-token evidence and cannot enable another attempt.
                        let done = {
                            let mut state = owner.lock();
                            state.abort_confirmed = true;
                            state.forwarding_done
                        };
                        if done {
                            self.native_retry.owners.lock().remove(id);
                        }
                        abort_response(id, true, None)
                    }
                    _ => abort_response(
                        id,
                        false,
                        Some("Native abort was not confirmed by its physical owner"),
                    ),
                }
            }
            _ => abort_response(id, false, Some("Native physical owner abort failed")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::PolicyConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;
    use tokio::sync::Notify;

    #[derive(Clone, Default)]
    struct Calls {
        generate: Arc<AtomicUsize>,
        abort: Arc<AtomicUsize>,
        health: Arc<AtomicUsize>,
        entered: Arc<Notify>,
        release: Arc<Notify>,
        health_entered: Arc<Notify>,
        health_release: Arc<Notify>,
        block_health: bool,
        block_other_requests: bool,
        reject_early_abort: bool,
        delay: Duration,
    }

    #[derive(Clone, Copy)]
    enum Behavior {
        Success,
        Reject,
        BadNonce,
        BadId,
        BadStatus,
        Generic502,
        Block,
    }

    async fn backend(
        behavior: Behavior,
        healthy: bool,
        calls: Calls,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new()
            .route("/health", axum::routing::get({
                let calls = calls.clone();
                move || { let calls = calls.clone(); async move {
                    calls.health.fetch_add(1, Ordering::SeqCst);
                    calls.health_entered.notify_one();
                    if calls.block_health { calls.health_release.notified().await; }
                    if healthy { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE }
                }}
            }))
            .route(GENERATE, axum::routing::post({
                let calls = calls.clone();
                move |headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                    let calls = calls.clone(); async move {
                        calls.generate.fetch_add(1, Ordering::SeqCst);
                        calls.entered.notify_one();
                        if !calls.delay.is_zero() { tokio::time::sleep(calls.delay).await; }
                        if calls.block_other_requests && body["request_id"] != "physical-a" {
                            calls.release.notified().await;
                            return Json(serde_json::json!({"token_ids": [43]})).into_response();
                        }
                        match behavior {
                            Behavior::Success => Json(serde_json::json!({
                                "request_id": body["request_id"], "session_id": body["session_id"],
                                "token_ids": [42], "stop_reason": "completed"
                            })).into_response(),
                            Behavior::Reject | Behavior::BadNonce | Behavior::BadId | Behavior::BadStatus => {
                                let attempt = if matches!(behavior, Behavior::BadNonce) { "stale-nonce" }
                                    else { headers[ATTEMPT].to_str().unwrap() };
                                Response::builder().status(if matches!(behavior, Behavior::BadStatus) { StatusCode::BAD_GATEWAY } else { StatusCode::SERVICE_UNAVAILABLE })
                                    .header(ADMISSION, "not-started-v1")
                                    .header(REQUEST_ID, if matches!(behavior, Behavior::BadId) { "different-physical-request" } else { body["request_id"].as_str().unwrap() })
                                    .header(ATTEMPT, attempt)
                                    .header(REJECTION, "engine-dead")
                                    .body(Body::from("engine is dead before generate")).unwrap()
                            }
                            Behavior::Generic502 => error_response(StatusCode::BAD_GATEWAY, "engine failed after ADD"),
                            Behavior::Block => {
                                calls.release.notified().await;
                                Json(serde_json::json!({"token_ids": [42]})).into_response()
                            }
                        }
                    }
                }
            }))
            .route(ABORT, axum::routing::post(move |Json(body): Json<serde_json::Value>| {
                let calls = calls.clone(); async move {
                    calls.abort.fetch_add(1, Ordering::SeqCst);
                    if !calls.reject_early_abort { calls.release.notify_one(); }
                    Json(serde_json::json!({"request_id": body["request_id"], "aborted": !calls.reject_early_abort}))
                }
            }));
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, handle)
    }

    fn router(urls: &[String], retries: u8, timeout: Duration) -> Router {
        let worker_registry = Arc::new(WorkerRegistry::new());
        for url in urls {
            worker_registry.register(Arc::new(BasicWorker::new(url.clone(), WorkerType::Regular)));
        }
        let (_, rx) = tokio::sync::watch::channel(HashMap::new());
        Router {
            worker_registry,
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::ConsistentHash {
                virtual_nodes: 100,
            })),
            client: Client::builder().no_proxy().build().unwrap(),
            worker_startup_timeout_secs: 1,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: 1,
            api_key: None,
            retry_config: RetryConfig::default(),
            native_retry: NativeRetry::new(retries, timeout, Duration::from_secs(5)),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            _worker_loads: Arc::new(rx),
            _load_monitor_handle: None,
        }
    }

    fn request(router: &Router, primary: &str, id: &str) -> (HeaderMap, serde_json::Value) {
        let workers = router.worker_registry.get_all();
        let policy = router.policy_registry.get_default_policy();
        for n in 0..10000 {
            let session = format!("prompt.same.{n}");
            let mut headers = HeaderMap::new();
            headers.insert("x-session-id", HeaderValue::from_str(&session).unwrap());
            let body = serde_json::json!({"request_id": id, "session_id": session, "prompt_ids": [1, 2], "sampling_params": {}});
            let text = serde_json::to_string(&body).unwrap();
            let request_headers = Router::headers_to_request_headers(Some(&headers));
            let index = policy
                .select_worker_with_headers(&workers, Some(&text), request_headers.as_ref())
                .unwrap();
            if workers[index].url() == primary {
                return (headers, body);
            }
        }
        panic!("no session assigned to primary")
    }

    async fn frontend(router: Arc<Router>) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new()
            .route(
                GENERATE,
                axum::routing::post({
                    let router = router.clone();
                    move |headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                        let router = router.clone();
                        async move {
                            router
                                .route_transparent(Some(&headers), GENERATE, &Method::POST, body)
                                .await
                        }
                    }
                }),
            )
            .route(
                ABORT,
                axum::routing::post(
                    move |headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                        let router = router.clone();
                        async move {
                            router
                                .route_transparent(Some(&headers), ABORT, &Method::POST, body)
                                .await
                        }
                    },
                ),
            );
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, handle)
    }

    #[tokio::test]
    async fn abort_before_dispatch_cannot_settle_unregistered_native_request() {
        let calls = Calls {
            reject_early_abort: true,
            ..Calls::default()
        };
        let other_calls = Calls::default();
        let (primary, primary_task) = backend(Behavior::Block, true, calls.clone()).await;
        let (secondary, secondary_task) =
            backend(Behavior::Success, true, other_calls.clone()).await;
        let router = router(&[primary.clone(), secondary], 1, Duration::from_secs(2));
        let (headers, body) = request(&router, &primary, "before-dispatch");
        let mut generation = Box::pin(router.route_transparent(
            Some(&headers),
            GENERATE,
            &Method::POST,
            body.clone(),
        ));
        // The first poll publishes Unknown ownership and parks in the fresh
        // connection future. Do not poll it again before forwarding the abort.
        assert!(futures::poll!(generation.as_mut()).is_pending());
        let owner = router.native_retry.owners.lock()["before-dispatch"].clone();
        assert_eq!(owner.lock().admission, Admission::Unknown);
        let aborted = router
            .route_transparent(Some(&headers), ABORT, &Method::POST, body)
            .await;
        let receipt: serde_json::Value =
            serde_json::from_slice(&to_bytes(aborted.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(receipt["aborted"], false);
        assert_eq!(calls.generate.load(Ordering::SeqCst), 0);
        assert_eq!(other_calls.generate.load(Ordering::SeqCst), 0);
        assert!(owner.lock().cancelled);
        assert!(!owner.lock().abort_confirmed);
        drop(generation);
        assert!(owner.lock().forwarding_done);
        assert!(router
            .native_retry
            .owners
            .lock()
            .contains_key("before-dispatch"));
        assert_eq!(owner.lock().worker.as_ref().unwrap().url(), primary);
        primary_task.abort();
        secondary_task.abort();
    }

    #[tokio::test]
    async fn refused_connection_retries_one_distinct_healthy_replica() {
        let refused = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let refused_url = format!("http://{}", refused.local_addr().unwrap());
        drop(refused);
        let calls = Calls::default();
        let (secondary, backend) = backend(Behavior::Success, true, calls.clone()).await;
        let router = Arc::new(router(
            &[refused_url.clone(), secondary],
            1,
            Duration::from_secs(2),
        ));
        let (headers, body) = request(&router, &refused_url, "physical-refused");
        let (url, server) = frontend(router.clone()).await;
        let response = Client::new()
            .post(format!("{url}{GENERATE}"))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let output: serde_json::Value = response.json().await.unwrap();
        assert_eq!(output["token_ids"], serde_json::json!([42]));
        assert_eq!(output["request_id"], body["request_id"]);
        assert_eq!(output["session_id"], body["session_id"]);
        assert_eq!(calls.generate.load(Ordering::SeqCst), 1);
        assert_eq!(calls.health.load(Ordering::SeqCst), 1);
        assert!(router.native_retry.owners.lock().is_empty());
        server.abort();
        backend.abort();
    }

    #[tokio::test]
    async fn accepted_bytes_then_disconnect_does_not_retry_and_abort_keeps_owner() {
        use tokio::io::AsyncReadExt;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let primary = format!("http://{}", listener.local_addr().unwrap());
        let connection = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = [0; 4096];
            assert!(socket.read(&mut bytes).await.unwrap() > 0);
        });
        let calls = Calls::default();
        let (secondary, backend) = backend(Behavior::Success, true, calls.clone()).await;
        let router = router(&[primary.clone(), secondary], 1, Duration::from_secs(2));
        let (headers, body) = request(&router, &primary, "physical-unknown");
        let response = router
            .route_transparent(Some(&headers), GENERATE, &Method::POST, body.clone())
            .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        connection.await.unwrap();
        assert_eq!(calls.generate.load(Ordering::SeqCst), 0);
        assert_eq!(calls.health.load(Ordering::SeqCst), 0);
        let owner = router.native_retry.owners.lock()["physical-unknown"].clone();
        assert_eq!(owner.lock().worker.as_ref().unwrap().url(), primary);
        let abort = router
            .route_transparent(Some(&headers), ABORT, &Method::POST, body)
            .await;
        let receipt: serde_json::Value =
            serde_json::from_slice(&to_bytes(abort.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(receipt["aborted"], false);
        assert_eq!(calls.abort.load(Ordering::SeqCst), 0);
        backend.abort();
    }

    #[tokio::test]
    async fn only_nonce_bound_rejection_retries_not_generic_502_or_stale_proof() {
        for behavior in [
            Behavior::Reject,
            Behavior::BadNonce,
            Behavior::BadId,
            Behavior::BadStatus,
            Behavior::Generic502,
        ] {
            let first = Calls::default();
            let second = Calls::default();
            let (primary, primary_task) = backend(behavior, true, first.clone()).await;
            let (secondary, secondary_task) =
                backend(Behavior::Success, true, second.clone()).await;
            let router = router(&[primary.clone(), secondary], 1, Duration::from_secs(2));
            let (headers, body) = request(&router, &primary, "physical-proof");
            let response = router
                .route_transparent(Some(&headers), GENERATE, &Method::POST, body)
                .await;
            let retried = matches!(behavior, Behavior::Reject);
            assert_eq!(
                response.status(),
                if retried {
                    StatusCode::OK
                } else if matches!(behavior, Behavior::BadNonce | Behavior::BadId) {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::BAD_GATEWAY
                }
            );
            assert_eq!(first.generate.load(Ordering::SeqCst), 1);
            assert_eq!(second.generate.load(Ordering::SeqCst), usize::from(retried));
            primary_task.abort();
            secondary_task.abort();
        }
    }

    #[tokio::test]
    async fn default_off_keeps_single_dispatch_and_does_not_probe_alternates() {
        let first = Calls::default();
        let second = Calls::default();
        let (primary, primary_task) = backend(Behavior::Generic502, true, first.clone()).await;
        let (secondary, secondary_task) = backend(Behavior::Success, true, second.clone()).await;
        let router = router(&[primary.clone(), secondary], 0, Duration::from_secs(2));
        let (headers, body) = request(&router, &primary, "off");
        assert_eq!(
            router
                .route_transparent(Some(&headers), GENERATE, &Method::POST, body)
                .await
                .status(),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(first.generate.load(Ordering::SeqCst), 1);
        assert_eq!(second.generate.load(Ordering::SeqCst), 0);
        assert_eq!(second.health.load(Ordering::SeqCst), 0);
        assert!(router.native_retry.owners.lock().is_empty());
        primary_task.abort();
        secondary_task.abort();
    }

    #[tokio::test]
    async fn retry_abort_tracks_physical_owner_without_moving_session_sibling() {
        let first = Calls {
            block_other_requests: true,
            ..Calls::default()
        };
        let second = Calls::default();
        let (primary, primary_task) = backend(Behavior::Reject, true, first.clone()).await;
        let (secondary, secondary_task) = backend(Behavior::Block, true, second.clone()).await;
        let router = Arc::new(router(
            &[primary.clone(), secondary],
            1,
            Duration::from_secs(3),
        ));
        let (headers, body) = request(&router, &primary, "physical-a");
        let (url, server) = frontend(router.clone()).await;
        let mut sibling = body.clone();
        sibling["request_id"] = serde_json::json!("physical-b");
        let sibling_url = format!("{url}{GENERATE}");
        let sibling_headers = headers.clone();
        let sibling_body = sibling.clone();
        let sibling_generation = tokio::spawn(async move {
            Client::new()
                .post(sibling_url)
                .headers(sibling_headers)
                .json(&sibling_body)
                .send()
                .await
                .unwrap()
        });
        first.entered.notified().await;
        let generate_url = format!("{url}{GENERATE}");
        let generate_headers = headers.clone();
        let generate_body = body.clone();
        let generation = tokio::spawn(async move {
            Client::new()
                .post(generate_url)
                .headers(generate_headers)
                .json(&generate_body)
                .send()
                .await
                .unwrap()
        });
        second.entered.notified().await;
        let duplicate = router
            .route_transparent(Some(&headers), GENERATE, &Method::POST, body.clone())
            .await;
        assert_eq!(duplicate.status(), StatusCode::CONFLICT);
        let abort = Client::new()
            .post(format!("{url}{ABORT}"))
            .headers(headers.clone())
            .json(&body)
            .send()
            .await
            .unwrap();
        let receipt: serde_json::Value = abort.json().await.unwrap();
        assert_eq!(receipt["aborted"], true);
        assert_eq!(second.abort.load(Ordering::SeqCst), 1);
        assert_eq!(first.abort.load(Ordering::SeqCst), 0);
        assert_eq!(generation.await.unwrap().status(), StatusCode::OK);
        assert!(!sibling_generation.is_finished());
        let abort = Client::new()
            .post(format!("{url}{ABORT}"))
            .headers(headers)
            .json(&sibling)
            .send()
            .await
            .unwrap();
        let receipt: serde_json::Value = abort.json().await.unwrap();
        assert_eq!(receipt["aborted"], true);
        assert_eq!(first.abort.load(Ordering::SeqCst), 1);
        assert_eq!(sibling_generation.await.unwrap().status(), StatusCode::OK);
        assert_eq!(first.generate.load(Ordering::SeqCst), 2);
        assert_eq!(second.generate.load(Ordering::SeqCst), 1);
        assert!(router.native_retry.owners.lock().is_empty());
        server.abort();
        primary_task.abort();
        secondary_task.abort();
    }

    #[tokio::test]
    async fn original_deadline_covers_both_attempts() {
        let first = Calls::default();
        let second = Calls::default();
        let first = Calls {
            delay: Duration::from_millis(300),
            ..first
        };
        let (primary, primary_task) = backend(Behavior::Reject, true, first.clone()).await;
        let (secondary, secondary_task) = backend(Behavior::Block, true, second.clone()).await;
        let router = router(&[primary.clone(), secondary], 1, Duration::from_millis(500));
        let (headers, body) = request(&router, &primary, "deadline");
        let start = Instant::now();
        let response = router
            .route_transparent(Some(&headers), GENERATE, &Method::POST, body)
            .await;
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(start.elapsed() < Duration::from_millis(700));
        assert_eq!(second.generate.load(Ordering::SeqCst), 1);
        assert_eq!(
            router.native_retry.owners.lock()["deadline"]
                .lock()
                .admission,
            Admission::Unknown
        );
        primary_task.abort();
        secondary_task.abort();
    }

    #[tokio::test]
    async fn unhealthy_alternate_and_exhausted_retry_never_dispatch_third_attempt() {
        for healthy in [false, true] {
            let first = Calls::default();
            let second = Calls::default();
            let (primary, primary_task) = backend(Behavior::Reject, true, first.clone()).await;
            let (secondary, secondary_task) =
                backend(Behavior::Reject, healthy, second.clone()).await;
            let router = router(&[primary.clone(), secondary], 1, Duration::from_secs(2));
            let (headers, body) = request(&router, &primary, "bounded");
            let response = router
                .route_transparent(Some(&headers), GENERATE, &Method::POST, body)
                .await;
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(first.generate.load(Ordering::SeqCst), 1);
            assert_eq!(second.generate.load(Ordering::SeqCst), usize::from(healthy));
            primary_task.abort();
            secondary_task.abort();
        }
    }
    #[tokio::test]
    async fn abort_between_proof_and_retry_prevents_any_secondary_generation() {
        let first = Calls::default();
        let second = Calls {
            block_health: true,
            ..Calls::default()
        };
        let (primary, primary_task) = backend(Behavior::Reject, true, first.clone()).await;
        let (secondary, secondary_task) = backend(Behavior::Success, true, second.clone()).await;
        let router = Arc::new(router(
            &[primary.clone(), secondary],
            1,
            Duration::from_secs(2),
        ));
        let (headers, body) = request(&router, &primary, "cancel-before-retry");
        let generation_router = router.clone();
        let generation_headers = headers.clone();
        let generation_body = body.clone();
        let generation = tokio::spawn(async move {
            generation_router
                .route_transparent(
                    Some(&generation_headers),
                    GENERATE,
                    &Method::POST,
                    generation_body,
                )
                .await
        });
        second.health_entered.notified().await;
        let abort = router
            .route_transparent(Some(&headers), ABORT, &Method::POST, body)
            .await;
        let receipt: serde_json::Value =
            serde_json::from_slice(&to_bytes(abort.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(receipt["aborted"], true);
        second.health_release.notify_one();
        assert_eq!(generation.await.unwrap().status(), StatusCode::CONFLICT);
        assert_eq!(second.generate.load(Ordering::SeqCst), 0);
        assert_eq!(first.abort.load(Ordering::SeqCst), 0);
        assert!(router.native_retry.owners.lock().is_empty());
        primary_task.abort();
        secondary_task.abort();
    }

    #[tokio::test]
    async fn exhausted_proven_rejection_can_settle_abort_without_engine_rpc() {
        let calls = Calls::default();
        let (primary, primary_task) = backend(Behavior::Reject, true, calls.clone()).await;
        let router = router(&[primary.clone()], 1, Duration::from_secs(2));
        let (headers, body) = request(&router, &primary, "no-alternate");
        assert_eq!(
            router
                .route_transparent(Some(&headers), GENERATE, &Method::POST, body.clone())
                .await
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        let abort = router
            .route_transparent(Some(&headers), ABORT, &Method::POST, body)
            .await;
        let receipt: serde_json::Value =
            serde_json::from_slice(&to_bytes(abort.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(receipt["aborted"], true);
        assert_eq!(calls.abort.load(Ordering::SeqCst), 0);
        assert!(router.native_retry.owners.lock().is_empty());
        primary_task.abort();
    }
    #[tokio::test]
    async fn cancelled_forwarding_keeps_the_secondary_address_for_abort() {
        let first = Calls::default();
        let second = Calls::default();
        let (primary, primary_task) = backend(Behavior::Reject, true, first.clone()).await;
        let (secondary, secondary_task) = backend(Behavior::Block, true, second.clone()).await;
        let router = Arc::new(router(
            &[primary.clone(), secondary.clone()],
            1,
            Duration::from_secs(2),
        ));
        let (headers, body) = request(&router, &primary, "dropped-forwarder");
        let generation_router = router.clone();
        let generation_headers = headers.clone();
        let generation_body = body.clone();
        let generation = tokio::spawn(async move {
            generation_router
                .route_transparent(
                    Some(&generation_headers),
                    GENERATE,
                    &Method::POST,
                    generation_body,
                )
                .await
        });
        second.entered.notified().await;
        generation.abort();
        assert!(generation.await.unwrap_err().is_cancelled());
        let owner = router.native_retry.owners.lock()["dropped-forwarder"].clone();
        assert_eq!(owner.lock().worker.as_ref().unwrap().url(), secondary);
        let abort = router
            .route_transparent(Some(&headers), ABORT, &Method::POST, body)
            .await;
        let receipt: serde_json::Value =
            serde_json::from_slice(&to_bytes(abort.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(receipt["aborted"], true);
        assert_eq!(second.abort.load(Ordering::SeqCst), 1);
        assert_eq!(first.abort.load(Ordering::SeqCst), 0);
        assert!(router.native_retry.owners.lock().is_empty());
        primary_task.abort();
        secondary_task.abort();
    }

    #[tokio::test]
    async fn truncated_token_response_is_never_retried() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let primary = format!("http://{}", listener.local_addr().unwrap());
        let connection = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = [0; 4096];
            assert!(socket.read(&mut bytes).await.unwrap() > 0);
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{\"token_ids\":[42]").await.unwrap();
        });
        let calls = Calls::default();
        let (secondary, secondary_task) = backend(Behavior::Success, true, calls.clone()).await;
        let router = router(&[primary.clone(), secondary], 1, Duration::from_secs(2));
        let (headers, body) = request(&router, &primary, "partial-token-body");
        let response = router
            .route_transparent(Some(&headers), GENERATE, &Method::POST, body)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(to_bytes(response.into_body(), usize::MAX).await.is_err());
        connection.await.unwrap();
        assert_eq!(calls.generate.load(Ordering::SeqCst), 0);
        assert_eq!(calls.health.load(Ordering::SeqCst), 0);
        secondary_task.abort();
    }

    #[tokio::test]
    async fn token_placement_is_rejected_only_when_native_retry_is_selected() {
        use crate::policies::{ConsistentHashPolicy, TokenPlacement};
        for retries in [0, 1] {
            let config = crate::config::RouterConfig {
                native_pre_admission_retries: retries,
                ..Default::default()
            };
            let mut context =
                crate::server::AppContext::new(config, Client::new(), 10, None, vec![]).unwrap();
            context.policy_registry = Arc::new(PolicyRegistry::with_default_policy(Arc::new(
                ConsistentHashPolicy::with_placement(Some(TokenPlacement::new(
                    Duration::from_secs(900),
                ))),
            )));
            let result = Router::new(vec![], &Arc::new(context)).await;
            if retries == 0 {
                assert!(result.is_ok());
            } else {
                assert!(result
                    .unwrap_err()
                    .contains("require hash session placement"));
            }
        }
    }
}
