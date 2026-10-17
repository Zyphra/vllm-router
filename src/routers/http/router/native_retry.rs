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
    owners: Owners,
}

impl NativeRetry {
    pub(super) fn new(retries: u8, timeout: Duration) -> Self {
        Self { retries, timeout, owners: Arc::default() }
    }

    pub(super) fn enabled(&self) -> bool { self.retries != 0 }
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
            owner.admission != Admission::Unknown || owner.abort_confirmed
        };
        if remove { self.owners.lock().remove(&self.id); }
    }
}

fn identity(body: &serde_json::Value) -> Option<(&str, &str)> {
    let id = body.get("request_id")?.as_str()?;
    let session = body.get("session_id")?.as_str()?;
    if id.is_empty() || session.is_empty() { return None; }
    // IDs are echoed in admission proof headers. Refuse invalid HTTP values.
    HeaderValue::from_str(id).ok()?;
    Some((id, session))
}

fn error_response(status: StatusCode, reason: &str) -> Response {
    (status, Json(serde_json::json!({"error": reason}))).into_response()
}

fn abort_response(id: &str, confirmed: bool, error: Option<&str>) -> Response {
    Json(serde_json::json!({"request_id": id, "aborted": confirmed, "error": error})).into_response()
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
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "Failed to build native response")
    })
}

impl Router {
    pub(super) async fn route_native_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: serde_json::Value,
    ) -> Response {
        let Some((id, session)) = identity(&body) else {
            return error_response(StatusCode::BAD_REQUEST, "Native generation requires request_id and session_id");
        };
        if headers.and_then(|h| h.get("x-session-id")).and_then(|v| v.to_str().ok())
            .is_some_and(|value| value != session)
        {
            return error_response(StatusCode::BAD_REQUEST, "Native routing session does not match body");
        }
        let owner = Arc::new(Mutex::new(Ownership {
            session: session.to_owned(), worker: None, admission: Admission::NotStarted,
            cancelled: false, abort_confirmed: false, forwarding_done: false,
        }));
        {
            let mut owners = self.native_retry.owners.lock();
            if owners.contains_key(id) {
                return error_response(StatusCode::CONFLICT, "Native request already has an owner");
            }
            owners.insert(id.to_owned(), owner.clone());
        }
        let _forwarding = Forwarding {
            id: id.to_owned(), owner: owner.clone(), owners: self.native_retry.owners.clone(),
        };
        let deadline = Deadline::now() + self.native_retry.timeout;
        let workers: Vec<_> = self.worker_registry.get_all().into_iter()
            .filter(|worker| worker.is_available()).collect();
        let request_text = serde_json::to_string(&body).ok();
        let request_headers = Self::headers_to_request_headers(headers);
        let policy = self.policy_registry.get_default_policy();
        let Some((index, lease)) = policy.select_worker_with_lease(
            &workers, request_text.as_deref(), request_headers.as_ref(),
        ) else {
            return error_response(StatusCode::SERVICE_UNAVAILABLE, "No available native worker");
        };
        let mut leases: Vec<_> = lease.into_iter().collect();
        let mut worker = workers[index].clone();
        for attempt_number in 0..=self.native_retry.retries {
            let attempt = uuid::Uuid::new_v4().to_string();
            {
                let mut state = owner.lock();
                if state.cancelled {
                    return error_response(StatusCode::CONFLICT, "Native generation cancelled before dispatch");
                }
                // Setting the active physical owner and starting its dispatch is
                // one cancellation transaction. Abort can never trigger a retry.
                state.worker = Some(worker.clone());
                state.admission = Admission::Unknown;
            }
            let url = worker.endpoint_url(GENERATE);
            let mut request = self.client.post(&url).json(&body).header(ATTEMPT, &attempt);
            request = dp_utils::add_dp_rank_header(request, worker.dp_rank());
            if let Some(key) = &self.api_key {
                request = request.bearer_auth(key);
            }
            let result = timeout_at(deadline, otel_http::send_client_request(
                request, headers, ClientRequestOptions {
                    method: "POST", url: &url, route: Some(GENERATE), request_phase: Some("inference"),
                },
            )).await;
            let proven_not_started = match result {
                Ok(Ok(response)) => {
                    if rejected_before_generate(&response, id, &attempt) {
                        owner.lock().admission = Admission::NotStarted;
                        if attempt_number == self.native_retry.retries {
                            return stream_response(response, leases);
                        }
                        true
                    } else {
                        // Native /generate sends successful response headers only
                        // after generation has finished. Other errors stay unknown.
                        if response.status().is_success() {
                            owner.lock().admission = Admission::Completed;
                        }
                        return stream_response(response, leases);
                    }
                }
                Ok(Err(error)) if error.is_connect() && !error.is_timeout() => {
                    owner.lock().admission = Admission::NotStarted;
                    true
                }
                Ok(Err(error)) => {
                    error!(request_id = id, worker = worker.url(), "Native delivery unknown: {error:?}");
                    return error_response(StatusCode::BAD_GATEWAY, "Native backend delivery is unknown; retry forbidden");
                }
                Err(_) => {
                    return error_response(StatusCode::GATEWAY_TIMEOUT, "Native generation deadline exceeded; retry forbidden");
                }
            };
            if !proven_not_started || attempt_number == self.native_retry.retries {
                return error_response(StatusCode::BAD_GATEWAY, "Native backend connection failed before delivery");
            }
            if owner.lock().cancelled {
                return error_response(StatusCode::CONFLICT, "Native generation cancelled before retry");
            }
            // A DP rank on the same endpoint is not an independent replica.
            // Probe one distinct available worker; this is not a search loop.
            let alternate = workers.iter().filter(|candidate| {
                candidate.is_available() && candidate.base_url() != worker.base_url()
            }).min_by_key(|candidate| candidate.load()).cloned();
            let Some(alternate) = alternate else {
                return error_response(StatusCode::SERVICE_UNAVAILABLE, "No distinct healthy native replica");
            };
            let health_url = alternate.endpoint_url("/health");
            let mut probe = self.client.get(&health_url);
            probe = dp_utils::add_dp_rank_header(probe, alternate.dp_rank());
            if let Some(key) = &self.api_key { probe = probe.bearer_auth(key); }
            let health = timeout_at(deadline, probe.send()).await;
            if !matches!(health, Ok(Ok(ref response)) if response.status() == StatusCode::OK) {
                return error_response(StatusCode::SERVICE_UNAVAILABLE, "Alternate native engine health was not confirmed");
            }
            alternate.increment_load();
            leases.push(SessionLease::Stateless(alternate.clone()));
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
            return error_response(StatusCode::BAD_REQUEST, "Native abort requires request_id and session_id");
        };
        let owner = self.native_retry.owners.lock().get(id).cloned();
        let Some(owner) = owner else {
            return abort_response(id, false, Some("No live physical request owner"));
        };
        let worker = {
            let mut state = owner.lock();
            if state.session != session {
                return abort_response(id, false, Some("Native abort session does not match request owner"));
            }
            state.cancelled = true;
            if state.admission != Admission::Unknown {
                state.abort_confirmed = true;
                if state.forwarding_done { self.native_retry.owners.lock().remove(id); }
                return abort_response(id, true, None);
            }
            state.worker.clone()
        };
        let Some(worker) = worker else {
            return abort_response(id, false, Some("Unknown native delivery has no physical owner"));
        };
        let url = worker.endpoint_url(ABORT);
        let mut request = self.client.post(&url).json(&body);
        request = dp_utils::add_dp_rank_header(request, worker.dp_rank());
        if let Some(key) = &self.api_key { request = request.bearer_auth(key); }
        let response = otel_http::send_client_request(request, headers, ClientRequestOptions {
            method: "POST", url: &url, route: Some(ABORT), request_phase: Some("abort"),
        }).await;
        match response {
            Ok(response) if response.status().is_success() => {
                match response.json::<serde_json::Value>().await {
                    Ok(receipt) if receipt.get("aborted").and_then(|v| v.as_bool()) == Some(true)
                        && receipt.get("request_id").and_then(|v| v.as_str()) == Some(id) => {
                        // This ACK settles cancellation only. It is never used
                        // as no-token evidence and cannot enable another attempt.
                        let done = {
                            let mut state = owner.lock();
                            state.abort_confirmed = true;
                            state.forwarding_done
                        };
                        if done { self.native_retry.owners.lock().remove(id); }
                        abort_response(id, true, None)
                    }
                    _ => abort_response(id, false, Some("Native abort was not confirmed by its physical owner")),
                }
            }
            _ => abort_response(id, false, Some("Native physical owner abort failed")),
        }
    }
}
