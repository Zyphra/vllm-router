//! Actual transparent routes must retain the plain-hash available-worker pool.
//! This isolated integration target has one test; set placement before starting Tokio.

mod common;

use axum::{body::Body, extract::State, http::Request, routing::post, Json};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig},
    core::{BasicWorker, Worker, WorkerType},
    policies::RequestHeaders,
    routers::http::router::Router,
};

struct Backend {
    url: String,
    task: JoinHandle<()>,
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn backend(label: &'static str) -> Backend {
    async fn reply(State(label): State<&'static str>, Json(body): Json<Value>) -> Json<Value> {
        Json(json!({"worker": label, "request_id": body["request_id"], "aborted": true}))
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new()
        .route("/verl/v1/generate", post(reply))
        .route("/verl/v1/abort", post(reply))
        .with_state(label);
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Backend { url, task }
}

async fn route(app: &axum::Router, path: &str, session: &str) -> Value {
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("x-session-id", session)
                .body(Body::from(json!({"request_id": "physical-G"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn plain_hash_abort_uses_generation_pool_with_unrelated_unhealthy_worker() {
    let previous = std::env::var_os("VLLM_ROUTER_SESSION_PLACEMENT");
    std::env::set_var("VLLM_ROUTER_SESSION_PLACEMENT", "hash");
    // No other tests run in this integration executable. Configure before threads exist.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let a = backend("A").await;
        let b = backend("B").await;
        let c = backend("C").await;
        let config = RouterConfig {
            policy: PolicyConfig::ConsistentHash { virtual_nodes: 160 },
            ..RouterConfig::default()
        };
        let context = common::create_test_context(config.clone());
        for (origin, healthy) in [(&a, true), (&b, true), (&c, false)] {
            let worker = Arc::new(BasicWorker::new(origin.url.clone(), WorkerType::Regular));
            worker.set_healthy(healthy);
            context.worker_registry.register(worker);
        }
        let policy = context.policy_registry.get_default_policy();
        let registered = context.worker_registry.get_all();
        let available: Vec<Arc<dyn Worker>> = registered
            .iter()
            .filter(|worker| worker.is_available())
            .cloned()
            .collect();
        assert_eq!(available.len(), 2);
        // Pick a witness for these ephemeral origins: adding C changes the hash ring,
        // even though C cannot receive generation. Compare actual policy decisions.
        let session = (0..4096)
            .map(|i| format!("availability-{i}"))
            .find(|session| {
                let headers: RequestHeaders =
                    [("x-session-id".to_string(), session.clone())].into();
                let normal = policy
                    .select_worker_with_headers(&available, None, Some(&headers))
                    .unwrap();
                let expanded = policy
                    .select_worker_with_headers(&registered, None, Some(&headers))
                    .unwrap();
                available[normal].url() != registered[expanded].url()
            })
            .expect("Unhealthy C must expose an expanded-pool hash disagreement");
        let router = Arc::new(Router::new(Vec::new(), &context).await.unwrap());
        let app = common::test_app::create_test_app(router, context.client.clone(), &config);
        let generation = route(&app, "/verl/v1/generate", &session).await;
        let abort = route(&app, "/verl/v1/abort", &session).await;
        assert!(generation["worker"] == "A" || generation["worker"] == "B");
        assert_eq!(generation["worker"], abort["worker"]);
        assert_eq!(abort["request_id"], "physical-G");
    });
    drop(runtime);
    match previous {
        Some(value) => std::env::set_var("VLLM_ROUTER_SESSION_PLACEMENT", value),
        None => std::env::remove_var("VLLM_ROUTER_SESSION_PLACEMENT"),
    }
}
