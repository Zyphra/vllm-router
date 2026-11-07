#![cfg(all(feature = "cpu", feature = "ready"))]

//! A chat request's `tools[].function.strict` must reach the worker as sent.
//!
//! The typed chat path re-serializes the parsed request, so a field missing
//! from `Function` is silently dropped and the worker never enforces the
//! caller's strict schema. This drives the real Router over HTTP.

use std::future::IntoFuture;
use std::net::SocketAddr;
use std::time::Duration;

use axum::{routing::get, routing::post, Json};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use vllm_router_rs::config::{RouterConfig, RoutingMode};
use vllm_router_rs::server::{startup, ServerConfig, DEFAULT_POOL_IDLE_TIMEOUT_SECS};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_tool_strict_option_reaches_worker() {
    let backend = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend_url = format!("http://{}", backend.local_addr().unwrap());
    let (forwarded_tx, mut forwarded_rx) = mpsc::unbounded_channel::<Value>();
    let app = axum::Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let forwarded_tx = forwarded_tx.clone();
                async move {
                    forwarded_tx.send(body).unwrap();
                    Json(json!({
                        "id": "chatcmpl-strict",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "test-model",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"
                        }]
                    }))
                }
            }),
        );
    let backend_task = tokio::spawn(axum::serve(backend, app).into_future());

    let port = portpicker::pick_unused_port().unwrap();
    let address: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let mut router_config = RouterConfig::default();
    router_config.mode = RoutingMode::Regular {
        worker_urls: vec![backend_url],
    };
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port,
        router_config,
        max_payload_size: 1024 * 1024,
        wasm_middleware: None,
        wasm_middleware_sha256: None,
        wasm_middleware_routes: Vec::new(),
        log_dir: None,
        log_level: Some("error".to_string()),
        service_discovery_config: None,
        prometheus_config: None,
        request_timeout_secs: 30,
        pool_idle_timeout_secs: DEFAULT_POOL_IDLE_TIMEOUT_SECS,
        request_id_headers: None,
        trace_config: None,
    };
    let router_task = tokio::spawn(async move {
        startup(config).await.expect("router startup");
    });
    tokio::time::timeout(Duration::from_secs(30), async {
        while tokio::net::TcpStream::connect(address).await.is_err() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("router listens");

    let request = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [
            {"type": "function", "function": {
                "name": "strict_true",
                "description": "Strict true tool",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": false
                },
                "strict": true
            }},
            {"type": "function", "function": {
                "name": "strict_false",
                "parameters": {"type": "object", "properties": {}},
                "strict": false
            }},
            {"type": "function", "function": {
                "name": "strict_absent",
                "parameters": {"type": "object", "properties": {}}
            }}
        ]
    });
    let response = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
        .post(format!("http://{address}/v1/chat/completions"))
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);

    let forwarded = tokio::time::timeout(Duration::from_secs(5), forwarded_rx.recv())
        .await
        .unwrap()
        .unwrap();
    for index in 0..3 {
        assert_eq!(
            forwarded["tools"][index]["function"], request["tools"][index]["function"],
            "tool {index} reached the worker changed"
        );
    }
    assert_eq!(forwarded["tools"][0]["function"]["strict"], true);
    assert_eq!(forwarded["tools"][1]["function"]["strict"], false);
    assert!(forwarded["tools"][2]["function"].get("strict").is_none());

    router_task.abort();
    backend_task.abort();
}
