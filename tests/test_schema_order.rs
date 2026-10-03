// Portable native forwarding regression. No fixture files or model backend.
mod common;
use axum::{body::Bytes, extract::State, http::Method, routing::any, Json, Router};
use serde_json::{json, Value};
use std::{
    future::IntoFuture,
    sync::{Arc, Mutex},
};
use vllm_router_rs::{
    config::{RouterConfig, RoutingMode},
    core::{BasicWorker, Worker, WorkerType},
    policies::{CacheAwarePolicy, ConsistentHashPolicy, LoadBalancingPolicy},
    protocols::spec::{ChatCompletionRequest, GenerationRequest},
    routers::{RouterFactory, RouterTrait},
};

const RESPONSES: &str = r#"{
  "model":"test-model","input":[{"role":"user","content":"Compare two answers."}],
  "temperature":0.0,"max_output_tokens":4096,"top_logprobs":20,
  "include":["message.output_text.logprobs"],"chat_template_kwargs":{"enable_thinking":false},
  "text":{"format":{"type":"json_schema","name":"verdict","strict":true,"schema":{
    "type":"object","properties":{
      "verdict":{"type":"string","enum":["EQUIVALENT","NOT_EQUIVALENT","UNSURE"]},
      "reason":{"type":"string","maxLength":600}
    },"required":["verdict","reason"],"propertyOrdering":["verdict","reason"]
  }}}
}"#;
const THINKING: &str = r#"{
  "model":"test-model","input":[{"role":"user","content":"Compare two answers."}],
  "temperature":0.0,"max_output_tokens":8192,"top_logprobs":20,
  "include":["message.output_text.logprobs"],"chat_template_kwargs":{"enable_thinking":true},
  "text":{"format":{"type":"json_schema","name":"verdict","strict":true,"schema":{
    "type":"object","properties":{
      "verdict":{"type":"string","enum":["EQUIVALENT","NOT_EQUIVALENT","UNSURE"]},
      "reason":{"type":"string","maxLength":600}
    },"required":["verdict","reason"],"propertyOrdering":["verdict","reason"]
  }}}
}"#;
const CHAT: &str = r#"{
  "model":"test-model","messages":[{"role":"user","content":"Compare two answers."}],
  "temperature":0.0,"max_tokens":4096,"top_logprobs":20,"logprobs":true,
  "response_format":{"type":"json_schema","json_schema":{"name":"verdict","strict":true,"schema":{
    "type":"object","properties":{
      "verdict":{"type":"string","enum":["EQUIVALENT","NOT_EQUIVALENT","UNSURE"]},
      "reason":{"type":"string","maxLength":600}
    },"required":["verdict","reason"],"propertyOrdering":["verdict","reason"]
  }}}
}"#;

fn order(v: &Value) -> Vec<String> {
    v.as_object().unwrap().keys().cloned().collect()
}

async fn capture(
    State(records): State<Arc<Mutex<Vec<Vec<u8>>>>>,
    method: Method,
    body: Bytes,
) -> Json<Value> {
    if method == Method::POST {
        records.lock().unwrap().push(body.to_vec());
    }
    Json(json!({
        "status":"healthy","model_path":"test-model","model":"test-model","is_generation":true,
        "object":"list","data":[{"id":"test-model","object":"model"}],"ok":true
    }))
}

#[tokio::test]
async fn responses_and_chat_parse_forward_order() {
    let records = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}", listener.local_addr().unwrap());
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let backend = tokio::spawn(
        axum::serve(
            listener,
            Router::new()
                .fallback(any(capture))
                .with_state(records.clone()),
        )
        .with_graceful_shutdown(async {
            let _ = stop_rx.await;
        })
        .into_future(),
    );
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![upstream],
        },
        worker_startup_timeout_secs: 3,
        worker_startup_check_interval_secs: 1,
        ..Default::default()
    };
    let context = common::create_test_context(config.clone());
    let router: Arc<dyn RouterTrait> =
        Arc::from(RouterFactory::create_router(&context).await.unwrap());
    let app = common::test_app::create_test_app_with_tracing(
        router,
        reqwest::Client::new(),
        &config,
        false,
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (app_stop_tx, app_stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = app_stop_rx.await;
            })
            .into_future(),
    );
    let client = reqwest::Client::new();
    for (name, path, raw, pointer) in [
        (
            "primary",
            "/v1/responses",
            RESPONSES,
            "/text/format/schema/properties",
        ),
        (
            "thinking",
            "/v1/responses",
            THINKING,
            "/text/format/schema/properties",
        ),
        (
            "chat",
            "/v1/chat/completions",
            CHAT,
            "/response_format/json_schema/schema/properties",
        ),
    ] {
        let sent: Value = serde_json::from_str(raw).unwrap();
        let response = client
            .post(format!("{base}{path}"))
            .header("content-type", "application/json")
            .body(raw)
            .send()
            .await
            .unwrap();
        assert!(
            response.status().is_success(),
            "{name} forwarding failed: {}",
            response.status()
        );
        let forwarded = records.lock().unwrap().last().unwrap().clone();
        let got: Value = serde_json::from_slice(&forwarded).unwrap();
        for field in [
            "temperature",
            "max_output_tokens",
            "max_tokens",
            "top_logprobs",
            "include",
            "logprobs",
            "chat_template_kwargs",
        ] {
            if let Some(v) = sent.get(field) {
                assert_eq!(got.get(field), Some(v), "sampling field {field}");
            }
        }
        let schema_pointer = if name == "chat" {
            "/response_format/json_schema"
        } else {
            "/text/format"
        };
        assert_eq!(got.pointer(schema_pointer), sent.pointer(schema_pointer));
        assert_eq!(
            order(got.pointer(pointer).unwrap()),
            vec!["verdict", "reason"],
            "{name} actual upstream schema property order",
        );
    }
    // The actual native Axum extractor rejects malformed JSON before forwarding.
    for path in ["/v1/responses", "/v1/chat/completions"] {
        let before = records.lock().unwrap().len();
        let response = client
            .post(format!("{base}{path}"))
            .header("content-type", "application/json")
            .body("{\"model\":")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(
            records.lock().unwrap().len(),
            before,
            "malformed body reached backend"
        );
    }
    // Nested duplicate properties retain their existing last-value-wins semantics.
    let dup = r#"{"model":"test-model","text":{"format":{"type":"json_schema","name":"v","strict":true,"schema":{"type":"object","properties":{"verdict":{"type":"string"},"reason":{"type":"string","maxLength":1},"reason":{"type":"string","maxLength":600}},"required":["verdict","reason"],"propertyOrdering":["verdict","reason"],"additionalProperties":false}}}}"#;
    let response = client
        .post(format!("{base}/v1/responses"))
        .header("content-type", "application/json")
        .body(dup)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let got: Value = serde_json::from_slice(records.lock().unwrap().last().unwrap()).unwrap();
    assert_eq!(
        got.pointer("/text/format/schema/properties/reason/maxLength"),
        Some(&json!(600))
    );
    assert_eq!(
        order(got.pointer("/text/format/schema/properties").unwrap()),
        vec!["verdict", "reason"]
    );
    let chat_dup = dup.replace(
        "\"text\":{\"format\":{\"type\":\"json_schema\",",
        "\"messages\":[{\"role\":\"user\",\"content\":\"control\"}],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{",
    );
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(chat_dup)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let got: Value = serde_json::from_slice(records.lock().unwrap().last().unwrap()).unwrap();
    assert_eq!(
        got.pointer("/response_format/json_schema/schema/properties/reason/maxLength"),
        Some(&json!(600)),
    );
    assert_eq!(
        order(
            got.pointer("/response_format/json_schema/schema/properties")
                .unwrap()
        ),
        vec!["verdict", "reason"],
    );
    assert_eq!(
        records.lock().unwrap().len(),
        5,
        "expected five valid forwarded requests"
    );
    let _ = app_stop_tx.send(());
    let _ = stop_tx.send(());
    server.await.unwrap().unwrap();
    backend.await.unwrap().unwrap();
}

#[test]
fn value_serialization_and_native_routing_consumers() {
    let left: Value = serde_json::from_str(r#"{"verdict":1,"reason":2}"#).unwrap();
    let right: Value = serde_json::from_str(r#"{"reason":2,"verdict":1}"#).unwrap();
    assert_eq!(left, right, "semantic JSON equality changes");
    let a = serde_json::to_string(&left).unwrap();
    let b = serde_json::to_string(&right).unwrap();
    assert_ne!(
        a, b,
        "serialized bodies must retain their input property order"
    );
    let roundtrip: Value = serde_json::from_str(&a).unwrap();
    assert_eq!(serde_json::to_string(&roundtrip).unwrap(), a);
    let policy = ConsistentHashPolicy::new();
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|i| {
            Arc::new(BasicWorker::new(
                format!("http://worker{i}:8000"),
                WorkerType::Regular,
            )) as Arc<dyn Worker>
        })
        .collect();
    let headers =
        std::collections::HashMap::from([("x-session-id".into(), "order-stable-session".into())]);
    let selected = policy
        .select_worker_with_headers(&workers, Some(&a), Some(&headers))
        .unwrap();
    assert_eq!(
        policy.select_worker_with_headers(&workers, Some(&b), Some(&headers)),
        Some(selected)
    );
    let fallback = policy.select_worker_with_headers(&workers, Some(&a), None);
    for _ in 0..10 {
        assert_eq!(
            policy.select_worker_with_headers(&workers, Some(&a), None),
            fallback
        );
    }
    let cache_policy = CacheAwarePolicy::new();
    for worker in &workers {
        cache_policy.add_worker(worker.as_ref());
    }
    let cached = cache_policy.select_worker(&workers, Some(&a)).unwrap();
    assert_eq!(cache_policy.select_worker(&workers, Some(&a)), Some(cached));
    let chat: ChatCompletionRequest = serde_json::from_str(CHAT).unwrap();
    let prompt = chat.extract_text_for_routing();
    let got: ChatCompletionRequest =
        serde_json::from_slice(&serde_json::to_vec(&chat).unwrap()).unwrap();
    assert_eq!(got.extract_text_for_routing(), prompt);
}
