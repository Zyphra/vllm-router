#![cfg(all(feature = "cpu", feature = "ready"))]

//! A transparent request must not keep its body resident while the worker
//! generates.
//!
//! Token-id generation requests (for example `/verl/v1/generate` with a long
//! `prompt_ids` array) reach the worker through the transparent fallback. The
//! worker answers only when generation finishes, so anything the Router still
//! owns at that point stays allocated for the whole generation: the raw body,
//! its parsed JSON tree (tens of bytes per array element) and, for hash
//! policies, a serialized copy. At tens of thousands of concurrent requests that
//! is hundreds of gigabytes. This test counts live heap bytes in the process
//! while real HTTP requests are held at a real worker.

use std::alloc::{GlobalAlloc, Layout, System};
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{body::Bytes, routing::get, routing::post, Json};
use serde_json::json;
use tokio::sync::Notify;
use vllm_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::server::{startup, ServerConfig, DEFAULT_POOL_IDLE_TIMEOUT_SECS};

struct CountingAllocator;

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            LIVE_BYTES.fetch_add(new_size as isize - layout.size() as isize, Ordering::Relaxed);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const HELD_REQUESTS: usize = 2;
const PROMPT_TOKENS: usize = 1_000_000;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transparent_request_body_is_released_while_the_worker_generates() {
    let received = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(Notify::new());
    let backend = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend_url = format!("http://{}", backend.local_addr().unwrap());
    let app = axum::Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/verl/v1/generate",
            post({
                let received = received.clone();
                let release = release.clone();
                move |body: Bytes| {
                    let received = received.clone();
                    let release = release.clone();
                    async move {
                        let length = body.len();
                        drop(body);
                        received.fetch_add(1, Ordering::SeqCst);
                        release.notified().await;
                        Json(json!({"output_ids": [1], "received_bytes": length}))
                    }
                }
            }),
        )
        .layer(axum::extract::DefaultBodyLimit::disable());
    let backend_task = tokio::spawn(axum::serve(backend, app).into_future());

    let port = portpicker::pick_unused_port().unwrap();
    let address: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let mut router_config = RouterConfig::default();
    router_config.mode = RoutingMode::Regular {
        worker_urls: vec![backend_url],
    };
    router_config.policy = PolicyConfig::ConsistentHash { virtual_nodes: 160 };
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port,
        router_config,
        max_payload_size: 64 * 1024 * 1024,
        wasm_middleware: None,
        wasm_middleware_sha256: None,
        wasm_middleware_routes: Vec::new(),
        log_dir: None,
        log_level: Some("error".to_string()),
        service_discovery_config: None,
        prometheus_config: None,
        request_timeout_secs: 60,
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

    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();
    let payloads: Vec<Bytes> = (0..HELD_REQUESTS)
        .map(|index| {
            Bytes::from(
                serde_json::to_vec(&json!({
                    "request_id": format!("held-{index}"),
                    "session_id": format!("session-{index}"),
                    "prompt_ids": vec![123_u32; PROMPT_TOKENS],
                    "sampling_params": {"max_tokens": 1}
                }))
                .unwrap(),
            )
        })
        .collect();
    let body_bytes: usize = payloads.iter().map(Bytes::len).sum();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let before = LIVE_BYTES.load(Ordering::SeqCst);

    let mut pending = Vec::new();
    for (index, payload) in payloads.iter().enumerate() {
        let request = client
            .post(format!("http://{address}/verl/v1/generate"))
            .header("content-type", "application/json")
            .header("x-session-id", format!("session-{index}"))
            .body(payload.clone());
        pending.push(tokio::spawn(async move { request.send().await }));
    }
    tokio::time::timeout(Duration::from_secs(30), async {
        while received.load(Ordering::SeqCst) < HELD_REQUESTS {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("worker receives every request");
    tokio::time::sleep(Duration::from_millis(500)).await;
    let held = LIVE_BYTES.load(Ordering::SeqCst) - before;
    println!(
        "TRANSPARENT_BODY_RETENTION held_requests={HELD_REQUESTS} body_bytes={body_bytes} \
         live_heap_growth_while_held={held}"
    );

    release.notify_waiters();
    for request in pending {
        let response = request.await.unwrap().unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let _ = response.bytes().await.unwrap();
    }
    router_task.abort();
    backend_task.abort();

    // While a request is held, the HTTP client keeps the one serialized copy it
    // sent (reqwest retains a reusable body for retry and redirect) plus
    // connection buffers. The Router's own raw bytes, parsed tree and text
    // copy (about 20 times the body for token arrays) must be gone.
    assert!(
        held < 2 * body_bytes as isize,
        "the Router kept {held} heap bytes for {body_bytes} bytes of held request bodies"
    );
}
