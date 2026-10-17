#![cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]

use std::{
    ffi::{c_int, c_uint, c_void},
    net::{SocketAddr, TcpStream as StdTcpStream},
    os::fd::{AsRawFd, FromRawFd},
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, Notify},
};
use vllm_router_rs::{
    config::{RouterConfig, RoutingMode},
    server::{startup, ServerConfig},
};

// Linux socket ABI on the supported runtime architectures. Keep inspection
// test-only so this mitigation needs no manifest/dependency change.
unsafe extern "C" {
    fn dup(fd: c_int) -> c_int;
    fn getsockopt(
        fd: c_int,
        level: c_int,
        name: c_int,
        value: *mut c_void,
        size: *mut c_uint,
    ) -> c_int;
}
const SOL_SOCKET: c_int = 1;
const SO_KEEPALIVE: c_int = 9;
const IPPROTO_TCP: c_int = 6;
const TCP_NODELAY: c_int = 1;
const TCP_KEEPIDLE: c_int = 4;
const TCP_KEEPINTVL: c_int = 5;
const TCP_KEEPCNT: c_int = 6;
const TCP_USER_TIMEOUT: c_int = 18;

fn socket_option(socket: &StdTcpStream, level: i32, name: i32) -> i32 {
    let mut value: c_int = 0;
    let mut size = std::mem::size_of_val(&value) as c_uint;
    // The owned duplicate remains open for this call and the buffer/length agree.
    let result = unsafe {
        getsockopt(
            socket.as_raw_fd(),
            level,
            name,
            (&mut value as *mut c_int).cast(),
            &mut size,
        )
    };
    assert_eq!(result, 0, "getsockopt: {}", std::io::Error::last_os_error());
    assert_eq!(size as usize, std::mem::size_of_val(&value));
    value
}

fn outbound_socket(local: SocketAddr, remote: SocketAddr) -> StdTcpStream {
    for entry in std::fs::read_dir("/proc/self/fd").unwrap().flatten() {
        let Ok(fd) = entry.file_name().to_string_lossy().parse::<i32>() else {
            continue;
        };
        // dup atomically takes independent ownership; stale descriptors return EBADF.
        let duplicated = unsafe { dup(fd) };
        if duplicated < 0 {
            continue;
        }
        // This descriptor was just returned by dup and is closed only by this owner.
        let socket = unsafe { StdTcpStream::from_raw_fd(duplicated) };
        if socket.local_addr().ok() == Some(local) && socket.peer_addr().ok() == Some(remote) {
            return socket;
        }
    }
    panic!("actual Router outbound socket {local} -> {remote} not found");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn actual_router_client_sets_transport_floor_and_keeps_request_deadline() {
    for request_budget in [1_u64, 1800, 5400] {
        let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend_address = backend.local_addr().unwrap();
        let (observed, mut peers) = mpsc::channel(4);
        let release = std::sync::Arc::new(Notify::new());
        let backend_release = release.clone();
        let backend_task = tokio::spawn(async move {
            loop {
                let (mut socket, peer) = backend.accept().await.unwrap();
                let observed = observed.clone();
                let release = backend_release.clone();
                tokio::spawn(async move {
                    let mut header = Vec::new();
                    while !header.ends_with(b"\r\n\r\n") {
                        let mut byte = [0_u8; 1];
                        if socket.read_exact(&mut byte).await.is_err() {
                            return;
                        }
                        header.push(byte[0]);
                        assert!(header.len() < 65536);
                    }
                    if header.starts_with(b"POST /timeout_probe ") {
                        observed.send(peer).await.unwrap();
                        release.notified().await;
                    }
                    let _ = socket
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                        )
                        .await;
                });
            }
        });
        let router_port = portpicker::pick_unused_port().unwrap();
        let router_address: SocketAddr = format!("127.0.0.1:{router_port}").parse().unwrap();
        let mut router_config = RouterConfig::default();
        router_config.mode = RoutingMode::Regular {
            worker_urls: vec![format!("http://{backend_address}")],
        };
        router_config.request_timeout_secs = request_budget;
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: router_port,
            router_config,
            max_payload_size: 4096,
            log_dir: None,
            log_level: Some("ERROR".to_string()),
            service_discovery_config: None,
            prometheus_config: None,
            request_timeout_secs: request_budget,
            pool_idle_timeout_secs: 4,
            request_id_headers: None,
            trace_config: None,
        };
        let router_task = tokio::spawn(async move {
            if let Err(error) = startup(config).await {
                panic!("Router startup: {error}");
            }
        });
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if TcpStream::connect(router_address).await.is_ok() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let frontend = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(8))
            .build()
            .unwrap();
        let started = Instant::now();
        let request = tokio::spawn(async move {
            frontend
                .post(format!("http://{router_address}/timeout_probe"))
                .json(&serde_json::json!({}))
                .send()
                .await
        });
        let actual_peer = tokio::time::timeout(Duration::from_secs(5), peers.recv())
            .await
            .unwrap()
            .unwrap();
        let outbound = outbound_socket(actual_peer, backend_address);
        let transport_ms = socket_option(&outbound, IPPROTO_TCP, TCP_USER_TIMEOUT);
        assert_eq!(transport_ms as u64, request_budget.max(600) * 1000);
        assert_eq!(socket_option(&outbound, SOL_SOCKET, SO_KEEPALIVE), 1);
        assert_eq!(socket_option(&outbound, IPPROTO_TCP, TCP_KEEPIDLE), 30);
        assert_eq!(socket_option(&outbound, IPPROTO_TCP, TCP_KEEPINTVL), 15);
        assert_eq!(socket_option(&outbound, IPPROTO_TCP, TCP_KEEPCNT), 3);
        assert_eq!(socket_option(&outbound, IPPROTO_TCP, TCP_NODELAY), 1);
        println!("ACTUAL_OUTBOUND_SOCKET budget_s={request_budget} tcp_user_timeout_ms={transport_ms} local={actual_peer} remote={backend_address} keepalive=1 idle_s=30 interval_s=15 retries=3");
        if request_budget > 1 {
            tokio::time::sleep(Duration::from_millis(1100)).await;
            release.notify_one();
        }
        let response = tokio::time::timeout(Duration::from_secs(5), request)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if request_budget == 1 {
            assert_eq!(response.status(), reqwest::StatusCode::BAD_GATEWAY);
            assert!(started.elapsed() >= Duration::from_millis(800));
            assert!(started.elapsed() < Duration::from_secs(5));
            release.notify_one();
        } else {
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            assert!(started.elapsed() >= Duration::from_millis(1100));
        }
        println!(
            "UNCHANGED_HTTP_DEADLINE budget_s={request_budget} status={} elapsed_ms={}",
            response.status(),
            started.elapsed().as_millis()
        );
        drop(outbound);
        router_task.abort();
        backend_task.abort();
        let _ = router_task.await;
        let _ = backend_task.await;
    }
}
