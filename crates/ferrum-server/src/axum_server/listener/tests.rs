use super::*;
use crate::model_registry::LoraAdapterModel;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn bound_listener_reserves_the_port_until_ownership_is_dropped() {
    let listener = BoundHttpListener::bind("127.0.0.1", 0).await.unwrap();
    let address = listener.local_addr();
    assert_ne!(address.port(), 0);
    let error = BoundHttpListener::bind("127.0.0.1", address.port())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Network { .. }));
    assert!(error.to_string().contains(&address.to_string()));
    assert!(error.to_string().contains("already in use"));
    assert!(error.to_string().contains("--port"));
    drop(listener);
    let replacement = BoundHttpListener::bind("127.0.0.1", address.port())
        .await
        .unwrap();
    assert_eq!(replacement.local_addr(), address);
}

#[test]
fn permission_failure_explains_socket_permissions_and_windows_reservations() {
    // Permission-denied ports depend on host policy and privileges, so exercise
    // that OS error class directly instead of requiring a privileged machine.
    let error = bind_error(
        "127.0.0.1",
        18000,
        io::Error::from(io::ErrorKind::PermissionDenied),
    );
    assert!(matches!(error, Error::Network { .. }));
    let message = error.to_string();
    assert!(message.contains("127.0.0.1:18000"));
    assert!(message.contains("Permission"));
    assert!(message.contains("--port"));
    assert!(message.contains("reserved TCP port"));
}

#[test]
fn bind_failure_formats_ipv6_with_an_unambiguous_port() {
    let error = bind_error("::1", 8080, io::Error::from(io::ErrorKind::AddrInUse));
    assert!(error.to_string().contains("[::1]:8080"));
}

#[tokio::test]
async fn required_profile_failure_does_not_announce_ready_and_keeps_original_cleanup() {
    let listener = BoundHttpListener::bind("127.0.0.1", 0).await.unwrap();
    let address = listener.local_addr();
    let path = std::env::temp_dir().join(format!(
        "ferrum-listener-unavailable-{}.jsonl",
        Uuid::new_v4()
    ));
    let scopes = HttpJournalScopes::default()
        .with_open_failure(&path, io::Error::from(io::ErrorKind::PermissionDenied))
        .unwrap();
    let server = AxumServer::from_state(AppState::default()).with_journal_scopes(scopes);
    let config = ServerConfig {
        profile_jsonl: Some(path),
        profile_detail: ferrum_types::ObservabilityProfileDetail::Verify,
        ..Default::default()
    };
    let result = server
        .start_with_listener(&config, listener, |_| {
            panic!("failed required profile must not announce readiness")
        })
        .await;
    assert!(result.is_err());
    assert!(!server.is_running());
    assert_eq!(server.address(), None);
    assert!(server.profile_delivery_report().is_some());
    BoundHttpListener::bind("127.0.0.1", address.port())
        .await
        .unwrap();
    assert!(server.finish_stop().await.is_err());
    assert!(
        server
            .profile_delivery_report()
            .unwrap()
            .close_attempt_finished
    );
}

#[tokio::test]
async fn trait_start_reports_occupied_port_without_a_running_server() {
    let occupied = BoundHttpListener::bind("127.0.0.1", 0).await.unwrap();
    let server = AxumServer::from_state(AppState::default());
    let config = ServerConfig {
        host: "127.0.0.1".to_owned(),
        port: occupied.local_addr().port(),
        ..Default::default()
    };
    let error = server.start(&config).await.unwrap_err();
    assert!(matches!(error, Error::Network { .. }));
    assert!(!server.is_running());
    assert_eq!(server.address(), None);
    server.finish_stop().await.unwrap();
}

#[tokio::test]
async fn failed_capability_validation_never_announces_ready_and_releases_prebound_socket() {
    let listener = BoundHttpListener::bind("127.0.0.1", 0).await.unwrap();
    let address = listener.local_addr();
    let registry = ServedModelRegistry::try_new(
        "base",
        ServedModelKind::Llm,
        vec!["public".to_owned()],
        vec![LoraAdapterModel::new("sql", "public:sql", "/unused")],
    )
    .unwrap();
    let server = AxumServer::from_state(AppState::default()).with_served_model_registry(registry);
    let announcements = AtomicUsize::new(0);
    let result = server
        .start_with_listener(&ServerConfig::default(), listener, |_| {
            announcements.fetch_add(1, Ordering::AcqRel);
        })
        .await;
    assert!(matches!(result, Err(Error::Unsupported { .. })));
    assert_eq!(announcements.load(Ordering::Acquire), 0);
    assert!(!server.is_running());
    assert_eq!(server.address(), None);
    BoundHttpListener::bind("127.0.0.1", address.port())
        .await
        .unwrap();
    server.finish_stop().await.unwrap();
}

#[tokio::test]
async fn prebound_listener_serves_its_original_port_and_gracefully_stops() {
    let listener = BoundHttpListener::bind("127.0.0.1", 0).await.unwrap();
    let address = listener.local_addr();
    let server = Arc::new(AxumServer::from_state(AppState::default()));
    assert_eq!(server.address(), None);
    let (ready, announced) = tokio::sync::oneshot::channel();
    let task = {
        let server = Arc::clone(&server);
        tokio::spawn(async move {
            // This configuration cannot replace the original listener. Any
            // accidental second bind would listen elsewhere and fail this test.
            let config = ServerConfig {
                host: "127.0.0.1".to_owned(),
                port: 0,
                ..Default::default()
            };
            server
                .start_with_listener(&config, listener, |address| {
                    ready.send(address).unwrap();
                })
                .await
        })
    };
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), announced)
            .await
            .unwrap()
            .unwrap(),
        address
    );
    assert!(server.is_running());
    assert_eq!(server.address(), Some(address));
    let response = tokio::time::timeout(Duration::from_secs(5), async {
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        response
    })
    .await
    .unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    tokio::time::timeout(Duration::from_secs(5), server.finish_stop())
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!server.is_running());
    assert_eq!(server.address(), None);
    assert!(tokio::net::TcpStream::connect(address).await.is_err());
}

#[tokio::test]
async fn shutdown_before_start_drops_listener_without_announcing_ready() {
    let listener = BoundHttpListener::bind("127.0.0.1", 0).await.unwrap();
    let address = listener.local_addr();
    let server = AxumServer::from_state(AppState::default());
    server.finish_stop().await.unwrap();
    let result = server
        .start_with_listener(&ServerConfig::default(), listener, |_| {
            panic!("stopped server must not announce readiness")
        })
        .await;
    assert!(result.is_err());
    assert!(!server.is_running());
    BoundHttpListener::bind("127.0.0.1", address.port())
        .await
        .unwrap();
}
