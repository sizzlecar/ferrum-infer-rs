use super::*;
use std::time::Duration;

async fn start_server(
    metrics_enabled: bool,
) -> (
    Arc<AxumServer>,
    tokio::task::JoinHandle<ferrum_types::Result<()>>,
) {
    let mut engine = StubLlm::new("ok");
    engine.config.monitoring.enable_metrics = metrics_enabled;
    let server = Arc::new(AxumServer::from_llm(Arc::new(engine)));
    let running = Arc::clone(&server);
    let task = tokio::spawn(async move {
        running
            .start(&ServerConfig {
                host: "127.0.0.1".to_owned(),
                port: 0,
                ..Default::default()
            })
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        while !server.is_running() {
            assert!(
                !task.is_finished(),
                "server failed before entering running state"
            );
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("server reached its real start lifecycle");
    (server, task)
}

async fn metrics_text(server: &AxumServer) -> String {
    let response = server
        .build_router()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), AxumStatusCode::OK);
    response_text(response).await
}

fn counter_value(body: &str, name: &str, case: &str) -> f64 {
    body.lines()
        .find(|line| line.starts_with(name) && line.contains(case))
        .unwrap_or_else(|| panic!("missing real facade counter {name} for {case}"))
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap()
}

#[tokio::test]
async fn server_start_installs_and_reuses_real_metrics_recorder() {
    // No direct init helper: both instances must traverse HttpServer::start.
    let (first, first_task) = start_server(true).await;
    let case = uuid::Uuid::new_v4().to_string();
    let metric = "ferrum_test_server_startup_total";
    metrics::counter!(metric, "case" => case.clone()).increment(3);
    let before = metrics_text(&first).await;
    assert_eq!(counter_value(&before, metric, &case), 3.0);
    assert!(!before.contains("recorder not initialized"));

    let (second, second_task) = start_server(true).await;
    metrics::counter!(metric, "case" => case.clone()).increment(5);
    for server in [&first, &second] {
        let body = metrics_text(server).await;
        assert_eq!(counter_value(&body, metric, &case), 8.0);
        assert!(body.contains("ferrum_admission_"));
    }
    first.stop(Duration::from_secs(2)).await.unwrap();
    second.stop(Duration::from_secs(2)).await.unwrap();
    first_task.await.unwrap().unwrap();
    second_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn disabled_engine_does_not_export_another_servers_process_metrics() {
    let (enabled, enabled_task) = start_server(true).await;
    let case = uuid::Uuid::new_v4().to_string();
    let metric = "ferrum_test_disabled_server_total";
    metrics::counter!(metric, "case" => case.clone()).increment(1);
    let (disabled, disabled_task) = start_server(false).await;
    let body = metrics_text(&disabled).await;
    assert!(body.contains("Metrics disabled by engine monitoring configuration"));
    assert!(!body.contains(metric));
    assert!(!body.contains("ferrum_admission_"));
    assert_eq!(
        counter_value(&metrics_text(&enabled).await, metric, &case),
        1.0
    );
    enabled.stop(Duration::from_secs(2)).await.unwrap();
    disabled.stop(Duration::from_secs(2)).await.unwrap();
    enabled_task.await.unwrap().unwrap();
    disabled_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn recorder_installation_conflict_is_explicit_and_retains_existing_recorder() {
    let (server, task) = start_server(true).await;
    assert!(matches!(PROM_HANDLE.get(), Some(Ok(_))));
    // Attempt a real second global installation with independent ownership.
    // This is the same global-recorder conflict as an embedding application;
    // no invented handle may be rendered, nor may the caller panic.
    let competing = std::sync::OnceLock::new();
    install_prometheus_recorder(&competing);
    assert!(competing.get().unwrap().is_err());
    let unavailable = competing.get().unwrap();
    install_prometheus_recorder(&competing);
    assert!(std::ptr::eq(unavailable, competing.get().unwrap()));
    assert!(render_prometheus_recorder(competing.get()).contains("recorder unavailable"));

    let case = uuid::Uuid::new_v4().to_string();
    let metric = "ferrum_test_retained_recorder_total";
    metrics::counter!(metric, "case" => case.clone()).increment(2);
    assert_eq!(
        counter_value(
            &render_prometheus_recorder(PROM_HANDLE.get()),
            metric,
            &case
        ),
        2.0
    );
    server.stop(Duration::from_secs(2)).await.unwrap();
    task.await.unwrap().unwrap();
}

#[test]
fn metrics_setting_comes_from_installed_engines_or_typed_default() {
    assert_eq!(
        AppState::default().metrics_enabled(),
        ferrum_types::MonitoringConfig::default().enable_metrics
    );
    let mut engine = StubLlm::new("ok");
    engine.config.monitoring.enable_metrics = false;
    assert!(!AppState::default()
        .with_llm(Arc::new(engine))
        .metrics_enabled());
}
