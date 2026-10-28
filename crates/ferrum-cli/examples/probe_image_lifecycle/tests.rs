use super::*;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use ferrum_bench_core::images::{ImageRequestMode, ImageSample};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn dataset() -> ImageDataset {
    ImageDataset {
        schema_version: 1,
        mode: ImageRequestMode::Generate,
        samples: vec![ImageSample {
            id: "fixed-0".into(),
            prompt: "fixture".into(),
            seed: 42,
            width: 1,
            height: 1,
            num_inference_steps: None,
            mode: None,
        }],
    }
}
fn arguments(endpoint: String, mode: Mode) -> Arguments {
    Arguments {
        health_endpoint: endpoint.replace("/v1/images/generations", "/health"),
        endpoint,
        edit_endpoint: None,
        model: "fixture".into(),
        dataset: "fixture.json".into(),
        output: "unused.json".into(),
        mode,
        requests: 1,
        cancel_after_ms: 5,
        hold_body_ms: 20,
        observation_ms: 30,
        poll_ms: 10,
        request_timeout_ms: 1000,
        health_timeout_ms: 500,
        recovery_requests: 1,
        recovery_delay_ms: 0,
        maximum_health_snapshots: 128,
        maximum_response_bytes: 64 * 1024 * 1024,
        maximum_health_bytes: 64 * 1024,
        maximum_dataset_bytes: 4 * 1024 * 1024,
        maximum_reference_bytes: 64 * 1024 * 1024,
    }
}
fn valid_body() -> String {
    use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
    let mut png = Vec::new();
    write_rgba8_png(
        &mut png,
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        |sink| sink.write_all(&[1, 2, 3, 255]),
    )
    .unwrap();
    serde_json::json!({"data":[{"b64_json":STANDARD.encode(png)}]}).to_string()
}
fn health_json(accepting: bool) -> serde_json::Value {
    serde_json::json!({"images": {
        "scheduler": {"accepting":accepting,"active_requests":0,"pending_requests":0,"preparing_requests":0,"pending_bytes":0,
            "rejected_requests":0,"executed_batches":0,"executed_requests":0,"last_batch_size":0,"capacity_reductions":0,
            "maximum_pending":16,"maximum_pending_bytes":4096,"max_num_seqs":4},
        "output":{"sessions":0,"encoded_bytes":0},"host":{"committed_bytes":1024,"allocated_bytes":1024}
    }})
}
#[derive(Clone)]
struct Fake {
    calls: Arc<AtomicUsize>,
    health_calls: Arc<AtomicUsize>,
    first_delay_ms: u64,
    body: String,
    status: StatusCode,
    stop_after_baseline: bool,
}
impl Fake {
    fn normal() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            health_calls: Arc::new(AtomicUsize::new(0)),
            first_delay_ms: 0,
            body: valid_body(),
            status: StatusCode::OK,
            stop_after_baseline: false,
        }
    }
}
async fn image(State(state): State<Fake>) -> impl IntoResponse {
    if state.calls.fetch_add(1, Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(state.first_delay_ms)).await;
    }
    (
        state.status,
        [("content-type", "application/json")],
        state.body,
    )
}
async fn health_response(State(state): State<Fake>) -> impl IntoResponse {
    let first = state.health_calls.fetch_add(1, Ordering::SeqCst) == 0;
    axum::Json(health_json(!state.stop_after_baseline || first))
}
struct Server(tokio::task::JoinHandle<()>);
impl Drop for Server {
    fn drop(&mut self) {
        self.0.abort();
    }
}
async fn server(state: Fake) -> (String, Server) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!(
        "http://{}/v1/images/generations",
        listener.local_addr().unwrap()
    );
    let app = Router::new()
        .route("/v1/images/generations", post(image))
        .route("/health", get(health_response))
        .with_state(state);
    (
        endpoint,
        Server(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        })),
    )
}
async fn probe(args: &Arguments) -> serde_json::Value {
    let dataset = dataset();
    args.validate(&dataset).unwrap();
    let report = run::run(args, request::prepare(args, dataset).unwrap())
        .await
        .unwrap();
    serde_json::to_value(report).unwrap()
}

#[tokio::test]
async fn holding_headers_is_a_client_action_and_does_not_invent_server_retention() {
    let (endpoint, _server) = server(Fake::normal()).await;
    let report = probe(&arguments(endpoint, Mode::HoldBody)).await;
    assert_eq!(report["unexpected_failures"], 0);
    assert_eq!(report["findings"]["client_body_hold"]["status"], "observed");
    assert_eq!(
        report["findings"]["output_retained_during_client_hold"]["status"],
        "not_observed"
    );
    assert_eq!(
        report["findings"]["validated_recovery_requests"]["status"],
        "observed"
    );
    assert!(report["requests"][0]["headers_received_ms"].is_number());
    let before = report["requests"][0]["body_hold_started_ms"]
        .as_f64()
        .unwrap();
    let after = report["requests"][0]["body_hold_finished_ms"]
        .as_f64()
        .unwrap();
    assert!(after >= before + 20.0);
    assert!(report["scope"]
        .as_str()
        .unwrap()
        .contains("does not establish server/backend cancellation"));
}

#[tokio::test]
async fn cancellation_and_validated_recovery_are_separate_observations() {
    let mut fake = Fake::normal();
    fake.first_delay_ms = 100;
    let (endpoint, _server) = server(fake).await;
    let report = probe(&arguments(endpoint, Mode::Cancel)).await;
    assert_eq!(report["expected_client_cancellations"], 1);
    assert_eq!(report["unexpected_failures"], 0);
    assert_eq!(report["requests"][0]["outcome"], "client_future_dropped");
    assert_eq!(report["requests"][1]["outcome"], "validated");
    assert!(report["findings"].get("server_cancelled").is_none());
}

#[tokio::test]
async fn observed_shutdown_503_is_distinct_from_unexpected_png_corruption() {
    let mut stopped = Fake::normal();
    stopped.stop_after_baseline = true;
    stopped.status = StatusCode::SERVICE_UNAVAILABLE;
    let (endpoint, _server) = server(stopped).await;
    let report = probe(&arguments(endpoint, Mode::ObserveShutdown)).await;
    assert_eq!(report["unexpected_failures"], 0);
    assert_eq!(report["expected_shutdown_transport_or_503"], 2);
    assert_eq!(
        report["findings"]["shutdown_or_unavailability"]["status"],
        "observed"
    );
    assert_eq!(
        report["findings"]["validated_recovery_requests"]["status"],
        "not_observed"
    );
    let mut corrupt = Fake::normal();
    corrupt.stop_after_baseline = true;
    corrupt.body = "invalid-json".into();
    let (endpoint, _server) = server(corrupt).await;
    let report = probe(&arguments(endpoint, Mode::ObserveShutdown)).await;
    assert_eq!(report["unexpected_failures"], 2);
    assert_eq!(
        report["findings"]["unexpected_failures"]["status"],
        "failed"
    );
}

#[tokio::test]
async fn body_capacity_failure_is_not_hidden_by_observe_shutdown_mode() {
    let mut fake = Fake::normal();
    fake.stop_after_baseline = true;
    let (endpoint, _server) = server(fake).await;
    let mut args = arguments(endpoint, Mode::ObserveShutdown);
    args.maximum_response_bytes = 1;
    let client = reqwest::Client::new();
    let prepared = request::prepare(
        &arguments(args.endpoint.clone(), Mode::ObserveShutdown),
        dataset(),
    )
    .unwrap();
    let record = request::execute(
        &client,
        &args,
        &prepared.requests[0],
        0,
        request::Phase::Perturbation,
        Instant::now(),
    )
    .await;
    assert_eq!(record.outcome, request::Outcome::InvalidResponse);
}

#[test]
fn strict_health_missing_values_and_bounded_log_never_become_zero_evidence() {
    assert!(health::parse(b"{}").is_err());
    let values = health::parse(&serde_json::to_vec(&health_json(true)).unwrap()).unwrap();
    assert_eq!(values.get("output.sessions"), Some(0));
    let mut log = health::Log::default();
    for _ in 0..4 {
        log.push(
            health::Snapshot {
                phase: "test",
                started_ms: 0.0,
                finished_ms: 1.0,
                http_status: None,
                values: None,
                failure: Some(health::Failure::Transport),
                error: Some("fixture".into()),
            },
            2,
        );
    }
    assert_eq!(log.snapshots.len(), 2);
    assert_eq!(log.omitted, 2);
    assert!(health::comparisons(&log)
        .values()
        .all(|delta| delta.baseline.is_none() && delta.final_minus_baseline.is_none()));
}

#[test]
fn probe_rejects_remote_urls_and_unbounded_request_sets() {
    assert!(local_url("http://example.com/health").is_err());
    assert!(local_url("http://localhost/health").is_err());
    assert!(local_url("http://127.0.0.1/health?token=secret").is_err());
    let mut args = arguments(
        "http://127.0.0.1:1/v1/images/generations".into(),
        Mode::Cancel,
    );
    args.validate(&dataset()).unwrap();
    args.requests = 33;
    assert!(args.validate(&dataset()).is_err());
}
