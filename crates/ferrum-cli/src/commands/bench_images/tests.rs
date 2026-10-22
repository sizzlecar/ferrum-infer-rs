use super::*;
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use ferrum_bench_core::images::{ImageFailureKind, ImageSample};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use tokio::sync::Barrier;

struct Server(tokio::task::JoinHandle<()>);
impl Drop for Server {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn serve(router: Router) -> (String, Server) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/v1/images/generations",
        listener.local_addr().unwrap()
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (url, Server(task))
}

fn dataset(count: usize) -> ImageDataset {
    ImageDataset {
        schema_version: 1,
        samples: (0..count)
            .map(|i| ImageSample {
                id: format!("image-{i}"),
                prompt: format!("prompt-{i}"),
                seed: i as u64 + 100,
                width: 1,
                height: 1,
                num_inference_steps: None,
            })
            .collect(),
    }
}

fn config(url: String) -> ImageBenchmarkConfig {
    ImageBenchmarkConfig {
        endpoints: vec![ImageBenchmarkEndpoint {
            url,
            model: "image-model".into(),
            server_config_declaration: "CPU protocol fixture, no model performance evidence".into(),
        }],
        concurrency: 8,
        repetitions: 2,
        warmup_repetitions: 0,
        initial_server_state: InitialServerState::Unknown,
        timeout_seconds: 5,
        latency_slo_p99_ms: None,
        limits: ImageResponseLimits::default(),
        maximum_total_requests: 100,
    }
}

// Exercise the production streaming writer. Encoding does not bypass the client's
// independent png decoder or its wire checks. No GPU/model substitute is claimed.
fn valid_body() -> String {
    use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
    let mut bytes = Vec::new();
    let dimensions = PngDimensions::new(1, 1).unwrap();
    write_rgba8_png(&mut bytes, dimensions, PngLimits::default(), |sink| {
        sink.write_all(&[1, 2, 3, 255])
    })
    .unwrap();
    serde_json::json!({"data":[{"b64_json":STANDARD.encode(bytes)}]}).to_string()
}

#[derive(Clone)]
struct BatchState {
    barrier: Arc<Barrier>,
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<serde_json::Value>>>,
    body: String,
}

async fn batched(
    State(state): State<BatchState>,
    Json(request): Json<serde_json::Value>,
) -> Response {
    state.requests.lock().unwrap().push(request);
    let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
    state.peak.fetch_max(active, Ordering::SeqCst);
    state.barrier.wait().await;
    state.active.fetch_sub(1, Ordering::SeqCst);
    ([("content-type", "application/json")], state.body).into_response()
}

#[tokio::test]
async fn drives_eight_in_flight_without_mutating_requests_across_endpoints_and_repetitions() {
    let state = BatchState {
        barrier: Arc::new(Barrier::new(8)),
        active: Arc::new(AtomicUsize::new(0)),
        peak: Arc::new(AtomicUsize::new(0)),
        requests: Arc::new(Mutex::new(Vec::new())),
        body: valid_body(),
    };
    let router = Router::new()
        .route("/v1/images/generations", post(batched))
        .with_state(state.clone());
    let (url, _server) = serve(router).await;
    let mut cfg = config(url);
    cfg.endpoints.push(cfg.endpoints[0].clone());
    let report = run_benchmark(cfg, dataset(8)).await.unwrap();
    assert_eq!(state.peak.load(Ordering::SeqCst), 8);
    assert_eq!(state.active.load(Ordering::SeqCst), 0);
    for endpoint in &report.endpoints {
        assert_eq!(endpoint.measured.succeeded, 16);
        assert_eq!(endpoint.measured.errors, 0);
        assert_eq!(
            endpoint.measured.latency_slo_status,
            ImageSloStatus::NotEvaluated
        );
        for (index, record) in endpoint.requests.iter().enumerate() {
            assert_eq!(record.sample_index, index % 8);
            assert_eq!(record.image.as_ref().unwrap().rgba8_bytes, 4);
        }
    }
    let requests = state.requests.lock().unwrap();
    for set in requests.chunks(8) {
        let mut set = set.to_vec();
        set.sort_by_key(|r| r["seed"].as_u64());
        for (index, request) in set.iter().enumerate() {
            assert_eq!(request["seed"], 100 + index as u64);
            assert_eq!(request["n"], 1);
            assert_eq!(request["size"], "1x1");
            assert_eq!(request["response_format"], "b64_json");
            assert!(request.get("max_num_seqs").is_none());
        }
    }
    assert!(!serde_json::to_string(&report).unwrap().contains("b64_json"));
    assert_eq!(report.evidence.quality_status, "not_evaluated");
}

async fn failures(Json(request): Json<serde_json::Value>) -> Response {
    match request["seed"].as_u64().unwrap() {
        100 => (StatusCode::TOO_MANY_REQUESTS, "secret-base64-payload").into_response(),
        101 => "{\"data\":[{\"b64_json\":\"secret-base64-payload\"}]}".into_response(),
        102 => Response::builder()
            .body(Body::from_stream(futures::stream::iter([
                Ok::<_, std::io::Error>(vec![b'x'; 96]),
                Ok(vec![b'y'; 96]),
            ])))
            .unwrap(),
        _ => Response::builder()
            .header("content-length", "500")
            .body(Body::from(vec![b'x'; 500]))
            .unwrap(),
    }
}

#[tokio::test]
async fn protocol_and_streaming_size_errors_are_recorded_without_body_leak_or_success_slo() {
    let (url, _server) = serve(Router::new().route("/v1/images/generations", post(failures))).await;
    let mut cfg = config(url);
    cfg.repetitions = 1;
    cfg.limits.maximum_response_bytes = 128;
    cfg.latency_slo_p99_ms = Some(100_000.0);
    let report = run_benchmark(cfg, dataset(4)).await.unwrap();
    let endpoint = &report.endpoints[0];
    assert_eq!(endpoint.measured.errors, 4);
    assert_eq!(
        endpoint.requests[0].failure,
        Some(ImageFailureKind::HttpStatus)
    );
    assert_eq!(
        endpoint.requests[1].failure,
        Some(ImageFailureKind::InvalidResponse)
    );
    assert_eq!(
        endpoint.requests[2].failure,
        Some(ImageFailureKind::ResponseLimit)
    );
    assert_eq!(
        endpoint.requests[3].failure,
        Some(ImageFailureKind::ResponseLimit)
    );
    assert_eq!(
        endpoint.measured.latency_slo_status,
        ImageSloStatus::Violated
    );
    assert_eq!(endpoint.measured.e2e_p50_ms, None);
    assert!(!serde_json::to_string(&report)
        .unwrap()
        .contains("secret-base64-payload"));
}

#[tokio::test]
async fn records_timeout_and_keeps_warmup_out_of_measured_counts() {
    let (url, _server) = serve(Router::new().route(
        "/v1/images/generations",
        post(|| async { std::future::pending::<String>().await }),
    ))
    .await;
    let mut cfg = config(url);
    cfg.timeout_seconds = 1;
    cfg.repetitions = 1;
    let timeout = run_benchmark(cfg, dataset(1)).await.unwrap();
    assert_eq!(
        timeout.endpoints[0].requests[0].failure,
        Some(ImageFailureKind::Timeout)
    );

    let body = valid_body();
    let (url, _server) = serve(Router::new().route(
        "/v1/images/generations",
        post(move || {
            let body = body.clone();
            async move { body }
        }),
    ))
    .await;
    let mut cfg = config(url);
    cfg.repetitions = 1;
    cfg.warmup_repetitions = 1;
    let warm = run_benchmark(cfg, dataset(1)).await.unwrap();
    assert_eq!(warm.endpoints[0].requests.len(), 2);
    assert_eq!(warm.endpoints[0].measured.attempted, 1);
    assert_eq!(
        warm.endpoints[0].repetitions[0].phase,
        ImageRequestPhase::Warmup
    );
    assert_eq!(
        warm.endpoints[0].repetitions[0].metrics.latency_slo_status,
        ImageSloStatus::NotEvaluated
    );
}

#[derive(clap::Parser)]
struct TestCli {
    #[command(flatten)]
    command: BenchImagesCommand,
}

#[tokio::test]
async fn cli_writes_reusable_artifact_and_refuses_existing_path_before_new_requests() {
    use clap::Parser;
    let body = valid_body();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let (url, _server) = serve(Router::new().route(
        "/v1/images/generations",
        post(move || {
            let body = body.clone();
            seen.fetch_add(1, Ordering::SeqCst);
            async move { body }
        }),
    ))
    .await;
    let directory = tempfile::tempdir().unwrap();
    let dataset_path = directory.path().join("dataset.json");
    let output = directory.path().join("report.json");
    let dataset_bytes = serde_json::to_vec(&dataset(1)).unwrap();
    std::fs::write(&dataset_path, &dataset_bytes).unwrap();
    let cmd = TestCli::try_parse_from([
        "bench-images",
        "--endpoint",
        &url,
        "--model",
        "fixture",
        "--server-config",
        "CPU protocol fixture",
        "--dataset",
        dataset_path.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--repetitions",
        "1",
        "--concurrency",
        "1",
    ])
    .unwrap()
    .command;
    execute_inner(cmd.clone()).await.unwrap();
    let original = std::fs::read(&output).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&original).unwrap();
    assert_eq!(report["dataset_file_sha256"], sha256(&dataset_bytes));
    assert_eq!(report["endpoints"][0]["measured"]["succeeded"], 1);
    assert_eq!(report["evidence"]["g3_status"], "not_evaluated");
    let error = execute_inner(cmd).await.unwrap_err();
    assert!(error.to_string().contains("reserve new report path"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(std::fs::read(&output).unwrap(), original);
}

#[test]
fn rejects_invalid_limits_endpoint_and_total_work_before_http() {
    let data = dataset(8);
    let mut cfg = config("http://127.0.0.1/v1/images/generations".into());
    cfg.validate(&data).unwrap();
    cfg.concurrency = 0;
    assert!(cfg.validate(&data).is_err());
    cfg.concurrency = 8;
    cfg.latency_slo_p99_ms = Some(f64::NAN);
    assert!(cfg.validate(&data).is_err());
    cfg.latency_slo_p99_ms = None;
    cfg.maximum_total_requests = 15;
    assert!(cfg.validate(&data).is_err());
    cfg.maximum_total_requests = 100;
    cfg.endpoints[0].url = "http://user:secret@example.test/path".into();
    assert!(cfg.validate(&data).is_err());
}
