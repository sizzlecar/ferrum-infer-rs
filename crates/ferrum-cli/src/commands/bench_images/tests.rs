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

pub(super) struct Server(tokio::task::JoinHandle<()>);
impl Drop for Server {
    fn drop(&mut self) {
        self.0.abort();
    }
}

pub(super) async fn serve(router: Router) -> (String, Server) {
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

pub(super) fn dataset(count: usize) -> ImageDataset {
    ImageDataset {
        schema_version: 1,
        mode: ImageRequestMode::Generate,
        samples: (0..count)
            .map(|i| ImageSample {
                id: format!("image-{i}"),
                prompt: format!("prompt-{i}"),
                seed: i as u64 + 100,
                width: 1,
                height: 1,
                num_inference_steps: None,
                mode: None,
            })
            .collect(),
    }
}

pub(super) fn config(url: String) -> ImageBenchmarkConfig {
    ImageBenchmarkConfig {
        endpoints: vec![ImageBenchmarkEndpoint {
            url,
            edit_url: None,
            model: "image-model".into(),
            server_config_declaration: "CPU protocol fixture, no model performance evidence".into(),
        }],
        concurrency: 8,
        arrival: ImageArrivalPolicy::ClosedLoop,
        maximum_in_flight: None,
        repetitions: 2,
        warmup_repetitions: 0,
        initial_server_state: InitialServerState::Unknown,
        timeout_seconds: 5,
        latency_slo_p99_ms: None,
        limits: ImageResponseLimits::default(),
        maximum_total_requests: 100,
        maximum_reference_image_bytes: ferrum_models::qwen_image21::image_input::ImageInputLimits::DEFAULT_MAXIMUM_ENCODED_BYTES,
        maximum_total_reference_image_bytes: 128 * 1024 * 1024,
        capture: None,
    }
}

// Exercise the production streaming writer. Encoding does not bypass the client's
// independent png decoder or its wire checks. No GPU/model substitute is claimed.
pub(super) fn valid_body() -> String {
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
    assert!(report.capture.is_none());
    assert_eq!(state.peak.load(Ordering::SeqCst), 8);
    assert_eq!(state.active.load(Ordering::SeqCst), 0);
    for endpoint in &report.endpoints {
        assert!(endpoint.captures.is_none());
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
    let directory = tempfile::tempdir().unwrap();
    let capture = directory.path().join("capture");
    cfg.capture = Some(PngCaptureConfig {
        directory: capture.clone(),
        maximum_capture_bytes: NonZeroU64::new(1024).unwrap(),
    });
    let report = run_benchmark(cfg, dataset(4)).await.unwrap();
    let endpoint = &report.endpoints[0];
    assert_eq!(endpoint.measured.errors, 4);
    assert_eq!(endpoint.measured.rejected, 1);
    assert_eq!(endpoint.measured.client_not_dispatched, 0);
    assert_eq!(endpoint.measured.other_errors, 3);
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
    let captured = report.capture.as_ref().unwrap();
    assert_eq!(
        (captured.saved, captured.failed, captured.not_produced),
        (0, 0, 4)
    );
    assert!(!captured.complete);
    assert_eq!(std::fs::read_dir(&capture).unwrap().count(), 0);
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

#[tokio::test]
async fn burst_reports_unsent_arrivals_separately_from_server_rejections() {
    let calls = Arc::new(AtomicUsize::new(0));
    let received = calls.clone();
    let (url, _server) = serve(Router::new().route(
        "/v1/images/generations",
        post(move || {
            received.fetch_add(1, Ordering::SeqCst);
            async { StatusCode::TOO_MANY_REQUESTS }
        }),
    ))
    .await;
    let mut cfg = config(url);
    cfg.repetitions = 1;
    cfg.arrival = ImageArrivalPolicy::Burst;
    cfg.maximum_in_flight = Some(2);
    cfg.latency_slo_p99_ms = Some(100_000.0);
    let report = run_benchmark(cfg, dataset(5)).await.unwrap();
    let endpoint = &report.endpoints[0];
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(endpoint.measured.attempted, 5);
    assert_eq!(endpoint.measured.dispatched, 2);
    assert_eq!(endpoint.measured.rejected, 2);
    assert_eq!(endpoint.measured.client_not_dispatched, 3);
    assert_eq!(endpoint.measured.other_errors, 0);
    assert_eq!(endpoint.measured.errors, 5);
    assert_eq!(
        endpoint.measured.latency_slo_status,
        ImageSloStatus::Violated
    );
    assert_eq!(endpoint.arrivals.len(), 5);
    for (index, arrival) in endpoint.arrivals.iter().enumerate() {
        assert_eq!(arrival.sample_index, index);
        assert_eq!(arrival.scheduled_offset_ms, Some(0.0));
        assert_eq!(arrival.dispatched_offset_ms.is_some(), index < 2);
        assert_eq!(arrival.dispatch_lag_ms.is_some(), index < 2);
    }
    for record in &endpoint.requests[2..] {
        assert_eq!(record.failure, Some(ImageFailureKind::ClientCapacity));
        assert_eq!(record.http_status, None);
    }
}

#[tokio::test]
async fn fixed_arrivals_do_not_wait_for_slow_responses_or_retry_unsent_samples() {
    let calls = Arc::new(AtomicUsize::new(0));
    let received = calls.clone();
    let ready = Arc::new(tokio::sync::Notify::new());
    let ready_server = ready.clone();
    let release = Arc::new(tokio::sync::Notify::new());
    let release_server = release.clone();
    let body = valid_body();
    let (url, _server) = serve(Router::new().route(
        "/v1/images/generations",
        post(move || {
            let ready = ready_server.clone();
            let release = release_server.clone();
            let received = received.clone();
            let body = body.clone();
            async move {
                let released = release.notified();
                tokio::pin!(released);
                released.as_mut().enable();
                if received.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                    ready.notify_one();
                }
                released.await;
                body
            }
        }),
    ))
    .await;
    let mut cfg = config(url);
    cfg.repetitions = 1;
    cfg.concurrency = 1;
    cfg.arrival = ImageArrivalPolicy::FixedInterval { interval_ms: 20.0 };
    cfg.maximum_in_flight = Some(2);
    let running = tokio::spawn(run_benchmark(cfg, dataset(3)));
    tokio::time::timeout(Duration::from_secs(2), ready.notified())
        .await
        .unwrap();
    // Both requests stay active beyond the final planned arrival. A closed-loop
    // implementation would defer the third request and send it after release.
    tokio::time::sleep(Duration::from_millis(100)).await;
    release.notify_waiters();
    let report = running.await.unwrap().unwrap();
    let endpoint = &report.endpoints[0];
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(endpoint.measured.succeeded, 2);
    assert_eq!(endpoint.measured.client_not_dispatched, 1);
    assert_eq!(endpoint.measured.rejected, 0);
    assert_eq!(endpoint.measured.other_errors, 0);
    assert_eq!(
        endpoint
            .arrivals
            .iter()
            .map(|record| record.scheduled_offset_ms)
            .collect::<Vec<_>>(),
        vec![Some(0.0), Some(20.0), Some(40.0)]
    );
    assert_eq!(endpoint.requests[2].sample_id, "image-2");
    assert_eq!(
        endpoint.requests[2].failure,
        Some(ImageFailureKind::ClientCapacity)
    );
}

#[test]
fn arrival_options_are_explicit_and_bounded() {
    for interval in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e-20] {
        assert!(
            ImageArrivalPolicy::from_cli(ImageArrivalMode::FixedInterval, Some(interval)).is_err()
        );
    }
    assert!(ImageArrivalPolicy::from_cli(ImageArrivalMode::FixedInterval, None).is_err());
    assert!(ImageArrivalPolicy::from_cli(ImageArrivalMode::ClosedLoop, Some(1.0)).is_err());
    assert!(ImageArrivalPolicy::from_cli(ImageArrivalMode::Burst, Some(1.0)).is_err());
    let mut cfg = config("http://127.0.0.1/v1/images/generations".into());
    cfg.maximum_in_flight = Some(2);
    assert!(cfg.validate(&dataset(1)).is_err());
    cfg.arrival = ImageArrivalPolicy::Burst;
    cfg.maximum_in_flight = Some(0);
    assert!(cfg.validate(&dataset(1)).is_err());
}

#[derive(clap::Parser)]
pub(super) struct TestCli {
    #[command(flatten)]
    pub(super) command: BenchImagesCommand,
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
    assert_eq!(
        cmd.maximum_reference_image_bytes,
        ferrum_models::qwen_image21::image_input::ImageInputLimits::default()
            .maximum_encoded_bytes()
    );
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

#[tokio::test]
async fn captures_original_pngs_for_every_endpoint_phase_and_repetition_with_numeric_paths() {
    use ferrum_bench_core::images::validate_image_response_with_png;
    let body = valid_body();
    let (expected, original) =
        validate_image_response_with_png(body.as_bytes(), 1, 1, &ImageResponseLimits::default())
            .unwrap()
            .into_parts();
    let (url, _server) = serve(Router::new().route(
        "/v1/images/generations",
        post(move || {
            let body = body.clone();
            async move { body }
        }),
    ))
    .await;
    let directory = tempfile::tempdir().unwrap();
    let capture_path = directory.path().join("capture");
    let mut cfg = config(url);
    cfg.endpoints.push(cfg.endpoints[0].clone());
    cfg.concurrency = 2;
    cfg.warmup_repetitions = 1;
    cfg.capture = Some(PngCaptureConfig {
        directory: capture_path.clone(),
        maximum_capture_bytes: NonZeroU64::new((original.len() * 12) as u64).unwrap(),
    });
    let mut data = dataset(2);
    data.samples[0].id = "../../escape.png".into();
    data.samples[1].id = "/absolute/path.png".into();
    data.samples[0].prompt = "../prompt/must/not/be/a/path".into();
    let report = run_benchmark(cfg, data).await.unwrap();
    let summary = report.capture.as_ref().unwrap();
    assert!(summary.complete);
    assert_eq!(
        (
            summary.requested,
            summary.saved,
            summary.failed,
            summary.not_produced
        ),
        (12, 12, 0, 0)
    );
    assert_eq!(summary.reserved_png_bytes, (original.len() * 12) as u64);
    let mut names = std::collections::BTreeSet::new();
    for endpoint in &report.endpoints {
        let captures = endpoint.captures.as_ref().unwrap();
        assert_eq!(captures.len(), endpoint.requests.len());
        for (captured, request) in captures.iter().zip(&endpoint.requests) {
            assert_eq!(captured.sample_id, request.sample_id);
            assert_eq!(captured.sample_index, request.sample_index);
            assert_eq!(captured.phase, request.phase);
            assert_eq!(captured.repetition, request.repetition);
            let PngCaptureResult::Saved {
                relative_path,
                png_bytes,
                png_sha256,
                capture_ms,
            } = &captured.result
            else {
                panic!("validated PNG was not captured");
            };
            assert!(names.insert(relative_path.clone()));
            assert_eq!(std::path::Path::new(relative_path).components().count(), 1);
            assert!(!relative_path.contains("escape") && !relative_path.contains("absolute"));
            assert_eq!(
                std::fs::read(capture_path.join(relative_path)).unwrap(),
                original
            );
            assert_eq!(*png_bytes, original.len());
            assert_eq!(png_sha256, &expected.png_sha256);
            assert!(capture_ms.is_finite() && *capture_ms >= 0.0);
            assert!(request.succeeded());
        }
    }
    assert_eq!(std::fs::read_dir(&capture_path).unwrap().count(), 12);
    assert!(!directory.path().join("escape.png").exists());
    assert!(!serde_json::to_string(&report).unwrap().contains("b64_json"));
}

#[test]
fn capture_cli_requires_both_options_and_a_nonzero_budget() {
    use clap::Parser;
    let base = [
        "bench-images",
        "--endpoint",
        "http://127.0.0.1/v1/images/generations",
        "--model",
        "fixture",
        "--server-config",
        "CPU fixture",
        "--dataset",
        "dataset.json",
        "--output",
        "report.json",
    ];
    for extra in [
        vec!["--capture-dir", "new"],
        vec!["--maximum-capture-bytes", "1"],
        vec!["--capture-dir", "new", "--maximum-capture-bytes", "0"],
    ] {
        assert!(TestCli::try_parse_from(base.into_iter().chain(extra)).is_err());
    }
    let plain = TestCli::try_parse_from(base).unwrap().command;
    assert!(plain.capture_dir.is_none() && plain.maximum_capture_bytes.is_none());
    let capture = TestCli::try_parse_from(base.into_iter().chain([
        "--capture-dir",
        "new",
        "--maximum-capture-bytes",
        "1",
    ]))
    .unwrap()
    .command;
    assert_eq!(capture.maximum_capture_bytes.unwrap().get(), 1);
}

#[tokio::test]
async fn capture_capacity_failure_saves_report_without_turning_http_success_into_server_error() {
    use clap::Parser;
    let body = valid_body();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let (url, _server) = serve(Router::new().route(
        "/v1/images/generations",
        post(move || {
            seen.fetch_add(1, Ordering::SeqCst);
            let body = body.clone();
            async move { body }
        }),
    ))
    .await;
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("dataset.json");
    let output = directory.path().join("report.json");
    let capture = directory.path().join("capture");
    std::fs::write(&data, serde_json::to_vec(&dataset(1)).unwrap()).unwrap();
    let mut cmd = TestCli::try_parse_from([
        "bench-images",
        "--endpoint",
        &url,
        "--model",
        "fixture",
        "--server-config",
        "CPU fixture",
        "--dataset",
        data.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--repetitions",
        "1",
        "--capture-dir",
        capture.to_str().unwrap(),
        "--maximum-capture-bytes",
        "1",
    ])
    .unwrap()
    .command;
    assert!(execute_inner(cmd.clone()).await.is_err());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["endpoints"][0]["measured"]["succeeded"], 1);
    assert_eq!(report["endpoints"][0]["measured"]["errors"], 0);
    assert!(report["endpoints"][0]["requests"][0]["failure"].is_null());
    assert_eq!(report["capture"]["complete"], false);
    assert_eq!(report["capture"]["failed"], 1);
    assert_eq!(
        report["endpoints"][0]["captures"][0]["result"]["kind"],
        "capacity"
    );
    assert_eq!(std::fs::read_dir(&capture).unwrap().count(), 0);
    // A different report path cannot make reuse of an existing capture directory
    // permissible. Refusal happens before another request is sent.
    cmd.output = directory.path().join("second-report.json");
    assert!(execute_inner(cmd).await.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
