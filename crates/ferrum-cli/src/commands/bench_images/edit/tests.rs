use super::*;
use crate::commands::bench_images::{execute_inner, run_benchmark, tests, PngCaptureConfig};
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    routing::post,
    Router,
};
use clap::Parser;
use ferrum_bench_core::images::{
    sha256, validate_image_response_with_png, ImageFailureKind, ImageReference, ImageResponseLimits,
};
use std::{
    collections::BTreeMap,
    num::NonZeroU64,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use tokio::sync::Barrier;

fn reference_bytes() -> Vec<u8> {
    validate_image_response_with_png(
        tests::valid_body().as_bytes(),
        1,
        1,
        &ImageResponseLimits::default(),
    )
    .unwrap()
    .into_parts()
    .1
}

fn edit_mode(path: PathBuf, expected: Option<String>) -> ImageRequestMode {
    ImageRequestMode::Edit {
        image: ImageReference {
            path,
            encoded_sha256: expected,
        },
    }
}

#[test]
fn reference_preparation_checks_actual_bytes_limits_hash_and_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.png");
    let png = reference_bytes();
    std::fs::write(&path, &png).unwrap();
    let mode = edit_mode(path.clone(), Some(sha256(&png)));
    let mut cfg = tests::config("http://127.0.0.1/v1/images/edits".into());
    cfg.maximum_reference_image_bytes = png.len();
    let prepared = PreparedReference::load(&mode, &cfg).unwrap().unwrap();
    assert_eq!(prepared.png.as_ref(), png);
    assert_eq!(prepared.identity.png_sha256, sha256(&png));
    cfg.maximum_reference_image_bytes -= 1;
    assert!(PreparedReference::load(&mode, &cfg).is_err());
    cfg.maximum_reference_image_bytes += 1;
    assert!(PreparedReference::load(&edit_mode(path.clone(), Some("0".repeat(64))), &cfg).is_err());
    assert!(PreparedReference::load(&edit_mode(dir.path().to_owned(), None), &cfg).is_err());
    std::fs::write(&path, b"not PNG").unwrap();
    assert!(PreparedReference::load(&edit_mode(path, None), &cfg).is_err());
    assert!(PreparedReference::load(&ImageRequestMode::Generate, &cfg)
        .unwrap()
        .is_none());
}

#[derive(Clone)]
struct EditState {
    calls: Arc<Mutex<Vec<(BTreeMap<String, String>, Vec<u8>)>>>,
    barrier: Arc<Barrier>,
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    source: Option<PathBuf>,
    reject_steps: bool,
}

async fn edit_endpoint(
    State(state): State<EditState>,
    mut multipart: Multipart,
) -> (StatusCode, String) {
    let mut fields = BTreeMap::new();
    let mut png = None;
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_owned();
        if name == "image" {
            assert_eq!(field.file_name(), Some("reference.png"));
            assert_eq!(field.content_type(), Some("image/png"));
            assert!(png.replace(field.bytes().await.unwrap().to_vec()).is_none());
        } else {
            assert!(fields.insert(name, field.text().await.unwrap()).is_none());
        }
    }
    let reject = state.reject_steps && fields.contains_key("num_inference_steps");
    {
        let mut calls = state.calls.lock().unwrap();
        // Subsequent repetitions/endpoints must send the retained, validated
        // file snapshot, even if the path changes after requests start.
        if calls.is_empty() {
            if let Some(path) = &state.source {
                std::fs::write(path, b"file replaced after preparation").unwrap();
            }
        }
        calls.push((fields, png.unwrap()));
    }
    let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
    state.peak.fetch_max(active, Ordering::SeqCst);
    state.barrier.wait().await;
    state.active.fetch_sub(1, Ordering::SeqCst);
    if reject {
        (StatusCode::BAD_REQUEST, "unsupported field".into())
    } else {
        (StatusCode::OK, tests::valid_body())
    }
}

fn state(concurrency: usize) -> EditState {
    EditState {
        calls: Arc::new(Mutex::new(Vec::new())),
        barrier: Arc::new(Barrier::new(concurrency)),
        active: Arc::new(AtomicUsize::new(0)),
        peak: Arc::new(AtomicUsize::new(0)),
        source: None,
        reject_steps: false,
    }
}

#[tokio::test]
async fn edits_share_one_actual_png_across_bounded_concurrency_endpoints_and_capture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.png");
    let png = reference_bytes();
    std::fs::write(&path, &png).unwrap();
    let mut state = state(8);
    state.source = Some(path.clone());
    let (url, _server) = tests::serve(
        Router::new()
            .route("/v1/images/edits", post(edit_endpoint))
            .with_state(state.clone()),
    )
    .await;
    let mut cfg = tests::config(url.replace("/generations", "/edits"));
    cfg.endpoints.push(cfg.endpoints[0].clone());
    cfg.capture = Some(PngCaptureConfig {
        directory: dir.path().join("capture"),
        maximum_capture_bytes: NonZeroU64::new((png.len() * 32) as u64).unwrap(),
    });
    let mut data = tests::dataset(8);
    data.schema_version = 2;
    data.mode = edit_mode(path, None);
    data.samples[0].prompt = "Make the teapot blue.\nKeep \"绿叶\" and lighting.".into();
    let expected_samples = data.samples.clone();
    let report = run_benchmark(cfg, data).await.unwrap();
    assert_eq!(report.schema_version, 2);
    assert_eq!(
        report.reference_image.as_ref().unwrap().png_sha256,
        sha256(&png)
    );
    let ImageRequestMode::Edit { image } = &report.dataset.mode else {
        unreachable!()
    };
    assert_eq!(image.encoded_sha256.as_ref().unwrap(), &sha256(&png));
    assert_eq!(report.dataset_sha256, report.dataset.fingerprint().unwrap());
    assert_eq!(state.peak.load(Ordering::SeqCst), 8);
    assert_eq!(state.active.load(Ordering::SeqCst), 0);
    assert!(report.capture.as_ref().unwrap().complete);
    let calls = state.calls.lock().unwrap();
    assert_eq!(calls.len(), 32);
    for round in calls.chunks(8) {
        let mut sorted = round.iter().collect::<Vec<_>>();
        sorted.sort_by_key(|(fields, _)| fields["seed"].parse::<u64>().unwrap());
        for ((fields, uploaded), sample) in sorted.into_iter().zip(&expected_samples) {
            assert_eq!(uploaded, &png);
            assert_eq!(fields["prompt"], sample.prompt);
            assert_eq!(fields["n"], "1");
            assert_eq!(fields["response_format"], "b64_json");
            assert_eq!(fields["size"], "1x1");
            assert!(!fields.contains_key("num_inference_steps"));
            assert!(!fields.contains_key("max_num_seqs"));
        }
    }
    for endpoint in &report.endpoints {
        assert_eq!(endpoint.measured.succeeded, 16);
        assert_eq!(endpoint.measured.errors, 0);
    }
}

#[tokio::test]
async fn explicit_steps_are_sent_and_server_rejection_is_a_recorded_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.png");
    std::fs::write(&path, reference_bytes()).unwrap();
    let mut state = state(1);
    state.reject_steps = true;
    let (url, _server) = tests::serve(
        Router::new()
            .route("/v1/images/edits", post(edit_endpoint))
            .with_state(state.clone()),
    )
    .await;
    let mut cfg = tests::config(url.replace("/generations", "/edits"));
    cfg.repetitions = 1;
    let mut data = tests::dataset(1);
    data.schema_version = 2;
    data.mode = edit_mode(path, None);
    data.samples[0].num_inference_steps = Some(8);
    let report = run_benchmark(cfg, data).await.unwrap();
    assert_eq!(state.calls.lock().unwrap()[0].0["num_inference_steps"], "8");
    assert_eq!(report.endpoints[0].measured.errors, 1);
    assert_eq!(
        report.endpoints[0].requests[0].failure,
        Some(ImageFailureKind::HttpStatus)
    );
}

#[tokio::test]
async fn edit_cli_resolves_dataset_relative_image_and_records_real_identity() {
    let dir = tempfile::tempdir().unwrap();
    let png = reference_bytes();
    std::fs::write(dir.path().join("source.png"), &png).unwrap();
    let data_path = dir.path().join("dataset.json");
    let output = dir.path().join("report.json");
    let mut data = tests::dataset(1);
    data.schema_version = 2;
    data.mode = edit_mode("source.png".into(), Some(sha256(&png)));
    let original = serde_json::to_vec(&data).unwrap();
    std::fs::write(&data_path, &original).unwrap();
    let state = state(1);
    let (url, _server) = tests::serve(
        Router::new()
            .route("/v1/images/edits", post(edit_endpoint))
            .with_state(state.clone()),
    )
    .await;
    let url = url.replace("/generations", "/edits");
    let command = tests::TestCli::try_parse_from([
        "bench-images",
        "--endpoint",
        &url,
        "--model",
        "fixture",
        "--server-config",
        "CPU protocol fixture; declared fixed schedule 8, unverified",
        "--dataset",
        data_path.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--concurrency",
        "1",
        "--repetitions",
        "1",
        "--maximum-reference-image-bytes",
        &png.len().to_string(),
    ])
    .unwrap()
    .command;
    execute_inner(command).await.unwrap();
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(report["reference_image"]["png_sha256"], sha256(&png));
    assert_eq!(report["dataset_file_sha256"], sha256(&original));
    assert_eq!(report["reference_image"]["png_bytes"], png.len());
    assert_eq!(report["evidence"]["server_configuration_verified"], false);
    assert!(!state.calls.lock().unwrap()[0]
        .0
        .contains_key("num_inference_steps"));
    let mut cfg = tests::config("http://127.0.0.1/v1/images/generations".into());
    assert!(cfg.validate(&data).is_err());
    cfg.endpoints[0].url = "http://127.0.0.1/v1/images/edits".into();
    cfg.maximum_reference_image_bytes = usize::MAX;
    assert!(cfg.validate(&data).is_err());
}

type MixedCalls = Arc<Mutex<Vec<(bool, u64, String, Option<String>)>>>;

fn sized_body(size: &str) -> String {
    use base64::Engine;
    use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
    let (width, height) = size.split_once('x').unwrap();
    let dimensions = PngDimensions::new(width.parse().unwrap(), height.parse().unwrap()).unwrap();
    let mut png = Vec::new();
    write_rgba8_png(&mut png, dimensions, PngLimits::default(), |sink| {
        sink.write_all(&vec![128; dimensions.rgba_bytes()])
    })
    .unwrap();
    serde_json::json!({"data":[{"b64_json":base64::engine::general_purpose::STANDARD.encode(png)}]})
        .to_string()
}

async fn mixed_generate(
    State(calls): State<MixedCalls>,
    axum::Json(body): axum::Json<serde_json::Value>,
) -> String {
    let size = body["size"].as_str().unwrap();
    calls
        .lock()
        .unwrap()
        .push((true, body["seed"].as_u64().unwrap(), size.into(), None));
    sized_body(size)
}

async fn mixed_edit(State(calls): State<MixedCalls>, mut multipart: Multipart) -> String {
    let mut fields = BTreeMap::new();
    let mut hash = None;
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_owned();
        if name == "image" {
            hash = Some(sha256(&field.bytes().await.unwrap()));
        } else {
            fields.insert(name, field.text().await.unwrap());
        }
    }
    calls.lock().unwrap().push((
        false,
        fields["seed"].parse().unwrap(),
        fields["size"].clone(),
        hash,
    ));
    sized_body(&fields["size"])
}

#[tokio::test]
async fn mixed_requests_preserve_modes_geometry_references_and_class_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let small = dir.path().join("small.png");
    let wide = dir.path().join("wide.png");
    let first = reference_bytes();
    let second = validate_image_response_with_png(
        sized_body("2x1").as_bytes(),
        2,
        1,
        &ImageResponseLimits::default(),
    )
    .unwrap()
    .into_parts()
    .1;
    std::fs::write(&small, &first).unwrap();
    std::fs::write(&wide, &second).unwrap();
    let calls = MixedCalls::default();
    let (url, _server) = tests::serve(
        Router::new()
            .route("/v1/images/generations", post(mixed_generate))
            .route("/v1/images/edits", post(mixed_edit))
            .with_state(calls.clone()),
    )
    .await;
    let mut cfg = tests::config(url.clone());
    cfg.endpoints[0].edit_url = Some(url.replace("/generations", "/edits"));
    cfg.repetitions = 1;
    cfg.maximum_total_reference_image_bytes = first.len() + second.len();
    let mut data = tests::dataset(5);
    data.schema_version = 3;
    data.samples[1].mode = Some(edit_mode(small.clone(), None));
    data.samples[2].width = 2;
    data.samples[3].width = 2;
    data.samples[3].mode = Some(edit_mode(wide, None));
    data.samples[4].mode = Some(edit_mode(small, None));
    let mut missing_route = cfg.clone();
    missing_route.endpoints[0].edit_url = None;
    assert!(missing_route.validate(&data).is_err());
    let mut short = cfg.clone();
    short.maximum_total_reference_image_bytes -= 1;
    assert!(run_benchmark(short, data.clone()).await.is_err());
    assert!(
        calls.lock().unwrap().is_empty(),
        "reference preflight must precede HTTP sends"
    );
    let report = run_benchmark(cfg, data).await.unwrap();
    assert_eq!(report.schema_version, 3);
    assert!(report.reference_image.is_none());
    assert_eq!(report.reference_images.len(), 2);
    assert_eq!(
        report.sample_reference_indices,
        vec![None, Some(0), None, Some(1), Some(0)]
    );
    assert_eq!(report.dataset_sha256, report.dataset.fingerprint().unwrap());
    let endpoint = &report.endpoints[0];
    assert_eq!(endpoint.measured.succeeded, 5);
    assert_eq!(endpoint.measured.errors, 0);
    assert_eq!(endpoint.measured_workloads.len(), 4);
    assert_eq!(
        endpoint
            .measured_workloads
            .iter()
            .map(|group| group.metrics.succeeded)
            .sum::<usize>(),
        5
    );
    for group in &endpoint.measured_workloads {
        assert!(group.metrics.e2e_p95_ms.is_some());
        assert_eq!(
            group.metrics.elapsed_seconds,
            endpoint.measured.elapsed_seconds
        );
    }
    let mut calls = calls.lock().unwrap().clone();
    calls.sort_by_key(|call| call.1);
    assert_eq!(
        calls.iter().map(|call| call.0).collect::<Vec<_>>(),
        vec![true, false, true, false, false]
    );
    assert_eq!(calls[1].3, Some(sha256(&first)));
    assert_eq!(calls[3].3, Some(sha256(&second)));
    assert_eq!(calls[4].3, Some(sha256(&first)));
}
