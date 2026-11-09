//! Actual HTTP protocol consumers, with PNGs from the production Rust writer.
//! These fixtures are neither GPU/model substitutes nor performance evidence.

use super::super::tests::{config, dataset, serve, valid_body, TestCli};
use super::*;
use axum::{
    body::Body,
    extract::{
        ws::{Message as WsMessage, WebSocketUpgrade},
        Multipart, Path, Query, State,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use ferrum_bench_core::images::{ImageFailureKind, ImageReference, ImageRequestMode};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Normal,
    BadFilename,
    MultipleImages,
    ExecutionError,
    Pending,
    UnknownSubmission,
    Redirect,
    OversizedMetadata,
    WrongAcknowledgement,
    CachedRequired,
    MissingExecution,
    ForeignExecution,
    AuditDisconnect,
    AuditTail,
    MutatedHistory,
    MetadataHistory,
    OversizedAudit,
}

#[derive(Clone)]
struct Fixture {
    mode: Mode,
    jobs: Arc<Mutex<BTreeMap<String, Value>>>,
    submitted: Arc<Mutex<Vec<Value>>>,
    cancelled: Arc<Mutex<Vec<String>>>,
    uploads: Arc<Mutex<Vec<Vec<u8>>>>,
    views: Arc<AtomicUsize>,
    polls: Arc<Mutex<BTreeMap<String, usize>>>,
    rewrite: Arc<Mutex<Option<PathBuf>>>,
    release_late: Arc<tokio::sync::Notify>,
    late_enqueued: Arc<tokio::sync::Notify>,
    png: Vec<u8>,
    audit_clients: Arc<Mutex<BTreeMap<String, tokio::sync::mpsc::Sender<WsMessage>>>>,
}

impl Fixture {
    fn new(mode: Mode) -> Self {
        let body: Value = serde_json::from_str(&valid_body()).unwrap();
        Self {
            mode,
            jobs: Arc::default(),
            submitted: Arc::default(),
            cancelled: Arc::default(),
            uploads: Arc::default(),
            views: Arc::default(),
            polls: Arc::default(),
            rewrite: Arc::default(),
            release_late: Arc::default(),
            late_enqueued: Arc::default(),
            audit_clients: Arc::default(),
            png: STANDARD
                .decode(body["data"][0]["b64_json"].as_str().unwrap())
                .unwrap(),
        }
    }
    fn router(&self) -> Router {
        Router::new()
            .route("/prompt", post(submit))
            .route("/ws", get(websocket))
            .route("/history/:id", get(history))
            .route("/view", get(view))
            .route("/upload/image", post(upload))
            .route("/api/jobs/:id/cancel", post(cancel))
            .route("/api/jobs/:id", get(job))
            .with_state(self.clone())
    }
}

async fn websocket(
    State(state): State<Fixture>,
    Query(query): Query<BTreeMap<String, String>>,
    socket: WebSocketUpgrade,
) -> Response {
    let id = query.get("clientId").unwrap().clone();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(16);
    // Register before returning the upgrade response so POST cannot race setup.
    state.audit_clients.lock().unwrap().insert(id, sender);
    socket
        .on_upgrade(move |mut socket| async move {
            let _ = socket
                .send(WsMessage::Text(
                    json!({"type":"status","data":{"sid":"fixture"}}).to_string(),
                ))
                .await;
            // Reconnect-style events lack a prompt identity and cannot prove work.
            let _ = socket
                .send(WsMessage::Text(
                    json!({"type":"executing","data":{"node":"input"}}).to_string(),
                ))
                .await;
            while let Some(message) = receiver.recv().await {
                if socket.send(message).await.is_err() {
                    break;
                }
            }
        })
        .into_response()
}

async fn audit_events(state: &Fixture, id: &str) {
    let Some(sender) = state.audit_clients.lock().unwrap().get(id).cloned() else {
        return;
    };
    if state.mode == Mode::OversizedAudit {
        let _ = sender.send(WsMessage::Text("x".repeat(2048))).await;
        return;
    }
    let events = [
        json!({"type":"execution_start","data":{"prompt_id":id}}),
        json!({"type":"execution_cached","data":{"prompt_id":id,"nodes":if state.mode == Mode::CachedRequired {vec!["input"]}else{vec![]}}}),
    ];
    for event in events {
        let _ = sender.send(WsMessage::Text(event.to_string())).await;
    }
    if state.mode == Mode::AuditDisconnect {
        let _ = sender.send(WsMessage::Close(None)).await;
        return;
    }
    for node in ["input", "out"] {
        if node == "input" && state.mode == Mode::MissingExecution {
            continue;
        }
        let prompt = if state.mode == Mode::ForeignExecution {
            "foreign-prompt"
        } else {
            id
        };
        let event = json!({"type":"executing","data":{"prompt_id":prompt,"node":node,"display_node":"ignored"}});
        let _ = sender.send(WsMessage::Text(event.to_string())).await;
    }
    let terminal =
        WsMessage::Text(json!({"type":"execution_success","data":{"prompt_id":id}}).to_string());
    if state.mode == Mode::AuditTail {
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = sender.send(terminal).await;
        });
    } else {
        let _ = sender.send(terminal).await;
    }
}

async fn submit(State(state): State<Fixture>, Json(body): Json<Value>) -> Response {
    let id = body["prompt_id"].as_str().unwrap().to_owned();
    assert_eq!(uuid::Uuid::parse_str(&id).unwrap().to_string(), id);
    assert_eq!(body["client_id"], id);
    state.submitted.lock().unwrap().push(body.clone());
    if let Some(path) = state.rewrite.lock().unwrap().take() {
        std::fs::write(path, b"not JSON anymore").unwrap();
    }
    if state.mode == Mode::UnknownSubmission {
        let late = state.clone();
        tokio::spawn(async move {
            late.release_late.notified().await;
            late.jobs.lock().unwrap().insert(id, body);
            late.late_enqueued.notify_one();
        });
        return Response::builder()
            .status(200)
            .body(Body::from_stream(futures::stream::pending::<
                Result<bytes::Bytes, std::io::Error>,
            >()))
            .unwrap();
    }
    state.jobs.lock().unwrap().insert(id.clone(), body);
    audit_events(&state, &id).await;
    if state.mode == Mode::OversizedMetadata {
        return Response::builder()
            .status(200)
            .body(Body::from_stream(futures::stream::iter([
                Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b' '; 256])),
                Ok(bytes::Bytes::from(vec![b' '; 256])),
            ])))
            .unwrap();
    }
    let acknowledged = if state.mode == Mode::WrongAcknowledgement {
        uuid::Uuid::new_v4().to_string()
    } else {
        id
    };
    Json(json!({"prompt_id":acknowledged,"node_errors":{}})).into_response()
}

async fn history(State(state): State<Fixture>, Path(id): Path<String>) -> Json<Value> {
    if matches!(
        state.mode,
        Mode::Pending | Mode::UnknownSubmission | Mode::OversizedMetadata
    ) {
        return Json(json!({}));
    }
    let request = state.jobs.lock().unwrap().get(&id).cloned();
    let Some(mut request) = request else {
        return Json(json!({}));
    };
    let mut polls = state.polls.lock().unwrap();
    let count = polls.entry(id.clone()).or_default();
    *count += 1;
    if *count == 1 {
        return Json(json!({}));
    }
    let name = if state.mode == Mode::BadFilename {
        "https://foreign.invalid/image.png"
    } else {
        "result.png"
    };
    let descriptor = json!({"filename":name,"subfolder":"owned","type":"output"});
    let images = if state.mode == Mode::MultipleImages {
        vec![descriptor.clone(), descriptor]
    } else {
        vec![descriptor]
    };
    let error = state.mode == Mode::ExecutionError;
    if state.mode == Mode::MetadataHistory {
        request["prompt"]["input"]["is_changed"] = json!("server metadata");
    }
    if state.mode == Mode::MutatedHistory {
        request["prompt"]["input"]["inputs"]["prompt"] = json!("different request");
    }
    Json(
        json!({(id.clone()):{"prompt":[0,id.clone(),request["prompt"],{"client_id":id},["out"]],
        "status":{"status_str":if error {"error"} else {"success"},"completed":!error},
        "outputs":{"out":{"images":images}}}}),
    )
}

async fn view(State(state): State<Fixture>) -> Response {
    state.views.fetch_add(1, Ordering::SeqCst);
    if state.mode == Mode::Redirect {
        return (
            StatusCode::FOUND,
            [(header::LOCATION, "http://foreign.invalid/image.png")],
        )
            .into_response();
    }
    ([(header::CONTENT_TYPE, "image/png")], state.png.clone()).into_response()
}

async fn upload(State(state): State<Fixture>, mut form: Multipart) -> Json<Value> {
    let mut folder = String::new();
    while let Some(field) = form.next_field().await.unwrap() {
        match field.name().unwrap() {
            "image" => {
                let bytes = field.bytes().await.unwrap().to_vec();
                state.uploads.lock().unwrap().push(bytes);
            }
            "subfolder" => {
                folder = field.text().await.unwrap();
            }
            "overwrite" => {
                assert_eq!(field.text().await.unwrap(), "false");
            }
            "type" => {
                assert_eq!(field.text().await.unwrap(), "input");
            }
            _ => panic!("unexpected upload field"),
        }
    }
    Json(json!({"name":"reference.png","subfolder":folder,"type":"input"}))
}

async fn cancel(State(state): State<Fixture>, Path(id): Path<String>) -> Json<Value> {
    state.cancelled.lock().unwrap().push(id.clone());
    let removed = state.jobs.lock().unwrap().remove(&id).is_some();
    Json(json!({"cancelled":removed}))
}
async fn job(State(state): State<Fixture>, Path(id): Path<String>) -> Response {
    if state.jobs.lock().unwrap().contains_key(&id) {
        Json(json!({"id":id,"status":"in_progress"})).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

fn template(directory: &std::path::Path, edit: bool) -> ComfyWorkflowSpec {
    let path = directory.join(if edit { "edit.json" } else { "generation.json" });
    let mut inputs = json!({"prompt":"original","seed":0,"width":1,"height":1,"prefix":"original"});
    if edit {
        inputs["image"] = Value::String("original.png".into());
    }
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({"prompt":{
            "input":{"class_type":"FixtureInputs","inputs":inputs},
            "out":{"class_type":"FixturePng","inputs":{"image":["input",0]}}
        }}))
        .unwrap(),
    )
    .unwrap();
    let mut bindings = vec![
        ("prompt", ComfyInputValue::Prompt),
        ("seed", ComfyInputValue::Seed),
        ("width", ComfyInputValue::Width),
        ("height", ComfyInputValue::Height),
        ("prefix", ComfyInputValue::OutputPrefix),
    ];
    if edit {
        bindings.push(("image", ComfyInputValue::UploadedImage));
    }
    ComfyWorkflowSpec {
        path,
        expected_sha256: None,
        output_node: "out".into(),
        steps: ComfySteps::Fixed { count: 8 },
        required_execution_nodes: Vec::new(),
        bindings: bindings
            .into_iter()
            .map(|(input, value)| ComfyBinding {
                node: "input".into(),
                input: input.into(),
                value,
                expected_class: "FixtureInputs".into(),
            })
            .collect(),
    }
}

fn comfy_config(origin: String, directory: &std::path::Path) -> ImageBenchmarkConfig {
    let mut cfg = config(origin);
    cfg.concurrency = 1;
    cfg.repetitions = 1;
    cfg.endpoints[0].protocol = ImageEndpointProtocol::Comfy(ComfyProtocolConfig {
        generation: Some(template(directory, false)),
        edit: Some(template(directory, true)),
        limits: ComfyProtocolLimits {
            poll_interval_ms: 1,
            cleanup_timeout_ms: 200,
            ..Default::default()
        },
        graph_cache_disabled_declared: true,
        execution_audit: None,
    });
    cfg
}

mod audit_tests;

async fn server(state: &Fixture) -> (String, super::super::tests::Server) {
    let (url, server) = serve(state.router()).await;
    let mut origin = reqwest::Url::parse(&url).unwrap();
    origin.set_path("/");
    (origin.to_string(), server)
}

#[tokio::test]
async fn comfy_protocol_consumes_original_png_and_freezes_workflow_across_repetitions() {
    let state = Fixture::new(Mode::Normal);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let mut cfg = comfy_config(origin, directory.path());
    cfg.repetitions = 2;
    if let ImageEndpointProtocol::Comfy(c) = &cfg.endpoints[0].protocol {
        *state.rewrite.lock().unwrap() = Some(c.generation.as_ref().unwrap().path.clone());
    }
    let data = dataset(2);
    let report = run_benchmark(cfg, data.clone()).await.unwrap();
    assert!(report.complete);
    assert_eq!(report.endpoints[0].measured.succeeded, 4);
    assert_eq!(state.views.load(Ordering::SeqCst), 4);
    let submitted = state.submitted.lock().unwrap();
    let mut ids = std::collections::BTreeSet::new();
    for (index, request) in submitted.iter().enumerate() {
        let sample = &data.samples[index % 2];
        let inputs = &request["prompt"]["input"]["inputs"];
        assert_eq!(inputs["prompt"], sample.prompt);
        assert_eq!(inputs["seed"], sample.seed);
        assert_eq!(inputs["width"], sample.width);
        assert_eq!(inputs["height"], sample.height);
        assert!(ids.insert(request["prompt_id"].as_str().unwrap()));
    }
    assert!(report.endpoints[0]
        .protocol_requests
        .iter()
        .all(|r| r.submission_acknowledged
            && matches!(r.cleanup, ComfyCleanupOutcome::Terminal)
            && r.polls == 2));
    assert!(state.cancelled.lock().unwrap().is_empty());
}

#[tokio::test]
async fn comfy_protocol_edit_uploads_actual_shared_reference_inside_each_request() {
    let state = Fixture::new(Mode::Normal);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let mut cfg = comfy_config(origin, directory.path());
    cfg.repetitions = 2;
    let image = directory.path().join("reference.png");
    std::fs::write(&image, &state.png).unwrap();
    let mut data = dataset(1);
    data.schema_version = 2;
    data.mode = ImageRequestMode::Edit {
        image: ImageReference {
            path: image,
            encoded_sha256: Some(sha256(&state.png)),
        },
    };
    let report = run_benchmark(cfg, data).await.unwrap();
    assert_eq!(report.endpoints[0].measured.succeeded, 2);
    assert_eq!(
        *state.uploads.lock().unwrap(),
        vec![state.png.clone(), state.png.clone()]
    );
    for request in state.submitted.lock().unwrap().iter() {
        assert_eq!(
            request["prompt"]["input"]["inputs"]["image"],
            format!(
                "ferrum-bench/{}/reference.png",
                request["prompt_id"].as_str().unwrap()
            )
        );
    }
    assert!(report.endpoints[0]
        .protocol_requests
        .iter()
        .all(|r| r.uploaded_png_bytes == state.png.len()));
}

#[tokio::test]
async fn comfy_protocol_and_openai_compare_the_identical_dataset_without_rewriting_endpoints() {
    let state = Fixture::new(Mode::Normal);
    let (origin, _comfy) = server(&state).await;
    let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured = seen.clone();
    let (openai, _openai) = serve(Router::new().route(
        "/v1/images/generations",
        post(move |Json(body): Json<Value>| {
            let captured = captured.clone();
            async move {
                captured.lock().unwrap().push(body);
                valid_body()
            }
        }),
    ))
    .await;
    let directory = tempfile::tempdir().unwrap();
    let mut cfg = comfy_config(origin, directory.path());
    cfg.endpoints.push(config(openai).endpoints.remove(0));
    let data = dataset(2);
    let report = run_benchmark(cfg, data.clone()).await.unwrap();
    assert!(report.complete);
    assert!(report
        .endpoints
        .iter()
        .all(|endpoint| endpoint.measured.succeeded == 2
            && endpoint.measured.latency_slo_status == ImageSloStatus::NotEvaluated));
    let native = seen.lock().unwrap();
    let reference = state.submitted.lock().unwrap();
    for (index, sample) in data.samples.iter().enumerate() {
        let inputs = &reference[index]["prompt"]["input"]["inputs"];
        assert_eq!(native[index]["prompt"], inputs["prompt"]);
        assert_eq!(native[index]["seed"], inputs["seed"]);
        assert_eq!(
            native[index]["size"],
            format!("{}x{}", sample.width, sample.height)
        );
    }
}

#[tokio::test]
async fn comfy_protocol_rejects_foreign_outputs_multiple_images_errors_and_redirects() {
    for mode in [
        Mode::BadFilename,
        Mode::MultipleImages,
        Mode::ExecutionError,
        Mode::Redirect,
    ] {
        let state = Fixture::new(mode);
        let (origin, _server) = server(&state).await;
        let directory = tempfile::tempdir().unwrap();
        let cfg = comfy_config(origin, directory.path());
        let report = run_benchmark(cfg, dataset(1)).await.unwrap();
        assert_eq!(report.endpoints[0].measured.errors, 1);
        assert!(report.endpoints[0].requests[0].image.is_none());
        assert_eq!(
            state.views.load(Ordering::SeqCst),
            usize::from(mode == Mode::Redirect)
        );
        assert!(state.cancelled.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn comfy_protocol_deadline_cancels_only_owned_uuid_and_waits_for_dequeue() {
    let state = Fixture::new(Mode::Pending);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let mut cfg = comfy_config(origin, directory.path());
    cfg.timeout_seconds = 1;
    let report = run_benchmark(cfg, dataset(1)).await.unwrap();
    assert!(report.complete);
    assert_eq!(
        report.endpoints[0].requests[0].failure,
        Some(ImageFailureKind::Timeout)
    );
    let evidence = &report.endpoints[0].protocol_requests[0];
    assert!(matches!(
        evidence.cleanup,
        ComfyCleanupOutcome::CancelledQuiescent
    ));
    assert_eq!(
        *state.cancelled.lock().unwrap(),
        vec![evidence.prompt_id.clone()]
    );
    assert!(state.jobs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn comfy_protocol_wrong_ack_never_cancels_the_foreign_response_uuid() {
    let state = Fixture::new(Mode::WrongAcknowledgement);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let cfg = comfy_config(origin, directory.path());
    let report = run_benchmark(cfg, dataset(1)).await.unwrap();
    assert_eq!(
        report.endpoints[0].requests[0].failure,
        Some(ImageFailureKind::InvalidResponse)
    );
    let own = state.submitted.lock().unwrap()[0]["prompt_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(*state.cancelled.lock().unwrap(), vec![own]);
    assert_eq!(state.views.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn comfy_protocol_fixed_arrivals_keep_unsent_capacity_separate_from_owned_timeout() {
    let state = Fixture::new(Mode::Pending);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let mut cfg = comfy_config(origin, directory.path());
    cfg.arrival = ImageArrivalPolicy::FixedInterval { interval_ms: 1.0 };
    cfg.maximum_in_flight = Some(1);
    cfg.timeout_seconds = 1;
    let report = run_benchmark(cfg, dataset(3)).await.unwrap();
    assert_eq!(state.submitted.lock().unwrap().len(), 1);
    assert_eq!(report.endpoints[0].measured.client_not_dispatched, 2);
    assert_eq!(report.endpoints[0].measured.rejected, 0);
    assert!(report.endpoints[0].arrivals[1..]
        .iter()
        .all(|a| a.scheduled_offset_ms.is_some() && a.dispatched_offset_ms.is_none()));
}

#[tokio::test]
async fn comfy_protocol_unknown_late_submission_stops_all_new_dispatch_and_is_incomplete() {
    let state = Fixture::new(Mode::UnknownSubmission);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let mut cfg = comfy_config(origin, directory.path());
    cfg.timeout_seconds = 1;
    let report = run_benchmark(cfg, dataset(3)).await.unwrap();
    assert!(!report.complete);
    assert!(matches!(
        report.stop_reason,
        Some(BenchmarkStopReason::RemoteStateUnknown)
    ));
    assert_eq!(state.submitted.lock().unwrap().len(), 1);
    assert_eq!(report.endpoints[0].requests.len(), 3);
    assert_eq!(report.endpoints[0].measured.dispatched, 1);
    assert_eq!(report.endpoints[0].measured.client_not_dispatched, 2);
    assert!(report.endpoints[0].arrivals[1..]
        .iter()
        .all(|a| a.dispatched_offset_ms.is_none()));
    assert!(matches!(
        report.endpoints[0].protocol_requests[0].cleanup,
        ComfyCleanupOutcome::Unknown
    ));
    // The original server-side validation really can enqueue after cancel
    // returned false. No second inference POST was issued to hide this case.
    state.release_late.notify_one();
    tokio::time::timeout(Duration::from_secs(2), state.late_enqueued.notified())
        .await
        .unwrap();
    assert_eq!(state.jobs.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn comfy_protocol_preflight_rejects_binding_steps_and_physical_byte_overflow_before_http() {
    let state = Fixture::new(Mode::Normal);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let cfg = comfy_config(origin, directory.path());
    let mut wrong = cfg.clone();
    if let ImageEndpointProtocol::Comfy(c) = &mut wrong.endpoints[0].protocol {
        c.generation.as_mut().unwrap().bindings[0].expected_class = "OtherClass".into();
    }
    assert!(run_benchmark(wrong, dataset(1)).await.is_err());
    let mut data = dataset(1);
    data.samples[0].num_inference_steps = Some(9);
    assert!(run_benchmark(cfg.clone(), data).await.is_err());
    let mut small = cfg.clone();
    if let ImageEndpointProtocol::Comfy(c) = &mut small.endpoints[0].protocol {
        c.limits.maximum_workflow_bytes = std::fs::metadata(&c.generation.as_ref().unwrap().path)
            .unwrap()
            .len() as usize
            - 1;
    }
    assert!(run_benchmark(small, dataset(1)).await.is_err());
    assert!(state.submitted.lock().unwrap().is_empty());
}

#[tokio::test]
async fn comfy_protocol_chunked_metadata_capacity_is_not_silently_accepted() {
    let state = Fixture::new(Mode::OversizedMetadata);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let mut cfg = comfy_config(origin, directory.path());
    if let ImageEndpointProtocol::Comfy(c) = &mut cfg.endpoints[0].protocol {
        c.limits.maximum_metadata_response_bytes = 128;
    }
    let report = run_benchmark(cfg, dataset(1)).await.unwrap();
    assert_eq!(
        report.endpoints[0].requests[0].failure,
        Some(ImageFailureKind::ResponseLimit)
    );
    assert_eq!(state.views.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn comfy_protocol_upload_budget_and_cache_declaration_are_checked_before_dispatch() {
    let state = Fixture::new(Mode::Normal);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let cfg = comfy_config(origin, directory.path());
    let mut unsafe_cache = cfg.clone();
    if let ImageEndpointProtocol::Comfy(c) = &mut unsafe_cache.endpoints[0].protocol {
        c.graph_cache_disabled_declared = false;
    }
    assert!(run_benchmark(unsafe_cache, dataset(1)).await.is_err());
    let image = directory.path().join("shared.png");
    std::fs::write(&image, &state.png).unwrap();
    let mut edits = dataset(2);
    edits.schema_version = 2;
    edits.mode = ImageRequestMode::Edit {
        image: ImageReference {
            path: image,
            encoded_sha256: None,
        },
    };
    let mut denied = cfg;
    if let ImageEndpointProtocol::Comfy(c) = &mut denied.endpoints[0].protocol {
        c.limits.maximum_uploaded_bytes = (2 * state.png.len() - 1) as u64;
    }
    assert!(run_benchmark(denied, edits).await.is_err());
    assert!(state.submitted.lock().unwrap().is_empty());
    assert!(state.uploads.lock().unwrap().is_empty());
}

#[test]
fn comfy_protocol_wire_configuration_loads_relative_immutable_workflows_and_rejects_foreign_hash() {
    let directory = tempfile::tempdir().unwrap();
    let mut cfg = comfy_config("http://127.0.0.1:1/".into(), directory.path());
    let path = directory.path().join("protocol.json");
    if let ImageEndpointProtocol::Comfy(c) = &mut cfg.endpoints[0].protocol {
        c.generation.as_mut().unwrap().path = PathBuf::from("generation.json");
        c.edit.as_mut().unwrap().path = PathBuf::from("edit.json");
    }
    std::fs::write(
        &path,
        serde_json::to_vec(&cfg.endpoints[0].protocol).unwrap(),
    )
    .unwrap();
    let loaded = ImageEndpointProtocol::from_path(&path, 4096).unwrap();
    let ImageEndpointProtocol::Comfy(mut config) = loaded else {
        panic!("wrong protocol")
    };
    let generation = config.generation.as_mut().unwrap();
    assert!(PreparedWorkflow::load(generation, false, &config.limits).is_ok());
    generation.expected_sha256 = Some("0".repeat(64));
    assert!(PreparedWorkflow::load(generation, false, &config.limits).is_err());
}

#[test]
fn comfy_protocol_owned_png_keeps_the_original_allocation_and_shared_validation_contract() {
    let state = Fixture::new(Mode::Normal);
    let bytes = state.png.clone();
    let ptr = bytes.as_ptr();
    let result = ferrum_bench_core::images::validate_png_response(
        bytes,
        1,
        1,
        &ImageResponseLimits::default(),
    )
    .unwrap();
    let (identity, bytes) = result.into_parts();
    assert_eq!(bytes.as_ptr(), ptr);
    assert_eq!(identity.png_sha256, sha256(&state.png));
    assert!(ferrum_bench_core::images::validate_png_response(
        bytes.clone(),
        2,
        1,
        &ImageResponseLimits::default()
    )
    .is_err());
    let mut corrupt = bytes;
    corrupt[20] ^= 1;
    assert!(ferrum_bench_core::images::validate_png_response(
        corrupt,
        1,
        1,
        &ImageResponseLimits::default()
    )
    .is_err());
}

#[test]
fn comfy_protocol_cli_exposes_gpu_query_deadline_without_relaxing_sampling_bounds() {
    use clap::Parser;
    let args = [
        "bench-images",
        "--endpoint",
        "http://127.0.0.1:1/v1/images/generations",
        "--model",
        "fixture",
        "--server-config",
        "fixture",
        "--dataset",
        "input.json",
        "--output",
        "output.json",
    ];
    assert_eq!(
        TestCli::try_parse_from(args)
            .unwrap()
            .command
            .resource_gpu_query_timeout_ms,
        2000
    );
    for value in ["100", "10000"] {
        let mut argv = args.to_vec();
        argv.extend(["--resource-gpu-query-timeout-ms", value]);
        assert_eq!(
            TestCli::try_parse_from(argv)
                .unwrap()
                .command
                .resource_gpu_query_timeout_ms,
            value.parse::<u64>().unwrap()
        );
    }
    for value in ["99", "10001"] {
        let mut argv = args.to_vec();
        argv.extend(["--resource-gpu-query-timeout-ms", value]);
        assert!(TestCli::try_parse_from(argv).is_err());
    }
}
