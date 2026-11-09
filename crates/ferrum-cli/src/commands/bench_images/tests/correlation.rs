//! Real local HTTP/PNG protocol tests. UUID fields are response fixtures,
//! not fabricated model execution, trace-clock or retirement evidence.
use super::*;
use axum::{body::Bytes, http::HeaderMap};
use ferrum_types::image_trace::{
    IMAGE_ATTEMPT_ID_HEADER, IMAGE_REQUEST_ID_HEADER, IMAGE_TRACE_SESSION_HEADER,
};
use futures::StreamExt;
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

#[derive(Clone)]
struct Correlated {
    seen: Arc<Mutex<BTreeMap<u64, (Uuid, Uuid)>>>,
    trace: Uuid,
    body: String,
}

fn sent_attempt(headers: &HeaderMap) -> Uuid {
    Uuid::parse_str(headers[IMAGE_ATTEMPT_ID_HEADER].to_str().unwrap()).unwrap()
}

fn add_headers(mut response: Response, attempt: Uuid, request: Uuid, trace: Uuid) -> Response {
    for (name, value) in [
        (IMAGE_ATTEMPT_ID_HEADER, attempt),
        (IMAGE_REQUEST_ID_HEADER, request),
        (IMAGE_TRACE_SESSION_HEADER, trace),
    ] {
        response
            .headers_mut()
            .insert(name, value.to_string().parse().unwrap());
    }
    response
}

async fn responses(
    State(state): State<Correlated>,
    headers: HeaderMap,
    Json(value): Json<serde_json::Value>,
) -> Response {
    let seed = value["seed"].as_u64().unwrap();
    let attempt = sent_attempt(&headers);
    let request = Uuid::new_v4();
    state.seen.lock().unwrap().insert(seed, (attempt, request));
    if seed == 3 {
        return std::future::pending().await; // No HTTP headers delivered.
    }
    let response = match seed {
        0 => (StatusCode::OK, state.body).into_response(),
        1 => (StatusCode::TOO_MANY_REQUESTS, "bounded fixture rejection").into_response(),
        2 => {
            let stream =
                futures::stream::once(async { Ok::<_, std::io::Error>(Bytes::from_static(b"{")) })
                    .chain(futures::stream::pending());
            Response::new(Body::from_stream(stream))
        }
        _ => unreachable!(),
    };
    add_headers(response, attempt, request, state.trace)
}

#[tokio::test]
async fn image_correlation_bench_keeps_attempts_and_response_ids_on_rejection_and_timeouts() {
    let state = Correlated {
        seen: Arc::new(Mutex::new(BTreeMap::new())),
        trace: Uuid::new_v4(),
        body: valid_body(),
    };
    let (url, _server) = serve(
        Router::new()
            .route("/v1/images/generations", post(responses))
            .with_state(state.clone()),
    )
    .await;
    let mut config = config(url);
    config.concurrency = 4;
    config.repetitions = 1;
    config.timeout_seconds = 1;
    let mut samples = dataset(4);
    for (seed, sample) in samples.samples.iter_mut().enumerate() {
        sample.seed = seed as u64;
    }
    let report = run_benchmark(config, samples).await.unwrap();
    let records = &report.endpoints[0].requests;
    let attempts = records
        .iter()
        .map(|x| x.client_attempt_id.unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(attempts.len(), records.len());
    let seen = state.seen.lock().unwrap();
    for record in records {
        let seed = record.sample_index as u64;
        assert_eq!(record.client_attempt_id, Some(seen[&seed].0));
        if seed == 3 {
            assert_eq!(record.http_status, None);
            assert_eq!(record.server_request_id, None);
            assert_eq!(record.trace_session_uuid, None);
        } else {
            assert_eq!(record.server_request_id, Some(seen[&seed].1));
            assert_eq!(record.trace_session_uuid, Some(state.trace));
        }
        match seed {
            0 => assert!(record.succeeded()),
            1 => assert_eq!(record.failure, Some(ImageFailureKind::HttpStatus)),
            2 | 3 => assert_eq!(record.failure, Some(ImageFailureKind::Timeout)),
            _ => unreachable!(),
        }
    }
}

#[tokio::test]
async fn image_correlation_bench_cancel_retains_dispatched_attempt_and_leaves_unsent_without_one() {
    #[derive(Clone)]
    struct CancelState {
        started: Arc<tokio::sync::Notify>,
        attempt: Arc<Mutex<Option<Uuid>>>,
    }
    async fn waiting(State(state): State<CancelState>, headers: HeaderMap) -> Response {
        *state.attempt.lock().unwrap() = Some(sent_attempt(&headers));
        state.started.notify_one();
        std::future::pending().await
    }
    let state = CancelState {
        started: Arc::new(tokio::sync::Notify::new()),
        attempt: Arc::new(Mutex::new(None)),
    };
    let (url, _server) = serve(
        Router::new()
            .route("/v1/images/generations", post(waiting))
            .with_state(state.clone()),
    )
    .await;
    let mut config = config(url);
    config.concurrency = 1;
    config.repetitions = 1;
    let control = super::super::protocol::RunControl::new();
    let active_control = control.clone();
    let task = tokio::spawn(async move {
        super::super::runner::run_benchmark_with_control(config, dataset(2), active_control).await
    });
    tokio::time::timeout(Duration::from_secs(10), state.started.notified())
        .await
        .unwrap();
    control.stop(super::super::protocol::BenchmarkStopReason::Cancelled);
    let report = tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let records = &report.endpoints[0].requests;
    let sent = records.iter().find(|r| r.sample_index == 0).unwrap();
    assert_eq!(sent.client_attempt_id, *state.attempt.lock().unwrap());
    assert!(sent.client_attempt_id.is_some());
    assert_eq!(sent.server_request_id, None);
    assert_eq!(sent.failure, Some(ImageFailureKind::ClientTask));
    let unsent = records.iter().find(|r| r.sample_index == 1).unwrap();
    assert_eq!(unsent.client_attempt_id, None);
    assert_eq!(unsent.failure, Some(ImageFailureKind::ClientStopped));
}

#[tokio::test]
async fn image_correlation_bench_rejects_malformed_duplicate_and_foreign_response_metadata() {
    async fn malformed(headers: HeaderMap, Json(value): Json<serde_json::Value>) -> Response {
        let seed = value["seed"].as_u64().unwrap();
        let mut response = add_headers(
            valid_body().into_response(),
            sent_attempt(&headers),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        match seed {
            0 => {
                response.headers_mut().insert(
                    IMAGE_TRACE_SESSION_HEADER,
                    "private-response-metadata-not-a-uuid".parse().unwrap(),
                );
            }
            1 => {
                response.headers_mut().append(
                    IMAGE_REQUEST_ID_HEADER,
                    Uuid::new_v4().to_string().parse().unwrap(),
                );
            }
            2 => {
                response.headers_mut().insert(
                    IMAGE_ATTEMPT_ID_HEADER,
                    Uuid::new_v4().to_string().parse().unwrap(),
                );
            }
            _ => unreachable!(),
        }
        response
    }
    let (url, _server) =
        serve(Router::new().route("/v1/images/generations", post(malformed))).await;
    let mut config = config(url);
    config.repetitions = 1;
    let mut samples = dataset(3);
    for (seed, sample) in samples.samples.iter_mut().enumerate() {
        sample.seed = seed as u64;
    }
    let report = run_benchmark(config, samples).await.unwrap();
    for record in &report.endpoints[0].requests {
        assert!(record.client_attempt_id.is_some());
        assert_eq!(record.failure, Some(ImageFailureKind::InvalidResponse));
        assert!(record.image.is_none());
        assert!(!record
            .error
            .as_ref()
            .unwrap()
            .contains("private-response-metadata"));
    }
}

#[tokio::test]
async fn image_correlation_legacy_openai_response_works_and_every_repetition_has_fresh_attempts() {
    async fn legacy() -> String {
        valid_body()
    }
    let (url, _server) = serve(Router::new().route("/v1/images/generations", post(legacy))).await;
    let mut config = config(url);
    config.warmup_repetitions = 1;
    config.repetitions = 2;
    let report = run_benchmark(config, dataset(2)).await.unwrap();
    let records = &report.endpoints[0].requests;
    assert!(records.iter().all(ImageRequestRecord::succeeded));
    assert!(records
        .iter()
        .all(|r| r.server_request_id.is_none() && r.trace_session_uuid.is_none()));
    assert_eq!(
        records
            .iter()
            .map(|r| r.client_attempt_id.unwrap())
            .collect::<BTreeSet<_>>()
            .len(),
        records.len()
    );
}

#[tokio::test]
async fn image_correlation_mixed_generation_and_edit_share_dispatch_header_contract() {
    use axum::extract::Multipart;
    use ferrum_bench_core::images::{validate_image_response_with_png, ImageReference};
    #[derive(Clone)]
    struct Mixed {
        requests: Arc<Mutex<Vec<(bool, Uuid)>>>,
        reference_sha: String,
        trace: Uuid,
    }
    async fn generated(
        State(state): State<Mixed>,
        headers: HeaderMap,
        Json(_): Json<serde_json::Value>,
    ) -> Response {
        let attempt = sent_attempt(&headers);
        state.requests.lock().unwrap().push((false, attempt));
        add_headers(
            valid_body().into_response(),
            attempt,
            Uuid::new_v4(),
            state.trace,
        )
    }
    async fn edited(
        State(state): State<Mixed>,
        headers: HeaderMap,
        mut form: Multipart,
    ) -> Response {
        let attempt = sent_attempt(&headers);
        let mut image = None;
        while let Some(field) = form.next_field().await.unwrap() {
            if field.name() == Some("image") {
                image = Some(sha256(&field.bytes().await.unwrap()));
            } else {
                let _ = field.bytes().await.unwrap();
            }
        }
        assert_eq!(image.as_deref(), Some(state.reference_sha.as_str()));
        state.requests.lock().unwrap().push((true, attempt));
        add_headers(
            valid_body().into_response(),
            attempt,
            Uuid::new_v4(),
            state.trace,
        )
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reference.png");
    let png = validate_image_response_with_png(
        valid_body().as_bytes(),
        1,
        1,
        &ImageResponseLimits::default(),
    )
    .unwrap()
    .into_parts()
    .1;
    std::fs::write(&path, &png).unwrap();
    let state = Mixed {
        requests: Arc::new(Mutex::new(Vec::new())),
        reference_sha: sha256(&png),
        trace: Uuid::new_v4(),
    };
    let (url, _server) = serve(
        Router::new()
            .route("/v1/images/generations", post(generated))
            .route("/v1/images/edits", post(edited))
            .with_state(state.clone()),
    )
    .await;
    let mut config = config(url.clone());
    config.repetitions = 1;
    config.endpoints[0].edit_url = Some(format!("{}/edits", url.rsplit_once('/').unwrap().0));
    let mut samples = dataset(2);
    samples.schema_version = 3;
    samples.samples[1].mode = Some(ImageRequestMode::Edit {
        image: ImageReference {
            path,
            encoded_sha256: Some(state.reference_sha.clone()),
        },
    });
    let report = run_benchmark(config, samples).await.unwrap();
    for record in &report.endpoints[0].requests {
        assert!(record.succeeded());
        assert!(record.server_request_id.is_some());
        assert_eq!(record.trace_session_uuid, Some(state.trace));
        assert!(state
            .requests
            .lock()
            .unwrap()
            .contains(&(record.sample_index == 1, record.client_attempt_id.unwrap())));
    }
}
