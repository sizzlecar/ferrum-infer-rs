//! Handler-boundary timing through actual HTTP conversion and projection. The
//! capturing engine checks transport of the clock, not model numerical timing.
use super::*;
use ferrum_bench_core::{JsonlJournalOpenMode, JsonlJournalOwner};
use ferrum_types::ObservabilityProfileDetail;
use std::time::{Duration, Instant};

struct ReleaseConstruction(Option<std::sync::mpsc::Sender<()>>);
impl ReleaseConstruction {
    fn release(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}
impl Drop for ReleaseConstruction {
    fn drop(&mut self) {
        self.release();
    }
}

async fn handler_clock_case(endpoint: &'static str, streaming: bool) {
    let path = unique_profile_jsonl("handler-clock");
    let mut journal =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Truncate, Default::default()).unwrap();
    let engine = Arc::new(CapturingLlm::new());
    let mut state = AppState::default()
        .with_llm(engine.clone())
        .with_profile_jsonl(Some(path.clone()))
        .with_profile_detail(ObservabilityProfileDetail::Basic)
        .with_prompt_template(Some(ModelChatTemplate::new(
            "{% for message in messages %}{{ message.role }}:{{ message.content }}\n{% endfor %}assistant:",
            "qwen3",
        )));
    let (entered, entry) = tokio::sync::oneshot::channel();
    let entered = Mutex::new(Some(entered));
    let (resume, resumed) = std::sync::mpsc::channel();
    let resumed = Mutex::new(resumed);
    let mut release = ReleaseConstruction(Some(resume));
    let observations = Arc::new(Mutex::new(Vec::new()));
    let observe = observations.clone();
    state.request_construction_observer = Some(Arc::new(move |anchor, phase| {
        let construction_start = Instant::now();
        observe
            .lock()
            .unwrap()
            .push((anchor, phase, construction_start));
        let sender = entered.lock().unwrap().take();
        if let Some(sender) = sender {
            let _ = sender.send((anchor, phase, construction_start));
            resumed
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
    }));
    let server = AxumServer::from_state(state).with_journal_scopes(
        HttpJournalScopes::new(vec![journal.producer_scope().unwrap()]).unwrap(),
    );
    let router = server.build_router().unwrap();
    let request = match endpoint {
        "/v1/chat/completions" => json!({
            "model":"qwen3", "messages":[{"role":"user","content":"clock prompt"}],
            "max_completion_tokens":1, "stream":streaming
        }),
        "/v1/responses" => json!({
            "model":"qwen3", "instructions":"Keep the original instruction.",
            "input":[{"role":"user","content":[{"type":"input_text","text":"clock prompt"}]}],
            "max_output_tokens":1, "stream":streaming
        }),
        "/v1/completions" => json!({
            "model":"qwen3", "prompt":"clock prompt", "max_tokens":1, "stream":streaming
        }),
        _ => unreachable!(),
    };
    // The synchronous conversion gate blocks a separate runtime worker. The
    // controlling task can time out and release it even on a broken handler.
    let request_task = tokio::spawn(async move { post_json(router, endpoint, request).await });
    let (anchor, phase, construction_start) = tokio::time::timeout(Duration::from_secs(5), entry)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(anchor.boundary(), RequestIngressBoundary::HttpHandler);
    assert!(anchor.instant() <= construction_start);
    assert_eq!(
        phase,
        match endpoint {
            "/v1/chat/completions" => "chat_session_and_prompt",
            "/v1/responses" => "responses_conversion",
            "/v1/completions" => "completion_prompt",
            _ => unreachable!(),
        }
    );
    assert!(
        !engine.has_captured_request(),
        "actual request conversion has not run past its gate"
    );
    let release_at = Instant::now();
    let construction_wait_us = anchor.elapsed_at(release_at).as_micros() as u64;
    release.release();
    let response = tokio::time::timeout(Duration::from_secs(5), request_task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), AxumStatusCode::OK);
    let body = response_text(response).await;
    assert!(body.contains("captured"), "{endpoint}: {body}");
    if streaming && endpoint != "/v1/responses" {
        assert!(body.contains("data: [DONE]"));
    } else if streaming {
        assert!(body.contains("response.completed"));
    }
    let request = engine.last_request();
    assert_eq!(
        request.ingress,
        Some(anchor),
        "conversion must not recapture the clock"
    );
    assert_eq!(request.stream, streaming);
    assert_eq!(request.sampling_params.max_tokens, 1);
    assert!(request.prompt.contains("clock prompt"));
    if endpoint == "/v1/responses" {
        assert!(request.prompt.contains("Keep the original instruction."));
        assert_eq!(
            observations.lock().unwrap().len(),
            2,
            "Responses conversion and Chat preparation both ran"
        );
    }
    for &(observed, _, started) in observations.lock().unwrap().iter() {
        assert_eq!(observed, anchor);
        assert!(observed.instant() <= started);
    }
    server.stop(Duration::from_secs(2)).await.unwrap();
    let final_receipt = journal.close().unwrap();
    final_receipt.result().unwrap();
    assert!(final_receipt.worker_joined);
    let events = read_profile_events(&path);
    if endpoint == "/v1/completions" {
        assert!(
            events.is_empty(),
            "clock propagation does not add a new profile pipeline"
        );
    } else {
        let expected_phase = if streaming {
            "chat_completions_stream_complete"
        } else {
            "chat_completions_sync_complete"
        };
        let event = events
            .iter()
            .find(|event| event["phase"] == expected_phase)
            .unwrap();
        assert_eq!(
            event["attributes"]["request_start_boundary"],
            "http_handler"
        );
        assert!(event["attributes"]["request_start_excludes"]
            .as_str()
            .unwrap()
            .contains("JSON extraction"));
        assert!(event["duration_us"].as_u64().unwrap() >= construction_wait_us);
        if streaming {
            assert!(
                event["attributes"]["engine_stream_first_chunk_received_us"]
                    .as_u64()
                    .unwrap()
                    >= construction_wait_us
            );
        }
        let memory = events
            .iter()
            .find(|event| event["phase"] == "actual_serve_first_request_done")
            .unwrap();
        assert_eq!(
            memory["attributes"]["memory_before_boundary"],
            "before_engine_call_after_request_preparation"
        );
    }
    let _ = fs::remove_file(path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_handler_clock_covers_chat_and_responses_construction_and_profile() {
    for endpoint in ["/v1/chat/completions", "/v1/responses"] {
        for streaming in [false, true] {
            handler_clock_case(endpoint, streaming).await;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_handler_clock_reaches_sync_and_stream_completion_requests() {
    for streaming in [false, true] {
        handler_clock_case("/v1/completions", streaming).await;
    }
}
