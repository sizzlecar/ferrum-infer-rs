use super::*;
use ferrum_interfaces::{
    output_credit::{OutputAccountLimits, OutputCreditAmount, OutputCreditPool, OutputPoolLimits},
    output_flow::{
        BoundedChatProjection, CreditedOutputFrame, CreditedOutputSession, OutputCompletion,
        OutputConsumerControl, OutputFrameAttempt, OutputFrameMetadata, OutputHistory,
        OutputProjectionContract, OutputTerminal, RequestOutputBudget, RequestOutputPlan,
    },
};
use ferrum_tokenizer::implementations::huggingface::HuggingFaceTokenizer;
use ferrum_types::{SloMode, SloOutputTransport};
use std::num::NonZeroUsize;

const TEXT: &str = "quote\"\\\n中";

#[derive(Default)]
struct ConsumerClosed(AtomicBool);

impl OutputConsumerControl for ConsumerClosed {
    fn consumer_dropped(&self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Uses the actual output budget and codec, with immediate token production.
/// This isolates routing/HTTP ownership from the engine's separately tested
/// wave scheduler. No StreamChunk-to-credit adapter is used by this fixture.
struct CreditedRouteLlm {
    base: StubLlm,
    text: String,
    last_request: Mutex<Option<InferenceRequest>>,
    tokenizer: HuggingFaceTokenizer,
    pool: OutputCreditPool,
    control: Arc<ConsumerClosed>,
    credited_calls: AtomicUsize,
    legacy_calls: AtomicUsize,
    first_wire_pointer: AtomicUsize,
    last_context: Mutex<Option<InferenceRequestContext>>,
    startup_failure: Option<Error>,
}

impl CreditedRouteLlm {
    async fn new(transport: SloOutputTransport) -> Self {
        Self::with_text(transport, TEXT).await
    }

    async fn with_text(transport: SloOutputTransport, text: &str) -> Self {
        let mut base = StubLlm::new("legacy");
        base.config.model.model_id = ModelId::new("wire-model");
        base.config.scheduler.slo.output.transport = transport;
        let vocabulary: tokenizers::models::bpe::Vocab = [
            (text.to_owned(), 0),
            ("a".to_owned(), 1),
            ("</think>".to_owned(), 2),
            ("</s>".to_owned(), 3),
        ]
        .into_iter()
        .collect();
        let mut hf = tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocabulary, Vec::new())
                .build()
                .unwrap(),
        );
        hf.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        hf.add_special_tokens(&[tokenizers::AddedToken::from("</s>", true)]);
        let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
        let limits = OutputPoolLimits::from_slo(
            &base.config.scheduler.slo.output,
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap();
        Self {
            base,
            text: text.to_owned(),
            last_request: Mutex::new(None),
            tokenizer,
            pool: OutputCreditPool::new(limits).unwrap(),
            control: Arc::new(ConsumerClosed::default()),
            credited_calls: AtomicUsize::new(0),
            legacy_calls: AtomicUsize::new(0),
            first_wire_pointer: AtomicUsize::new(0),
            last_context: Mutex::new(None),
            startup_failure: None,
        }
    }

    fn assert_released(&self) {
        let snapshot = self.pool.snapshot();
        assert_eq!(snapshot.retained_accounts, 0);
        assert_eq!(snapshot.data_used, OutputCreditAmount::ZERO);
        assert_eq!(snapshot.terminal_held, OutputCreditAmount::ZERO);
    }
}

#[async_trait]
impl InferenceEngine for CreditedRouteLlm {
    async fn status(&self) -> EngineStatus {
        self.base.status().await
    }
    async fn shutdown(&self) -> ferrum_types::Result<()> {
        self.base.shutdown().await
    }
    fn config(&self) -> &EngineConfig {
        self.base.config()
    }
    fn metrics(&self) -> EngineMetrics {
        self.base.metrics()
    }
    async fn health_check(&self) -> EngineHealthStatus {
        self.base.health_check().await
    }
}

#[async_trait]
impl LlmInferenceEngine for CreditedRouteLlm {
    async fn infer(&self, request: InferenceRequest) -> ferrum_types::Result<InferenceResponse> {
        self.legacy_calls.fetch_add(1, Ordering::Relaxed);
        self.base.infer(request).await
    }
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ferrum_types::Result<Pin<Box<dyn Stream<Item = ferrum_types::Result<StreamChunk>> + Send>>>
    {
        self.legacy_calls.fetch_add(1, Ordering::Relaxed);
        self.base.infer_stream(request).await
    }
    async fn infer_with_context(
        &self,
        request: InferenceRequest,
        context: InferenceRequestContext,
    ) -> ferrum_types::Result<InferenceResponse> {
        *self.last_context.lock().unwrap() = Some(context);
        self.infer(request).await
    }
    async fn infer_stream_with_context(
        &self,
        request: InferenceRequest,
        context: InferenceRequestContext,
    ) -> ferrum_types::Result<Pin<Box<dyn Stream<Item = ferrum_types::Result<StreamChunk>> + Send>>>
    {
        *self.last_context.lock().unwrap() = Some(context);
        self.infer_stream(request).await
    }
    async fn infer_credited_stream(
        &self,
        request: InferenceRequest,
        context: InferenceRequestContext,
        contract: Arc<OutputProjectionContract>,
    ) -> ferrum_types::Result<CreditedOutputSession> {
        self.credited_calls.fetch_add(1, Ordering::Relaxed);
        *self.last_context.lock().unwrap() = Some(context);
        if let Some(error) = &self.startup_failure {
            return Err(error.clone());
        }
        *self.last_request.lock().unwrap() = Some(request.clone());
        let plan = RequestOutputPlan::derive(contract, &self.tokenizer, &request, 3).unwrap();
        let limits = OutputAccountLimits::from_slo(
            &self.base.config.scheduler.slo.output,
            NonZeroUsize::new(plan.terminal_credit().events).unwrap(),
        )
        .unwrap();
        let is_chat = plan.text_reasoning_policy().is_some();
        let mut budget = RequestOutputBudget::open(&self.pool, limits, plan).unwrap();
        let mut projection = is_chat.then(|| BoundedChatProjection::new(budget.plan()).unwrap());
        let mut data = Vec::new();
        if let Some(projection) = projection.as_mut() {
            projection.prepare(&self.text, true).unwrap();
            while let Some(delta) = projection.pending_delta() {
                let permit = match budget.try_begin_frame().unwrap() {
                    OutputFrameAttempt::Reserved(permit) => permit,
                    OutputFrameAttempt::Full(_) => {
                        panic!("fixture must admit its bounded two-channel output")
                    }
                };
                data.push(budget.encode_chat_data_frame(permit, delta, 17).unwrap());
                projection.advance().unwrap();
            }
        } else {
            let permit = match budget.try_begin_frame().unwrap() {
                OutputFrameAttempt::Reserved(permit) => permit,
                OutputFrameAttempt::Full(_) => panic!("fixture must admit its first frame"),
            };
            data.push(budget.encode_data_frame(permit, &self.text, 17).unwrap());
        }
        if let Some(first) = data.first() {
            self.first_wire_pointer
                .store(first.payload().as_ptr() as usize, Ordering::Relaxed);
        }
        drop(projection);
        let usage = TokenUsage::new(3, 1);
        let terminal = budget
            .encode_terminal(OutputTerminal::Success {
                reason: FinishReason::Length,
                usage: &usage,
                created: 17,
            })
            .unwrap();
        let completion = budget
            .into_retained_projection(OutputCompletion::Succeeded {
                execution_evidence: (request.evidence_request.capture_engine_token_timing
                    || request.evidence_request.capture_prompt_token_ids)
                    .then(|| InferenceExecutionEvidence {
                        prompt_token_ids: if request.evidence_request.capture_prompt_token_ids {
                            vec![TokenId::new(0); 3]
                        } else {
                            vec![]
                        },
                        output_token_ids: vec![TokenId::new(0)],
                        engine_token_timing: request
                            .evidence_request
                            .capture_engine_token_timing
                            .then(|| EngineTokenTimingEvidence {
                                clock_source: "rust_std_instant".into(),
                                wall_anchor_unix_nanos: 1,
                                wall_anchor_max_error_nanos: 0,
                                decode_ready_nanos_since_request_start: None,
                                token_commit_nanos_since_request_start: vec![1000],
                                decode_stage_intervals: vec![],
                                decode_stage_intervals_omitted: 0,
                            }),
                    }),
                history: Some(OutputHistory {
                    text: self.text.clone(),
                    tokens: vec![TokenId::new(0)],
                }),
                reason: FinishReason::Length,
                usage,
            })
            .unwrap();
        let (sender, frames) = mpsc::channel(3);
        for (index, (wire, terminal)) in data
            .into_iter()
            .map(|wire| (wire, false))
            .chain(std::iter::once((terminal, true)))
            .enumerate()
        {
            assert!(sender
                .try_send(CreditedOutputFrame::new(
                    wire,
                    OutputFrameMetadata {
                        ordinal: 1,
                        token: (!terminal && index == 0).then_some(TokenId::new(0)),
                        generated_tokens: 1,
                        terminal,
                    },
                ))
                .is_ok());
        }
        drop(sender);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        assert!(sender.send(completion).is_ok());
        Ok(CreditedOutputSession::from_receivers(
            frames,
            receiver,
            self.control.clone(),
        ))
    }
}

fn completion_request() -> Value {
    json!({"model": "wire-model", "prompt": "hello", "max_tokens": 2, "stream": true})
}

#[tokio::test]
async fn credited_completion_route_uses_codec_once_and_preserves_usage_contract() {
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let started = Instant::now();
    let response = post_json(
        AxumServer::from_llm(engine.clone()).build_router(),
        "/v1/completions",
        completion_request(),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");
    assert_eq!(engine.legacy_calls.load(Ordering::Relaxed), 0);
    assert_eq!(engine.credited_calls.load(Ordering::Relaxed), 1);
    assert!(
        engine
            .last_context
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .ingress()
            >= started
    );
    assert_eq!(engine.pool.snapshot().data_used.projection_bytes, 0);
    let body = response_text(response).await;
    let events: Vec<Value> = body
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|event| *event != "[DONE]")
        .map(|event| serde_json::from_str(event).unwrap())
        .collect();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0]["choices"][0]["text"], TEXT);
    assert_eq!(events[1]["choices"][0]["finish_reason"], "length");
    assert_eq!(events[2]["choices"], json!([]));
    assert_eq!(events[2]["usage"]["completion_tokens"], 1);
    assert_eq!(events[2]["usage"]["prompt_tokens"], 3);
    assert_eq!(events[2]["usage"]["total_tokens"], 4);
    for event in &events {
        assert_eq!(event["model"], "wire-model");
        assert_eq!(event["created"], 17);
        assert_eq!(event["id"], events[0]["id"]);
    }
    assert!(body.ends_with("data: [DONE]\n\n"));
    assert!(!engine.control.0.load(Ordering::Acquire));
    engine.assert_released();
}

#[tokio::test]
async fn credited_completion_body_keeps_returned_bytes_until_last_transport_owner() {
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let response = post_json(
        AxumServer::from_llm(engine.clone()).build_router(),
        "/v1/completions",
        completion_request(),
    )
    .await;
    let mut body = response.into_body().into_data_stream();
    let bytes = body.next().await.unwrap().unwrap();
    assert_eq!(
        bytes.as_ptr() as usize,
        engine.first_wire_pointer.load(Ordering::Relaxed),
        "HTTP must transfer the codec allocation without copying"
    );
    let length = bytes.len();
    let alias = bytes.clone().slice(1..);
    drop(bytes);
    drop(body);
    assert!(engine.control.0.load(Ordering::Acquire));
    assert_eq!(engine.pool.snapshot().data_used.bytes, length);
    assert_eq!(engine.pool.snapshot().data_used.events, 1);
    assert_eq!(engine.pool.snapshot().retained_accounts, 1);
    drop(alias);
    engine.assert_released();
}

#[tokio::test]
async fn credited_completion_unpolled_body_drop_cancels_and_releases_queued_frames() {
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let response = post_json(
        AxumServer::from_llm(engine.clone()).build_router(),
        "/v1/completions",
        completion_request(),
    )
    .await;
    assert_eq!(engine.pool.snapshot().retained_accounts, 1);
    assert!(!engine.control.0.load(Ordering::Acquire));
    drop(response);
    assert!(engine.control.0.load(Ordering::Acquire));
    engine.assert_released();
}

#[tokio::test]
async fn credited_unsupported_routes_reject_before_either_inference_entrypoint() {
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    for (path, payload) in [
        (
            "/v1/completions",
            json!({"model":"wire-model", "prompt":"hello", "stream":false}),
        ),
        (
            "/v1/chat/completions",
            json!({"model":"wire-model", "messages":[{"role":"user", "content":"hello"}], "stream":true, "response_format":{"type":"json_object"}}),
        ),
        (
            "/v1/chat/completions",
            json!({"model":"wire-model", "messages":[{"role":"user", "content":"hello"}], "stream":false}),
        ),
        (
            "/v1/responses",
            json!({"model":"wire-model", "input":"hello", "stream":true}),
        ),
        (
            "/v1/responses",
            json!({"model":"wire-model", "input":"hello", "stream":false}),
        ),
    ] {
        let response = post_json(
            AxumServer::from_llm(engine.clone()).build_router(),
            path,
            payload,
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST, "{path}");
        assert!(response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("application/json"));
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("credited output does not yet support"));
    }
    assert_eq!(engine.credited_calls.load(Ordering::Relaxed), 0);
    assert_eq!(engine.legacy_calls.load(Ordering::Relaxed), 0);
    engine.assert_released();
}

#[tokio::test]
async fn credited_startup_rejections_remain_json_before_sse_headers() {
    for (failure, status) in [
        (
            Error::unsupported("unproved decoder"),
            AxumStatusCode::BAD_REQUEST,
        ),
        (
            Error::resource_exhausted("output budget full"),
            AxumStatusCode::SERVICE_UNAVAILABLE,
        ),
    ] {
        let mut engine = CreditedRouteLlm::new(SloOutputTransport::Credited).await;
        engine.startup_failure = Some(failure);
        let engine = Arc::new(engine);
        let response = post_json(
            AxumServer::from_llm(engine.clone()).build_router(),
            "/v1/completions",
            completion_request(),
        )
        .await;
        assert_eq!(response.status(), status);
        assert!(response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("application/json"));
        assert!(response_json(response).await["error"].is_object());
        assert_eq!(engine.credited_calls.load(Ordering::Relaxed), 1);
        assert_eq!(engine.legacy_calls.load(Ordering::Relaxed), 0);
        engine.assert_released();
    }
}

#[tokio::test]
async fn credited_selection_leaves_default_off_and_observe_legacy_routes_unchanged() {
    for mode in [SloMode::Off, SloMode::Observe] {
        let mut engine = CreditedRouteLlm::new(SloOutputTransport::Legacy).await;
        engine.base.config.scheduler.slo.mode = mode;
        let engine = Arc::new(engine);
        let response = post_json(
            AxumServer::from_llm(engine.clone()).build_router(),
            "/v1/completions",
            completion_request(),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("legacy"));
        assert!(body.contains("[DONE]"));
        assert_eq!(engine.legacy_calls.load(Ordering::Relaxed), 1);
        assert_eq!(engine.credited_calls.load(Ordering::Relaxed), 0);
        assert!(engine.last_context.lock().unwrap().is_some());
        engine.assert_released();
    }
}

fn ordinary_chat_request(usage: Option<bool>) -> Value {
    let mut request = json!({"model":"wire-model", "messages":[{"role":"user", "content":"hello"}], "max_tokens":4, "stream":true});
    if let Some(usage) = usage {
        request["stream_options"] = json!({"include_usage":usage});
    }
    request
}

#[tokio::test]
async fn credited_chat_route_uses_rendered_template_state_and_opt_in_usage() {
    let text = "reason</think>\nanswer \"中\"";
    for include_usage in [None, Some(false), Some(true)] {
        let engine =
            Arc::new(CreditedRouteLlm::with_text(SloOutputTransport::Credited, text).await);
        let response = post_json(
            AxumServer::from_llm(engine.clone())
                .with_prompt_template(Some(prompt_opened_literal_json_template()))
                .build_router(),
            "/v1/chat/completions",
            ordinary_chat_request(include_usage),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let request = engine.last_request.lock().unwrap().clone().unwrap();
        assert!(request.prompt.ends_with("<think>"));
        assert_eq!(request.metadata[PROMPT_OPENED_REASONING_METADATA_KEY], true);
        assert_eq!(
            request.sampling_params.response_completion_boundary,
            ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
                delimiter: "</think>".into(),
                alternate_envelope: None,
            }
        );
        assert_eq!(request.sampling_params.max_tokens, 4);
        let body = response_text(response).await;
        let events: Vec<Value> = body
            .split("\n\n")
            .filter_map(|event| event.strip_prefix("data: "))
            .filter(|event| *event != "[DONE]")
            .map(|event| serde_json::from_str(event).unwrap())
            .collect();
        assert_eq!(events[0]["choices"][0]["delta"]["reasoning"], "reason");
        assert_eq!(events[0]["choices"][0]["delta"]["content"], "");
        assert_eq!(events[1]["choices"][0]["delta"]["content"], "answer \"中\"");
        assert_eq!(events[2]["choices"][0]["finish_reason"], "length");
        assert!(events[2]["choices"][0].get("delta").is_some());
        assert_eq!(events.len(), 3 + usize::from(include_usage == Some(true)));
        if include_usage == Some(true) {
            assert_eq!(events[3]["usage"]["completion_tokens"], 1);
        }
        for event in &events {
            assert_eq!(event["id"], request.id.to_string());
            assert_eq!(event["model"], "wire-model");
            assert_eq!(event["created"], 17);
        }
        assert!(body.ends_with("data: [DONE]\n\n"));
        assert_eq!(engine.credited_calls.load(Ordering::Relaxed), 1);
        assert_eq!(engine.legacy_calls.load(Ordering::Relaxed), 0);
        engine.assert_released();
    }
}

#[tokio::test]
async fn credited_chat_body_uses_same_allocation_and_holds_last_bytes_owner() {
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let response = post_json(
        AxumServer::from_llm(engine.clone()).build_router(),
        "/v1/chat/completions",
        ordinary_chat_request(None),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let bytes = body.next().await.unwrap().unwrap();
    assert_eq!(
        bytes.as_ptr() as usize,
        engine.first_wire_pointer.load(Ordering::Relaxed)
    );
    let held = bytes.slice(..);
    drop(bytes);
    drop(body);
    assert!(engine.control.0.load(Ordering::Acquire));
    assert_eq!(engine.pool.snapshot().retained_accounts, 1);
    assert_eq!(engine.pool.snapshot().data_used.events, 1);
    drop(held);
    engine.assert_released();
}

#[tokio::test]
async fn credited_chat_startup_error_stays_json_and_legacy_selection_is_unchanged() {
    let mut failed = CreditedRouteLlm::new(SloOutputTransport::Credited).await;
    failed.startup_failure = Some(Error::resource_exhausted("output budget full"));
    let failed = Arc::new(failed);
    let response = post_json(
        AxumServer::from_llm(failed.clone()).build_router(),
        "/v1/chat/completions",
        ordinary_chat_request(None),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::SERVICE_UNAVAILABLE);
    assert!(response_json(response).await["error"].is_object());
    assert_eq!(failed.credited_calls.load(Ordering::Relaxed), 1);
    failed.assert_released();
    for mode in [SloMode::Off, SloMode::Observe] {
        let mut engine = CreditedRouteLlm::new(SloOutputTransport::Legacy).await;
        engine.base.config.scheduler.slo.mode = mode;
        let engine = Arc::new(engine);
        let response = post_json(
            AxumServer::from_llm(engine.clone()).build_router(),
            "/v1/chat/completions",
            ordinary_chat_request(None),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        assert!(response_text(response).await.contains("legacy"));
        assert_eq!(engine.legacy_calls.load(Ordering::Relaxed), 1);
        assert_eq!(engine.credited_calls.load(Ordering::Relaxed), 0);
        engine.assert_released();
    }
}

#[tokio::test]
async fn credited_text_latency_and_kernel_profiles_consume_terminal_evidence_and_release_credit() {
    for detail in [
        ferrum_types::ObservabilityProfileDetail::Latency,
        ferrum_types::ObservabilityProfileDetail::Kernel,
    ] {
        for chat in [false, true] {
            let dump = unique_request_dump_dir("credited-terminal-evidence");
            let profile = unique_profile_jsonl("credited-terminal-evidence");
            let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
            let router = AxumServer::from_state(
                AppState::default()
                    .with_llm(engine.clone())
                    .with_prompt_template(Some(prompt_opened_literal_json_template()))
                    .with_profile_detail(detail)
                    .with_profile_jsonl(Some(profile.clone()))
                    .with_request_dump_dir(Some(dump.clone())),
            )
            .build_router();
            let endpoint = if chat {
                "/v1/chat/completions"
            } else {
                "/v1/completions"
            };
            let response = post_json(
                router,
                endpoint,
                if chat {
                    ordinary_chat_request(None)
                } else {
                    completion_request()
                },
            )
            .await;
            assert_eq!(response.status(), AxumStatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .unwrap();
            assert!(String::from_utf8_lossy(&bytes).contains("[DONE]"));
            let mut changed = engine.pool.subscribe();
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while engine.pool.snapshot().retained_accounts != 0 {
                    changed.changed().await.unwrap();
                }
            })
            .await
            .unwrap();
            engine.assert_released();
            let request = engine.last_request.lock().unwrap().clone().unwrap();
            assert!(request.evidence_request.capture_engine_token_timing);
            assert!(request.evidence_request.capture_prompt_token_ids);
            let lines = std::fs::read_to_string(&profile).unwrap();
            let events: Vec<Value> = lines
                .lines()
                .map(|s| serde_json::from_str(s).unwrap())
                .collect();
            let event = events
                .iter()
                .find(|v| v["phase"] == "credited_generation")
                .unwrap();
            assert_eq!(event["status"], "ok");
            assert_eq!(event["attributes"]["endpoint"], endpoint);
            assert_eq!(event["attributes"]["engine_token_commit_count"], 1);
            assert_eq!(
                event["attributes"]["engine_token_commit_nanos_since_request_start"],
                json!([1000])
            );
            let prompt: Value = serde_json::from_str(
                &std::fs::read_to_string(
                    dump.join(request.id.to_string())
                        .join("prompt_token_ids.json"),
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(prompt["token_count"], 3);
            assert_eq!(engine.legacy_calls.load(Ordering::Relaxed), 0);
            std::fs::remove_dir_all(dump).unwrap();
            std::fs::remove_file(profile).unwrap();
        }
    }
}

mod shutdown;
