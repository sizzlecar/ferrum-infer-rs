use super::*;
use ferrum_models::test_support::{
    tiny_llama_model_with_execution_config, tiny_model_info, tiny_tokenizer, TinyLlamaConfig,
};
use futures::StreamExt;

struct Recorder {
    records: Mutex<Vec<RequestStoppingObservation>>,
    max_rule_bytes: usize,
    dropped: AtomicU64,
    panic_on_record: bool,
}

impl Default for Recorder {
    fn default() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
            max_rule_bytes: 4096,
            dropped: AtomicU64::new(0),
            panic_on_record: false,
        }
    }
}

impl RequestStoppingObserver for Recorder {
    fn limits(&self) -> RequestObservationLimits {
        RequestObservationLimits {
            max_rule_bytes: self.max_rule_bytes,
        }
    }
    fn try_record(&self, observation: &RequestStoppingObservation) -> RequestObservationDelivery {
        assert!(!self.panic_on_record, "injected request observer panic");
        assert!(observation.is_consistent(), "{observation:?}");
        let Some(mut records) = self.records.try_lock() else {
            return RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull);
        };
        if records.len() >= 128 {
            return RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull);
        }
        records.push(observation.clone());
        RequestObservationDelivery::Recorded
    }
    fn record_drop(&self, reason: RequestObservationDrop) {
        assert_eq!(reason, RequestObservationDrop::ProjectionLimit);
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

fn request() -> InferenceRequest {
    let mut request = InferenceRequest::new("hello world", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    request
}

async fn engine(observer: Arc<Recorder>) -> Box<dyn LlmInferenceEngine> {
    engine_with_tokenizer(observer, tiny_tokenizer(&TinyLlamaConfig::default())).await
}

async fn engine_with_tokenizer(
    observer: Arc<Recorder>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
) -> Box<dyn LlmInferenceEngine> {
    let cfg = TinyLlamaConfig::default();
    let model = tiny_llama_model_with_execution_config(
        &cfg,
        ferrum_types::LlamaExecutionConfig {
            prefix_cache: true,
            ..Default::default()
        },
    );
    crate::EngineBuilder::new(EngineConfig::default())
        .with_custom_tokenizer(tokenizer)
        .with_custom_sampler(Arc::new(crate::registry::GreedySampler))
        .with_owned_executor(ExecutorOwner::new(Box::new(
            ferrum_models::LlmExecutor::new(Box::new(model), tiny_model_info(&cfg)),
        )))
        .unwrap()
        .with_request_stopping_observer(observer)
        .build()
        .await
        .unwrap()
}

#[tokio::test]
async fn stopping_observer_preserves_verified_eos_and_user_stop_before_wire_projection() {
    let baseline = engine(Arc::new(Recorder::default())).await;
    let output = baseline.infer(request()).await.unwrap();
    let first = output.response().tokens[0].get();
    baseline.shutdown().await.unwrap();
    baseline.wait_for_shutdown_completion().await.unwrap();
    for reason in [FinishReason::EOS, FinishReason::Stop] {
        let cfg = TinyLlamaConfig::default();
        let tokenizer = if reason == FinishReason::EOS {
            ferrum_models::test_support::TinyTokenizer::new(cfg.vocab_size).with_eos(first)
        } else {
            ferrum_models::test_support::TinyTokenizer::new(cfg.vocab_size)
                .with_composite_token(first, "beforeHALTafter")
        };
        let recorder = Arc::new(Recorder::default());
        let engine = engine_with_tokenizer(recorder.clone(), Arc::new(tokenizer)).await;
        let mut input = request();
        if reason == FinishReason::EOS {
            input.metadata.remove("ferrum_ignore_eos");
        } else {
            input.sampling_params.stop_sequences = vec!["HALT".into()];
        }
        let output = engine.infer(input).await.unwrap();
        assert_eq!(output.response().finish_reason, reason);
        engine.shutdown().await.unwrap();
        engine.wait_for_shutdown_completion().await.unwrap();
        let records = recorder.records.lock();
        let end = records.last().unwrap();
        assert_eq!(end.terminal, Some(terminal_reason(reason)));
        assert_eq!(end.completion, Some(RequestCompletionOutcome::Succeeded));
        assert_eq!(end.generated_tokens, 1);
        assert_eq!(end.verified_survival_through, 0);
        assert_eq!(end.rules.mechanism(), StoppingMechanism::Observed);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn abandoned_stream_records_cancellation_without_claiming_a_natural_end() {
    let recorder = Arc::new(Recorder::default());
    let engine = engine(recorder.clone()).await;
    let stream = engine.infer_stream(request()).await.unwrap();
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if recorder
                .records
                .lock()
                .iter()
                .any(|record| record.terminal == Some(RequestTerminalReason::Cancelled))
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("abandoned request must emit its direct cancellation fact");
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    let records = recorder.records.lock();
    assert_eq!(
        records.last().unwrap().completion,
        Some(RequestCompletionOutcome::Failed)
    );
    assert!(!records.iter().any(|record| matches!(
        record.terminal,
        Some(
            RequestTerminalReason::Eos
                | RequestTerminalReason::Stop
                | RequestTerminalReason::Length
        )
    )));
}

struct FailAfterOutputTokenizer(ferrum_models::test_support::TinyTokenizer, Arc<AtomicU64>);
impl Tokenizer for FailAfterOutputTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        self.0.encode(text, add_special)
    }
    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        if skip_special && tokens.len() >= 3 {
            self.1.fetch_add(1, Ordering::Relaxed);
            return Err(FerrumError::tokenizer(
                "injected post-output decode failure",
            ));
        }
        self.0.decode(tokens, skip_special)
    }
    fn decode_incremental(&self, prev: &[TokenId], next: TokenId) -> Result<String> {
        self.0.decode_incremental(prev, next)
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        self.0.special_tokens()
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.0.token_id(text)
    }
    fn token_text(&self, token: TokenId) -> Option<&str> {
        self.0.token_text(token)
    }
    fn token_bytes(&self, token: TokenId) -> Option<Vec<u8>> {
        self.0.token_bytes(token)
    }
    fn info(&self) -> ferrum_interfaces::tokenizer::TokenizerInfo {
        self.0.info()
    }
}

fn assert_injected_output_failure(error: &FerrumError) {
    // Token validation may reject every candidate before final text decoding.
    assert!(
        matches!(error,
        FerrumError::Tokenizer { message } if message == "injected post-output decode failure")
            || matches!(error, FerrumError::Backend { message } if message == "No finite logits available for sampling"),
        "{error:?}"
    );
}

#[tokio::test]
async fn post_output_error_keeps_real_generated_count_and_verified_prefix_for_both_flows() {
    for streaming in [false, true] {
        let recorder = Arc::new(Recorder::default());
        let decode_failures = Arc::new(AtomicU64::new(0));
        let tokenizer = FailAfterOutputTokenizer(
            ferrum_models::test_support::TinyTokenizer::new(TinyLlamaConfig::default().vocab_size),
            decode_failures.clone(),
        );
        let engine = engine_with_tokenizer(recorder.clone(), Arc::new(tokenizer)).await;
        if streaming {
            let mut stream = engine.infer_stream(request()).await.unwrap();
            let mut failed = false;
            while let Some(chunk) = stream.next().await {
                if let Err(error) = chunk {
                    assert_injected_output_failure(&error);
                    failed = true;
                }
            }
            assert!(failed);
        } else {
            let error = engine.infer(request()).await.unwrap_err();
            assert_injected_output_failure(&error);
        }
        // The injected failure is also reported by the output-control owner;
        // shutdown still joins the original worker and releases its resources.
        if let Err(error) = engine.shutdown().await {
            assert_injected_output_failure(&error);
        }
        if let Err(error) = engine.wait_for_shutdown_completion().await {
            assert_injected_output_failure(&error);
        }
        assert!(decode_failures.load(Ordering::Relaxed) > 0);
        let records = recorder.records.lock();
        let end = records.last().unwrap();
        assert_eq!(end.terminal, Some(RequestTerminalReason::Error));
        assert_eq!(end.completion, Some(RequestCompletionOutcome::Failed));
        let last_verified = records
            .iter()
            .filter(|record| record.terminal.is_none())
            .map(|record| record.verified_survival_through)
            .max()
            .unwrap();
        assert!(last_verified >= 2);
        assert!(
            end.generated_tokens >= last_verified,
            "completion cannot use the emptied token vector after payload transfer"
        );
        assert!(
            end.verified_survival_through >= last_verified,
            "a failure cannot truncate already verified natural survival"
        );
        assert!(end.is_consistent());
    }
}

#[tokio::test]
async fn stopping_feedback_runs_through_stream_and_nonstream_with_unique_product_lifetimes() {
    let recorder = Arc::new(Recorder::default());
    let engine = engine(recorder.clone()).await;
    let request = request();
    let output = engine.infer(request.clone()).await.unwrap();
    assert_eq!(output.response().finish_reason, FinishReason::Length);
    let mut stream = engine.infer_stream(request).await.unwrap();
    let mut terminal = None;
    while let Some(chunk) = stream.next().await {
        terminal = chunk.unwrap().chunk().finish_reason.or(terminal);
    }
    assert_eq!(terminal, Some(FinishReason::Length));
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    let records = recorder.records.lock();
    let admissions: Vec<_> = records
        .iter()
        .filter(|record| record.revision == 1)
        .collect();
    assert_eq!(admissions.len(), 2);
    assert_eq!(
        admissions[0].lifecycle.producer,
        admissions[1].lifecycle.producer
    );
    assert_ne!(
        admissions[0].lifecycle, admissions[1].lifecycle,
        "reused public RequestId must not reuse lifetime"
    );
    for admission in admissions {
        let lifetime: Vec<_> = records
            .iter()
            .filter(|record| record.lifecycle == admission.lifecycle)
            .collect();
        assert!(lifetime
            .windows(2)
            .all(|pair| pair[0].revision < pair[1].revision
                && pair[0].verified_survival_through <= pair[1].verified_survival_through));
        let end = lifetime.last().unwrap();
        assert_eq!(end.terminal, Some(RequestTerminalReason::Length));
        assert_eq!(end.completion, Some(RequestCompletionOutcome::Succeeded));
        assert!(lifetime
            .iter()
            .all(|fact| fact.admitted_at == admission.admitted_at));
        assert_eq!(end.generated_tokens, 4);
        assert_eq!(end.verified_survival_through, 4);
        assert_eq!(end.rules.mechanism(), StoppingMechanism::CapOnly);
    }
}

#[tokio::test]
async fn generation_endpoint_does_not_claim_success_before_actual_close_and_output_delivery() {
    use crate::continuous_engine::token_completion_tests as actual;
    for (close_failure, closed_stream) in [(false, false), (true, false), (false, true)] {
        let recorder = Arc::new(Recorder::default());
        let mut fixture = actual::fixture_with_feedback(
            true,
            close_failure,
            |vocab| Arc::new(ferrum_models::test_support::TinyTokenizer::new(vocab)),
            Some(recorder.clone()),
        )
        .await;
        let inner = fixture.engine.active_inner().unwrap();
        let handle = inner.sequences.get(&fixture.request_id).unwrap();
        let reason = inner
            .stop_reason_for_sequence(&handle)
            .expect("real first-token stopping boundary");
        assert!(matches!(
            reason,
            FinishReason::EOS | FinishReason::Stop | FinishReason::Length
        ));
        if closed_stream {
            drop(fixture.stream);
        }
        let mut completion = Box::pin(inner.complete_request_for_test(&fixture.request_id, reason));
        tokio::select! {
            result = completion.as_mut() => panic!("completion escaped actual close gate: {result:?}"),
            started = &mut fixture.started => started.unwrap(),
        }
        {
            let records = recorder.records.lock();
            let endpoint = records.last().unwrap();
            assert_eq!(endpoint.terminal, Some(terminal_reason(reason)));
            assert_eq!(
                endpoint.completion, None,
                "an endpoint waiting on physical close is not product success"
            );
        }
        fixture.gate.take().unwrap().send(()).unwrap();
        let result = completion.await;
        if close_failure {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("actual close completed but publication failed"));
        } else {
            result.unwrap();
        }
        let records = recorder.records.lock();
        let end = records.last().unwrap();
        assert_eq!(
            end.terminal,
            Some(terminal_reason(reason)),
            "delivery failure does not erase true stopping evidence"
        );
        assert_eq!(
            end.completion,
            Some(if close_failure || closed_stream {
                RequestCompletionOutcome::Failed
            } else {
                RequestCompletionOutcome::Succeeded
            })
        );
        drop(records);
        drop(handle);
        drop(inner);
        let shutdown = tokio::time::timeout(Duration::from_secs(5), fixture.engine.shutdown())
            .await
            .unwrap();
        if let Err(error) = shutdown {
            assert!(
                close_failure
                    && error
                        .to_string()
                        .contains("actual close completed but publication failed"),
                "{error:?}"
            );
        }
    }
}

#[tokio::test]
async fn oversized_identity_and_panicking_observer_cannot_change_output_or_fail_requests() {
    let baseline = engine(Arc::new(Recorder::default())).await;
    let expected = baseline
        .infer(request())
        .await
        .unwrap()
        .response()
        .tokens
        .clone();
    baseline.shutdown().await.unwrap();
    baseline.wait_for_shutdown_completion().await.unwrap();
    for recorder in [
        Recorder {
            max_rule_bytes: 1,
            ..Default::default()
        },
        Recorder {
            panic_on_record: true,
            ..Default::default()
        },
    ] {
        let recorder = Arc::new(recorder);
        let engine = engine(recorder.clone()).await;
        let actual = engine.infer(request()).await.unwrap();
        assert_eq!(actual.response().tokens, expected);
        if recorder.max_rule_bytes == 1 {
            assert_eq!(recorder.dropped.load(Ordering::Relaxed), 1);
        }
        engine.shutdown().await.unwrap();
        engine.wait_for_shutdown_completion().await.unwrap();
    }
}

#[test]
fn complete_effective_rule_identity_separates_stops_grammar_activation_and_response_boundaries() {
    let mut sequence = SequenceState::new(request(), vec![TokenId::new(1)]);
    let plain = effective_rules(&sequence, 4096).unwrap();
    sequence.stop_text_seqs = vec!["STOP".into()];
    let user_stop = effective_rules(&sequence, 4096).unwrap();
    assert_ne!(plain, user_stop);
    assert_eq!(user_stop.mechanism(), StoppingMechanism::Observed);
    sequence.stop_text_seqs.clear();
    sequence.sampling_params.structured_output_start =
        ferrum_types::StructuredOutputStart::AfterDelimiter("</think>".into());
    let activation = effective_rules(&sequence, 4096).unwrap();
    assert_ne!(plain, activation);
    sequence.sampling_params.structured_output_start = Default::default();
    sequence.sampling_params.response_completion_boundary =
        ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "</think>".into(),
            alternate_envelope: None,
        };
    let boundary = effective_rules(&sequence, 4096).unwrap();
    assert_ne!(plain, boundary);
    assert_eq!(boundary.mechanism(), StoppingMechanism::Observed);
    sequence.sampling_params.response_completion_boundary = Default::default();
    sequence.sampling_params.response_format =
        ferrum_types::ResponseFormat::JsonSchema("{}".into());
    assert_ne!(plain, effective_rules(&sequence, 4096).unwrap());
}

#[test]
fn bounded_serialization_rejects_long_rule_without_truncating_domain_identity() {
    let mut sequence = SequenceState::new(request(), vec![TokenId::new(1)]);
    sequence.stop_text_seqs = vec!["x".repeat(4097)];
    assert!(effective_rules(&sequence, 4096).is_none());
    sequence.stop_text_seqs = vec!["x".repeat(32)];
    assert!(effective_rules(&sequence, 4096).is_some());
}

#[test]
fn kv_resource_release_does_not_replace_product_lifecycle_or_verified_prefix() {
    let mut sequence = SequenceState::new(request(), vec![TokenId::new(1)]);
    let fact = RequestStoppingObservation {
        lifecycle: RequestLifecycleId {
            producer: 7,
            admission: 9,
        },
        admitted_at: sequence.start_time,
        revision: 3,
        observed_at: Instant::now(),
        rules: effective_rules(&sequence, 4096).unwrap(),
        cap: 4,
        generated_tokens: 2,
        verified_survival_through: 2,
        terminal: None,
        completion: None,
    };
    sequence.stopping_observation = Some(fact.clone());
    sequence.generated_tokens = vec![TokenId::new(2), TokenId::new(3)];
    let _resources = sequence.take_completion_resources();
    sequence.prefill_complete = false;
    sequence.prefill_tokens_processed = 0;
    assert_eq!(sequence.stopping_observation, Some(fact));
}

#[tokio::test]
async fn planning_recaptures_after_admission_cleanup_and_invalidates_failed_projection() {
    use ferrum_scheduler::cost_prediction::{
        CostExecution, CostPrediction, CostQuery, CostSnapshot, CostUnknownReason,
    };

    struct Snapshot;
    impl CostSnapshot for Snapshot {
        fn revision(&self) -> u64 {
            73
        }
        fn query_limits(&self) -> ferrum_types::CostQueryLimits {
            Default::default()
        }
        fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
            CostPrediction::Unknown(CostUnknownReason::NoCoverage)
        }
    }

    let cfg = TinyLlamaConfig::default();
    let config = EngineConfig::default();
    let engine = ContinuousBatchEngine::new_observed_with_requests(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tiny_tokenizer(&cfg),
        Arc::new(crate::registry::GreedySampler),
        ExecutorOwner::new(Box::new(ferrum_models::LlmExecutor::new(
            Box::new(tiny_llama_model_with_execution_config(
                &cfg,
                Default::default(),
            )),
            tiny_model_info(&cfg),
        ))),
        None,
        None,
        None,
        Some(Arc::new(Recorder::default())),
    )
    .unwrap();
    let mut ids = Vec::new();
    let mut responses = Vec::new();
    for _ in 0..2 {
        let mut request = request();
        request.ensure_ingress(ferrum_types::RequestIngressBoundary::EngineApi);
        let mut sequence = SequenceState::new(request.clone(), vec![TokenId::new(1)]);
        let (sender, response) = tokio::sync::oneshot::channel();
        sequence.response_sender = Some(sender);
        responses.push(Some(response));
        ids.push(request.id.clone());
        engine.publish_request(request, sequence).await.unwrap();
    }
    let inner = engine.active_inner().unwrap();
    let mut round = TimePlanningRound::new(
        Arc::new(Snapshot),
        "ttft:1000,tpot:100,itl:100".parse().unwrap(),
        ferrum_execution::model_executor::PrefillShapePolicy::Identity,
        CostExecution::SingleInvocation,
    );
    inner
        .prepare_dynamic_admission_round(0, Some(&mut round))
        .await
        .unwrap();
    assert_eq!(round.request_facts().unwrap().len(), 2);
    assert_eq!(round.request_clocks().unwrap().len(), 2);

    // The real scheduler failure queue and engine terminal path remove one
    // lifetime during a subround. A subsequent subround must not reuse it.
    inner.scheduler.fail_waiting_admission(
        &ids[0],
        FerrumError::request_validation("injected admission rejection"),
    );
    inner
        .prepare_dynamic_admission_round(0, Some(&mut round))
        .await
        .unwrap();
    assert!(inner.sequences.get(&ids[0]).is_none());
    assert!(matches!(
        responses[0].take().unwrap().await.unwrap(),
        Err(FerrumError::RequestValidation { message }) if message == "injected admission rejection"
    ));
    inner
        .prepare_dynamic_admission_round(0, Some(&mut round))
        .await
        .unwrap();
    let facts = round.request_facts().unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].0, ids[1]);
    assert_eq!(round.request_clocks().unwrap()[0].0, ids[1]);
    assert_eq!(
        round.revision(),
        73,
        "cohort refresh must not reload the model"
    );

    let handle = inner.sequences.get(&ids[1]).unwrap();
    handle.lock().as_mut().unwrap().stopping_observation = None;
    inner.capture_planning_request_facts(&mut round, 2);
    assert!(round.request_facts().is_none());
    assert!(round.request_clocks().is_none());
    assert_eq!(round.revision(), 73);
    drop(handle);
    drop(inner);
    // The output-control owner also reports the injected request failure;
    // this must not be mistaken for a new cleanup or planning error.
    let error = engine.shutdown().await.unwrap_err();
    assert!(matches!(error, FerrumError::RequestValidation { message }
        if message == "injected admission rejection"));
    if let Err(error) = engine.wait_for_shutdown_completion().await {
        assert!(matches!(error, FerrumError::RequestValidation { message }
            if message == "injected admission rejection"));
    }
}
