use super::*;
use futures::StreamExt;

// Preserve the existing controlled token stream, including its repeated first
// decode token. Only Engine plumbing changes: opaque admission/frontiers and
// explicit closure now flow through the common token protocol. Fake logits
// remain protocol fixtures, not model numerical evidence.
struct StopBoundaryExecutor {
    inner: ferrum_testkit::ConfigurableModelExecutor,
    finish_signal: Mutex<Option<tokio::sync::oneshot::Sender<ExecutionEndReason>>>,
}

impl StopBoundaryExecutor {
    fn with_token_sequence(vocabulary: usize, tokens: Vec<u32>) -> Self {
        Self {
            inner: ferrum_testkit::ConfigurableModelExecutor::with_token_sequence(
                vocabulary, tokens,
            ),
            finish_signal: Mutex::new(None),
        }
    }
}

#[async_trait::async_trait]
impl ModelExecutor for StopBoundaryExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        input.validate()?;
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }
    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        assert!(input.adapter.is_none());
        self.inner.admit_tokens(input).await
    }
    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        self.inner.execute_tokens(work).await
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        let outcome = self.inner.finish_tokens(sequence, reason).await?;
        if let Some(signal) = self.finish_signal.lock().take() {
            assert!(outcome.error.is_none());
            assert!(matches!(
                &outcome.disposition,
                ferrum_execution::token_execution::ExecutionCloseDisposition::Ended(_)
            ));
            let _ = signal.send(reason);
        }
        Ok(outcome)
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.inner.cancel_tokens(sequence)
    }
}

fn stop_boundary_engine(pieces: &[&str], end_with_eos: bool) -> ContinuousBatchEngine {
    stop_boundary_engine_with_options(pieces, end_with_eos, StopBoundaryOptions::default())
}

#[derive(Default)]
struct StopBoundaryOptions {
    output_queue_items: Option<usize>,
    finish_signal: Option<tokio::sync::oneshot::Sender<ExecutionEndReason>>,
}

fn stop_boundary_engine_with_options(
    pieces: &[&str],
    end_with_eos: bool,
    options: StopBoundaryOptions,
) -> ContinuousBatchEngine {
    let mut pairs = vec![("test", 5), ("", 6)];
    pairs.extend(
        pieces
            .iter()
            .enumerate()
            .map(|(i, text)| (*text, i as u32 + 7)),
    );
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(128, &pairs));
    // The reusable executor repeats its first token at prefill and first decode.
    // An empty decoded piece leaves the visible sequence exactly as specified.
    let mut tokens = (6..7 + pieces.len() as u32).collect::<Vec<_>>();
    if end_with_eos {
        tokens.push(3); // PolicyTokenizer's ordinary EOS has no decoded text.
    }
    let mut config = EngineConfig::default();
    if let Some(items) = options.output_queue_items {
        config.output.engine_queue_items = items;
    }
    let executor = StopBoundaryExecutor::with_token_sequence(128, tokens);
    *executor.finish_signal.lock() = options.finish_signal;
    ContinuousBatchEngine::new(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        Arc::new(MockKvCacheManager::new(256)),
        Arc::new(executor),
        Arc::new(MockTensorFactory),
    )
    .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn output_stop_prefix_flush_preserves_two_pending_spans_before_terminal() {
    let (finish, finished) = tokio::sync::oneshot::channel();
    let engine = stop_boundary_engine_with_options(
        &["first", "answerab"],
        false,
        StopBoundaryOptions {
            output_queue_items: Some(1),
            finish_signal: Some(finish),
        },
    );
    let mut request = policy_request();
    request.sampling_params.stop_sequences = vec!["abc".into()];
    request.sampling_params.max_tokens = 4;
    request.evidence_request.capture_prompt_token_ids = true;
    let request_id = request.id.clone();
    let stream = engine.infer_stream(request).await.unwrap();

    // No client poll has happened. Tokens 6,6 are empty, 7 fills the queue,
    // and 8 creates "answer" plus the unmatched terminal "ab" flush. On
    // this single-thread runtime the synchronous completion handoff registers
    // both spans before the separately owned finish task can send this signal.
    let reason = tokio::time::timeout(Duration::from_secs(2), finished)
        .await
        .expect("physical finish must not wait for the client to drain output")
        .unwrap();
    assert_eq!(
        reason,
        ExecutionEndReason::Completed {
            input_tokens: 1,
            output_tokens: 4
        }
    );
    assert!(engine.inner.sequences.is_empty());

    let chunks = tokio::time::timeout(Duration::from_secs(2), stream.collect::<Vec<_>>())
        .await
        .expect("both pending spans and the terminal must drain in order")
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk().text.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "answer", "ab", ""]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk().text.as_str())
            .collect::<String>(),
        "firstanswerab"
    );
    assert!(chunks
        .iter()
        .all(|chunk| chunk.chunk().request_id == request_id));
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk().token.map(TokenId::get))
            .collect::<Vec<_>>(),
        vec![Some(7), Some(8), None, None]
    );
    assert!(chunks[..3]
        .iter()
        .all(|chunk| chunk.chunk().finish_reason.is_none() && chunk.chunk().usage.is_none()));
    let terminal = chunks.last().unwrap().chunk();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, 1);
    assert_eq!(usage.completion_tokens, 4);
    let evidence = terminal.execution_evidence.as_ref().unwrap();
    assert_eq!(evidence.prompt_token_ids, vec![TokenId::new(5)]);
    assert_eq!(
        evidence.output_token_ids,
        vec![
            TokenId::new(6),
            TokenId::new(6),
            TokenId::new(7),
            TokenId::new(8)
        ]
    );
    engine.shutdown().await.unwrap();
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    drop(chunks);
    assert_eq!(engine.inner.output_budget.usage().sessions, 0);
    assert_eq!(engine.inner.output_budget.usage().queued_bytes, 0);
}

async fn harmony_pending_terminal_preserves_published_range(marker: &str) {
    use ferrum_execution::token_execution::{ExecutionProduct, TokenWork};

    let mut tokenizer = PolicyTokenizer::new(128, &[("test", 5), (marker, 7)]);
    tokenizer.special.extra_eos_tokens = vec![TokenId::new(7)];
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let executor = Arc::new(StopBoundaryExecutor::with_token_sequence(128, vec![7]));
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        Arc::new(crate::registry::GreedySampler),
        Arc::new(MockKvCacheManager::new(256)),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    let mut request = policy_request();
    request.stream = true;
    request.sampling_params.model_output_protocol =
        ferrum_types::ModelOutputProtocol::HarmonyGptOss;
    let request_id = request.id.clone();
    let prompt = tokenizer.encode(&request.prompt, true).unwrap();
    scheduler.submit_request(request.clone()).unwrap();
    let scheduled = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    assert_eq!(scheduled.requests.len(), 1);

    // Use the existing controlled executor's actual token admission and
    // sampling product. This is a protocol/publication regression, not a
    // numerical model or native-failure fixture.
    let TokenAdmissionOutcome::Admitted(start) = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_product_request(
                &request_id,
                &prompt,
                executor.info().max_sequence_length,
                prompt.len(),
                0,
            )
            .unwrap(),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("controlled token admission must succeed")
    };
    let policy = ferrum_execution::model_executor::LogitsReturnPolicy::FullLogits;
    let work = ExecutionWork::new(
        &start,
        TokenWork::Prefill(PrefillChunk::new(0, prompt.len(), prompt.len()).unwrap()),
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(completed) =
        executor.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("controlled prefill must complete")
    };
    let mut completed = completed.into_completions();
    assert_eq!(completed.len(), 1);
    let (_, frontier, _, product) = completed.pop().unwrap().into_parts();
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(mut logits)) = product else {
        panic!("controlled prefill must preserve its sampling product")
    };
    let identity = frontier.sequence().clone();
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        prompt.clone(),
        Some(tokenizer.clone()),
        Some(executor.info().vocab_size),
    );
    let token = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();
    assert_eq!(token, TokenId::new(7));
    assert_eq!(
        state.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::EOS)
    );
    assert!(state.should_stream_generated_token(
        Some(tokenizer.as_ref()),
        token,
        Some(FinishReason::EOS),
    ));
    state.execution_frontier = Some(frontier);
    state.prefill_complete = true;
    state.prefill_tokens_processed = prompt.len();
    scheduler.mark_prefill_complete(&request_id, prompt.len());
    let (response_sender, mut response) = tokio::sync::oneshot::channel();
    let (sender, mut receiver) = engine.inner.output_channel(request_id.clone(), 1).unwrap();
    state.response_sender = Some(response_sender);
    state.stream_sender = Some(sender.clone());
    let mut slot = RequestSlotLease::open(&engine.inner, request_id.clone());
    slot.admit(&engine.inner);
    state.request_slot = Some(slot);
    assert!(engine
        .inner
        .sequences
        .insert(request_id.clone(), state)
        .is_none());
    let handle = engine.inner.sequences.get(&request_id).unwrap();
    sender.bind_sequence(Arc::downgrade(&handle));

    // Hold every real queued-byte permit before publication. A full mpsc
    // channel alone is insufficient: its pending span may already be charged.
    let blocker = engine.inner.output_budget.try_open_session().unwrap();
    let held = blocker
        .reserve_queued(blocker.config().queued_bytes_total)
        .await
        .unwrap();
    engine.inner.send_stream_update(&request_id, token).await;
    assert_eq!(
        handle.lock().as_ref().unwrap().streamed_text_len,
        marker.len()
    );
    assert_eq!(
        engine.inner.output_budget.usage().queued_bytes,
        held.bytes()
    );
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(sender.failure().is_none());
    assert!(
        scheduler
            .next_batch(ferrum_interfaces::BatchHint::simple(1))
            .await
            .is_none(),
        "the published, uncharged span must actually pause this request"
    );

    let original = FerrumError::request_validation("explicit failure after Harmony publication");
    engine
        .inner
        .complete_request_exact_frontier_detached(
            &handle,
            &identity,
            FinishReason::EOS,
            Some(original.clone()),
        )
        .unwrap();
    let client_error = response.try_recv().unwrap().unwrap_err();
    assert!(matches!(
        client_error,
        FerrumError::RequestValidation { .. }
    ));
    assert_eq!(client_error.to_string(), original.to_string());
    assert!(handle.lock().is_none());
    assert!(engine.inner.sequences.is_empty());
    assert_eq!(
        sender.completed_text(0..marker.len()).unwrap().as_deref(),
        Some(marker)
    );
    assert_eq!(
        engine.inner.output_budget.usage().queued_bytes,
        held.bytes()
    );
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));

    drop(held);
    drop(blocker);
    let chunk = tokio::time::timeout(Duration::from_secs(2), receiver.recv())
        .await
        .expect("the original published span must rebuild after credit returns")
        .unwrap()
        .unwrap();
    assert_eq!(chunk.chunk().request_id, request_id);
    assert_eq!(chunk.chunk().text, marker);
    assert_eq!(chunk.chunk().token, Some(token));
    assert!(chunk.chunk().finish_reason.is_none());
    assert!(chunk.chunk().usage.is_none());
    let stream_error = tokio::time::timeout(Duration::from_secs(2), receiver.recv())
        .await
        .expect("the original error follows the previously published marker")
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        stream_error,
        FerrumError::RequestValidation { .. }
    ));
    assert_eq!(stream_error.to_string(), original.to_string());
    engine.inner.output_tasks.join_current().await;
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert!(identity.observed_end().is_some());
    assert!(
        sender.failure().is_none(),
        "no secondary span-projection error"
    );
    assert!(engine.inner.output_tasks.take_control_error().is_none());
    drop(sender);
    assert!(
        receiver.recv().await.is_none(),
        "an error must not gain a successful terminal"
    );
    engine.shutdown().await.unwrap();
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    drop(chunk);
    assert_eq!(engine.inner.output_budget.usage().queued_bytes, 0);
    assert_eq!(engine.inner.output_budget.usage().sessions, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn harmony_pending_call_marker_survives_explicit_completion_error() {
    harmony_pending_terminal_preserves_published_range("<|call|>").await;
}

#[tokio::test(flavor = "current_thread")]
async fn harmony_pending_return_marker_survives_explicit_completion_error() {
    harmony_pending_terminal_preserves_published_range("<|return|>").await;
}

async fn stop_boundary_outputs(
    pieces: &[&str],
    stops: &[&str],
) -> (InferenceOutput, Vec<ferrum_output::OutputChunk>) {
    stop_boundary_outputs_with_eos(pieces, stops, false).await
}

async fn stop_boundary_outputs_with_eos(
    pieces: &[&str],
    stops: &[&str],
    end_with_eos: bool,
) -> (InferenceOutput, Vec<ferrum_output::OutputChunk>) {
    let mut request = policy_request();
    request.sampling_params.stop_sequences = stops.iter().map(|s| (*s).to_owned()).collect();
    request.sampling_params.max_tokens = pieces.len() + 2 + usize::from(end_with_eos);
    stop_boundary_outputs_for_request(pieces, end_with_eos, request).await
}

async fn stop_boundary_outputs_for_request(
    pieces: &[&str],
    end_with_eos: bool,
    mut request: InferenceRequest,
) -> (InferenceOutput, Vec<ferrum_output::OutputChunk>) {
    let engine = stop_boundary_engine(pieces, end_with_eos);
    let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request.clone()))
        .await
        .expect("fixed token sequence must complete")
        .unwrap();
    engine.shutdown().await.unwrap();

    let engine = stop_boundary_engine(pieces, end_with_eos);
    request.id = RequestId::new();
    request.stream = true;
    let mut stream = engine.infer_stream(request).await.unwrap();
    let chunks = tokio::time::timeout(Duration::from_secs(2), async {
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        chunks
    })
    .await
    .expect("fixed token stream must terminate");
    engine.shutdown().await.unwrap();
    (response, chunks)
}

#[tokio::test]
async fn harmony_user_stop_preserves_exact_payload_and_terminal_usage() {
    const FINAL: &str = "<|channel|>final<|message|>";
    for pieces in [
        vec![FINAL, "答案 ", "ST", "OPtail"],
        vec![FINAL, "答案 STOPtail"],
    ] {
        let mut request = policy_request();
        request.sampling_params.model_output_protocol =
            ferrum_types::ModelOutputProtocol::HarmonyGptOss;
        request.sampling_params.stop_sequences = vec!["STOP".to_string()];
        request.sampling_params.max_tokens = pieces.len() + 2;
        let (response, chunks) = stop_boundary_outputs_for_request(&pieces, false, request).await;
        let expected = format!("{FINAL}答案 ");
        let mut streamed = String::new();
        for chunk in &chunks {
            streamed.push_str(&chunk.chunk().text);
            assert!(
                expected.starts_with(&streamed),
                "a stop prefix leaked: {streamed:?}"
            );
        }
        assert_eq!(response.response().text, expected);
        assert_eq!(streamed, expected);
        assert_eq!(response.response().finish_reason, FinishReason::Stop);
        let terminal = chunks.last().unwrap();
        assert!(terminal.chunk().text.is_empty());
        assert_eq!(terminal.chunk().finish_reason, Some(FinishReason::Stop));
        let usage = terminal.chunk().usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, response.response().usage.prompt_tokens);
        assert_eq!(
            usage.completion_tokens,
            response.response().usage.completion_tokens
        );
        assert_eq!(usage.total_tokens, response.response().usage.total_tokens);
        assert_eq!(
            response.response().usage.completion_tokens,
            response.response().tokens.len()
        );
    }
}

#[tokio::test]
async fn stop_boundary_react_marker_never_leaks_into_stream() {
    let (response, chunks) = stop_boundary_outputs(
        &["Action Input: wiki", "\n", "Observ", "ation"],
        &["\nObservation"],
    )
    .await;
    let text = chunks
        .iter()
        .map(|chunk| chunk.chunk().text.as_str())
        .collect::<String>();
    assert_eq!(
        text, "Action Input: wiki",
        "stop prefixes must remain buffered"
    );
    assert_eq!(
        response.response().text,
        text,
        "sync and stream must have the same boundary"
    );
    assert_eq!(response.response().finish_reason, FinishReason::Stop);
    let terminal = chunks.last().unwrap();
    assert_eq!(terminal.chunk().finish_reason, Some(FinishReason::Stop));
    let usage = terminal.chunk().usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, response.response().usage.prompt_tokens);
    assert_eq!(
        usage.completion_tokens,
        response.response().usage.completion_tokens
    );
    assert_eq!(usage.total_tokens, response.response().usage.total_tokens);
    assert_eq!(
        response.response().usage.completion_tokens,
        6,
        "stop tokens still consumed model work"
    );
}

#[tokio::test]
async fn stop_boundary_inside_token_preserves_preceding_text() {
    let (response, chunks) = stop_boundary_outputs(&["OK \n\n"], &["\n"]).await;
    let text = chunks
        .iter()
        .map(|chunk| chunk.chunk().text.as_str())
        .collect::<String>();
    assert_eq!(
        text, "OK ",
        "a stop must not discard the valid text before it"
    );
    assert_eq!(response.response().text, text);
    assert_eq!(response.response().finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn stop_boundary_flushes_and_selects_the_earliest_match() {
    struct Case<'a> {
        pieces: &'a [&'a str],
        stops: &'a [&'a str],
        expected: &'a str,
        finish: FinishReason,
        end_with_eos: bool,
    }
    for case in [
        Case {
            pieces: &["answer", "ab"],
            stops: &["abc"],
            expected: "answerab",
            finish: FinishReason::Length,
            end_with_eos: false,
        },
        Case {
            pieces: &["answer", "ab"],
            stops: &["abc"],
            expected: "answerab",
            finish: FinishReason::EOS,
            end_with_eos: true,
        },
        Case {
            pieces: &["keep:", "ab", "cSTOPtail"],
            stops: &["", "STOP", "bc", "abc", "abcd"],
            expected: "keep:",
            finish: FinishReason::Stop,
            end_with_eos: false,
        },
        Case {
            pieces: &["答案：", "终", "止后续"],
            stops: &["终止"],
            expected: "答案：",
            finish: FinishReason::Stop,
            end_with_eos: false,
        },
    ] {
        let (response, chunks) =
            stop_boundary_outputs_with_eos(case.pieces, case.stops, case.end_with_eos).await;
        let text = chunks
            .iter()
            .map(|chunk| chunk.chunk().text.as_str())
            .collect::<String>();
        assert_eq!(
            text, case.expected,
            "pieces={:?}, stops={:?}",
            case.pieces, case.stops
        );
        assert_eq!(response.response().text, text);
        assert_eq!(response.response().finish_reason, case.finish);
        assert_eq!(
            chunks.last().unwrap().chunk().finish_reason,
            Some(case.finish)
        );
    }
}

#[tokio::test]
async fn stop_boundary_mismatch_releases_the_prefix_before_later_output() {
    let (response, chunks) = stop_boundary_outputs(&["ab", "x", " tail"], &["abc"]).await;
    let mut text = String::new();
    let mut released_after_mismatch = false;
    for chunk in chunks {
        text.push_str(&chunk.chunk().text);
        released_after_mismatch |= text == "abx" && chunk.chunk().finish_reason.is_none();
    }
    assert!(
        released_after_mismatch,
        "a disproven stop prefix must become visible before subsequent output"
    );
    assert_eq!(text, "abx tail");
    assert_eq!(response.response().text, text);
    assert_eq!(response.response().finish_reason, FinishReason::Length);
}

#[test]
fn stop_boundary_buffer_does_not_hide_a_non_stop_control_token() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[("test", 0), ("ab", 4), ("x", 5), ("<think>", 7)],
    ));
    let mut request = policy_request();
    request.sampling_params.stop_sequences = vec!["abc".to_owned()];
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer.clone()));
    state.generated_tokens.push(TokenId::new(4));
    // "ab" is valid decoded text, withheld solely because "abc" may follow.
    state.streamed_text_len = 0;
    state.decoded_text_len = 2;
    assert!(state.allowed_extended_token_ids.contains(&7));
    assert!(!state.stop_token_ids.contains(&7));

    let mut logits = vec![f32::NEG_INFINITY; 8];
    logits[5] = 1.0;
    logits[7] = 100.0;
    let sampled = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();

    assert_eq!(
        sampled,
        TokenId::new(5),
        "hidden non-stop control must be rejected even while visible text is buffered"
    );
    assert_eq!(logits[7], f32::NEG_INFINITY);
}

#[tokio::test]
async fn stop_boundary_rejects_strict_json_truncated_inside_a_token() {
    let make_engine = |generated: &str, stop: &str, terminal_token: Option<u32>| {
        let mut pairs = vec![("test", 5), (generated, 7), (stop, 8)];
        if terminal_token == Some(3) {
            pairs.push(("</s>", 3));
        }
        let mut tokenizer = PolicyTokenizer::new(16, &pairs);
        if terminal_token == Some(8) {
            // After grammar acceptance the fixture's tied fallback logits
            // must choose the explicit whitespace stop before ordinary EOS.
            tokenizer.special.eos_token = Some(TokenId::new(15));
        }
        let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
        let config = EngineConfig::default();
        ContinuousBatchEngine::new(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
            tokenizer,
            Arc::new(crate::registry::GreedySampler),
            Arc::new(MockKvCacheManager::new(256)),
            Arc::new(StopBoundaryExecutor::with_token_sequence(16, vec![7])),
            Arc::new(MockTensorFactory),
        )
        .unwrap()
    };
    let object_schema = r#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}"#;
    for (generated, schema, stop, expected, terminal_token) in [
        (r#"{"value":1}"#, object_schema, "}", None, None),
        ("123", r#"{"enum":[123]}"#, "3", None, None),
        (
            r#"{"value":1}"#,
            object_schema,
            "\n",
            Some(r#"{"value":1}"#),
            Some(8),
        ),
        ("123", r#"{"enum":[123]}"#, "</s>", Some("123"), Some(3)),
    ] {
        let mut request = policy_request();
        request.sampling_params.max_tokens = 1;
        request.sampling_params.response_format =
            ferrum_types::ResponseFormat::JsonSchema(schema.to_string());

        // The same ordinary token must produce valid structured output without
        // a stop. A failure setting up the grammar is not evidence of this bug.
        let engine = make_engine(generated, stop, terminal_token);
        let response = engine.infer(request.clone()).await.unwrap_or_else(|error| {
            panic!("no-stop control failed: generated={generated:?}, stop={stop:?}: {error}")
        });
        assert_eq!(response.response().text, generated);
        engine.shutdown().await.unwrap();

        // Stop token 8 does not mask ordinary token 7. A stop inside its JSON
        // value must fail, while a stop in trailing whitespace is harmless.
        request.id = RequestId::new();
        request.sampling_params.stop_sequences = vec![stop.to_string()];
        request.sampling_params.max_tokens += usize::from(terminal_token.is_some());
        let engine = make_engine(generated, stop, terminal_token);
        let result = engine.infer(request.clone()).await;
        engine.shutdown().await.unwrap();
        if let Some(expected) = expected {
            let response = result.unwrap_or_else(|error| {
                panic!("stop after complete value failed: generated={generated:?}, stop={stop:?}: {error}")
            });
            assert_eq!(response.response().text, expected);
            assert_eq!(response.response().finish_reason, FinishReason::Stop);
            if let Some(terminal_token) = terminal_token {
                assert_eq!(
                    response.response().tokens.last(),
                    Some(&TokenId::new(terminal_token))
                );
            }
        } else {
            assert!(
                result.is_err(),
                "stop {stop:?} must not turn {generated:?} into successful strict output: {result:?}"
            );
        }

        let engine = make_engine(generated, stop, terminal_token);
        request.id = RequestId::new();
        request.stream = true;
        let mut stream = engine.infer_stream(request).await.unwrap();
        let chunks = tokio::time::timeout(Duration::from_secs(2), async {
            let mut chunks = Vec::new();
            while let Some(chunk) = stream.next().await {
                chunks.push(chunk);
            }
            chunks
        })
        .await
        .expect("structured output must terminate");
        engine.shutdown().await.unwrap();
        if let Some(expected) = expected {
            let chunks = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
            let text = chunks
                .iter()
                .map(|chunk| chunk.chunk().text.as_str())
                .collect::<String>();
            assert_eq!(text, expected);
            assert_eq!(
                chunks.last().unwrap().chunk().finish_reason,
                Some(FinishReason::Stop)
            );
        } else {
            assert!(chunks.iter().any(Result::is_err), "chunks: {chunks:?}");
            assert!(
                chunks
                    .iter()
                    .filter_map(|chunk| chunk.as_ref().ok())
                    .all(|chunk| {
                        chunk.chunk().text.is_empty()
                            && chunk.chunk().finish_reason.is_none()
                            && chunk.chunk().usage.is_none()
                    }),
                "invalid structured tail must not be flushed as successful output: {chunks:?}"
            );
        }
    }
}
