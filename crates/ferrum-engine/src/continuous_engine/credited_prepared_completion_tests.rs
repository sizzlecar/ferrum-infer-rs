//! Real credited admission/waves, with legacy completion behavior as oracle.
use super::*;
use ferrum_interfaces::tokenizer::BoundedIncrementalDecodePolicy;
use ferrum_types::{ResponseCompletionBoundary, ResponseCompletionEnvelope};

struct PreparedTokenizer {
    base: PolicyTokenizer,
    calls: Arc<CreditedLegacyDecodeCalls>,
    encode_calls: AtomicUsize,
}
impl PreparedTokenizer {
    fn new(calls: Arc<CreditedLegacyDecodeCalls>) -> Self {
        let mut base = PolicyTokenizer::new(
            64,
            &[
                ("test", 5),
                ("</", 6),
                ("think", 7),
                (">", 8),
                ("\n", 9),
                ("answer", 10),
                ("<tool_", 11),
                ("call", 12),
                ("</tool_", 13),
                ("payload", 14),
                ("<eos>", 3),
                ("X", 16),
            ],
        );
        for (text, ids) in [
            ("</think>", [6, 7, 8]),
            ("<tool_call>", [11, 12, 8]),
            ("</tool_call>", [13, 12, 8]),
        ] {
            base.encoded_sequences
                .insert(text.into(), ids.map(TokenId::new).to_vec());
        }
        base.hidden_decode_token_ids.insert(3);
        Self {
            base,
            calls,
            encode_calls: AtomicUsize::new(0),
        }
    }
}
impl Tokenizer for PreparedTokenizer {
    fn encode(&self, text: &str, special: bool) -> Result<Vec<TokenId>> {
        self.encode_calls.fetch_add(1, Ordering::Relaxed);
        self.base.encode(text, special)
    }
    fn decode(&self, tokens: &[TokenId], skip: bool) -> Result<String> {
        self.calls.decode.fetch_add(1, Ordering::Relaxed);
        self.base.decode(tokens, skip)
    }
    fn decode_incremental(&self, previous: &[TokenId], next: TokenId) -> Result<String> {
        self.calls.incremental.fetch_add(1, Ordering::Relaxed);
        let before = self.base.decode(previous, true)?;
        let mut full = previous.to_vec();
        full.push(next);
        let full = self.base.decode(&full, true)?;
        Ok(full.strip_prefix(&before).unwrap().to_owned())
    }
    fn vocab_size(&self) -> usize {
        self.base.vocab_size()
    }
    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        self.base.special_tokens()
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.base.token_id(text)
    }
    fn token_text(&self, token: TokenId) -> Option<&str> {
        self.base.token_text(token)
    }
    fn prepared_completion_tokens(&self, text: &str) -> Option<&[TokenId]> {
        self.base.encoded_sequences.get(text).map(Vec::as_slice)
    }
    fn info(&self) -> TokenizerInfo {
        self.base.info()
    }
    fn bounded_decode_bound(&self) -> Option<BoundedDecodeBound> {
        Some(BoundedDecodeBound::new(NonZeroUsize::new(8).unwrap(), 0))
    }
    fn bounded_incremental_decode_policy(&self) -> Option<BoundedIncrementalDecodePolicy> {
        Some(BoundedIncrementalDecodePolicy::StrictDecodedPrefix)
    }
    fn bounded_token_bytes_bound(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(8)
    }
    fn token_bytes_bounded_into(
        &self,
        token: TokenId,
        output: &mut [u8],
    ) -> std::result::Result<Option<usize>, BoundedDecodeError> {
        if output.len() < 8 {
            return Err(BoundedDecodeError::InsufficientScratch {
                required: 8,
                available: output.len(),
            });
        }
        let Some(text) = self.token_text(token) else {
            return Ok(None);
        };
        output[..text.len()].copy_from_slice(text.as_bytes());
        Ok(Some(text.len()))
    }
    fn decode_bounded_into(
        &self,
        tokens: &[TokenId],
        skip: bool,
        _: &mut [u8],
        output: &mut String,
    ) -> std::result::Result<(), BoundedDecodeError> {
        let required = self
            .bounded_decode_bound()
            .unwrap()
            .requirements(tokens.len())?
            .text_bytes;
        if output.capacity() < required {
            return Err(BoundedDecodeError::InsufficientOutput {
                required,
                available: output.capacity(),
            });
        }
        output.clear();
        for token in tokens {
            if skip && self.base.hidden_decode_token_ids.contains(&token.get()) {
                continue;
            }
            if let Some(text) = self.token_text(*token) {
                output.push_str(text);
            }
        }
        Ok(())
    }
}

struct ScriptExecutor {
    base: Arc<PlanRuntimeBatchDecodeTestExecutor>,
    script: Vec<(u32, bool)>,
    next: AtomicUsize,
}
impl ScriptExecutor {
    fn logits(&self) -> Vec<f32> {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        let (token, prefer_eos) = self.script[index];
        let mut logits = vec![f32::NEG_INFINITY; 64];
        logits[3] = if prefer_eos { 100. } else { 1. };
        logits[token as usize] = if token == 3 { 100. } else { 10. };
        logits
    }
}
#[async_trait::async_trait]
impl ModelExecutor for ScriptExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.base.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.base.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.base.status()
    }
    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }
    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        self.base.plan_runtime_resource_snapshot()
    }
    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        self.base.execution_capacity_epochs()
    }
    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<ferrum_interfaces::vnext::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        self.base.write_execution_capacity_snapshot(availability)
    }
    fn write_execution_capacity_release_sources(
        &self,
        preemption: &ExecutorExecutionCapacityPreemption,
        sources: &mut Vec<ferrum_interfaces::vnext::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        self.base
            .write_execution_capacity_release_sources(preemption, sources)
    }
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.base.prefill(input).await
    }
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.base.decode(input).await
    }
    async fn plan_runtime_prefill_with_capacity(
        &self,
        input: &PlanRuntimePrefillInput,
    ) -> Result<PlanRuntimePrefillOutcome> {
        let PlanRuntimePrefillOutcome::Completed(completion) =
            self.base.plan_runtime_prefill_with_capacity(input).await?
        else {
            panic!("exact fixture deferred")
        };
        let (output, planned, completed, probes) = completion.into_parts();
        let output = if completed.is_final() {
            PlanRuntimePrefillOutput::final_logits(
                output.request_id().clone(),
                output.committed_tokens(),
                self.logits(),
                output.kv_cache().clone(),
            )?
        } else {
            output
        };
        Ok(PlanRuntimePrefillOutcome::Completed(
            PlanRuntimePrefillCompletion::new(output, planned, completed, probes)?,
        ))
    }
    async fn plan_runtime_batch_decode_with_capacity(
        &self,
        inputs: &[PlanRuntimeDecodeInput],
    ) -> Result<PlanRuntimeBatchDecodeOutcome> {
        assert_eq!(inputs.len(), 1);
        Ok(PlanRuntimeBatchDecodeOutcome::Completed(vec![
            PlanRuntimeDecodeOutput::new(
                ExecutorSamplingOutput::FullLogits(self.logits()),
                inputs[0].kv_cache.clone(),
            ),
        ]))
    }
    fn release_cache(&self, id: &str) {
        self.base.release_cache(id);
    }
}
fn request(alternate: bool, max_envelopes: usize, max_tokens: usize) -> InferenceRequest {
    let mut request = policy_request();
    request.stream = true;
    request.sampling_params.max_tokens = max_tokens;
    request.sampling_params.response_completion_boundary =
        ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "</think>".into(),
            alternate_envelope: alternate.then(|| ResponseCompletionEnvelope {
                open_token_text: "<tool_call>".into(),
                close_token_text: "</tool_call>".into(),
                max_envelopes,
            }),
        };
    request
}
fn fixture(
    script: Vec<(u32, bool)>,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ScriptExecutor>,
    Arc<PreparedTokenizer>,
) {
    let ((mut engine, scheduler, base), calls) = credited_counted_fixture();
    let executor = Arc::new(ScriptExecutor {
        base,
        script,
        next: AtomicUsize::new(0),
    });
    let tokenizer = Arc::new(PreparedTokenizer::new(calls));
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.tokenizer = tokenizer.clone();
    inner.model_executor = executor.clone();
    (engine, scheduler, executor, tokenizer)
}
async fn submit(
    engine: &ContinuousBatchEngine,
    request: InferenceRequest,
) -> (RequestId, CreditedOutputSession) {
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    (id, session)
}

#[tokio::test]
async fn credited_prepared_completion_multitoken_and_alternate_match_legacy_each_step() {
    for (alternate, trace, second_open) in [
        (false, vec![6, 7, 8, 9, 10, 3], None),
        (
            true,
            vec![11, 12, 8, 14, 13, 12, 8, 11, 12, 8, 14, 13, 12, 8, 3],
            Some(7),
        ),
        (true, vec![6, 7, 8, 11, 16, 3], None),
    ] {
        let script: Vec<_> = trace
            .iter()
            .enumerate()
            .map(|(i, t)| (*t, Some(i) != second_open))
            .collect();
        let (engine, scheduler, executor, tokenizer) = fixture(script.clone());
        let request = request(alternate, 2, trace.len() + 1);
        let legacy_tok = Arc::new(PreparedTokenizer::new(Arc::new(
            CreditedLegacyDecodeCalls::default(),
        )));
        let mut oracle = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            vec![TokenId::new(5)],
            Some(legacy_tok.clone()),
            Some(64),
        );
        let (id, mut session) = submit(&engine, request).await;
        assert_eq!(
            tokenizer.encode_calls.load(Ordering::Relaxed),
            1,
            "only the prompt may use legacy encode"
        );
        tokenizer.calls.reset();
        let mut wire = String::new();
        for (index, (expected, prefer_eos)) in script.into_iter().enumerate() {
            let mut logits = vec![f32::NEG_INFINITY; 64];
            logits[3] = if prefer_eos { 100. } else { 1. };
            logits[expected as usize] = if expected == 3 { 100. } else { 10. };
            let sampled = oracle
                .sample_and_commit_with_processors_and_tokenizer(
                    &mut logits,
                    Some(legacy_tok.as_ref()),
                )
                .unwrap();
            assert_eq!(sampled, TokenId::new(expected));
            credited_ready(&engine, &id).await;
            engine
                .inner
                .process_batch(&credited_next(&scheduler, 1).await)
                .await
                .unwrap();
            if expected != 3 {
                let states = engine.inner.sequences.read();
                let actual = states
                    .get(&id)
                    .unwrap_or_else(|| panic!("sequence missing at trace {index}"));
                assert_eq!(actual.generated_tokens, oracle.generated_tokens);
                assert_eq!(
                    actual.response_completion_state.allows_model_eos(),
                    oracle.response_completion_state.allows_model_eos()
                );
                assert_eq!(
                    actual
                        .argmax_token_mask
                        .as_ref()
                        .map(|m| m.valid_token_mask[3]),
                    oracle
                        .argmax_token_mask
                        .as_ref()
                        .map(|m| m.valid_token_mask[3])
                );
                assert_eq!(
                    actual.stop_reason(Some(tokenizer.as_ref())),
                    oracle.stop_reason(Some(legacy_tok.as_ref()))
                );
                drop(states);
                let frame = credited_bound(session.frames.next()).await.unwrap();
                assert_eq!(frame.metadata().token, Some(TokenId::new(expected)));
                wire.push_str(std::str::from_utf8(frame.wire().payload()).unwrap());
                drop(frame);
            }
        }
        assert!(!engine.inner.sequences.read().contains_key(&id));
        let terminal = credited_bound(session.frames.next()).await.unwrap();
        assert!(terminal.metadata().terminal);
        drop(terminal);
        let completion = credited_bound(session.completion).await.unwrap();
        match completion.payload() {
            OutputCompletion::Succeeded {
                history: Some(history),
                usage,
                reason,
                ..
            } => {
                assert_eq!(*reason, FinishReason::EOS);
                assert_eq!(usage.completion_tokens, trace.len());
                assert_eq!(history.tokens, oracle.generated_tokens);
                assert_eq!(history.text, wire);
            }
            _ => panic!("prepared completion failed"),
        }
        assert_eq!(executor.next.load(Ordering::Relaxed), trace.len());
        assert_eq!(executor.base.inner.prefill_count(), 1);
        assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
        tokenizer.calls.assert_unused();
        drop((completion, session.frames));
        credited_drained(&engine).await;
    }
}

#[tokio::test]
async fn credited_prepared_completion_rejects_exact_delimiter_budget_and_eos_collision_before_submit(
) {
    for collision in [false, true] {
        let (mut engine, _, executor, mut tokenizer) = fixture(Vec::new());
        if collision {
            // Replace both shared handles before borrowing the immutable table.
            let mut changed = PreparedTokenizer::new(tokenizer.calls.clone());
            changed
                .base
                .encoded_sequences
                .insert("</think>".into(), vec![TokenId::new(3)]);
            tokenizer = Arc::new(changed);
            Arc::get_mut(&mut engine.inner).unwrap().tokenizer = tokenizer.clone();
        }
        let result = engine
            .infer_credited_stream(
                request(false, 1, 3),
                InferenceRequestContext::capture(),
                Arc::new(OutputProjectionContract::cli_text()),
            )
            .await;
        assert!(result.is_err());
        assert_eq!(executor.next.load(Ordering::Relaxed), 0);
        assert_eq!(executor.base.inner.prefill_count(), 0);
        assert!(engine.inner.sequences.read().is_empty());
        credited_drained(&engine).await;
        tokenizer.calls.assert_unused();
        assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
    }
}

#[tokio::test]
async fn credited_prepared_completion_envelope_limit_rejects_without_committing_or_retry() {
    let trace = [11, 12, 8, 14, 13, 12, 8, 11, 12, 8];
    let script: Vec<_> = trace
        .iter()
        .enumerate()
        .map(|(index, token)| (*token, index != 7))
        .collect();
    let (engine, scheduler, executor, tokenizer) = fixture(script.clone());
    let request = request(true, 1, trace.len() + 1);
    let legacy_tok = Arc::new(PreparedTokenizer::new(Arc::new(
        CreditedLegacyDecodeCalls::default(),
    )));
    let mut oracle = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        vec![TokenId::new(5)],
        Some(legacy_tok.clone()),
        Some(64),
    );
    let (id, mut session) = submit(&engine, request).await;
    tokenizer.calls.reset();
    for (index, (expected, prefer_eos)) in script.into_iter().enumerate() {
        let mut logits = vec![f32::NEG_INFINITY; 64];
        logits[3] = if prefer_eos { 100. } else { 1. };
        logits[expected as usize] = 10.;
        let before = oracle.generated_tokens.len();
        let result = oracle.sample_and_commit_with_processors_and_tokenizer(
            &mut logits,
            Some(legacy_tok.as_ref()),
        );
        credited_ready(&engine, &id).await;
        engine
            .inner
            .process_batch(&credited_next(&scheduler, 1).await)
            .await
            .unwrap();
        if index + 1 < trace.len() {
            assert_eq!(result.unwrap(), TokenId::new(expected));
            assert_eq!(
                engine.inner.sequences.read()[&id].generated_tokens,
                oracle.generated_tokens
            );
            drop(credited_bound(session.frames.next()).await.unwrap());
        } else {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("1-envelope protocol limit"));
            assert_eq!(oracle.generated_tokens.len(), before);
            assert!(!engine.inner.sequences.read().contains_key(&id));
        }
    }
    let terminal = credited_bound(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    drop(terminal);
    let completion = credited_bound(session.completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    engine.inner.run_iteration().await.unwrap();
    assert_eq!(executor.next.load(Ordering::Relaxed), trace.len());
    assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
    tokenizer.calls.assert_unused();
    drop((completion, session.frames));
    credited_drained(&engine).await;
}

#[tokio::test]
async fn credited_prepared_completion_cancel_retains_matcher_credit_until_sequence_cleanup() {
    let (engine, scheduler, executor, tokenizer) = fixture(vec![(6, true)]);
    let (id, mut session) = submit(&engine, request(true, 2, 12)).await;
    tokenizer.calls.reset();
    credited_ready(&engine, &id).await;
    engine
        .inner
        .process_batch(&credited_next(&scheduler, 1).await)
        .await
        .unwrap();
    let frame = credited_bound(session.frames.next()).await.unwrap();
    assert_eq!(frame.wire().payload(), b"</");
    drop(session.frames);
    assert!(engine.inner.sequences.read().contains_key(&id));
    assert!(credited_pool(&engine).snapshot().data_used.projection_bytes > 0);
    engine.inner.run_iteration().await.unwrap();
    assert!(!engine.inner.sequences.read().contains_key(&id));
    let completion = credited_bound(session.completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    drop(completion);
    assert!(
        credited_pool(&engine).snapshot().data_used.bytes > 0,
        "retained delivered bytes remain charged"
    );
    assert_eq!(executor.next.load(Ordering::Relaxed), 1);
    tokenizer.calls.assert_unused();
    assert_eq!(tokenizer.encode_calls.load(Ordering::Relaxed), 1);
    drop(frame);
    credited_drained(&engine).await;
}

#[tokio::test]
async fn credited_prepared_completion_ignore_eos_keeps_original_one_token_budget() {
    let (mut engine, scheduler, executor, tokenizer) = fixture(vec![(3, true)]);
    // With ignore_eos this ID becomes an ordinary non-stop control. Existing
    // sampling quality rejects controls whose decode is empty. Use the same
    // visible EOS surface as the legacy ignore-EOS regression, rather than
    // demanding that the bounded path bypass the quality policy.
    let mut visible_eos = PreparedTokenizer::new(tokenizer.calls.clone());
    visible_eos.base.hidden_decode_token_ids.remove(&3);
    let tokenizer = Arc::new(visible_eos);
    Arc::get_mut(&mut engine.inner).unwrap().tokenizer = tokenizer.clone();
    let mut request = request(true, 2, 1);
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    let mut oracle = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        vec![TokenId::new(5)],
        Some(tokenizer.clone()),
        Some(64),
    );
    let mut logits = vec![f32::NEG_INFINITY; 64];
    logits[3] = 100.;
    assert_eq!(
        oracle
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap(),
        TokenId::new(3)
    );
    assert_eq!(
        oracle.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::Length)
    );
    let (id, mut session) = submit(&engine, request).await;
    tokenizer.calls.reset();
    {
        let states = engine.inner.sequences.read();
        assert!(states[&id].response_completion_state.allows_model_eos());
        assert_eq!(states[&id].sampling_params.max_tokens, 1);
    }
    credited_ready(&engine, &id).await;
    engine
        .inner
        .process_batch(&credited_next(&scheduler, 1).await)
        .await
        .unwrap();
    let data = credited_bound(session.frames.next()).await.unwrap();
    assert_eq!(data.metadata().token, Some(TokenId::new(3)));
    assert_eq!(data.wire().payload(), b"<eos>");
    drop(data);
    let terminal = credited_bound(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    drop(terminal);
    let completion = credited_bound(session.completion).await.unwrap();
    match completion.payload() {
        OutputCompletion::Succeeded {
            reason,
            usage,
            history: Some(history),
            ..
        } => {
            assert_eq!(*reason, FinishReason::Length);
            assert_eq!(usage.completion_tokens, 1);
            assert_eq!(history.tokens, vec![TokenId::new(3)]);
            assert_eq!(history.text, "<eos>");
        }
        OutputCompletion::Failed(error) => panic!("ignore EOS failed: {}", error.message()),
        _ => panic!("ignore EOS lost its original completion history"),
    }
    assert_eq!(executor.next.load(Ordering::Relaxed), 1);
    tokenizer.calls.assert_unused();
    drop((completion, session.frames));
    credited_drained(&engine).await;
}
