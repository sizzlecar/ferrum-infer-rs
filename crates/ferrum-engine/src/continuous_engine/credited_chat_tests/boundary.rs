use super::*;
use ferrum_interfaces::tokenizer::BoundedIncrementalDecodePolicy;

/// Literal concatenation with an explicitly declared strict-prefix incremental
/// contract. The parent PolicyTokenizer intentionally has different marker
/// semantics, so it cannot establish this capability.
struct BoundaryTokenizer {
    base: PolicyTokenizer,
    legacy: Arc<CreditedLegacyDecodeCalls>,
}
impl BoundaryTokenizer {
    fn new(legacy: Arc<CreditedLegacyDecodeCalls>) -> Self {
        Self {
            base: PolicyTokenizer::new(
                64,
                &[
                    ("test", 5),
                    ("</think>", 6),
                    ("\n", 7),
                    ("\t", 8),
                    ("answer", 9),
                    ("�", 10),
                    ("</s>", 3),
                ],
            ),
            legacy,
        }
    }
    fn text(&self, tokens: &[TokenId], skip: bool, output: &mut String) {
        output.clear();
        for token in tokens {
            if skip && token.get() == 3 {
                continue;
            }
            if let Some(text) = self.base.token_text(*token) {
                output.push_str(text);
            }
        }
    }
}
impl Tokenizer for BoundaryTokenizer {
    fn encode(&self, text: &str, special: bool) -> Result<Vec<TokenId>> {
        self.base.encode(text, special)
    }
    fn decode(&self, tokens: &[TokenId], skip: bool) -> Result<String> {
        self.legacy.decode.fetch_add(1, Ordering::Relaxed);
        let mut output = String::new();
        self.text(tokens, skip, &mut output);
        Ok(output)
    }
    fn decode_incremental(&self, previous: &[TokenId], next: TokenId) -> Result<String> {
        self.legacy.incremental.fetch_add(1, Ordering::Relaxed);
        let old = self.decode(previous, true)?;
        let mut combined = previous.to_vec();
        combined.push(next);
        let full = self.decode(&combined, true)?;
        Ok(full.strip_prefix(&old).unwrap().to_owned())
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
        self.text(tokens, skip, output);
        Ok(())
    }
}

struct BoundaryExecutor {
    base: Arc<PlanRuntimeBatchDecodeTestExecutor>,
    decoded: AtomicUsize,
}
#[async_trait::async_trait]
impl ModelExecutor for BoundaryExecutor {
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
        self.base.plan_runtime_prefill_with_capacity(input).await
    }
    async fn plan_runtime_batch_decode_with_capacity(
        &self,
        inputs: &[PlanRuntimeDecodeInput],
    ) -> Result<PlanRuntimeBatchDecodeOutcome> {
        let step = self.decoded.fetch_add(1, Ordering::Relaxed);
        let selected = [7, 8, 9, 3][step];
        let outputs = inputs
            .iter()
            .map(|input| {
                let mut logits = vec![f32::NEG_INFINITY; 64];
                logits[selected] = 10.;
                // Actual sequence sampling must suppress the otherwise preferred
                // EOS until a non-whitespace, non-replacement payload appears.
                logits[3] = 100.;
                PlanRuntimeDecodeOutput::new(
                    ExecutorSamplingOutput::FullLogits(logits),
                    input.kv_cache.clone(),
                )
            })
            .collect();
        Ok(PlanRuntimeBatchDecodeOutcome::Completed(outputs))
    }
    fn release_cache(&self, id: &str) {
        self.base.release_cache(id);
    }
}

#[tokio::test]
async fn credited_engine_chat_real_delayed_boundary_masks_eos_until_bounded_payload() {
    let ((mut engine, scheduler, base), legacy) = credited_counted_fixture();
    enable_chat_events(&mut engine);
    let executor = Arc::new(BoundaryExecutor {
        base,
        decoded: AtomicUsize::new(0),
    });
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.tokenizer = Arc::new(BoundaryTokenizer::new(legacy.clone()));
    inner.model_executor = executor.clone();
    let mut request = chat_request(6, true, true);
    request.sampling_params.response_completion_boundary =
        ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "</think>".into(),
            alternate_envelope: None,
        };
    let id = request.id.clone();
    let mut session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::chat_sse(
                id.to_string(),
                "wire-alias".into(),
                true,
            )),
        )
        .await
        .unwrap();
    legacy.reset();
    {
        let states = engine.inner.sequences.read();
        let state = &states[&id];
        let tokenizer = engine.inner.tokenizer.as_ref();
        let decoder = state.credited_output.as_ref().unwrap().decoder;
        // U+FFFD is not completion payload. The engine's existing raw-byte
        // quality policy also rejects this literal scalar before committing
        // it, so it cannot be part of a successful physical token trace.
        assert!(!decoder
            .incremental_has_payload(tokenizer, &[TokenId::new(6)], TokenId::new(10))
            .unwrap());
        assert!(state.sample_candidate_decodes_to_forbidden_output(
            Some(tokenizer),
            0,
            TokenId::new(10),
            None,
        ));
        assert!(!state.response_completion_state.allows_model_eos());
    }
    let mut content = String::new();
    for (step, expected) in [6u32, 7, 8, 9, 3].into_iter().enumerate() {
        credited_ready(&engine, &id).await;
        engine
            .inner
            .process_batch(&credited_next(&scheduler, 1).await)
            .await
            .unwrap();
        if expected != 3 {
            let states = engine.inner.sequences.read();
            let state = states.get(&id).unwrap_or_else(|| {
                panic!("request disappeared after boundary trace step {step}, token {expected}")
            });
            assert_eq!(state.generated_tokens.last(), Some(&TokenId::new(expected)));
            assert_eq!(
                state.response_completion_state.allows_model_eos(),
                step == 3
            );
        }
        if matches!(expected, 8 | 9) {
            let frame = credited_bound(session.frames.next()).await.unwrap();
            content.push_str(
                events(&frame)[0]["choices"][0]["delta"]["content"]
                    .as_str()
                    .unwrap(),
            );
            drop(frame);
        }
    }
    assert_eq!(content, "\tanswer");
    assert_eq!(executor.base.inner.prefill_count(), 1);
    assert_eq!(executor.decoded.load(Ordering::Relaxed), 4);
    assert!(!engine.inner.sequences.read().contains_key(&id));
    let terminal = credited_bound(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    assert_eq!(events(&terminal)[0]["choices"][0]["finish_reason"], "stop");
    assert_eq!(events(&terminal)[1]["usage"]["completion_tokens"], 5);
    drop(terminal);
    let completion = credited_bound(session.completion).await.unwrap();
    match completion.payload() {
        OutputCompletion::Succeeded {
            reason,
            history: Some(history),
            ..
        } => {
            assert_eq!(*reason, FinishReason::EOS);
            assert_eq!(history.tokens, [6, 7, 8, 9, 3].map(TokenId::new));
            assert_eq!(history.text, "</think>\n\tanswer");
        }
        _ => panic!("delayed Chat completion failed"),
    }
    drop((completion, session.frames));
    credited_drained(&engine).await;
    legacy.assert_unused();
}
