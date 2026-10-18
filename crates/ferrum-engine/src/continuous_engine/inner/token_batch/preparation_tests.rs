//! Real tiny CPU execution behind a deliberately invalid prompt-tail hook.
//! The replacement case exercises exact-incarnation protection, not a GPU path.

use super::*;
use ferrum_execution::model_executor::{
    ExecutorCapabilities, ExecutorPrefillAdmission, ExecutorStatus, PrefixCapturePlan,
};
use ferrum_execution::token_execution::{
    ExecutionAdmission, ExecutionCloseDisposition, ExecutionEndReason, ExecutorSequence,
    TokenAdmissionOutcome,
};
use ferrum_kv::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_models::test_support::{tiny_llama_executor, TinyLlamaConfig, TinyTokenizer};

struct InvalidTailExecutor {
    inner: Arc<dyn ModelExecutor>,
    bad_hook: AtomicBool,
    executed: Mutex<Vec<ExecutorSequence>>,
    cancelled: Mutex<Vec<ExecutorSequence>>,
    healthy_at_close: Mutex<Option<(std::sync::Weak<EngineInner>, RequestId)>>,
    close_progress: Mutex<Vec<(usize, usize)>>,
}

#[async_trait::async_trait]
impl ModelExecutor for InvalidTailExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        self.inner.admit_tokens(input).await
    }
    fn plan_prompt_tail_capture_boundary(&self, chunk: PrefillChunk) -> Option<PrefixCapturePlan> {
        (self.bad_hook.load(Ordering::Relaxed) && chunk.total_prompt_tokens() == 3).then(|| {
            PrefixCapturePlan {
                // A capture boundary must precede the prompt's final token.
                boundary: chunk.total_prompt_tokens(),
                span: ferrum_runtime::CheckpointTokenSpanConstraint::any_positive(),
            }
        })
    }
    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        self.executed.lock().extend(
            work.iter()
                .map(|work| work.expected_frontier().sequence().clone()),
        );
        self.inner.execute_tokens(work).await
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.inner.finish_tokens(sequence, reason).await
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        if let Some((engine, request)) = self.healthy_at_close.lock().as_ref() {
            let engine = engine.upgrade().expect("test engine remains live");
            let states = engine.sequences.read();
            let healthy = &states[request];
            self.close_progress.lock().push((
                healthy
                    .execution_frontier
                    .as_ref()
                    .unwrap()
                    .consumed_tokens(),
                healthy.generated_tokens.len(),
            ));
        }
        self.cancelled.lock().push(sequence.clone());
        self.inner.cancel_tokens(sequence)
    }
}

async fn fixture(
    prompt_lengths: &[usize],
) -> (
    ContinuousBatchEngine,
    Arc<InvalidTailExecutor>,
    ferrum_interfaces::BatchPlan,
    Vec<(RequestId, ExecutionFrontier)>,
) {
    let cfg = TinyLlamaConfig::default();
    let executor = Arc::new(InvalidTailExecutor {
        inner: tiny_llama_executor(&cfg),
        bad_hook: AtomicBool::new(true),
        executed: Mutex::new(Vec::new()),
        cancelled: Mutex::new(Vec::new()),
        healthy_at_close: Mutex::new(None),
        close_progress: Mutex::new(Vec::new()),
    });
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(TinyTokenizer::new(cfg.vocab_size));
    let kv = Arc::new(
        PagedKvCacheManager::new(
            Device::CPU,
            PagedKvCacheConfig {
                block_size: 16,
                max_gpu_blocks: 16,
                max_cpu_blocks: 0,
                enable_swapping: false,
                num_layers: cfg.num_layers,
                num_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        Arc::new(crate::registry::GreedySampler),
        kv,
        executor.clone(),
        Arc::new(crate::tensor_factory::CandleTensorFactory::new(Device::CPU)),
    )
    .unwrap();
    let mut identities = Vec::new();
    for &length in prompt_lengths {
        let mut request = InferenceRequest::new("preparation", "tiny-llama-test");
        request.sampling_params.max_tokens = 4;
        request
            .metadata
            .insert("ferrum_ignore_eos".into(), serde_json::json!(true));
        request.sampling_params.temperature = 0.0;
        request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.into(), serde_json::json!(length));
        let prompt = (0..length)
            .map(|i| TokenId((i + 3) as u32))
            .collect::<Vec<_>>();
        let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            prompt.clone(),
            Some(tokenizer.clone()),
            Some(cfg.vocab_size),
        );
        let TokenAdmissionOutcome::Admitted(frontier) = executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    &request.id,
                    &prompt,
                    state.model_maximum_sequence_tokens(),
                    prompt.len(),
                    0,
                )
                .unwrap(),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("real CPU admission must succeed");
        };
        state.execution_frontier = Some(frontier.clone());
        let mut slot = RequestSlotLease::open(&engine.inner, request.id.clone());
        slot.admit(&engine.inner);
        state.request_slot = Some(slot);
        engine
            .inner
            .sequences
            .write()
            .insert(request.id.clone(), state);
        identities.push((request.id.clone(), frontier));
        scheduler.submit(request).await.unwrap();
    }
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(prompt_lengths.len()))
        .await
        .unwrap();
    assert_eq!(batch.requests.len(), prompt_lengths.len());
    (engine, executor, batch, identities)
}

#[tokio::test]
async fn token_preparation_bad_tail_commits_healthy_cpu_peer_before_closing_failure() {
    let (engine, executor, batch, identities) = fixture(&[3, 4]).await;
    let (bad, bad_frontier) = &identities[0];
    let (healthy, healthy_frontier) = &identities[1];
    *executor.healthy_at_close.lock() = Some((Arc::downgrade(&engine.inner), healthy.clone()));
    let error = engine.inner.process_batch(&batch).await.unwrap_err();
    assert!(error
        .to_string()
        .contains("invalid prompt-tail capture boundary"));
    assert_eq!(
        *executor.executed.lock(),
        vec![healthy_frontier.sequence().clone()]
    );
    assert_eq!(
        *executor.cancelled.lock(),
        vec![bad_frontier.sequence().clone()]
    );
    assert_eq!(*executor.close_progress.lock(), vec![(4, 1)]);
    assert!(bad_frontier.sequence().observed_end().is_some());
    assert!(!engine.inner.sequences.read().contains_key(bad));
    assert!(healthy_frontier.sequence().observed_end().is_none());
    let next = engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .unwrap();
    assert_eq!(next.requests.len(), 1);
    assert_eq!(&next.requests[0].request.id, healthy);
    engine.inner.process_batch(&next).await.unwrap();
    {
        let states = engine.inner.sequences.read();
        let state = &states[healthy];
        assert_eq!(
            state.execution_frontier.as_ref().unwrap().consumed_tokens(),
            5
        );
        assert_eq!(state.generated_tokens.len(), 2);
    }
    *executor.healthy_at_close.lock() = None;
    engine
        .inner
        .complete_request_for_test(healthy, FinishReason::Error)
        .await
        .unwrap();
    assert!(healthy_frontier.sequence().observed_end().is_some());
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
}

#[tokio::test]
async fn token_preparation_failure_preserves_same_request_replacement_incarnation() {
    let (engine, executor, batch, identities) = fixture(&[3]).await;
    let PreparedTokenBatch {
        inputs,
        mut failures,
    } = engine.inner.prepare_token_batch(&batch);
    assert!(inputs.is_empty());
    assert_eq!(failures.len(), 1);
    let failure = failures.pop().unwrap();
    let (request_id, original) = &identities[0];
    // End the original model owner, then genuinely admit a new incarnation for
    // the same product ID before processing the captured old preparation error.
    let old_close = executor.inner.cancel_tokens(original.sequence()).unwrap();
    assert!(old_close.error.is_none());
    assert!(matches!(
        old_close.disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    let replacement = {
        let mut states = engine.inner.sequences.write();
        let state = states.get_mut(request_id).unwrap();
        let TokenAdmissionOutcome::Admitted(frontier) = executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    request_id,
                    &state.input_tokens,
                    state.model_maximum_sequence_tokens(),
                    state.input_tokens.len(),
                    0,
                )
                .unwrap(),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("replacement must genuinely admit");
        };
        assert_ne!(frontier.sequence(), original.sequence());
        state.execution_frontier = Some(frontier.clone());
        frontier
    };
    engine
        .inner
        .fail_token_frontier(
            &failure.request_id,
            failure.frontier.as_ref().unwrap(),
            failure.error,
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(
        *executor.cancelled.lock(),
        vec![original.sequence().clone()]
    );
    assert_eq!(
        engine.inner.sequences.read()[request_id]
            .execution_frontier
            .as_ref(),
        Some(&replacement)
    );
    assert!(replacement.sequence().observed_end().is_none());
    assert!(engine.inner.sequences.read()[request_id]
        .request_slot
        .is_some());
    executor.bad_hook.store(false, Ordering::Relaxed);
    engine.inner.process_batch(&batch).await.unwrap();
    {
        let states = engine.inner.sequences.read();
        let state = &states[request_id];
        assert_eq!(
            state.execution_frontier.as_ref().unwrap().consumed_tokens(),
            3
        );
        assert_eq!(state.generated_tokens.len(), 1);
    }
    engine
        .inner
        .complete_request_for_test(request_id, FinishReason::Error)
        .await
        .unwrap();
    assert!(replacement.sequence().observed_end().is_some());
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
}
