//! Real tiny CPU execution with a deliberately stale scheduled prefill range.
//! The replacement case exercises exact-incarnation protection, not a GPU path.

use super::*;
use ferrum_execution::model_executor::{
    ExecutorCapabilities, ExecutorPrefillAdmission, ExecutorStatus,
};
use ferrum_execution::token_execution::{
    ExecutionAdmission, ExecutionCloseDisposition, ExecutionEndReason, ExecutorSequence,
    TokenAdmissionOutcome,
};
use ferrum_kv::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_models::test_support::{tiny_llama_executor_owned, TinyLlamaConfig, TinyTokenizer};

struct StalePrefillExecutor {
    inner: Arc<ExecutorScope>,
    owner: ExecutorOwner,
    observation: Arc<StalePrefillObservation>,
}

struct StalePrefillObservation {
    inner: crate::continuous_engine::token_completion_tests::ScopedExecutorObservation,
    executed: Mutex<Vec<ExecutorSequence>>,
    cancelled: Mutex<Vec<ExecutorSequence>>,
    healthy_at_close: Mutex<Option<(std::sync::Weak<EngineInner>, RequestId)>>,
    close_progress: Mutex<Vec<(usize, usize)>>,
}

impl std::ops::Deref for StalePrefillExecutor {
    type Target = StalePrefillObservation;
    fn deref(&self) -> &Self::Target {
        &self.observation
    }
}

#[async_trait::async_trait]
impl ModelExecutor for StalePrefillExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::ExecutorShutdownPermit,
    ) -> ferrum_execution::ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ferrum_execution::ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        match Arc::try_unwrap(self) {
            Ok(Self {
                inner,
                owner,
                observation,
            }) => {
                drop((inner, observation));
                owner.begin_shutdown()
            }
            Err(original) => ferrum_execution::ExecutorShutdownObserver::retained(
                Arc::new(original),
                FerrumError::internal("preparation fixture has an unscoped owner"),
            ),
        }
    }
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
    fn prefill_shape_policy(&self) -> PrefillShapePolicy {
        self.inner.prefill_shape_policy()
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
            let states = &engine.sequences;
            let healthy_handle = states.get(request).unwrap();
            let healthy_guard = healthy_handle.lock();
            let healthy = healthy_guard.as_ref().unwrap();
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
    Arc<StalePrefillObservation>,
    ferrum_interfaces::BatchPlan,
    Vec<(RequestId, ExecutionFrontier)>,
) {
    let cfg = TinyLlamaConfig::default();
    let owner = ExecutorOwner::new(tiny_llama_executor_owned(&cfg));
    let inner = Arc::new(owner.scope().unwrap());
    let executor = Arc::new(StalePrefillObservation {
        inner: crate::continuous_engine::token_completion_tests::ScopedExecutorObservation::bound(
            &inner,
        ),
        executed: Mutex::new(Vec::new()),
        cancelled: Mutex::new(Vec::new()),
        healthy_at_close: Mutex::new(None),
        close_progress: Mutex::new(Vec::new()),
    });
    let implementation = StalePrefillExecutor {
        inner,
        owner,
        observation: Arc::clone(&executor),
    };
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
        ExecutorOwner::new(Box::new(implementation)),
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
        let TokenAdmissionOutcome::Admitted(frontier) = engine
            .active_inner()
            .unwrap()
            .model_executor
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
        let mut slot = RequestSlotLease::open(
            &engine.active_inner().expect("engine execution is open"),
            request.id.clone(),
        );
        slot.admit(&engine.active_inner().expect("engine execution is open"));
        state.request_slot = Some(slot);
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
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
async fn token_preparation_stale_range_commits_healthy_cpu_peer_before_closing_failure() {
    let (engine, executor, mut batch, identities) = fixture(&[3, 4]).await;
    let (bad, bad_frontier) = &identities[0];
    let (healthy, healthy_frontier) = &identities[1];
    batch
        .requests
        .iter_mut()
        .find(|request| &request.request.id == bad)
        .unwrap()
        .tokens_processed = 1;
    *executor.healthy_at_close.lock() = Some((
        Arc::downgrade(&engine.active_inner().expect("engine execution is open")),
        healthy.clone(),
    ));
    let error = engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("scheduler prefill position differs from the actual token frontier"));
    assert_eq!(
        *executor.executed.lock(),
        vec![healthy_frontier.sequence().clone()]
    );
    // Closing is independently managed; healthy numerical publication above
    // remains synchronous and precedes the actual close invocation.
    engine
        .active_inner()
        .expect("engine execution is open")
        .execution_closes
        .join_observers()
        .await;
    engine
        .active_inner()
        .expect("engine execution is open")
        .output_tasks
        .join_current()
        .await;
    assert_eq!(
        *executor.cancelled.lock(),
        vec![bad_frontier.sequence().clone()]
    );
    assert_eq!(*executor.close_progress.lock(), vec![(4, 1)]);
    assert!(bad_frontier.sequence().observed_end().is_some());
    assert!(!engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(bad)
        .is_some());
    assert!(healthy_frontier.sequence().observed_end().is_none());
    let next = engine
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .unwrap();
    assert_eq!(next.requests.len(), 1);
    assert_eq!(&next.requests[0].request.id, healthy);
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&next)
        .await
        .unwrap();
    {
        let states = &engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        let state_handle = states.get(healthy).unwrap();
        let state_guard = state_handle.lock();
        let state = state_guard.as_ref().unwrap();
        assert_eq!(
            state.execution_frontier.as_ref().unwrap().consumed_tokens(),
            5
        );
        assert_eq!(state.generated_tokens.len(), 2);
    }
    *executor.healthy_at_close.lock() = None;
    engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_for_test(healthy, FinishReason::Error)
        .await
        .unwrap();
    assert!(healthy_frontier.sequence().observed_end().is_some());
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        0
    );
}

#[tokio::test]
async fn token_preparation_rejects_unprojected_priced_range_with_original_frontier() {
    let (engine, executor, batch, identities) = fixture(&[3]).await;
    let inner = engine.active_inner().unwrap();
    let error = inner
        .process_batch_with_shape(
            &batch,
            PrefillShapePolicy::PromptTail(
                ferrum_runtime::CheckpointBoundaryConstraint::any_positive(),
            ),
            true,
        )
        .await
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("selected prefill range differs from its projected execution shape"));
    assert!(executor.executed.lock().is_empty());
    inner.execution_closes.join_observers().await;
    inner.output_tasks.join_current().await;
    assert_eq!(
        *executor.cancelled.lock(),
        [identities[0].1.sequence().clone()]
    );
    assert!(identities[0].1.sequence().observed_end().is_some());
    assert_eq!(inner.execution_closes.unreclaimed_count(), 0);
    drop(inner);
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
}

#[tokio::test]
async fn token_preparation_failure_preserves_same_request_replacement_incarnation() {
    let (engine, executor, mut batch, identities) = fixture(&[3]).await;
    batch.requests[0].tokens_processed = 1;
    let PreparedTokenBatch {
        inputs,
        mut failures,
    } = engine
        .active_inner()
        .expect("engine execution is open")
        .prepare_token_batch(&batch, PrefillShapePolicy::Identity, false);
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
    let state_handle = engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(request_id)
        .unwrap();
    let (input_tokens, maximum_sequence_tokens) = {
        let state_guard = state_handle.lock();
        let state = state_guard.as_ref().unwrap();
        (
            state.input_tokens.clone(),
            state.model_maximum_sequence_tokens(),
        )
    };
    let replacement = {
        let TokenAdmissionOutcome::Admitted(frontier) = engine
            .active_inner()
            .unwrap()
            .model_executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    request_id,
                    &input_tokens,
                    maximum_sequence_tokens,
                    input_tokens.len(),
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
        let mut state_guard = state_handle.lock();
        let state = state_guard.as_mut().unwrap();
        state.execution_frontier = Some(frontier.clone());
        frontier
    };
    engine
        .active_inner()
        .expect("engine execution is open")
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
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .get(request_id)
            .unwrap()
            .lock()
            .as_ref()
            .unwrap()
            .execution_frontier
            .as_ref(),
        Some(&replacement)
    );
    assert!(replacement.sequence().observed_end().is_none());
    assert!(engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(request_id)
        .unwrap()
        .lock()
        .as_ref()
        .unwrap()
        .request_slot
        .is_some());
    batch.requests[0].tokens_processed = 0;
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap();
    {
        let states = &engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        let state_handle = states.get(request_id).unwrap();
        let state_guard = state_handle.lock();
        let state = state_guard.as_ref().unwrap();
        assert_eq!(
            state.execution_frontier.as_ref().unwrap().consumed_tokens(),
            3
        );
        assert_eq!(state.generated_tokens.len(), 1);
    }
    engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_for_test(request_id, FinishReason::Error)
        .await
        .unwrap();
    assert!(replacement.sequence().observed_end().is_some());
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        0
    );
}
