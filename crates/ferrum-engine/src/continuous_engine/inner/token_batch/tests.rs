//! Protocol-consumer tests: issuers create genuine opaque identities and Held
//! observations. No model forward or GPU numerical qualification is claimed.

use super::*;
use ferrum_execution::model_executor::{
    DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus, PrefillInput, PrefillOutput,
};
use ferrum_execution::token_execution::{
    ExecutedTokenWork, ExecutionCompletion, ExecutionProduct, ExecutionProgress,
    ExecutorSequenceIssuer, LastKnownExecutionStage, NoSubmissionStage,
};
use ferrum_kv::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_models::test_support::{tiny_llama_executor, TinyLlamaConfig, TinyTokenizer};

struct ProtocolExecutor {
    metadata: Arc<dyn ModelExecutor>,
    issuer: ExecutorSequenceIssuer,
    closed: Mutex<Vec<ExecutorSequence>>,
}

impl ProtocolExecutor {
    fn close(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.closed.lock().push(sequence.clone());
        Ok(ExecutionCloseOutcome::ended(self.issuer.ended(sequence)?))
    }
}

#[async_trait::async_trait]
impl ModelExecutor for ProtocolExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.metadata.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.metadata.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.metadata.status()
    }
    async fn prefill(&self, _: &PrefillInput) -> Result<PrefillOutput> {
        panic!("protocol test does not execute a model")
    }
    async fn decode(&self, _: &DecodeInput) -> Result<DecodeOutput> {
        panic!("protocol test does not execute a model")
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        _: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.close(sequence)
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.close(sequence)
    }
}

async fn fixture() -> (
    ContinuousBatchEngine,
    Arc<ProtocolExecutor>,
    Vec<TokenBatchInput>,
) {
    fixture_with_request_count(2).await
}

async fn fixture_with_request_count(
    request_count: usize,
) -> (
    ContinuousBatchEngine,
    Arc<ProtocolExecutor>,
    Vec<TokenBatchInput>,
) {
    let cfg = TinyLlamaConfig::default();
    let executor = Arc::new(ProtocolExecutor {
        metadata: tiny_llama_executor(&cfg),
        issuer: ExecutorSequenceIssuer::new(),
        closed: Mutex::new(Vec::new()),
    });
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer = Arc::new(TinyTokenizer::new(cfg.vocab_size));
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
    let mut inputs = Vec::new();
    for _ in 0..request_count {
        let mut request = InferenceRequest::new("protocol", "tiny-llama-test");
        request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.into(), serde_json::json!(2));
        let request_id = request.id.clone();
        let identity = executor.issuer.issue_sequence().unwrap();
        let frontier = executor.issuer.frontier(&identity, 0).unwrap();
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            vec![TokenId(3), TokenId(4)],
            Some(tokenizer.clone()),
            Some(cfg.vocab_size),
        );
        sequence.execution_frontier = Some(frontier.clone());
        sequence.phase = RequestPhase::Prefilling;
        engine
            .inner
            .sequences
            .write()
            .insert(request_id.clone(), sequence);
        scheduler.submit(request).await.unwrap();
        inputs.push(TokenBatchInput {
            request_id,
            frontier,
            work: TokenWork::Prefill(PrefillChunk::new(0, 1, 2).unwrap()),
            logits_policy: LogitsReturnPolicy::FullLogits,
            scheduled_tokens: 1,
            context_tokens: 2,
        });
    }
    let scheduled = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(request_count))
        .await
        .unwrap();
    assert_eq!(scheduled.requests.len(), request_count);
    (engine, executor, inputs)
}

fn completions(
    scope: &ferrum_execution::token_execution::ExecutionBatch<'_>,
) -> Vec<ExecutionCompletion> {
    (0..2)
        .map(|index| {
            scope
                .completion(
                    index,
                    ExecutedTokenWork::Prefill {
                        chunk: PrefillChunk::new(0, 1, 2).unwrap(),
                        capacity_probe_count: 1,
                    },
                    ExecutionProduct::Intermediate,
                )
                .unwrap()
        })
        .collect()
}

fn held(
    issuer: &ExecutorSequenceIssuer,
    sequence: &ExecutorSequence,
) -> (
    ContinuationDisposition,
    tokio::sync::oneshot::Sender<
        Result<ferrum_execution::token_execution::ExecutionTerminalOutcome>,
    >,
) {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let held = issuer
        .held(issuer.recovery_key(sequence).unwrap(), async move {
            receiver
                .await
                .map_err(|_| FerrumError::internal("protocol observer lost"))?
        })
        .unwrap();
    (ContinuationDisposition::Held(held), sender)
}

fn assert_healthy(engine: &ContinuousBatchEngine, input: &TokenBatchInput) {
    let sequences = engine.inner.sequences.read();
    let sequence = &sequences[&input.request_id];
    let frontier = sequence.execution_frontier.as_ref().unwrap();
    assert_eq!(frontier.sequence(), input.frontier.sequence());
    assert_eq!(frontier.consumed_tokens(), 1);
    assert_eq!(sequence.prefill_tokens_processed, 1);
    assert!(sequence.execution_close.is_none());
    assert!(frontier.sequence().observed_end().is_none());
}

#[tokio::test]
async fn token_outcome_wrong_index_held_conflicts_only_with_its_actual_incarnation() {
    for wrong_index in [usize::MAX, 1] {
        let (engine, executor, inputs) = fixture().await;
        let selected = inputs.iter().collect::<Vec<_>>();
        let works = inputs
            .iter()
            .map(TokenBatchInput::as_work)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let scope = executor
            .issuer
            .batch(&works, executor.info().vocab_size)
            .unwrap();
        let (continuation, sender) = held(&executor.issuer, inputs[0].frontier.sequence());
        // Public validation rejects this index/identity pair and preserves the
        // real Held, exactly as a broken producer's cold outcome does.
        let rejected = scope
            .participant(
                wrong_index,
                ExecutionProgress::Unknown {
                    last_known: LastKnownExecutionStage::Submitted,
                    known_executed_range: None,
                },
                continuation,
                0,
            )
            .unwrap_err();
        let outcome = rejected.into_outcome((
            completions(&scope),
            Vec::new(),
            vec![FerrumError::backend("conflicting actual incarnation")],
        ));
        assert!(engine
            .inner
            .consume_token_outcome(&selected, &works, outcome, None)
            .await
            .is_err());
        engine.inner.execution_closes.join_observers().await;
        engine.inner.drain_execution_closes().unwrap();
        assert!(!engine
            .inner
            .sequences
            .read()
            .contains_key(&inputs[0].request_id));
        assert_healthy(&engine, &inputs[1]);
        assert_eq!(
            executor.closed.lock().as_slice(),
            &[inputs[0].frontier.sequence().clone()]
        );
        assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
        assert!(
            !sender.is_closed(),
            "invalid index must not discard the actual Held"
        );
        drop(engine);
        assert!(sender.is_closed());
    }
}

#[tokio::test]
async fn token_outcome_foreign_held_does_not_poison_the_declared_healthy_peer() {
    let (engine, executor, inputs) = fixture().await;
    let selected = inputs.iter().collect::<Vec<_>>();
    let works = inputs
        .iter()
        .map(TokenBatchInput::as_work)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let scope = executor
        .issuer
        .batch(&works, executor.info().vocab_size)
        .unwrap();
    let foreign = executor.issuer.issue_sequence().unwrap();
    let (continuation, sender) = held(&executor.issuer, &foreign);
    let rejected = scope
        .participant(
            1,
            ExecutionProgress::Unknown {
                last_known: LastKnownExecutionStage::Submitted,
                known_executed_range: None,
            },
            continuation,
            0,
        )
        .unwrap_err();
    let outcome = rejected.into_outcome((
        completions(&scope),
        Vec::new(),
        vec![FerrumError::backend("foreign producer observation")],
    ));
    engine
        .inner
        .consume_token_outcome(&selected, &works, outcome, None)
        .await
        .unwrap();
    for input in &inputs {
        assert_healthy(&engine, input);
    }
    assert!(executor.closed.lock().is_empty());
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
    assert!(!sender.is_closed());
    assert!(foreign.observed_end().is_none());
    drop(engine);
    assert!(sender.is_closed());
}

#[tokio::test]
async fn token_outcome_foreign_completion_does_not_claim_a_healthy_index() {
    let (engine, executor, inputs) = fixture().await;
    let selected = inputs.iter().collect::<Vec<_>>();
    let works = inputs
        .iter()
        .map(TokenBatchInput::as_work)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let scope = executor
        .issuer
        .batch(&works, executor.info().vocab_size)
        .unwrap();
    let foreign = executor.issuer.issue_sequence().unwrap();
    let foreign_frontier = executor.issuer.frontier(&foreign, 0).unwrap();
    let foreign_works = [
        inputs[0].as_work().unwrap(),
        ExecutionWork::new(&foreign_frontier, inputs[1].work, &inputs[1].logits_policy).unwrap(),
    ];
    let foreign_scope = executor
        .issuer
        .batch(&foreign_works, executor.info().vocab_size)
        .unwrap();
    let mut completed = completions(&scope);
    completed.push(
        foreign_scope
            .completion(
                1,
                ExecutedTokenWork::Prefill {
                    chunk: PrefillChunk::new(0, 1, 2).unwrap(),
                    capacity_probe_count: 1,
                },
                ExecutionProduct::Intermediate,
            )
            .unwrap(),
    );
    let outcome = ExecutionBatchOutcome::ContractViolation(ExecutionContractViolation::new(
        FerrumError::internal("foreign completion"),
        (completed, Vec::new(), Vec::new()),
    ));
    engine
        .inner
        .consume_token_outcome(&selected, &works, outcome, None)
        .await
        .unwrap();
    for input in &inputs {
        assert_healthy(&engine, input);
    }
    assert!(executor.closed.lock().is_empty());
}

#[tokio::test]
async fn token_outcome_foreign_retryable_preserves_another_live_split_request() {
    // All three requests are actually published and selected by the real
    // scheduler. The third remains pending while the first split is consumed;
    // it is foreign only to this two-member executor call.
    let (engine, executor, inputs) = fixture_with_request_count(3).await;
    let selected = inputs[..2].iter().collect::<Vec<_>>();
    let works = selected
        .iter()
        .map(|input| input.as_work())
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let scope = executor
        .issuer
        .batch(&works, executor.info().vocab_size)
        .unwrap();
    let pending = &inputs[2];
    let rejected = scope
        .participant(
            1,
            ExecutionProgress::NotSubmitted {
                stage: NoSubmissionStage::Preparation,
            },
            ContinuationDisposition::Retryable(pending.frontier.clone()),
            0,
        )
        .unwrap_err();
    let outcome = rejected.into_outcome((
        completions(&scope),
        Vec::new(),
        vec![FerrumError::backend(
            "retryable belongs to another live split",
        )],
    ));
    engine
        .inner
        .consume_token_outcome(&selected, &works, outcome, None)
        .await
        .unwrap();

    for input in &inputs[..2] {
        assert_healthy(&engine, input);
    }
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&pending.request_id];
        assert_eq!(
            sequence.execution_frontier.as_ref(),
            Some(&pending.frontier)
        );
        assert_eq!(sequence.prefill_tokens_processed, 0);
        assert!(!sequence.prefill_complete);
        assert_eq!(sequence.phase, RequestPhase::Prefilling);
        assert!(sequence.execution_close.is_none());
    }
    assert!(pending.frontier.sequence().observed_end().is_none());
    assert!(executor.closed.lock().is_empty());
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);

    // The original pending frontier still participates in its own legal
    // product commit; no cancellation or invented Ended was applied to it.
    let pending_work = [pending.as_work().unwrap()];
    let pending_scope = executor
        .issuer
        .batch(&pending_work, executor.info().vocab_size)
        .unwrap();
    let completion = pending_scope
        .completion(
            0,
            ExecutedTokenWork::Prefill {
                chunk: PrefillChunk::new(0, 1, 2).unwrap(),
                capacity_probe_count: 1,
            },
            ExecutionProduct::Intermediate,
        )
        .unwrap();
    let outcome = pending_scope.completed(vec![completion]).unwrap();
    engine
        .inner
        .consume_token_outcome(&[pending], &pending_work, outcome, None)
        .await
        .unwrap();
    assert_healthy(&engine, pending);
    assert!(executor.closed.lock().is_empty());
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
}
