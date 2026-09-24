//! Real engine leaves with a deterministic, explicitly observed executor.
use super::*;
use crate::continuous_engine::{ContinuousBatchEngine, EngineInner, SequenceState};
use ferrum_interfaces::{
    engine::InferenceEngine, model_executor::*, scheduler::Scheduler, ModelExecutor,
};
use ferrum_scheduler::implementations::continuous::{
    cost_model as model, ContinuousBatchScheduler,
};
use ferrum_testkit::{MockKvCacheHandle, MockModelExecutor, MockTensorFactory};
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::{EngineConfig, FerrumError, InferenceRequest, Result, SamplingParams, TokenId};
use parking_lot::Mutex;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Weak,
};

struct LeafExecutor {
    base: MockModelExecutor,
    physical_calls: AtomicU64,
    observed_calls: AtomicU64,
    fail: AtomicBool,
    cancel: Mutex<Option<Weak<EngineInner>>>,
    shapes: Mutex<Vec<ActualWaveShape>>,
}
fn known_identity() -> ExecutorCostIdentityAvailability {
    ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }))
}
impl LeafExecutor {
    fn execute_prefill(
        &self,
        input: &PlanRuntimePrefillInput,
    ) -> Result<PlanRuntimePrefillOutcome> {
        self.physical_calls.fetch_add(1, Ordering::Relaxed);
        if self.fail.load(Ordering::Relaxed) {
            return Err(FerrumError::backend("submitted device failure"));
        }
        // Actual capacity narrowing differs from the engine's requested shape.
        let chunk = PrefillChunk::new(
            input.chunk.tokens_processed(),
            input.chunk.tokens_to_process().min(2),
            input.input_tokens.len(),
        )?;
        let cache = Arc::new(MockKvCacheHandle::new(
            input.request_id.clone(),
            1,
            chunk.end(),
        ));
        let output = if chunk.is_final() {
            let mut logits = vec![0.0; 8];
            logits[4] = 1.0;
            PlanRuntimePrefillOutput::final_logits(
                input.request_id.clone(),
                chunk.end(),
                logits,
                cache,
            )?
        } else {
            PlanRuntimePrefillOutput::intermediate(input.request_id.clone(), chunk.end(), cache)
        };
        if let Some(engine) = self.cancel.lock().take().and_then(|weak| weak.upgrade()) {
            engine.sequences.write().remove(&input.request_id);
        }
        Ok(PlanRuntimePrefillOutcome::Completed(
            PlanRuntimePrefillCompletion::new(
                output,
                input.chunk,
                chunk,
                u32::from(chunk != input.chunk),
            )?,
        ))
    }
}
#[async_trait::async_trait]
impl ModelExecutor for LeafExecutor {
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
    fn execution_cost_identity(&self) -> ExecutorCostIdentityAvailability {
        known_identity()
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
        self.execute_prefill(input)
    }
    async fn plan_runtime_prefill_with_capacity_observed(
        &self,
        input: &PlanRuntimePrefillInput,
        context: &mut PlanRuntimeCostObservationContext<'_>,
    ) -> ObservedDispatch<PlanRuntimePrefillOutcome> {
        self.observed_calls.fetch_add(1, Ordering::Relaxed);
        let row = context.participant(&input.request_id).unwrap();
        let shape = ActualWaveShape {
            statistical_evidence: None,
            kind: ActualWaveKind::Prefill,
            path: ActualWavePath::PlanRuntime,
            graph: ActualWaveGraphState::Disabled,
            row_order: ActualWaveRowOrder::Ordered,
            provider_signature: [5; 32],
            output_policy_signature: row.output_policy_signature.unwrap(),
            numeric_features: None,
            host_content_features: None,
            row_multiset_features: None,
            rows: vec![ActualWaveRow {
                request_id: input.request_id.clone(),
                owner_incarnation: row.owner_incarnation,
                work_generation: row.work_generation,
                input_index: row.input_index,
                work: ActualRowWork::Prefill {
                    offset: input.chunk.tokens_processed() as u32,
                    count: input.chunk.tokens_to_process().min(2) as u32,
                    total_prompt_tokens: input.input_tokens.len() as u32,
                },
            }],
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        };
        self.shapes.lock().push(shape.clone());
        context.physical_wave(Ok(shape), context.now_ns());
        let result = self.execute_prefill(input);
        context.terminal(
            if result.is_ok() {
                ActualWaveOutcome::Completed
            } else {
                ActualWaveOutcome::FailedAfterSubmit
            },
            None,
        );
        context.finish_call(if result.is_ok() {
            ObservedCallOutcome::Completed
        } else {
            ObservedCallOutcome::Failed
        });
        ObservedDispatch::Executed(result)
    }
}
async fn fixture(observe: bool) -> (ContinuousBatchEngine, Arc<LeafExecutor>) {
    let vocab: tokenizers::models::bpe::Vocab = ["a", "b", "c", "d", "e", "f", "g", "h"]
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t.to_owned(), i as u32))
        .collect();
    let mut raw = tokenizers::Tokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap(),
    );
    raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    let tokenizer = Arc::new(HuggingFaceTokenizer::new(raw).await.unwrap());
    let executor = Arc::new(LeafExecutor {
        base: MockModelExecutor::instant(8),
        physical_calls: AtomicU64::new(0),
        observed_calls: AtomicU64::new(0),
        fail: AtomicBool::new(false),
        cancel: Mutex::new(None),
        shapes: Mutex::new(Vec::new()),
    });
    let mut config = EngineConfig::default();
    if observe {
        config.scheduler.slo.mode = ferrum_types::SloMode::Observe;
        config.scheduler.slo.default_service_class = Some("test".into());
        config
            .scheduler
            .slo
            .services
            .push(ferrum_types::ServiceSloConfig {
                id: "test".into(),
                server_token_commit: ferrum_types::SloLatencyBudgets {
                    ttft_ms: NonZeroU64::new(1000).unwrap(),
                    tpot_ms: NonZeroU64::new(100).unwrap(),
                    itl_ms: NonZeroU64::new(100).unwrap(),
                },
                client_visible: None,
                attainment: Default::default(),
            });
    }
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    assert_eq!(engine.inner.cost_runtime.is_some(), observe);
    (engine, executor)
}
async fn install(
    engine: &ContinuousBatchEngine,
) -> (
    ferrum_interfaces::BatchPlan,
    RequestId,
    tokio::sync::oneshot::Receiver<Result<ferrum_types::InferenceResponse>>,
) {
    let mut request = InferenceRequest::new("abcd", "test");
    request.sampling_params = SamplingParams {
        max_tokens: 16,
        ..SamplingParams::greedy()
    };
    request.metadata.insert(
        crate::continuous_engine::PROMPT_TOKENS_METADATA_KEY.to_owned(),
        serde_json::json!(4),
    );
    let id = request.id.clone();
    engine
        .inner
        .scheduler
        .submit(request.clone())
        .await
        .unwrap();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0); 4],
        Some(engine.inner.tokenizer.clone()),
        Some(8),
    );
    let (tx, rx) = tokio::sync::oneshot::channel();
    sequence.response_sender = Some(tx);
    engine.inner.initialize_sequence_cost(&mut sequence);
    if engine.inner.cost_runtime.is_some() {
        assert!(sequence.cost_frontier.is_some());
        assert!(sequence.cost_policy_signature.is_some());
    }
    engine.inner.sequences.write().insert(id.clone(), sequence);
    let mut batch = engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    batch.requests.retain(|request| request.request.id == id);
    assert_eq!(batch.requests.len(), 1);
    batch.requests[0].tokens_to_process = Some(4);
    (batch, id, rx)
}

#[tokio::test]
async fn actual_narrowed_leaf_publishes_host_commit_and_trains_without_replay() {
    let (engine, executor) = fixture(true).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap();
    // This tests exact real-leaf evidence and coverage, not the allowed
    // try-lock loss policy. Keep trainer pops from racing the calibration set.
    let (held, waiting) = std::sync::mpsc::channel();
    let (release, resume) = std::sync::mpsc::channel();
    let paused = runtime.clone();
    let holder = std::thread::spawn(move || {
        paused.with_training_paused(|| {
            held.send(()).unwrap();
            let _ = resume.recv();
        });
    });
    waiting
        .recv_timeout(std::time::Duration::from_secs(3))
        .unwrap();
    let n = model::CostModelSettings::default().min_samples.get();
    for _ in 0..n {
        let (batch, id, _receiver) = install(&engine).await;
        engine.inner.process_batch(&batch).await.unwrap();
        {
            let sequences = engine.inner.sequences.read();
            let sequence = &sequences[&id];
            assert_eq!(sequence.prefill_tokens_processed, 2);
            assert!(sequence.generated_tokens.is_empty());
            assert_eq!(sequence.cost_frontier.unwrap().work_generation.get(), 2);
        }
        // Remove this unfinished request before the next identical calibration
        // request; its distinct identity never becomes part of the shape key.
        engine.inner.scheduler.cancel(id.clone()).await.unwrap();
        engine.inner.sequences.write().remove(&id);
    }
    assert_eq!(executor.physical_calls.load(Ordering::Relaxed), n as u64);
    assert_eq!(executor.observed_calls.load(Ordering::Relaxed), n as u64);
    release.send(()).unwrap();
    holder.join().unwrap();
    // Shutdown joins the real background trainer after all producers stop.
    // Training completion is deliberately not synchronous with process_batch.
    engine.shutdown().await.unwrap();
    assert_eq!(runtime.trained_samples(), n as u64);
    assert_eq!(runtime.sink.stats().published, n as u64);
    let shape = sample::scheduler_shape(&executor.shapes.lock()[0]).unwrap();
    let snapshot = runtime.snapshot().unwrap();
    assert!(matches!(
        snapshot.predict(
            snapshot.fingerprint(),
            &shape,
            model::CostBoundary::PreparationToCommit,
            runtime.clock.now_ns().unwrap()
        ),
        model::CostPrediction::Known(_)
    ));
}

#[tokio::test]
async fn off_and_observe_execute_the_same_actual_frontier() {
    for observe in [false, true] {
        let (engine, executor) = fixture(observe).await;
        let (batch, id, _receiver) = install(&engine).await;
        engine.inner.process_batch(&batch).await.unwrap();
        assert_eq!(executor.physical_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            executor.observed_calls.load(Ordering::Relaxed),
            u64::from(observe)
        );
        let sequences = engine.inner.sequences.read();
        assert_eq!(sequences[&id].prefill_tokens_processed, 2);
        assert!(sequences[&id].generated_tokens.is_empty());
    }
}

#[tokio::test]
async fn actual_inference_progresses_while_the_cost_trainer_is_blocked() {
    let (engine, executor) = fixture(true).await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let (held, waiting) = std::sync::mpsc::channel();
    let (release, resume) = std::sync::mpsc::channel();
    let paused = runtime.clone();
    let holder = std::thread::spawn(move || {
        paused.with_training_paused(|| {
            held.send(()).unwrap();
            // Dropping release on a failed assertion also unblocks this guard.
            let _ = resume.recv();
        });
    });
    waiting
        .recv_timeout(std::time::Duration::from_secs(3))
        .unwrap();
    let (batch, id, _receiver) = install(&engine).await;
    tokio::time::timeout(
        std::time::Duration::from_secs(3),
        engine.inner.process_batch(&batch),
    )
    .await
    .expect("inference waited for cost training")
    .unwrap();
    assert_eq!(
        engine.inner.sequences.read()[&id].prefill_tokens_processed,
        2
    );
    assert_eq!(executor.physical_calls.load(Ordering::Relaxed), 1);
    assert_eq!(runtime.trained_samples(), 0);
    assert_eq!(runtime.sink.stats().published, 1);
    release.send(()).unwrap();
    holder.join().unwrap();
    engine.inner.scheduler.cancel(id.clone()).await.unwrap();
    engine.inner.sequences.write().remove(&id);
    engine.shutdown().await.unwrap();
    assert_eq!(runtime.trained_samples(), 1);
    assert_eq!(runtime.sink.stats().drained, 1);
}

#[tokio::test]
async fn submitted_failure_and_cancelled_host_never_train_or_replay() {
    for cancelled in [false, true] {
        let (engine, executor) = fixture(true).await;
        let (batch, _id, _receiver) = install(&engine).await;
        if cancelled {
            *executor.cancel.lock() = Some(Arc::downgrade(&engine.inner));
        } else {
            executor.fail.store(true, Ordering::Relaxed);
        }
        engine.inner.process_batch(&batch).await.unwrap();
        assert_eq!(executor.physical_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.observed_calls.load(Ordering::Relaxed), 1);
        let runtime = engine.inner.cost_runtime.as_ref().unwrap();
        assert_eq!(runtime.trained_samples(), 0);
        assert_eq!(runtime.sink.stats().published, 0);
        assert!(runtime.snapshot().is_none());
        assert_eq!(
            runtime.sink.stats().rejected(if cancelled {
                CostCallRejection::HostCancelled
            } else {
                CostCallRejection::ExecutorFailed
            }),
            1
        );
    }
}

#[tokio::test]
async fn same_kv_budget_keeps_actual_host_history_lengths_separate() {
    let (engine, _) = fixture(true).await;
    let (_batch, id, _receiver) = install(&engine).await;
    let capture = || {
        let mut preparation = engine.inner.prepare_cost_observation().unwrap();
        preparation.capture(&engine.inner.sequences.read()[&id]);
        let call = preparation.begin().unwrap();
        call.participants[0].output_policy_signature
    };
    let empty_history = capture().unwrap();
    let cached_policy = engine.inner.sequences.read()[&id].cost_policy_signature;
    {
        let mut sequences = engine.inner.sequences.write();
        sequences
            .get_mut(&id)
            .unwrap()
            .generated_tokens
            .push(TokenId::new(4));
    }
    let one_token = capture().unwrap();
    assert_ne!(empty_history, one_token);
    assert_eq!(
        engine.inner.sequences.read()[&id].cost_policy_signature,
        cached_policy
    );
    // The random realization is not an identity: length and installed policy
    // remain the same, and token-dependent cost remains measured variability.
    engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .generated_tokens[0] = TokenId::new(5);
    assert_eq!(capture(), Some(one_token));
    engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .cost_policy_signature = None;
    assert_eq!(
        capture(),
        None,
        "history cannot manufacture an unknown policy"
    );
}
