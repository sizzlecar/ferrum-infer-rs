//! Actual CPU model closure behind the product completion path. The only
//! injected behavior is close-observation scheduling or a post-close error;
//! admission, full-vocabulary forward, KV and terminal identity stay real.

use super::*;
use ferrum_execution::model_executor::{
    ExecutorCapabilities, ExecutorStatus, KvSlotRequest, PrefillChunk,
};
use ferrum_execution::token_execution::*;
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kv::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_models::common::llm::{
    ModelKvPreparation, ModelKvReservation, ModelTokenInput, ModelTokenOutcome, ModelTokenProduct,
};
use ferrum_models::models::LlamaFamilyModel;
use ferrum_models::test_support::{
    tiny_llama_model, tiny_model_info, TinyLlamaConfig, TinyTokenizer,
};
use ferrum_models::{DecoderOnlyLLM, LlmExecutor, LlmRuntimeConfig};
use std::sync::atomic::AtomicUsize;

#[path = "completion_failure_tests.rs"]
mod failure;

pub(super) struct ModelObservation {
    pub(super) model: LlamaFamilyModel<CpuBackend>,
    pub(super) releases: Vec<Vec<usize>>,
}

struct ObservedModel {
    state: Arc<Mutex<ModelObservation>>,
    config: LlmRuntimeConfig,
}

impl DecoderOnlyLLM for ObservedModel {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }
    fn supports_direct_token_execution(&self) -> bool {
        true
    }
    fn prepare_token_kv(&mut self, request: ModelKvPreparation<'_>) -> Result<ModelKvReservation> {
        self.state.lock().model.prepare_token_kv(request)
    }
    fn reserve_kv_slots(
        &mut self,
        requests: &[KvSlotRequest],
    ) -> Result<Option<ferrum_execution::model_executor::KvSlotReservation>> {
        self.state.lock().model.reserve_kv_slots(requests)
    }
    fn execute_token_work(
        &mut self,
        input: &ModelTokenInput<'_>,
    ) -> ModelTokenOutcome<ModelTokenProduct> {
        self.state.lock().model.execute_token_work(input)
    }
    fn close_token_sequence(&mut self, id: &str) -> Result<()> {
        let mut state = self.state.lock();
        let lengths = state
            .model
            .kv_caches
            .get(id)
            .expect("actual model KV before close")
            .iter()
            .map(|layer| layer.len)
            .collect();
        state.model.close_token_sequence(id)?;
        state.releases.push(lengths);
        Ok(())
    }
    fn release(&mut self, id: &str) {
        self.state.lock().model.release(id);
    }
    fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
        panic!("test must use the direct token core")
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("test must use the direct token core")
    }
}

pub(super) struct CloseGate {
    inner: Arc<dyn ModelExecutor>,
    gate: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    publication_error: bool,
    fail_confirmation: AtomicBool,
    pub(super) fail_cancellation: AtomicBool,
    finishes: AtomicUsize,
    cancels: AtomicUsize,
}

#[async_trait::async_trait]
impl ModelExecutor for CloseGate {
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
    async fn execute_tokens(&self, input: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        self.inner.execute_tokens(input).await
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.finishes.fetch_add(1, Ordering::SeqCst);
        if self.fail_confirmation.swap(false, Ordering::SeqCst) {
            return Err(FerrumError::backend(
                "injected terminal confirmation failure",
            ));
        }
        if let Some(started) = self.started.lock().take() {
            let _ = started.send(());
        }
        let gate = self.gate.lock().take();
        if let Some(gate) = gate {
            gate.await
                .map_err(|_| FerrumError::internal("close gate lost"))?;
        }
        let outcome = self.inner.finish_tokens(sequence, reason).await?;
        Ok(if self.publication_error {
            outcome.with_error(FerrumError::backend(
                "actual close completed but publication failed",
            ))
        } else {
            outcome
        })
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.cancels.fetch_add(1, Ordering::SeqCst);
        if self.fail_cancellation.load(Ordering::SeqCst) {
            return Err(FerrumError::backend(
                "injected cancellation without terminal proof",
            ));
        }
        self.inner.cancel_tokens(sequence)
    }
}

pub(super) struct Fixture {
    pub(super) engine: ContinuousBatchEngine,
    pub(super) executor: Arc<CloseGate>,
    pub(super) model: Arc<Mutex<ModelObservation>>,
    pub(super) identity: ExecutorSequence,
    pub(super) request_id: RequestId,
    pub(super) response: tokio::sync::oneshot::Receiver<Result<InferenceResponse>>,
    pub(super) stream: OutputReceiver,
    pub(super) started: tokio::sync::oneshot::Receiver<()>,
    pub(super) gate: Option<tokio::sync::oneshot::Sender<()>>,
    pub(super) trace: tempfile::TempDir,
    pub(super) prompt_len: usize,
    pub(super) layers: usize,
}

pub(super) fn observed_cpu_model() -> (
    Arc<LlmExecutor>,
    Arc<Mutex<ModelObservation>>,
    TinyLlamaConfig,
) {
    let cfg = TinyLlamaConfig::default();
    let model = tiny_llama_model(&cfg);
    let runtime_config = model.config().clone();
    let model = Arc::new(Mutex::new(ModelObservation {
        model,
        releases: Vec::new(),
    }));
    let inner = Arc::new(LlmExecutor::new(
        Box::new(ObservedModel {
            state: Arc::clone(&model),
            config: runtime_config,
        }),
        tiny_model_info(&cfg),
    ));
    (inner, model, cfg)
}

pub(super) async fn fixture(delayed: bool, publication_error: bool) -> Fixture {
    let (inner, model, cfg) = observed_cpu_model();
    let (gate, receiver) = tokio::sync::oneshot::channel();
    let (started_sender, started) = tokio::sync::oneshot::channel();
    let executor = Arc::new(CloseGate {
        inner,
        gate: Mutex::new(delayed.then_some(receiver)),
        started: Mutex::new(Some(started_sender)),
        publication_error,
        fail_confirmation: AtomicBool::new(false),
        fail_cancellation: AtomicBool::new(false),
        finishes: AtomicUsize::new(0),
        cancels: AtomicUsize::new(0),
    });
    let trace = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.backend.device = Device::CPU;
    config.runtime.scheduler_trace_jsonl = Some(trace.path().join("trace.jsonl"));
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
    let mut request = InferenceRequest::new("real token closure", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request.evidence_request.capture_engine_token_timing = true;
    let request_id = request.id.clone();
    let prompt = tokenizer.encode(&request.prompt, true).unwrap();
    request.metadata.insert(
        PROMPT_TOKENS_METADATA_KEY.into(),
        serde_json::json!(prompt.len()),
    );
    scheduler.submit(request.clone()).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    assert_eq!(batch.requests.len(), 1);
    let TokenAdmissionOutcome::Admitted(start) = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&request_id, &prompt, cfg.max_seq_len),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("real CPU admission")
    };
    let policy = LogitsReturnPolicy::FullLogits;
    let work = ExecutionWork::new(
        &start,
        TokenWork::Prefill(PrefillChunk::new(0, prompt.len(), prompt.len()).unwrap()),
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("real CPU prefill")
    };
    let completion = batch.into_completions().pop().unwrap();
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        completion.product()
    else {
        panic!("full numerical output")
    };
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
    let values: Vec<_> = prompt.iter().map(|token| token.get()).collect();
    let expected = tiny_llama_model(&cfg).prefill("independent-reference", &values);
    assert_eq!(logits, &expected);
    let sampled = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let identity = completion.next_frontier().sequence().clone();
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        prompt.clone(),
        Some(tokenizer),
        Some(cfg.vocab_size),
    );
    state.execution_frontier = Some(completion.next_frontier().clone());
    state.prefill_complete = true;
    state.prefill_tokens_processed = prompt.len();
    state.generated_tokens.push(TokenId(sampled as u32));
    state.record_decode_ready();
    state.record_generated_token_commit();
    let (response_sender, response) = tokio::sync::oneshot::channel();
    let (stream_sender, stream) = engine.inner.output_channel(request_id.clone(), 16).unwrap();
    state.response_sender = Some(response_sender);
    state.stream_sender = Some(stream_sender);
    let mut slot = RequestSlotLease::open(&engine.inner, request_id.clone());
    slot.admit(&engine.inner);
    state.request_slot = Some(slot);
    engine.inner.sequences.insert(request_id.clone(), state);
    assert_eq!(model.lock().model.kv_caches.len(), 1);
    Fixture {
        engine,
        executor,
        model,
        identity,
        request_id,
        response,
        stream,
        started,
        gate: delayed.then_some(gate),
        trace,
        prompt_len: prompt.len(),
        layers: cfg.num_layers,
    }
}

pub(super) fn slot_events(fixture: &Fixture, action: ResourceAction) -> usize {
    fixture
        .engine
        .inner
        .scheduler_trace_jsonl
        .as_ref()
        .unwrap()
        .flush()
        .unwrap();
    std::fs::read_to_string(fixture.trace.path().join("trace.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<FerrumProfileEvent>(line).unwrap())
        .filter_map(|event| event.resource)
        .filter(|resource| resource.resource_kind == "request_slot" && resource.action == action)
        .count()
}

pub(super) async fn assert_closed_once(fixture: &Fixture) {
    fixture.engine.inner.execution_closes.join_observers().await;
    fixture.engine.inner.drain_execution_closes().unwrap();
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        fixture.identity.observed_end().unwrap().sequence(),
        &fixture.identity
    );
    {
        let model = fixture.model.lock();
        assert!(model.model.kv_caches.is_empty());
        assert_eq!(
            model.releases,
            vec![vec![fixture.prompt_len; fixture.layers]]
        );
    }
    assert_eq!(slot_events(fixture, ResourceAction::Release), 1);
    assert_eq!(slot_events(fixture, ResourceAction::RequestClose), 1);
    assert_eq!(
        fixture
            .engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    fixture
        .engine
        .inner
        .complete_request_for_test(&fixture.request_id, FinishReason::Length)
        .await
        .unwrap();
    fixture.engine.inner.drain_execution_closes().unwrap();
    assert_eq!(slot_events(fixture, ResourceAction::Release), 1);
    assert_eq!(fixture.model.lock().releases.len(), 1);
}

#[tokio::test]
async fn token_completion_cpu_caller_cancel_preserves_actual_close_owner() {
    let mut fixture = fixture(true, false).await;
    let mut completion = Box::pin(
        fixture
            .engine
            .inner
            .complete_request_for_test(&fixture.request_id, FinishReason::Length),
    );
    tokio::select! {
        result = completion.as_mut() => panic!("completion escaped the blocked actual close: {result:?}"),
        started = &mut fixture.started => started.unwrap(),
    }
    assert!(!fixture
        .engine
        .inner
        .sequences
        .get(&fixture.request_id)
        .is_some());
    assert_eq!(
        fixture.engine.inner.scheduler.trace_snapshot().active_len,
        0
    );
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 1);
    assert!(fixture.identity.observed_end().is_none());
    assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 0);
    drop(completion);
    fixture.gate.take().unwrap().send(()).unwrap();
    assert_closed_once(&fixture).await;
    assert_eq!(fixture.executor.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.executor.cancels.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn token_completion_cpu_real_ended_preserves_publication_error() {
    let mut fixture = fixture(false, true).await;
    let error = fixture
        .engine
        .inner
        .complete_request_for_test(&fixture.request_id, FinishReason::Length)
        .await
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("actual close completed but publication failed"));
    assert_eq!(
        fixture
            .response
            .try_recv()
            .unwrap()
            .unwrap_err()
            .to_string(),
        error.to_string()
    );
    let mut terminal_errors = 0;
    while let Ok(chunk) = fixture.stream.try_recv() {
        match chunk {
            Ok(chunk) => assert!(
                chunk.chunk().finish_reason.is_none(),
                "no successful terminal after publication failure"
            ),
            Err(actual) => {
                assert_eq!(actual.to_string(), error.to_string());
                terminal_errors += 1;
            }
        }
    }
    assert_eq!(terminal_errors, 1);
    assert_closed_once(&fixture).await;
    assert_eq!(fixture.executor.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.executor.cancels.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn token_completion_cpu_observed_close_is_consumed_without_reclosing() {
    let mut fixture = fixture(false, false).await;
    let outcome = fixture
        .executor
        .inner
        .cancel_tokens(&fixture.identity)
        .unwrap()
        .with_error(FerrumError::backend("original interrupted close error"));
    fixture
        .engine
        .inner
        .sequences
        .get(&fixture.request_id)
        .unwrap()
        .lock()
        .as_mut()
        .unwrap()
        .execution_close = Some(Box::new(Mutex::new(outcome)));
    let error = fixture
        .engine
        .inner
        .complete_request_for_test(&fixture.request_id, FinishReason::Error)
        .await
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("original interrupted close error"));
    assert_eq!(
        fixture
            .response
            .try_recv()
            .unwrap()
            .unwrap_err()
            .to_string(),
        error.to_string()
    );
    assert_closed_once(&fixture).await;
    assert_eq!(fixture.executor.finishes.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.executor.cancels.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn token_completion_cpu_stale_failure_does_not_repeat_release_or_close_replacement() {
    let fixture = fixture(false, false).await;
    let (request, prompt, original) = {
        let states = &fixture.engine.inner.sequences;
        let state_handle = states.get(&fixture.request_id).unwrap();
        let state_guard = state_handle.lock();
        let state = state_guard.as_ref().unwrap();
        (
            state.original_request.clone(),
            state.input_tokens.clone(),
            state.execution_frontier.clone().unwrap(),
        )
    };
    fixture
        .engine
        .inner
        .complete_request_for_test(&fixture.request_id, FinishReason::Length)
        .await
        .unwrap();
    assert_closed_once(&fixture).await;

    // Reuse the product ID only after the original Engine obligation has
    // actually drained. Both incarnations use the same real CPU model owner.
    let cfg = TinyLlamaConfig::default();
    let TokenAdmissionOutcome::Admitted(start) = fixture
        .executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(
                &fixture.request_id,
                &prompt,
                cfg.max_seq_len,
            ),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("replacement must genuinely admit");
    };
    assert_ne!(start.sequence(), original.sequence());
    fixture
        .engine
        .inner
        .scheduler
        .submit(request.clone())
        .await
        .unwrap();
    let scheduled = fixture
        .engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    assert_eq!(scheduled.requests.len(), 1);
    let policy = LogitsReturnPolicy::FullLogits;
    let work = ExecutionWork::new(
        &start,
        TokenWork::Prefill(PrefillChunk::new(0, prompt.len(), prompt.len()).unwrap()),
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(batch) =
        fixture.executor.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("replacement prefill must actually execute");
    };
    let completion = batch.into_completions().pop().unwrap();
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        completion.product()
    else {
        panic!("replacement requires full logits");
    };
    let mut reference = tiny_llama_model(&cfg);
    let expected = reference.prefill(
        "replacement-reference",
        &prompt.iter().map(|token| token.get()).collect::<Vec<_>>(),
    );
    assert_eq!(logits, &expected);
    let sampled = TokenId(
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32,
    );
    let replacement = completion.next_frontier().clone();
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        prompt.clone(),
        Some(fixture.engine.inner.tokenizer.clone()),
        Some(cfg.vocab_size),
    );
    state.execution_frontier = Some(replacement.clone());
    state.prefill_complete = true;
    state.prefill_tokens_processed = prompt.len();
    state.generated_tokens.push(sampled);
    state.record_decode_ready();
    state.record_generated_token_commit();
    let mut slot = RequestSlotLease::open(&fixture.engine.inner, fixture.request_id.clone());
    slot.admit(&fixture.engine.inner);
    state.request_slot = Some(slot);
    assert!(fixture
        .engine
        .inner
        .sequences
        .insert(fixture.request_id.clone(), state)
        .is_none());
    fixture
        .engine
        .inner
        .scheduler
        .mark_prefill_complete(&fixture.request_id, prompt.len());

    for supplied_outcome in [false, true] {
        let old_error = FerrumError::backend("late old result failure");
        let outcome = supplied_outcome.then(|| {
            ExecutionCloseOutcome::ended(original.sequence().observed_end().unwrap())
                .with_error(old_error.clone())
        });
        let error = fixture
            .engine
            .inner
            .fail_token_frontier(&fixture.request_id, &original, old_error, outcome)
            .await
            .unwrap_err();
        if supplied_outcome {
            assert!(error.to_string().contains("late old result failure"));
        } else {
            // Even an already-published end cannot bypass actual executor
            // validation. Its not-live error is preserved, not called success.
            assert!(error.to_string().contains("token incarnation is not live"));
        }
        fixture.engine.inner.execution_closes.join_observers().await;
        fixture.engine.inner.drain_execution_closes().unwrap();
        assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
        assert_eq!(
            fixture
                .engine
                .inner
                .scheduler
                .trace_snapshot()
                .capacity_release_epoch,
            1
        );
        assert_eq!(fixture.executor.cancels.load(Ordering::SeqCst), 1);
        assert_eq!(slot_events(&fixture, ResourceAction::Release), 1);
        assert_eq!(slot_events(&fixture, ResourceAction::RequestClose), 1);
        {
            let states = &fixture.engine.inner.sequences;
            let state_handle = states.get(&fixture.request_id).unwrap();
            let state_guard = state_handle.lock();
            let state = state_guard.as_ref().unwrap();
            assert_eq!(state.execution_frontier.as_ref(), Some(&replacement));
            assert!(state.request_slot.as_ref().unwrap().armed);
        }
        assert!(replacement.sequence().observed_end().is_none());
        let model = fixture.model.lock();
        assert_eq!(model.releases.len(), 1);
        assert_eq!(model.model.kv_caches.len(), 1);
        assert!(model
            .model
            .kv_caches
            .values()
            .next()
            .unwrap()
            .iter()
            .all(|layer| layer.len == prompt.len()));
    }

    // Full-vocabulary continuation proves the replacement's real KV survives
    // both old callbacks, not merely its outer identity or queue membership.
    let decode = ExecutionWork::new(
        &replacement,
        TokenWork::Decode {
            token: sampled,
            position: prompt.len(),
        },
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(batch) =
        fixture.executor.execute_tokens(&[decode]).await.unwrap()
    else {
        panic!("replacement decode must continue after stale failure");
    };
    let completion = batch.into_completions().pop().unwrap();
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        completion.product()
    else {
        panic!("replacement decode requires full logits");
    };
    assert_eq!(
        logits,
        &reference.decode("replacement-reference", sampled.get(), prompt.len() as u32)
    );
    {
        let states = &fixture.engine.inner.sequences;
        let state_handle = states.get(&fixture.request_id).unwrap();
        let mut state_guard = state_handle.lock();
        let state = state_guard.as_mut().unwrap();
        state.execution_frontier = Some(completion.next_frontier().clone());
        state.generated_tokens.push(TokenId(
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32,
        ));
        state.record_decode_ready();
        state.record_generated_token_commit();
    }
    fixture
        .engine
        .inner
        .complete_request_for_test(&fixture.request_id, FinishReason::Error)
        .await
        .unwrap();
    fixture.engine.inner.execution_closes.join_observers().await;
    fixture.engine.inner.drain_execution_closes().unwrap();
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        fixture
            .engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        2
    );
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 2);
    assert_eq!(slot_events(&fixture, ResourceAction::RequestClose), 2);
    assert!(replacement.sequence().observed_end().is_some());
    let model = fixture.model.lock();
    assert!(model.model.kv_caches.is_empty());
    assert_eq!(
        model.releases,
        vec![
            vec![prompt.len(); cfg.num_layers],
            vec![prompt.len() + 1; cfg.num_layers]
        ]
    );
}
