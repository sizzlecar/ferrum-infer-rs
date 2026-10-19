//! Actual tiny CPU forwards with a separate numerical reference. Gates delay
//! observation only after the real output and per-layer KV have been checked.

use super::*;
use ferrum_execution::executor_lifecycle::{
    ExecutorOwner, ExecutorScope, ExecutorShutdownObserver, ExecutorShutdownPermit,
};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_models::models::LlamaFamilyModel;
use ferrum_models::test_support::{tiny_llama_model, TinyTokenizer};
use std::sync::atomic::AtomicUsize;

struct ReferenceSequence {
    cache: String,
    prompt: Vec<u32>,
    frontier: usize,
    outputs: usize,
}

struct ReferenceState {
    model: LlamaFamilyModel<CpuBackend>,
    sequences: HashMap<ExecutorSequence, ReferenceSequence>,
}

struct ExecutionGate {
    reached: tokio::sync::oneshot::Sender<()>,
    release: tokio::sync::oneshot::Receiver<()>,
}

pub(super) struct NumericalExecutor {
    inner: Arc<ExecutorScope>,
    owner: ExecutorOwner,
    reference: Mutex<ReferenceState>,
    observation: Arc<NumericalObservation>,
    drop_signal: ExecutorDropSignal,
    layers: usize,
}

impl std::ops::Deref for NumericalExecutor {
    type Target = NumericalObservation;
    fn deref(&self) -> &Self::Target {
        &self.observation
    }
}

pub(super) struct NumericalObservation {
    pub(super) observed: Arc<Mutex<closure::ModelObservation>>,
    info: ferrum_types::ModelInfo,
    gate: Mutex<Option<ExecutionGate>>,
    next_prefill: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    in_flight: AtomicUsize,
    pub(super) maximum_in_flight: AtomicUsize,
    pub(super) admissions: AtomicUsize,
    pub(super) closes: AtomicUsize,
    close_notify: Notify,
    pub(super) checked_outputs: AtomicUsize,
    output_notify: Notify,
    pub(super) dropped: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

struct ExecutorDropSignal(Arc<NumericalObservation>);
impl Drop for ExecutorDropSignal {
    fn drop(&mut self) {
        if let Some(sender) = self.0.dropped.lock().take() {
            let _ = sender.send(());
        }
    }
}

struct Flight<'a>(&'a AtomicUsize);

impl Drop for Flight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl ModelExecutor for NumericalExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        // The numerical oracle is another real CPU model, owned only here.
        let reference_close = self.reference.lock().model.close();
        if let Err(error) = reference_close {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        match Arc::try_unwrap(self) {
            Ok(Self {
                inner,
                owner,
                reference,
                observation,
                drop_signal,
                layers: _,
            }) => {
                drop(inner);
                drop(reference);
                drop(observation);
                drop(drop_signal);
                owner.begin_shutdown()
            }
            Err(original) => ExecutorShutdownObserver::retained(
                Arc::new(original),
                FerrumError::internal("numerical fixture still has an unscoped execution owner"),
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
    fn kv_capacity(&self) -> Option<usize> {
        self.inner.kv_capacity()
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        let before: HashSet<_> = self
            .observed
            .lock()
            .model
            .kv_caches
            .keys()
            .cloned()
            .collect();
        let outcome = self.inner.admit_tokens(input).await?;
        if let TokenAdmissionOutcome::Admitted(frontier) = &outcome {
            let observed = self.observed.lock();
            let new_caches: Vec<_> = observed
                .model
                .kv_caches
                .keys()
                .filter(|id| !before.contains(*id))
                .cloned()
                .collect();
            assert_eq!(new_caches.len(), 1, "one real cache per actual admission");
            assert!(self
                .reference
                .lock()
                .sequences
                .insert(
                    frontier.sequence().clone(),
                    ReferenceSequence {
                        cache: new_caches[0].clone(),
                        prompt: input
                            .context
                            .input_tokens
                            .iter()
                            .map(|token| token.get())
                            .collect(),
                        frontier: 0,
                        outputs: 0,
                    }
                )
                .is_none());
            self.admissions.fetch_add(1, Ordering::SeqCst);
        }
        Ok(outcome)
    }

    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        let active = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.maximum_in_flight.fetch_max(active, Ordering::SeqCst);
        let _flight = Flight(&self.in_flight);
        let outcome = self.inner.execute_tokens(work).await?;
        let ExecutionBatchOutcome::Completed(batch) = &outcome else {
            panic!("actual CPU execution must complete without a model fault");
        };
        assert_eq!(batch.completions().len(), work.len());
        {
            let observed = self.observed.lock();
            let mut reference = self.reference.lock();
            for completion in batch.completions() {
                let input = &work[completion.input_index()];
                let ReferenceState { model, sequences } = &mut *reference;
                let state = sequences.get_mut(input.sequence()).unwrap();
                assert_eq!(
                    state.frontier,
                    input.expected_frontier().consumed_tokens(),
                    "one committed numerical frontier, never replay an earlier step"
                );
                let expected = match input.work() {
                    TokenWork::Prefill(chunk) => {
                        let start = chunk.tokens_processed();
                        let end = start + chunk.tokens_to_process();
                        model.prefill(&state.cache, &state.prompt[start..end])
                    }
                    TokenWork::Decode { token, position } => {
                        model.decode(&state.cache, token.get(), u32::try_from(position).unwrap())
                    }
                };
                let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(actual)) =
                    completion.product()
                else {
                    panic!("numerical oracle requires the full vocabulary");
                };
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                assert!(actual.iter().all(|value| value.is_finite()));
                state.frontier = completion.next_frontier().consumed_tokens();
                state.outputs += 1;
                assert_eq!(
                    observed.model.kv_caches[&state.cache]
                        .iter()
                        .map(|layer| layer.len)
                        .collect::<Vec<_>>(),
                    vec![state.frontier; self.layers]
                );
                self.checked_outputs.fetch_add(1, Ordering::SeqCst);
            }
        }
        self.output_notify.notify_waiters();
        if work
            .iter()
            .any(|input| matches!(input.work(), TokenWork::Prefill(_)))
        {
            let hook = self.next_prefill.lock().take();
            if let Some(hook) = hook {
                hook();
            }
        }
        let gate = self.gate.lock().take();
        if let Some(gate) = gate {
            gate.reached.send(()).unwrap();
            gate.release
                .await
                .expect("test must release actual completed forward");
        }
        Ok(outcome)
    }

    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        let result = self.inner.finish_tokens(sequence, reason).await?;
        self.check_close(sequence, &result);
        Ok(result)
    }

    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        let result = self.inner.cancel_tokens(sequence)?;
        self.check_close(sequence, &result);
        Ok(result)
    }
}

impl NumericalObservation {
    pub(super) fn info(&self) -> &ferrum_types::ModelInfo {
        &self.info
    }
    pub(super) fn remove_execution_gate(&self) {
        self.gate.lock().take();
    }

    pub(super) fn after_next_prefill(&self, hook: impl FnOnce() + Send + 'static) {
        assert!(self.next_prefill.lock().replace(Box::new(hook)).is_none());
    }

    pub(super) async fn wait_for_outputs(&self, count: usize) {
        loop {
            let changed = self.output_notify.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.checked_outputs.load(Ordering::SeqCst) >= count {
                return;
            }
            changed.await;
        }
    }

    pub(super) async fn wait_for_closes(&self, count: usize) {
        loop {
            let changed = self.close_notify.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.closes.load(Ordering::SeqCst) >= count {
                return;
            }
            changed.await;
        }
    }
}

impl NumericalExecutor {
    fn check_close(&self, sequence: &ExecutorSequence, result: &ExecutionCloseOutcome) {
        let ExecutionCloseDisposition::Ended(ended) = &result.disposition else {
            panic!("the actual synchronous CPU close must prove terminal");
        };
        assert_eq!(ended.sequence(), sequence);
        assert!(result.error.is_none());
        let mut reference = self.reference.lock();
        let state = reference
            .sequences
            .remove(sequence)
            .expect("close each original once");
        assert!(!self
            .observed
            .lock()
            .model
            .kv_caches
            .contains_key(&state.cache));
        reference.model.release(&state.cache);
        self.closes.fetch_add(1, Ordering::SeqCst);
        self.close_notify.notify_waiters();
    }
}

pub(super) struct Fixture {
    pub(super) engine: Arc<ContinuousBatchEngine>,
    pub(super) original_inner: Weak<EngineInner>,
    pub(super) output_budget: OutputBudget,
    pub(super) executor: Arc<NumericalObservation>,
    pub(super) reached: tokio::sync::oneshot::Receiver<()>,
    pub(super) release: tokio::sync::oneshot::Sender<()>,
    pub(super) trace: tempfile::TempDir,
}

impl Fixture {
    pub(super) fn new() -> Self {
        Self::with_config(|_| {})
    }

    pub(super) fn with_config(configure: impl FnOnce(&mut EngineConfig)) -> Self {
        Self::with_tokenizer(configure, |vocab| Arc::new(TinyTokenizer::new(vocab)))
    }

    pub(super) fn with_tokenizer(
        configure: impl FnOnce(&mut EngineConfig),
        tokenizer: impl FnOnce(usize) -> Arc<dyn Tokenizer>,
    ) -> Self {
        Self::with_options(configure, tokenizer, |_| None)
    }

    pub(super) fn with_journal_scopes(
        configure: impl FnOnce(&mut EngineConfig),
        scopes: impl FnOnce(&EngineConfig) -> EngineJournalScopes,
    ) -> Self {
        Self::with_options(
            configure,
            |vocab| Arc::new(TinyTokenizer::new(vocab)),
            |config| Some(scopes(config)),
        )
    }

    fn with_options(
        configure: impl FnOnce(&mut EngineConfig),
        tokenizer: impl FnOnce(usize) -> Arc<dyn Tokenizer>,
        scopes: impl FnOnce(&EngineConfig) -> Option<EngineJournalScopes>,
    ) -> Self {
        let (inner, observed, cfg) = closure::observed_cpu_model_owned();
        let owner = ExecutorOwner::new(inner);
        let inner = Arc::new(owner.scope().unwrap());
        let (reached_tx, reached) = tokio::sync::oneshot::channel();
        let (release, release_rx) = tokio::sync::oneshot::channel();
        let executor = Arc::new(NumericalObservation {
            info: inner.info().clone(),
            observed,
            gate: Mutex::new(Some(ExecutionGate {
                reached: reached_tx,
                release: release_rx,
            })),
            next_prefill: Mutex::new(None),
            in_flight: AtomicUsize::new(0),
            maximum_in_flight: AtomicUsize::new(0),
            admissions: AtomicUsize::new(0),
            closes: AtomicUsize::new(0),
            close_notify: Notify::new(),
            checked_outputs: AtomicUsize::new(0),
            output_notify: Notify::new(),
            dropped: Mutex::new(None),
        });
        let owned_executor = ExecutorOwner::new(Box::new(NumericalExecutor {
            inner,
            owner,
            reference: Mutex::new(ReferenceState {
                model: tiny_llama_model(&cfg),
                sequences: HashMap::new(),
            }),
            observation: Arc::clone(&executor),
            drop_signal: ExecutorDropSignal(Arc::clone(&executor)),
            layers: cfg.num_layers,
        }));
        let trace = tempfile::tempdir().unwrap();
        let mut config = EngineConfig::default();
        config.backend.device = Device::CPU;
        config.runtime.scheduler_trace_jsonl = Some(trace.path().join("trace.jsonl"));
        configure(&mut config);
        let journal_scopes = scopes(&config);
        let engine = ContinuousBatchEngine::new_with_recurrent_state_manager_and_journal_scopes(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
            tokenizer(cfg.vocab_size),
            Arc::new(crate::registry::GreedySampler),
            Arc::new(MockKvCacheManager::new(128)),
            owned_executor,
            Arc::new(crate::tensor_factory::CandleTensorFactory::new(Device::CPU)),
            None,
            journal_scopes,
        )
        .unwrap();
        let original_inner = Arc::downgrade(&engine.active_inner().unwrap());
        let output_budget = engine.active_inner().unwrap().output_budget.clone();
        Self {
            engine: Arc::new(engine),
            original_inner,
            output_budget,
            executor,
            reached,
            release,
            trace,
        }
    }

    pub(super) fn request(&self) -> InferenceRequest {
        let mut request = InferenceRequest::new("same real numerical prompt", "tiny-llama-test");
        request.sampling_params.max_tokens = 4;
        request.sampling_params.temperature = 0.0;
        request.sampling_params.top_k = Some(self.executor.info().vocab_size);
        request
            .metadata
            .insert("ferrum_ignore_eos".into(), serde_json::json!(true));
        request
    }
}

pub(super) fn slot_events(
    engine: &ContinuousBatchEngine,
    path: &std::path::Path,
    action: ResourceAction,
) -> usize {
    engine
        .active_inner()
        .unwrap()
        .scheduler_trace_jsonl
        .as_ref()
        .unwrap()
        .flush()
        .unwrap();
    closed_slot_events(path, action)
}

// Use only after a successful driver cleanup has closed and joined the writer.
pub(super) fn closed_slot_events(path: &std::path::Path, action: ResourceAction) -> usize {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<FerrumProfileEvent>(line).unwrap())
        .filter_map(|event| event.resource)
        .filter(|resource| resource.resource_kind == "request_slot" && resource.action == action)
        .count()
}
