//! Synthetic logits only; stop handling and responses use the production engine.
use super::*;
use ferrum_execution::{
    model_executor::{
        ExecutionResourceAuthority, ExecutorCapabilities, ExecutorSamplingOutput,
        ExecutorSequenceCompletion, ExecutorStatus, PlanRuntimeResourceSnapshot,
    },
    token_execution::*,
    ExecutorShutdownObserver, ExecutorShutdownPermit, ModelExecutor,
};
use ferrum_testkit::MockModelExecutor;
use ferrum_types::{ExecutorAdmissionLimits, ModelInfo, RequestId, Result};

struct Cursor {
    request_id: RequestId,
    prompt: Arc<[TokenId]>,
    maximum: usize,
    frontier: ExecutionFrontier,
    next: usize,
}

#[derive(Clone, Debug)]
pub(super) struct LogitStep {
    candidates: Vec<(TokenId, f32)>,
}
impl LogitStep {
    pub fn only(token: TokenId) -> Self {
        Self::candidates(vec![(token, 1.0)])
    }
    pub fn candidates(candidates: Vec<(TokenId, f32)>) -> Self {
        Self { candidates }
    }
}

pub(super) struct ScriptedExecutor {
    metadata: MockModelExecutor,
    script: Vec<LogitStep>,
    issuer: ExecutorSequenceIssuer,
    active: Arc<Mutex<HashMap<ExecutorSequence, Cursor>>>,
    observation: Arc<ScriptedObservation>,
}

/// Inspect results without keeping the executable or any live sequence alive.
pub(super) struct ScriptedObservation {
    decoded_inputs: Mutex<Vec<TokenId>>,
    prefill_tokens: Mutex<Vec<TokenId>>,
    active: std::sync::Weak<Mutex<HashMap<ExecutorSequence, Cursor>>>,
    active_at_retirement: Mutex<Option<usize>>,
    pub prompt_tokens: AtomicUsize,
    pub generated_tokens: AtomicUsize,
    completed: Mutex<Vec<ExecutorSequenceCompletion>>,
}

impl ScriptedExecutor {
    pub fn new(vocab_size: usize, script: Vec<TokenId>) -> Self {
        Self::from_steps(
            vocab_size,
            script.into_iter().map(LogitStep::only).collect(),
        )
    }
    pub fn from_steps(vocab_size: usize, steps: Vec<LogitStep>) -> Self {
        assert!(!steps.is_empty());
        let active = Arc::new(Mutex::new(HashMap::new()));
        let observation = Arc::new(ScriptedObservation {
            decoded_inputs: Mutex::new(Vec::new()),
            prefill_tokens: Mutex::new(Vec::new()),
            active: Arc::downgrade(&active),
            active_at_retirement: Mutex::new(None),
            prompt_tokens: AtomicUsize::new(0),
            generated_tokens: AtomicUsize::new(0),
            completed: Mutex::new(Vec::new()),
        });
        Self {
            metadata: MockModelExecutor::instant(vocab_size),
            script: steps,
            issuer: ExecutorSequenceIssuer::new(),
            active,
            observation,
        }
    }
    pub fn observer(&self) -> Arc<ScriptedObservation> {
        Arc::clone(&self.observation)
    }
    fn logits(&self, index: usize) -> Result<Vec<f32>> {
        let step = self.script.get(index).ok_or_else(|| {
            FerrumError::backend("script exhausted: engine did not stop on the final token")
        })?;
        let mut logits = vec![f32::NEG_INFINITY; self.info().vocab_size];
        for &(token, logit) in &step.candidates {
            *logits
                .get_mut(token.get() as usize)
                .ok_or_else(|| FerrumError::backend("script token outside vocabulary"))? = logit;
        }
        Ok(logits)
    }
    fn close(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.issuer.frontier(sequence, 0)?;
        let mut active = self.active.lock().unwrap();
        let Some(cursor) = active.get(sequence) else {
            return sequence
                .observed_end()
                .map(ExecutionCloseOutcome::ended)
                .ok_or_else(|| FerrumError::backend("script close has no live incarnation"));
        };
        if let ExecutionEndReason::Completed {
            input_tokens,
            output_tokens,
        } = reason
        {
            assert_eq!(input_tokens, cursor.prompt.len());
            assert_eq!(output_tokens, cursor.next);
            self.observation
                .completed
                .lock()
                .unwrap()
                .push(ExecutorSequenceCompletion::new(
                    cursor.request_id.clone(),
                    format!("script-{}", cursor.request_id),
                    input_tokens,
                    output_tokens,
                )?);
        }
        active.remove(sequence);
        Ok(ExecutionCloseOutcome::ended(self.issuer.ended(sequence)?))
    }
}

impl ScriptedObservation {
    pub fn decoded_inputs(&self) -> Vec<TokenId> {
        self.decoded_inputs.lock().unwrap().clone()
    }
    pub fn prefill_tokens(&self) -> Vec<TokenId> {
        self.prefill_tokens.lock().unwrap().clone()
    }
    pub fn assert_released(&self) {
        if let Some(active) = self.active.upgrade() {
            assert!(active.lock().unwrap().is_empty());
        } else {
            assert_eq!(
                *self.active_at_retirement.lock().unwrap(),
                Some(0),
                "request cleanup must finish before whole-executor retirement"
            );
        }
    }
    pub fn assert_completed(&self) {
        let completed = self.completed.lock().unwrap();
        assert_eq!(
            completed.len(),
            1,
            "one request completed inside the engine"
        );
        assert_eq!(
            completed[0].input_tokens(),
            self.prompt_tokens.load(Ordering::Relaxed) as u64
        );
        assert_eq!(
            completed[0].output_tokens(),
            self.generated_tokens.load(Ordering::Relaxed) as u64
        );
    }
}

#[async_trait]
impl ModelExecutor for ScriptedExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        let original = match Arc::try_unwrap(self) {
            Ok(original) => original,
            Err(original) => {
                return ExecutorShutdownObserver::retained(
                    Arc::new(original),
                    FerrumError::internal("script executor still has live users"),
                )
            }
        };
        let sequences: Vec<_> = original.active.lock().unwrap().keys().cloned().collect();
        *original.observation.active_at_retirement.lock().unwrap() = Some(sequences.len());
        for sequence in sequences {
            if let Err(error) = original.close(&sequence, ExecutionEndReason::Cancelled) {
                return ExecutorShutdownObserver::retained(Arc::new(original), error);
            }
        }
        let (reporter, observer) = ExecutorShutdownObserver::channel(original.observer());
        drop(original);
        reporter.retired(None);
        observer
    }
    fn info(&self) -> &ModelInfo {
        self.metadata.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.metadata.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.metadata.status()
    }
    // The explicit composition fixture retains its declared limits. There are
    // no fake native epochs: actual token closure signals scheduler progress.
    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }
    fn admission_limits(&self) -> Result<Option<ExecutorAdmissionLimits>> {
        ExecutorAdmissionLimits::new(1, 256)
            .map(Some)
            .map_err(FerrumError::internal)
    }
    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        PlanRuntimeResourceSnapshot::new(0, 0, 0, 0, 0, 0, 0, 0, 0).map(Some)
    }
    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        input.validate()?;
        if input.adapter.is_some() {
            return Err(FerrumError::unsupported(
                "script has no LoRA implementation",
            ));
        }
        let mut active = self.active.lock().unwrap();
        if active
            .values()
            .any(|cursor| &cursor.request_id == input.context.request_id)
        {
            return Err(FerrumError::backend("duplicate live scripted request"));
        }
        let sequence = self.issuer.issue_sequence()?;
        let frontier = self.issuer.frontier(&sequence, 0)?;
        active.insert(
            sequence,
            Cursor {
                request_id: input.context.request_id.clone(),
                prompt: input.context.input_tokens.into(),
                maximum: input.context.maximum_sequence_tokens,
                frontier: frontier.clone(),
                next: 0,
            },
        );
        Ok(TokenAdmissionOutcome::Admitted(frontier))
    }
    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        let batch = self.issuer.batch(work, self.info().vocab_size)?;
        let mut active = self.active.lock().unwrap();
        for input in work {
            let cursor = active
                .get(input.sequence())
                .ok_or_else(|| FerrumError::backend("unknown scripted incarnation"))?;
            if &cursor.frontier != input.expected_frontier()
                || input.work().range()?.end() > cursor.maximum
            {
                return Err(FerrumError::backend("stale scripted frontier"));
            }
            match input.work() {
                TokenWork::Prefill(chunk) if chunk.total_prompt_tokens() == cursor.prompt.len() => {
                }
                TokenWork::Decode { position, .. } if position >= cursor.prompt.len() => {}
                _ => {
                    return Err(FerrumError::backend(
                        "script work does not match admitted prompt",
                    ))
                }
            }
        }
        let prefill = matches!(work[0].work(), TokenWork::Prefill(_));
        if work
            .iter()
            .any(|input| matches!(input.work(), TokenWork::Prefill(_)) != prefill)
        {
            return Ok(batch.not_submitted(NoSubmissionReason::UnsupportedShape));
        }
        // Pure products are validated before any frontier/counter publication.
        let mut completions = Vec::with_capacity(work.len());
        for (index, input) in work.iter().enumerate() {
            let cursor = &active[input.sequence()];
            let executed = match input.work() {
                TokenWork::Prefill(chunk) => ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count: 0,
                },
                TokenWork::Decode { position, .. } => ExecutedTokenWork::Decode { position },
            };
            let product = match input.work() {
                TokenWork::Prefill(chunk) if !chunk.is_final() => ExecutionProduct::Intermediate,
                _ => match self.logits(cursor.next) {
                    Ok(values) => {
                        ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values))
                    }
                    Err(error) => {
                        return Ok(batch.not_submitted(NoSubmissionReason::Rejected(error)))
                    }
                },
            };
            match batch.completion(index, executed, product) {
                Ok(completion) => completions.push(completion),
                Err(error) => {
                    return Ok(
                        batch.not_submitted(NoSubmissionReason::Rejected(error.into_parts().0))
                    )
                }
            }
        }
        for (input, completion) in work.iter().zip(&completions) {
            let cursor = active.get_mut(input.sequence()).expect("validated cursor");
            cursor.frontier = completion.next_frontier().clone();
            match input.work() {
                TokenWork::Prefill(chunk) if chunk.is_final() => {
                    *self.observation.prefill_tokens.lock().unwrap() = cursor.prompt.to_vec();
                    self.observation
                        .prompt_tokens
                        .store(cursor.prompt.len(), Ordering::Relaxed);
                }
                TokenWork::Decode { token, .. } => {
                    self.observation.decoded_inputs.lock().unwrap().push(token)
                }
                _ => {}
            }
            if matches!(completion.product(), ExecutionProduct::Sampling(_)) {
                cursor.next += 1;
                self.observation
                    .generated_tokens
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(batch
            .completed(completions)
            .unwrap_or_else(|error| error.into_outcome()))
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.close(sequence, reason)
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.close(sequence, ExecutionEndReason::Cancelled)
    }
}
