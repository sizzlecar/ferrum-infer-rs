//! One borrowed causal phase. This never creates a Runtime, closes the whole
//! executor owner, or retries work that may already have reached the device.
use ferrum_execution::{
    executor_lifecycle::ExecutorScope,
    model_executor::{
        ExecutorAdmissionEpochs, ExecutorPrefillAdmission, ExecutorPrefillMaintenanceOutcome,
        ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
    },
    token_execution::*,
};
use ferrum_models::p_h3_image::PH3ImagePromptExpansionGeneration;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBuffer};
use ferrum_types::{FerrumError, FinishReason, RequestId, TokenId};

mod batch;
#[cfg(test)]
mod tests;
pub(super) use batch::{advance_batch, ExpansionWave, ExpansionWaveOutcome};

/// Initialized, uniquely writable Host Output backing. There is deliberately no
/// TokenId cast: that public newtype has no representation contract.
pub(super) struct ManagedTokenHistory {
    buffer: ManagedHostBuffer,
    capacity: usize,
    length: usize,
    backing_bytes: usize,
}
impl std::fmt::Debug for ManagedTokenHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedTokenHistory")
            .field("length", &self.length)
            .field("capacity", &self.capacity)
            .field("backing_bytes", &self.backing_bytes)
            .finish()
    }
}

fn history_logical_bytes(max_new_tokens: u32) -> ferrum_types::Result<usize> {
    let capacity = usize::try_from(max_new_tokens).map_err(|_| {
        FerrumError::request_validation("expansion token capacity exceeds address space")
    })?;
    if capacity == 0 {
        return Err(FerrumError::request_validation(
            "expansion requires positive max_new_tokens",
        ));
    }
    capacity
        .checked_mul(std::mem::size_of::<u32>())
        .and_then(|n| n.checked_add(std::mem::align_of::<u32>() - 1))
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| FerrumError::request_validation("expansion token history size overflows"))
}

/// Physical Output demand only; no grant, payload allocation or cached account.
pub(super) fn history_backing_bytes(
    host: &HostMemoryAccount,
    max_new_tokens: u32,
) -> ferrum_types::Result<usize> {
    host.backing_bytes(history_logical_bytes(max_new_tokens)?)
        .map_err(|e| FerrumError::resource_exhausted(format!("expansion token history: {e}")))
}

impl ManagedTokenHistory {
    pub(super) fn new(host: &HostMemoryAccount, max_new_tokens: u32) -> ferrum_types::Result<Self> {
        let logical = history_logical_bytes(max_new_tokens)?;
        let backing_bytes = history_backing_bytes(host, max_new_tokens)?;
        let buffer = host
            .allocate_zeroed(HostMemoryClass::Output, logical)
            .map_err(|e| {
                FerrumError::resource_exhausted(format!("expansion token history: {e}"))
            })?;
        let capacity = usize::try_from(max_new_tokens)
            .map_err(|_| FerrumError::internal("checked token capacity changed"))?;
        // SAFETY: every byte was initialized, u32 admits every bit pattern,
        // and the returned view is checked before any indexing. Three spare
        // bytes allow any legal u8 allocation address to reach u32 alignment.
        let (_, words, _) = unsafe { buffer.as_ref().align_to::<u32>() };
        if words.len() < capacity {
            return Err(FerrumError::internal(
                "expansion history lacks its checked aligned capacity",
            ));
        }
        Ok(Self {
            buffer,
            capacity,
            length: 0,
            backing_bytes,
        })
    }
    pub(super) fn ids(&self) -> &[u32] {
        // SAFETY: the backing remains initialized, immutable for this borrow,
        // and its aligned capacity was checked before publication.
        let (_, words, _) = unsafe { self.buffer.as_ref().align_to::<u32>() };
        &words[..self.length]
    }
    #[cfg(test)]
    pub(super) fn backing_bytes(&self) -> usize {
        self.backing_bytes
    }
    pub(super) fn push(&mut self, token: TokenId) -> ferrum_types::Result<()> {
        if self.length >= self.capacity {
            return Err(FerrumError::internal(
                "expansion history exceeded max_new_tokens",
            ));
        }
        // SAFETY: this is the unique mutable borrow of initialized managed
        // backing, u32 has no invalid bit patterns, and length < capacity.
        let (_, words, _) = unsafe { self.buffer.as_mut().align_to_mut::<u32>() };
        words[self.length] = token.get(); // native-endian u32, consumed as u32.
        self.length += 1;
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct ExpansionExecution {
    pub(super) history: ManagedTokenHistory,
    pub(super) finish: FinishReason,
    pub(super) ended: ExecutionEnded,
    pub(super) frontier: ExecutionFrontier,
}

/// Whether this request ever entered an executor admission call. A bare error
/// after entry does not certify absence of native or retained authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AdmissionProbe {
    NeverEntered,
    Entered,
}

/// Keep exact executor observations, especially Held/Unknown authorities. This
/// error does not certify zero submission or authorize replay. The caller must
/// retain the original owner until terminal proof or its original retirement.
pub(super) struct ExpansionExecutionFailure {
    pub(super) finish: FinishReason,
    pub(super) error: FerrumError,
    pub(super) history: Option<ManagedTokenHistory>,
    pub(super) frontier: Option<ExecutionFrontier>,
    pub(super) admission: Option<TokenAdmissionOutcome>,
    pub(super) maintenance: Option<ExecutorPrefillMaintenanceOutcome>,
    pub(super) outcome: Option<ExecutionBatchOutcome>,
    pub(super) close: Option<ExecutionCloseOutcome>,
    pub(super) close_error: Option<FerrumError>,
    pub(super) admission_probe: AdmissionProbe,
    pub(super) wave_observation: Option<ExpansionWaveReference>,
}

/// Index into the enclosing cohort's uniquely retained original wave ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ExpansionWaveReference {
    pub(super) wave: usize,
    pub(super) input_index: usize,
}
impl std::fmt::Debug for ExpansionExecutionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let outcome = self.outcome.as_ref().map(|outcome| match outcome {
            ExecutionBatchOutcome::Completed(_) => "completed",
            ExecutionBatchOutcome::NotSubmitted(_) => "not_submitted",
            ExecutionBatchOutcome::Interrupted(_) => "interrupted",
            ExecutionBatchOutcome::ContractViolation(_) => "contract_violation",
        });
        f.debug_struct("ExpansionExecutionFailure")
            .field("finish", &self.finish)
            .field("error", &self.error)
            .field("history", &self.history)
            .field("frontier", &self.frontier)
            .field("admission", &self.admission)
            .field("maintenance", &self.maintenance)
            .field("outcome", &outcome)
            .field("close", &self.close)
            .field("close_error", &self.close_error)
            .field("admission_probe", &self.admission_probe)
            .field("wave_observation", &self.wave_observation)
            .finish()
    }
}
impl ExpansionExecutionFailure {
    fn new(error: FerrumError, finish: FinishReason, history: Option<ManagedTokenHistory>) -> Self {
        Self {
            error,
            finish,
            history,
            frontier: None,
            admission: None,
            maintenance: None,
            outcome: None,
            close: None,
            close_error: None,
            admission_probe: AdmissionProbe::NeverEntered,
            wave_observation: None,
        }
    }
}

fn active_failure(
    scope: &ExecutorScope,
    history: ManagedTokenHistory,
    frontier: ExecutionFrontier,
    finish: FinishReason,
    error: FerrumError,
    outcome: Option<ExecutionBatchOutcome>,
) -> ExpansionExecutionFailure {
    let mut failure = ExpansionExecutionFailure::new(error, finish, Some(history));
    failure.admission_probe = AdmissionProbe::Entered;
    match scope.cancel_tokens(frontier.sequence()) {
        Ok(close) => failure.close = Some(close),
        Err(error) => failure.close_error = Some(error),
    }
    failure.frontier = Some(frontier);
    failure.outcome = outcome;
    failure
}

/// Project only the failure cause, never products, token history or recovery
/// authorities. The original outcome remains owned by ExpansionExecutionFailure.
fn incomplete_wave_error(outcome: &ExecutionBatchOutcome) -> FerrumError {
    incomplete_participant_error(outcome, 0)
}

fn incomplete_participant_error(
    outcome: &ExecutionBatchOutcome,
    input_index: usize,
) -> FerrumError {
    fn deferred_error(
        deferral: &ferrum_execution::model_executor::ExecutorExecutionDeferral,
    ) -> FerrumError {
        let reason = match deferral {
            ferrum_execution::model_executor::ExecutorExecutionDeferral::Capacity(_) => "capacity",
            ferrum_execution::model_executor::ExecutorExecutionDeferral::AllocatorCapacity(_) => {
                "allocator_capacity"
            }
            ferrum_execution::model_executor::ExecutorExecutionDeferral::RequestState(_) => {
                "request_state"
            }
        };
        FerrumError::backend(format!(
            "expansion wave deferred: {reason} at {:?}; original outcome retained",
            deferral.stage()
        ))
    }
    match outcome {
        ExecutionBatchOutcome::NotSubmitted(batch) => match batch.reason() {
            NoSubmissionReason::Rejected(error) => error.clone(),
            NoSubmissionReason::Deferred(deferral) => deferred_error(deferral),
            NoSubmissionReason::UnsupportedShape => FerrumError::backend(
                "expansion wave not submitted: unsupported shape; original outcome retained",
            ),
        },
        ExecutionBatchOutcome::Interrupted(batch) => {
            // Respect the original input/cause association, independent of
            // completion order or the cause ledger's storage order.
            match batch
                .remaining()
                .iter()
                .find(|participant| participant.input_index() == input_index)
                .and_then(|participant| batch.causes().get(participant.cause_index()))
            {
                Some(ExecutionInterruptionCause::Error(error)) => error.clone(),
                Some(ExecutionInterruptionCause::Deferred(deferral)) => deferred_error(deferral),
                None => FerrumError::backend(
                    "expansion wave interrupted without an associated participant cause; original outcome retained",
                ),
            }
        }
        ExecutionBatchOutcome::ContractViolation(violation) => violation.error.clone(),
        ExecutionBatchOutcome::Completed(_) => FerrumError::internal(
            "completed expansion wave was passed to incomplete-wave diagnostics",
        ),
    }
}

fn selected(product: &ExecutionProduct) -> ferrum_types::Result<Option<TokenId>> {
    match product {
        ExecutionProduct::Intermediate => Ok(None),
        ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(token)) => Ok(Some(*token)),
        ExecutionProduct::Sampling(ExecutorSamplingOutput::NoSelectableFiniteLogit) => Err(
            FerrumError::backend("expansion greedy output has no selectable finite logit"),
        ),
        // Legal under the original greedy contract. Borrow the existing output
        // without creating another logits/history payload. Product composition
        // selects managed greedy IO; this does not account arbitrary legacy
        // executor allocations, tokenizer metadata or whole-process RSS.
        ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) => {
            let mut best: Option<(usize, f32)> = None;
            for (index, &value) in logits
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
            {
                if best.is_none_or(|(_, prior)| value.total_cmp(&prior).is_gt()) {
                    best = Some((index, value));
                }
            }
            let (index, _) = best.ok_or_else(|| {
                FerrumError::backend("expansion greedy output has no selectable finite logit")
            })?;
            Ok(Some(TokenId::new(u32::try_from(index).map_err(|_| {
                FerrumError::backend("expansion selected token exceeds u32")
            })?)))
        }
    }
}

async fn finish(
    scope: &ExecutorScope,
    history: ManagedTokenHistory,
    frontier: ExecutionFrontier,
    prompt_tokens: usize,
    reason: FinishReason,
) -> Result<ExpansionExecution, ExpansionExecutionFailure> {
    let closed = scope
        .finish_tokens(
            frontier.sequence(),
            ExecutionEndReason::Completed {
                input_tokens: prompt_tokens,
                output_tokens: history.ids().len(),
            },
        )
        .await;
    let close = match closed {
        Ok(close) => close,
        Err(error) => {
            let mut failure = ExpansionExecutionFailure::new(error.clone(), reason, Some(history));
            failure.frontier = Some(frontier);
            failure.close_error = Some(error);
            return Err(failure);
        }
    };
    if close.error.is_none() && reason == FinishReason::EOS {
        if let ExecutionCloseDisposition::Ended(ended) = &close.disposition {
            if ended.sequence() == frontier.sequence() {
                return Ok(ExpansionExecution {
                    history,
                    finish: reason,
                    ended: ended.clone(),
                    frontier,
                });
            }
        }
    }
    let error = close.error.clone().unwrap_or_else(|| {
        if reason == FinishReason::Length {
            FerrumError::request_validation(
                "expansion reached max_new_tokens before the protocol EOS",
            )
        } else {
            FerrumError::backend("expansion close did not prove this exact sequence ended")
        }
    });
    let mut failure = ExpansionExecutionFailure::new(error, reason, Some(history));
    failure.admission_probe = AdmissionProbe::Entered;
    failure.frontier = Some(frontier);
    failure.close = Some(close);
    Err(failure)
}

/// A retained admission attempt, before any token frontier has been issued.
/// The original phase supplies its immutable prompt on every advancement;
/// the executor validates that binding when retrying a retained deferral.
pub(super) struct ExpansionAdmission {
    history: ManagedTokenHistory,
    request: RequestId,
    prompt_tokens: usize,
    maximum: usize,
    generation: PH3ImagePromptExpansionGeneration,
    prefill_chunk_limit: usize,
    prior_maintenance: Option<ExecutorAdmissionEpochs>,
    admission: Option<TokenAdmissionOutcome>,
    maintenance: Option<ExecutorPrefillMaintenanceOutcome>,
    inputs: Option<std::sync::Arc<ferrum_execution::CausalRequestInputs>>,
    probe: AdmissionProbe,
}
pub(super) enum ExpansionAdmissionStep {
    Pending(ExpansionAdmission),
    Ready(ExpansionCursor),
}
impl ExpansionAdmission {
    pub(super) fn new(
        scope: &ExecutorScope,
        request: &RequestId,
        prompt: &[TokenId],
        generation: PH3ImagePromptExpansionGeneration,
        prefill_chunk_limit: usize,
        host: &HostMemoryAccount,
    ) -> Result<Self, ExpansionExecutionFailure> {
        let validation = (|| {
            let maximum = prompt
                .len()
                .checked_add(usize::try_from(generation.max_new_tokens).map_err(|_| {
                    FerrumError::request_validation("expansion token bound exceeds usize")
                })?)
                .ok_or_else(|| {
                    FerrumError::request_validation("expansion context bound overflows")
                })?;
            if prompt.is_empty()
                || prefill_chunk_limit == 0
                || generation.max_new_tokens == 0
                || generation.do_sample
                || scope.info().vocab_size == 0
                || maximum > scope.info().max_sequence_length
                || prompt
                    .iter()
                    .any(|t| t.get() as usize >= scope.info().vocab_size)
                || generation.eos_token_id as usize >= scope.info().vocab_size
                || generation.pad_token_id as usize >= scope.info().vocab_size
            {
                return Err(FerrumError::request_validation(
                    "invalid greedy expansion context, protocol tokens or chunk bound",
                ));
            }
            Ok(maximum)
        })();
        let maximum =
            validation.map_err(|e| ExpansionExecutionFailure::new(e, FinishReason::Error, None))?;
        let history = ManagedTokenHistory::new(host, generation.max_new_tokens)
            .map_err(|e| ExpansionExecutionFailure::new(e, FinishReason::Error, None))?;
        Ok(Self {
            history,
            request: request.clone(),
            prompt_tokens: prompt.len(),
            maximum,
            generation,
            prefill_chunk_limit,
            prior_maintenance: None,
            admission: None,
            maintenance: None,
            inputs: None,
            probe: AdmissionProbe::NeverEntered,
        })
    }

    pub(super) fn with_inputs(
        mut self,
        inputs: Option<std::sync::Arc<ferrum_execution::CausalRequestInputs>>,
    ) -> Result<Self, ExpansionExecutionFailure> {
        if inputs
            .as_ref()
            .is_some_and(|i| i.maximum_tokens() != self.maximum)
        {
            return Err(self.failed(
                FerrumError::request_validation(
                    "visual input coverage differs from the admitted causal context",
                ),
                FinishReason::Error,
            ));
        }
        self.inputs = inputs;
        Ok(self)
    }

    fn failed(self, error: FerrumError, finish: FinishReason) -> ExpansionExecutionFailure {
        let mut failure = ExpansionExecutionFailure::new(error, finish, Some(self.history));
        failure.admission = self.admission;
        failure.maintenance = self.maintenance;
        failure.admission_probe = self.probe;
        failure
    }

    /// One authoritative admission probe and at most one backing maintenance.
    /// Returning Pending hands the same history/facts back to the phase worker,
    /// so a changing capacity epoch cannot monopolize a native advancement.
    pub(super) async fn advance<F: Fn() -> bool + Sync + ?Sized>(
        mut self,
        scope: &ExecutorScope,
        prompt: &[TokenId],
        cancelled: &F,
    ) -> Result<ExpansionAdmissionStep, ExpansionExecutionFailure> {
        if cancelled() {
            return Err(self.failed(
                FerrumError::cancelled("expansion cancelled before admission"),
                FinishReason::Cancelled,
            ));
        }
        if prompt.len() != self.prompt_tokens {
            return Err(self.failed(
                FerrumError::request_validation("expansion admission prompt extent changed"),
                FinishReason::Error,
            ));
        }
        let context = match ExecutorPrefillAdmission::for_product_request(
            &self.request,
            prompt,
            self.maximum,
            self.prompt_tokens,
            0,
        ) {
            Ok(context) => context,
            Err(error) => return Err(self.failed(error, FinishReason::Error)),
        };
        let admission = ExecutionAdmission {
            context,
            adapter: None,
        };
        // Even a bare Err after entry is not proof that no executor authority
        // was acquired. Keep this fact across maintenance retries and errors.
        self.probe = AdmissionProbe::Entered;
        let admitted = if let Some(inputs) = &self.inputs {
            scope
                .admit_tokens_with_inputs(admission, std::sync::Arc::clone(inputs))
                .await
        } else {
            scope.admit_tokens(admission).await
        };
        let outcome = match admitted {
            Ok(outcome) => outcome,
            Err(error) => return Err(self.failed(error, FinishReason::Error)),
        };
        match outcome {
            TokenAdmissionOutcome::Admitted(frontier) => {
                if frontier.consumed_tokens() != 0 {
                    return Err(active_failure(
                        scope,
                        self.history,
                        frontier,
                        FinishReason::Error,
                        FerrumError::backend(
                            "fresh expansion admission returned a nonzero frontier",
                        ),
                        None,
                    ));
                }
                Ok(ExpansionAdmissionStep::Ready(ExpansionCursor {
                    history: self.history,
                    frontier,
                    prompt_tokens: self.prompt_tokens,
                    generation: self.generation,
                    prefill_chunk_limit: self.prefill_chunk_limit,
                }))
            }
            TokenAdmissionOutcome::MaintenanceDeferred(ref deferred) => {
                if deferred.request_id() != &self.request {
                    self.admission = Some(outcome);
                    return Err(self.failed(
                        FerrumError::backend(
                            "expansion maintenance deferral belongs to another request",
                        ),
                        FinishReason::Error,
                    ));
                }
                if self.prior_maintenance == Some(deferred.observed()) {
                    self.admission = Some(outcome);
                    return Err(self.failed(
                        FerrumError::resource_exhausted(
                            "expansion maintenance requires a new capacity observation",
                        ),
                        FinishReason::Error,
                    ));
                }
                self.prior_maintenance = Some(deferred.observed());
                self.admission = Some(outcome);
                self.maintenance = None;
                let maintenance = match scope.maintain_prefill_backing(&self.request) {
                    Ok(m) => m,
                    Err(error) => return Err(self.failed(error, FinishReason::Error)),
                };
                let retry = matches!(
                    maintenance,
                    ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                );
                self.maintenance = Some(maintenance);
                if retry {
                    return Ok(ExpansionAdmissionStep::Pending(self));
                }
                Err(self.failed(
                    FerrumError::resource_exhausted(
                        "expansion backing is waiting for original capacity release",
                    ),
                    FinishReason::Error,
                ))
            }
            other => {
                self.admission = Some(other);
                self.maintenance = None;
                Err(self.failed(
                    FerrumError::resource_exhausted(
                        "expansion token admission did not issue a frontier",
                    ),
                    FinishReason::Error,
                ))
            }
        }
    }
}

#[cfg(test)]
pub(super) async fn begin<F: Fn() -> bool + Sync + ?Sized>(
    scope: &ExecutorScope,
    request: &RequestId,
    prompt: &[TokenId],
    generation: PH3ImagePromptExpansionGeneration,
    prefill_chunk_limit: usize,
    host: &HostMemoryAccount,
    cancelled: &F,
) -> Result<ExpansionCursor, ExpansionExecutionFailure> {
    let mut admission = ExpansionAdmission::new(
        scope,
        request,
        prompt,
        generation,
        prefill_chunk_limit,
        host,
    )?;
    loop {
        match admission.advance(scope, prompt, cancelled).await? {
            ExpansionAdmissionStep::Ready(cursor) => return Ok(cursor),
            ExpansionAdmissionStep::Pending(next) => admission = next,
        }
        // The borrowed async wrapper uses the same individual transitions.
        // Wake its actual caller, independent of an ambient Tokio scheduler.
        let mut yielded = false;
        std::future::poll_fn(|context| {
            if yielded {
                return std::task::Poll::Ready(());
            }
            yielded = true;
            context.waker().wake_by_ref();
            std::task::Poll::Pending
        })
        .await;
    }
}

/// A quiescent original token frontier. Its enclosing phase retains the
/// original ExecutorOwner until shutdown; dropping a cursor is not retirement.
pub(super) struct ExpansionCursor {
    history: ManagedTokenHistory,
    frontier: ExecutionFrontier,
    prompt_tokens: usize,
    generation: PH3ImagePromptExpansionGeneration,
    prefill_chunk_limit: usize,
}
pub(super) enum ExpansionStep {
    Continue(ExpansionCursor),
    Complete(ExpansionExecution),
}
impl ExpansionCursor {
    pub(super) fn frontier_tokens(&self) -> usize {
        self.frontier.consumed_tokens()
    }

    #[cfg(test)]
    pub(super) fn consumed_tokens(&self) -> usize {
        self.frontier.consumed_tokens()
    }

    /// Execute at most one actual prefill chunk or decode, waiting for its
    /// original completion before returning custody at a safe step boundary.
    /// Cancellation is observed at boundaries, never by dropping live work.
    pub(super) async fn advance<F: Fn() -> bool + Sync + ?Sized>(
        self,
        scope: &ExecutorScope,
        cancelled: &F,
    ) -> Result<ExpansionStep, ExpansionExecutionFailure> {
        let Self {
            mut history,
            mut frontier,
            prompt_tokens,
            generation,
            prefill_chunk_limit,
        } = self;
        let policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: None,
        };
        if cancelled() {
            return Err(active_failure(
                scope,
                history,
                frontier,
                FinishReason::Cancelled,
                FerrumError::cancelled("expansion cancelled at an execution boundary"),
                None,
            ));
        }
        let consumed = frontier.consumed_tokens();
        let work = if consumed < prompt_tokens {
            PrefillChunk::new(
                consumed,
                prefill_chunk_limit.min(prompt_tokens - consumed),
                prompt_tokens,
            )
            .map(TokenWork::Prefill)
        } else {
            history
                .ids()
                .last()
                .copied()
                .map(|token| TokenWork::Decode {
                    token: TokenId::new(token),
                    position: consumed,
                })
                .ok_or_else(|| {
                    FerrumError::backend("expansion decode has no actual previous token")
                })
        };
        let work = match work {
            Ok(work) => work,
            Err(error) => {
                return Err(active_failure(
                    scope,
                    history,
                    frontier,
                    FinishReason::Error,
                    error,
                    None,
                ));
            }
        };
        let expected_frontier = frontier.clone();
        let input = match ExecutionWork::new(&expected_frontier, work, &policy) {
            Ok(input) => input,
            Err(error) => {
                return Err(active_failure(
                    scope,
                    history,
                    frontier,
                    FinishReason::Error,
                    error,
                    None,
                ));
            }
        };
        // Do not race this future with cancellation. A boundary observation
        // preserves the original executor's submitted/unknown progress facts.
        let outcome = match scope.execute_tokens(std::slice::from_ref(&input)).await {
            Ok(outcome) => outcome,
            Err(error) => {
                return Err(active_failure(
                    scope,
                    history,
                    frontier,
                    FinishReason::Error,
                    error,
                    None,
                ));
            }
        };
        let ExecutionBatchOutcome::Completed(batch) = &outcome else {
            return Err(active_failure(
                scope,
                history,
                frontier,
                FinishReason::Error,
                incomplete_wave_error(&outcome),
                Some(outcome),
            ));
        };
        let [completion] = batch.completions() else {
            return Err(active_failure(
                scope,
                history,
                frontier,
                FinishReason::Error,
                FerrumError::backend("expansion wave returned an invalid completion count"),
                Some(outcome),
            ));
        };
        if let Err(error) = completion.validate_for(0, &input, scope.info().vocab_size) {
            return Err(active_failure(
                scope,
                history,
                frontier,
                FinishReason::Error,
                error,
                Some(outcome),
            ));
        }
        let token = selected(completion.product());
        frontier = completion.next_frontier().clone();
        let token = match token {
            Ok(token) => token,
            Err(error) => {
                return Err(active_failure(
                    scope,
                    history,
                    frontier,
                    FinishReason::Error,
                    error,
                    Some(outcome),
                ));
            }
        };
        if let Some(token) = token {
            if let Err(error) = history.push(token) {
                return Err(active_failure(
                    scope,
                    history,
                    frontier,
                    FinishReason::Error,
                    error,
                    Some(outcome),
                ));
            }
            if cancelled() {
                return Err(active_failure(
                    scope,
                    history,
                    frontier,
                    FinishReason::Cancelled,
                    FerrumError::cancelled("expansion cancelled after completed model work"),
                    Some(outcome),
                ));
            }
            if token.get() == generation.eos_token_id {
                return finish(scope, history, frontier, prompt_tokens, FinishReason::EOS)
                    .await
                    .map(ExpansionStep::Complete);
            }
            if history.ids().len() == generation.max_new_tokens as usize {
                return finish(
                    scope,
                    history,
                    frontier,
                    prompt_tokens,
                    FinishReason::Length,
                )
                .await
                .map(ExpansionStep::Complete);
            }
        }

        Ok(ExpansionStep::Continue(Self {
            history,
            frontier,
            prompt_tokens,
            generation,
            prefill_chunk_limit,
        }))
    }
}
#[cfg(test)]
pub(super) async fn execute<F: Fn() -> bool + Sync + ?Sized>(
    scope: &ExecutorScope,
    request: &RequestId,
    prompt: &[TokenId],
    generation: PH3ImagePromptExpansionGeneration,
    prefill_chunk_limit: usize,
    host: &HostMemoryAccount,
    cancelled: &F,
) -> Result<ExpansionExecution, ExpansionExecutionFailure> {
    let mut cursor = begin(
        scope,
        request,
        prompt,
        generation,
        prefill_chunk_limit,
        host,
        cancelled,
    )
    .await?;
    loop {
        match cursor.advance(scope, cancelled).await? {
            ExpansionStep::Continue(next) => cursor = next,
            ExpansionStep::Complete(done) => return Ok(done),
        }
    }
}
