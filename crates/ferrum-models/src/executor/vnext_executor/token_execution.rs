//! Token protocol over the original native sessions, steps and completion worker.
//! The weak index names existing owners; it never creates resource authority.

use super::sequence_settlement::WaveProgress;
use super::*;
use ferrum_execution::token_execution::*;

pub(super) struct AdmissionBinding {
    tokens: Vec<u32>,
    maximum_tokens: usize,
    product_prompt_tokens: usize,
    replayed_output_tokens: usize,
    origin: ExecutorRequestOrigin,
}

impl AdmissionBinding {
    fn from_input(input: ExecutorPrefillAdmission<'_>) -> Self {
        Self {
            tokens: input.input_tokens.iter().map(|token| token.get()).collect(),
            maximum_tokens: input.maximum_sequence_tokens,
            product_prompt_tokens: input.product_prompt_tokens,
            replayed_output_tokens: input.replayed_output_tokens,
            origin: input.request_origin,
        }
    }

    fn matches(&self, input: ExecutorPrefillAdmission<'_>) -> bool {
        self.maximum_tokens == input.maximum_sequence_tokens
            && self.product_prompt_tokens == input.product_prompt_tokens
            && self.replayed_output_tokens == input.replayed_output_tokens
            && self.origin == input.request_origin
            && self
                .tokens
                .iter()
                .copied()
                .eq(input.input_tokens.iter().map(|token| token.get()))
    }
}

/// Armed only once immutable validation and the original operation gates have
/// succeeded. Cancellation owns cleanup even before a native wave is created.
struct TokenBatchGuard<'a, R: DeviceRuntime> {
    executor: &'a VNextModelExecutor<R>,
    sequences: &'a [Arc<VNextSequence<R>>],
    armed: bool,
}

fn retain_rejected_products(
    outcome: ExecutionBatchOutcome,
    rejected: Vec<CompletionParts>,
    unassigned: Vec<ExecutionProduct>,
) -> ExecutionBatchOutcome {
    let mut violation = match outcome {
        ExecutionBatchOutcome::ContractViolation(violation) => violation,
        ExecutionBatchOutcome::Interrupted(interruption) => {
            let parts = interruption.into_parts();
            let error = parts.2.first().cloned().unwrap_or_else(|| {
                FerrumError::backend("native token results violated their participant contract")
            });
            ExecutionContractViolation::new(error, parts)
        }
        ExecutionBatchOutcome::Completed(completed) => ExecutionContractViolation::new(
            FerrumError::internal("native rejected products accompanied a completed batch"),
            (completed.into_completions(), Vec::new(), Vec::new()),
        ),
        ExecutionBatchOutcome::NotSubmitted(_) => ExecutionContractViolation::new(
            FerrumError::internal("native retired results were incorrectly marked not submitted"),
            (Vec::new(), Vec::new(), Vec::new()),
        ),
    };
    violation.rejected_completions.extend(rejected);
    violation.unassigned_products.extend(unassigned);
    ExecutionBatchOutcome::ContractViolation(violation)
}

impl<R: DeviceRuntime> Drop for TokenBatchGuard<'_, R> {
    fn drop(&mut self) {
        if self.armed {
            let mut registry = self.executor.sequences.lock();
            for sequence in self.sequences {
                sequence.abort();
                registry.remove_ended_exact(sequence);
            }
        }
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn admit_bound_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
        repeat_ready: bool,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        input.validate()?;
        if input.maximum_sequence_tokens > self.maximum_model_tokens {
            return Err(FerrumError::request_validation(
                "sequence ceiling exceeds the compiled model",
            ));
        }
        if input
            .input_tokens
            .iter()
            .any(|token| token.get() as usize >= self.io.output_elements)
        {
            return Err(FerrumError::request_validation(
                "prompt token is outside the model vocabulary",
            ));
        }
        let tokens = input
            .input_tokens
            .iter()
            .map(|token| token.get())
            .collect::<Vec<_>>();
        let span =
            TokenSpanWork::from_token_ids_with_fit(&tokens, 0..1, input.maximum_sequence_tokens)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
        let work = ResourceWorkShape::single(span)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let slot = {
            let mut registry = self.sequences.lock();
            if let Some(slot) = registry.prefills.get(input.request_id) {
                if slot
                    .admission
                    .lock()
                    .as_ref()
                    .is_some_and(|binding| !binding.matches(input))
                {
                    return Err(FerrumError::request_validation(
                        "prefill retry changed its immutable admission",
                    ));
                }
                if repeat_ready && !slot.cancelled.load(Ordering::Acquire) {
                    if let VNextPrefillSlotState::Ready(sequence) = &*slot.state.lock() {
                        if sequence.token_frontier_is_open()
                            && sequence.prefill_tokens_processed.load(Ordering::Acquire) == 0
                        {
                            return Ok(ExecutorPrefillAdmissionDecision::Admitted(
                                ExecutorPrefillAdmissionReceipt {
                                    request_id: input.request_id.clone(),
                                },
                            ));
                        }
                    }
                }
            }
            let slot = registry.begin_prefill_probe(input.request_id, &work)?;
            let mut binding = slot.admission.lock();
            if binding.is_none() {
                *binding = Some(AdmissionBinding::from_input(input));
            }
            drop(binding);
            slot
        };
        let resolution = match self.resolve_prefill_probe(
            input.request_id,
            input.request_origin,
            input.maximum_sequence_tokens,
            tokens,
            input.product_prompt_tokens,
            input.replayed_output_tokens,
            work,
        ) {
            Ok(resolution) => resolution,
            Err(error) => {
                self.fail_prefill_probe(&slot);
                return Err(error);
            }
        };
        self.publish_prefill_probe(&slot, resolution)
    }

    pub(super) fn admit_native_tokens(
        &self,
        input: ExecutionAdmission<'_>,
    ) -> Result<TokenAdmissionOutcome> {
        input.validate()?;
        if input.adapter.is_some() {
            return Err(FerrumError::unsupported(
                "native token execution does not implement LoRA",
            ));
        }
        match self.admit_bound_prefill(input.context, true)? {
            ExecutorPrefillAdmissionDecision::Admitted(_) => {
                let registry = self.sequences.lock();
                let slot = registry
                    .prefills
                    .get(input.context.request_id)
                    .ok_or_else(|| {
                        FerrumError::cancelled("token admission was cancelled before publication")
                    })?;
                let state = slot.state.lock();
                let VNextPrefillSlotState::Ready(sequence) = &*state else {
                    return Err(FerrumError::cancelled(
                        "token admission no longer owns a ready session",
                    ));
                };
                Ok(TokenAdmissionOutcome::Admitted(
                    self.token_issuer.frontier(
                        &sequence.token_identity,
                        sequence.prefill_tokens_processed.load(Ordering::Acquire),
                    )?,
                ))
            }
            ExecutorPrefillAdmissionDecision::Deferred(value) => {
                Ok(TokenAdmissionOutcome::Deferred(value))
            }
            ExecutorPrefillAdmissionDecision::MaintenanceDeferred(value) => {
                Ok(TokenAdmissionOutcome::MaintenanceDeferred(value))
            }
            ExecutorPrefillAdmissionDecision::PermanentRejected(value) => {
                Ok(TokenAdmissionOutcome::PermanentRejected(value))
            }
        }
    }

    fn sequence_for_token(&self, identity: &ExecutorSequence) -> Result<Arc<VNextSequence<R>>> {
        self.token_issuer.frontier(identity, 0)?;
        self.sequences
            .lock()
            .token_index
            .get(identity)
            .and_then(Weak::upgrade)
            .ok_or_else(|| {
                FerrumError::request_validation(
                    "token incarnation is not owned by this live executor",
                )
            })
    }

    /// A zero-submit capacity result does not by itself preserve continuation.
    /// The caller holds all original operation gates; validate the whole cohort
    /// before changing any prefill slot back to Ready.
    pub(super) fn restore_token_batch_retryable(
        &self,
        inputs: &[ExecutionWork<'_>],
        sequences: &[Arc<VNextSequence<R>>],
        indices: &[usize],
        slots: &[Option<Arc<VNextPrefillSlot<R>>>],
    ) -> Result<()> {
        let mut registry = self.sequences.lock();
        for ((sequence, &index), slot) in sequences.iter().zip(indices).zip(slots) {
            let input = &inputs[index];
            if !sequence.token_frontier_is_open()
                || input.sequence() != &sequence.token_identity
                || !registry
                    .token_index
                    .get(input.sequence())
                    .is_some_and(|owner| owner.ptr_eq(&Arc::downgrade(sequence)))
            {
                return Err(FerrumError::cancelled(
                    "capacity deferral lost an open token incarnation",
                ));
            }
            let matches_frontier = match (input.work(), slot) {
                (TokenWork::Prefill(chunk), Some(slot)) => {
                    chunk.total_prompt_tokens() == sequence.tokens.lock().len()
                        && chunk.tokens_processed()
                            == sequence.prefill_tokens_processed.load(Ordering::Acquire)
                        && input.expected_frontier().consumed_tokens() == chunk.tokens_processed()
                        && !slot.cancelled.load(Ordering::Acquire)
                        && registry
                            .prefills
                            .get(sequence.request_id())
                            .is_some_and(|current| Arc::ptr_eq(current, slot))
                        && matches!(&*slot.state.lock(), VNextPrefillSlotState::Executing(current)
                            if Arc::ptr_eq(current, sequence))
                }
                (TokenWork::Decode { position, .. }, None) => {
                    position == sequence.tokens.lock().len()
                        && input.expected_frontier().consumed_tokens() == position
                        && registry
                            .active
                            .get(&sequence.cache_id)
                            .is_some_and(|current| Arc::ptr_eq(current, sequence))
                }
                _ => false,
            };
            if !matches_frontier {
                return Err(FerrumError::request_validation(
                    "capacity deferral did not preserve the original token frontier",
                ));
            }
        }
        for (slot, sequence) in slots.iter().zip(sequences) {
            if let Some(slot) = slot {
                registry.restore_prefill_ready(slot, sequence)?;
            }
        }
        Ok(())
    }

    pub(super) fn cancel_native_tokens(
        &self,
        identity: &ExecutorSequence,
    ) -> Result<ExecutionCloseOutcome> {
        let sequence = self.sequence_for_token(identity)?;
        self.release_resident_decode();
        sequence.abort();
        // This strong reference retains the real terminal state even if the
        // worker removes the weak index first. Held registers before return.
        let observation = sequence.token_close_observation()?;
        self.sequences.lock().remove_ended_exact(&sequence);
        Ok(observation)
    }

    pub(super) async fn finish_native_tokens(
        &self,
        identity: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        if reason == ExecutionEndReason::Cancelled {
            return self.cancel_native_tokens(identity);
        }
        let sequence = self.sequence_for_token(identity)?;
        let ExecutionEndReason::Completed {
            input_tokens,
            output_tokens,
        } = reason
        else {
            unreachable!()
        };
        let completion = ExecutorSequenceCompletion::new(
            sequence.request_id().clone(),
            sequence.cache_id.clone(),
            input_tokens,
            output_tokens,
        )?;
        validate_sequence_completion_accounting(
            sequence.request_id(),
            sequence.product_prompt_tokens,
            sequence.replayed_output_tokens,
            &completion,
        )?;
        self.release_resident_decode();
        sequence.begin_closing();
        let mut pending = PendingSequenceCompletion {
            sequence: &sequence,
            operation: None,
            completed: false,
        };
        pending.operation = Some(sequence.operation.lock().await);
        let completion_result = match self.retain_completed_sequence_boundary(&sequence).await {
            Ok(()) => sequence.complete(&completion),
            Err(error) => Err(error),
        };
        if completion_result.is_err() && sequence.settlement.lock().terminal.is_none() {
            sequence.abort();
        }
        let observation = sequence.token_close_observation()?;
        self.sequences.lock().remove_ended_exact(&sequence);
        pending.completed = true;
        Ok(match completion_result {
            Ok(()) => observation,
            Err(error) => observation.with_error(error),
        })
    }

    fn interrupted_native_tokens(
        &self,
        scope: &ferrum_execution::token_execution::ExecutionBatch<'_>,
        inputs: &[ExecutionWork<'_>],
        sequences: &[Arc<VNextSequence<R>>],
        indices: &[usize],
        actual: &[ExecutedTokenWork],
        mut completed: Vec<ExecutionCompletion>,
        error: FerrumError,
        progress: &WaveProgress,
    ) -> ExecutionBatchOutcome {
        let mut remaining = Vec::with_capacity(inputs.len().saturating_sub(completed.len()));
        let mut causes = vec![error];
        let mut violation: Option<ExecutionContractViolation> = None;
        for (canonical, sequence) in sequences.iter().enumerate() {
            let index = indices[canonical];
            if completed
                .iter()
                .any(|completion| completion.input_index() == index)
            {
                continue;
            }
            sequence.abort();
            let close = match sequence.token_close_observation() {
                Ok(close) => close,
                Err(error) => {
                    violation.get_or_insert_with(|| {
                        ExecutionContractViolation::new(
                            FerrumError::internal(
                                "native close observation violated its original issuer binding",
                            ),
                            (Vec::new(), Vec::new(), Vec::new()),
                        )
                    });
                    causes.push(error);
                    // The original registry/reaper still owns this sequence;
                    // a broken identity contract grants no release permission.
                    continue;
                }
            };
            self.sequences.lock().remove_ended_exact(sequence);
            let ended = matches!(&close.disposition, ExecutionCloseDisposition::Ended(_));
            if let Some(error) = close.error {
                causes.push(error);
            }
            let range = match actual[canonical].range() {
                Ok(range) => range,
                Err(error) => {
                    // Actual work was checked before preparing the native wave.
                    // Keep the original close observer if this invariant fails.
                    let continuation = match close.disposition {
                        ExecutionCloseDisposition::Ended(ended) => {
                            ContinuationDisposition::Ended(ended)
                        }
                        ExecutionCloseDisposition::Held(held) => {
                            ContinuationDisposition::Held(held)
                        }
                    };
                    let broken = violation.get_or_insert_with(|| {
                        ExecutionContractViolation::new(
                            FerrumError::internal("native actual token range became invalid"),
                            (Vec::new(), Vec::new(), Vec::new()),
                        )
                    });
                    broken.rejected_participants.push((
                        index,
                        ExecutionProgress::Unknown {
                            last_known: LastKnownExecutionStage::EnteredSubmit,
                            known_executed_range: None,
                        },
                        continuation,
                        0,
                    ));
                    causes.push(error);
                    continue;
                }
            };
            let state = match progress {
                WaveProgress::BeforeSubmit => ExecutionProgress::NotSubmitted {
                    stage: NoSubmissionStage::BeforeSubmit,
                },
                WaveProgress::Terminal(receipt) => {
                    let succeeded = matches!(
                        receipt.disposition(),
                        OperationCompletionDisposition::Succeeded
                    );
                    ExecutionProgress::Terminal {
                        disposition: if succeeded {
                            TerminalModelDisposition::Successful
                        } else {
                            TerminalModelDisposition::Failed
                        },
                        executed_range: succeeded.then_some(range),
                    }
                }
                WaveProgress::Retired(receipt) => {
                    let committed = receipt.participants().iter().any(|participant| {
                        participant.assignment().sequence_authority()
                            == sequence.session.sequence_authority()
                            && participant.disposition()
                                == StepParticipantRetirementDisposition::Committed
                    });
                    ExecutionProgress::Terminal {
                        disposition: if committed {
                            TerminalModelDisposition::Successful
                        } else {
                            TerminalModelDisposition::UnknownAfterQuiescence
                        },
                        executed_range: committed.then_some(range),
                    }
                }
                WaveProgress::Drained(_) => ExecutionProgress::Terminal {
                    disposition: TerminalModelDisposition::UnknownAfterQuiescence,
                    executed_range: None,
                },
                WaveProgress::Unknown if ended => ExecutionProgress::Terminal {
                    disposition: TerminalModelDisposition::UnknownAfterQuiescence,
                    executed_range: None,
                },
                WaveProgress::Unknown => ExecutionProgress::Unknown {
                    last_known: LastKnownExecutionStage::EnteredSubmit,
                    known_executed_range: None,
                },
            };
            let continuation = match close.disposition {
                ExecutionCloseDisposition::Ended(ended) => ContinuationDisposition::Ended(ended),
                ExecutionCloseDisposition::Held(held) => ContinuationDisposition::Held(held),
            };
            match scope.participant(index, state, continuation, 0) {
                Ok(participant) => remaining.push(participant),
                Err(rejected) => {
                    let (error, parts) = rejected.into_parts();
                    let broken = violation.get_or_insert_with(|| {
                        ExecutionContractViolation::new(
                            FerrumError::internal(
                                "native participant report violated the batch contract",
                            ),
                            (Vec::new(), Vec::new(), Vec::new()),
                        )
                    });
                    broken.rejected_participants.push(parts);
                    causes.push(error);
                }
            }
        }
        completed.sort_by_key(ExecutionCompletion::input_index);
        remaining.sort_by_key(InterruptedParticipant::input_index);
        if let Some(mut violation) = violation {
            violation.completed = completed;
            violation.remaining = remaining;
            violation.causes = causes;
            return ExecutionBatchOutcome::ContractViolation(violation);
        }
        match scope.interrupted(completed, remaining, causes) {
            Ok(outcome) => outcome,
            Err(rejected) => rejected.into_outcome(),
        }
    }

    pub(super) async fn execute_native_tokens(
        &self,
        inputs: &[ExecutionWork<'_>],
    ) -> Result<ExecutionBatchOutcome> {
        self.execute_native_tokens_inner(
            inputs,
            #[cfg(test)]
            None,
        )
        .await
    }

    pub(super) async fn execute_native_tokens_inner(
        &self,
        inputs: &[ExecutionWork<'_>],
        #[cfg(test)] inspect_output: Option<fn(&mut Vec<ExecutorSamplingOutput>)>,
    ) -> Result<ExecutionBatchOutcome> {
        let scope = self.token_issuer.batch(inputs, self.io.output_elements)?;
        let candidates = inputs
            .iter()
            .map(|input| self.sequence_for_token(input.sequence()))
            .collect::<Result<Vec<_>>>()?;
        let batch = ExecutionBatchParticipants::new(
            candidates
                .iter()
                .map(|sequence| Arc::clone(&sequence.session))
                .collect(),
        )
        .map_err(|error| FerrumError::request_validation(error.to_string()))?;
        let mut by_authority = candidates
            .into_iter()
            .enumerate()
            .map(|(index, sequence)| (sequence.session.sequence_authority(), (index, sequence)))
            .collect::<BTreeMap<_, _>>();
        let mut indices = Vec::with_capacity(inputs.len());
        let mut sequences = Vec::with_capacity(inputs.len());
        for session in batch.sessions() {
            let (index, sequence) = by_authority
                .remove(&session.sequence_authority())
                .ok_or_else(|| FerrumError::internal("canonical token participant is absent"))?;
            indices.push(index);
            sequences.push(sequence);
        }
        let mut operations = Vec::with_capacity(sequences.len());
        for sequence in &sequences {
            operations.push(sequence.operation.lock().await);
        }

        let mut token_batches = Vec::with_capacity(sequences.len());
        let mut actual = Vec::with_capacity(sequences.len());
        let mut slots = Vec::with_capacity(sequences.len());
        let mut has_prefill = false;
        let mut has_decode = false;
        // Every participant is checked before any resource preparation or slot mutation.
        for (sequence, &index) in sequences.iter().zip(&indices) {
            let input = &inputs[index];
            if !sequence.active.load(Ordering::Acquire) {
                return Err(FerrumError::cancelled("token sequence is closing"));
            }
            let mut tokens = sequence.tokens.lock().clone();
            let (executed, slot) = match input.work() {
                TokenWork::Prefill(chunk) => {
                    has_prefill = true;
                    if chunk.total_prompt_tokens() != tokens.len()
                        || chunk.tokens_processed()
                            != sequence.prefill_tokens_processed.load(Ordering::Acquire)
                    {
                        return Err(FerrumError::request_validation(
                            "prefill does not match the live admitted frontier",
                        ));
                    }
                    let registry = self.sequences.lock();
                    let slot = registry
                        .prefills
                        .get(sequence.request_id())
                        .cloned()
                        .ok_or_else(|| {
                            FerrumError::request_validation("prefill incarnation is already active")
                        })?;
                    if slot.cancelled.load(Ordering::Acquire)
                        || !matches!(&*slot.state.lock(), VNextPrefillSlotState::Ready(current) if Arc::ptr_eq(current, sequence))
                    {
                        return Err(FerrumError::request_validation(
                            "prefill incarnation is not ready",
                        ));
                    }
                    (
                        ExecutedTokenWork::Prefill {
                            chunk,
                            capacity_probe_count: 0,
                        },
                        Some(slot),
                    )
                }
                TokenWork::Decode { token, position } => {
                    has_decode = true;
                    if position != tokens.len()
                        || position >= sequence.maximum_tokens
                        || !self
                            .sequences
                            .lock()
                            .active
                            .get(&sequence.cache_id)
                            .is_some_and(|current| Arc::ptr_eq(current, sequence))
                    {
                        return Err(FerrumError::request_validation(
                            "decode does not match a live final-prefill frontier",
                        ));
                    }
                    tokens.push(token.get());
                    (ExecutedTokenWork::Decode { position }, None)
                }
            };
            if input.expected_frontier().consumed_tokens() != executed.range()?.start() {
                return Err(FerrumError::request_validation("stale token frontier"));
            }
            token_batches.push(tokens);
            actual.push(executed);
            slots.push(slot);
        }
        let kind = match (has_prefill, has_decode) {
            (true, true) => VNextExecutionWaveKind::Mixed,
            (true, false) => VNextExecutionWaveKind::Prefill,
            (false, true) => VNextExecutionWaveKind::Decode,
            (false, false) => unreachable!("batch scope rejects empty work"),
        };
        if kind == VNextExecutionWaveKind::Mixed
            && (self.checkpoint_capture.is_some() || self.diagnostic_fault.is_some())
        {
            return Ok(scope.not_submitted(NoSubmissionReason::UnsupportedShape));
        }
        let started = Instant::now();
        // Freeze slot ownership for the entire native wave. Existing operation
        // gates were acquired in the runtime's canonical order above.
        {
            let _registry = self.sequences.lock();
            for (slot, sequence) in slots.iter().zip(&sequences) {
                if !sequence.active.load(Ordering::Acquire) || slot.as_ref().is_some_and(|slot| {
                    slot.cancelled.load(Ordering::Acquire)
                        || !matches!(&*slot.state.lock(), VNextPrefillSlotState::Ready(current) if Arc::ptr_eq(current, sequence))
                }) {
                    return Err(FerrumError::cancelled("token participant changed before preparation"));
                }
            }
            for (slot, sequence) in slots.iter().zip(&sequences) {
                if let Some(slot) = slot {
                    *slot.state.lock() = VNextPrefillSlotState::Executing(Arc::clone(sequence));
                }
            }
        }
        let mut cancellation = TokenBatchGuard {
            executor: self,
            sequences: &sequences,
            armed: true,
        };
        let restore = || self.restore_token_batch_retryable(inputs, &sequences, &indices, &slots);

        let retired = loop {
            let mut spans = Vec::with_capacity(sequences.len());
            let mut roles = Vec::with_capacity(sequences.len());
            let mut deferral = None;
            let mut failure = None;
            let mut narrower = None;
            for (canonical, ((sequence, tokens), work)) in sequences
                .iter()
                .zip(&token_batches)
                .zip(&actual)
                .enumerate()
            {
                let end = work.range()?.end();
                let extension_span = TokenSpanWork::from_token_ids(&tokens[..end], 0..end)
                    .map_err(|error| FerrumError::backend(error.to_string()));
                let extension = extension_span.and_then(|span| {
                    ResourceWorkShape::single(span)
                        .map_err(|error| FerrumError::backend(error.to_string()))
                });
                match extension
                    .and_then(|extension| self.extend_sequence_with_capacity(sequence, extension))
                {
                    Ok(VNextExecutionCapacityDecision::Ready(())) => {}
                    Ok(VNextExecutionCapacityDecision::Deferred(value)) => {
                        if actual.len() == 1 {
                            if let ExecutedTokenWork::Prefill {
                                chunk,
                                capacity_probe_count,
                            } = *work
                            {
                                if let Some(width) =
                                    value.narrower_prefill_tokens(chunk.tokens_to_process())
                                {
                                    narrower = Some((chunk, width, capacity_probe_count));
                                }
                            }
                        }
                        deferral = Some(ExecutorExecutionDeferral::from(value));
                        break;
                    }
                    Ok(VNextExecutionCapacityDecision::RequestStateDeferred(value)) => {
                        deferral = Some(ExecutorExecutionDeferral::from(value));
                        break;
                    }
                    Err(error) => {
                        failure = Some(WaveFailure::not_submitted(error));
                        break;
                    }
                }
                match TokenSpanWork::from_token_ids_with_fit(
                    tokens,
                    work.range()?.as_range(),
                    sequence.maximum_tokens,
                ) {
                    Ok(span) => spans.push(span),
                    Err(error) => {
                        failure = Some(WaveFailure::not_submitted(FerrumError::backend(
                            error.to_string(),
                        )));
                        break;
                    }
                }
                roles.push(match *work {
                    ExecutedTokenWork::Prefill { chunk, .. } => {
                        VNextParticipantOutputRole::prefill(chunk)
                    }
                    ExecutedTokenWork::Decode { .. } => VNextParticipantOutputRole::Decode(
                        inputs[indices[canonical]].logits_policy().clone(),
                    ),
                });
            }
            if let Some((chunk, width, probes)) = narrower {
                actual[0] = ExecutedTokenWork::Prefill {
                    chunk: PrefillChunk::new(
                        chunk.tokens_processed(),
                        width,
                        chunk.total_prompt_tokens(),
                    )?,
                    capacity_probe_count: probes
                        .checked_add(1)
                        .ok_or_else(|| FerrumError::internal("prefill probe overflow"))?,
                };
                self.metrics
                    .prefill_frontier_narrowings
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let decision = if let Some(failure) = failure {
                Err(failure)
            } else if let Some(deferral) = deferral {
                match restore() {
                    Ok(()) => {
                        cancellation.armed = false;
                        return Ok(scope.not_submitted(NoSubmissionReason::Deferred(deferral)));
                    }
                    Err(error) => Err(WaveFailure::not_submitted(error)),
                }
            } else {
                self.execute_batch_step(&batch, &sequences, &token_batches, &spans, kind, &roles)
                    .await
            };
            match decision {
                Ok(VNextExecutionCapacityDecision::Ready(retired)) => break retired,
                Ok(VNextExecutionCapacityDecision::Deferred(deferred)) => {
                    // Reuse the existing legal adaptive single-prefill narrowing.
                    if actual.len() == 1 {
                        if let ExecutedTokenWork::Prefill {
                            chunk,
                            capacity_probe_count,
                        } = actual[0]
                        {
                            if let Some(width) =
                                deferred.narrower_prefill_tokens(chunk.tokens_to_process())
                            {
                                actual[0] = ExecutedTokenWork::Prefill {
                                    chunk: PrefillChunk::new(
                                        chunk.tokens_processed(),
                                        width,
                                        chunk.total_prompt_tokens(),
                                    )?,
                                    capacity_probe_count: capacity_probe_count
                                        .checked_add(1)
                                        .ok_or_else(|| {
                                            FerrumError::internal("prefill probe overflow")
                                        })?,
                                };
                                self.metrics
                                    .prefill_frontier_narrowings
                                    .fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                        }
                    }
                    match restore() {
                        Ok(()) => {
                            cancellation.armed = false;
                            return Ok(
                                scope.not_submitted(NoSubmissionReason::Deferred(deferred.into()))
                            );
                        }
                        Err(error) => {
                            drop(cancellation);
                            return Ok(self.interrupted_native_tokens(
                                &scope,
                                inputs,
                                &sequences,
                                &indices,
                                &actual,
                                Vec::new(),
                                error,
                                &WaveProgress::BeforeSubmit,
                            ));
                        }
                    }
                }
                Ok(VNextExecutionCapacityDecision::RequestStateDeferred(deferred)) => {
                    match restore() {
                        Ok(()) => {
                            cancellation.armed = false;
                            return Ok(
                                scope.not_submitted(NoSubmissionReason::Deferred(deferred.into()))
                            );
                        }
                        Err(error) => {
                            drop(cancellation);
                            return Ok(self.interrupted_native_tokens(
                                &scope,
                                inputs,
                                &sequences,
                                &indices,
                                &actual,
                                Vec::new(),
                                error,
                                &WaveProgress::BeforeSubmit,
                            ));
                        }
                    }
                }
                Err(failure) => {
                    let (error, progress) = failure.into_parts();
                    drop(cancellation);
                    return Ok(self.interrupted_native_tokens(
                        &scope,
                        inputs,
                        &sequences,
                        &indices,
                        &actual,
                        Vec::new(),
                        error,
                        &progress,
                    ));
                }
            }
        };
        let RetiredWave { output, retirement } = retired;
        #[cfg(test)]
        let output = {
            let mut output = output;
            if let Some(inspect) = inspect_output {
                inspect(&mut output);
            }
            output
        };
        if output.len() != sequences.len()
            || retirement.receipt().participants().len() != sequences.len()
        {
            let receipt = retirement.finish();
            drop(cancellation);
            let outcome = self.interrupted_native_tokens(
                &scope,
                inputs,
                &sequences,
                &indices,
                &actual,
                Vec::new(),
                FerrumError::backend("native token wave returned incomplete participant results"),
                &WaveProgress::Retired(receipt),
            );
            return Ok(retain_rejected_products(
                outcome,
                Vec::new(),
                output.into_iter().map(ExecutionProduct::Sampling).collect(),
            ));
        }
        let mut completed = Vec::with_capacity(sequences.len());
        // Empty on the normal path; malformed rows allocate only on failure.
        let mut rejected_products = Vec::new();
        let mut first_error = None;
        for (canonical, sampling) in output.into_iter().enumerate() {
            let sequence = &sequences[canonical];
            let index = indices[canonical];
            let work = actual[canonical];
            let participant = &retirement.receipt().participants()[canonical];
            if participant.assignment().sequence_authority()
                != sequence.session.sequence_authority()
            {
                first_error.get_or_insert_with(|| {
                    FerrumError::backend("native retirement names a different token participant")
                });
                rejected_products.push((index, work, ExecutionProduct::Sampling(sampling)));
                continue;
            }
            if participant.disposition() != StepParticipantRetirementDisposition::Committed
                || !sequence.active.load(Ordering::Acquire)
            {
                first_error.get_or_insert_with(|| {
                    FerrumError::cancelled(
                        "native token participant was discarded before product commit",
                    )
                });
                continue;
            }
            let product = match work {
                ExecutedTokenWork::Prefill { chunk, .. } if !chunk.is_final() => {
                    ExecutionProduct::Intermediate
                }
                _ => ExecutionProduct::Sampling(sampling),
            };
            let completion = match scope.completion(index, work, product) {
                Ok(completion) => completion,
                Err(rejected) => {
                    let (error, parts) = rejected.into_parts();
                    first_error.get_or_insert(error);
                    rejected_products.push(parts);
                    continue;
                }
            };
            let commit = match work {
                ExecutedTokenWork::Prefill { chunk, .. } => {
                    match self
                        .retain_prefill_boundary(sequence, &token_batches[canonical], chunk)
                        .await
                    {
                        Ok(()) => {
                            let slot = slots[canonical]
                                .as_ref()
                                .expect("validated prefill has its original slot");
                            let transition = if chunk.is_final() {
                                self.sequences.lock().activate(slot, sequence)
                            } else {
                                self.sequences.lock().restore_prefill_ready(slot, sequence)
                            };
                            if transition.is_ok() {
                                sequence
                                    .prefill_tokens_processed
                                    .store(chunk.end(), Ordering::Release);
                            }
                            transition
                        }
                        Err(error) => Err(error),
                    }
                }
                ExecutedTokenWork::Decode { .. } => {
                    *sequence.tokens.lock() = std::mem::take(&mut token_batches[canonical]);
                    Ok(())
                }
            };
            if let Err(error) = commit {
                first_error.get_or_insert(error);
                let (index, _, work, product) = completion.into_parts();
                rejected_products.push((index, work, product));
                continue;
            }
            match work {
                ExecutedTokenWork::Prefill { .. } => {
                    self.metrics
                        .prefill_operations
                        .fetch_add(1, Ordering::Relaxed);
                }
                ExecutedTokenWork::Decode { .. } => {
                    self.metrics
                        .decode_operations
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            completed.push(completion);
        }
        let receipt = retirement.finish();
        cancellation.armed = false;
        if let Some(error) = first_error {
            let outcome = self.interrupted_native_tokens(
                &scope,
                inputs,
                &sequences,
                &indices,
                &actual,
                completed,
                error,
                &WaveProgress::Retired(receipt),
            );
            return Ok(if rejected_products.is_empty() {
                outcome
            } else {
                retain_rejected_products(outcome, rejected_products, Vec::new())
            });
        }
        let elapsed = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        let prefills = actual
            .iter()
            .filter(|work| matches!(work, ExecutedTokenWork::Prefill { .. }))
            .count() as u64;
        self.metrics
            .total_prefill_us
            .fetch_add(elapsed.saturating_mul(prefills), Ordering::Relaxed);
        self.metrics.total_decode_us.fetch_add(
            elapsed.saturating_mul(actual.len() as u64 - prefills),
            Ordering::Relaxed,
        );
        completed.sort_by_key(ExecutionCompletion::input_index);
        Ok(match scope.completed(completed) {
            Ok(outcome) => outcome,
            Err(rejected) => rejected.into_outcome(),
        })
    }
}
