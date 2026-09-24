//! One exact resource step for admitted prefill chunks and live decode frontiers.
//! Capacity fallback is allowed only before encoding; every submitted participant
//! shares terminal completion and keeps its own product-state transition.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MixedParticipantIndex {
    Prefill(usize),
    Decode(usize),
}

/// Cancellation or a post-submit failure invalidates every decode participant.
/// The corresponding prefill guards own the retained-prefill registry entries.
/// Device completion keeps the physical leases alive until the fence retires.
struct MixedDecodeGuard<'a, R: DeviceRuntime> {
    executor: &'a VNextModelExecutor<R>,
    candidates: &'a [VNextDecodeCandidate<R>],
    armed: bool,
}

impl<R: DeviceRuntime> Drop for MixedDecodeGuard<'_, R> {
    fn drop(&mut self) {
        if self.armed {
            self.executor.abort_decode_candidates(self.candidates);
        }
    }
}

fn validate_mixed_request_ids(
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for request_id in prefills
        .iter()
        .map(|input| &input.request_id)
        .chain(decodes.iter().map(|input| &input.request_id))
    {
        if !seen.insert(request_id) {
            return Err(FerrumError::request_validation(
                "mixed execution contains a duplicate request frontier",
            ));
        }
    }
    Ok(())
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) async fn execute_plan_runtime_mixed_batch(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
        cost_observation: Option<&mut PlanRuntimeCostObservationContext<'_>>,
    ) -> Result<PlanRuntimeMixedBatchOutcome> {
        self.execute_plan_runtime_mixed_batch_inner(prefills, decodes, cost_observation, None)
            .await
    }

    pub(super) async fn execute_plan_runtime_mixed_batch_inner(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
        cost_observation: Option<&mut PlanRuntimeCostObservationContext<'_>>,
        guarded: Option<&GuardedExecution<'_>>,
    ) -> Result<PlanRuntimeMixedBatchOutcome> {
        // Diagnostic checkpoint capture assigns separate phase counters and
        // teacher-forced ownership. Preserve its existing execution contract.
        if prefills.is_empty()
            || decodes.is_empty()
            || self.checkpoint_capture.is_some()
            || self.diagnostic_fault.is_some()
        {
            return Ok(PlanRuntimeMixedBatchOutcome::Unsupported);
        }
        validate_mixed_request_ids(prefills, decodes)?;
        let started = Instant::now();

        let mut parsed_prefills = Vec::with_capacity(prefills.len());
        for input in prefills {
            if input.maximum_sequence_tokens < input.input_tokens.len()
                || input.maximum_sequence_tokens > self.maximum_model_tokens
                || input.chunk.total_prompt_tokens() != input.input_tokens.len()
            {
                return Err(FerrumError::request_validation(
                    "mixed prefill chunk or sequence ceiling differs from its input",
                ));
            }
            parsed_prefills.push(
                input
                    .input_tokens
                    .iter()
                    .map(|token| token.get())
                    .collect::<Vec<_>>(),
            );
        }

        let decode_candidates = decodes
            .iter()
            .enumerate()
            .map(|(original_index, input)| {
                let cache_id = input.kv_cache.cache_id();
                let sequence = self.sequence_for_cache(&cache_id)?;
                if &input.request_id != sequence.request_id() {
                    return Err(FerrumError::request_validation(
                        "mixed decode request differs from its cache owner",
                    ));
                }
                Ok(VNextDecodeCandidate {
                    original_index,
                    sequence,
                    cache_id,
                    next_token: input.input_token.get(),
                    logits_policy: input.logits_policy.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let request_ids = prefills
            .iter()
            .map(|input| input.request_id.clone())
            .collect::<Vec<_>>();
        let executions = self
            .sequences
            .lock()
            .begin_prefill_batch_execution(&request_ids)?;
        let prefill_candidates = prefills
            .iter()
            .zip(parsed_prefills)
            .zip(executions)
            .enumerate()
            .map(
                |(original_index, ((input, tokens), (slot, sequence)))| VNextPrefillCandidate {
                    original_index,
                    slot,
                    sequence,
                    tokens,
                    maximum_tokens: input.maximum_sequence_tokens,
                    planned_chunk: input.chunk,
                },
            )
            .collect::<Vec<_>>();
        let mut prefill_guards = prefill_candidates
            .iter()
            .map(|candidate| {
                VNextPrefillExecutionGuard::new(
                    &self.sequences,
                    Arc::clone(&candidate.slot),
                    Arc::clone(&candidate.sequence),
                )
            })
            .collect::<Vec<_>>();

        let batch = ExecutionBatchParticipants::new(
            prefill_candidates
                .iter()
                .map(|candidate| Arc::clone(&candidate.sequence.session))
                .chain(
                    decode_candidates
                        .iter()
                        .map(|candidate| Arc::clone(&candidate.sequence.session)),
                )
                .collect(),
        )
        .map_err(|error| FerrumError::request_validation(error.to_string()))?;
        let mut indices_by_authority = BTreeMap::new();
        for (index, candidate) in prefill_candidates.iter().enumerate() {
            indices_by_authority.insert(
                candidate.sequence.session.sequence_authority(),
                MixedParticipantIndex::Prefill(index),
            );
        }
        for (index, candidate) in decode_candidates.iter().enumerate() {
            indices_by_authority.insert(
                candidate.sequence.session.sequence_authority(),
                MixedParticipantIndex::Decode(index),
            );
        }
        let canonical_indices = batch
            .sessions()
            .iter()
            .map(|session| {
                indices_by_authority
                    .remove(&session.sequence_authority())
                    .ok_or_else(|| {
                        FerrumError::internal("mixed canonical participant has no input frontier")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let sequences = canonical_indices
            .iter()
            .map(|index| match *index {
                MixedParticipantIndex::Prefill(index) => {
                    Arc::clone(&prefill_candidates[index].sequence)
                }
                MixedParticipantIndex::Decode(index) => {
                    Arc::clone(&decode_candidates[index].sequence)
                }
            })
            .collect::<Vec<_>>();

        // Lock both phases in the runtime's canonical authority order. A phase-
        // specific lock order could deadlock overlapping calls at this boundary.
        let mut operation_guards = Vec::with_capacity(sequences.len());
        for sequence in &sequences {
            operation_guards.push(sequence.operation.lock().await);
            if !sequence.active.load(Ordering::Acquire) {
                return Err(FerrumError::cancelled(
                    "mixed execution sequence is no longer active",
                ));
            }
        }
        let mut token_batches = Vec::with_capacity(sequences.len());
        let mut spans = Vec::with_capacity(sequences.len());
        let mut output_roles = Vec::with_capacity(sequences.len());
        for index in &canonical_indices {
            let (tokens, range, output_role) = match *index {
                MixedParticipantIndex::Prefill(index) => {
                    let candidate = &prefill_candidates[index];
                    if candidate.sequence.request_id() != &candidate.slot.request_id
                        || candidate.sequence.maximum_tokens != candidate.maximum_tokens
                        || *candidate.sequence.tokens.lock() != candidate.tokens
                        || candidate
                            .sequence
                            .prefill_tokens_processed
                            .load(Ordering::Acquire)
                            != candidate.planned_chunk.tokens_processed()
                    {
                        return Err(FerrumError::request_validation(
                            "mixed prefill differs from its admitted token frontier",
                        ));
                    }
                    (
                        candidate.tokens.clone(),
                        candidate.planned_chunk.range(),
                        VNextParticipantOutputRole::prefill(candidate.planned_chunk),
                    )
                }
                MixedParticipantIndex::Decode(index) => {
                    let candidate = &decode_candidates[index];
                    let mut tokens = candidate.sequence.tokens.lock().clone();
                    let previous_len = tokens.len();
                    if previous_len >= candidate.sequence.maximum_tokens {
                        return Err(FerrumError::request_validation(
                            "mixed decode reached its sequence token ceiling",
                        ));
                    }
                    tokens.push(candidate.next_token);
                    let end = tokens.len();
                    (
                        tokens,
                        previous_len..end,
                        VNextParticipantOutputRole::Decode(candidate.logits_policy.clone()),
                    )
                }
            };
            spans.push(
                TokenSpanWork::from_token_ids(&tokens, range)
                    .map_err(|error| FerrumError::backend(error.to_string()))?,
            );
            token_batches.push(tokens);
            output_roles.push(output_role);
        }

        let restore_prefills = |guards: &mut [VNextPrefillExecutionGuard<'_, R>]| -> Result<()> {
            let authority = prefill_candidates
                .iter()
                .map(|candidate| (&candidate.slot, &candidate.sequence))
                .collect::<Vec<_>>();
            self.sequences
                .lock()
                .restore_prefill_batch_ready(&authority)?;
            for guard in guards {
                guard.disarm();
            }
            Ok(())
        };
        for ((sequence, tokens), span) in sequences.iter().zip(&token_batches).zip(&spans) {
            let end = usize::try_from(span.immediate_token_range().end).map_err(|_| {
                FerrumError::backend("mixed token frontier exceeds host address space")
            })?;
            let extension_span = TokenSpanWork::from_token_ids(&tokens[..end], 0..end)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            let extension = ResourceWorkShape::single(extension_span)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            let extension = match guarded {
                Some(selected) => self.guarded_extend_once(sequence, extension, selected),
                None => self.extend_sequence_with_capacity(sequence, extension),
            };
            let extension = match extension {
                Ok(decision) => decision,
                Err(error) => {
                    if let Some(selected) = guarded.filter(|selected| selected.preserves_request())
                    {
                        self.restore_guarded_prefills(
                            &prefill_candidates,
                            &mut prefill_guards,
                            selected,
                        )?;
                    }
                    return Err(error);
                }
            };
            match extension {
                VNextExecutionCapacityDecision::Ready(()) => {}
                VNextExecutionCapacityDecision::Deferred(deferral) => {
                    restore_prefills(&mut prefill_guards)?;
                    return Ok(PlanRuntimeMixedBatchOutcome::NotSubmitted(deferral.into()));
                }
                VNextExecutionCapacityDecision::RequestStateDeferred(_) => {
                    return Err(FerrumError::internal(
                        "mixed sequence extension produced a request-state deferral",
                    ));
                }
            }
        }

        let mut decode_guard = MixedDecodeGuard {
            executor: self,
            candidates: &decode_candidates,
            armed: true,
        };
        let execution = self
            .execute_batch_step(
                &batch,
                &sequences,
                &token_batches,
                &spans,
                VNextExecutionWaveKind::Mixed,
                &output_roles,
                cost_observation,
                guarded,
            )
            .await;
        let execution = match execution {
            Ok(decision) => decision,
            Err(error) => {
                if let Some(selected) = guarded.filter(|selected| selected.preserves_request()) {
                    self.restore_guarded_prefills(
                        &prefill_candidates,
                        &mut prefill_guards,
                        selected,
                    )?;
                    decode_guard.armed = false;
                }
                return Err(error);
            }
        };
        let sampling_outputs = match execution {
            VNextExecutionCapacityDecision::Ready(outputs) => outputs,
            VNextExecutionCapacityDecision::Deferred(deferral) => {
                restore_prefills(&mut prefill_guards)?;
                decode_guard.armed = false;
                return Ok(PlanRuntimeMixedBatchOutcome::NotSubmitted(deferral.into()));
            }
            VNextExecutionCapacityDecision::RequestStateDeferred(deferral) => {
                restore_prefills(&mut prefill_guards)?;
                decode_guard.armed = false;
                return Ok(PlanRuntimeMixedBatchOutcome::NotSubmitted(deferral.into()));
            }
        };
        if sampling_outputs.len() != canonical_indices.len() {
            return Err(FerrumError::internal(
                "mixed execution output count differs from its participant set",
            ));
        }
        for (output, role) in sampling_outputs.iter().zip(&output_roles) {
            match role {
                // The complete device plan still computes an output for this
                // participant. Its readback is discarded internally and does
                // not grant a sampling policy or publish a product token.
                VNextParticipantOutputRole::IntermediatePrefill => {}
                VNextParticipantOutputRole::FinalPrefill => {
                    output.validate_for_policy(
                        &LogitsReturnPolicy::FullLogits,
                        self.io.output_elements,
                    )?;
                }
                VNextParticipantOutputRole::Decode(policy) => {
                    output.validate_for_policy(policy, self.io.output_elements)?;
                }
            }
        }

        // The shared FullPlan fence has retired. Checkpoint copies belong only
        // to prefill boundaries, never to their decode peers' token histories.
        if guarded.is_none() {
            for candidate in &prefill_candidates {
                self.retain_prefill_boundary(
                    &candidate.sequence,
                    &candidate.tokens,
                    candidate.planned_chunk,
                )
                .await?;
            }
        }

        let mut ordered_prefills = (0..prefills.len()).map(|_| None).collect::<Vec<_>>();
        let mut ordered_decodes = (0..decodes.len()).map(|_| None).collect::<Vec<_>>();
        for ((index, output), span) in canonical_indices.iter().zip(sampling_outputs).zip(&spans) {
            match *index {
                MixedParticipantIndex::Prefill(index) => {
                    let candidate = &prefill_candidates[index];
                    let chunk = candidate.planned_chunk;
                    let cache = self.cache_handle(&candidate.sequence, chunk.end());
                    let product = if chunk.is_final() {
                        PlanRuntimePrefillOutput::final_logits(
                            candidate.slot.request_id.clone(),
                            chunk.end(),
                            output.into_full_logits()?,
                            cache,
                        )?
                    } else {
                        PlanRuntimePrefillOutput::intermediate(
                            candidate.slot.request_id.clone(),
                            chunk.end(),
                            cache,
                        )
                    };
                    product.validate_for_completion(
                        &candidate.slot.request_id,
                        chunk,
                        self.io.output_elements,
                    )?;
                    ordered_prefills[candidate.original_index] =
                        Some(PlanRuntimePrefillCompletion::exact(product, chunk));
                }
                MixedParticipantIndex::Decode(index) => {
                    let candidate = &decode_candidates[index];
                    let end = usize::try_from(span.immediate_token_range().end).map_err(|_| {
                        FerrumError::backend("mixed decode frontier exceeds host address space")
                    })?;
                    ordered_decodes[candidate.original_index] = Some(PlanRuntimeDecodeOutput::new(
                        output,
                        self.cache_handle(&candidate.sequence, end),
                    ));
                }
            }
        }
        let prefills = ordered_prefills
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| FerrumError::internal("mixed execution lost a prefill output"))
            })
            .collect::<Result<Vec<_>>>()?;
        let decodes = ordered_decodes
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| FerrumError::internal("mixed execution lost a decode output"))
            })
            .collect::<Result<Vec<_>>>()?;

        let authority = prefill_candidates
            .iter()
            .map(|candidate| {
                (
                    &candidate.slot,
                    &candidate.sequence,
                    candidate.planned_chunk.is_final(),
                )
            })
            .collect::<Vec<_>>();
        for candidate in &prefill_candidates {
            candidate
                .sequence
                .prefill_tokens_processed
                .store(candidate.planned_chunk.end(), Ordering::Release);
        }
        self.sequences
            .lock()
            .commit_prefill_batch_execution(&authority)?;
        for (index, tokens) in canonical_indices.iter().zip(token_batches) {
            if let MixedParticipantIndex::Decode(index) = *index {
                let sequence = &decode_candidates[index].sequence;
                if sequence.active.load(Ordering::Acquire) {
                    *sequence.tokens.lock() = tokens;
                }
            }
        }
        for guard in &mut prefill_guards {
            guard.disarm();
        }
        decode_guard.armed = false;
        self.metrics
            .prefill_operations
            .fetch_add(prefills.len() as u64, Ordering::Relaxed);
        self.metrics
            .decode_operations
            .fetch_add(decodes.len() as u64, Ordering::Relaxed);
        // These status counters report each participant's observed call latency.
        // Physical device time is separately attributed only to the mixed wave.
        let elapsed_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        self.metrics.total_prefill_us.fetch_add(
            elapsed_us.saturating_mul(prefills.len() as u64),
            Ordering::Relaxed,
        );
        self.metrics.total_decode_us.fetch_add(
            elapsed_us.saturating_mul(decodes.len() as u64),
            Ordering::Relaxed,
        );
        // Shared wave timing is reported under "mixed", never added to both
        // phase timers as if it were two disjoint device intervals.
        Ok(PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes })
    }
}

#[cfg(test)]
mod tests;
