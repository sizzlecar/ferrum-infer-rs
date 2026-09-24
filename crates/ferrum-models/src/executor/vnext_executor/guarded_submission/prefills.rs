//! Exact admitted prefill and mixed waves. The ordinary adaptive path remains
//! separate: a selected wave neither narrows its chunk nor copies checkpoints.
use super::*;
use ferrum_interfaces::execution_cost::ExpectedWaveInput;

fn finish<T, R: DeviceRuntime>(
    executor: &VNextModelExecutor<R>,
    selected: &GuardedExecution<'_>,
    observation: GuardedCostObservation<'_, '_>,
    outcome: GuardedDispatchOutcome<T>,
) -> GuardedDispatchOutcome<T> {
    let outcome = if selected.replan_before_encode.load(Ordering::Acquire) {
        GuardedDispatchOutcome::ReplanBeforeEncode
    } else if let Some(receipt) = selected.reconciled.lock().take() {
        GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt)
    } else {
        executor.attach_guarded_maintenance(selected, outcome)
    };
    if let Some(observation) = observation {
        observation.finish_call(match &outcome {
            GuardedDispatchOutcome::Submitted(Ok(_)) => ObservedCallOutcome::Completed,
            GuardedDispatchOutcome::Submitted(Err(_)) => ObservedCallOutcome::Failed,
            GuardedDispatchOutcome::Deferred(_)
            | GuardedDispatchOutcome::MaintenanceDeferred { .. } => ObservedCallOutcome::Deferred,
            GuardedDispatchOutcome::Unsupported
            | GuardedDispatchOutcome::ReplanBeforeEncode
            | GuardedDispatchOutcome::NotSubmittedAfterPreparation(_) => {
                ObservedCallOutcome::NotSubmitted
            }
        });
    }
    outcome
}

/// Inputs may arrive in phase order; the expected rows are in physical authority
/// order. Match full identities exactly, then the common dispatch rechecks that
/// canonical order against the actual participants at the commit boundary.
fn matches_inputs(
    expected: &ExpectedExecutionWave,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
) -> bool {
    if expected.work().participants().len() != prefills.len() + decodes.len() {
        return false;
    }
    let prefill_match =
        |input: &PlanRuntimePrefillInput,
         row: &ferrum_interfaces::execution_cost::ExpectedWorkParticipant| {
            let row = row.selection();
            input.request_id == row.request_id
                && row.input == ExpectedWaveInput::Prefill { chunk: input.chunk }
        };
    let decode_match =
        |input: &PlanRuntimeDecodeInput,
         row: &ferrum_interfaces::execution_cost::ExpectedWorkParticipant| {
            let row = row.selection();
            input.request_id == row.request_id
                && matches!(&row.input, ExpectedWaveInput::Decode { cache_id }
                    if cache_id == &input.kv_cache.cache_id())
                && row
                    .decode_policy
                    .as_ref()
                    .is_some_and(|policy| policy.same_captured_input(&input.logits_policy))
        };
    // Check both directions: repeated inputs cannot impersonate a missing row.
    expected.work().participants().iter().all(|row| {
        prefills
            .iter()
            .filter(|input| prefill_match(input, row))
            .count()
            + decodes
                .iter()
                .filter(|input| decode_match(input, row))
                .count()
            == 1
    }) && prefills.iter().all(|input| {
        expected
            .work()
            .participants()
            .iter()
            .any(|row| prefill_match(input, row))
    }) && decodes.iter().all(|input| {
        expected
            .work()
            .participants()
            .iter()
            .any(|row| decode_match(input, row))
    })
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    fn supports_guarded_prefill_wave(&self, expected: &ExpectedExecutionWave) -> bool {
        self.supports_guarded_work(expected)
            // Boundary retention can submit an independent CheckpointTransfer.
            // Until that maintenance has its own controller phase, reject this
            // policy before changing any retained-prefill registry state.
            && self.resolved_plan.execution_plan().payload().memory().checkpoint_capacity().is_none()
    }

    pub(in crate::executor::vnext_executor) fn restore_guarded_prefills(
        &self,
        candidates: &[VNextPrefillCandidate<R>],
        guards: &mut [VNextPrefillExecutionGuard<'_, R>],
        selected: &GuardedExecution<'_>,
    ) -> Result<()> {
        let result = restore_prefill_execution_batch(&self.sequences, candidates, guards);
        if result.is_err() {
            selected.invalidate_preservation();
        }
        result
    }

    pub(in crate::executor::vnext_executor) async fn execute_guarded_prefill(
        &self,
        inputs: &[PlanRuntimePrefillInput],
        expected: &ExpectedExecutionWave,
        host: &dyn NonblockingHostSubmissionGuard,
        mut observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<Vec<PlanRuntimePrefillCompletion>> {
        if !self.supports_guarded_prefill_wave(expected)
            || expected.work().kind() != ActualWaveKind::Prefill
            || inputs.is_empty()
            || !matches_inputs(expected, inputs, &[])
        {
            return GuardedDispatchOutcome::Unsupported;
        }
        let selected = GuardedExecution {
            expected,
            host,
            reconciled: Mutex::new(None),
            replan_before_encode: AtomicBool::new(false),
            maintenance: Mutex::new(None),
        };
        // A single participant deliberately uses the exact batch path too.
        let result = self
            .execute_plan_runtime_prefill_batch_inner(
                inputs,
                observation.as_deref_mut(),
                Some(&selected),
            )
            .await;
        let outcome = match result {
            Ok(PlanRuntimeBatchPrefillOutcome::Completed(outputs)) => {
                GuardedDispatchOutcome::Submitted(Ok(outputs))
            }
            Ok(PlanRuntimeBatchPrefillOutcome::NotSubmitted(reason)) => {
                GuardedDispatchOutcome::Deferred(reason)
            }
            Ok(PlanRuntimeBatchPrefillOutcome::Unsupported) => GuardedDispatchOutcome::Unsupported,
            Err(error) => GuardedDispatchOutcome::Submitted(Err(error)),
        };
        finish(self, &selected, observation, outcome)
    }

    pub(in crate::executor::vnext_executor) async fn execute_guarded_mixed(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
        expected: &ExpectedExecutionWave,
        host: &dyn NonblockingHostSubmissionGuard,
        mut observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<PlanRuntimeMixedBatchOutput> {
        if !self.supports_guarded_prefill_wave(expected)
            || expected.work().kind() != ActualWaveKind::Mixed
            || prefills.is_empty()
            || decodes.is_empty()
            || !matches_inputs(expected, prefills, decodes)
        {
            return GuardedDispatchOutcome::Unsupported;
        }
        let selected = GuardedExecution {
            expected,
            host,
            reconciled: Mutex::new(None),
            replan_before_encode: AtomicBool::new(false),
            maintenance: Mutex::new(None),
        };
        let result = self
            .execute_plan_runtime_mixed_batch_inner(
                prefills,
                decodes,
                observation.as_deref_mut(),
                Some(&selected),
            )
            .await;
        let outcome = match result {
            Ok(PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes }) => {
                GuardedDispatchOutcome::Submitted(Ok(PlanRuntimeMixedBatchOutput {
                    prefills,
                    decodes,
                }))
            }
            Ok(PlanRuntimeMixedBatchOutcome::NotSubmitted(reason)) => {
                GuardedDispatchOutcome::Deferred(reason)
            }
            Ok(PlanRuntimeMixedBatchOutcome::Unsupported) => GuardedDispatchOutcome::Unsupported,
            Err(error) => GuardedDispatchOutcome::Submitted(Err(error)),
        };
        finish(self, &selected, observation, outcome)
    }
}

/// The same atomic registry transition is used for both phases. No row is
/// reopened if any original slot was cancelled, replaced, or otherwise lost.
fn restore_prefill_execution_batch<R: DeviceRuntime>(
    registry: &Mutex<VNextSequenceRegistry<R>>,
    candidates: &[VNextPrefillCandidate<R>],
    guards: &mut [VNextPrefillExecutionGuard<'_, R>],
) -> Result<()> {
    let authority = candidates
        .iter()
        .map(|candidate| (&candidate.slot, &candidate.sequence))
        .collect::<Vec<_>>();
    registry.lock().restore_prefill_batch_ready(&authority)?;
    for guard in guards {
        guard.disarm();
    }
    Ok(())
}

#[cfg(test)]
mod tests;
