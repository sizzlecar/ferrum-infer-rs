//! The controller never calls the adaptive or unguarded executor entrypoints.
use super::super::calibration::CalibrationSubmissionState;
use super::owner::{ControllerFlight, ControllerWork, HostGuard};
use super::*;
use ferrum_interfaces::execution_cost::{
    ExpectedWaveInput, GuardedCostObservation, GuardedDispatchOutcome,
    NonblockingHostSubmissionGuard,
};
use ferrum_interfaces::model_executor::PlanRuntimeMixedBatchOutput;

impl EngineInner {
    pub(super) async fn dispatch_controller_wave(
        &self,
        flight: &ControllerFlight,
    ) -> Result<EngineIterationOutcome> {
        let work = &flight.work;
        tracing::trace!(batch_id = %work.batch.batch_id, rows = work.expected.work().participants().len(),
            "entering one guarded controller wave");
        let guard = HostGuard { engine: self, work };
        if let Err(reason) = guard.check() {
            tracing::trace!(?reason, "controller evidence changed before preparation");
            self.arm_controller_retry(retry::ControllerRetryReason::ChangedEvidence);
            if matches!(
                reason,
                ferrum_interfaces::execution_cost::HostSubmissionRejection::WitnessExpired
                    | ferrum_interfaces::execution_cost::HostSubmissionRejection::CostModelChanged
            ) {
                self.request_completion_turn(
                    ferrum_interfaces::execution_cost::CompletionOnlyReason::WitnessExpired,
                );
            }
            self.withdraw_controller_flight(flight)?;
            return self.slo_controller_idle_outcome();
        }
        let preparation_timing = work.proof.budget.stage(ControllerStage::InputPreparation);
        let request_ids: Vec<_> = work.rows().map(|row| row.request_id.clone()).collect();
        let mut preparation = self.prepare_cost_observation();
        let prepared = (|| {
            let mut prefills = Vec::new();
            let mut decodes = Vec::new();
            // Preserve physical participant order for cost correlation even
            // though the executor input API has one vector per phase.
            for participant in work.rows() {
                match &participant.input {
                    ExpectedWaveInput::Prefill { chunk } => {
                        let scheduled = work
                            .batch
                            .requests
                            .iter()
                            .find(|row| row.request.id == participant.request_id)
                            .ok_or_else(|| {
                                FerrumError::internal("guarded prefill lost selected request")
                            })?;
                        let input = self
                            .prepare_guarded_plan_runtime_prefill(
                                scheduled,
                                *chunk,
                                &mut preparation,
                            )?
                            .ok_or_else(|| {
                                FerrumError::cancelled("guarded prefill owner disappeared")
                            })?;
                        prefills.push(input);
                    }
                    ExpectedWaveInput::Decode { .. } => {
                        let mut inputs = self.prepare_plan_runtime_decodes(
                            std::slice::from_ref(&participant.request_id),
                            &mut preparation,
                        );
                        if inputs.len() != 1 {
                            return Err(FerrumError::cancelled("guarded decode owner disappeared"));
                        }
                        let mut input = inputs.pop().unwrap();
                        // Use the captured immutable host policy. Rebuilding a
                        // repetition history would allocate a different Arc and
                        // correctly fail the final native identity check.
                        input.logits_policy =
                            participant.decode_policy.clone().ok_or_else(|| {
                                FerrumError::internal("guarded decode lost captured policy")
                            })?;
                        decodes.push(input);
                    }
                }
            }
            Ok::<_, FerrumError>((prefills, decodes))
        })();
        let (prefills, decodes) = match prepared {
            Ok(inputs) => inputs,
            Err(error) => {
                self.withdraw_controller_flight(flight)?;
                return Err(error);
            }
        };
        let decode_ids: Vec<_> = decodes
            .iter()
            .map(|input| input.request_id.clone())
            .collect();
        let mut cost = preparation.and_then(EngineCostPreparation::begin);
        if let (Some(call), Some(receipt)) = (cost.as_deref_mut(), &flight.calibration) {
            call.attach_calibration_capture(Arc::clone(&receipt.capture));
        }
        let started_at = self.close_plan_runtime_decode_scheduling(&decode_ids);
        drop(preparation_timing);
        let executor_timing = work.proof.budget.stage(ControllerStage::ExecutorAwait);
        if let Some(receipt) = &flight.calibration {
            receipt.record(CalibrationSubmissionState::InFlightUnknown);
        }
        let outcome = {
            let mut context = cost.as_deref_mut().and_then(|call| call.context().ok());
            self.dispatch_guarded_controller_inputs(
                &prefills,
                &decodes,
                work,
                &guard,
                context.as_deref_mut(),
            )
            .await
        };
        if matches!(&outcome, GuardedDispatchOutcome::Submitted(_)) {
            self.record_recovery_submission(work);
            if let owner::ControllerTimingCommitment::Witness {
                admission: Some(admission),
                ..
            } = &work.timing
            {
                self.record_started_time_witness(admission);
            }
        }
        if let Some(receipt) = &flight.calibration {
            receipt.record(
                if matches!(&outcome, GuardedDispatchOutcome::Submitted(_)) {
                    CalibrationSubmissionState::Submitted
                } else {
                    CalibrationSubmissionState::NotSubmitted
                },
            );
        }
        drop(executor_timing);
        match outcome {
            GuardedDispatchOutcome::ReplanBeforeEncode => {
                self.arm_controller_retry(retry::ControllerRetryReason::ChangedEvidence);
                // A logical backing extension may have changed the captured
                // resource view. No encode/submission occurred, but the next
                // attempt must capture fresh evidence instead of reusing it.
                self.record_controller(ControllerObservation {
                    obligations: work.proof.queue.requests().len(),
                    disposition: "unknown",
                    reason: "resource_changed_before_encode",
                });
                self.withdraw_controller_flight(flight)?;
                self.slo_controller_idle_outcome()
            }
            GuardedDispatchOutcome::Unsupported => {
                self.arm_controller_retry(retry::ControllerRetryReason::ChangedEvidence);
                self.record_controller(ControllerObservation {
                    obligations: work.proof.queue.requests().len(),
                    disposition: "unknown",
                    reason: "final_submit_guard_unavailable",
                });
                self.withdraw_controller_flight(flight)?;
                self.slo_controller_idle_outcome()
            }
            GuardedDispatchOutcome::Deferred(deferral) => {
                // This exact attempt performed no encode/submission. Retain
                // the original publication until the scheduler accepts undo.
                self.withdraw_controller_flight(flight)?;
                match deferral {
                    ExecutorExecutionDeferral::RequestState(deferral) => {
                        if deferral
                            .request_ids()
                            .iter()
                            .any(|id| !request_ids.contains(id))
                        {
                            return Err(FerrumError::internal(
                                "guarded deferral names another frontier",
                            ));
                        }
                        self.defer_for_request_state_readiness(deferral).await?;
                    }
                    ExecutorExecutionDeferral::Capacity(deferral) => {
                        deferral.validated_maintenance_retry_scope(&request_ids)?;
                        self.retain_controller_capacity_wait(work.expected.work(), deferral)?;
                        // The following iteration owns capacity handling.
                        self.record_controller(ControllerObservation {
                            obligations: work.proof.queue.requests().len(),
                            disposition: "blocked",
                            reason: "execution_capacity",
                        });
                    }
                }
                self.slo_controller_idle_outcome()
            }
            GuardedDispatchOutcome::MaintenanceDeferred { deferral, ticket } => {
                let ExecutorExecutionDeferral::Capacity(capacity) = deferral else {
                    return Err(FerrumError::internal(
                        "capacity maintenance ticket carries a state-only deferral",
                    ));
                };
                capacity.validated_maintenance_retry_scope(&request_ids)?;
                self.withdraw_controller_flight(flight)?;
                self.retain_controller_maintenance(work.expected.work(), ticket)?;
                self.slo_controller_idle_outcome()
            }
            GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt) => {
                tracing::trace!(reason=?receipt.reason(), "prepared controller wave withdrawn after cleanup");
                let completion_reason = match receipt.reason() {
                    ferrum_interfaces::execution_cost::GuardedNotSubmittedReason::HostRejected(
                        ferrum_interfaces::execution_cost::HostSubmissionRejection::WitnessExpired
                        | ferrum_interfaces::execution_cost::HostSubmissionRejection::CostModelChanged,
                    ) => ferrum_interfaces::execution_cost::CompletionOnlyReason::WitnessExpired,
                    _ => ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
                };
                self.request_completion_turn(completion_reason);
                self.withdraw_controller_flight(flight)?;
                self.slo_controller_idle_outcome()
            }
            GuardedDispatchOutcome::Submitted(result) => {
                let _stage = work.proof.budget.stage(ControllerStage::Reconciliation);
                // No future cancellation can drop the backend call: the durable
                // single-flight task owns it and its terminal reconciliation.
                // This receipt must never enter the no-submission undo path.
                flight.receipt.lock().take();
                let outputs = match result {
                    Ok(outputs) => outputs,
                    Err(error) => {
                        self.fail_controller_participants(work, &error.to_string())
                            .await?;
                        return Err(error);
                    }
                };
                let completed_at = Instant::now();
                let validation = (|| {
                    if outputs.prefills.len() != prefills.len() {
                        return Err(FerrumError::backend(
                            "guarded prefill receipt cardinality changed",
                        ));
                    }
                    for (input, output) in prefills.iter().zip(&outputs.prefills) {
                        output.validate_for(
                            &input.request_id,
                            input.chunk,
                            self.model_executor.info().vocab_size,
                        )?;
                        if output.completed_chunk() != input.chunk
                            || output.capacity_probe_count() != 0
                        {
                            return Err(FerrumError::backend(
                                "guarded prefill receipt narrowed selected work",
                            ));
                        }
                    }
                    self.validate_plan_runtime_decode_outputs(&decodes, &outputs.decodes)
                })();
                if let Err(error) = validation {
                    let cleanup = self.discard_plan_runtime_prefill_completions(outputs.prefills);
                    self.fail_controller_participants(work, &error.to_string())
                        .await?;
                    cleanup?;
                    return Err(error);
                }
                if let Some(start) = started_at {
                    self.record_plan_runtime_decode_execution(&decode_ids, start, completed_at);
                }
                let commit_fences = work.commit_fences();
                let decode_fences: Vec<_> = commit_fences
                    .iter()
                    .filter(|row| matches!(row.input(), ExpectedWaveInput::Decode { .. }))
                    .copied()
                    .collect();
                if let Err(error) = self
                    .commit_plan_runtime_decode_outputs_fenced(
                        &decode_ids,
                        outputs.decodes,
                        Some(completed_at),
                        &mut cost,
                        Some(&decode_fences),
                    )
                    .await
                {
                    let cleanup = self.discard_plan_runtime_prefill_completions(outputs.prefills);
                    self.fail_controller_participants(work, &error.to_string())
                        .await?;
                    cleanup?;
                    return Err(error);
                }
                let mut remaining = outputs.prefills.into_iter();
                for input in prefills {
                    let output = remaining.next().expect("validated prefill cardinality");
                    let fence = commit_fences
                        .iter()
                        .find(|row| row.request_id() == &input.request_id)
                        .expect("prepared input belongs to exact selection");
                    if let Err(error) = self
                        .commit_plan_runtime_prefill_completion_fenced(
                            &input.request_id,
                            input.input_tokens.len(),
                            input.chunk,
                            output,
                            &mut cost,
                            Some(*fence),
                        )
                        .await
                    {
                        let cleanup = self.discard_plan_runtime_prefill_completions(remaining);
                        self.fail_controller_participants(work, &error.to_string())
                            .await?;
                        cleanup?;
                        return Err(error);
                    }
                }
                self.finish_controller_output(work);
                if let Some(receipt) = &flight.calibration {
                    receipt.record(CalibrationSubmissionState::HostReconciled);
                }
                Ok(EngineIterationOutcome::Progressed)
            }
        }
    }

    async fn dispatch_guarded_controller_inputs(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[ferrum_interfaces::model_executor::PlanRuntimeDecodeInput],
        work: &ControllerWork,
        guard: &HostGuard<'_>,
        observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<PlanRuntimeMixedBatchOutput> {
        if prefills.is_empty() {
            map_guarded_outcome(
                self.model_executor
                    .plan_runtime_batch_decode_guarded_work_observed(
                        decodes,
                        &work.expected,
                        guard,
                        observation,
                    )
                    .await,
                |decodes| PlanRuntimeMixedBatchOutput {
                    prefills: Vec::new(),
                    decodes,
                },
            )
        } else if decodes.is_empty() {
            map_guarded_outcome(
                self.model_executor
                    .plan_runtime_batch_prefill_guarded_work_observed(
                        prefills,
                        &work.expected,
                        guard,
                        observation,
                    )
                    .await,
                |prefills| PlanRuntimeMixedBatchOutput {
                    prefills,
                    decodes: Vec::new(),
                },
            )
        } else {
            self.model_executor
                .plan_runtime_mixed_batch_guarded_work_observed(
                    prefills,
                    decodes,
                    &work.expected,
                    guard,
                    observation,
                )
                .await
        }
    }

    /// The caller owns iteration_lock. Incarnation checks exclude a reused ID;
    /// a surviving original peer is cleaned even if another peer fails cleanup.
    pub(super) async fn fail_controller_participants(
        &self,
        work: &ControllerWork,
        message: &str,
    ) -> Result<()> {
        let mut failure = None;
        for participant in work.rows() {
            let current = self
                .sequences
                .read()
                .get(&participant.request_id)
                .is_some_and(|sequence| {
                    sequence.cost_frontier.is_some_and(|frontier| {
                        frontier.owner_incarnation == participant.owner_incarnation
                    })
                });
            if current {
                if let Err(error) = self
                    .complete_request_with_error(
                        &participant.request_id,
                        FerrumError::backend(message),
                    )
                    .await
                {
                    failure.get_or_insert(error);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }

    fn finish_controller_output(&self, work: &ControllerWork) {
        use crate::continuous_engine::output_flow_runtime::OutputDelta;
        for participant in work.rows() {
            let mut sequences = self.sequences.write();
            let Some(sequence) = sequences.get_mut(&participant.request_id) else {
                continue;
            };
            if !sequence
                .cost_frontier
                .is_some_and(|frontier| frontier.owner_incarnation == participant.owner_incarnation)
            {
                continue;
            }
            let Some(output) = sequence.credited_output.as_mut() else {
                continue;
            };
            let Some(grant) = output.grant.take() else {
                continue;
            };
            if output.tokens_before_grant.checked_add(1) == Some(sequence.generated_tokens.len()) {
                output.accepted_ordinal = grant.committed(OutputDelta {
                    text: String::new(),
                    token: sequence.generated_tokens.last().copied(),
                    generated_tokens: sequence.generated_tokens.len(),
                    created: chrono::Utc::now().timestamp().max(0) as u64,
                });
            } else {
                // Submission happened but its host frontier is not known. Do
                // not turn this grant into a fresh device authorization.
                drop(grant);
                output.port.cancel();
            }
        }
    }
}

fn map_guarded_outcome<T, U>(
    outcome: GuardedDispatchOutcome<T>,
    map: impl FnOnce(T) -> U,
) -> GuardedDispatchOutcome<U> {
    match outcome {
        GuardedDispatchOutcome::Unsupported => GuardedDispatchOutcome::Unsupported,
        GuardedDispatchOutcome::ReplanBeforeEncode => GuardedDispatchOutcome::ReplanBeforeEncode,
        GuardedDispatchOutcome::Deferred(reason) => GuardedDispatchOutcome::Deferred(reason),
        GuardedDispatchOutcome::MaintenanceDeferred { deferral, ticket } => {
            GuardedDispatchOutcome::MaintenanceDeferred { deferral, ticket }
        }
        GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt) => {
            GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt)
        }
        GuardedDispatchOutcome::Submitted(result) => {
            GuardedDispatchOutcome::Submitted(result.map(map))
        }
    }
}
