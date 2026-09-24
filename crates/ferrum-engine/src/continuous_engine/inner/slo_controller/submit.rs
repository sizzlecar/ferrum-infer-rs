pub(in crate::continuous_engine) use super::owner::PreparedControllerWave;
use super::*;
use ferrum_interfaces::execution_cost::{
    host_history_cost_signature, CostObservationParticipant, ExpectedExecutionCostWave,
    ExpectedExecutionWave, ExpectedWaveInput, ExpectedWaveParticipant,
};

impl EngineInner {
    pub(in crate::continuous_engine) fn slo_controller_idle_outcome(
        &self,
    ) -> Result<EngineIterationOutcome> {
        if self.consume_controller_maintenance_fairness_turn()? {
            return Ok(EngineIterationOutcome::Progressed);
        }
        if let Some(observed) = self.time_admission_capacity_wait()? {
            if let Some(waiter) = self
                .model_executor
                .register_execution_capacity_waiter(&observed)?
            {
                return Ok(EngineIterationOutcome::CapacityBlocked(waiter));
            }
        }
        Ok(EngineIterationOutcome::Idle)
    }
    pub(super) fn retry_slo_publication_release(&self) -> Result<bool> {
        let mut state = self.slo_controller.lock();
        let Some(receipt) = state.pending_release.as_ref() else {
            return Ok(true);
        };
        match self.scheduler.release_unsubmitted_planned_wave(receipt) {
            Ok(_) => {
                state.pending_release = None;
                Ok(true)
            }
            Err(PlanningStateUnavailable::Busy) => Ok(false),
            Err(reason) => Err(FerrumError::scheduler(format!(
                "release SLO publication: {reason:?}"
            ))),
        }
    }

    pub(super) fn keep_unsubmitted_publication(
        &self,
        receipt: PlanningPublicationReceipt,
    ) -> Result<()> {
        {
            let mut state = self.slo_controller.lock();
            if state.pending_release.is_some() {
                return Err(FerrumError::internal(
                    "SLO publication release slot already occupied",
                ));
            }
            state.pending_release = Some(receipt);
        }
        let _ = self.retry_slo_publication_release()?;
        // Retain a notification across registration of the next iteration wait.
        self.work_notify.notify_one();
        Ok(())
    }

    fn controller_frontiers_match(&self, captured: &ControllerSnapshot) -> bool {
        let Some(sequences) = self.sequences.try_read() else {
            return false;
        };
        if sequences.len() != captured.fences.len() {
            return false;
        }
        captured.fences.iter().all(|fence| {
            if !captured.budget.poll() {
                return false;
            }
            let Some(sequence) = sequences.get(&fence.key.request_id) else {
                return false;
            };
            fence.matches_sequence(sequence)
                && sequence.credited_output.as_ref().is_some_and(|output| {
                    output.grant.is_none() && output.port.planning_snapshot() == fence.output
                })
        })
    }

    #[cfg(test)]
    pub(super) fn prepare_slo_controller_wave(
        &self,
        captured: ControllerSnapshot,
        selected: SelectedWave,
        hint: &ferrum_interfaces::BatchHint,
    ) -> Result<SloIterationPlan> {
        self.prepare_slo_controller_wave_with_admission(captured, selected, hint, None)
    }

    pub(super) fn prepare_slo_controller_wave_with_admission(
        &self,
        captured: ControllerSnapshot,
        selected: SelectedWave,
        hint: &ferrum_interfaces::BatchHint,
        admission: Option<super::admission::PendingTimeWitness>,
    ) -> Result<SloIterationPlan> {
        // Diagnostic only: preserve the original short-circuit evaluation and
        // do not reread clocks, budgets, or live evidence to explain a failure.
        let mut rejection_reason = "unspecified";
        macro_rules! rejects {
            ($condition:expr, $reason:literal) => {{
                let rejected = $condition;
                if rejected {
                    rejection_reason = $reason;
                }
                rejected
            }};
        }
        macro_rules! publication_idle {
            ($reason:expr) => {{
                tracing::trace!(
                    reason = $reason,
                    selected_rows = selected.candidate.work.len(),
                    snapshot_requests = captured.fences.len(),
                    "SLO witness publication rejected before dispatch"
                );
                return Ok(SloIterationPlan::Idle);
            }};
        }
        if !captured.budget.poll() {
            publication_idle!("entry_budget_expired");
        }
        if rejects!(
            selected.protection.as_ref().is_some_and(|scope| {
                scope.as_ref() != captured.protection.as_ref() || admission.is_some()
            }),
            "recovery_scope_or_admission_conflict"
        ) || rejects!(
            captured.protection.needs_recovery() && selected.protection.is_none(),
            "missing_recovery_protection"
        ) {
            publication_idle!(rejection_reason);
        }
        let valid_until_ns = selected
            .planning_observed_at_ns
            .checked_add(selected.witness_valid_for_ns);
        let Some(valid_until) =
            valid_until_ns.and_then(|ns| captured.origin.instant_at_ns(ns).ok())
        else {
            publication_idle!("witness_validity_conversion_failed");
        };
        if rejects!(
            selected.snapshot_generation != captured.snapshot.generation,
            "snapshot_generation_changed"
        ) || rejects!(
            selected.cost_model_version != captured.snapshot.cost_model_version,
            "snapshot_model_version_mismatch"
        ) || rejects!(
            selected.snapshot_observed_at_ns != captured.snapshot.observed_at_ns,
            "snapshot_observed_at_mismatch"
        ) || rejects!(
            !self.controller_model_current(selected.cost_model_version),
            "live_cost_model_changed_or_unavailable"
        ) || rejects!(!captured.budget.poll(), "budget_before_revalidation")
            || rejects!(
                slo_clock_now() > valid_until,
                "witness_expired_before_revalidation"
            )
            || rejects!(
                !self.controller_frontiers_match(&captured),
                "frontier_or_output_evidence_changed"
            )
        {
            publication_idle!(rejection_reason);
        }
        let requests: Vec<_> = captured
            .fences
            .iter()
            .map(|fence| ExecutorResourcePlanningRequest {
                request_id: &fence.key.request_id,
                cache_id: fence.resource_cache_id(),
            })
            .collect();
        let rechecked = self
            .model_executor
            .revalidate_execution_resource_planning_view(
                &requests,
                &captured.resources,
                &mut || captured.budget.poll() && slo_clock_now() <= valid_until,
            );
        if !matches!(rechecked, ResourcePlanningAvailability::Known(true)) {
            publication_idle!("resource_revalidation_unavailable");
        }
        let route = self.model_executor.execution_cost_route_view(
            &requests,
            captured.resources.limits(),
            &mut || captured.budget.poll() && slo_clock_now() <= valid_until,
        );
        if !matches!(route, ferrum_interfaces::vnext::ExecutionCostRouteAvailability::Known(ref current)
            if captured.route.same_live_evidence(current))
        {
            publication_idle!("execution_route_changed_or_unavailable");
        }
        let mut selected_work = Vec::with_capacity(selected.candidate.work.len());
        let mut participants = Vec::with_capacity(selected.candidate.work.len());
        for work in &selected.candidate.work {
            if !captured.budget.poll() {
                publication_idle!("participant_budget_expired");
            }
            let Some(fence) = captured.fences.iter().find(|fence| {
                fence.key.request_id == work.key.request_id
                    && fence.incarnation == work.key.incarnation
                    && fence.key.generation == work.key.work_generation
            }) else {
                publication_idle!("participant_identity_missing");
            };
            let (action, input) = match work.action {
                WaveAction::Decode if fence.prefill_complete => {
                    let Some(cache_id) = &fence.cache_id else {
                        publication_idle!("decode_cache_missing");
                    };
                    (
                        PlanningWorkAction::Decode,
                        ExpectedWaveInput::Decode {
                            cache_id: cache_id.clone(),
                        },
                    )
                }
                WaveAction::Prefill { offset, count } if !fence.prefill_complete => {
                    let chunk = ferrum_interfaces::model_executor::PrefillChunk::new(
                        offset as usize,
                        count.get() as usize,
                        fence.prefill_total,
                    )?;
                    if chunk.tokens_processed() != fence.prefill_tokens_processed {
                        publication_idle!("prefill_frontier_changed");
                    }
                    (
                        PlanningWorkAction::Prefill {
                            offset: offset as usize,
                            count: NonZeroUsize::new(count.get() as usize).unwrap(),
                        },
                        ExpectedWaveInput::Prefill { chunk },
                    )
                }
                _ => publication_idle!("participant_phase_changed"),
            };
            selected_work.push(PlanningWorkSelection {
                key: fence.key.clone(),
                action,
            });
            let index = captured
                .fences
                .iter()
                .position(|candidate| candidate.key == fence.key)
                .expect("selected fence belongs to captured complete set");
            participants.push(ExpectedWaveParticipant {
                participant_index: index,
                request_id: work.key.request_id.clone(),
                input,
                host: CostObservationParticipant {
                    request_id: work.key.request_id.clone(),
                    owner_incarnation: fence.incarnation,
                    work_generation: fence.generation,
                    input_index: u32::try_from(participants.len())
                        .map_err(|_| FerrumError::internal("guarded wave row overflow"))?,
                    output_policy_signature: Some(host_history_cost_signature(
                        captured.snapshot.requests[index].output_policy_signature,
                        fence.generated as u64,
                    )),
                    host_features: fence.host_features,
                },
            });
        }
        let Some(canonical) = self.controller_first_wave_shape(&captured, &selected, valid_until)
        else {
            publication_idle!("first_wave_canonical_unavailable");
        };
        let expected =
            ExpectedExecutionCostWave::new(captured.route.clone(), canonical, participants)?;
        let expected = ExpectedExecutionWave::from_cost_witness(expected, |id| {
            captured
                .fences
                .iter()
                .find(|fence| fence.key.request_id == *id)
                .map(|fence| &fence.logits_policy)
        })?;
        let mut availability = match self.dynamic_admission_availability.try_lock() {
            Some(value) => value,
            None => publication_idle!("capacity_snapshot_lock_busy"),
        };
        let Some(epochs) = self
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)?
        else {
            publication_idle!("capacity_snapshot_unavailable");
        };
        if rejects!(!captured.budget.poll(), "budget_before_publication")
            || rejects!(
                slo_clock_now() > valid_until,
                "witness_expired_before_publication"
            )
            || rejects!(
                !self.controller_frontiers_match(&captured),
                "frontier_or_output_changed_before_publication"
            )
        {
            publication_idle!(rejection_reason);
        }
        let publication = self.scheduler.try_select_planned_wave(
            &captured.queue,
            &selected_work,
            hint,
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(
                    epochs.coordinator_id,
                    epochs.release_epoch,
                    epochs.capacity_epoch,
                    0,
                ),
                &availability,
            ),
        );
        drop(availability);
        let (batch, receipt) = match publication {
            Ok(PlanningSelectionOutcome::Published { batch, receipt }) => (batch, receipt),
            Ok(_) | Err(PlanningStateUnavailable::Busy) => {
                publication_idle!("scheduler_publication_unavailable")
            }
            Err(error) => {
                return Err(FerrumError::scheduler(format!(
                    "publish SLO wave: {error:?}"
                )));
            }
        };
        if !captured.budget.poll() {
            self.keep_unsubmitted_publication(receipt)?;
            publication_idle!("budget_after_publication");
        }
        let reserved = match self.reserve_batch_output(&batch) {
            Ok(reserved) => reserved,
            Err(error) => {
                self.finish_batch_output(&batch, true);
                self.keep_unsubmitted_publication(receipt)?;
                return Err(error);
            }
        };
        if !reserved {
            self.keep_unsubmitted_publication(receipt)?;
            publication_idle!("output_credit_unavailable");
        }
        if rejects!(!captured.budget.poll(), "budget_after_output_reservation")
            || rejects!(
                slo_clock_now() > valid_until,
                "witness_expired_after_output_reservation"
            )
            || rejects!(
                !self.controller_model_current(selected.cost_model_version),
                "cost_model_changed_or_unavailable_after_output_reservation"
            )
        {
            self.finish_batch_output(&batch, true);
            self.keep_unsubmitted_publication(receipt)?;
            publication_idle!(rejection_reason);
        }
        self.install_controller_wave(
            owner::ControllerWork {
                batch,
                timing: owner::ControllerTimingCommitment::Witness {
                    valid_until,
                    model_version: selected.cost_model_version,
                    admission,
                },
                proof: captured.into_safety(),
                expected,
            },
            receipt,
        )
        .map(SloIterationPlan::Selected)
    }

    fn controller_model_current(&self, version: u64) -> bool {
        self.cost_runtime
            .as_ref()
            .and_then(|runtime| runtime.try_snapshot().flatten())
            .is_some_and(|model| model.model_version() == version)
    }
}
