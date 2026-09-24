//! Capacity maintenance is a distinct iteration, never an inner model retry.
use super::*;
use ferrum_interfaces::execution_cost::{
    ActualRowWork, ExpectedWaveInput, ExpectedWaveWork, HostSubmissionRejection,
    NonblockingHostSubmissionGuard,
};
use ferrum_interfaces::model_executor::{
    ExecutorExecutionCapacityDeferral, ExecutorExecutionMaintenanceOutcome,
    ExecutorExecutionMaintenanceTicket,
};

#[cfg(test)]
mod tests;

pub(super) struct ControllerMaintenance {
    work: ExpectedWaveWork,
    action: MaintenanceAction,
}

enum MaintenanceAction {
    Backing(ExecutorExecutionMaintenanceTicket),
    Capacity(ExecutorExecutionCapacityDeferral),
}

struct MaintenanceGuard<'a> {
    engine: &'a EngineInner,
    work: &'a ExpectedWaveWork,
}

impl NonblockingHostSubmissionGuard for MaintenanceGuard<'_> {
    fn check(&self) -> std::result::Result<(), HostSubmissionRejection> {
        use HostSubmissionRejection::*;
        if self.engine.shutdown_started.load(Ordering::Acquire) {
            return Err(Cancelled);
        }
        let sequences = self.engine.sequences.try_read().ok_or(Busy)?;
        for row in self.work.participants() {
            let row = row.selection();
            let sequence = sequences.get(&row.request_id).ok_or(Cancelled)?;
            if !sequence.cost_frontier.is_some_and(|frontier| {
                frontier.owner_incarnation == row.owner_incarnation
                    && frontier.work_generation == row.work_generation
            }) {
                return Err(FrontierChanged);
            }
            let matches = match (&row.input, row.work) {
                (ExpectedWaveInput::Prefill { chunk }, ActualRowWork::Prefill { .. }) => {
                    !sequence.prefill_complete
                        && sequence.prefill_tokens_processed == chunk.tokens_processed()
                        && sequence.prefill_context_len() == chunk.total_prompt_tokens()
                }
                (ExpectedWaveInput::Decode { cache_id }, ActualRowWork::Decode { kv_tokens }) => {
                    sequence.prefill_complete
                        && sequence.model_cache_id() == Some(cache_id.as_str())
                        && sequence
                            .model_kv
                            .as_ref()
                            .is_some_and(|kv| kv.handle().num_tokens() == kv_tokens as usize)
                }
                _ => false,
            };
            if !matches {
                return Err(FrontierChanged);
            }
            let output = sequence.credited_output.as_ref().ok_or(OutputRevoked)?;
            if output.failure.is_some() || output.port.consumer_closed() {
                return Err(OutputRevoked);
            }
        }
        Ok(())
    }
}

impl EngineInner {
    pub(super) fn retain_controller_maintenance(
        &self,
        work: &ExpectedWaveWork,
        ticket: ExecutorExecutionMaintenanceTicket,
    ) -> Result<()> {
        if ticket.request_ids().iter().any(|id| {
            !work
                .participants()
                .iter()
                .any(|row| row.selection().request_id == *id)
        }) {
            return Err(FerrumError::internal(
                "maintenance ticket names an unselected owner",
            ));
        }
        self.retain_controller_capacity_action(work, MaintenanceAction::Backing(ticket))
    }

    pub(super) fn retain_controller_capacity_wait(
        &self,
        work: &ExpectedWaveWork,
        deferral: ExecutorExecutionCapacityDeferral,
    ) -> Result<()> {
        self.retain_controller_capacity_action(work, MaintenanceAction::Capacity(deferral))
    }

    fn retain_controller_capacity_action(
        &self,
        work: &ExpectedWaveWork,
        action: MaintenanceAction,
    ) -> Result<()> {
        let mut state = self.slo_controller.lock();
        if state.pending_maintenance.is_some() {
            return Err(FerrumError::internal(
                "controller maintenance slot already occupied",
            ));
        }
        state.pending_maintenance = Some(ControllerMaintenance {
            work: work.clone(),
            action,
        });
        drop(state);
        self.work_notify.notify_one();
        Ok(())
    }

    /// The caller owns iteration_lock and has reaped the previous exact wave.
    /// One existing bounded maintenance transaction may touch several pools;
    /// no user model work or replacement wave is submitted in this turn.
    pub(in crate::continuous_engine) fn clear_controller_maintenance(&self) {
        let pending = {
            let mut state = self.slo_controller.lock();
            state.pending_maintenance_fairness = false;
            state.pending_maintenance.take()
        };
        drop(pending);
    }

    /// Called after an actual controller attempt returned Idle. A maintenance
    /// retry yields one scheduling opportunity to peers, but an empty queue of
    /// runnable peers must still consume that opportunity. Never allocate or
    /// submit work, clear a ticket, or treat a changed capacity epoch as a grant.
    pub(super) fn consume_controller_maintenance_fairness_turn(&self) -> Result<bool> {
        if !self.completion_allowed()
            || self.controller_retry_pending()
            || !self.slo_controller.lock().pending_maintenance_fairness
        {
            return Ok(false);
        }
        let Some(mut availability) = self.dynamic_admission_availability.try_lock() else {
            self.arm_controller_retry(retry::ControllerRetryReason::SnapshotBusy);
            return Ok(false);
        };
        let epochs = self
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)?
            .ok_or_else(|| FerrumError::internal("maintenance fairness lost capacity identity"))?;
        let wake = AdmissionWakeSnapshot::new(
            AdmissionWakeEpochs::new(
                epochs.coordinator_id,
                epochs.release_epoch,
                epochs.capacity_epoch,
                0,
            ),
            &availability,
        );
        let result = self
            .scheduler
            .planning_state(NonZeroUsize::new(4096).unwrap(), wake)
            .and_then(|snapshot| {
                self.scheduler
                    .try_consume_maintenance_fairness_turn(&snapshot, wake)
            });
        drop(availability);
        match result {
            Ok(PlanningMaintenanceFairnessOutcome::Advanced {
                previous_iteration,
                next_iteration,
                matured_tickets,
            }) => {
                self.slo_controller.lock().pending_maintenance_fairness = false;
                counter!("ferrum.engine.slo_maintenance_fairness_turns_total").increment(1);
                tracing::trace!(
                    previous_iteration,
                    next_iteration,
                    matured_tickets,
                    "consumed empty maintenance fairness turn; physical work requires recapture"
                );
                Ok(true)
            }
            Ok(PlanningMaintenanceFairnessOutcome::NoPending) => {
                // A peer publication already consumed the fairness turn, or
                // the exact request was completed/cancelled in the meantime.
                self.slo_controller.lock().pending_maintenance_fairness = false;
                Ok(false)
            }
            Ok(PlanningMaintenanceFairnessOutcome::Stale) | Err(PlanningStateUnavailable::Busy) => {
                self.arm_controller_retry(retry::ControllerRetryReason::SnapshotBusy);
                Ok(false)
            }
            Err(reason) => Err(FerrumError::scheduler(format!(
                "consume maintenance fairness turn: {reason:?}"
            ))),
        }
    }

    pub(in crate::continuous_engine::inner) async fn prepare_slo_maintenance_turn(
        &self,
    ) -> Result<Option<EngineIterationOutcome>> {
        let has_pending = { self.slo_controller.lock().pending_maintenance.is_some() };
        if has_pending && self.controller_retry_pending() {
            return Ok(Some(EngineIterationOutcome::Idle));
        }
        if has_pending && !self.retry_slo_publication_release()? {
            self.arm_controller_retry(retry::ControllerRetryReason::SnapshotBusy);
            return Ok(Some(EngineIterationOutcome::Idle));
        }
        let pending = self.slo_controller.lock().pending_maintenance.take();
        let Some(pending) = pending else {
            return Ok(None);
        };
        let guard = MaintenanceGuard {
            engine: self,
            work: &pending.work,
        };
        match guard.check() {
            Ok(()) => {}
            Err(HostSubmissionRejection::Busy) => {
                self.slo_controller.lock().pending_maintenance = Some(pending);
                self.arm_controller_retry(retry::ControllerRetryReason::SnapshotBusy);
                return Ok(Some(EngineIterationOutcome::Idle));
            }
            Err(_) => return Ok(Some(EngineIterationOutcome::Progressed)), // Dropping the one-use ticket does no maintenance.
        }
        let outcome = match pending.action {
            MaintenanceAction::Backing(ticket) => self
                .model_executor
                .maintain_execution_capacity_once(ticket, &guard)?,
            MaintenanceAction::Capacity(deferral) => {
                ExecutorExecutionMaintenanceOutcome::Wait(deferral)
            }
        };
        match outcome {
            ExecutorExecutionMaintenanceOutcome::Recapture { observed, progress } => {
                tracing::trace!(
                    ?observed,
                    ?progress,
                    "controller capacity maintenance completed; recapturing"
                );
            }
            ExecutorExecutionMaintenanceOutcome::Wait(deferral) => {
                self.defer_controller_capacity(&pending.work, &deferral)
                    .await?;
            }
            ExecutorExecutionMaintenanceOutcome::Rejected(reason) => {
                tracing::trace!(?reason, "controller capacity maintenance owner changed");
                if reason == HostSubmissionRejection::Busy {
                    self.arm_controller_retry(retry::ControllerRetryReason::SnapshotBusy);
                }
            }
            ExecutorExecutionMaintenanceOutcome::Unsupported => {
                self.record_controller(ControllerObservation {
                    obligations: self.scheduler.active_count() + self.scheduler.waiting_count(),
                    disposition: "completion_wait",
                    reason: "maintenance_unsupported",
                });
                return Err(FerrumError::unsupported(
                    "selected executor cannot consume its maintenance ticket",
                ));
            }
        }
        Ok(Some(EngineIterationOutcome::Progressed))
    }

    async fn defer_controller_capacity(
        &self,
        work: &ExpectedWaveWork,
        deferral: &ExecutorExecutionCapacityDeferral,
    ) -> Result<()> {
        let ids: Vec<_> = work
            .participants()
            .iter()
            .map(|row| row.selection().request_id.clone())
            .collect();
        if let Some(retry) = deferral.validated_maintenance_retry_scope(&ids)? {
            self.scheduler
                .defer_retry_after_execution_maintenance(retry)?;
            self.slo_controller.lock().pending_maintenance_fairness = true;
            return Ok(());
        }
        let observed = deferral.observed();
        let make_deferral = || {
            AdmissionDeferral::new(
                DeferredAction::WaitForRelease,
                AdmissionWakeEpochs::new(
                    observed.coordinator_id,
                    observed.release_epoch,
                    observed.capacity_epoch,
                    0,
                ),
                deferral.wait_condition().clone(),
            )
        };
        let release = self.execution_capacity_release_snapshot()?;
        match self
            .scheduler
            .defer_wave_for_execution_capacity(&ids, make_deferral(), &release)?
        {
            ExecutionCapacityAction::Deferred { .. } => {}
            ExecutionCapacityAction::YieldPlanned { transaction } => {
                self.execute_capacity_yield(&transaction, ids.len(), None)
                    .await?;
            }
            ExecutionCapacityAction::InvariantViolation { violation } => {
                return Err(FerrumError::internal(format!(
                    "controller capacity episode violated {:?}",
                    violation.class()
                )))
            }
        }
        Ok(())
    }
}
