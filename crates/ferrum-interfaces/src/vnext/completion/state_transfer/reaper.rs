use super::super::{
    CheckpointTimingCounters, CheckpointTimingPhase, CompletionRecord, CompletionRecoveryCause,
    CompletionRecoveryState, LaneSubmitOutcome, SharedCompletionRecord,
};
use super::*;
use crate::vnext::{
    CheckpointCapturePermit, CompletionReaper, CompletionSlotId, DeferredDeviceCleanupDomainId,
    DeviceExecutionTiming, DeviceTerminal, DeviceTimingMeasurement, DeviceTimingMode, FenceQuery,
    SequenceCheckpointLayout,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Weak;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateTransferObservation {
    Pending,
    Indeterminate,
    Quarantined,
    Ready,
}

#[derive(Debug)]
pub(crate) struct StateTransferSweepEntry {
    pub(crate) slot_id: CompletionSlotId,
    pub(crate) observation: StateTransferObservation,
}

enum TransferPhase<F> {
    InFlight {
        fence: F,
        recovery: CompletionRecoveryState,
    },
    SubmissionIndeterminate,
    Quarantined {
        fence: Option<F>,
    },
    Ready,
}

/// This is a record in CompletionReaper's existing slot table. It has no
/// allocator, stream, recovery worker or capacity ledger of its own.
pub(in crate::vnext::completion) struct StateTransferRecord<R: DeviceRuntime> {
    timing_kind: StateTransferKind,
    timings: Arc<CheckpointTimingCounters>,
    timing_mode: DeviceTimingMode,
    resources: Option<StateTransferLease<R>>,
    lane: Arc<ExecutionLane<R>>,
    outbox: Arc<StateTransferResultSlot<R>>,
    phase: TransferPhase<R::Fence>,
    cleanup_domain: DeferredDeviceCleanupDomainId,
    contract_failure: Option<String>,
}

impl<R: DeviceRuntime> StateTransferRecord<R> {
    pub(in crate::vnext::completion) fn deferred_cleanup_domain(
        &self,
    ) -> Option<DeferredDeviceCleanupDomainId> {
        // Ready restore outcomes also own gated resources until consumed or
        // abandoned; dropping the reaper must cancel those outcomes safely.
        Some(self.cleanup_domain)
    }

    pub(in crate::vnext::completion) fn is_quarantined(&self) -> bool {
        matches!(self.phase, TransferPhase::Quarantined { .. })
    }

    pub(in crate::vnext::completion) fn fail_close(&self) {
        self.lane.fail_closed();
    }

    fn publish_terminal(
        &mut self,
        reason: Option<StateTransferFailureReason>,
        device_timing: Option<DeviceTimingMeasurement<DeviceExecutionTiming>>,
    ) -> Result<(), VNextError> {
        let _timing = self
            .timings
            .start(self.timing_kind, CheckpointTimingPhase::Publication);
        let resources = self
            .resources
            .take()
            .ok_or_else(|| invalid_completion("native transfer was already settled"))?;
        let identity = Arc::clone(resources.identity());
        let slot_id = self.outbox.slot_id();
        let result = match reason {
            Some(reason) => {
                resources.finish_failed();
                StateTransferResult::failed(slot_id, identity, reason)
            }
            None => match resources.finish_succeeded(slot_id) {
                Ok(result) => result,
                Err(error) => StateTransferResult::failed(
                    slot_id,
                    identity,
                    StateTransferFailureReason::ContractFailedButQuiescent(error.to_string()),
                ),
            },
        };
        self.timings.record_device_terminal(
            self.timing_kind,
            self.timing_mode,
            matches!(
                &result,
                StateTransferResult::Captured(_) | StateTransferResult::RestoreReady(_)
            ),
            device_timing,
        );
        self.phase = TransferPhase::Ready;
        self.outbox.publish(result).map_err(|failure| {
            // Dropping a rejected restore result cancels its exact target.
            drop(failure.result);
            failure.error
        })
    }

    fn observe(&mut self, blocking: bool) -> Result<StateTransferObservation, VNextError> {
        let fence = match &self.phase {
            TransferPhase::Ready => return Ok(StateTransferObservation::Ready),
            TransferPhase::SubmissionIndeterminate => {
                return Ok(StateTransferObservation::Indeterminate)
            }
            TransferPhase::Quarantined { .. } => return Ok(StateTransferObservation::Quarantined),
            TransferPhase::InFlight { fence, .. } => fence,
        };
        let (observation, _) = {
            let _timing = self
                .timings
                .start(self.timing_kind, CheckpointTimingPhase::FenceRecovery);
            super::super::observe_device_fence(&self.lane, fence, blocking, self.timing_mode)
        };
        let terminal = match observation {
            Ok(FenceQuery::Pending) => return Ok(StateTransferObservation::Pending),
            Ok(FenceQuery::Terminal(terminal)) => terminal,
            Ok(FenceQuery::Indeterminate(_)) | Err(_) => {
                if let TransferPhase::InFlight { recovery, .. } = &mut self.phase {
                    if !matches!(recovery, CompletionRecoveryState::DrainEligible(_)) {
                        *recovery = if blocking {
                            CompletionRecoveryState::DrainEligible(
                                CompletionRecoveryCause::FenceIndeterminate,
                            )
                        } else {
                            CompletionRecoveryState::QueryIndeterminate
                        };
                    }
                }
                return Ok(StateTransferObservation::Indeterminate);
            }
        };
        // The terminal receipt itself proves quiescence. A panic while
        // checking its metadata cannot turn it into successful capture.
        let (terminal, execution_timing, _) = terminal.into_parts();
        let reason = catch_unwind(AssertUnwindSafe(|| {
            if !self.lane.current_descriptor_matches_snapshot() {
                return Some(StateTransferFailureReason::ContractFailedButQuiescent(
                    "native transfer runtime descriptor changed".into(),
                ));
            }
            match terminal {
                DeviceTerminal::Succeeded => None,
                DeviceTerminal::FailedButQuiescent(error) => {
                    Some(match self.lane.runtime().describe_error(&error) {
                        Ok(report) => StateTransferFailureReason::FailedButQuiescent(report),
                        Err(error) => StateTransferFailureReason::ContractFailedButQuiescent(
                            error.to_string(),
                        ),
                    })
                }
            }
        }))
        .unwrap_or_else(|_| {
            Some(StateTransferFailureReason::ContractFailedButQuiescent(
                "native terminal metadata inspection panicked".into(),
            ))
        });
        let reason = match self.lane.finish_one_terminal() {
            Ok(()) => reason,
            Err(error) => Some(StateTransferFailureReason::ContractFailedButQuiescent(
                format!("native lane retirement failed: {error:?}"),
            )),
        };
        let reason = self
            .contract_failure
            .take()
            .map(StateTransferFailureReason::ContractFailedButQuiescent)
            .or(reason);
        self.publish_terminal(reason, Some(execution_timing))?;
        Ok(StateTransferObservation::Ready)
    }

    fn recover(&mut self) -> Result<StateTransferObservation, VNextError> {
        let has_fence = match &self.phase {
            TransferPhase::InFlight {
                recovery: CompletionRecoveryState::DrainEligible(_),
                ..
            } => true,
            TransferPhase::SubmissionIndeterminate => false,
            TransferPhase::Quarantined { fence } => fence.is_some(),
            TransferPhase::Ready => return Ok(StateTransferObservation::Ready),
            _ => {
                return Err(invalid_completion(
                    "native recovery requires a blocking indeterminate observation",
                ))
            }
        };
        self.lane.fail_closed();
        let drained = {
            let _timing = self
                .timings
                .start(self.timing_kind, CheckpointTimingPhase::FenceRecovery);
            self.lane.drain(has_fence)
        };
        if drained {
            self.publish_terminal(Some(StateTransferFailureReason::AbandonedAfterDrain), None)?;
            return Ok(StateTransferObservation::Ready);
        }
        let old = std::mem::replace(&mut self.phase, TransferPhase::SubmissionIndeterminate);
        let fence = match old {
            TransferPhase::InFlight { fence, .. } => Some(fence),
            TransferPhase::Quarantined { fence } => fence,
            TransferPhase::SubmissionIndeterminate => None,
            TransferPhase::Ready => unreachable!("ready transfers returned above"),
        };
        self.phase = TransferPhase::Quarantined { fence };
        Ok(StateTransferObservation::Quarantined)
    }

    pub(in crate::vnext::completion) fn cleanup_abandoned(&mut self) -> bool {
        if matches!(self.phase, TransferPhase::Ready) {
            return true;
        }
        if self
            .observe(false)
            .is_ok_and(|result| result == StateTransferObservation::Ready)
        {
            return true;
        }
        if self
            .observe(true)
            .is_ok_and(|result| result == StateTransferObservation::Ready)
        {
            return true;
        }
        self.recover()
            .is_ok_and(|result| result == StateTransferObservation::Ready)
    }
}

#[must_use = "the reaper owns device work; consume its terminal result once"]
pub(crate) struct StateTransferHandle<R: DeviceRuntime> {
    reaper: Weak<CompletionReaper<R>>,
    outbox: Weak<StateTransferResultSlot<R>>,
    slot_id: CompletionSlotId,
}

impl<R: DeviceRuntime> StateTransferHandle<R> {
    /// Only this exact result slot is abandoned. A consumed slot has already
    /// released the outbox and needs no further cleanup. No device wait occurs.
    pub(in crate::vnext::completion) fn abandon_consumer(&self) {
        if let Some(outbox) = self.outbox.upgrade() {
            outbox.abandon_consumer();
        }
    }

    pub(crate) fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }

    pub(crate) fn poll(&self) -> Result<StateTransferObservation, VNextError> {
        self.observe(false)
    }
    pub(crate) fn wait_for_recovery(&self) -> Result<StateTransferObservation, VNextError> {
        self.observe(true)
    }

    fn observe(&self, blocking: bool) -> Result<StateTransferObservation, VNextError> {
        let reaper = self
            .reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("native transfer reaper was dropped"))?;
        let outbox = self
            .outbox
            .upgrade()
            .ok_or_else(|| invalid_completion("native result was already consumed"))?;
        reaper.with_state_transfer(self.slot_id, Some(&outbox), |transfer| {
            transfer.observe(blocking)
        })
    }

    pub(crate) fn recover_by_draining_lane(&self) -> Result<StateTransferObservation, VNextError> {
        let reaper = self
            .reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("native transfer reaper was dropped"))?;
        let outbox = self
            .outbox
            .upgrade()
            .ok_or_else(|| invalid_completion("native result was already consumed"))?;
        reaper.with_state_transfer(self.slot_id, Some(&outbox), StateTransferRecord::recover)
    }

    pub(crate) fn take(&self) -> Result<Option<StateTransferResult<R>>, VNextError> {
        let reaper = self
            .reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("native transfer reaper was dropped"))?;
        let outbox = self
            .outbox
            .upgrade()
            .ok_or_else(|| invalid_completion("native result was already consumed"))?;
        reaper.take_state_transfer(self.slot_id, Some(&outbox))
    }
}

pub(crate) enum StateTransferSubmission<R: DeviceRuntime> {
    Submitted(StateTransferHandle<R>),
    Indeterminate(StateTransferHandle<R>),
    ContractAfterSubmission {
        error: VNextError,
        handle: StateTransferHandle<R>,
    },
}

struct TransferReservation<R: DeviceRuntime> {
    reaper: Arc<CompletionReaper<R>>,
    slot_id: CompletionSlotId,
    record: SharedCompletionRecord<R>,
    resources: Option<StateTransferLease<R>>,
    lane: Arc<ExecutionLane<R>>,
    timing_mode: DeviceTimingMode,
    outbox: Arc<StateTransferResultSlot<R>>,
    submission_started: bool,
    finished: bool,
}

impl<R: DeviceRuntime> TransferReservation<R> {
    fn install(&mut self, phase: TransferPhase<R::Fence>) -> Result<(), VNextError> {
        let mut resources = self
            .resources
            .take()
            .expect("reservation owns native resources");
        let cleanup_domain = resources.deferred_cleanup_domain();
        let mut record = self
            .record
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !matches!(*record, CompletionRecord::Reserved) {
            self.lane.fail_closed();
            // Unreachable through safe reservation APIs. Do not release an
            // unknown device write if the registry contract was corrupted.
            std::mem::forget((resources, phase));
            self.finished = true;
            panic!("native completion reservation changed during submission");
        }
        // Install resource transitions while holding the same record lock used
        // by sweeps. No observer can reap this fence before initialization has
        // entered InFlight or recorded a terminal contract failure.
        let transition = if matches!(phase, TransferPhase::InFlight { .. }) {
            resources.mark_fence_installed()
        } else {
            Ok(())
        };
        let contract_failure = transition.as_ref().err().map(ToString::to_string);
        if transition.is_err() {
            self.lane.fail_closed();
        }
        *record = CompletionRecord::StateTransfer(StateTransferRecord {
            timing_kind: resources.identity().kind,
            timings: Arc::clone(&self.reaper.checkpoint_timings),
            timing_mode: self.timing_mode,
            resources: Some(resources),
            lane: Arc::clone(&self.lane),
            outbox: Arc::clone(&self.outbox),
            phase,
            cleanup_domain,
            contract_failure,
        });
        self.finished = true;
        transition
    }

    fn handle(&self) -> StateTransferHandle<R> {
        StateTransferHandle {
            reaper: Arc::downgrade(&self.reaper),
            outbox: Arc::downgrade(&self.outbox),
            slot_id: self.slot_id,
        }
    }
}

impl<R: DeviceRuntime> Drop for TransferReservation<R> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if self.submission_started {
            self.lane.fail_closed();
            let _ = self.install(TransferPhase::SubmissionIndeterminate);
        } else {
            self.reaper.remove_exact(self.slot_id, &self.record);
        }
    }
}

impl<R: DeviceRuntime> CompletionReaper<R> {
    #[cfg(test)]
    pub(crate) fn submit_capture(
        self: &Arc<Self>,
        source: PreparedSequenceStateTransfer<R>,
        permit: CheckpointCapturePermit<R>,
        byte_plan: Arc<SequenceCheckpointBytePlan>,
        lane: Arc<ExecutionLane<R>>,
    ) -> Result<StateTransferSubmission<R>, VNextError> {
        self.submit_capture_with_timing(source, permit, byte_plan, lane, DeviceTimingMode::Off)
    }

    pub(crate) fn submit_capture_with_timing(
        self: &Arc<Self>,
        source: PreparedSequenceStateTransfer<R>,
        permit: CheckpointCapturePermit<R>,
        byte_plan: Arc<SequenceCheckpointBytePlan>,
        lane: Arc<ExecutionLane<R>>,
        timing_mode: DeviceTimingMode,
    ) -> Result<StateTransferSubmission<R>, VNextError> {
        let _timing = self.checkpoint_timings.start(
            StateTransferKind::Capture,
            CheckpointTimingPhase::EncodeSubmit,
        );
        let resources = StateTransferLease::capture(source, permit, byte_plan, &lane)?;
        self.submit_state_transfer(resources, lane, timing_mode)
    }

    #[cfg(test)]
    pub(crate) fn submit_restore(
        self: &Arc<Self>,
        target: PreparedSequenceStateTransfer<R>,
        checkpoint: Arc<CapturedCheckpoint<R>>,
        layout: &SequenceCheckpointLayout,
        lane: Arc<ExecutionLane<R>>,
    ) -> Result<StateTransferSubmission<R>, VNextError> {
        self.submit_restore_with_timing(target, checkpoint, layout, lane, DeviceTimingMode::Off)
    }

    pub(crate) fn submit_restore_with_timing(
        self: &Arc<Self>,
        target: PreparedSequenceStateTransfer<R>,
        checkpoint: Arc<CapturedCheckpoint<R>>,
        layout: &SequenceCheckpointLayout,
        lane: Arc<ExecutionLane<R>>,
        timing_mode: DeviceTimingMode,
    ) -> Result<StateTransferSubmission<R>, VNextError> {
        let _timing = self.checkpoint_timings.start(
            StateTransferKind::Restore,
            CheckpointTimingPhase::EncodeSubmit,
        );
        let resources = StateTransferLease::restore(target, checkpoint, layout, &lane)?;
        self.submit_state_transfer(resources, lane, timing_mode)
    }

    fn submit_state_transfer(
        self: &Arc<Self>,
        resources: StateTransferLease<R>,
        lane: Arc<ExecutionLane<R>>,
        timing_mode: DeviceTimingMode,
    ) -> Result<StateTransferSubmission<R>, VNextError> {
        // Transfers have no compute kernels to attribute. Reuse terminal
        // elapsed timing without enabling counter buffers or encoder changes.
        let timing_mode = if timing_mode.completion_enabled() {
            DeviceTimingMode::Completion
        } else {
            DeviceTimingMode::Off
        };
        let (slot_id, record) = self.reserve_slot()?;
        let outbox = StateTransferResultSlot::new(slot_id, Arc::clone(resources.identity()));
        let mut reservation = TransferReservation {
            reaper: Arc::clone(self),
            slot_id,
            record,
            resources: Some(resources),
            lane: Arc::clone(&lane),
            timing_mode,
            outbox,
            submission_started: false,
            finished: false,
        };
        let resources = reservation
            .resources
            .as_mut()
            .expect("reservation owns native resources");
        let kind = resources.identity().kind;
        let (commands, geometry) = resources.encode(&lane, timing_mode)?;
        let mut enqueue = lane.reserve_enqueue()?;
        resources.mark_possibly_submitted()?;
        reservation.submission_started = true;
        let submitted = enqueue.submit(commands);
        drop(enqueue);
        match submitted {
            LaneSubmitOutcome::Submitted(fence) => {
                self.checkpoint_timings.record_copies(kind, geometry, false);
                let transition = reservation.install(TransferPhase::InFlight {
                    fence,
                    recovery: CompletionRecoveryState::Unobserved,
                });
                let handle = reservation.handle();
                Ok(match transition {
                    Ok(()) => StateTransferSubmission::Submitted(handle),
                    Err(error) => {
                        StateTransferSubmission::ContractAfterSubmission { error, handle }
                    }
                })
            }
            LaneSubmitOutcome::DefinitelyNotSubmitted(error) => {
                reservation.submission_started = false;
                let resources = reservation
                    .resources
                    .take()
                    .expect("unsubmitted transfer owns resources");
                resources.definitely_not_submitted()?;
                Err(invalid_completion(format!(
                    "native state transfer was not submitted: {error}"
                )))
            }
            LaneSubmitOutcome::PossiblySubmittedPanic => {
                self.checkpoint_timings.record_copies(kind, geometry, true);
                reservation.install(TransferPhase::SubmissionIndeterminate)?;
                Ok(StateTransferSubmission::Indeterminate(reservation.handle()))
            }
        }
    }

    fn with_state_transfer<T>(
        &self,
        slot_id: CompletionSlotId,
        expected: Option<&Arc<StateTransferResultSlot<R>>>,
        action: impl FnOnce(&mut StateTransferRecord<R>) -> Result<T, VNextError>,
    ) -> Result<T, VNextError> {
        let record = self.lookup(slot_id)?;
        let mut guard = record
            .lock()
            .map_err(|_| invalid_completion("native completion slot mutex is poisoned"))?;
        let CompletionRecord::StateTransfer(transfer) = &mut *guard else {
            return Err(invalid_completion(
                "completion slot is not a native state transfer",
            ));
        };
        if expected.is_some_and(|expected| !Arc::ptr_eq(expected, &transfer.outbox)) {
            return Err(invalid_completion(
                "native transfer handle belongs to another result slot",
            ));
        }
        action(transfer)
    }

    pub(in crate::vnext::completion) fn poll_state_transfer_slot(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<Option<StateTransferObservation>, VNextError> {
        self.sweep_state_transfer_slot(slot_id, StateTransferSweepMode::Poll)
    }

    pub(in crate::vnext::completion) fn recover_abandoned_state_transfer_slot(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<Option<StateTransferObservation>, VNextError> {
        self.sweep_state_transfer_slot(slot_id, StateTransferSweepMode::RecoverAbandoned)
    }

    fn sweep_state_transfer_slot(
        &self,
        slot_id: CompletionSlotId,
        mode: StateTransferSweepMode,
    ) -> Result<Option<StateTransferObservation>, VNextError> {
        // A consumer or another sweep may have taken this bounded-snapshot
        // entry already. It is not an outstanding owner in that case.
        let record = self
            .state
            .lock()
            .map_err(|_| invalid_completion("completion reaper state mutex is poisoned"))?
            .slots
            .get(&slot_id)
            .cloned();
        let Some(record) = record else {
            return Ok(None);
        };
        let mut guard = record
            .lock()
            .map_err(|_| invalid_completion("completion slot mutex is poisoned"))?;
        let CompletionRecord::StateTransfer(transfer) = &mut *guard else {
            return Ok(None);
        };
        let observation = match mode {
            StateTransferSweepMode::Poll => transfer.observe(false)?,
            StateTransferSweepMode::RecoverAbandoned => {
                if !transfer.outbox.consumer_abandoned() {
                    return Ok(None);
                }
                match transfer.observe(true)? {
                    StateTransferObservation::Indeterminate
                    | StateTransferObservation::Quarantined => transfer.recover()?,
                    observation => observation,
                }
            }
        };
        if observation == StateTransferObservation::Ready && transfer.outbox.consumer_abandoned() {
            // A terminal fence or successful drain has settled native writes.
            // Move the whole record out so dropping a restore result (which
            // cancels its target) never runs under either registry lock.
            let settled = std::mem::replace(&mut *guard, CompletionRecord::Reaped);
            drop(guard);
            self.remove_exact(slot_id, &record);
            drop(settled);
        }
        Ok(Some(observation))
    }

    pub(crate) fn take_completed_state_transfer(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<Option<StateTransferResult<R>>, VNextError> {
        self.take_state_transfer(slot_id, None)
    }

    /// Recovery workers use the scheduler-owned sweep slot after external
    /// handles detach. The existing registry remains the sole owner.
    pub(crate) fn wait_state_transfer_for_recovery(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<StateTransferObservation, VNextError> {
        self.with_state_transfer(slot_id, None, |transfer| transfer.observe(true))
    }

    pub(crate) fn recover_state_transfer_by_draining_lane(
        &self,
        slot_id: CompletionSlotId,
    ) -> Result<StateTransferObservation, VNextError> {
        self.with_state_transfer(slot_id, None, StateTransferRecord::recover)
    }

    fn take_state_transfer(
        &self,
        slot_id: CompletionSlotId,
        expected: Option<&Arc<StateTransferResultSlot<R>>>,
    ) -> Result<Option<StateTransferResult<R>>, VNextError> {
        let record = self.lookup(slot_id)?;
        let mut guard = record
            .lock()
            .map_err(|_| invalid_completion("native completion slot mutex is poisoned"))?;
        let CompletionRecord::StateTransfer(transfer) = &mut *guard else {
            return Err(invalid_completion(
                "completion slot is not a native state transfer",
            ));
        };
        if expected.is_some_and(|expected| !Arc::ptr_eq(expected, &transfer.outbox)) {
            return Err(invalid_completion(
                "native result consumer belongs to another reservation",
            ));
        }
        let result = transfer.outbox.take()?;
        if result.is_some() {
            *guard = CompletionRecord::Reaped;
            drop(guard);
            self.remove_exact(slot_id, &record);
        }
        Ok(result)
    }
}

#[derive(Clone, Copy)]
enum StateTransferSweepMode {
    Poll,
    RecoverAbandoned,
}
