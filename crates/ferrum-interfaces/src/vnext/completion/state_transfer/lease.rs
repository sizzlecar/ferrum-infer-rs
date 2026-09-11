use super::*;
use crate::vnext::{
    BackingInitializationEncodeError, CheckpointCapturePermit, CompletedSequenceBoundary,
    CompletionSlotId, DeferredDeviceCleanupDomainId, DeviceCommandBatch,
    PreparedBackingInitializations, SequenceCheckpointLayout,
};

enum TransferDestination<R: DeviceRuntime> {
    Capture {
        permit: Option<CheckpointCapturePermit<R>>,
        boundary: Arc<CompletedSequenceBoundary>,
        byte_plan: Arc<SequenceCheckpointBytePlan>,
    },
    Restore {
        checkpoint: Arc<CapturedCheckpoint<R>>,
        initializations: PreparedBackingInitializations,
    },
}

/// All source and destination authority stays together until an exact fence or
/// a lane drain proves that device access has stopped.
pub(in crate::vnext::completion) struct StateTransferLease<R: DeviceRuntime> {
    guard: Option<PreparedSequenceStateTransfer<R>>,
    destination: TransferDestination<R>,
    identity: Arc<StateTransferIdentity>,
    copy_retentions: Option<StateTransferCopyRetentions>,
    may_have_written_target: bool,
}

impl<R: DeviceRuntime> StateTransferLease<R> {
    pub(super) fn capture(
        guard: PreparedSequenceStateTransfer<R>,
        permit: CheckpointCapturePermit<R>,
        byte_plan: Arc<SequenceCheckpointBytePlan>,
        lane: &ExecutionLane<R>,
    ) -> Result<Self, VNextError> {
        let boundary = guard.completed_boundary()?;
        let identity = Arc::new(StateTransferIdentity::from_prepared(
            &guard,
            permit.backing(),
            &byte_plan,
            lane,
            Some(permit.attempt_id()),
        )?);
        Ok(Self {
            guard: Some(guard),
            destination: TransferDestination::Capture {
                permit: Some(permit),
                boundary,
                byte_plan,
            },
            identity,
            copy_retentions: None,
            may_have_written_target: false,
        })
    }

    pub(super) fn restore(
        guard: PreparedSequenceStateTransfer<R>,
        checkpoint: Arc<CapturedCheckpoint<R>>,
        layout: &SequenceCheckpointLayout,
        lane: &ExecutionLane<R>,
    ) -> Result<Self, VNextError> {
        let identity = Arc::new(StateTransferIdentity::from_prepared(
            &guard,
            checkpoint.backing(),
            checkpoint.byte_plan(),
            lane,
            None,
        )?);
        let initializations = PreparedBackingInitializations::prepare_restore(
            &guard,
            layout,
            checkpoint.byte_plan(),
            &identity.fingerprint(),
        )?;
        Ok(Self {
            guard: Some(guard),
            destination: TransferDestination::Restore {
                checkpoint,
                initializations,
            },
            identity,
            copy_retentions: None,
            may_have_written_target: false,
        })
    }

    pub(super) fn identity(&self) -> &Arc<StateTransferIdentity> {
        &self.identity
    }

    pub(in crate::vnext::completion) fn deferred_cleanup_domain(
        &self,
    ) -> DeferredDeviceCleanupDomainId {
        self.guard
            .as_ref()
            .expect("active transfer owns its sequence guard")
            .deferred_cleanup_domain()
    }

    pub(super) fn encode(
        &mut self,
        lane: &ExecutionLane<R>,
    ) -> Result<DeviceCommandBatch<R::Command>, VNextError> {
        if self.copy_retentions.is_some() {
            return Err(invalid_completion("state transfer was already encoded"));
        }
        let guard = self
            .guard
            .as_ref()
            .expect("prepared transfer owns its sequence guard");
        let (checkpoint, byte_plan) = match &self.destination {
            TransferDestination::Capture {
                permit, byte_plan, ..
            } => (
                permit
                    .as_ref()
                    .expect("prepared capture owns a write permit")
                    .backing(),
                byte_plan,
            ),
            TransferDestination::Restore { checkpoint, .. } => {
                (checkpoint.backing(), checkpoint.byte_plan())
            }
        };
        let copies = PreparedStateTransferCopies::encode(guard, checkpoint, byte_plan, lane)
            .map_err(|error| match error {
                StateTransferCopyEncodeError::Contract(error) => error,
                StateTransferCopyEncodeError::Runtime {
                    resource_id,
                    region,
                    error,
                } => invalid_completion(format!(
                    "state copy encode failed for {resource_id:?} at {region:?}: {error}"
                )),
            })?;
        let mut commands = DeviceCommandBatch::with_capacity(copies.len());
        if let TransferDestination::Restore {
            initializations, ..
        } = &self.destination
        {
            initializations
                .encode_restore(guard, lane.runtime(), &mut commands)
                .map_err(|error| match error {
                    BackingInitializationEncodeError::Contract(error) => error,
                    BackingInitializationEncodeError::Runtime { error, .. } => {
                        invalid_completion(format!("restore initialization encode failed: {error}"))
                    }
                })?;
        }
        self.copy_retentions = Some(copies.append_to(&mut commands));
        Ok(commands)
    }

    pub(super) fn mark_possibly_submitted(&mut self) -> Result<(), VNextError> {
        match &mut self.destination {
            TransferDestination::Capture { permit, .. } => permit
                .as_mut()
                .expect("capture owns permit")
                .mark_possibly_submitted()?,
            TransferDestination::Restore { .. } => self.may_have_written_target = true,
        }
        Ok(())
    }

    pub(super) fn mark_fence_installed(&mut self) -> Result<(), VNextError> {
        if let TransferDestination::Restore {
            initializations, ..
        } = &mut self.destination
        {
            initializations.mark_in_flight()?;
        }
        Ok(())
    }

    pub(super) fn definitely_not_submitted(mut self) -> Result<(), VNextError> {
        self.may_have_written_target = false;
        if let TransferDestination::Capture { permit, .. } = &mut self.destination {
            if let Err(failure) = permit
                .take()
                .expect("capture owns permit")
                .definitely_not_submitted()
            {
                let (error, permit) = failure.into_parts();
                // The backend has proved no write happened. A contract failure
                // still must not restore this allocation to the Fresh state.
                let _ = permit.finish_failed_but_quiescent();
                return Err(error);
            }
        }
        Ok(())
    }

    pub(super) fn finish_failed(mut self) {
        if let TransferDestination::Capture { permit, .. } = &mut self.destination {
            if let Some(permit) = permit.take() {
                let _ = permit.finish_failed_but_quiescent();
            }
        }
        if let TransferDestination::Restore {
            initializations, ..
        } = &mut self.destination
        {
            initializations.mark_indeterminate();
        }
        // Drop cancels a possibly-written target before releasing its gate.
    }

    pub(super) fn finish_succeeded(
        mut self,
        slot_id: CompletionSlotId,
    ) -> Result<StateTransferResult<R>, VNextError> {
        match &mut self.destination {
            TransferDestination::Capture {
                permit,
                boundary,
                byte_plan,
            } => {
                let permit = permit.take().expect("capture owns its completion permit");
                let seal = SuccessfulCheckpointCaptureSeal {
                    attempt_id: permit.attempt_id(),
                };
                let backing = permit.finish_succeeded(&seal).map_err(|failure| {
                    let (error, permit) = failure.into_parts();
                    let _ = permit.finish_failed_but_quiescent();
                    error
                })?;
                StateTransferResult::captured(
                    slot_id,
                    Arc::clone(&self.identity),
                    backing,
                    Arc::clone(boundary),
                    Arc::clone(byte_plan),
                )
            }
            TransferDestination::Restore {
                checkpoint,
                initializations,
            } => {
                initializations.finish(true)?;
                StateTransferResult::restore_ready(
                    slot_id,
                    Arc::clone(&self.identity),
                    self.guard.take().expect("restore owns target guard"),
                    Arc::clone(checkpoint),
                )
            }
        }
    }
}

impl<R: DeviceRuntime> Drop for StateTransferLease<R> {
    fn drop(&mut self) {
        if self.may_have_written_target {
            if let Some(guard) = &self.guard {
                let _ = guard.session().request_cancel();
            }
        }
    }
}
