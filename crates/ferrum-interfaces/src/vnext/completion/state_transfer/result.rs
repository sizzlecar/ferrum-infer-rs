use super::{invalid_completion, StateTransferIdentity, StateTransferKind};
use crate::vnext::{
    CapturedCheckpointBacking, CheckpointBackingOwner, CompletedSequenceBoundary, CompletionSlotId,
    DeviceErrorReport, DeviceRuntime, PreparedSequenceStateTransfer, SequenceCheckpointBytePlan,
    VNextError,
};
use std::sync::Arc;

/// A successful native capture retains independent checkpoint storage and host
/// evidence only. The source sequence/session is deliberately not retained.
pub(crate) struct CapturedCheckpoint<R: DeviceRuntime> {
    slot_id: CompletionSlotId,
    identity: Arc<StateTransferIdentity>,
    backing: CapturedCheckpointBacking<R>,
    boundary: Arc<CompletedSequenceBoundary>,
    byte_plan: Arc<SequenceCheckpointBytePlan>,
}

impl<R: DeviceRuntime> CapturedCheckpoint<R> {
    pub(crate) fn identity(&self) -> &Arc<StateTransferIdentity> {
        &self.identity
    }

    /// Read/retention projection only. This Arc is never authority to reuse an
    /// already-published checkpoint as another capture's writable destination.
    pub(crate) fn backing(&self) -> &Arc<CheckpointBackingOwner<R>> {
        self.backing.backing()
    }

    pub(crate) fn boundary(&self) -> &Arc<CompletedSequenceBoundary> {
        &self.boundary
    }

    pub(crate) fn byte_plan(&self) -> &Arc<SequenceCheckpointBytePlan> {
        &self.byte_plan
    }
}

/// Device completion has made the copied bytes quiescent, but has not committed
/// the target's execution frontier. Engine commit must consume this owner in a
/// later integration step; no guard-extraction or redispatch API is provided.
#[must_use = "restore remains gated until engine commit or cancellation"]
pub(crate) struct PendingRestoreCommit<R: DeviceRuntime> {
    slot_id: CompletionSlotId,
    identity: Arc<StateTransferIdentity>,
    target: PreparedSequenceStateTransfer<R>,
    checkpoint: Arc<CapturedCheckpoint<R>>,
}

impl<R: DeviceRuntime> PendingRestoreCommit<R> {
    pub(crate) fn identity(&self) -> &Arc<StateTransferIdentity> {
        &self.identity
    }

    pub(crate) fn checkpoint(&self) -> &Arc<CapturedCheckpoint<R>> {
        &self.checkpoint
    }
}

impl<R: DeviceRuntime> Drop for PendingRestoreCommit<R> {
    fn drop(&mut self) {
        // request_cancel validates the exact session epoch/fingerprint. It must
        // happen before the target field releases its exclusive reservation.
        // A poisoned slot remains fail-closed in the reservation's own Drop.
        let _ = self.target.session().request_cancel();
    }
}

#[derive(Debug, Clone)]
pub(crate) enum StateTransferFailureReason {
    FailedButQuiescent(DeviceErrorReport),
    ContractFailedButQuiescent(String),
    /// Drain proved safety to release, never successful execution of the copy.
    AbandonedAfterDrain,
}

#[derive(Debug, Clone)]
pub(crate) struct StateTransferFailure {
    slot_id: CompletionSlotId,
    identity: Arc<StateTransferIdentity>,
    reason: StateTransferFailureReason,
}

impl StateTransferFailure {
    pub(crate) fn reason(&self) -> &StateTransferFailureReason {
        &self.reason
    }
}

/// Only terminal, owned outcomes enter the outbox. Pending or unknown device
/// work remains owned by the parent reaper's active/quarantined record.
#[must_use = "a native terminal result carries ownership or a terminal failure"]
pub(crate) enum StateTransferResult<R: DeviceRuntime> {
    Captured(Arc<CapturedCheckpoint<R>>),
    RestoreReady(PendingRestoreCommit<R>),
    Failed(StateTransferFailure),
}

impl<R: DeviceRuntime> StateTransferResult<R> {
    pub(crate) fn slot_id(&self) -> CompletionSlotId {
        match self {
            Self::Captured(checkpoint) => checkpoint.slot_id,
            Self::RestoreReady(restore) => restore.slot_id,
            Self::Failed(failure) => failure.slot_id,
        }
    }

    pub(crate) fn identity(&self) -> &Arc<StateTransferIdentity> {
        match self {
            Self::Captured(checkpoint) => checkpoint.identity(),
            Self::RestoreReady(restore) => restore.identity(),
            Self::Failed(failure) => &failure.identity,
        }
    }

    /// The parent completion owner calls this only after the exact native
    /// capture fence and all resource terminal transitions succeeded.
    /// The consumed backing projection proves the exact destination permit
    /// reached successful completion and can never be reserved for writing again.
    pub(in crate::vnext::completion) fn captured(
        slot_id: CompletionSlotId,
        identity: Arc<StateTransferIdentity>,
        backing: CapturedCheckpointBacking<R>,
        boundary: Arc<CompletedSequenceBoundary>,
        byte_plan: Arc<SequenceCheckpointBytePlan>,
    ) -> Result<Self, VNextError> {
        if identity.kind() != StateTransferKind::Capture
            || identity.capture_attempt != Some(backing.attempt_id())
            || !identity.matches_checkpoint(backing.backing(), &byte_plan)
            || boundary.plan_hash() != byte_plan.plan_hash()
            || u64::try_from(boundary.completed_tokens()).ok() != Some(byte_plan.boundary())
            || boundary.epoch() != identity.epoch
            || boundary.session_fingerprint() != &identity.session_fingerprint
            || boundary.backing_generation() != identity.backing_generation
            || Some(boundary.frame_id()) != identity.source_frame
        {
            return Err(invalid_completion(
                "completed capture owners do not match its identity",
            ));
        }
        Ok(Self::Captured(Arc::new(CapturedCheckpoint {
            slot_id,
            identity,
            backing,
            boundary,
            byte_plan,
        })))
    }

    /// Even a constructor mismatch cancels before dropping an already-written
    /// target; it must never roll that target back into normal admission.
    pub(in crate::vnext::completion) fn restore_ready(
        slot_id: CompletionSlotId,
        identity: Arc<StateTransferIdentity>,
        target: PreparedSequenceStateTransfer<R>,
        checkpoint: Arc<CapturedCheckpoint<R>>,
    ) -> Result<Self, VNextError> {
        let pending = PendingRestoreCommit {
            slot_id,
            identity,
            target,
            checkpoint,
        };
        if pending.identity.kind() != StateTransferKind::Restore
            || !pending.identity.matches_guard(&pending.target)
            || !pending
                .identity
                .matches_checkpoint(pending.checkpoint.backing(), pending.checkpoint.byte_plan())
        {
            return Err(invalid_completion(
                "completed restore owners do not match its identity",
            ));
        }
        Ok(Self::RestoreReady(pending))
    }

    /// Resource cleanup/target cancellation is the consuming completion owner's
    /// responsibility and must already be safe when this metadata is published.
    pub(in crate::vnext::completion) fn failed(
        slot_id: CompletionSlotId,
        identity: Arc<StateTransferIdentity>,
        reason: StateTransferFailureReason,
    ) -> Self {
        Self::Failed(StateTransferFailure {
            slot_id,
            identity,
            reason,
        })
    }
}
