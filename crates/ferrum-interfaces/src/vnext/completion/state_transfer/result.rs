use super::{
    canonical_completion_fingerprint, invalid_completion, StateTransferIdentity, StateTransferKind,
};
use crate::vnext::{
    CapturedCheckpointBacking, CheckpointBackingOwner, CheckpointCaptureAttemptId,
    CheckpointInputDependency, CheckpointPartitionNumerics, CheckpointTokenSpanConstraint,
    CompletedSequenceBoundary, CompletedSequenceProvenance, CompletionSlotId, DeviceErrorReport,
    DeviceRuntime, PreparedSequenceStateTransfer, SequenceCheckpointBytePlan,
    SequenceCheckpointLayout, SequenceSession, VNextError,
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

    /// Shared by the public pre-submit access boundary and terminal install.
    /// A native copy never supplies missing conditioning or partition evidence.
    pub(crate) fn validate_reuse_contract(
        &self,
        layout: &SequenceCheckpointLayout,
        full_input: &[u32],
    ) -> Result<(), VNextError> {
        let source = self.boundary();
        let boundary = u64::try_from(source.completed_tokens())
            .map_err(|_| invalid_completion("restore boundary exceeds u64"))?;
        let source_start = u64::try_from(source.capture_span_start())
            .map_err(|_| invalid_completion("capture span start exceeds u64"))?;
        let source_length = u64::try_from(source.full_input().len())
            .map_err(|_| invalid_completion("capture input length exceeds u64"))?;
        let target_length = u64::try_from(full_input.len())
            .map_err(|_| invalid_completion("restore input length exceeds u64"))?;
        if layout.fingerprint()? != self.identity.layout_fingerprint()
            || !layout.inputs().conditioning_inputs().is_empty()
            || layout.providers().iter().any(|provider| {
                provider.contract().partition_numerics()
                    == CheckpointPartitionNumerics::SamePartitionOnly
            })
            || !layout.permits_capture_from(source_start, boundary, source_length)
            || !layout.permits_suffix(boundary, target_length)
            || (layout.input_dependency() == CheckpointInputDependency::EntireTokenInput
                && source.full_input().as_ref() != full_input)
        {
            return Err(invalid_completion(
                "restore lacks the declared input, partition, or legal boundary evidence",
            ));
        }
        Ok(())
    }
}

/// Device completion has made the copied bytes quiescent, but has not committed
/// the target's execution frontier. Engine commit must consume this owner in a
/// later integration step; no guard-extraction or redispatch API is provided.
#[must_use = "restore remains gated until engine commit or cancellation"]
pub(crate) struct PendingRestoreCommit<R: DeviceRuntime> {
    slot_id: CompletionSlotId,
    identity: Arc<StateTransferIdentity>,
    target: Option<PreparedSequenceStateTransfer<R>>,
    checkpoint: Arc<CapturedCheckpoint<R>>,
    layout: SequenceCheckpointLayout,
}

impl<R: DeviceRuntime> PendingRestoreCommit<R> {
    pub(crate) fn identity(&self) -> &Arc<StateTransferIdentity> {
        &self.identity
    }

    pub(crate) fn checkpoint(&self) -> &Arc<CapturedCheckpoint<R>> {
        &self.checkpoint
    }

    /// Installs only the exact native result into its still-gated target.
    /// Full input is verified against the target's own immutable admission,
    /// not a caller assertion or its parent's request work.
    pub(crate) fn install_frontier(
        self,
        expected_target: &Arc<SequenceSession<R>>,
        full_input: Arc<[u32]>,
    ) -> Result<RestoreFrontierPublication<R>, VNextError> {
        let target = self.target.as_ref().expect("pending restore owns target");
        if !Arc::ptr_eq(expected_target, target.session())
            || self.identity.kind() != StateTransferKind::Restore
            || !self.identity.matches_guard(target)
            || !self
                .identity
                .matches_checkpoint(self.checkpoint.backing(), self.checkpoint.byte_plan())
        {
            return Err(invalid_completion(
                "restore publication names another exact target or checkpoint",
            ));
        }
        self.validate_reuse_contract(&full_input)?;
        let seal = SuccessfulRestoreFrontierSeal {
            identity: Arc::clone(&self.identity),
            capture_attempt: self.checkpoint.backing.attempt_id(),
            source: Arc::clone(self.checkpoint.boundary()),
            input_dependency: self.layout.input_dependency(),
            suffix_constraints: self
                .layout
                .providers()
                .iter()
                .map(|provider| provider.contract().boundaries().suffix())
                .collect(),
        };
        let boundary = target.install_imported_frontier(&seal, full_input)?;
        Ok(RestoreFrontierPublication {
            pending: Some(self),
            boundary,
        })
    }

    fn validate_reuse_contract(&self, full_input: &[u32]) -> Result<(), VNextError> {
        if self.layout.fingerprint()? != self.identity.layout_fingerprint() {
            return Err(invalid_completion(
                "restore layout differs from the native transfer identity",
            ));
        }
        self.checkpoint
            .validate_reuse_contract(&self.layout, full_input)
    }
}

impl<R: DeviceRuntime> Drop for PendingRestoreCommit<R> {
    fn drop(&mut self) {
        // request_cancel validates the exact session epoch/fingerprint. It must
        // happen before the target field releases its exclusive reservation.
        // A poisoned slot remains fail-closed in the reservation's own Drop.
        if let Some(target) = &self.target {
            let _ = target.session().request_cancel();
        }
    }
}

/// Cannot be constructed by resource consumers. Only an owned successful
/// native restore result can request a frontier installation.
pub(crate) struct SuccessfulRestoreFrontierSeal {
    identity: Arc<StateTransferIdentity>,
    capture_attempt: CheckpointCaptureAttemptId,
    source: Arc<CompletedSequenceBoundary>,
    input_dependency: CheckpointInputDependency,
    suffix_constraints: Vec<CheckpointTokenSpanConstraint>,
}

impl SuccessfulRestoreFrontierSeal {
    pub(crate) fn matches_target<R: DeviceRuntime>(
        &self,
        target: &PreparedSequenceStateTransfer<R>,
    ) -> bool {
        self.identity.kind() == StateTransferKind::Restore && self.identity.matches_guard(target)
    }

    pub(crate) fn source(&self) -> &Arc<CompletedSequenceBoundary> {
        &self.source
    }

    pub(crate) fn provenance(&self) -> CompletedSequenceProvenance {
        CompletedSequenceProvenance::ImportedCheckpoint {
            capture_attempt: self.capture_attempt,
            restore: Arc::clone(&self.identity),
        }
    }

    pub(crate) fn input_dependency(&self) -> CheckpointInputDependency {
        self.input_dependency
    }

    pub(crate) fn suffix_constraints(&self) -> &[CheckpointTokenSpanConstraint] {
        &self.suffix_constraints
    }
}

/// The core frontier is installed but its gate remains closed while outer
/// executor/scheduler state is published. No target guard is exposed. Dropping
/// or rejecting publication cancels the target before releasing its gate.
#[must_use = "acknowledge outer progress publication or cancel by dropping"]
pub(crate) struct RestoreFrontierPublication<R: DeviceRuntime> {
    pending: Option<PendingRestoreCommit<R>>,
    boundary: Arc<CompletedSequenceBoundary>,
}

impl<R: DeviceRuntime> RestoreFrontierPublication<R> {
    pub(crate) fn boundary(&self) -> &Arc<CompletedSequenceBoundary> {
        &self.boundary
    }

    pub(crate) fn completed_tokens(&self) -> usize {
        self.boundary.completed_tokens()
    }

    pub(crate) fn token_prefix(&self) -> &[u32] {
        self.boundary.token_prefix()
    }

    pub(crate) fn acknowledge(mut self) -> Result<Arc<CompletedSequenceBoundary>, VNextError> {
        let pending = self
            .pending
            .as_mut()
            .expect("publication owns pending target");
        let target = pending
            .target
            .as_ref()
            .expect("publication owns target gate");
        target.acknowledge_imported_frontier(&self.boundary)?;
        // The exact gate was released under the same lock that validated its
        // installed boundary. Suppress PendingRestoreCommit's cancellation.
        drop(pending.target.take());
        drop(self.pending.take());
        Ok(Arc::clone(&self.boundary))
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
            || boundary.frame_id() != identity.source_frame
            || identity.source_provenance_fingerprint.as_deref()
                != Some(
                    canonical_completion_fingerprint(&(
                        boundary.provenance(),
                        boundary.continuation_contract(),
                    ))
                    .as_str(),
                )
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
        layout: SequenceCheckpointLayout,
    ) -> Result<Self, VNextError> {
        let pending = PendingRestoreCommit {
            slot_id,
            identity,
            target: Some(target),
            checkpoint,
            layout,
        };
        if pending.identity.kind() != StateTransferKind::Restore
            || !pending
                .identity
                .matches_guard(pending.target.as_ref().expect("pending owns target"))
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
