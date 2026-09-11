//! Public, owner-preserving access to the existing native state-transfer path.
//! Raw writable backing, capture permits and terminal seals remain private.

use super::{
    invalid_completion, CapturedCheckpoint, CompletionReaper, CompletionSlotId, ExecutionLane,
    RestoreFrontierPublication, StateTransferFailureReason, StateTransferHandle,
    StateTransferObservation, StateTransferResult, StateTransferSubmission,
};
use crate::vnext::{
    AdmissionDeferred, AdmissionRejected, CheckpointAuthorityId, CheckpointCapacityMaintenance,
    CheckpointRetentionSkipReason, DeviceErrorReport, DeviceRuntime, DynamicBackingDeferred,
    PlanHash, SequenceAuthorityId, SequenceSession, SequenceSessionEpoch, VNextError,
};
use std::fmt;
use std::sync::Arc;

mod entry;
mod recovery;

/// Immutable copied state with independent physical/logical ownership. Clones
/// pin the same accounted extents; they do not retain the source execution slot.
pub struct SequenceCheckpoint<R: DeviceRuntime> {
    inner: Arc<CapturedCheckpoint<R>>,
}

impl<R: DeviceRuntime> Clone for SequenceCheckpoint<R> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<R: DeviceRuntime> fmt::Debug for SequenceCheckpoint<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SequenceCheckpoint")
            .field("authority", &self.authority())
            .field("completed_tokens", &self.completed_tokens())
            .field("retained_bytes", &self.retained_bytes())
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> SequenceCheckpoint<R> {
    pub fn authority(&self) -> CheckpointAuthorityId {
        self.inner.backing().authority()
    }

    pub fn plan_hash(&self) -> &PlanHash {
        self.inner.byte_plan().plan_hash()
    }

    pub fn layout_fingerprint(&self) -> &str {
        self.inner.byte_plan().layout_fingerprint()
    }

    pub fn completed_tokens(&self) -> usize {
        self.inner.boundary().completed_tokens()
    }

    pub fn token_prefix(&self) -> &[u32] {
        self.inner.boundary().token_prefix()
    }

    /// Original complete token input, including any not-yet-computed suffix.
    pub fn full_input(&self) -> &[u32] {
        self.inner.boundary().full_input()
    }

    pub fn logical_bytes(&self) -> u64 {
        self.inner.backing().logical_bytes()
    }

    /// Exclusive allocator-aligned extents charged to the existing plan ledger.
    pub fn retained_bytes(&self) -> u64 {
        self.inner.backing().extent_bytes()
    }
}

#[derive(Debug)]
pub enum CheckpointAccessSkipReason {
    Disabled,
    Unsupported,
    MissingConditioningEvidence,
    MissingPartitionEvidence,
    BoundaryNotPermitted,
    Busy,
    StaleBacking,
    Retention(CheckpointRetentionSkipReason),
    CapacityDeferred(AdmissionDeferred),
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

/// No device work has been submitted for `Skipped`, `CapacityMaintenance`, or
/// `NotSubmitted`. Maintenance does not reserve the source: after consuming its
/// authority the caller must repeat capture and all boundary/ownership checks.
/// Every possibly-submitted outcome retains a handle in the same native reaper.
#[must_use = "possibly-submitted transfers must reach a terminal or recovery outcome"]
pub enum NativeCheckpointStart<R: DeviceRuntime> {
    Skipped(CheckpointAccessSkipReason),
    CapacityMaintenance {
        reason: CheckpointAccessSkipReason,
        maintenance: CheckpointCapacityMaintenance<R>,
    },
    NotSubmitted(VNextError),
    Submitted(NativeCheckpointTransfer<R>),
    Indeterminate(NativeCheckpointTransfer<R>),
    ContractAfterSubmission {
        error: VNextError,
        transfer: NativeCheckpointTransfer<R>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeCheckpointObservation {
    Pending,
    Indeterminate,
    Quarantined,
    Ready,
}

impl From<StateTransferObservation> for NativeCheckpointObservation {
    fn from(value: StateTransferObservation) -> Self {
        match value {
            StateTransferObservation::Pending => Self::Pending,
            StateTransferObservation::Indeterminate => Self::Indeterminate,
            StateTransferObservation::Quarantined => Self::Quarantined,
            StateTransferObservation::Ready => Self::Ready,
        }
    }
}

#[derive(Debug, Clone)]
pub enum NativeCheckpointFailure {
    FailedButQuiescent(DeviceErrorReport),
    ContractFailedButQuiescent(String),
    AbandonedAfterDrain,
}

impl From<&StateTransferFailureReason> for NativeCheckpointFailure {
    fn from(value: &StateTransferFailureReason) -> Self {
        match value {
            StateTransferFailureReason::FailedButQuiescent(report) => {
                Self::FailedButQuiescent(report.clone())
            }
            StateTransferFailureReason::ContractFailedButQuiescent(reason) => {
                Self::ContractFailedButQuiescent(reason.clone())
            }
            StateTransferFailureReason::AbandonedAfterDrain => Self::AbandonedAfterDrain,
        }
    }
}

#[must_use = "restored state stays gated until its publication is acknowledged"]
pub enum NativeCheckpointResult<R: DeviceRuntime> {
    Captured(SequenceCheckpoint<R>),
    Restored(CheckpointRestorePublication<R>),
    Failed(NativeCheckpointFailure),
}

struct RestoreInput<R: DeviceRuntime> {
    target: Arc<SequenceSession<R>>,
    full_input: Arc<[u32]>,
}

/// One native reaper slot. Polling errors are not proof of quiescence. Keep the
/// handle for blocking recovery or lane drain. Dropping it never releases an
/// unknown write: it marks the exact result abandoned for the existing reaper's
/// sweep/recovery path, which continues owning all native resources meanwhile.
#[must_use = "retain the transfer through terminal delivery or recovery"]
pub struct NativeCheckpointTransfer<R: DeviceRuntime> {
    handle: StateTransferHandle<R>,
    restore: Option<RestoreInput<R>>,
}

impl<R: DeviceRuntime> NativeCheckpointTransfer<R> {
    pub fn slot_id(&self) -> CompletionSlotId {
        self.handle.slot_id()
    }

    pub fn poll(&self) -> Result<NativeCheckpointObservation, VNextError> {
        self.handle.poll().map(Into::into)
    }

    /// Blocking recovery; callers should use their existing completion worker.
    pub fn wait_for_recovery(&self) -> Result<NativeCheckpointObservation, VNextError> {
        self.handle.wait_for_recovery().map(Into::into)
    }

    pub fn recover_by_draining_lane(&self) -> Result<NativeCheckpointObservation, VNextError> {
        self.handle.recover_by_draining_lane().map(Into::into)
    }

    /// Takes a terminal result exactly once. A successful restore installs the
    /// core frontier for the target/input bound at submission, but keeps its
    /// gate closed while the caller publishes executor/scheduler progress.
    pub fn take_result(&mut self) -> Result<Option<NativeCheckpointResult<R>>, VNextError> {
        let Some(result) = self.handle.take()? else {
            return Ok(None);
        };
        match result {
            StateTransferResult::Captured(inner) => {
                if self.restore.is_some() {
                    return Err(invalid_completion(
                        "restore access received a capture result",
                    ));
                }
                Ok(Some(NativeCheckpointResult::Captured(SequenceCheckpoint {
                    inner,
                })))
            }
            StateTransferResult::RestoreReady(pending) => {
                let input = self.restore.take().ok_or_else(|| {
                    invalid_completion("restore result lost its bound target input")
                })?;
                let inner = pending.install_frontier(&input.target, input.full_input)?;
                Ok(Some(NativeCheckpointResult::Restored(
                    CheckpointRestorePublication {
                        inner,
                        target: input.target,
                    },
                )))
            }
            StateTransferResult::Failed(failure) => {
                // The native terminal path already cancelled a failed restore.
                self.restore.take();
                Ok(Some(NativeCheckpointResult::Failed(
                    failure.reason().into(),
                )))
            }
        }
    }
}

impl<R: DeviceRuntime> Drop for NativeCheckpointTransfer<R> {
    fn drop(&mut self) {
        if let Some(input) = &self.restore {
            // Cancellation never releases the native transfer's retained gate.
            // The same reaper must still prove quiescence before freeing bytes.
            let _ = input.target.request_cancel();
        }
        self.handle.abandon_consumer();
    }
}

/// Non-cloneable outer publication owner. Dropping or rejecting it cancels the
/// exact target before its state gate opens. It exposes no writable resources.
#[must_use = "acknowledge only after executor and scheduler progress is published"]
pub struct CheckpointRestorePublication<R: DeviceRuntime> {
    inner: RestoreFrontierPublication<R>,
    target: Arc<SequenceSession<R>>,
}

impl<R: DeviceRuntime> CheckpointRestorePublication<R> {
    pub fn completed_tokens(&self) -> usize {
        self.inner.completed_tokens()
    }

    pub fn token_prefix(&self) -> &[u32] {
        self.inner.token_prefix()
    }

    pub fn target_sequence_authority(&self) -> SequenceAuthorityId {
        self.target.sequence_authority()
    }

    pub fn target_epoch(&self) -> SequenceSessionEpoch {
        self.target.epoch()
    }

    pub fn matches_target(&self, target: &Arc<SequenceSession<R>>) -> bool {
        Arc::ptr_eq(&self.target, target)
    }

    pub fn acknowledge(self) -> Result<(), VNextError> {
        self.inner.acknowledge().map(|_| ())
    }
}
