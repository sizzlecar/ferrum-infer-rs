//! Native transfer evidence and owned terminal delivery. The parent completion
//! registry owns submission, fences and recovery; this module never submits or
//! decides that a device operation has completed.

use super::{canonical_completion_fingerprint, invalid_completion, ExecutionLane};
use crate::vnext::{
    CheckpointAuthorityId, CheckpointBackingOwner, DeviceId, DeviceRuntime, ExecutionFrameId,
    ExecutionLaneId, PlanHash, PreparedSequenceStateTransfer, RequestAuthorityId,
    SequenceAuthorityId, SequenceBackingGeneration, SequenceCheckpointBytePlan,
    SequenceSessionEpoch, SequenceSessionFingerprint, SequenceStateTransferKind, VNextError,
};
use serde::Serialize;
use std::num::NonZeroU64;
use std::sync::Arc;

mod result;
pub(crate) use result::*;
mod outbox;
pub(crate) use outbox::*;
mod commands;
use commands::*;
mod lease;
pub(in crate::vnext::completion) use lease::*;
mod reaper;
pub(crate) use reaper::*;

/// Only the native completion path can mint this after the exact capture fence
/// succeeds and the execution lane has retired that submission.
pub(crate) struct SuccessfulCheckpointCaptureSeal {
    attempt_id: crate::vnext::CheckpointCaptureAttemptId,
}

impl SuccessfulCheckpointCaptureSeal {
    pub(crate) fn attempt_id(&self) -> crate::vnext::CheckpointCaptureAttemptId {
        self.attempt_id
    }

    #[cfg(test)]
    pub(crate) fn for_test(attempt_id: crate::vnext::CheckpointCaptureAttemptId) -> Self {
        Self { attempt_id }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateTransferKind {
    Capture,
    Restore,
}

impl From<SequenceStateTransferKind> for StateTransferKind {
    fn from(kind: SequenceStateTransferKind) -> Self {
        match kind {
            SequenceStateTransferKind::CaptureRead => Self::Capture,
            SequenceStateTransferKind::RestoreWrite => Self::Restore,
        }
    }
}

/// Auditable metadata derived from retained owners. It is not a successful-copy
/// certificate, and deserialization cannot recreate it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct StateTransferIdentity {
    kind: StateTransferKind,
    plan_hash: PlanHash,
    layout_fingerprint: String,
    byte_plan_fingerprint: String,
    checkpoint: CheckpointAuthorityId,
    capture_attempt: Option<crate::vnext::CheckpointCaptureAttemptId>,
    boundary_tokens: u64,
    sequence: SequenceAuthorityId,
    request: RequestAuthorityId,
    epoch: SequenceSessionEpoch,
    session_fingerprint: SequenceSessionFingerprint,
    reservation_serial: NonZeroU64,
    backing_generation: SequenceBackingGeneration,
    source_frame: Option<ExecutionFrameId>,
    // The source may itself be imported state and therefore have no model
    // frame. A digest binds its complete typed host provenance without
    // retaining an unbounded chain of prior transfer identities.
    source_provenance_fingerprint: Option<String>,
    lane_id: ExecutionLaneId,
    device_id: DeviceId,
    runtime_implementation_fingerprint: String,
}

impl StateTransferIdentity {
    pub(crate) fn from_prepared<R: DeviceRuntime>(
        guard: &PreparedSequenceStateTransfer<R>,
        checkpoint: &Arc<CheckpointBackingOwner<R>>,
        byte_plan: &SequenceCheckpointBytePlan,
        lane: &ExecutionLane<R>,
        capture_attempt: Option<crate::vnext::CheckpointCaptureAttemptId>,
    ) -> Result<Self, VNextError> {
        checkpoint.validate_transfer_binding(guard, byte_plan, lane)?;
        if (guard.kind() == SequenceStateTransferKind::CaptureRead) != capture_attempt.is_some() {
            return Err(invalid_completion(
                "capture identity requires its exclusive write attempt",
            ));
        }
        if !lane.current_descriptor_matches_snapshot() {
            return Err(invalid_completion(
                "native transfer runtime differs from its execution lane snapshot",
            ));
        }
        let (source_frame, source_provenance_fingerprint) = match guard.kind() {
            SequenceStateTransferKind::CaptureRead => {
                let boundary = guard.completed_boundary()?;
                if boundary.plan_hash() != byte_plan.plan_hash()
                    || u64::try_from(boundary.completed_tokens()).ok() != Some(byte_plan.boundary())
                {
                    return Err(invalid_completion(
                        "transfer byte plan differs from the reserved completed boundary",
                    ));
                }
                (
                    boundary.frame_id(),
                    Some(canonical_completion_fingerprint(&(
                        boundary.provenance(),
                        boundary.continuation_contract(),
                    ))),
                )
            }
            SequenceStateTransferKind::RestoreWrite => {
                guard.ensure_fresh_restore_target()?;
                (None, None)
            }
        };
        let session = guard.session();
        Ok(Self {
            kind: guard.kind().into(),
            plan_hash: byte_plan.plan_hash().clone(),
            layout_fingerprint: byte_plan.layout_fingerprint().to_owned(),
            byte_plan_fingerprint: canonical_completion_fingerprint(byte_plan),
            checkpoint: checkpoint.authority(),
            capture_attempt,
            boundary_tokens: byte_plan.boundary(),
            sequence: session.sequence_authority(),
            request: session.request_authority(),
            epoch: session.epoch(),
            session_fingerprint: session.fingerprint().clone(),
            reservation_serial: guard.reservation_serial(),
            backing_generation: guard.backing().generation(),
            source_frame,
            source_provenance_fingerprint,
            lane_id: lane.id(),
            device_id: lane.descriptor().id.clone(),
            runtime_implementation_fingerprint: lane
                .descriptor()
                .runtime_implementation_fingerprint
                .clone(),
        })
    }

    pub(crate) fn kind(&self) -> StateTransferKind {
        self.kind
    }

    pub(crate) fn lane_id(&self) -> ExecutionLaneId {
        self.lane_id
    }

    pub(crate) fn checkpoint_authority(&self) -> CheckpointAuthorityId {
        self.checkpoint
    }

    pub(crate) fn boundary_tokens(&self) -> u64 {
        self.boundary_tokens
    }

    pub(crate) fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub(crate) fn layout_fingerprint(&self) -> &str {
        &self.layout_fingerprint
    }

    pub(crate) fn fingerprint(&self) -> String {
        #[derive(Serialize)]
        struct Fingerprint<'a> {
            domain: &'static str,
            identity: &'a StateTransferIdentity,
        }
        canonical_completion_fingerprint(&Fingerprint {
            domain: "ferrum.runtime-vnext.native-state-transfer.v1",
            identity: self,
        })
    }

    fn matches_guard<R: DeviceRuntime>(&self, guard: &PreparedSequenceStateTransfer<R>) -> bool {
        let session = guard.session();
        self.kind == guard.kind().into()
            && self.sequence == session.sequence_authority()
            && self.request == session.request_authority()
            && self.epoch == session.epoch()
            && self.session_fingerprint == *session.fingerprint()
            && self.reservation_serial == guard.reservation_serial()
            && self.backing_generation == guard.backing().generation()
    }

    fn matches_checkpoint<R: DeviceRuntime>(
        &self,
        checkpoint: &CheckpointBackingOwner<R>,
        byte_plan: &SequenceCheckpointBytePlan,
    ) -> bool {
        self.checkpoint == checkpoint.authority()
            && self.plan_hash == *byte_plan.plan_hash()
            && self.layout_fingerprint == byte_plan.layout_fingerprint()
            && self.byte_plan_fingerprint == canonical_completion_fingerprint(byte_plan)
            && self.boundary_tokens == byte_plan.boundary()
    }
}
