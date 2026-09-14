use crate::vnext::{BatchInvocationId, BatchOperationIdentity, BatchStepId, ExecutionLaneId};

/// Process-local evidence issued only after a successful device terminal,
/// initialization completion and request hazard completion. Resource code
/// must still validate this exact wave and commit its participant frames.
/// It is deliberately neither cloneable nor deserializable.
pub(crate) struct SuccessfulWaveCompletionSeal {
    batch_step_id: BatchStepId,
    batch_invocation_id: BatchInvocationId,
    lane_id: ExecutionLaneId,
    wave_fingerprint: String,
}

impl SuccessfulWaveCompletionSeal {
    // Only the completion owner can mint production success evidence. There
    // is no constructor that accepts a host-reported token position.
    pub(super) fn from_terminal_identity(identity: &BatchOperationIdentity) -> Self {
        Self {
            batch_step_id: identity.batch_step_id(),
            batch_invocation_id: identity.batch_invocation_id(),
            lane_id: identity.lane_id(),
            wave_fingerprint: identity.claimed_backing_fingerprint().to_owned(),
        }
    }

    pub(crate) fn batch_step_id(&self) -> BatchStepId {
        self.batch_step_id
    }

    pub(crate) fn batch_invocation_id(&self) -> BatchInvocationId {
        self.batch_invocation_id
    }

    pub(crate) fn lane_id(&self) -> ExecutionLaneId {
        self.lane_id
    }

    pub(crate) fn wave_fingerprint(&self) -> &str {
        &self.wave_fingerprint
    }

    #[cfg(test)]
    pub(crate) fn test_only(
        batch_step_id: BatchStepId,
        batch_invocation_id: BatchInvocationId,
        lane_id: ExecutionLaneId,
        wave_fingerprint: String,
    ) -> Self {
        Self {
            batch_step_id,
            batch_invocation_id,
            lane_id,
            wave_fingerprint,
        }
    }
}
