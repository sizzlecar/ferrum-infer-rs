//! A terminal receipt for a typed, pre-encode inference-capacity deferral.
//! It says nothing about allocator maintenance or other runtime GPU activity.
//! Only the original observation context can mint this receipt; serde wire
//! diagnostics cannot be converted back into live settlement evidence.
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Wire/audit identity only. This public DTO cannot mint a live receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallNoSubmissionParticipantV1 {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub input_index: u32,
}
impl From<&CostObservationParticipant> for CallNoSubmissionParticipantV1 {
    fn from(value: &CostObservationParticipant) -> Self {
        Self {
            request_id: value.request_id.clone(),
            owner_incarnation: value.owner_incarnation,
            work_generation: value.work_generation,
            input_index: value.input_index,
        }
    }
}

pub fn no_submission_participant_signature(
    rows: &[CallNoSubmissionParticipantV1],
) -> Option<[u8; 32]> {
    signature(rows.len(), |index| {
        let row = &rows[index];
        (
            &row.request_id,
            row.owner_incarnation,
            row.work_generation,
            row.input_index,
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallNoSubmissionStageV1 {
    SequenceExtension,
    StepAdmission,
    SubmissionWave,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CallNoSubmissionReasonV1 {
    Capacity {
        stage: CallNoSubmissionStageV1,
    },
    RequestState {
        stage: CallNoSubmissionStageV1,
    },
    GuardRollback {
        batch_step: u64,
        batch_invocation: u64,
        lane_id: u64,
        reason: GuardedNotSubmittedReason,
    },
}

impl CallNoSubmissionReasonV1 {
    pub const fn protocol(self) -> &'static str {
        match self {
            Self::GuardRollback { .. } => "ferrum.inference-guard-rollback.v1",
            _ => "ferrum.inference-not-submitted.v1",
        }
    }
    pub const fn valid(self) -> bool {
        match self {
            Self::Capacity { .. } => true,
            Self::RequestState { stage } => {
                matches!(stage, CallNoSubmissionStageV1::SubmissionWave)
            }
            Self::GuardRollback {
                batch_step,
                batch_invocation,
                lane_id,
                ..
            } => batch_step != 0 && batch_invocation != 0 && lane_id != 0,
        }
    }
    pub(super) fn from_deferral(
        value: &crate::model_executor::ExecutorExecutionDeferral,
    ) -> Option<Self> {
        use crate::model_executor::ExecutorExecutionCapacityStage as Stage;
        let stage = match value.stage() {
            Stage::SequenceExtension => CallNoSubmissionStageV1::SequenceExtension,
            Stage::StepAdmission => CallNoSubmissionStageV1::StepAdmission,
            Stage::SubmissionWave => CallNoSubmissionStageV1::SubmissionWave,
        };
        match value {
            crate::model_executor::ExecutorExecutionDeferral::Capacity(_) => {
                Some(Self::Capacity { stage })
            }
            crate::model_executor::ExecutorExecutionDeferral::RequestState(_)
                if stage == CallNoSubmissionStageV1::SubmissionWave =>
            {
                Some(Self::RequestState { stage })
            }
            _ => None,
        }
    }
}

/// Join checked host selections to the same original Step's participant set.
/// The full no-submission record additionally binds original input indices.
pub(crate) fn guarded_participant_signature<'a>(
    count: usize,
    row: impl Fn(usize) -> (&'a RequestId, u64, u64),
) -> Option<[u8; 32]> {
    if count == 0 || count > MAX_COST_ROWS {
        return None;
    }
    let mut hash = Sha256::new();
    hash.update(b"ferrum.guard-rollback.participants.v1\0");
    hash.update((count as u64).to_le_bytes());
    for i in 0..count {
        let (request, owner, generation) = row(i);
        if owner == 0 || generation == 0 || (0..i).any(|j| row(j).0 == request) {
            return None;
        }
        hash.update(request.0.as_bytes());
        hash.update(owner.to_le_bytes());
        hash.update(generation.to_le_bytes());
    }
    Some(hash.finalize().into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CallNoSubmissionV1 {
    pub(super) call_id: u64,
    pub(super) prepare_started_at_ns: u64,
    pub(super) returned_at_ns: u64,
    pub(super) participant_count: usize,
    pub(super) participant_signature: [u8; 32],
    pub(super) reason: CallNoSubmissionReasonV1,
}
impl CallNoSubmissionV1 {
    pub const fn call_id(&self) -> u64 {
        self.call_id
    }
    pub const fn prepare_started_at_ns(&self) -> u64 {
        self.prepare_started_at_ns
    }
    pub const fn returned_at_ns(&self) -> u64 {
        self.returned_at_ns
    }
    pub const fn participant_count(&self) -> usize {
        self.participant_count
    }
    pub const fn participant_signature(&self) -> &[u8; 32] {
        &self.participant_signature
    }
    pub const fn reason(&self) -> CallNoSubmissionReasonV1 {
        self.reason
    }
    pub fn matches_participants(&self, participants: &[CostObservationParticipant]) -> bool {
        participants.len() == self.participant_count
            && participant_signature(participants) == Some(self.participant_signature)
    }
}

/// Original preparation order and owner/frontier, without tokens, runtime
/// objects, heap allocation, or any inference about a missing observation.
pub(super) fn participant_signature(
    participants: &[CostObservationParticipant],
) -> Option<[u8; 32]> {
    signature(participants.len(), |index| {
        let row = &participants[index];
        (
            &row.request_id,
            row.owner_incarnation,
            row.work_generation,
            row.input_index,
        )
    })
}

fn signature<'a>(
    count: usize,
    row: impl Fn(usize) -> (&'a RequestId, u64, u64, u32),
) -> Option<[u8; 32]> {
    if count == 0 || count > 1024 {
        return None;
    }
    let mut digest = Sha256::new();
    digest.update(b"ferrum.no-inference-submission.participants.v1\0");
    digest.update((count as u64).to_le_bytes());
    for index in 0..count {
        let (request, owner, generation, input) = row(index);
        if owner == 0 || generation == 0 {
            return None;
        }
        if (0..index).any(|prior| {
            let r = row(prior);
            r.0 == request || r.3 == input
        }) {
            return None;
        }
        digest.update(request.0.as_bytes());
        digest.update(owner.to_le_bytes());
        digest.update(generation.to_le_bytes());
        digest.update(input.to_le_bytes());
    }
    Some(digest.finalize().into())
}

#[cfg(test)]
mod tests;
