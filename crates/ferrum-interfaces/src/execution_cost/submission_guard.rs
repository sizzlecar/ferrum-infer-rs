//! One guarded dispatch, including the distinction between no submission and
//! an entered device submission. Expected evidence is never execution authority.
use super::{
    ActualRowWork, ActualWaveKind, CanonicalWaveCostShape, CostObservationParticipant,
    PlanRuntimeCostObservationContext, MAX_COST_ROWS,
};
use crate::model_executor::{ExecutorExecutionDeferral, PrefillChunk};
use crate::vnext::ExecutionCostRouteView;
use ferrum_types::{FerrumError, RequestId, Result};

/// Exact phase identity. Prefill remains identified by its admitted request and
/// route authority, never by an invented decode cache handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedWaveInput {
    Prefill { chunk: PrefillChunk },
    Decode { cache_id: String },
}

/// Correlation in the actual physical row order. The route view independently
/// identifies the real sequence authority; these names grant no resource access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedWaveParticipant {
    pub participant_index: usize,
    pub request_id: RequestId,
    pub input: ExpectedWaveInput,
    /// Needed even when no diagnostic recorder is enabled: actual canonical
    /// reconstruction must include the host policy and its captured state.
    pub host: CostObservationParticipant,
}

impl ExpectedWaveParticipant {
    pub fn decode_cache_id(&self) -> Option<&str> {
        match &self.input {
            ExpectedWaveInput::Decode { cache_id } => Some(cache_id),
            ExpectedWaveInput::Prefill { .. } => None,
        }
    }
}

/// Immutable complete cost-route expectation for exactly one physical wave.
/// The core must compare the actual encoded route and its own physical claims,
/// not compare the initial free capacity after allocating those claims.
#[derive(Debug, Clone)]
pub struct ExpectedExecutionCostWave {
    route_view: ExecutionCostRouteView,
    canonical: CanonicalWaveCostShape,
    participants: Vec<ExpectedWaveParticipant>,
}

impl ExpectedExecutionCostWave {
    pub fn new(
        route_view: ExecutionCostRouteView,
        canonical: CanonicalWaveCostShape,
        participants: Vec<ExpectedWaveParticipant>,
    ) -> Result<Self> {
        if participants.is_empty()
            || participants.len() > MAX_COST_ROWS
            || participants.len() != canonical.rows.len()
            || canonical
                .numeric_features
                .as_ref()
                .is_some_and(|features| features.validate(participants.len()).is_err())
        {
            return Err(FerrumError::invalid_request("invalid guarded wave shape"));
        }
        for (index, row) in participants.iter().enumerate() {
            if row.participant_index >= route_view.participant_count()
                || !match (&row.input, &canonical.rows[index]) {
                    (
                        ExpectedWaveInput::Prefill { chunk },
                        ActualRowWork::Prefill {
                            offset,
                            count,
                            total_prompt_tokens,
                        },
                    ) => {
                        canonical.kind != ActualWaveKind::Decode
                            && usize::try_from(*offset).ok() == Some(chunk.tokens_processed())
                            && usize::try_from(*count).ok() == Some(chunk.tokens_to_process())
                            && usize::try_from(*total_prompt_tokens).ok()
                                == Some(chunk.total_prompt_tokens())
                    }
                    (ExpectedWaveInput::Decode { cache_id }, ActualRowWork::Decode { .. }) => {
                        canonical.kind != ActualWaveKind::Prefill && !cache_id.is_empty()
                    }
                    _ => false,
                }
                || row.host.request_id != row.request_id
                || usize::try_from(row.host.input_index).ok() != Some(index)
                || row.host.owner_incarnation == 0
                || row.host.work_generation == 0
                || row.host.output_policy_signature.is_none()
                || participants[..index].iter().any(|prior| {
                    prior.participant_index == row.participant_index
                        || prior.request_id == row.request_id
                        || row
                            .decode_cache_id()
                            .is_some_and(|cache| prior.decode_cache_id() == Some(cache))
                })
            {
                return Err(FerrumError::invalid_request(
                    "invalid guarded wave participant correlation",
                ));
            }
        }
        Ok(Self {
            route_view,
            canonical,
            participants,
        })
    }

    pub fn route_view(&self) -> &ExecutionCostRouteView {
        &self.route_view
    }
    pub fn canonical(&self) -> &CanonicalWaveCostShape {
        &self.canonical
    }
    pub fn participants(&self) -> &[ExpectedWaveParticipant] {
        &self.participants
    }
}

/// Rejecting an expired witness invalidates that proposed submission, not the
/// request. Completion-first scheduling may obtain fresh bounded evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostSubmissionRejection {
    WitnessExpired,
    CostModelChanged,
    FrontierChanged,
    OutputRevoked,
    Cancelled,
    Busy,
}

/// Invoked synchronously after actual route validation and immediately before
/// the device commit. Implementations must never wait, perform I/O, or acquire
/// a blocking lock. An unavailable read returns `Busy` instead.
pub trait NonblockingHostSubmissionGuard: Send + Sync {
    fn check(&self) -> std::result::Result<(), HostSubmissionRejection>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardedNotSubmittedReason {
    ActualRouteMismatch,
    AttributionUnavailable,
    ResourceClaimMismatch,
    HostRejected(HostSubmissionRejection),
}

/// Produced only by the core after all encoded-but-unsubmitted work and its
/// step/readback resources are reconciled. It permits logical publication
/// withdrawal; it does not authorize another submission.
#[derive(Debug)]
pub struct GuardedNotSubmitted {
    reason: GuardedNotSubmittedReason,
}

impl GuardedNotSubmitted {
    pub(crate) fn reconciled(reason: GuardedNotSubmittedReason) -> Self {
        Self { reason }
    }
    pub fn reason(&self) -> GuardedNotSubmittedReason {
        self.reason
    }
}

/// No outer `Result`: even failure must retain its submission phase.
pub enum GuardedDispatchOutcome<T> {
    /// No preparation, encode, or device submission has occurred.
    Unsupported,
    /// The selected resource/frame evidence changed before provider encoding
    /// or device submission. All temporary preparation owners are released.
    /// A prior row may have extended its logical backing, so the caller must
    /// capture a fresh resource view instead of reusing the selected witness.
    ReplanBeforeEncode,
    /// Ordinary capacity deferral before encode or submission.
    Deferred(ExecutorExecutionDeferral),
    /// No submission occurred; any temporary Step was already rolled back.
    /// The one-use maintenance continuation belongs to a later engine turn.
    MaintenanceDeferred {
        deferral: ExecutorExecutionDeferral,
        ticket: crate::model_executor::ExecutorExecutionMaintenanceTicket,
    },
    /// Preparation occurred, no device submission occurred, cleanup completed.
    NotSubmittedAfterPreparation(GuardedNotSubmitted),
    /// Submission was entered or cannot be ruled out. Never reopen its logical
    /// publication on error. The executor owns physical terminal reconciliation.
    Submitted(Result<T>),
}

/// Cost observation is optional; safe guarded execution does not depend on
/// enabling diagnostic instrumentation.
pub type GuardedCostObservation<'a, 'b> = Option<&'a mut PlanRuntimeCostObservationContext<'b>>;
