//! Loss-tolerant product execution facts, never execution or recovery authority.

use crate::token_execution::{
    ExecutedTokenWork, ExecutionProgress, ExecutionRange, NoSubmissionStage,
};
use ferrum_types::RequestId;
use std::time::{Duration, Instant};

/// An observer bounds the projection before the engine allocates its members.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionAttemptLimits {
    pub max_participants: usize,
    pub max_summary_bytes: usize,
}

impl ExecutionAttemptLimits {
    pub fn accepts(self, participants: usize) -> bool {
        participants <= self.max_participants
            && ExecutionAttemptSummary::projected_bytes(participants)
                .is_some_and(|bytes| bytes <= self.max_summary_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionAttemptDrop {
    ProjectionLimit,
    QueueFull,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionAttemptDelivery {
    Recorded,
    Dropped(ExecutionAttemptDrop),
}

/// Optional, nonblocking observation of one actual `execute_tokens` invocation.
///
/// Implementations must use bounded, nonblocking publication and retain only
/// these pure values. They must not fit a model, wait on a worker, or acquire an
/// execution/scheduler lock in this call. Delivery is independent of execution:
/// dropping a sample never permits retrying model work. No runtime event permit
/// is consumed or duplicated by this interface.
pub trait ExecutionAttemptObserver: Send + Sync {
    fn limits(&self) -> ExecutionAttemptLimits;
    /// Called once after bounded work projection, before the actual invocation.
    /// Optional private observation state must be bounded and nonblocking; it
    /// confers no execution authority and must not retain execution owners.
    /// Outcomes in `participants` are not yet known. A panic is isolated by the
    /// engine. Implementations must obey their CPU budget: synchronous callbacks
    /// cannot be safely interrupted by a timeout.
    fn prepare(
        &self,
        _participants: &[ExecutionAttemptParticipant],
    ) -> Option<Box<dyn ExecutionAttemptCompletion>> {
        None
    }
    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery;
    fn record_drop(&self, _reason: ExecutionAttemptDrop) {}
}

/// Private per-invocation observation continuation. Dropping it has no effect on
/// execution. The engine calls it at most once with the validated outcome and
/// does not also call the observer's ordinary `try_record` for that invocation.
pub trait ExecutionAttemptCompletion: Send {
    fn try_record(self: Box<Self>, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptWorkKind {
    Prefill,
    Decode,
}

/// Only an acknowledged restore into this exact incarnation is a restore fact.
/// `NoneObserved` does not prove that the executor did no internal prefix reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptPrefixRestore {
    NoneObserved,
    Restored { tokens: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptNoSubmissionCause {
    Deferred,
    UnsupportedShape,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptParticipantOutcome {
    /// Missing, ambiguous, or invalid result; no terminal fact is inferred.
    Unverified,
    Completed {
        work: ExecutedTokenWork,
        frontier_committed: bool,
    },
    NotSubmitted {
        stage: NoSubmissionStage,
        cause: AttemptNoSubmissionCause,
    },
    /// A validated participant's original progress, including failed terminal
    /// and unknown submission. It is not a successful service sample.
    Interrupted(ExecutionProgress),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionAttemptEligibility {
    Completed,
    CompletedWithNotSubmitted,
    NotSubmitted,
    /// Error returned without a participant ledger. Submission is unknown.
    ExecutorError,
    /// Invalid mapping, missing/conflicting claim, or failed frontier commit.
    InvalidOutcome,
    /// At least one participant failed or has unknown execution progress.
    Interrupted,
}

/// A fixed-size pure-value member. The UUID is diagnostic only: consumers must
/// not join an asynchronous request/resource ledger by this id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionAttemptParticipant {
    pub request_id: RequestId,
    pub kind: AttemptWorkKind,
    pub requested: ExecutionRange,
    pub context_tokens: usize,
    pub prefix_restore: AttemptPrefixRestore,
    /// Executor-internal reuse is not exposed by the product token protocol.
    /// `None` means unknown, never zero. Engine acknowledgement above is a
    /// separate fact and must not fill in this missing physical-work evidence.
    pub internally_reused_tokens: Option<usize>,
    pub outcome: AttemptParticipantOutcome,
    /// This participant failed during result reconciliation or product commit.
    /// It does not erase already validated and committed model work.
    pub product_failed: bool,
}

/// The wall interval encloses only the executor invocation, excluding Engine
/// sampling, output waits and cleanup. It is process-local monotonic time, not
/// device time or a client SLO. Ranges describe product token work: internal
/// prefix reuse, physical kernel shape and resident state remain unknown.
/// Cancellation before a result may leave no summary; absence is not success.
#[derive(Debug, Clone)]
pub struct ExecutionAttemptSummary {
    pub started_at: Instant,
    pub completed_at: Instant,
    pub participants: Vec<ExecutionAttemptParticipant>,
    pub eligibility: ExecutionAttemptEligibility,
}

impl ExecutionAttemptSummary {
    pub fn executor_wall(&self) -> Option<Duration> {
        self.completed_at.checked_duration_since(self.started_at)
    }

    pub fn projected_bytes(participants: usize) -> Option<usize> {
        participants
            .checked_mul(std::mem::size_of::<ExecutionAttemptParticipant>())?
            .checked_add(std::mem::size_of::<Self>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_limits_check_count_bytes_and_overflow() {
        let bytes = ExecutionAttemptSummary::projected_bytes(2).unwrap();
        let limits = ExecutionAttemptLimits {
            max_participants: 2,
            max_summary_bytes: bytes,
        };
        assert!(limits.accepts(2));
        assert!(!limits.accepts(3));
        assert!(!ExecutionAttemptLimits {
            max_summary_bytes: bytes - 1,
            ..limits
        }
        .accepts(2));
        assert!(ExecutionAttemptSummary::projected_bytes(usize::MAX).is_none());
    }
}
