//! Read-only callbacks at original projection/lookup boundaries. These values
//! cannot be deserialized into planning, execution or qualification authority.
use super::super::cost_model::structured_v2::{StructuredQueryV2, StructuredUnknownV2};
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningQueryPhase {
    Search,
    IndependentReplay { replay: u64 },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningQueryKey {
    pub attempt: u64,
    pub alternative: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningQueryAttemptEnd {
    Completed,
    Unknown(PlanningUnknownReason),
    SequenceViolation,
    Abandoned,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningQueryOutcome {
    Known(PlanningCost),
    StructuredUnknown(StructuredUnknownV2),
    ModelUnavailable,
    ClockMappingFailed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningObservedCost {
    pub outcome: PlanningQueryOutcome,
    /// Original model-clock lookup time; absent if anchor mapping failed.
    pub cost_now_ns: Option<u64>,
}
impl PlanningObservedCost {
    pub fn cost(self) -> Option<PlanningCost> {
        match self.outcome {
            PlanningQueryOutcome::Known(cost) => Some(cost),
            _ => None,
        }
    }
}
/// Callbacks must never block, project, look up a model, or grant authority.
/// A writer copies only bounded passive data. Disabled models return no observer.
pub trait PlanningQueryObserver: Send + Sync {
    fn begin_replay(&self, waves: usize) -> u64;
    fn end_replay(&self, replay: u64, reason: PlanningQueryAttemptEnd);
    /// The planner chose this replay; outer publication/dispatch guards remain.
    fn selected_replay(&self, replay: u64);
    fn begin_attempt(
        &self,
        phase: PlanningQueryPhase,
        depth: usize,
        requests: &[RequestSchedulingView],
        work: &[CandidateWork],
    ) -> u64;
    fn constructed(
        &self,
        key: PlanningQueryKey,
        query: Result<&StructuredQueryV2, StructuredUnknownV2>,
    );
    fn lookup(&self, key: PlanningQueryKey, planning_now_ns: u64, result: PlanningObservedCost);
    /// [queried, constructed) is the exact NotQueried suffix, not missing data.
    /// Unconstructed alternatives and unvisited search tails remain unknown.
    fn end_attempt(
        &self,
        attempt: u64,
        constructed: usize,
        queried: usize,
        reason: PlanningQueryAttemptEnd,
    );
}

pub(super) struct AttemptObservation<'a> {
    observer: &'a dyn PlanningQueryObserver,
    pub id: u64,
    constructed: std::cell::Cell<usize>,
    queried: std::cell::Cell<usize>,
    finished: std::cell::Cell<bool>,
}
impl<'a> AttemptObservation<'a> {
    pub fn new(
        observer: &'a dyn PlanningQueryObserver,
        phase: PlanningQueryPhase,
        depth: usize,
        requests: &[RequestSchedulingView],
        work: &[CandidateWork],
    ) -> Self {
        Self {
            observer,
            id: observer.begin_attempt(phase, depth, requests, work),
            constructed: 0.into(),
            queried: 0.into(),
            finished: false.into(),
        }
    }
    pub fn constructed(
        &self,
        alternative: usize,
        query: Result<&StructuredQueryV2, StructuredUnknownV2>,
    ) {
        self.constructed
            .set(self.constructed.get().saturating_add(1));
        self.observer.constructed(
            PlanningQueryKey {
                attempt: self.id,
                alternative,
            },
            query,
        );
    }
    pub fn lookup(&self, alternative: usize, planning_now_ns: u64, result: PlanningObservedCost) {
        self.queried.set(self.queried.get().saturating_add(1));
        self.observer.lookup(
            PlanningQueryKey {
                attempt: self.id,
                alternative,
            },
            planning_now_ns,
            result,
        );
    }
    pub fn finish(&self, reason: PlanningQueryAttemptEnd) {
        if !self.finished.replace(true) {
            self.observer
                .end_attempt(self.id, self.constructed.get(), self.queried.get(), reason);
        }
    }
}
impl Drop for AttemptObservation<'_> {
    fn drop(&mut self) {
        self.finish(PlanningQueryAttemptEnd::Abandoned);
    }
}

/// Passive detail at an existing rejected projection boundary. This is not a
/// query receipt or execution authority, and does not identify a unique attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningProjectionDiagnostic {
    RootNotExact,
    ProjectionKind,
    FrontierEnvelope,
    FrontierIdentity,
    FrontierPrefill,
    FrontierPhase,
    WorkOwnerMissing,
    WorkRowMissing,
    WorkDuplicateOwner,
    WorkIdentity,
    WorkRange,
    WorkPhase,
    WorkRowOrCompleted,
    WorkKindOrRecurrent,
    CanonicalRows,
    CanonicalKind,
    CanonicalRecurrent,
    CanonicalWork,
    CanonicalHostContent,
    ExecutionPath,
    GraphState,
    RowOrder,
    ExecutionRoute(ferrum_interfaces::vnext::ExecutionCostRouteUnknown),
}
impl PlanningProjectionDiagnostic {
    /// Uses the existing explicit tracing filter. Disabled tracing performs no
    /// formatting, allocation or clock read. Enabled output can consume budget.
    pub fn trace(
        self,
        snapshot: &SchedulerSnapshot,
        depth: Option<usize>,
        kind: ferrum_interfaces::execution_cost::ActualWaveKind,
        rows: usize,
    ) {
        tracing::trace!(
            target: "ferrum::slo_projection",
            diagnostic = ?self,
            generation = snapshot.generation,
            cost_model_version = snapshot.cost_model_version,
            ?depth,
            ?kind,
            rows,
            "SLO candidate projection rejected"
        );
    }
}
