//! Checked bridges between request Instants, one planning origin and the
//! independent engine cost clock. No clock conversion refreshes observations.
use super::super::cost_model::{ExecutionFingerprint, WaveExecutionShape};
use super::{types::*, BoundedSloPlanner, PlanningExecutionContext};
use ferrum_interfaces::slo::RequestSloState;
use std::{
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PlanningTimeError {
    #[error("request count exceeds the configured planning snapshot bound")]
    RequestCapacity,
    #[error("request timing is untrusted")]
    UntrustedRequest,
    #[error("request ingress or commit follows the snapshot observation")]
    FutureRequestTiming,
    #[error("request timing observations or endpoint/count state are inconsistent")]
    InconsistentRequestTiming,
    #[error("timestamp precedes this planning origin")]
    BeforeOrigin,
    #[error("time mapping overflows its representable domain")]
    TimeOverflow,
    #[error("cost clock acquisition bracket is reversed")]
    ReversedCostBracket,
    #[error("query precedes the validated cost-clock anchor or acquisition bracket")]
    BeforeCostAnchor,
    #[error("planning clock moved backwards")]
    ClockMovedBackwards,
    #[error("snapshot observation does not use this time origin")]
    SnapshotOriginMismatch,
}

/// One immutable planning epoch selected from actual request ingress Instants.
/// Ingress may precede engine startup; it is never subtracted from a cost epoch.
#[derive(Debug, Clone, Copy)]
pub struct PlanningTimeOrigin {
    origin: Instant,
    observed_at: Instant,
    observed_at_ns: u64,
}

impl PlanningTimeOrigin {
    /// Assess an already accepted or prospective request using the caller's
    /// existing transaction deadline. Policy checks, search and final replay
    /// all consume that same wall-clock allowance; no nested full budget is
    /// created. The evaluator itself preserves the acceptance boundary.
    pub fn assess_admission_with_deadline(
        &self,
        evaluator: &super::time_admission::TimeAdmissionEvaluator<'_>,
        query: super::time_admission::TimeAdmissionQuery<'_>,
        deadline: Instant,
        read: impl FnMut() -> Instant,
    ) -> Result<super::time_admission::TimeAdmissionDecision, PlanningTimeError> {
        self.assess_admission_with_budget_window(
            evaluator,
            query,
            PlanningBudgetWindow {
                started_at_ns: self.observed_at_ns,
                deadline_ns: self.at_ns(deadline)?,
            },
            read,
        )
    }

    /// Preserve the actual outer transaction start, including pre-snapshot work.
    pub fn assess_admission_with_budget_window(
        &self,
        evaluator: &super::time_admission::TimeAdmissionEvaluator<'_>,
        query: super::time_admission::TimeAdmissionQuery<'_>,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<super::time_admission::TimeAdmissionDecision, PlanningTimeError> {
        self.assess_admission_transaction(
            evaluator.planner,
            query,
            window.into(),
            read,
            |query, clock| evaluator.assess(query, clock),
        )
    }

    pub fn assess_admission_execution_with_budget_window(
        &self,
        evaluator: &super::time_admission::TimeAdmissionExecutionEvaluator<'_>,
        query: super::time_admission::TimeAdmissionQuery<'_>,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<super::time_admission::TimeAdmissionDecision, PlanningTimeError> {
        self.assess_admission_transaction(
            evaluator.planner,
            query,
            window.into(),
            read,
            |query, clock| evaluator.assess(query, clock),
        )
    }

    fn assess_admission_transaction<'a>(
        &self,
        planner: &BoundedSloPlanner,
        query: super::time_admission::TimeAdmissionQuery<'a>,
        mut phase: PlanningPhaseBudget,
        read: impl FnMut() -> Instant,
        assess: impl FnOnce(
            super::time_admission::TimeAdmissionQuery<'a>,
            &mut dyn PlanningClock,
        ) -> super::time_admission::TimeAdmissionDecision,
    ) -> Result<super::time_admission::TimeAdmissionDecision, PlanningTimeError> {
        use super::time_admission::*;
        if query.snapshot.observed_at_ns != self.observed_at_ns {
            return Err(PlanningTimeError::SnapshotOriginMismatch);
        }
        let exhausted = || TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::Planning(PlanningUnknownReason::ComputeBudgetExhausted),
            continuation: TimeAdmissionContinuation::WaitForEvidence,
            wait: TimeAdmissionWait {
                review_at_ns: None,
                strict_expiry_at_ns: None,
                snapshot_generation: query.snapshot.generation,
                cost_model_version: query.snapshot.cost_model_version,
            },
        };
        phase.window = self.checked_budget_window(phase.window, &planner.settings.search)?;
        let (_, planner_deadline_ns) = match phase.phase_deadlines(&planner.settings.search) {
            Ok(value) => value,
            Err(reason) => {
                return Ok(TimeAdmissionDecision::Unknown {
                    reason: TimeAdmissionUnknown::Planning(reason),
                    continuation: TimeAdmissionContinuation::WaitForEvidence,
                    wait: TimeAdmissionWait {
                        review_at_ns: None,
                        strict_expiry_at_ns: None,
                        snapshot_generation: query.snapshot.generation,
                        cost_model_version: query.snapshot.cost_model_version,
                    },
                })
            }
        };
        let mut clock = CheckedPlanningClock {
            origin: *self,
            read,
            last_ns: self.observed_at_ns,
            error: None,
            budget: phase,
        };
        let initial_ns = clock.now_ns();
        if let Some(error) = clock.error {
            return Err(error);
        }
        if initial_ns >= planner_deadline_ns {
            return Ok(exhausted());
        }
        let decision = assess(query, &mut clock);
        let final_ns = clock.now_ns();
        if let Some(error) = clock.error {
            return Err(error);
        }
        if final_ns >= planner_deadline_ns {
            return Ok(exhausted());
        }
        Ok(decision)
    }

    /// `maximum_requests` must come from the same bounded snapshot policy as
    /// the planner. Iteration consumes at most that bound plus one; overflow
    /// returns an error instead of omitting an outstanding obligation.
    pub fn from_ingress<'a>(
        observed_at: Instant,
        maximum_requests: NonZeroUsize,
        states: impl IntoIterator<Item = &'a RequestSloState>,
    ) -> Result<Self, PlanningTimeError> {
        let mut origin = observed_at;
        for (index, state) in states.into_iter().enumerate() {
            if index >= maximum_requests.get() {
                return Err(PlanningTimeError::RequestCapacity);
            }
            if !state.is_trusted() {
                return Err(PlanningTimeError::UntrustedRequest);
            }
            if state.ingress() > observed_at {
                return Err(PlanningTimeError::FutureRequestTiming);
            }
            origin = origin.min(state.ingress());
        }
        Self::from_origin(origin, observed_at)
    }

    /// An explicit trusted epoch is useful when snapshots must share an origin.
    /// Every later request projection still rejects an ingress before it.
    pub fn from_origin(origin: Instant, observed_at: Instant) -> Result<Self, PlanningTimeError> {
        let observed_at_ns = elapsed_ns(origin, observed_at)?;
        Ok(Self {
            origin,
            observed_at,
            observed_at_ns,
        })
    }

    pub fn observed_at(&self) -> Instant {
        self.observed_at
    }
    pub fn observed_at_ns(&self) -> u64 {
        self.observed_at_ns
    }
    pub fn at_ns(&self, at: Instant) -> Result<u64, PlanningTimeError> {
        elapsed_ns(self.origin, at)
    }
    pub fn instant_at_ns(&self, at_ns: u64) -> Result<Instant, PlanningTimeError> {
        self.origin
            .checked_add(Duration::from_nanos(at_ns))
            .ok_or(PlanningTimeError::TimeOverflow)
    }

    /// Preserve ingress elapsed time, committed endpoints and sticky failures.
    /// The clone is bounded cold-path projection; a future public observation
    /// getter could avoid cloning the service-class String merely to validate
    /// its hidden last_observation through `observe_wait`.
    pub fn project_request(
        &self,
        state: &RequestSloState,
        maximum_output_tokens: NonZeroU32,
    ) -> Result<RequestTimingView, PlanningTimeError> {
        if !state.is_trusted() {
            return Err(PlanningTimeError::UntrustedRequest);
        }
        if state.ingress() > self.observed_at
            || state.first_commit().is_some_and(|at| at > self.observed_at)
            || state.last_commit().is_some_and(|at| at > self.observed_at)
        {
            return Err(PlanningTimeError::FutureRequestTiming);
        }
        let committed_tokens = u32::try_from(state.committed_tokens())
            .map_err(|_| PlanningTimeError::InconsistentRequestTiming)?;
        if committed_tokens > maximum_output_tokens.get() {
            return Err(PlanningTimeError::InconsistentRequestTiming);
        }
        match (committed_tokens, state.first_commit(), state.last_commit()) {
            (0, None, None) => {}
            (n, Some(first), Some(last))
                if n > 0
                    && state.ingress() <= first
                    && first <= last
                    && (n != 1 || first == last) => {}
            _ => return Err(PlanningTimeError::InconsistentRequestTiming),
        }
        let mut checked = state.clone();
        checked
            .observe_wait(self.observed_at)
            .map_err(|_| PlanningTimeError::InconsistentRequestTiming)?;
        let budget = state.budgets();
        let timing = RequestTimingView {
            ingress_at_ns: self.at_ns(state.ingress())?,
            first_commit_at_ns: state.first_commit().map(|at| self.at_ns(at)).transpose()?,
            last_commit_at_ns: state.last_commit().map(|at| self.at_ns(at)).transpose()?,
            committed_tokens,
            maximum_output_tokens,
            budgets: PlannerLatencyBudgets {
                ttft_ns: duration_ns(budget.ttft())?,
                tpot_ns: duration_ns(budget.tpot())?,
                itl_ns: duration_ns(budget.itl())?,
            },
            // A completed request has no new next-token waiting obligation.
            // Validate its observation chronology above without inventing one.
            slo_failed: if committed_tokens == maximum_output_tokens.get() {
                state.violations().any()
            } else {
                checked.violations().any()
            },
        };
        let actual_deadline = state
            .next_deadline()
            .map_err(|_| PlanningTimeError::InconsistentRequestTiming)?;
        if timing.next_deadline_ns() != Some(self.at_ns(actual_deadline)?) {
            return Err(PlanningTimeError::InconsistentRequestTiming);
        }
        Ok(timing)
    }

    /// The existing PlanningClock trait cannot express a failed checked read.
    /// Keep that implementation private and override *every* decision on a
    /// sticky clock error, including an otherwise returned impossibility.
    pub fn propose_realtime(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        self.propose_with_clock(planner, snapshot, model, resolver, Instant::now)
    }

    pub fn propose_realtime_with_resources(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: &dyn PlanningResourceResolver,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        self.propose_with_clock_and_resources(
            planner,
            snapshot,
            model,
            resolver,
            Some(resources),
            Instant::now,
        )
    }

    fn propose_with_clock(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        read: impl FnMut() -> Instant,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        self.propose_with_clock_and_resources(planner, snapshot, model, resolver, None, read)
    }

    /// Uses one explicit request-clock source through snapshot construction,
    /// planning and final replay (including Tokio's paused test clock).
    pub fn propose_with_clock_and_resources(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: Option<&dyn PlanningResourceResolver>,
        read: impl FnMut() -> Instant,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        let deadline = self
            .observed_at
            .checked_add(planner.settings.search.planning_budget())
            .ok_or(PlanningTimeError::TimeOverflow)?;
        self.propose_with_deadline_and_resources(
            planner, snapshot, model, resolver, resources, deadline, read,
        )
    }

    /// Consume a caller-owned synchronous planning deadline, including work
    /// before this snapshot's observation. The request time origin is unchanged.
    /// The supplied deadline can shorten, never extend, the configured budget.
    pub fn propose_with_deadline_and_resources(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: Option<&dyn PlanningResourceResolver>,
        deadline: Instant,
        read: impl FnMut() -> Instant,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        self.propose_scoped_with_deadline(
            planner, snapshot, model, resolver, resources, None, deadline, read,
        )
    }

    pub fn propose_scoped_with_deadline(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: Option<&dyn PlanningResourceResolver>,
        protection: Option<std::sync::Arc<super::PlanningObligationSet>>,
        deadline: Instant,
        read: impl FnMut() -> Instant,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        self.propose_scoped_with_budget_window(
            planner,
            snapshot,
            model,
            resolver,
            resources,
            protection,
            PlanningBudgetWindow {
                started_at_ns: self.observed_at_ns,
                deadline_ns: self.at_ns(deadline)?,
            },
            read,
        )
    }

    pub fn propose_scoped_with_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: Option<&dyn PlanningResourceResolver>,
        protection: Option<std::sync::Arc<super::PlanningObligationSet>>,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        self.propose_transaction(planner, snapshot, window.into(), read, |clock| {
            if let Some(protection) = protection {
                planner.propose_recovery(snapshot, protection, model, resolver, resources, clock)
            } else {
                match resources {
                    Some(resources) => {
                        planner.propose_with_resources(snapshot, model, resolver, resources, clock)
                    }
                    None => planner.propose(snapshot, model, resolver, clock),
                }
            }
        })
    }

    /// Same clock/budget contract, with the executor's joint state transition.
    pub fn propose_scoped_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<std::sync::Arc<super::PlanningObligationSet>>,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        self.propose_transaction(planner, snapshot, window.into(), read, |clock| {
            if let Some(protection) = protection {
                planner.propose_recovery_with_execution(snapshot, protection, model, context, clock)
            } else {
                planner.propose_with_execution(snapshot, model, context, clock)
            }
        })
    }

    fn propose_transaction(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        mut phase: PlanningPhaseBudget,
        read: impl FnMut() -> Instant,
        propose: impl FnOnce(&mut dyn PlanningClock) -> PlanningDecision,
    ) -> Result<PlanningDecision, PlanningTimeError> {
        if snapshot.observed_at_ns != self.observed_at_ns {
            return Err(PlanningTimeError::SnapshotOriginMismatch);
        }
        phase.window = self.checked_budget_window(phase.window, &planner.settings.search)?;
        let (_, planner_deadline_ns) = match phase.phase_deadlines(&planner.settings.search) {
            Ok(value) => value,
            Err(reason) => {
                return Ok(PlanningDecision::Unknown {
                    reason,
                    search: Default::default(),
                })
            }
        };
        let mut clock = CheckedPlanningClock {
            origin: *self,
            read,
            last_ns: self.observed_at_ns,
            error: None,
            budget: phase,
        };
        let start_ns = clock.now_ns();
        if let Some(error) = clock.error {
            return Err(error);
        }
        if start_ns >= planner_deadline_ns {
            return Ok(budget_exhausted(Default::default()));
        }
        let decision = propose(&mut clock);
        let final_ns = clock.now_ns();
        if let Some(error) = clock.error {
            return Err(error);
        }
        if final_ns >= planner_deadline_ns {
            let search = match decision {
                PlanningDecision::FeasibleWithinHorizon { search, .. }
                | PlanningDecision::ProtectedWithinHorizon { search, .. }
                | PlanningDecision::Unknown { search, .. } => search,
                PlanningDecision::ProvenImpossibleUnderModel { .. } => Default::default(),
            };
            return Ok(budget_exhausted(search));
        }
        Ok(decision)
    }

    fn checked_budget_window(
        &self,
        mut window: PlanningBudgetWindow,
        settings: &ferrum_types::SloPlannerConfig,
    ) -> Result<PlanningBudgetWindow, PlanningTimeError> {
        if window.started_at_ns > self.observed_at_ns {
            return Err(PlanningTimeError::SnapshotOriginMismatch);
        }
        let configured_end = settings
            .max_planning_us
            .get()
            .checked_mul(1000)
            .and_then(|span| window.started_at_ns.checked_add(span))
            .ok_or(PlanningTimeError::TimeOverflow)?;
        window.deadline_ns = window.deadline_ns.min(configured_end);
        Ok(window)
    }
}

fn budget_exhausted(search: super::PlanningSearchStats) -> PlanningDecision {
    PlanningDecision::Unknown {
        reason: PlanningUnknownReason::ComputeBudgetExhausted,
        search,
    }
}

fn elapsed_ns(origin: Instant, at: Instant) -> Result<u64, PlanningTimeError> {
    let elapsed = at
        .checked_duration_since(origin)
        .ok_or(PlanningTimeError::BeforeOrigin)?;
    u64::try_from(elapsed.as_nanos()).map_err(|_| PlanningTimeError::TimeOverflow)
}

fn duration_ns(duration: Duration) -> Result<NonZeroU64, PlanningTimeError> {
    u64::try_from(duration.as_nanos())
        .ok()
        .and_then(NonZeroU64::new)
        .ok_or(PlanningTimeError::TimeOverflow)
}

struct CheckedPlanningClock<F> {
    origin: PlanningTimeOrigin,
    read: F,
    last_ns: u64,
    error: Option<PlanningTimeError>,
    budget: PlanningPhaseBudget,
}
impl<F: FnMut() -> Instant> PlanningClock for CheckedPlanningClock<F> {
    fn planning_budget_window(&self) -> Option<PlanningBudgetWindow> {
        Some(self.budget.window)
    }
    fn planning_phase_deadline_ns(&self) -> Option<u64> {
        self.budget.planner_deadline_ns
    }
    fn now_ns(&mut self) -> u64 {
        if self.error.is_none() {
            match self.origin.at_ns((self.read)()) {
                Ok(now) if now >= self.last_ns => self.last_ns = now,
                Ok(_) => self.error = Some(PlanningTimeError::ClockMovedBackwards),
                Err(error) => self.error = Some(error),
            }
        }
        // No synthetic future time can manufacture an expired-deadline proof.
        // The private wrapper returns Err regardless of the planner's result.
        self.last_ns
    }
}

/// Explicit conversion to an independent monotonic cost clock. The cost value
/// is the engine-local clock; an imported model owns any further profile bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningCostClockAnchor {
    planner_anchor_ns: u64,
    cost_anchor_ns: u64,
    earliest_query_ns: u64,
    acquisition_skew_ns: u64,
}

impl PlanningCostClockAnchor {
    /// Both values must refer to the same physical Instant. No clock epoch is
    /// inferred from a request's ingress or an imported sample timestamp.
    pub fn exact(planner_anchor_ns: u64, cost_anchor_ns: u64) -> Self {
        Self {
            planner_anchor_ns,
            cost_anchor_ns,
            earliest_query_ns: planner_anchor_ns,
            acquisition_skew_ns: 0,
        }
    }

    /// Read `before = Instant::now()`, then the real cost clock, then `after`.
    /// Mapping the later cost read onto `before` only advances the query age,
    /// by at most this bracket's width. It never makes an old sample younger.
    pub fn conservative_read(
        origin: &PlanningTimeOrigin,
        before: Instant,
        cost_ns: u64,
        after: Instant,
    ) -> Result<Self, PlanningTimeError> {
        if after < before {
            return Err(PlanningTimeError::ReversedCostBracket);
        }
        let planner_anchor_ns = origin.at_ns(before)?;
        let earliest_query_ns = origin.at_ns(after)?;
        let acquisition_skew_ns = earliest_query_ns
            .checked_sub(planner_anchor_ns)
            .ok_or(PlanningTimeError::ReversedCostBracket)?;
        Ok(Self {
            planner_anchor_ns,
            cost_anchor_ns: cost_ns,
            earliest_query_ns,
            acquisition_skew_ns,
        })
    }

    pub fn acquisition_skew_ns(&self) -> u64 {
        self.acquisition_skew_ns
    }
    pub fn cost_time_ns(&self, planner_ns: u64) -> Result<u64, PlanningTimeError> {
        if planner_ns < self.earliest_query_ns {
            return Err(PlanningTimeError::BeforeCostAnchor);
        }
        let elapsed = planner_ns
            .checked_sub(self.planner_anchor_ns)
            .ok_or(PlanningTimeError::BeforeCostAnchor)?;
        self.cost_anchor_ns
            .checked_add(elapsed)
            .ok_or(PlanningTimeError::TimeOverflow)
    }
}

pub struct AnchoredPlanningCostModel<'a> {
    model: &'a dyn PlanningCostModel,
    anchor: PlanningCostClockAnchor,
}

impl<'a> AnchoredPlanningCostModel<'a> {
    pub fn new(model: &'a dyn PlanningCostModel, anchor: PlanningCostClockAnchor) -> Self {
        Self { model, anchor }
    }
}

impl PlanningCostModel for AnchoredPlanningCostModel<'_> {
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        self.model.evidence_requirement()
    }
    fn requires_statistical_evidence(&self) -> bool {
        self.model.requires_statistical_evidence()
    }
    fn predict_with_evidence(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        self.model.predict_with_evidence(
            fingerprint,
            shape,
            evidence,
            self.anchor.cost_time_ns(now_ns).ok()?,
        )
    }

    fn supports_empirical_host_content(&self) -> bool {
        self.model.supports_empirical_host_content()
    }
    fn model_version(&self) -> u64 {
        self.model.model_version()
    }
    fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        // Remaining TTL is already relative to this (possibly conservatively
        // advanced) query. Passing it through preserves continued aging.
        self.model
            .predict(fingerprint, shape, self.anchor.cost_time_ns(now_ns).ok()?)
    }
}

#[cfg(test)]
mod tests;
