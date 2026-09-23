//! Checked bridges between request Instants, one planning origin and the
//! independent engine cost clock. No clock conversion refreshes observations.
use super::super::cost_model::{ExecutionFingerprint, WaveExecutionShape};
use super::types::*;
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
