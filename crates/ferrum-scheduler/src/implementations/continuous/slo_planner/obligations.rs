//! Immutable recovery scope. It never rewrites clocks, removes owners, or
//! grants execution authority. Fairness counts real service opportunities,
//! rather than repeatedly promising service at the end of a moving horizon.
use super::types::*;
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryServiceDebt {
    eligible_bypasses: usize,
    bound: NonZeroUsize,
}

impl RecoveryServiceDebt {
    pub const fn new(bound: NonZeroUsize) -> Self {
        Self {
            eligible_bypasses: 0,
            bound,
        }
    }
    pub const fn eligible_bypasses(self) -> usize {
        self.eligible_bypasses
    }
    pub const fn bound(self) -> NonZeroUsize {
        self.bound
    }
    pub fn due(self) -> bool {
        self.eligible_bypasses >= self.bound.get()
    }
    /// Charge only an eligible, unserved owner on an actually submitted peer
    /// wave. Saturation retains the oldest debt without integer wraparound.
    pub fn bypass(&mut self) {
        self.eligible_bypasses = self.eligible_bypasses.saturating_add(1);
    }
    /// Caller must possess the owner's real progress receipt, not a forecast.
    pub fn progressed(&mut self) {
        self.eligible_bypasses = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardObligation {
    Protected,
    CompletionOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestObligation {
    pub key: RequestWorkKey,
    pub forward: ForwardObligation,
    pub historical_violation: bool,
    pub expired_deadline_ns: Option<u64>,
    /// Original indices; an expired checkpoint is distinct from an SLO miss.
    pub expired_control_milestones: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanningObligationSet {
    generation: u64,
    model_version: u64,
    classified_at_ns: u64,
    initial: Vec<RequestSchedulingView>,
    rows: Vec<RequestObligation>,
}

impl PlanningObligationSet {
    /// Capture once at the transaction's real start. Final replay may reject
    /// this scope but may not demote newly-late protected owners.
    pub fn capture(
        snapshot: &SchedulerSnapshot,
        now_ns: u64,
    ) -> Result<Self, PlanningUnknownReason> {
        Self::capture_with_budget(snapshot, now_ns, &mut || Ok(()))
    }

    pub fn capture_with_budget(
        snapshot: &SchedulerSnapshot,
        now_ns: u64,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Self, PlanningUnknownReason> {
        if now_ns < snapshot.observed_at_ns || snapshot.requests.len() > 256 {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        let mut rows = Vec::with_capacity(snapshot.requests.len());
        let mut points = 0usize;
        for request in &snapshot.requests {
            poll()?;
            let timing = &request.timing;
            let historical_violation = historical_violation(timing)?;
            let expired_deadline_ns = if timing.completed() {
                None
            } else {
                let deadline = timing
                    .next_deadline_ns()
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
                (deadline < now_ns).then_some(deadline)
            };
            let mut expired_control_milestones = Vec::new();
            if timing.committed_tokens == 0 {
                if let RequestPhaseView::Prefill(progress) = &request.phase {
                    points = points
                        .checked_add(progress.milestones.len())
                        .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
                    if points > 4096 {
                        return Err(PlanningUnknownReason::InvalidSnapshot);
                    }
                    let completed = progress
                        .reference
                        .work_at(progress.logical_high_water)
                        .ok_or(PlanningUnknownReason::MissingReferenceWork)?;
                    for (index, milestone) in progress.milestones.iter().enumerate() {
                        poll()?;
                        if milestone.at_ns < now_ns
                            && completed < milestone.required_reference_work_ns
                        {
                            expired_control_milestones.push(index);
                        }
                    }
                }
            }
            rows.push(RequestObligation {
                key: request.key.clone(),
                historical_violation,
                forward: if expired_deadline_ns.is_some() {
                    ForwardObligation::CompletionOnly
                } else {
                    ForwardObligation::Protected
                },
                expired_deadline_ns,
                expired_control_milestones,
            });
        }
        Ok(Self {
            generation: snapshot.generation,
            model_version: snapshot.cost_model_version,
            classified_at_ns: now_ns,
            initial: snapshot.requests.clone(),
            rows,
        })
    }
    pub fn rows(&self) -> &[RequestObligation] {
        &self.rows
    }
    pub fn classified_at_ns(&self) -> u64 {
        self.classified_at_ns
    }
    pub fn needs_recovery(&self) -> bool {
        self.rows.iter().any(|row| {
            row.historical_violation
                || row.expired_deadline_ns.is_some()
                || !row.expired_control_milestones.is_empty()
        })
    }
    pub fn new_time_promises_closed(&self) -> bool {
        self.rows
            .iter()
            .any(|row| row.historical_violation || row.expired_deadline_ns.is_some())
    }
    pub(super) fn matches(&self, snapshot: &SchedulerSnapshot) -> bool {
        self.generation == snapshot.generation
            && self.model_version == snapshot.cost_model_version
            && self.initial == snapshot.requests
    }
    pub(super) fn protects(&self, index: usize) -> bool {
        self.rows[index].forward == ForwardObligation::Protected
    }
    pub(super) fn requires_milestone(&self, row: usize, index: usize) -> bool {
        self.protects(row) && !self.rows[row].expired_control_milestones.contains(&index)
    }
    pub fn required_first_service(&self) -> Option<&RequestWorkKey> {
        self.required_service(&self.initial)
            .map(|index| &self.rows[index].key)
    }
    pub(super) fn recovery_owner(&self, index: usize) -> bool {
        let row = &self.rows[index];
        row.historical_violation
            || row.expired_deadline_ns.is_some()
            || !row.expired_control_milestones.is_empty()
    }
    pub(super) fn required_service(&self, requests: &[RequestSchedulingView]) -> Option<usize> {
        requests
            .iter()
            .enumerate()
            .filter(|(i, row)| {
                self.recovery_owner(*i)
                    && row.readiness == RequestReadiness::Ready
                    && !row.timing.completed()
                    && row.recovery_service.due()
            })
            .min_by_key(|(_, row)| {
                (
                    std::cmp::Reverse(row.recovery_service.eligible_bypasses()),
                    row.timing.ingress_at_ns,
                    row.fairness_rank,
                )
            })
            .map(|(i, _)| i)
    }
}

pub fn historical_violation(timing: &RequestTimingView) -> Result<bool, PlanningUnknownReason> {
    let mut failed = timing.slo_failed;
    if let (Some(first), Some(last)) = (timing.first_commit_at_ns, timing.last_commit_at_ns) {
        let first_deadline = timing
            .ingress_at_ns
            .checked_add(timing.budgets.ttft_ns.get())
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        let elapsed = last
            .checked_sub(first)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        failed |= first > first_deadline
            || (timing.committed_tokens >= 2
                && u128::from(elapsed)
                    > u128::from(timing.committed_tokens - 1)
                        * u128::from(timing.budgets.tpot_ns.get()));
    }
    Ok(failed)
}
