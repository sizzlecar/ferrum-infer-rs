//! One synchronous planning deadline; execution timing is diagnostic only.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

mod timing_metrics;

#[derive(Debug, Clone, Copy)]
pub(in crate::continuous_engine) enum ControllerStage {
    Release,
    Capture,
    SearchReplay,
    Publication,
    ReadyQueue,
    IterationLock,
    InputPreparation,
    HostGuard,
    ExecutorAwait,
    Reconciliation,
}

const STAGES: [ControllerStage; 10] = [
    ControllerStage::Release,
    ControllerStage::Capture,
    ControllerStage::SearchReplay,
    ControllerStage::Publication,
    ControllerStage::ReadyQueue,
    ControllerStage::IterationLock,
    ControllerStage::InputPreparation,
    ControllerStage::HostGuard,
    ControllerStage::ExecutorAwait,
    ControllerStage::Reconciliation,
];
impl ControllerStage {
    fn label(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Capture => "capture",
            Self::SearchReplay => "search_replay",
            Self::Publication => "publication",
            Self::ReadyQueue => "ready_queue",
            Self::IterationLock => "iteration_lock_wait",
            Self::InputPreparation => "input_preparation",
            Self::HostGuard => "host_guard",
            Self::ExecutorAwait => "executor_await",
            Self::Reconciliation => "reconciliation",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(in crate::continuous_engine) struct ControllerStageAudit {
    pub stage: ControllerStage,
    pub wall_ns: u64,
    pub calls: u64,
}

/// The selected finite plan, not a count of physically executed waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine) struct ControllerWitnessAudit {
    pub waves: u64,
    pub tail_waves: u64,
}

/// Wall times, never thread CPU cycles. ExecutorAwait contains native guard
/// callbacks and device waiting, so these diagnostic stages must not be summed.
/// No value here is a cost-model training sample.
#[derive(Debug, Clone, Copy)]
pub(in crate::continuous_engine) struct ControllerAudit {
    pub planning_wall_ns: u64,
    pub transaction_wall_ns: u64,
    pub stages: [ControllerStageAudit; 10],
    pub search: PlanningSearchStats,
    pub witness: Option<ControllerWitnessAudit>,
    pub backend_submitted: bool,
    pub host_reconciled: bool,
    pub budget_ns: u64,
    pub budget_polls: u64,
    /// Search/replay returned a budget stop, including its reserved-phase end.
    /// This does not imply that the outer transaction's hard deadline elapsed.
    pub planner_budget_exhausted: bool,
    /// Only the original transaction's real hard-deadline clock sets this.
    pub budget_exhausted: bool,
    pub clock_invalid: bool,
    pub outcome: &'static str,
    pub decision: &'static str,
    pub reason: &'static str,
}

pub(super) struct ControllerBudget {
    started_at: Instant,
    deadline: Instant,
    budget_ns: u64,
    planning_end_ns: AtomicU64,
    elapsed: [AtomicU64; 10],
    calls: [AtomicU64; 10],
    polls: AtomicU64,
    planner_exhausted: AtomicBool,
    exhausted: AtomicBool,
    clock_invalid: AtomicBool,
    emitted: AtomicBool,
    search: Mutex<PlanningSearchStats>,
    witness: Mutex<Option<ControllerWitnessAudit>>,
    backend_submitted: AtomicBool,
    host_reconciled: AtomicBool,
    decision: Mutex<(&'static str, &'static str)>,
}

/// A local stopping boundary for optional cost capture and search. It retains
/// the original transaction clock and cannot grant publication time or work.
#[derive(Clone)]
pub(super) struct ControllerOptionalPhase {
    budget: Arc<ControllerBudget>,
    deadline: Instant,
}

impl ControllerOptionalPhase {
    pub fn poll(&self) -> bool {
        let now = slo_clock_now();
        if !self.budget.poll_at(now) {
            return false;
        }
        if now >= self.deadline {
            self.budget.planner_exhausted.store(true, Ordering::Release);
            return false;
        }
        true
    }

    pub fn planning_window(
        &self,
        origin: &PlanningTimeOrigin,
    ) -> std::result::Result<PlanningPhaseBudget, PlanningTimeError> {
        Ok(PlanningPhaseBudget {
            window: self.budget.planning_window(origin)?,
            planner_deadline_ns: Some(origin.at_ns(self.deadline)?),
        })
    }
}

impl ControllerBudget {
    pub fn new(started_at: Instant, allowance: std::time::Duration) -> Result<Arc<Self>> {
        let deadline = started_at
            .checked_add(allowance)
            .ok_or_else(|| FerrumError::config("controller planning deadline overflow"))?;
        let budget_ns = u64::try_from(allowance.as_nanos())
            .map_err(|_| FerrumError::config("controller planning budget overflow"))?;
        Ok(Arc::new(Self {
            started_at,
            deadline,
            budget_ns,
            planning_end_ns: AtomicU64::new(u64::MAX),
            elapsed: std::array::from_fn(|_| AtomicU64::new(0)),
            calls: std::array::from_fn(|_| AtomicU64::new(0)),
            polls: AtomicU64::new(0),
            planner_exhausted: AtomicBool::new(false),
            exhausted: AtomicBool::new(false),
            clock_invalid: AtomicBool::new(false),
            emitted: AtomicBool::new(false),
            search: Mutex::new(PlanningSearchStats::default()),
            witness: Mutex::new(None),
            backend_submitted: AtomicBool::new(false),
            host_reconciled: AtomicBool::new(false),
            decision: Mutex::new(("unknown", "not_evaluated")),
        }))
    }
    pub fn planning_window(
        &self,
        origin: &PlanningTimeOrigin,
    ) -> std::result::Result<PlanningBudgetWindow, PlanningTimeError> {
        Ok(PlanningBudgetWindow {
            started_at_ns: origin.at_ns(self.started_at)?,
            deadline_ns: origin.at_ns(self.deadline)?,
        })
    }
    /// Reserve the work measured while preparing this transaction's executable
    /// completion draft, plus the configured publication window. Like planner
    /// replay reservation, this stops optional work; it is not a duration proof.
    /// A slow callback still fails the unchanged hard deadline at publication.
    pub fn completion_optional_phase(
        self: &Arc<Self>,
        preparation_wall: std::time::Duration,
        publication_reserve_percent: u8,
    ) -> ControllerOptionalPhase {
        let configured = std::time::Duration::from_nanos(
            ((u128::from(self.budget_ns) * u128::from(publication_reserve_percent.min(100))) / 100)
                as u64,
        );
        let deadline = preparation_wall
            .checked_add(configured)
            .and_then(|reserve| self.deadline.checked_sub(reserve))
            .map_or(self.started_at, |end| end.max(self.started_at));
        ControllerOptionalPhase {
            budget: Arc::clone(self),
            deadline,
        }
    }
    pub fn poll(&self) -> bool {
        self.poll_at(slo_clock_now())
    }
    fn poll_at(&self, now: Instant) -> bool {
        add(&self.polls, 1);
        if self.clock_invalid.load(Ordering::Acquire) || now < self.started_at {
            self.clock_invalid.store(true, Ordering::Release);
            return false;
        }
        let live = now < self.deadline;
        if !live {
            self.exhausted.store(true, Ordering::Release);
        }
        live
    }
    pub fn finish_planning(&self) -> bool {
        let now = slo_clock_now();
        let live = self.poll_at(now);
        let _ = self.planning_end_ns.compare_exchange(
            u64::MAX,
            self.duration(self.started_at, now),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        live
    }
    pub fn record_unavailable(&self, reason: &'static str) {
        *self.decision.lock() = ("unknown", reason);
    }
    pub fn record_search(&self, decision: &PlanningDecision) {
        *self.witness.lock() = match decision {
            PlanningDecision::FeasibleWithinHorizon { witness, .. }
            | PlanningDecision::ProtectedWithinHorizon { witness, .. } => {
                u64::try_from(witness.waves).ok().and_then(|waves| {
                    waves
                        .checked_sub(1)
                        .map(|tail_waves| ControllerWitnessAudit { waves, tail_waves })
                })
            }
            _ => None,
        };
        let search = match decision {
            PlanningDecision::FeasibleWithinHorizon { search, .. }
            | PlanningDecision::ProtectedWithinHorizon { search, .. }
            | PlanningDecision::Unknown { search, .. } => *search,
            // These endpoint/deadline proofs precede candidate search. The
            // decision carries no search work; elapsed stage time is still
            // recorded independently by this transaction's stage guard.
            PlanningDecision::ProvenImpossibleUnderModel { .. } => Default::default(),
        };
        *self.search.lock() = search;
        *self.decision.lock() = match decision {
            PlanningDecision::FeasibleWithinHorizon { .. } => {
                ("feasible_within_horizon", "complete_obligation_set")
            }
            PlanningDecision::ProtectedWithinHorizon { .. } => {
                ("protected_within_horizon", "partial_forward_obligations")
            }
            PlanningDecision::ProvenImpossibleUnderModel { .. } => {
                ("impossible_under_model", "complete_obligation_set")
            }
            PlanningDecision::Unknown { reason, .. } => ("unknown", unknown_label(*reason)),
        };
        if matches!(
            decision,
            PlanningDecision::Unknown {
                reason: PlanningUnknownReason::ComputeBudgetExhausted,
                ..
            }
        ) {
            self.planner_exhausted.store(true, Ordering::Release);
            // The planner may stop at its replay boundary while publication
            // still has time. A reason alone cannot certify hard exhaustion.
            let _ = self.poll();
        }
    }
    /// Called only for GuardedDispatchOutcome::Submitted, even when its result
    /// fails. A decision or scheduler publication cannot set this fact.
    pub fn record_backend_submitted(&self) {
        self.backend_submitted.store(true, Ordering::Release);
    }
    /// Called after the submitted wave's real fenced commits and output cleanup.
    pub fn record_host_reconciled(&self) {
        self.host_reconciled.store(true, Ordering::Release);
    }
    pub fn stage(&self, stage: ControllerStage) -> ControllerStageTimer<'_> {
        ControllerStageTimer {
            budget: self,
            stage,
            began: slo_clock_now(),
        }
    }
    pub fn record_ready_queue(&self) {
        let end = self.planning_end_ns.load(Ordering::Acquire);
        if let Some(began) = (end != u64::MAX)
            .then(|| {
                self.started_at
                    .checked_add(std::time::Duration::from_nanos(end))
            })
            .flatten()
        {
            self.record(ControllerStage::ReadyQueue, began, slo_clock_now());
        }
    }
    fn duration(&self, began: Instant, ended: Instant) -> u64 {
        match ended
            .checked_duration_since(began)
            .and_then(|d| u64::try_from(d.as_nanos()).ok())
        {
            Some(value) => value,
            None => {
                self.clock_invalid.store(true, Ordering::Release);
                0
            }
        }
    }
    fn record(&self, stage: ControllerStage, began: Instant, ended: Instant) {
        add(&self.elapsed[stage as usize], self.duration(began, ended));
        add(&self.calls[stage as usize], 1);
    }
    fn take_audit(&self, outcome: &'static str) -> Option<ControllerAudit> {
        if self.emitted.swap(true, Ordering::AcqRel) {
            return None;
        }
        let (decision, reason) = *self.decision.lock();
        Some(ControllerAudit {
            planning_wall_ns: self.planning_end_ns.load(Ordering::Acquire),
            transaction_wall_ns: self.duration(self.started_at, slo_clock_now()),
            stages: std::array::from_fn(|i| ControllerStageAudit {
                stage: STAGES[i],
                wall_ns: self.elapsed[i].load(Ordering::Acquire),
                calls: self.calls[i].load(Ordering::Acquire),
            }),
            search: *self.search.lock(),
            witness: *self.witness.lock(),
            backend_submitted: self.backend_submitted.load(Ordering::Acquire),
            host_reconciled: self.host_reconciled.load(Ordering::Acquire),
            budget_ns: self.budget_ns,
            budget_polls: self.polls.load(Ordering::Acquire),
            planner_budget_exhausted: self.planner_exhausted.load(Ordering::Acquire),
            budget_exhausted: self.exhausted.load(Ordering::Acquire),
            clock_invalid: self.clock_invalid.load(Ordering::Acquire),
            outcome,
            decision,
            reason,
        })
    }
}

fn add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_add(value))
    });
}
pub(super) struct ControllerStageTimer<'a> {
    budget: &'a ControllerBudget,
    stage: ControllerStage,
    began: Instant,
}
impl Drop for ControllerStageTimer<'_> {
    fn drop(&mut self) {
        self.budget.record(self.stage, self.began, slo_clock_now());
    }
}

impl EngineInner {
    pub(super) fn finish_controller_audit(&self, budget: &ControllerBudget, outcome: &'static str) {
        let Some(audit) = budget.take_audit(outcome) else {
            return;
        };
        {
            let mut controller = self.slo_controller.lock();
            timing_metrics::accumulate(
                controller.timing.get_or_insert_with(Default::default),
                &audit,
            );
            controller.last_audit = Some(audit);
        }
        histogram!("ferrum.engine.slo_controller_planning_seconds", "outcome" => audit.outcome)
            .record(audit.planning_wall_ns as f64 / 1e9);
        histogram!("ferrum.engine.slo_controller_transaction_seconds", "outcome" => audit.outcome)
            .record(audit.transaction_wall_ns as f64 / 1e9);
        for stage in audit.stages {
            if stage.calls != 0 {
                histogram!("ferrum.engine.slo_controller_stage_seconds", "stage" => stage.stage.label())
                    .record(stage.wall_ns as f64 / 1e9);
                counter!("ferrum.engine.slo_controller_stage_calls_total", "stage" => stage.stage.label())
                    .increment(stage.calls);
            }
        }
        counter!("ferrum.engine.slo_controller_budget_exhausted_total")
            .increment(u64::from(audit.budget_exhausted));
        counter!("ferrum.engine.slo_controller_planner_budget_exhausted_total")
            .increment(u64::from(audit.planner_budget_exhausted));
        counter!("ferrum.engine.slo_controller_backend_submitted_total")
            .increment(u64::from(audit.backend_submitted));
        counter!("ferrum.engine.slo_controller_host_reconciled_total")
            .increment(u64::from(audit.host_reconciled));
        if let Some(witness) = audit.witness {
            for (stage, reached) in [
                ("decision", true),
                ("backend_submitted", audit.backend_submitted),
                ("host_reconciled", audit.host_reconciled),
            ] {
                if !reached {
                    continue;
                }
                counter!("ferrum.engine.slo_controller_witness_samples_total", "stage" => stage)
                    .increment(1);
                histogram!("ferrum.engine.slo_controller_witness_waves", "stage" => stage)
                    .record(witness.waves as f64);
                counter!("ferrum.engine.slo_controller_witness_tail_waves_total", "stage" => stage)
                    .increment(witness.tail_waves);
                counter!("ferrum.engine.slo_controller_witness_nonempty_tail_total", "stage" => stage)
                    .increment(u64::from(witness.tail_waves > 0));
            }
        }
        for (kind, value) in [
            ("enumeration_attempts", audit.search.enumeration_attempts),
            ("generated_candidates", audit.search.generated_candidates),
            ("expanded_candidates", audit.search.expanded_candidates),
            ("candidate_truncations", audit.search.candidate_truncations),
            ("search_soft_stops", audit.search.search_soft_stops),
            ("beam_pruned_nodes", audit.search.beam_pruned_nodes),
            (
                "cost_unknown_candidates",
                audit.search.cost_unknown_candidates,
            ),
            (
                "shape_unknown_candidates",
                audit.search.shape_unknown_candidates,
            ),
            (
                "resource_unknown_candidates",
                audit.search.resource_unknown_candidates,
            ),
        ] {
            counter!("ferrum.engine.slo_controller_search_work_total", "kind" => kind)
                .increment(value as u64);
        }
        tracing::trace!(?audit, budget_ns=audit.budget_ns, budget_polls=audit.budget_polls,
            planner_budget_exhausted=audit.planner_budget_exhausted,
            budget_exhausted=audit.budget_exhausted,
            decision=audit.decision, reason=audit.reason,
            clock_invalid=audit.clock_invalid, search=?audit.search,
            "SLO controller wall-time audit; executor await overlaps host guard, not additive");
    }
}

#[cfg(test)]
mod tests;
