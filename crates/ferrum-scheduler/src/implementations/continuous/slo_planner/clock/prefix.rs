use super::super::{prefix_rendezvous::*, PlanningSearchStats};
use super::*;

impl PlanningTimeOrigin {
    /// Consume the original controller window, including work before capture.
    /// Neither prefix comparison nor its independent replay renews that window.
    pub fn compare_prefix_rendezvous_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PrefixRendezvousDecision, PlanningTimeError> {
        self.compare_prefix_rendezvous_scoped_with_execution_budget_window(
            planner,
            snapshot,
            offer,
            model,
            maintenance,
            context,
            None,
            window,
            read,
        )
    }

    pub fn compare_prefix_rendezvous_scoped_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<std::sync::Arc<super::super::PlanningObligationSet>>,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PrefixRendezvousDecision, PlanningTimeError> {
        self.prefix_transaction(
            planner,
            snapshot,
            window.into(),
            read,
            |clock| {
                planner.compare_prefix_rendezvous_scoped(
                    snapshot,
                    offer,
                    model,
                    maintenance,
                    context,
                    protection,
                    clock,
                )
            },
            |reason, search| PrefixRendezvousDecision::Unknown { reason, search },
            |decision| match decision {
                PrefixRendezvousDecision::Compared { search, .. }
                | PrefixRendezvousDecision::Unknown { search, .. } => search.clone(),
            },
        )
    }

    /// Rebind the remaining cohort using the same checked request clock and
    /// transaction allowance as normal planning and its eventual fallback.
    pub fn continue_prefix_rendezvous_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        phase: PrefixContinuationPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PrefixContinuationDecision, PlanningTimeError> {
        self.continue_prefix_rendezvous_scoped_with_execution_budget_window(
            planner,
            snapshot,
            offer,
            phase,
            model,
            maintenance,
            context,
            None,
            window,
            read,
        )
    }

    pub fn continue_prefix_rendezvous_scoped_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        phase: PrefixContinuationPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<std::sync::Arc<super::super::PlanningObligationSet>>,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PrefixContinuationDecision, PlanningTimeError> {
        self.prefix_transaction(
            planner,
            snapshot,
            window.into(),
            read,
            |clock| {
                planner.continue_prefix_rendezvous_scoped(
                    snapshot,
                    offer,
                    phase,
                    model,
                    maintenance,
                    context,
                    protection,
                    clock,
                )
            },
            |reason, search| PrefixContinuationDecision::Unknown { reason, search },
            |decision| match decision {
                PrefixContinuationDecision::Ready { search, .. }
                | PrefixContinuationDecision::Unknown { search, .. } => search.clone(),
            },
        )
    }

    /// Rebind the remaining cohort using the same checked request clock and
    /// transaction allowance as normal planning and its eventual fallback.
    pub fn plan_ready_prefix_restore_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &ReadyPrefixRestoreOffer,
        phase: ReadyPrefixPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<ReadyPrefixDecision, PlanningTimeError> {
        self.plan_ready_prefix_restore_scoped_with_execution_budget_window(
            planner,
            snapshot,
            offer,
            phase,
            model,
            maintenance,
            context,
            None,
            window,
            read,
        )
    }

    pub fn plan_ready_prefix_restore_scoped_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &ReadyPrefixRestoreOffer,
        phase: ReadyPrefixPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<std::sync::Arc<super::super::PlanningObligationSet>>,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<ReadyPrefixDecision, PlanningTimeError> {
        self.prefix_transaction(
            planner,
            snapshot,
            window.into(),
            read,
            |clock| {
                planner.plan_ready_prefix_restore_scoped(
                    snapshot,
                    offer,
                    phase,
                    model,
                    maintenance,
                    context,
                    protection,
                    clock,
                )
            },
            |reason, search| ReadyPrefixDecision::Unknown { reason, search },
            |decision| match decision {
                ReadyPrefixDecision::Ready { search, .. }
                | ReadyPrefixDecision::PreferDirect { search, .. }
                | ReadyPrefixDecision::Unknown { search, .. } => search.clone(),
            },
        )
    }

    /// Source-only cache publication consumes the same original controller window.
    pub fn plan_prefix_cache_capture_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixCacheCaptureOffer,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PrefixCacheCaptureDecision, PlanningTimeError> {
        self.prefix_transaction(
            planner,
            snapshot,
            window.into(),
            read,
            |clock| {
                planner.plan_prefix_cache_capture(
                    snapshot,
                    offer,
                    model,
                    maintenance,
                    context,
                    clock,
                )
            },
            |reason, search| PrefixCacheCaptureDecision::Unknown { reason, search },
            |decision| match decision {
                PrefixCacheCaptureDecision::Known { search, .. }
                | PrefixCacheCaptureDecision::Unknown { search, .. } => search.clone(),
            },
        )
    }

    /// Source-only cache publication consumes the same original controller window.
    pub fn plan_prefix_cache_capture_in_phase_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixCacheCaptureOffer,
        phase: PrefixCacheCapturePhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PrefixCacheCaptureDecision, PlanningTimeError> {
        self.plan_prefix_cache_capture_in_phase_scoped_with_execution_budget_window(
            planner,
            snapshot,
            offer,
            phase,
            model,
            maintenance,
            context,
            None,
            window,
            read,
        )
    }

    pub fn plan_prefix_cache_capture_in_phase_scoped_with_execution_budget_window(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixCacheCaptureOffer,
        phase: PrefixCacheCapturePhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<std::sync::Arc<super::super::PlanningObligationSet>>,
        window: impl Into<PlanningPhaseBudget>,
        read: impl FnMut() -> Instant,
    ) -> Result<PrefixCacheCaptureDecision, PlanningTimeError> {
        self.prefix_transaction(
            planner,
            snapshot,
            window.into(),
            read,
            |clock| {
                planner.plan_prefix_cache_capture_in_phase_scoped(
                    snapshot,
                    offer,
                    phase,
                    model,
                    maintenance,
                    context,
                    protection,
                    clock,
                )
            },
            |reason, search| PrefixCacheCaptureDecision::Unknown { reason, search },
            |decision| match decision {
                PrefixCacheCaptureDecision::Known { search, .. }
                | PrefixCacheCaptureDecision::Unknown { search, .. } => search.clone(),
            },
        )
    }

    fn prefix_transaction<D>(
        &self,
        planner: &BoundedSloPlanner,
        snapshot: &SchedulerSnapshot,
        mut phase: PlanningPhaseBudget,
        read: impl FnMut() -> Instant,
        evaluate: impl FnOnce(&mut dyn PlanningClock) -> D,
        unknown: impl Fn(PlanningUnknownReason, PlanningSearchStats) -> D,
        statistics: impl FnOnce(&D) -> PlanningSearchStats,
    ) -> Result<D, PlanningTimeError> {
        if snapshot.observed_at_ns != self.observed_at_ns {
            return Err(PlanningTimeError::SnapshotOriginMismatch);
        }
        phase.window = self.checked_budget_window(phase.window, &planner.settings.search)?;
        let (_, deadline_ns) = match phase.phase_deadlines(&planner.settings.search) {
            Ok(value) => value,
            Err(reason) => return Ok(unknown(reason, Default::default())),
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
        if initial_ns >= deadline_ns {
            return Ok(unknown(
                PlanningUnknownReason::ComputeBudgetExhausted,
                Default::default(),
            ));
        }
        let decision = evaluate(&mut clock);
        let final_ns = clock.now_ns();
        if let Some(error) = clock.error {
            return Err(error);
        }
        if final_ns >= deadline_ns {
            return Ok(unknown(
                PlanningUnknownReason::ComputeBudgetExhausted,
                statistics(&decision),
            ));
        }
        Ok(decision)
    }
}
