//! Reprove the remaining trajectory of one existing hold on a fresh full queue.
use super::*;

pub(super) fn route(phase: PrefixContinuationPhase) -> Route {
    match phase {
        PrefixContinuationPhase::HeldAwaitingProducer
        | PrefixContinuationPhase::AtCaptureBoundary { .. } => Route::Producer,
        PrefixContinuationPhase::CheckpointReady { .. } => Route::Captured,
        PrefixContinuationPhase::Restored => Route::Restored,
    }
}

impl BoundedSloPlanner {
    /// The engine supplies the original cohort identity/expiry with this
    /// snapshot's current owner keys and generation. No continuation renews a
    /// hold, a model TTL, an ingress anchor or an original controller budget.
    pub fn continue_prefix_rendezvous(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        phase: PrefixContinuationPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        clock: &mut dyn PlanningClock,
    ) -> PrefixContinuationDecision {
        self.continue_prefix_rendezvous_scoped(
            snapshot,
            offer,
            phase,
            model,
            maintenance,
            context,
            None,
            clock,
        )
    }

    pub fn continue_prefix_rendezvous_scoped(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        phase: PrefixContinuationPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
    ) -> PrefixContinuationDecision {
        let mut stats = PlanningSearchStats::default();
        match self.continue_prefix(
            snapshot,
            offer,
            phase,
            model,
            maintenance,
            context,
            protection,
            clock,
            &mut stats,
        ) {
            Ok(continuation) => PrefixContinuationDecision::Ready {
                continuation,
                search: stats,
            },
            Err(reason) => PrefixContinuationDecision::Unknown {
                reason,
                search: stats,
            },
        }
    }

    fn continue_prefix(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        phase: PrefixContinuationPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
        stats: &mut PlanningSearchStats,
    ) -> Result<PrefixContinuationEvidence, PlanningUnknownReason> {
        validation::validate(&self.settings, snapshot, true)?;
        validate_current(snapshot, offer, phase)?;
        if model.model_version() != snapshot.cost_model_version {
            return Err(PlanningUnknownReason::ModelVersionMismatch);
        }
        let maintenance_version = maintenance.model_version();
        let started = clock.now_ns();
        if started < snapshot.observed_at_ns {
            return Err(PlanningUnknownReason::ClockMovedBackwards);
        }
        if started >= snapshot.scope.horizon_end_ns || started >= offer.expires_at_ns {
            return Err(PlanningUnknownReason::HorizonInsufficient);
        }
        let window = clock
            .planning_budget_window()
            .unwrap_or(PlanningBudgetWindow {
                started_at_ns: snapshot.observed_at_ns,
                deadline_ns: snapshot
                    .observed_at_ns
                    .checked_add(
                        self.settings
                            .search
                            .max_planning_us
                            .get()
                            .checked_mul(1000)
                            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?,
                    )
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?,
            });
        let mut budget = ComputeBudget::new(
            PlanningPhaseBudget {
                window,
                planner_deadline_ns: clock.planning_phase_deadline_ns(),
            },
            started,
            &self.settings.search,
        )?;
        let protection = prefix_protection(snapshot, started, protection, &mut budget, clock)?;
        // This path cannot remove late owners or turn a recovery-only turn into
        // a new promise. Failure returns Unknown for the engine's hold policy.
        if protection.new_time_promises_closed() {
            return Err(PlanningUnknownReason::RecoveryConflict);
        }
        let session = ExecutionSession::new(context, &self.settings);
        let future_controller_ns = self
            .settings
            .future_controller_time
            .reserved_ns(&self.settings.search)?;
        let saved = find_path(
            self,
            snapshot,
            PathOffer::Rendezvous(offer),
            model,
            maintenance,
            maintenance_version,
            &session,
            &protection,
            route(phase),
            Some(InitialPhase::Rendezvous(phase)),
            started,
            future_controller_ns,
            &mut budget,
            clock,
            stats,
        )?;
        (stats.measured_replay_work_ns, stats.replay_reserve_ns) =
            budget.reserve_prefix_pair(saved.measured, MeasuredReplayWork::default())?;
        stats.phase = PlanningSearchPhase::Finalization;
        let replayed = replay(
            self,
            snapshot,
            PathOffer::Rendezvous(offer),
            model,
            maintenance,
            maintenance_version,
            &session,
            &protection,
            &saved,
            future_controller_ns,
            &mut budget,
            clock,
        )?;
        let final_now = budget.read(clock)?;
        if model.model_version() != snapshot.cost_model_version
            || maintenance.model_version() != maintenance_version
        {
            return Err(PlanningUnknownReason::ModelVersionMismatch);
        }
        if final_now >= snapshot.scope.horizon_end_ns || final_now >= offer.expires_at_ns {
            return Err(PlanningUnknownReason::HorizonInsufficient);
        }
        let delay = final_now
            .checked_sub(replayed.state.started_at_ns)
            .ok_or(PlanningUnknownReason::ClockMovedBackwards)?;
        let slack = replayed
            .state
            .minimum_start_slack_ns
            .min(replayed.state.minimum_cost_freshness_slack_ns)
            .checked_sub(delay)
            .ok_or(PlanningUnknownReason::SearchIncomplete)?
            .min(snapshot.scope.horizon_end_ns - final_now - 1)
            .min(offer.expires_at_ns - final_now - 1);
        let valid_until_ns = final_now
            .checked_add(slack)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        let first_action_cost_ns = replayed.first_action_cost_ns;
        if first_action_cost_ns == 0 {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let action = match replayed
            .steps
            .first()
            .ok_or(PlanningUnknownReason::SearchIncomplete)?
        {
            PrefixPathStep::Wave(candidate) => {
                let bound = replayed
                    .state
                    .first_wave_candidate
                    .clone()
                    .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
                let canonical = replayed
                    .state
                    .first_wave_canonical
                    .clone()
                    .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
                PrefixContinuationAction::Wave(SelectedWave {
                    final_replay_first_wave: Some(Arc::new(FinalReplayFirstWave::from_replay(
                        snapshot,
                        bound,
                        canonical,
                        replayed.state.first_wave_statistics.clone(),
                    ))),
                    protection: Some(protection.clone()),
                    candidate: candidate.clone(),
                    predicted_wall_ns: first_action_cost_ns,
                    planning_observed_at_ns: final_now,
                    snapshot_observed_at_ns: snapshot.observed_at_ns,
                    snapshot_generation: snapshot.generation,
                    cost_model_version: snapshot.cost_model_version,
                    witness_valid_for_ns: slack,
                })
            }
            PrefixPathStep::Maintenance(evidence) => {
                PrefixContinuationAction::Maintenance(evidence.clone())
            }
            PrefixPathStep::ReadyRestore(_) => {
                return Err(PlanningUnknownReason::InvalidShapeEvidence)
            }
        };
        if let (Some(observer), Some(replay)) =
            (model.query_observer(), replayed.state.observed_replay)
        {
            observer.selected_replay(replay);
        }
        let mut remaining_slack = u64::MAX;
        let remaining = finalize_path(
            replayed,
            PathOffer::Rendezvous(offer),
            final_now,
            &mut remaining_slack,
        )?;
        if budget.read(clock)? > valid_until_ns {
            return Err(PlanningUnknownReason::SearchIncomplete);
        }
        Ok(PrefixContinuationEvidence {
            offer: offer.clone(),
            phase,
            maintenance_model_version: maintenance_version,
            validated_at_ns: final_now,
            valid_until_ns,
            first_action_cost_ns,
            protection,
            action,
            remaining,
        })
    }
}

fn validate_current(
    snapshot: &SchedulerSnapshot,
    offer: &PrefixRendezvousOffer,
    phase: PrefixContinuationPhase,
) -> Result<(), PlanningUnknownReason> {
    if offer.based_on_generation != snapshot.generation || offer.producer == offer.target {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    let source = snapshot
        .requests
        .iter()
        .find(|r| r.key == offer.producer)
        .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
    let target = snapshot
        .requests
        .iter()
        .find(|r| r.key == offer.target)
        .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
    let RequestPhaseView::Prefill(destination) = &target.phase else {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    };
    let boundary = offer.boundary_tokens.get();
    let restored = phase == PrefixContinuationPhase::Restored;
    if target.timing.committed_tokens != 0
        || target.timing.first_commit_at_ns.is_some()
        || boundary >= destination.total_prompt_tokens.get()
        || boundary > destination.executable_until
        || destination.reference.work_at(boundary).is_none()
        || if restored {
            target.readiness != RequestReadiness::Ready
                || destination.offset != boundary
                || target.context_tokens != boundary
                || destination.logical_high_water < boundary
        } else {
            target.readiness != RequestReadiness::StateBlocked
                || destination.offset != 0
                || destination.logical_high_water != 0
                || target.context_tokens != 0
        }
    {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    match phase {
        PrefixContinuationPhase::HeldAwaitingProducer
        | PrefixContinuationPhase::AtCaptureBoundary { .. } => {
            let RequestPhaseView::Prefill(progress) = &source.phase else {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            };
            if source.readiness != RequestReadiness::Ready
                || source.timing.committed_tokens != 0
                || boundary >= progress.total_prompt_tokens.get()
                || boundary > progress.executable_until
                || progress.reference.work_at(boundary).is_none()
                || match phase {
                    PrefixContinuationPhase::HeldAwaitingProducer => progress.offset >= boundary,
                    PrefixContinuationPhase::AtCaptureBoundary { capture_span_start } => {
                        progress.offset != boundary || capture_span_start >= boundary
                    }
                    _ => unreachable!(),
                }
            {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            }
        }
        PrefixContinuationPhase::CheckpointReady { capture_span_start }
            if capture_span_start >= boundary =>
        {
            return Err(PlanningUnknownReason::InvalidSnapshot)
        }
        _ => {}
    }
    Ok(())
}
