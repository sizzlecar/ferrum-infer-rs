//! Source-only cache publication, with the existing complete queue witness.
use super::*;

impl BoundedSloPlanner {
    pub fn plan_prefix_cache_capture(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixCacheCaptureOffer,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        clock: &mut dyn PlanningClock,
    ) -> PrefixCacheCaptureDecision {
        self.plan_prefix_cache_capture_in_phase(
            snapshot,
            offer,
            PrefixCacheCapturePhase::AtBoundary,
            model,
            maintenance,
            context,
            clock,
        )
    }

    pub fn plan_prefix_cache_capture_in_phase(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixCacheCaptureOffer,
        phase: PrefixCacheCapturePhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        clock: &mut dyn PlanningClock,
    ) -> PrefixCacheCaptureDecision {
        self.plan_prefix_cache_capture_in_phase_scoped(
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

    pub fn plan_prefix_cache_capture_in_phase_scoped(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixCacheCaptureOffer,
        phase: PrefixCacheCapturePhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
    ) -> PrefixCacheCaptureDecision {
        let mut search = PlanningSearchStats::default();
        match self.plan_cache_capture(
            snapshot,
            offer,
            phase,
            model,
            maintenance,
            context,
            protection,
            clock,
            &mut search,
        ) {
            Ok(evidence) => PrefixCacheCaptureDecision::Known { evidence, search },
            Err(reason) => PrefixCacheCaptureDecision::Unknown { reason, search },
        }
    }

    fn plan_cache_capture(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixCacheCaptureOffer,
        phase: PrefixCacheCapturePhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
        stats: &mut PlanningSearchStats,
    ) -> Result<PrefixCacheCaptureEvidence, PlanningUnknownReason> {
        validation::validate(&self.settings, snapshot, true)?;
        validate_capture(snapshot, offer, phase)?;
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
        if protection.new_time_promises_closed() || protection.required_first_service().is_some() {
            return Err(PlanningUnknownReason::RecoveryConflict);
        }
        let session = ExecutionSession::new(context, &self.settings);
        let future_controller_ns = self
            .settings
            .future_controller_time
            .reserved_ns(&self.settings.search)?;
        let path_offer = PathOffer::CacheCapture(offer);
        let saved = find_path(
            self,
            snapshot,
            path_offer,
            model,
            maintenance,
            maintenance_version,
            &session,
            &protection,
            match phase {
                PrefixCacheCapturePhase::AtBoundary => Route::CapturePending,
                PrefixCacheCapturePhase::Preparing => Route::CapturePreparing,
            },
            Some(InitialPhase::CacheCapture(phase)),
            started,
            future_controller_ns,
            &mut budget,
            clock,
            stats,
        )?;
        // One complete trajectory is replayed here. No hypothetical follower
        // or second path is created to manufacture a cache-hit benefit.
        (stats.measured_replay_work_ns, stats.replay_reserve_ns) =
            budget.reserve_prefix_pair(saved.measured, MeasuredReplayWork::default())?;
        stats.phase = PlanningSearchPhase::Finalization;
        let replayed = replay(
            self,
            snapshot,
            path_offer,
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
        let capture = replayed
            .steps
            .iter()
            .find_map(|step| match step {
                PrefixPathStep::Maintenance(e) if e.stage == PrefixMaintenanceStage::Capture => {
                    Some(e)
                }
                _ => None,
            })
            .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
        if first_action_cost_ns == 0
            || capture.stage != PrefixMaintenanceStage::Capture
            || capture.restored_frontier.is_some()
            || (phase == PrefixCacheCapturePhase::AtBoundary
                && capture.capture_span_start != offer.capture_span_start)
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let action = match replayed.steps.first() {
            Some(PrefixPathStep::Maintenance(first))
                if phase == PrefixCacheCapturePhase::AtBoundary && first == capture =>
            {
                PrefixCacheCaptureAction::Capture(first.clone())
            }
            Some(PrefixPathStep::Wave(candidate))
                if phase == PrefixCacheCapturePhase::Preparing =>
            {
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
                PrefixCacheCaptureAction::Wave(SelectedWave {
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
            _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
        };
        let completion_at_ns = replayed
            .state
            .now_ns
            .checked_add(delay)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        if budget.read(clock)? >= valid_until_ns {
            return Err(PlanningUnknownReason::SearchIncomplete);
        }
        if let (Some(observer), Some(replay)) =
            (model.query_observer(), replayed.state.observed_replay)
        {
            observer.selected_replay(replay);
        }
        if budget.read(clock)? >= valid_until_ns {
            return Err(PlanningUnknownReason::SearchIncomplete);
        }
        Ok(PrefixCacheCaptureEvidence {
            offer: offer.clone(),
            phase,
            action,
            inference_model_version: snapshot.cost_model_version,
            maintenance_model_version: maintenance_version,
            validated_at_ns: final_now,
            valid_until_ns,
            first_action_cost_ns,
            protection,
            capture: capture.clone(),
            steps: replayed.steps,
            completion_at_ns,
        })
    }
}

fn validate_capture(
    snapshot: &SchedulerSnapshot,
    offer: &PrefixCacheCaptureOffer,
    phase: PrefixCacheCapturePhase,
) -> Result<(), PlanningUnknownReason> {
    if offer.based_on_generation != snapshot.generation {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    let source = snapshot
        .requests
        .iter()
        .find(|r| r.key == offer.source)
        .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
    let RequestPhaseView::Prefill(progress) = &source.phase else {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    };
    let boundary = offer.boundary_tokens.get();
    if source.readiness != RequestReadiness::Ready
        || source.timing.committed_tokens != 0
        || source.timing.first_commit_at_ns.is_some()
        || offer.capture_span_start >= boundary
        || source.context_tokens != progress.offset
        || progress.logical_high_water < progress.offset
        || match phase {
            PrefixCacheCapturePhase::AtBoundary => progress.offset != boundary,
            PrefixCacheCapturePhase::Preparing => {
                progress.offset >= boundary || offer.capture_span_start != progress.offset
            }
        }
        || boundary >= progress.total_prompt_tokens.get()
        || boundary > progress.executable_until
        || progress.reference.work_at(boundary).is_none()
        || offer.expires_at_ns > snapshot.scope.horizon_end_ns
    {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    Ok(())
}
