//! Immediate ready-cache restoration uses the existing full-queue search and replay.
use super::*;

enum ReadyResult {
    Use(ReadyPrefixEvidence),
    Direct {
        direct: PrefixPathEvidence,
        restoring: PrefixPathEvidence,
    },
}

impl BoundedSloPlanner {
    /// Compare an immediate ready hit with direct prefill, or reprove the first
    /// model wave after its acknowledged restore. Neither path creates a hold
    /// or renews the original model, ingress or controller lifetime.
    pub fn plan_ready_prefix_restore(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &ReadyPrefixRestoreOffer,
        phase: ReadyPrefixPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        clock: &mut dyn PlanningClock,
    ) -> ReadyPrefixDecision {
        self.plan_ready_prefix_restore_scoped(
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

    pub fn plan_ready_prefix_restore_scoped(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &ReadyPrefixRestoreOffer,
        phase: ReadyPrefixPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
    ) -> ReadyPrefixDecision {
        let mut stats = PlanningSearchStats::default();
        match self.plan_ready_prefix(
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
            Ok(ReadyResult::Use(evidence)) => ReadyPrefixDecision::Ready {
                evidence,
                search: stats,
            },
            Ok(ReadyResult::Direct { direct, restoring }) => ReadyPrefixDecision::PreferDirect {
                direct,
                restoring,
                search: stats,
            },
            Err(reason) => ReadyPrefixDecision::Unknown {
                reason,
                search: stats,
            },
        }
    }

    fn plan_ready_prefix(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &ReadyPrefixRestoreOffer,
        phase: ReadyPrefixPhase,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
        stats: &mut PlanningSearchStats,
    ) -> Result<ReadyResult, PlanningUnknownReason> {
        validation::validate(&self.settings, snapshot, true)?;
        validate_ready(snapshot, offer, phase)?;
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
        // a new promise. Failure leaves ordinary scheduling on the same budget.
        if protection.new_time_promises_closed() {
            return Err(PlanningUnknownReason::RecoveryConflict);
        }
        let session = ExecutionSession::new(context, &self.settings);
        let future_controller_ns = self
            .settings
            .future_controller_time
            .reserved_ns(&self.settings.search)?;
        let direct = if phase == ReadyPrefixPhase::Ready {
            Some(find_path(
                self,
                snapshot,
                PathOffer::Ready(offer),
                model,
                maintenance,
                maintenance_version,
                &session,
                &protection,
                Route::Direct,
                None,
                started,
                future_controller_ns,
                &mut budget,
                clock,
                stats,
            )?)
        } else {
            None
        };
        let saved = find_path(
            self,
            snapshot,
            PathOffer::Ready(offer),
            model,
            maintenance,
            maintenance_version,
            &session,
            &protection,
            if phase == ReadyPrefixPhase::Ready {
                Route::Captured
            } else {
                Route::Restored
            },
            Some(InitialPhase::Ready(phase)),
            started,
            future_controller_ns,
            &mut budget,
            clock,
            stats,
        )?;
        (stats.measured_replay_work_ns, stats.replay_reserve_ns) = budget.reserve_prefix_pair(
            saved.measured,
            direct
                .as_ref()
                .map_or(MeasuredReplayWork::default(), |path| path.measured),
        )?;
        stats.phase = PlanningSearchPhase::Finalization;
        let direct = direct
            .as_ref()
            .map(|path| {
                replay(
                    self,
                    snapshot,
                    PathOffer::Ready(offer),
                    model,
                    maintenance,
                    maintenance_version,
                    &session,
                    &protection,
                    path,
                    future_controller_ns,
                    &mut budget,
                    clock,
                )
            })
            .transpose()?;
        let replayed = replay(
            self,
            snapshot,
            PathOffer::Ready(offer),
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
        let mut slack = replayed
            .state
            .minimum_start_slack_ns
            .min(replayed.state.minimum_cost_freshness_slack_ns)
            .checked_sub(delay)
            .ok_or(PlanningUnknownReason::SearchIncomplete)?
            .min(snapshot.scope.horizon_end_ns - final_now - 1)
            .min(offer.expires_at_ns - final_now - 1);
        let direct = direct
            .map(|path| finalize_path(path, PathOffer::Ready(offer), final_now, &mut slack))
            .transpose()?;
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
                ReadyPrefixAction::Wave(SelectedWave {
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
            PrefixPathStep::ReadyRestore(evidence) => ReadyPrefixAction::Restore(evidence.clone()),
            PrefixPathStep::Maintenance(_) => {
                return Err(PlanningUnknownReason::InvalidShapeEvidence)
            }
        };
        let observed_replay = replayed.state.observed_replay;
        let mut remaining_slack = u64::MAX;
        let remaining = finalize_path(
            replayed,
            PathOffer::Ready(offer),
            final_now,
            &mut remaining_slack,
        )?;
        if budget.read(clock)? > valid_until_ns {
            return Err(PlanningUnknownReason::SearchIncomplete);
        }
        if let Some(direct) = &direct {
            if remaining.first_commit_at_ns >= direct.first_commit_at_ns {
                return Ok(ReadyResult::Direct {
                    direct: direct.clone(),
                    restoring: remaining,
                });
            }
        }
        if let (Some(observer), Some(replay)) = (model.query_observer(), observed_replay) {
            observer.selected_replay(replay);
        }
        if budget.read(clock)? >= valid_until_ns {
            return Err(PlanningUnknownReason::SearchIncomplete);
        }
        Ok(ReadyResult::Use(ReadyPrefixEvidence {
            offer: offer.clone(),
            phase,
            maintenance_model_version: maintenance_version,
            validated_at_ns: final_now,
            valid_until_ns,
            first_action_cost_ns,
            protection,
            action,
            remaining,
            direct,
        }))
    }
}

fn validate_ready(
    snapshot: &SchedulerSnapshot,
    offer: &ReadyPrefixRestoreOffer,
    phase: ReadyPrefixPhase,
) -> Result<(), PlanningUnknownReason> {
    if offer.based_on_generation != snapshot.generation {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    let target = snapshot
        .requests
        .iter()
        .find(|r| r.key == offer.target)
        .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
    let RequestPhaseView::Prefill(progress) = &target.phase else {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    };
    let boundary = offer.boundary_tokens.get();
    if target.readiness != RequestReadiness::Ready
        || target.timing.committed_tokens != 0
        || target.timing.first_commit_at_ns.is_some()
        || boundary >= progress.total_prompt_tokens.get()
        || boundary > progress.executable_until
        || progress.reference.work_at(boundary).is_none()
        || match phase {
            ReadyPrefixPhase::Ready => {
                progress.offset != 0
                    || progress.logical_high_water != 0
                    || target.context_tokens != 0
            }
            ReadyPrefixPhase::Restored => {
                progress.offset != boundary
                    || progress.logical_high_water < boundary
                    || target.context_tokens != boundary
            }
        }
    {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    Ok(())
}
