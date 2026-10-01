use super::super::{
    candidates::FrontierCursor,
    execution::{ExecutionSession, PlanningExecutionContext},
    obligations::PlanningObligationSet,
    search::{BoundedSloPlanner, ComputeBudget, MeasuredReplayWork},
    simulation::{self, PlanningState, SimulationFailure},
    validation,
};
use super::*;
use std::num::NonZeroUsize;

mod cache_capture;
mod continuation;
mod ready;

#[derive(Clone, Copy)]
enum InitialPhase {
    Rendezvous(PrefixContinuationPhase),
    Ready(ReadyPrefixPhase),
    CacheCapture(PrefixCacheCapturePhase),
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(any(test, feature = "planning-diagnostics"), derive(Debug))]
enum Route {
    Direct,
    Producer,
    Captured,
    Restored,
    CapturePreparing,
    CapturePending,
    CaptureDone,
}

struct Path<'epoch> {
    state: PlanningState<'epoch>,
    steps: Vec<PrefixPathStep>,
    route: Route,
    measured: MeasuredReplayWork,
    capture_ready_at_ns: Option<u64>,
    restore_ready_at_ns: Option<u64>,
    initial_phase: Option<InitialPhase>,
    first_action_cost_ns: u64,
}

struct Frame<'epoch> {
    path: Path<'epoch>,
    cursor: Option<FrontierCursor>,
    resolved: usize,
}

impl BoundedSloPlanner {
    /// Compare direct prefill and producer/capture/restore/suffix trajectories.
    /// Both consume one original clock/action budget, retain every queue owner,
    /// and are independently replayed before returning numeric hold evidence.
    pub fn compare_prefix_rendezvous(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        clock: &mut dyn PlanningClock,
    ) -> PrefixRendezvousDecision {
        self.compare_prefix_rendezvous_scoped(
            snapshot,
            offer,
            model,
            maintenance,
            context,
            None,
            clock,
        )
    }

    pub fn compare_prefix_rendezvous_scoped(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
    ) -> PrefixRendezvousDecision {
        let mut stats = PlanningSearchStats::default();
        match self.compare_prefix(
            snapshot,
            offer,
            model,
            maintenance,
            context,
            protection,
            clock,
            &mut stats,
        ) {
            Ok(comparison) => PrefixRendezvousDecision::Compared {
                comparison,
                search: stats,
            },
            Err(reason) => PrefixRendezvousDecision::Unknown {
                reason,
                search: stats,
            },
        }
    }

    fn compare_prefix(
        &self,
        snapshot: &SchedulerSnapshot,
        offer: &PrefixRendezvousOffer,
        model: &dyn PlanningCostModel,
        maintenance: &dyn PlanningPrefixCostModel,
        context: &dyn PlanningExecutionContext,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
        stats: &mut PlanningSearchStats,
    ) -> Result<PrefixRendezvousComparison, PlanningUnknownReason> {
        validation::validate(&self.settings, snapshot, true)?;
        validate_offer(snapshot, offer)?;
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
        if protection.new_time_promises_closed() {
            return Err(PlanningUnknownReason::RecoveryConflict);
        }
        let session = ExecutionSession::new(context, &self.settings);
        let future_controller_ns = self
            .settings
            .future_controller_time
            .reserved_ns(&self.settings.search)?;
        // One cumulative candidate/raw/session ceiling; the second path never
        // reopens either the compute deadline or any consumed action allowance.
        let direct = find_path(
            self,
            snapshot,
            PathOffer::Rendezvous(offer),
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
        )?;
        let waiting = find_path(
            self,
            snapshot,
            PathOffer::Rendezvous(offer),
            model,
            maintenance,
            maintenance_version,
            &session,
            &protection,
            Route::Producer,
            None,
            started,
            future_controller_ns,
            &mut budget,
            clock,
            stats,
        )?;
        (stats.measured_replay_work_ns, stats.replay_reserve_ns) =
            budget.reserve_prefix_pair(direct.measured, waiting.measured)?;
        stats.phase = PlanningSearchPhase::Finalization;
        let direct = replay(
            self,
            snapshot,
            PathOffer::Rendezvous(offer),
            model,
            maintenance,
            maintenance_version,
            &session,
            &protection,
            &direct,
            future_controller_ns,
            &mut budget,
            clock,
        )?;
        let waiting = replay(
            self,
            snapshot,
            PathOffer::Rendezvous(offer),
            model,
            maintenance,
            maintenance_version,
            &session,
            &protection,
            &waiting,
            future_controller_ns,
            &mut budget,
            clock,
        )?;
        // Place BOTH independently replayed trajectories on one final clock.
        // All later CPU work is charged to both their deadline and TTL slack.
        let final_now = budget.read(clock)?;
        if model.model_version() != snapshot.cost_model_version
            || maintenance.model_version() != maintenance_version
        {
            return Err(PlanningUnknownReason::ModelVersionMismatch);
        }
        if final_now >= snapshot.scope.horizon_end_ns {
            return Err(PlanningUnknownReason::HorizonInsufficient);
        }
        let mut remaining_slack = u64::MAX;
        let direct_evidence = finalize_path(
            direct,
            PathOffer::Rendezvous(offer),
            final_now,
            &mut remaining_slack,
        )?;
        let waiting_evidence = finalize_path(
            waiting,
            PathOffer::Rendezvous(offer),
            final_now,
            &mut remaining_slack,
        )?;
        let target = snapshot
            .requests
            .iter()
            .find(|r| r.key == offer.target)
            .unwrap();
        let valid_until_ns = final_now
            .checked_add(remaining_slack)
            .unwrap_or(u64::MAX)
            .min(offer.expires_at_ns)
            .min(
                target
                    .timing
                    .next_deadline_ns()
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?,
            );
        // This read covers evidence construction, not only provider callbacks.
        let checked_now = budget.read(clock)?;
        if checked_now > final_now {
            // Output evidence remains on final_now; its explicit expiry still
            // permits the integrating controller to perform its last checks.
            if checked_now >= valid_until_ns {
                #[cfg(any(test, feature = "planning-diagnostics"))]
                eprintln!("prefix original comparison evidence expired: final_now={final_now} checked_now={checked_now} valid_until={valid_until_ns} remaining_slack={remaining_slack}");
                return Err(PlanningUnknownReason::SearchIncomplete);
            }
        }
        Ok(PrefixRendezvousComparison {
            offer: offer.clone(),
            inference_model_version: snapshot.cost_model_version,
            maintenance_model_version: maintenance_version,
            validated_at_ns: final_now,
            valid_until_ns,
            direct: direct_evidence,
            waiting: waiting_evidence,
        })
    }
}

/// The controller classifies once. Reusing that exact scope preserves protected
/// owners even if search/replay happens later; it cannot refresh classification.
fn prefix_protection(
    snapshot: &SchedulerSnapshot,
    started: u64,
    supplied: Option<Arc<PlanningObligationSet>>,
    budget: &mut ComputeBudget,
    clock: &mut dyn PlanningClock,
) -> Result<Arc<PlanningObligationSet>, PlanningUnknownReason> {
    match supplied {
        Some(scope) => {
            budget.read(clock)?;
            if !scope.matches(snapshot) || scope.classified_at_ns() > started {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            }
            budget.read(clock)?;
            Ok(scope)
        }
        None => PlanningObligationSet::capture_with_budget(snapshot, started, &mut || {
            budget.read(clock).map(|_| ())
        })
        .map(Arc::new),
    }
}

fn validate_offer(
    snapshot: &SchedulerSnapshot,
    offer: &PrefixRendezvousOffer,
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
    let (RequestPhaseView::Prefill(source_progress), RequestPhaseView::Prefill(target_progress)) =
        (&source.phase, &target.phase)
    else {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    };
    let boundary = offer.boundary_tokens.get();
    if source.readiness != RequestReadiness::Ready
        || target.readiness != RequestReadiness::Ready
        || source.timing.committed_tokens != 0
        || target.timing.committed_tokens != 0
        || source_progress.offset >= boundary
        || boundary >= source_progress.total_prompt_tokens.get()
        || boundary > source_progress.executable_until
        || target_progress.offset != 0
        || target_progress.logical_high_water != 0
        || target.context_tokens != 0
        || boundary >= target_progress.total_prompt_tokens.get()
        || boundary > target_progress.executable_until
        || source_progress.reference.work_at(boundary).is_none()
        || target_progress.reference.work_at(boundary).is_none()
    {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    Ok(())
}

fn maintenance_stage(path: &Path<'_>, offer: PathOffer<'_>) -> Option<PrefixMaintenanceStage> {
    match path.route {
        Route::CapturePending => Some(PrefixMaintenanceStage::Capture),
        Route::CapturePreparing
            if matches!(offer, PathOffer::CacheCapture(o) if path.state.requests.iter().any(|r| r.key == o.source
            && matches!(&r.phase, RequestPhaseView::Prefill(p) if p.offset == offer.boundary()))) =>
        {
            Some(PrefixMaintenanceStage::Capture)
        }
        Route::Producer
            if matches!(offer, PathOffer::Rendezvous(o) if path.state.requests.iter().any(|r| r.key == o.producer
            && matches!(&r.phase, RequestPhaseView::Prefill(p) if p.offset == offer.boundary()))) =>
        {
            Some(PrefixMaintenanceStage::Capture)
        }
        Route::Captured => Some(PrefixMaintenanceStage::Restore),
        _ => None,
    }
}

fn capture_span_start(
    path: &Path<'_>,
    offer: PathOffer<'_>,
) -> Result<Option<u32>, PlanningUnknownReason> {
    let source = match offer {
        PathOffer::CacheCapture(o) => {
            if matches!(
                path.initial_phase,
                Some(InitialPhase::CacheCapture(
                    PrefixCacheCapturePhase::AtBoundary
                ))
            ) {
                return Ok(Some(o.capture_span_start));
            }
            &o.source
        }
        PathOffer::Rendezvous(o) => &o.producer,
        PathOffer::Ready(_) => return Ok(None),
    };
    path.steps
        .iter()
        .rev()
        .find_map(|step| match step {
            PrefixPathStep::Wave(wave) => wave.work.iter().find_map(|work| {
                if &work.key != source {
                    return None;
                }
                match work.action {
                    WaveAction::Prefill { offset, count }
                        if offset.checked_add(count.get()) == Some(offer.boundary()) =>
                    {
                        Some(offset)
                    }
                    _ => None,
                }
            }),
            PrefixPathStep::Maintenance(evidence) => Some(evidence.capture_span_start),
            PrefixPathStep::ReadyRestore(_) => None,
        })
        .or_else(|| match path.initial_phase {
            Some(InitialPhase::Rendezvous(
                PrefixContinuationPhase::AtCaptureBoundary { capture_span_start }
                | PrefixContinuationPhase::CheckpointReady { capture_span_start },
            )) => Some(capture_span_start),
            _ => None,
        })
        .map(Some)
        .ok_or(PlanningUnknownReason::InvalidShapeEvidence)
}

fn complete(
    path: &Path<'_>,
    snapshot: &SchedulerSnapshot,
    offer: PathOffer<'_>,
    milestones: bool,
    protection: &PlanningObligationSet,
) -> Result<bool, PlanningUnknownReason> {
    if matches!(offer, PathOffer::CacheCapture(_)) {
        return if path.route == Route::CaptureDone {
            simulation::ready_witness(snapshot, &path.state, milestones, Some(protection))
        } else {
            Ok(false)
        };
    }
    let target = offer
        .target()
        .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
    if !matches!(path.route, Route::Direct | Route::Restored)
        || !path
            .state
            .requests
            .iter()
            .any(|r| r.key == *target && r.timing.first_commit_at_ns.is_some())
    {
        return Ok(false);
    }
    simulation::ready_witness(snapshot, &path.state, milestones, Some(protection))
}

fn find_path<'epoch>(
    planner: &BoundedSloPlanner,
    snapshot: &'epoch SchedulerSnapshot,
    offer: PathOffer<'_>,
    model: &dyn PlanningCostModel,
    maintenance: &dyn PlanningPrefixCostModel,
    maintenance_version: u64,
    context: &'epoch dyn PlanningExecutionContext,
    protection: &PlanningObligationSet,
    route: Route,
    initial_phase: Option<InitialPhase>,
    started: u64,
    future_controller_ns: u64,
    budget: &mut ComputeBudget,
    clock: &mut dyn PlanningClock,
    stats: &mut PlanningSearchStats,
) -> Result<Path<'epoch>, PlanningUnknownReason> {
    let begin = budget.read(clock)?;
    let mut state = simulation::begin_with_controller_time(
        snapshot,
        context,
        &mut || budget.read(clock).map(|_| ()),
        started,
        future_controller_ns,
    )
    .map_err(reason)?;
    prepare_initial(
        &mut state,
        snapshot,
        offer,
        route,
        initial_phase,
        &mut || budget.read(clock).map(|_| ()),
    )?;
    let measured = MeasuredReplayWork::default().with_span(begin, budget.read(clock)?)?;
    let horizon = planner.settings.search.lookahead_waves.get();
    let width = planner.settings.search.beam_width.get();
    let limit = planner.settings.search.candidate_limit.get();
    let expansions = limit * width * horizon;
    let raw_limit = expansions * 8;
    // H bounds inference waves. Exactly two typed checkpoint phases may be
    // interleaved; they still spend the unchanged candidate/session/clock caps.
    let step_horizon = horizon
        + if matches!(offer, PathOffer::CacheCapture(_)) {
            1
        } else {
            2
        };
    let mut pending: Vec<Vec<Frame<'epoch>>> = (0..step_horizon).map(|_| Vec::new()).collect();
    pending[0].push(Frame {
        path: Path {
            state,
            steps: Vec::new(),
            route,
            measured,
            capture_ready_at_ns: None,
            restore_ready_at_ns: None,
            initial_phase,
            first_action_cost_ns: 0,
        },
        cursor: None,
        resolved: 0,
    });
    let mut preferred = Some(0);
    #[cfg(any(test, feature = "planning-diagnostics"))]
    let mut diagnostic_transitions = 0usize;
    while let Some(depth) = preferred
        .take()
        .filter(|d| !pending[*d].is_empty())
        .or_else(|| pending.iter().position(|rows| !rows.is_empty()))
    {
        budget.read(clock)?;
        if stats.expanded_candidates >= expansions || stats.enumeration_attempts >= raw_limit {
            stats.candidate_truncations += 1;
            break;
        }
        let mut frame = pending[depth].remove(0);
        let model_depth = frame
            .path
            .steps
            .iter()
            .filter(|step| matches!(step, PrefixPathStep::Wave(_)))
            .count();
        let stage = maintenance_stage(&frame.path, offer);
        if stage.is_none() && model_depth >= horizon {
            continue;
        }
        stats.max_depth_reached = stats
            .max_depth_reached
            .max(model_depth + usize::from(stage.is_none()));
        let work = if stage.is_none() {
            if frame.resolved >= limit {
                stats.candidate_truncations += 1;
                continue;
            }
            if frame.cursor.is_none() {
                let goal = match frame.path.route {
                    Route::Producer => match offer {
                        PathOffer::Rendezvous(o) => Some(&o.producer),
                        _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
                    },
                    Route::CapturePreparing => match offer {
                        PathOffer::CacheCapture(o) => Some(&o.source),
                        _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
                    },
                    Route::Direct | Route::Restored => offer.target(),
                    Route::Captured | Route::CapturePending | Route::CaptureDone => None,
                };
                frame.cursor = Some(FrontierCursor::with_prefill_goal(
                    snapshot,
                    &frame.path.state.requests,
                    frame.path.state.now_ns,
                    limit,
                    Some(protection),
                    NonZeroUsize::new(horizon - model_depth),
                    goal,
                    &mut || budget.read(clock).map(|_| ()),
                )?);
            }
            match frame.cursor.as_mut().unwrap().next(
                snapshot,
                &frame.path.state.requests,
                &mut stats.enumeration_attempts,
                raw_limit,
                &mut || budget.read(clock).map(|_| ()),
            )? {
                Some(work) => Some(work),
                None => continue,
            }
        } else {
            stats.enumeration_attempts += 1;
            None
        };
        let edge_started = budget.read(clock)?;
        let transition = if let Some(stage) = stage {
            simulation::advance_prefix(
                snapshot,
                &frame.path.state,
                offer,
                stage,
                capture_span_start(&frame.path, offer)?,
                maintenance,
                maintenance_version,
                &mut || budget.read(clock).map(|_| ()),
                planner.settings.search.enable_prefill_milestones,
                protection,
            )
        } else {
            simulation::advance(
                snapshot,
                &frame.path.state,
                work.as_ref().unwrap(),
                model,
                true,
                &mut || budget.read(clock).map(|_| ()),
                planner.settings.search.enable_prefill_milestones,
                Some(protection),
            )
            .map(|transition| (PrefixPathStep::Wave(transition.wave), transition.state))
            .map_err(|failure| failure.cause)
        };
        // Failed projections also consume this route's candidate allowance;
        // Unknown must not create an unbounded replacement loop.
        frame.resolved += 1;
        stats.expanded_candidates += 1;
        stats.generated_candidates += 1;
        let (step, state) = match transition {
            Ok(value) => value,
            Err(error) => {
                let skip = match error {
                    SimulationFailure::SequenceViolation => true,
                    SimulationFailure::Unknown(PlanningUnknownReason::CostUnavailable) => {
                        stats.cost_unknown_candidates += 1;
                        true
                    }
                    SimulationFailure::Unknown(PlanningUnknownReason::ShapeUnavailable) => {
                        stats.shape_unknown_candidates += 1;
                        true
                    }
                    SimulationFailure::Unknown(PlanningUnknownReason::UnknownResourceEvidence) => {
                        stats.resource_unknown_candidates += 1;
                        true
                    }
                    SimulationFailure::Unknown(PlanningUnknownReason::OutputOrResourceBlocked) => {
                        true
                    }
                    _ => false,
                };
                if !skip {
                    return Err(reason(error));
                }
                if stage.is_none() {
                    pending[depth].insert(0, frame);
                }
                continue;
            }
        };
        let mut steps = frame.path.steps.clone();
        steps.push(step);
        let measured = frame
            .path
            .measured
            .with_span(edge_started, budget.read(clock)?)?;
        let next_route = match stage {
            Some(PrefixMaintenanceStage::Capture)
                if matches!(offer, PathOffer::CacheCapture(_)) =>
            {
                Route::CaptureDone
            }
            Some(PrefixMaintenanceStage::Capture) => Route::Captured,
            Some(PrefixMaintenanceStage::Restore) => Route::Restored,
            None => frame.path.route,
        };
        let next = Path {
            initial_phase: frame.path.initial_phase,
            first_action_cost_ns: if frame.path.steps.is_empty() {
                state
                    .now_ns
                    .checked_sub(frame.path.state.now_ns)
                    .ok_or(PlanningUnknownReason::ClockMovedBackwards)?
            } else {
                frame.path.first_action_cost_ns
            },
            capture_ready_at_ns: if stage == Some(PrefixMaintenanceStage::Capture) {
                Some(state.now_ns)
            } else {
                frame.path.capture_ready_at_ns
            },
            restore_ready_at_ns: if stage == Some(PrefixMaintenanceStage::Restore) {
                Some(state.now_ns)
            } else {
                frame.path.restore_ready_at_ns
            },
            state,
            steps,
            route: next_route,
            measured,
        };
        let is_complete = complete(
            &next,
            snapshot,
            offer,
            planner.settings.search.enable_prefill_milestones,
            protection,
        )?;
        #[cfg(any(test, feature = "planning-diagnostics"))]
        if diagnostic_transitions < 48 {
            diagnostic_transitions += 1;
            let work: Vec<_> = work
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(|row| {
                    let span = match row.action {
                        WaveAction::Prefill { offset, count } => Some((offset, count.get())),
                        WaveAction::Decode => None,
                    };
                    (row.key.incarnation, span)
                })
                .collect();
            let rows: Vec<_> = next
                .state
                .requests
                .iter()
                .enumerate()
                .map(|(index, row)| {
                    let prefill = match &row.phase {
                        RequestPhaseView::Prefill(p) => {
                            Some((p.offset, p.executable_until, p.total_prompt_tokens.get()))
                        }
                        RequestPhaseView::Decode => None,
                    };
                    (
                        row.key.incarnation,
                        row.readiness,
                        prefill,
                        row.timing.committed_tokens,
                        row.timing.next_deadline_ns(),
                        protection.protects(index),
                    )
                })
                .collect();
            eprintln!("prefix original transition: route={:?}->{:?} model_depth={} stage={stage:?} complete={is_complete} horizon={} now={} work={work:?} rows={rows:?}",
                route, next.route, model_depth + usize::from(stage.is_none()),
                snapshot.scope.horizon_end_ns, next.state.now_ns);
        }
        if is_complete {
            return Ok(next);
        }
        if stage.is_none() {
            pending[depth].insert(0, frame);
        }
        if depth + 1 < step_horizon {
            let level = &mut pending[depth + 1];
            level.insert(
                0,
                Frame {
                    path: next,
                    cursor: None,
                    resolved: 0,
                },
            );
            let previous = level.len();
            level.truncate(width);
            stats.beam_pruned_nodes += previous - level.len();
            preferred = Some(depth + 1);
        }
    }
    budget.read(clock)?;
    Err(PlanningUnknownReason::SearchIncomplete)
}

fn replay<'epoch>(
    planner: &BoundedSloPlanner,
    snapshot: &'epoch SchedulerSnapshot,
    offer: PathOffer<'_>,
    model: &dyn PlanningCostModel,
    maintenance: &dyn PlanningPrefixCostModel,
    maintenance_version: u64,
    context: &'epoch dyn PlanningExecutionContext,
    protection: &PlanningObligationSet,
    saved: &Path<'_>,
    future_controller_ns: u64,
    budget: &mut ComputeBudget,
    clock: &mut dyn PlanningClock,
) -> Result<Path<'epoch>, PlanningUnknownReason> {
    let observer = model.query_observer();
    let replay_id = observer.map(|observer| {
        observer.begin_replay(
            saved
                .steps
                .iter()
                .filter(|step| matches!(step, PrefixPathStep::Wave(_)))
                .count(),
        )
    });
    let replay_result = (|| {
        let started = budget.read(clock)?;
        let mut state = simulation::begin_with_controller_time(
            snapshot,
            context,
            &mut || budget.read(clock).map(|_| ()),
            started,
            future_controller_ns,
        )
        .map_err(reason)?;
        state.observed_replay = replay_id;
        let route = match saved.initial_phase {
            Some(InitialPhase::Rendezvous(phase)) => continuation::route(phase),
            Some(InitialPhase::CacheCapture(PrefixCacheCapturePhase::AtBoundary)) => {
                Route::CapturePending
            }
            Some(InitialPhase::CacheCapture(PrefixCacheCapturePhase::Preparing)) => {
                Route::CapturePreparing
            }
            Some(InitialPhase::Ready(ReadyPrefixPhase::Ready)) => Route::Captured,
            Some(InitialPhase::Ready(ReadyPrefixPhase::Restored)) => Route::Restored,
            None if saved.route == Route::Direct => Route::Direct,
            None => Route::Producer,
        };
        prepare_initial(
            &mut state,
            snapshot,
            offer,
            route,
            saved.initial_phase,
            &mut || budget.read(clock).map(|_| ()),
        )?;
        let mut result = Path {
            state,
            steps: Vec::new(),
            route,
            measured: MeasuredReplayWork::default(),
            capture_ready_at_ns: None,
            restore_ready_at_ns: None,
            initial_phase: saved.initial_phase,
            first_action_cost_ns: 0,
        };
        for expected in &saved.steps {
            let stage = maintenance_stage(&result, offer);
            let (step, mut state) = match expected {
                PrefixPathStep::Wave(wave) if stage.is_none() => {
                    let transition = simulation::advance_observed(
                        snapshot,
                        &result.state,
                        &wave.work,
                        model,
                        true,
                        &mut || budget.read(clock).map(|_| ()),
                        planner.settings.search.enable_prefill_milestones,
                        Some(protection),
                        super::super::observation::PlanningQueryPhase::IndependentReplay {
                            replay: replay_id.unwrap_or(0),
                        },
                    )
                    .map_err(|failure| {
                        #[cfg(any(test, feature = "planning-diagnostics"))]
                        eprintln!("prefix original replay rejected: route={:?} edge={} stage=wave cause={:?} projected={} started={} now={} horizon={} expires={} start_slack={} freshness_slack={}",
                            saved.route, result.steps.len(), failure.cause, failure.projected,
                            result.state.started_at_ns, result.state.now_ns, snapshot.scope.horizon_end_ns,
                            offer.expires_at_ns(), result.state.minimum_start_slack_ns,
                            result.state.minimum_cost_freshness_slack_ns);
                        reason(failure.cause)
                    })?;
                    (PrefixPathStep::Wave(transition.wave), transition.state)
                }
                expected if stage.is_some() && step_stage(expected) == stage => {
                    let stage = stage.unwrap();
                    let (step, state) = simulation::advance_prefix(
                        snapshot,
                        &result.state,
                        offer,
                        stage,
                        capture_span_start(&result, offer)?,
                        maintenance,
                        maintenance_version,
                        &mut || budget.read(clock).map(|_| ()),
                        planner.settings.search.enable_prefill_milestones,
                        protection,
                    )
                    .map_err(|error| {
                        #[cfg(any(test, feature = "planning-diagnostics"))]
                        eprintln!("prefix original replay rejected: route={:?} edge={} stage={stage:?} cause={error:?} started={} now={} horizon={} expires={} start_slack={} freshness_slack={}",
                            saved.route, result.steps.len(), result.state.started_at_ns,
                            result.state.now_ns, snapshot.scope.horizon_end_ns, offer.expires_at_ns(),
                            result.state.minimum_start_slack_ns, result.state.minimum_cost_freshness_slack_ns);
                        reason(error)
                    })?;
                    if stage == PrefixMaintenanceStage::Capture {
                        result.route = if matches!(offer, PathOffer::CacheCapture(_)) {
                            Route::CaptureDone
                        } else {
                            Route::Captured
                        };
                        result.capture_ready_at_ns = Some(state.now_ns);
                    } else {
                        result.route = Route::Restored;
                        result.restore_ready_at_ns = Some(state.now_ns);
                    }
                    (step, state)
                }
                _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
            };
            if &step != expected {
                return Err(PlanningUnknownReason::InvalidShapeEvidence);
            }
            if result.steps.is_empty() {
                result.first_action_cost_ns = state
                    .now_ns
                    .checked_sub(result.state.now_ns)
                    .ok_or(PlanningUnknownReason::ClockMovedBackwards)?;
                if let PrefixPathStep::Wave(wave) = &step {
                    state.first_wave_candidate = Some(Arc::new(wave.clone()));
                }
            }
            result.steps.push(step);
            result.state = state;
        }
        if !complete(
            &result,
            snapshot,
            offer,
            planner.settings.search.enable_prefill_milestones,
            protection,
        )? {
            #[cfg(any(test, feature = "planning-diagnostics"))]
            eprintln!("prefix original replay incomplete: route={:?} edges={} started={} now={} horizon={} expires={} start_slack={} freshness_slack={}",
                saved.route, result.steps.len(), result.state.started_at_ns,
                result.state.now_ns, snapshot.scope.horizon_end_ns, offer.expires_at_ns(),
                result.state.minimum_start_slack_ns, result.state.minimum_cost_freshness_slack_ns);
            return Err(PlanningUnknownReason::SearchIncomplete);
        }
        budget.read(clock)?;
        Ok(result)
    })();
    if let (Some(observer), Some(replay)) = (observer, replay_id) {
        observer.end_replay(
            replay,
            match &replay_result {
                Ok(_) => super::super::observation::PlanningQueryAttemptEnd::Completed,
                Err(reason) => super::super::observation::PlanningQueryAttemptEnd::Unknown(*reason),
            },
        );
    }
    replay_result
}

fn finalize_path(
    path: Path<'_>,
    offer: PathOffer<'_>,
    final_now: u64,
    remaining_slack: &mut u64,
) -> Result<PrefixPathEvidence, PlanningUnknownReason> {
    let delay = final_now
        .checked_sub(path.state.started_at_ns)
        .ok_or(PlanningUnknownReason::ClockMovedBackwards)?;
    let slack = path
        .state
        .minimum_start_slack_ns
        .min(path.state.minimum_cost_freshness_slack_ns)
        .checked_sub(delay)
        .ok_or_else(|| {
            #[cfg(any(test, feature = "planning-diagnostics"))]
            eprintln!("prefix original finalization rejected: route={:?} edges={} final_now={final_now} started={} delay={delay} simulated_end={} start_slack={} freshness_slack={} expires={}",
                path.route, path.steps.len(), path.state.started_at_ns, path.state.now_ns,
                path.state.minimum_start_slack_ns, path.state.minimum_cost_freshness_slack_ns,
                offer.expires_at_ns());
            PlanningUnknownReason::SearchIncomplete
        })?;
    *remaining_slack = (*remaining_slack).min(slack);
    let shift = |value: u64| {
        value
            .checked_add(delay)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)
    };
    let target = offer
        .target()
        .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
    let first = path
        .state
        .requests
        .iter()
        .find(|r| r.key == *target)
        .and_then(|r| r.timing.first_commit_at_ns)
        .ok_or(PlanningUnknownReason::SearchIncomplete)?;
    Ok(PrefixPathEvidence {
        first_commit_at_ns: shift(first)?,
        completion_at_ns: shift(path.state.now_ns)?,
        capture_ready_at_ns: path.capture_ready_at_ns.map(shift).transpose()?,
        restore_ready_at_ns: path.restore_ready_at_ns.map(shift).transpose()?,
        steps: path.steps,
    })
}

fn reason(error: SimulationFailure) -> PlanningUnknownReason {
    match error {
        SimulationFailure::Unknown(reason) => reason,
        SimulationFailure::SequenceViolation => PlanningUnknownReason::SearchIncomplete,
    }
}

fn prepare_initial<'epoch>(
    state: &mut PlanningState<'epoch>,
    snapshot: &SchedulerSnapshot,
    offer: PathOffer<'_>,
    route: Route,
    phase: Option<InitialPhase>,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    if let Some(phase) = phase {
        state.execution = super::super::execution::checked(poll, |poll| match (offer, phase) {
            (PathOffer::Rendezvous(offer), InitialPhase::Rendezvous(phase)) => {
                state.execution.bind_prefix_continuation(
                    &PlanningPrefixContinuationInput {
                        snapshot,
                        offer,
                        phase,
                    },
                    poll,
                )
            }
            (PathOffer::Ready(offer), InitialPhase::Ready(phase)) => {
                state.execution.bind_ready_prefix(
                    &PlanningReadyPrefixInput {
                        snapshot,
                        offer,
                        phase,
                    },
                    poll,
                )
            }
            (PathOffer::CacheCapture(offer), InitialPhase::CacheCapture(phase)) => {
                state.execution.bind_prefix_cache_capture(
                    &PlanningPrefixCacheCaptureBindingInput {
                        snapshot,
                        offer,
                        phase,
                    },
                    poll,
                )
            }
            _ => Err(PlanningUnknownReason::InvalidShapeEvidence),
        })?
        .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?;
    }
    if route == Route::CapturePreparing {
        let PathOffer::CacheCapture(offer) = offer else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        };
        simulation::restrict_cache_capture(state, offer)?;
    }
    if route == Route::Producer {
        let PathOffer::Rendezvous(offer) = offer else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        };
        simulation::restrict_prefix_wait(state, offer)?;
    }
    Ok(())
}

fn step_stage(step: &PrefixPathStep) -> Option<PrefixMaintenanceStage> {
    match step {
        PrefixPathStep::Wave(_) => None,
        PrefixPathStep::Maintenance(e) => Some(e.stage),
        PrefixPathStep::ReadyRestore(_) => Some(PrefixMaintenanceStage::Restore),
    }
}
