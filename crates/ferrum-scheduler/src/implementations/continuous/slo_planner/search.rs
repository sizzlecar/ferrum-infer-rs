use super::obligations::PlanningObligationSet;
use super::{
    candidates,
    execution::{ExecutionSession, PlanningExecutionContext, ReplayContext},
    simulation::{self, PlanningState, SimulatedSequence, SimulationFailure},
    types::*,
    validation,
};
use std::sync::Arc;

mod replay_budget;
use replay_budget::{MeasuredReplayWork, ReplayReserve};

/// Finite heuristic search. Neither candidate truncation, beam pruning, nor
/// exhausted depth is ever an impossibility proof. All surviving branches are
/// replayed from the current observation clock before proposing their first wave.
#[derive(Debug, Clone, Default)]
pub struct BoundedSloPlanner {
    pub settings: BoundedPlannerSettings,
}

#[derive(Clone)]
struct Node<'epoch> {
    waves: Vec<WaveCandidate>,
    state: PlanningState<'epoch>,
    replay_work: MeasuredReplayWork,
    score: f64,
    debt: u64,
    ordinal: usize,
}

struct Frame<'epoch> {
    node: Node<'epoch>,
    work: Option<candidates::FrontierCursor>,
    resolved: usize,
    /// One known child is waiting for a same-parent challenger. This is a
    /// proposal comparison, not an incumbent or a second planning budget.
    challenge_pending: bool,
}
impl<'epoch> Frame<'epoch> {
    fn new(node: Node<'epoch>) -> Self {
        Self {
            node,
            work: None,
            resolved: 0,
            challenge_pending: false,
        }
    }
}

/// A complete common tail, never just a high-scoring prefix. Construction and
/// improvement share the same transaction, action limits and final replay.
struct CommonPlan<'epoch>(Node<'epoch>);

struct ComputeBudget {
    last_ns: u64,
    replay_reserve: ReplayReserve,
    planner_deadline_ns: u64,
    optional_search: bool,
    soft_stopped: bool,
    stopped_for_replay_reserve: bool,
}

impl ComputeBudget {
    fn new(
        window: PlanningBudgetWindow,
        now_ns: u64,
        settings: &ferrum_types::SloPlannerConfig,
    ) -> Result<Self, PlanningUnknownReason> {
        let (search_deadline_ns, planner_deadline_ns) = window.phase_deadlines(settings)?;
        if window.started_at_ns > now_ns || now_ns >= planner_deadline_ns {
            return Err(PlanningUnknownReason::ComputeBudgetExhausted);
        }
        Ok(Self {
            last_ns: now_ns,
            replay_reserve: ReplayReserve::new(
                window.started_at_ns,
                search_deadline_ns,
                planner_deadline_ns,
            ),
            planner_deadline_ns,
            optional_search: false,
            soft_stopped: false,
            stopped_for_replay_reserve: false,
        })
    }

    fn read(&mut self, clock: &mut dyn PlanningClock) -> Result<u64, PlanningUnknownReason> {
        let now = clock.now_ns();
        if now < self.last_ns {
            return Err(PlanningUnknownReason::ClockMovedBackwards);
        }
        self.last_ns = now;
        if now >= self.planner_deadline_ns {
            return Err(PlanningUnknownReason::ComputeBudgetExhausted);
        }
        if self.optional_search && now >= self.replay_reserve.deadline_ns() {
            self.soft_stopped = true;
            self.stopped_for_replay_reserve = self.replay_reserve.is_early_stop(now);
            return Err(PlanningUnknownReason::SearchIncomplete);
        }
        Ok(now)
    }
}

impl BoundedSloPlanner {
    pub fn propose(
        &self,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        self.propose_internal(snapshot, model, resolver, None, None, None, clock)
    }

    /// Use the executor's complete resource model instead of the scalar
    /// KV/workspace approximation. The resolver is mandatory on this path;
    /// missing evidence cannot fall back to invented scalar capacity.
    pub fn propose_with_resources(
        &self,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: &dyn PlanningResourceResolver,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        self.propose_internal(
            snapshot,
            model,
            resolver,
            Some(resources),
            None,
            None,
            clock,
        )
    }

    /// A new-request witness must include its transition through prefill and
    /// one subsequent decode (or completion for a one-token request), together
    /// with every existing obligation. TTFT-only feasibility cannot establish
    /// that the newly introduced decoder can coexist with its peers.
    /// The result remains a finite empirical witness, never a lifetime promise
    /// or a physical admission permit.
    pub fn propose_admission_with_resources(
        &self,
        snapshot: &SchedulerSnapshot,
        target: &RequestWorkKey,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: &dyn PlanningResourceResolver,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        self.propose_internal(
            snapshot,
            model,
            resolver,
            Some(resources),
            Some(target),
            None,
            clock,
        )
    }

    /// Recover forward protection without changing the original failed clocks.
    /// Scope must be captured from exactly this full snapshot, before search.
    pub fn propose_recovery(
        &self,
        snapshot: &SchedulerSnapshot,
        protection: Arc<PlanningObligationSet>,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: Option<&dyn PlanningResourceResolver>,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        self.propose_internal(
            snapshot,
            model,
            resolver,
            resources,
            None,
            Some(protection),
            clock,
        )
    }

    fn propose_internal(
        &self,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        resolver: &dyn PlanningShapeResolver,
        resources: Option<&dyn PlanningResourceResolver>,
        admission_target: Option<&RequestWorkKey>,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        let context = ReplayContext {
            resolver,
            resources,
        };
        self.propose_joint(
            snapshot,
            model,
            &context,
            resources.is_some(),
            admission_target,
            protection,
            clock,
        )
    }

    pub fn propose_with_execution(
        &self,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        context: &dyn PlanningExecutionContext,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        self.propose_joint(snapshot, model, context, true, None, None, clock)
    }

    pub fn propose_admission_with_execution(
        &self,
        snapshot: &SchedulerSnapshot,
        target: &RequestWorkKey,
        model: &dyn PlanningCostModel,
        context: &dyn PlanningExecutionContext,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        self.propose_joint(snapshot, model, context, true, Some(target), None, clock)
    }

    pub fn propose_recovery_with_execution(
        &self,
        snapshot: &SchedulerSnapshot,
        protection: Arc<PlanningObligationSet>,
        model: &dyn PlanningCostModel,
        context: &dyn PlanningExecutionContext,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        self.propose_joint(
            snapshot,
            model,
            context,
            true,
            None,
            Some(protection),
            clock,
        )
    }

    fn propose_joint(
        &self,
        snapshot: &SchedulerSnapshot,
        model: &dyn PlanningCostModel,
        context: &dyn PlanningExecutionContext,
        complete_resources: bool,
        admission_target: Option<&RequestWorkKey>,
        protection: Option<Arc<PlanningObligationSet>>,
        clock: &mut dyn PlanningClock,
    ) -> PlanningDecision {
        let mut stats = PlanningSearchStats::default();
        let start_ns = clock.now_ns();
        if start_ns < snapshot.observed_at_ns {
            return unknown(PlanningUnknownReason::ClockMovedBackwards, stats);
        }
        if let Err(reason) = validation::validate(&self.settings, snapshot, complete_resources) {
            return unknown(reason, stats);
        }
        if admission_target.is_some_and(|key| {
            snapshot
                .requests
                .iter()
                .find(|request| &request.key == key)
                .is_none_or(|request| {
                    request.timing.committed_tokens != 0
                        || !matches!(request.phase, RequestPhaseView::Prefill(_))
                })
        }) {
            return unknown(PlanningUnknownReason::InvalidSnapshot, stats);
        }
        if model.model_version() != snapshot.cost_model_version {
            return unknown(PlanningUnknownReason::ModelVersionMismatch, stats);
        }
        if let Some(scope) = &protection {
            if !scope.matches(snapshot) || scope.classified_at_ns() > start_ns {
                return unknown(PlanningUnknownReason::InvalidSnapshot, stats);
            }
        } else if let Some(decision) = self.impossibility(snapshot, start_ns, stats) {
            return decision;
        }
        if start_ns >= snapshot.scope.horizon_end_ns {
            return unknown(PlanningUnknownReason::HorizonInsufficient, stats);
        }
        let transaction_start = if admission_target.is_some() {
            snapshot.observed_at_ns
        } else {
            start_ns
        };
        let configured_end =
            transaction_start.checked_add(self.settings.search.max_planning_us.get() * 1000);
        let window = match clock.planning_budget_window() {
            Some(window) => window,
            None => match configured_end {
                Some(deadline_ns) => PlanningBudgetWindow {
                    started_at_ns: transaction_start,
                    deadline_ns,
                },
                None => return unknown(PlanningUnknownReason::ArithmeticOverflow, stats),
            },
        };
        let mut budget = match ComputeBudget::new(window, start_ns, &self.settings.search) {
            Ok(value) => value,
            Err(reason) => return unknown(reason, stats),
        };
        let execution_session = ExecutionSession::new(context, &self.settings);
        let context: &dyn PlanningExecutionContext = &execution_session;
        let milestones = self.settings.search.enable_prefill_milestones;
        let future_controller_ns = match self
            .settings
            .future_controller_time
            .reserved_ns(&self.settings.search)
        {
            Ok(value) => value,
            Err(reason) => return unknown(reason, stats),
        };
        let begin_started = match budget.read(clock) {
            Ok(now) => now,
            Err(reason) => return unknown(reason, stats),
        };
        let initial = match simulation::begin_with_controller_time(
            snapshot,
            context,
            &mut || budget.read(clock).map(|_| ()),
            start_ns,
            future_controller_ns,
        ) {
            Ok(state) => state,
            Err(error) => return unknown(simulation_reason(error), stats),
        };
        let initial_replay_work = match budget
            .read(clock)
            .and_then(|now| MeasuredReplayWork::default().with_span(begin_started, now))
        {
            Ok(work) => work,
            Err(reason) => return unknown(reason, stats),
        };
        let horizon = self.settings.search.lookahead_waves.get();
        let width = self.settings.search.beam_width.get();
        let candidate_limit = self.settings.search.candidate_limit.get();
        // Validation bounds this product. Changing exploration order must not
        // increase the original K * B * H successful-expansion/work envelope.
        let expansion_limit = candidate_limit * width * horizon;
        let raw_limit = expansion_limit * 8;
        let mut pending: Vec<Vec<Frame>> = (0..horizon).map(|_| Vec::new()).collect();
        pending[0].push(Frame::new(Node {
            waves: Vec::new(),
            state: initial,
            replay_work: initial_replay_work,
            score: 0.0,
            debt: 0,
            ordinal: 0,
        }));
        let mut solutions = Vec::new();
        let mut saw_resource_block = false;
        let mut saw_candidate = false;
        macro_rules! stop_or_unknown {
            ($reason:expr, $label:lifetime) => {{
                let reason = $reason;
                if reason == PlanningUnknownReason::SearchIncomplete
                    && budget.soft_stopped
                    && !solutions.is_empty()
                {
                    stats.search_soft_stops += 1;
                    stats.replay_reserve_stops += usize::from(budget.stopped_for_replay_reserve);
                    break $label;
                }
                return unknown(reason, stats);
            }};
        }
        // Resolve and evaluate one candidate at a time, then first pursue its
        // complete shared witness. Parent cursors retain untried siblings: a
        // dead-end full decode prefix cannot permanently occupy a depth quota.
        // Pending nodes (not lifetime-expanded nodes) are beam-bounded per depth.
        let mut preferred_depth: Option<usize> = None;
        'exploration: loop {
            // Follow a newly viable prefix once. After a dead end or leaf,
            // return to its retained siblings rather than exhaust that entire
            // subtree before admitting a later mixed/prefill family to the beam.
            let Some(depth) = preferred_depth
                .take()
                .filter(|&depth| !pending[depth].is_empty())
                .or_else(|| pending.iter().position(|nodes| !nodes.is_empty()))
            else {
                break;
            };
            let phase = if solutions.is_empty() {
                PlanningSearchPhase::Construct
            } else {
                PlanningSearchPhase::Improve
            };
            stats.phase = phase;
            budget.optional_search = matches!(phase, PlanningSearchPhase::Improve);
            if let Err(reason) = budget.read(clock) {
                stop_or_unknown!(reason, 'exploration);
            }
            if stats.expanded_candidates == expansion_limit
                || stats.enumeration_attempts == raw_limit
            {
                stats.candidate_truncations += 1;
                break;
            }
            let ranking_now = match budget.read(clock) {
                Ok(now) => now,
                Err(reason) => stop_or_unknown!(reason, 'exploration),
            };
            let selected =
                if let Some(index) = pending[depth].iter().position(|f| f.challenge_pending) {
                    // Finish this bounded comparison against the same parent,
                    // rather than accidentally probing a differently scored peer.
                    index
                } else {
                    if let Err(reason) = rank_frames(
                        &mut pending[depth],
                        snapshot,
                        &self.settings,
                        window.started_at_ns,
                        ranking_now,
                        &mut || budget.read(clock).map(|_| ()),
                    ) {
                        stop_or_unknown!(reason, 'exploration);
                    }
                    0
                };
            let mut frame = pending[depth].remove(selected);
            stats.max_depth_reached = stats.max_depth_reached.max(depth + 1);
            if frame.work.is_none() {
                frame.work = Some(
                    match candidates::FrontierCursor::with_remaining_waves(
                        snapshot,
                        &frame.node.state.requests,
                        frame.node.state.now_ns,
                        candidate_limit,
                        protection.as_deref(),
                        std::num::NonZeroUsize::new(horizon - depth),
                        &mut || budget.read(clock).map(|_| ()),
                    ) {
                        Ok(cursor) => cursor,
                        Err(reason) => stop_or_unknown!(reason, 'exploration),
                    },
                );
            }
            let cursor = frame.work.as_mut().expect("initialized logical cursor");
            if frame.resolved == candidate_limit {
                stats.candidate_truncations += usize::from(cursor.may_have_more());
                if frame.challenge_pending {
                    preferred_depth = Some(depth + 1);
                }
                continue;
            }
            let work = match cursor.next(
                snapshot,
                &frame.node.state.requests,
                &mut stats.enumeration_attempts,
                raw_limit,
                &mut || budget.read(clock).map(|_| ()),
            ) {
                Ok(Some(work)) => work,
                Ok(None) => {
                    stats.candidate_truncations += usize::from(cursor.truncated);
                    if frame.challenge_pending {
                        preferred_depth = Some(depth + 1);
                    }
                    continue;
                }
                Err(reason) => stop_or_unknown!(reason, 'exploration),
            };
            let has_prefill_choices = cursor.has_prefill_choices();
            let was_challenger = std::mem::take(&mut frame.challenge_pending);
            if was_challenger {
                // A rejected/Unknown challenger still ends the pair. It must
                // not force repeated probes before the known child can run.
                preferred_depth = Some(depth + 1);
            }
            // All search nodes use one execution-time origin. CPU time is
            // charged by the shared real budget and fresh final replay. Each
            // future edge also reserves its own controller transaction, without
            // shifting ancestors or replaying their physical work.
            let advance_started = match budget.read(clock) {
                Ok(now) => now,
                Err(reason) => stop_or_unknown!(reason, 'exploration),
            };
            let parent_replay_work = frame.node.replay_work;
            let transition = simulation::advance(
                snapshot,
                &frame.node.state,
                &work,
                model,
                complete_resources,
                &mut || budget.read(clock).map(|_| ()),
                milestones,
                protection.as_deref(),
            );
            let projected = transition
                .as_ref()
                .map_or_else(|failure| failure.projected, |_| true);
            let transition = transition.map_err(|failure| failure.cause);
            if matches!(
                transition,
                Err(SimulationFailure::Unknown(
                    PlanningUnknownReason::ShapeUnavailable
                ))
            ) {
                stats.shape_unknown_candidates += 1;
                pending[depth].insert(0, frame);
                continue;
            }
            if projected {
                frame.resolved += 1;
                stats.generated_candidates += 1;
                stats.expanded_candidates += 1;
                saw_candidate = true;
            }
            let mut waves = frame.node.waves.clone();
            pending[depth].insert(0, frame);
            let transition = match transition {
                Ok(value) => value,
                Err(SimulationFailure::Unknown(PlanningUnknownReason::CostUnavailable)) => {
                    stats.cost_unknown_candidates += 1;
                    continue;
                }
                Err(SimulationFailure::Unknown(PlanningUnknownReason::UnknownResourceEvidence)) => {
                    stats.resource_unknown_candidates += 1;
                    continue;
                }
                Err(SimulationFailure::Unknown(PlanningUnknownReason::OutputOrResourceBlocked)) => {
                    saw_resource_block = true;
                    continue;
                }
                Err(SimulationFailure::SequenceViolation) => continue,
                Err(SimulationFailure::Unknown(reason)) => stop_or_unknown!(reason, 'exploration),
            };
            // Measure this edge and its adjacent successful-result bookkeeping,
            // never a previously measured parent or sibling. Failed branches do
            // not create an estimate; final ranking remains covered only by F.
            let replay_work = match budget
                .read(clock)
                .and_then(|now| parent_replay_work.with_span(advance_started, now))
            {
                Ok(work) => work,
                Err(reason) => stop_or_unknown!(reason, 'exploration),
            };
            waves.push(transition.wave);
            let state = transition.state;
            let node = Node {
                waves,
                state,
                replay_work,
                score: 0.0,
                debt: 0,
                ordinal: stats.expanded_candidates,
            };
            let mut complete = false;
            match simulation::ready_witness(
                snapshot,
                &node.state,
                milestones,
                protection.as_deref(),
            ) {
                Ok(true) if admission_serviced(admission_target, &node.state) => {
                    complete = true;
                    solutions.push(CommonPlan(node.clone()));
                    if let Err(reason) = budget.replay_reserve.observe_complete(node.replay_work) {
                        return unknown(reason, stats);
                    }
                    stats.measured_replay_work_ns = budget.replay_reserve.measured_ns();
                    stats.replay_reserve_ns = budget.replay_reserve.reserved_ns();
                    // Crossing Construct -> Improve does not create a new
                    // budget or permit an uncertified tail into solutions.
                    budget.optional_search = true;
                    stats.phase = PlanningSearchPhase::Improve;
                    let now = match budget.read(clock) {
                        Ok(now) => now,
                        Err(reason) => stop_or_unknown!(reason, 'exploration),
                    };
                    if let Err(reason) = rank_plans(
                        &mut solutions,
                        snapshot,
                        &self.settings,
                        window.started_at_ns,
                        now,
                        &mut || budget.read(clock).map(|_| ()),
                    ) {
                        stop_or_unknown!(reason, 'exploration);
                    }
                    solutions.truncate(width);
                }
                Ok(_) => {}
                Err(reason) => return unknown(reason, stats),
            }
            if depth + 1 < horizon {
                pending[depth + 1].push(Frame::new(node));
                let nodes = &mut pending[depth + 1];
                let now = match budget.read(clock) {
                    Ok(now) => now,
                    Err(reason) => stop_or_unknown!(reason, 'exploration),
                };
                if let Err(reason) = rank_frames(
                    nodes,
                    snapshot,
                    &self.settings,
                    window.started_at_ns,
                    now,
                    &mut || budget.read(clock).map(|_| ()),
                ) {
                    stop_or_unknown!(reason, 'exploration);
                }
                let before = nodes.len();
                nodes.truncate(width);
                stats.beam_pruned_nodes += before - nodes.len();
                let parent = &mut pending[depth][0];
                // Before any complete incumbent, continue the first viable prefix.
                // A cheap sibling has no certified tail and must not evict it just
                // on partial score. Dead ends still return to shallow siblings;
                // this is not an unbounded depth-first search or an extra beam.
                if !solutions.is_empty()
                    && !complete
                    && !was_challenger
                    && has_prefill_choices
                    && parent.resolved < candidate_limit
                    && parent
                        .work
                        .as_ref()
                        .is_some_and(|cursor| cursor.may_have_more())
                {
                    // At most one additional real transition before descent:
                    // compare useful work/cost at one ranking_now, not raw
                    // ascending chunk counts. Both transitions retain their
                    // original pure successors and consume the global limits.
                    parent.challenge_pending = true;
                    preferred_depth = Some(depth);
                } else {
                    preferred_depth = Some(depth + 1);
                }
            }
        }
        // Replay the complete shared witness after search overhead. The engine
        // must still revalidate identity/resources/credits and its real current
        // clock immediately before committing this *first* wave.
        budget.optional_search = false;
        stats.phase = PlanningSearchPhase::Finalization;
        // Re-rank every complete plan at the same end-of-search CPU instant.
        // A soft stop may interrupt an earlier ranking; cached scores cannot
        // select a winner at a different time basis from its peers.
        if !solutions.is_empty() {
            let now = match budget.read(clock) {
                Ok(now) => now,
                Err(reason) => return unknown(reason, stats),
            };
            if let Err(reason) = rank_plans(
                &mut solutions,
                snapshot,
                &self.settings,
                window.started_at_ns,
                now,
                &mut || budget.read(clock).map(|_| ()),
            ) {
                return unknown(reason, stats);
            }
        }
        for CommonPlan(solution) in solutions {
            let now_ns = match budget.read(clock) {
                Ok(now) => now,
                Err(reason) => return unknown(reason, stats),
            };
            if protection.is_none() {
                if let Some(decision) = self.impossibility(snapshot, now_ns, stats) {
                    return decision;
                }
            }
            let state = match simulation::replay(
                snapshot,
                &solution.waves,
                model,
                context,
                complete_resources,
                &mut || budget.read(clock).map(|_| ()),
                now_ns,
                future_controller_ns,
                milestones,
                protection.as_deref(),
            ) {
                Ok(state) => state,
                Err(SimulationFailure::SequenceViolation) => continue,
                Err(SimulationFailure::Unknown(PlanningUnknownReason::ShapeUnavailable)) => {
                    stats.shape_unknown_candidates += 1;
                    continue;
                }
                Err(SimulationFailure::Unknown(PlanningUnknownReason::CostUnavailable)) => {
                    stats.cost_unknown_candidates += 1;
                    continue;
                }
                Err(SimulationFailure::Unknown(PlanningUnknownReason::UnknownResourceEvidence)) => {
                    stats.resource_unknown_candidates += 1;
                    continue;
                }
                Err(SimulationFailure::Unknown(reason)) => return unknown(reason, stats),
            };
            match simulation::ready_witness(snapshot, &state, milestones, protection.as_deref()) {
                Ok(true) if admission_serviced(admission_target, &state) => {}
                Ok(_) => continue,
                Err(reason) => return unknown(reason, stats),
            }
            // Detect time spent inside the cost-model callback too. Crossing the
            // budget cannot be hidden by the last lookup returning a prediction.
            let final_now = match budget.read(clock) {
                Ok(now) => now,
                Err(reason) => return unknown(reason, stats),
            };
            let (score, debt) = match simulation::score_at(
                snapshot,
                &state,
                &self.settings,
                window.started_at_ns,
                final_now,
                &mut || budget.read(clock).map(|_| ()),
            ) {
                Ok(value) => value,
                Err(reason) => return unknown(reason, stats),
            };
            // Ranking is bounded but not free: guard the time it consumed too.
            // The score is only an ordering diagnostic, not a deadline permit.
            let final_now = match budget.read(clock) {
                Ok(now) => now,
                Err(reason) => return unknown(reason, stats),
            };
            let final_delay = final_now - now_ns;
            if final_now >= snapshot.scope.horizon_end_ns {
                return unknown(PlanningUnknownReason::HorizonInsufficient, stats);
            }
            if final_delay > state.minimum_cost_freshness_slack_ns {
                stats.cost_unknown_candidates += 1;
                continue;
            }
            if final_delay > state.minimum_start_slack_ns {
                continue;
            }
            let Some(completion_at_ns) = state.now_ns.checked_add(final_delay) else {
                return unknown(PlanningUnknownReason::ArithmeticOverflow, stats);
            };
            let Some(canonical) = state.first_wave_canonical.clone() else {
                return unknown(PlanningUnknownReason::InvalidShapeEvidence, stats);
            };
            let first_wave = SelectedWave {
                final_replay_first_wave: Some(Arc::new(FinalReplayFirstWave::from_replay(
                    snapshot,
                    solution.waves[0].clone(),
                    canonical,
                ))),
                protection: protection.clone(),
                candidate: solution.waves[0].clone(),
                predicted_wall_ns: state.first_wave_cost_ns,
                planning_observed_at_ns: final_now,
                snapshot_observed_at_ns: snapshot.observed_at_ns,
                snapshot_generation: snapshot.generation,
                cost_model_version: snapshot.cost_model_version,
                witness_valid_for_ns: (state.minimum_start_slack_ns - final_delay)
                    .min(state.minimum_cost_freshness_slack_ns - final_delay)
                    .min(snapshot.scope.horizon_end_ns - final_now - 1),
            };
            let witness = PlanningWitnessSummary {
                waves: solution.waves.len(),
                completion_at_ns,
                validated_through_ns: snapshot.scope.horizon_end_ns,
                predicted_output_tokens: state.output_tokens,
                net_prefill_reference_work_ns: state.net_prefill_work_ns,
                terminal_prefill_debt_ns: debt,
                proxy_score: score,
                requests_with_obligations_beyond_horizon: state
                    .requests
                    .iter()
                    .filter(|request| !request.timing.completed())
                    .count(),
            };
            return if let Some(protection) = protection.filter(|scope| scope.needs_recovery()) {
                PlanningDecision::ProtectedWithinHorizon {
                    first_wave,
                    witness,
                    protection,
                    search: stats,
                }
            } else {
                PlanningDecision::FeasibleWithinHorizon {
                    first_wave,
                    witness,
                    search: stats,
                }
            };
        }
        let reason = if protection
            .as_ref()
            .and_then(|scope| scope.required_first_service())
            .is_some()
        {
            PlanningUnknownReason::RecoveryConflict
        } else if stats.shape_unknown_candidates > 0 {
            PlanningUnknownReason::ShapeUnavailable
        } else if stats.cost_unknown_candidates > 0 {
            PlanningUnknownReason::CostUnavailable
        } else if stats.resource_unknown_candidates > 0 {
            PlanningUnknownReason::UnknownResourceEvidence
        } else if saw_resource_block {
            PlanningUnknownReason::OutputOrResourceBlocked
        } else if stats.candidate_truncations > 0 || stats.beam_pruned_nodes > 0 {
            PlanningUnknownReason::SearchIncomplete
        } else if !saw_candidate
            && snapshot
                .requests
                .iter()
                .all(|request| request.timing.completed())
        {
            PlanningUnknownReason::NoWork
        } else if !saw_candidate {
            PlanningUnknownReason::OutputOrResourceBlocked
        } else {
            PlanningUnknownReason::HorizonInsufficient
        };
        unknown(reason, stats)
    }

    fn impossibility(
        &self,
        snapshot: &SchedulerSnapshot,
        now_ns: u64,
        stats: PlanningSearchStats,
    ) -> Option<PlanningDecision> {
        match validation::prove_impossible(snapshot, now_ns) {
            Ok(Some(reason)) => Some(PlanningDecision::ProvenImpossibleUnderModel {
                reason,
                model_version: snapshot.cost_model_version,
                snapshot_generation: snapshot.generation,
            }),
            Ok(None) => None,
            Err(reason) => Some(unknown(reason, stats)),
        }
    }
}

fn admission_serviced(target: Option<&RequestWorkKey>, state: &SimulatedSequence) -> bool {
    target.is_none_or(|key| {
        state
            .requests
            .iter()
            .find(|request| &request.key == key)
            .is_some_and(|request| {
                request.timing.committed_tokens >= request.timing.maximum_output_tokens.get().min(2)
            })
    })
}

fn compare_nodes(left: &Node, right: &Node) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.debt.cmp(&right.debt))
        .then_with(|| {
            right
                .state
                .net_prefill_work_ns
                .cmp(&left.state.net_prefill_work_ns)
        })
        .then_with(|| {
            left.state
                .first_fairness_rank
                .cmp(&right.state.first_fairness_rank)
        })
        .then_with(|| left.ordinal.cmp(&right.ordinal))
}

fn rank_node(
    node: &mut Node,
    snapshot: &SchedulerSnapshot,
    settings: &BoundedPlannerSettings,
    origin: u64,
    now: u64,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    // The root has no work and is never compared against a non-root prefix.
    if node.waves.is_empty() {
        return poll();
    }
    (node.score, node.debt) =
        simulation::score_at(snapshot, &node.state, settings, origin, now, poll)?;
    Ok(())
}

fn rank_frames(
    nodes: &mut [Frame],
    snapshot: &SchedulerSnapshot,
    settings: &BoundedPlannerSettings,
    origin: u64,
    now: u64,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    // A sole branch needs no ranking; when a peer arrives both are scored
    // afresh. This avoids an O(N) debt walk merely to pick the only prefix.
    if nodes.len() < 2 {
        return poll();
    }
    for frame in nodes.iter_mut() {
        rank_node(&mut frame.node, snapshot, settings, origin, now, poll)?;
    }
    nodes.sort_by(|left, right| compare_nodes(&left.node, &right.node));
    poll()
}

fn rank_plans(
    nodes: &mut [CommonPlan],
    snapshot: &SchedulerSnapshot,
    settings: &BoundedPlannerSettings,
    origin: u64,
    now: u64,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    if nodes.len() < 2 {
        return poll();
    }
    for plan in nodes.iter_mut() {
        rank_node(&mut plan.0, snapshot, settings, origin, now, poll)?;
    }
    nodes.sort_by(|left, right| compare_nodes(&left.0, &right.0));
    poll()
}

fn unknown(reason: PlanningUnknownReason, search: PlanningSearchStats) -> PlanningDecision {
    PlanningDecision::Unknown { reason, search }
}
fn simulation_reason(reason: SimulationFailure) -> PlanningUnknownReason {
    match reason {
        SimulationFailure::Unknown(reason) => reason,
        SimulationFailure::SequenceViolation => PlanningUnknownReason::SearchIncomplete,
    }
}
