use super::obligations::PlanningObligationSet;
use super::{
    candidates,
    execution::{ExecutionSession, PlanningExecutionContext, ReplayContext},
    simulation::{self, PlanningState, SimulatedSequence, SimulationFailure},
    types::*,
    validation,
};
use std::sync::Arc;

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
    score: f64,
    debt: u64,
    ordinal: usize,
}

struct Frame<'epoch> {
    node: Node<'epoch>,
    work: Option<std::vec::IntoIter<Vec<CandidateWork>>>,
    resolved: usize,
}
impl<'epoch> Frame<'epoch> {
    fn new(node: Node<'epoch>) -> Self {
        Self {
            node,
            work: None,
            resolved: 0,
        }
    }
}

struct ComputeBudget {
    last_ns: u64,
    search_deadline_ns: u64,
    planner_deadline_ns: u64,
    optional_search: bool,
    soft_stopped: bool,
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
            search_deadline_ns,
            planner_deadline_ns,
            optional_search: false,
            soft_stopped: false,
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
        if self.optional_search && now >= self.search_deadline_ns {
            self.soft_stopped = true;
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
        let initial = match simulation::begin(
            snapshot,
            context,
            &mut || budget.read(clock).map(|_| ()),
            start_ns,
        ) {
            Ok(state) => state,
            Err(error) => return unknown(simulation_reason(error), stats),
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
            budget.optional_search = !solutions.is_empty();
            if let Err(reason) = budget.read(clock) {
                stop_or_unknown!(reason, 'exploration);
            }
            if stats.expanded_candidates == expansion_limit {
                stats.candidate_truncations += 1;
                break;
            }
            let mut frame = pending[depth].remove(0);
            stats.max_depth_reached = stats.max_depth_reached.max(depth + 1);
            if frame.work.is_none() {
                let remaining_raw = raw_limit - stats.enumeration_attempts;
                if remaining_raw == 0 {
                    stats.candidate_truncations += 1;
                    continue;
                }
                let generated = match candidates::logical_candidates(
                    snapshot,
                    &frame.node.state.requests,
                    frame.node.state.now_ns,
                    candidate_limit,
                    remaining_raw,
                    &mut stats.enumeration_attempts,
                    protection.as_deref(),
                    &mut || budget.read(clock).map(|_| ()),
                ) {
                    Ok(generated) => generated,
                    Err(reason) => stop_or_unknown!(reason, 'exploration),
                };
                stats.candidate_truncations += usize::from(generated.truncated);
                frame.work = Some(generated.work.into_iter());
            }
            let work = frame.work.as_mut().expect("initialized logical cursor");
            if frame.resolved == candidate_limit {
                stats.candidate_truncations += usize::from(work.len() != 0);
                continue;
            }
            let Some(work) = work.next() else { continue };
            // All search nodes use one execution-time origin. CPU time is
            // charged by the shared real budget and fresh final replay; extending
            // a node never shifts its ancestors or replays their physical work.
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
            waves.push(transition.wave);
            let state = transition.state;
            let (score, debt) = match simulation::score(snapshot, &state, &self.settings) {
                Ok(value) => value,
                Err(reason) => return unknown(reason, stats),
            };
            let node = Node {
                waves,
                state,
                score,
                debt,
                ordinal: stats.expanded_candidates,
            };
            match simulation::ready_witness(
                snapshot,
                &node.state,
                milestones,
                protection.as_deref(),
            ) {
                Ok(true) if admission_serviced(admission_target, &node.state) => {
                    solutions.push(node.clone());
                    retain_best(&mut solutions, width);
                }
                Ok(_) => {}
                Err(reason) => return unknown(reason, stats),
            }
            if depth + 1 < horizon {
                pending[depth + 1].push(Frame::new(node));
                let nodes = &mut pending[depth + 1];
                nodes.sort_by(|left, right| compare_nodes(&left.node, &right.node));
                let before = nodes.len();
                nodes.truncate(width);
                stats.beam_pruned_nodes += before - nodes.len();
                preferred_depth = Some(depth + 1);
            }
        }
        // Replay the complete shared witness after search overhead. The engine
        // must still revalidate identity/resources/credits and its real current
        // clock immediately before committing this *first* wave.
        budget.optional_search = false;
        for solution in solutions {
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
            let (score, debt) = match simulation::score(snapshot, &state, &self.settings) {
                Ok(value) => value,
                Err(reason) => return unknown(reason, stats),
            };
            // Detect time spent inside the cost-model callback too. Crossing the
            // budget cannot be hidden by the last lookup returning a prediction.
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
            let first_wave = SelectedWave {
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

fn retain_best(nodes: &mut Vec<Node>, limit: usize) {
    nodes.sort_by(compare_nodes);
    nodes.truncate(limit);
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
