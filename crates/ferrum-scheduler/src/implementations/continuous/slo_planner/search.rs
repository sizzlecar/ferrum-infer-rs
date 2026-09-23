use super::obligations::PlanningObligationSet;
use super::{
    candidates,
    simulation::{self, SimulatedSequence, SimulationFailure},
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
struct Node {
    waves: Vec<WaveCandidate>,
    state: SimulatedSequence,
    score: f64,
    debt: u64,
    ordinal: usize,
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
        let mut stats = PlanningSearchStats::default();
        let start_ns = clock.now_ns();
        if start_ns < snapshot.observed_at_ns {
            return unknown(PlanningUnknownReason::ClockMovedBackwards, stats);
        }
        if let Err(reason) = validation::validate(&self.settings, snapshot, resources.is_some()) {
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
        let resolution_session = super::shape::ResolutionSession::new(resolver, &self.settings);
        let resolver: &dyn PlanningShapeResolver = &resolution_session;
        let milestones = self.settings.search.enable_prefill_milestones;
        let initial = match simulation::simulate(
            snapshot,
            &[],
            model,
            resolver,
            resources,
            &mut || budget.read(clock).map(|_| ()),
            start_ns,
            milestones,
            protection.as_deref(),
        ) {
            Ok(state) => state,
            Err(error) => return unknown(simulation_reason(error), stats),
        };
        let mut beam = vec![Node {
            waves: Vec::new(),
            state: initial,
            score: 0.0,
            debt: 0,
            ordinal: 0,
        }];
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
        'exploration: for depth in 1..=self.settings.search.lookahead_waves.get() {
            let mut next = Vec::new();
            stats.max_depth_reached = depth;
            for node in &beam {
                budget.optional_search = !solutions.is_empty();
                if let Err(reason) = budget.read(clock) {
                    stop_or_unknown!(reason, 'exploration);
                }
                let generated = match candidates::enumerate(
                    snapshot,
                    &node.state.requests,
                    node.state.now_ns,
                    self.settings.search.candidate_limit.get(),
                    protection.as_deref(),
                    &super::shape::PriorWaveResolver {
                        resolver,
                        prior_waves: &node.waves,
                    },
                    &mut || budget.read(clock).map(|_| ()),
                ) {
                    Ok(generated) => generated,
                    Err(reason) => stop_or_unknown!(reason, 'exploration),
                };
                stats.enumeration_attempts += generated.attempts;
                stats.shape_unknown_candidates += generated.shape_unknown;
                stats.generated_candidates += generated.waves.len();
                stats.candidate_truncations += usize::from(generated.truncated);
                for candidate in generated.waves {
                    budget.optional_search = !solutions.is_empty();
                    saw_candidate = true;
                    let now_ns = match budget.read(clock) {
                        Ok(now) => now,
                        Err(reason) => stop_or_unknown!(reason, 'exploration),
                    };
                    let mut waves = node.waves.clone();
                    waves.push(candidate);
                    stats.expanded_candidates += 1;
                    let state = match simulation::simulate(
                        snapshot,
                        &waves,
                        model,
                        resolver,
                        resources,
                        &mut || budget.read(clock).map(|_| ()),
                        now_ns,
                        milestones,
                        protection.as_deref(),
                    ) {
                        Ok(state) => state,
                        Err(SimulationFailure::Unknown(
                            PlanningUnknownReason::ShapeUnavailable,
                        )) => {
                            stats.shape_unknown_candidates += 1;
                            continue;
                        }
                        Err(SimulationFailure::Unknown(PlanningUnknownReason::CostUnavailable)) => {
                            stats.cost_unknown_candidates += 1;
                            continue;
                        }
                        Err(SimulationFailure::Unknown(
                            PlanningUnknownReason::UnknownResourceEvidence,
                        )) => {
                            stats.resource_unknown_candidates += 1;
                            continue;
                        }
                        Err(SimulationFailure::Unknown(
                            PlanningUnknownReason::OutputOrResourceBlocked,
                        )) => {
                            saw_resource_block = true;
                            continue;
                        }
                        Err(SimulationFailure::SequenceViolation) => continue,
                        Err(SimulationFailure::Unknown(reason)) => {
                            stop_or_unknown!(reason, 'exploration)
                        }
                    };
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
                            retain_best(&mut solutions, self.settings.search.beam_width.get());
                        }
                        Ok(_) => {}
                        Err(reason) => return unknown(reason, stats),
                    }
                    next.push(node);
                }
            }
            let before = next.len();
            retain_best(&mut next, self.settings.search.beam_width.get());
            stats.beam_pruned_nodes += before - next.len();
            if next.is_empty() {
                break;
            }
            beam = next;
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
            let state = match simulation::simulate(
                snapshot,
                &solution.waves,
                model,
                resolver,
                resources,
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

fn retain_best(nodes: &mut Vec<Node>, limit: usize) {
    nodes.sort_by(|left, right| {
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
    });
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
