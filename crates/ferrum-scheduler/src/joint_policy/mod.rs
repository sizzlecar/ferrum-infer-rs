//! Bounded, conditional action comparison. Execution commits are timing proxies;
//! these simulations do not certify client-visible SSE latency or throughput.

mod admission;
mod aggregation;
mod decision_observation;
mod diagnostics;
mod phase;
mod risk;
mod scenarios;
pub use admission::*;
pub use aggregation::*;
pub use decision_observation::*;
pub use diagnostics::*;
pub use phase::{ContinuationPhase, PhaseBuilder, PrefillTarget};
pub use risk::*;
pub use scenarios::*;

use crate::cost_prediction::*;
use ferrum_execution::execution_attempt::AttemptWorkKind;
use ferrum_types::{CostQueryLimits, RequestId, SchedulerSloConfig};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Unsupported {
    EmptyCohort,
    ParticipantBudget,
    CandidateBudget,
    QueryBudget,
    TimeBudget,
    WaitingResources,
    UnknownKvCapacity,
    PressureActive,
    ColdFillFirst,
    PrefixRendezvous,
    DecodeBackpressure,
    MissingCapacity,
    InsufficientCapacity,
    AdmissionCapacity,
    PendingRequestState,
    InvalidClock,
    NonStreamingClock,
    HistoricalTextCoalescing,
    StoppingCoverage,
    FeasibilityUnresolved,
    InvalidState,
    InvalidPlan,
    NoProgress,
    Cost(CostUnknownReason),
    Arithmetic,
}

impl Unsupported {
    /// A refused planning operation invalidates the comparison, even when an
    /// earlier candidate completed. A per-query provider support limit is a
    /// different contract and remains local to that candidate.
    fn is_hard_budget_failure(self) -> bool {
        matches!(
            self,
            Self::ParticipantBudget
                | Self::CandidateBudget
                | Self::QueryBudget
                | Self::TimeBudget
                | Self::Cost(CostUnknownReason::PlanningBudget)
        )
    }

    /// Preserve the existing resource/unresolved partition when refining a
    /// previously generic resource reason. This is diagnostic classification,
    /// never authority to admit or change the unknown policy.
    pub(crate) fn is_resource_unknown(self) -> bool {
        matches!(
            self,
            Self::WaitingResources
                | Self::UnknownKvCapacity
                | Self::PressureActive
                | Self::ColdFillFirst
                | Self::PrefixRendezvous
                | Self::DecodeBackpressure
                | Self::MissingCapacity
                | Self::InsufficientCapacity
                | Self::AdmissionCapacity
        )
    }
}

/// The resource owner must guarantee headroom for the entire declared cohort.
/// Waiting requests additionally require an explicit bounded admission plan.
#[derive(Debug, Clone, Copy)]
pub struct ResourceSupport {
    pub waiting_requests: usize,
    pub kv_reserved_through_scenario_end: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub prefill_tokens: usize,
    pub output_tokens: usize,
}

impl Progress {
    fn covers(self, target: Self) -> bool {
        self.prefill_tokens >= target.prefill_tokens && self.output_tokens >= target.output_tokens
    }
}

/// All times are relative to this request's original ingress. Output times are
/// committed-output proxies; the transport's visible SSE clock is independent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RequestClock {
    pub first_output: Option<Duration>,
    pub last_output: Option<Duration>,
    pub maximum_output_interval: Duration,
    pub interval_count: usize,
    pub violating_interval_count: usize,
    pub completed: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct SimulatedRequest {
    pub id: RequestId,
    pub prompt_tokens: usize,
    pub prefill_tokens: usize,
    pub output_tokens: usize,
    pub output_cap: usize,
    /// A declared conditional scenario endpoint, never future observed EOS.
    pub scenario_end_tokens: usize,
    pub age: Duration,
    pub central_clock: RequestClock,
    pub risk_clock: RequestClock,
}

impl SimulatedRequest {
    pub fn progress(&self) -> Progress {
        Progress {
            prefill_tokens: self.prefill_tokens,
            output_tokens: self.output_tokens,
        }
    }

    pub fn finished(&self) -> bool {
        self.output_tokens >= self.scenario_end_tokens
    }

    fn valid(&self) -> bool {
        self.prompt_tokens > 0
            && self.prefill_tokens <= self.prompt_tokens
            && self.scenario_end_tokens > 0
            && self.scenario_end_tokens <= self.output_cap
            && self.output_tokens <= self.scenario_end_tokens
            && (self.output_tokens == 0 || self.prefill_tokens == self.prompt_tokens)
            && [self.central_clock, self.risk_clock].iter().all(|clock| {
                clock.first_output.is_none_or(|first| first <= self.age)
                    && clock.last_output.is_none_or(|last| last <= self.age)
                    && clock.violating_interval_count <= clock.interval_count
                    && clock.interval_count <= self.output_tokens.saturating_sub(1)
                    && (self.output_tokens == 0
                        || (clock.first_output.is_some() && clock.last_output.is_some()))
            })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GroupMember {
    pub request: usize,
    pub work: CostWork,
}

/// The adapter appends actual invocation groups through a bounded builder.
/// Split is at most prefill followed by decode; mixed is one invocation.
pub struct PlanBuilder {
    groups: Vec<Vec<GroupMember>>,
    participant_limit: usize,
    query_limits: CostQueryLimits,
    execution: CostExecution,
    active: Option<Vec<bool>>,
}

impl PlanBuilder {
    pub fn push_group(&mut self, members: &[GroupMember]) -> Result<(), Unsupported> {
        if members.is_empty()
            || self.groups.len() >= 2
            || !CostQuery::fits(members.len(), self.query_limits)
            || self
                .groups
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                .checked_add(members.len())
                .is_none_or(|count| count > self.participant_limit)
        {
            return Err(Unsupported::ParticipantBudget);
        }
        if self.execution == CostExecution::SingleInvocation && !self.groups.is_empty() {
            return Err(Unsupported::InvalidPlan);
        }
        if self.execution == CostExecution::SequentialPrefillThenDecode {
            let first_kind = members[0].work.kind;
            if members.iter().any(|member| member.work.kind != first_kind)
                || self.groups.first().is_some_and(|group| {
                    group[0].work.kind != AttemptWorkKind::Prefill
                        || first_kind != AttemptWorkKind::Decode
                })
            {
                return Err(Unsupported::InvalidPlan);
            }
        }
        for (position, member) in members.iter().enumerate() {
            if self
                .active
                .as_ref()
                .is_some_and(|active| !active.get(member.request).copied().unwrap_or(false))
            {
                return Err(Unsupported::WaitingResources);
            }
            if members[..position]
                .iter()
                .any(|prior| prior.request == member.request)
                || self
                    .groups
                    .iter()
                    .flatten()
                    .any(|prior| prior.request == member.request)
            {
                return Err(Unsupported::InvalidPlan);
            }
        }
        self.groups.push(members.to_vec());
        Ok(())
    }
}

/// By default only the first round receives an action, followed by this same
/// adapter with `None`. An explicit bounded phase may retain that action until
/// the core observes its frozen prefill endpoints; every later wave again uses
/// the original `None` continuation. The adapter cannot renew the phase fence.
pub trait TrajectoryAdapter<Action> {
    /// Absent support keeps any initial waiting cohort unresolved. A supplied
    /// plan is shared across actions and determines legal members once per
    /// wave; completion in an earlier execution group cannot admit into a
    /// later group of that same wave.
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        None
    }

    fn continuation_phase(
        &self,
        _action: &Action,
        _requests: &[SimulatedRequest],
        _phase: &mut PhaseBuilder<'_>,
    ) -> Result<(), Unsupported> {
        Ok(())
    }

    /// Active-phase work uses explicit core-owned state. Declaring a phase
    /// without implementing its subsequent waves fails closed.
    fn plan_phase(
        &self,
        _action: &Action,
        _phase: &ContinuationPhase,
        _requests: &[SimulatedRequest],
        _plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        Err(Unsupported::InvalidPlan)
    }

    fn plan(
        &self,
        first_action: Option<&Action>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported>;
}

pub struct EvaluationInput<'a, Action, Adapter> {
    pub snapshot: &'a dyn CostSnapshot,
    pub now: Instant,
    pub config: &'a SchedulerSloConfig,
    pub execution: CostExecution,
    pub requests: &'a [SimulatedRequest],
    pub resources: ResourceSupport,
    pub adapter: &'a Adapter,
    pub static_action: &'a Action,
    pub alternatives: &'a [Action],
    pub horizon: EvaluationHorizon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationHorizon {
    /// The static trajectory fixes the common window at this wave's final
    /// invocation boundary, or earlier when the cohort completes.
    StaticRounds(usize),
    /// Targets are the full, assigned endpoints of this scenario. This is not
    /// a latency SLO and does not relax the global query or elapsed budgets.
    /// Only a scenario evaluation with a known seed distribution may retain a
    /// later static support prefix; an unseeded evaluation still returns Err.
    ScenarioCompletion,
}

#[derive(Debug, Clone)]
pub struct TrajectoryResult {
    pub requests: Vec<SimulatedRequest>,
    pub window_progress: Vec<Progress>,
    pub target_reached_at: Duration,
    pub central_elapsed: Duration,
    pub risk_elapsed: Duration,
    pub queries: usize,
    pub cost_risk: CostRiskBudget,
}

/// Union accounting for the invocation risk costs used by this trajectory.
/// Empirical group quantiles are assumptions, not certified probabilities.
/// Engineering margins carry unresolved mass one, never an invented quantile.
#[derive(Debug, Clone, Copy, Default)]
pub struct CostRiskBudget {
    pub union_failure_upper: f64,
    pub empirical_quantiles: bool,
    pub conditional_scenario: Option<CostScenarioDependence>,
}

impl TrajectoryResult {
    /// Final endpoint evidence, independent of an earlier Exact window's
    /// progress. A token frontier without its completion clock is insufficient.
    fn has_completed_endpoints(&self) -> bool {
        !self.requests.is_empty()
            && self.requests.iter().all(|request| {
                request.prefill_tokens == request.prompt_tokens
                    && request.output_tokens == request.scenario_end_tokens
                    && request.central_clock.completed.is_some()
            })
    }

    pub fn has_window_debt(&self, targets: &[Progress]) -> bool {
        self.window_progress.len() != targets.len()
            || self
                .window_progress
                .iter()
                .zip(targets)
                .any(|(progress, target)| !progress.covers(*target))
    }
}

#[derive(Debug)]
pub struct Comparison {
    pub revision: u64,
    pub window: StaticWindow,
    pub targets: Vec<Progress>,
    pub static_result: StaticTrajectory,
    pub alternatives: Vec<Result<TrajectoryResult, Unsupported>>,
}

/// An observed prefix is a lower bound on the unknown full static window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticWindow {
    Exact(Duration),
    AtLeast(Duration),
}

#[derive(Debug)]
pub enum StaticTrajectory {
    Complete(TrajectoryResult),
    OutsideSupportPrefix(TrajectoryPrefix),
}

/// Only successfully priced invocation groups have been applied. The rejected
/// group contributes neither progress nor elapsed time. No completion/target
/// time is invented for the unknown suffix.
#[derive(Debug)]
pub struct TrajectoryPrefix {
    pub requests: Vec<SimulatedRequest>,
    pub central_elapsed: Duration,
    pub risk_elapsed: Duration,
    pub queries: usize,
    pub cost_risk: CostRiskBudget,
    pub unsupported: Unsupported,
}

impl StaticTrajectory {
    pub fn complete(&self) -> Option<&TrajectoryResult> {
        match self {
            Self::Complete(result) => Some(result),
            Self::OutsideSupportPrefix(_) => None,
        }
    }

    pub fn requests(&self) -> &[SimulatedRequest] {
        match self {
            Self::Complete(result) => &result.requests,
            Self::OutsideSupportPrefix(prefix) => &prefix.requests,
        }
    }

    fn cost_risk(&self) -> &CostRiskBudget {
        match self {
            Self::Complete(result) => &result.cost_risk,
            Self::OutsideSupportPrefix(prefix) => &prefix.cost_risk,
        }
    }
}

impl Comparison {
    fn partial_endpoints_covered(&self, result: &TrajectoryResult) -> bool {
        match self.window {
            StaticWindow::Exact(_) => true,
            // An AtLeast alternative has no window probe: its window_progress
            // is its final endpoint progress, not progress at the prefix time.
            StaticWindow::AtLeast(_) => {
                result.has_completed_endpoints() && !result.has_window_debt(&self.targets)
            }
        }
    }

    fn partial_window_covered(&self, result: &TrajectoryResult) -> bool {
        match self.window {
            StaticWindow::Exact(_) => true,
            StaticWindow::AtLeast(lower) => {
                self.partial_endpoints_covered(result) && result.central_elapsed <= lower
            }
        }
    }
}

/// Paired action-difference uncertainty comes from independent measurements;
/// marginal model residual spread or scenario ranks cannot fill this field.
#[derive(Debug, Clone, Copy)]
pub struct PairedActionEvidence {
    pub uncertainty: Duration,
    pub control_cost: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyChoice {
    Static,
    Alternative(usize),
    AllKnownInfeasible,
    Unresolved,
}

pub struct SloPolicy;

impl SloPolicy {
    pub fn evaluate<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
    ) -> Result<Comparison, Unsupported> {
        let diagnostics = diagnostics::Recorder::new(input.config.planning.max_queries);
        Self::evaluate_recorded(input, &diagnostics, None)
    }

    fn evaluate_recorded<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
        diagnostics: &diagnostics::Recorder,
        rank: Option<usize>,
    ) -> Result<Comparison, Unsupported> {
        if input.requests.is_empty() {
            return Err(Unsupported::EmptyCohort);
        }
        if input.requests.len() > input.config.planning.max_participants {
            return Err(Unsupported::ParticipantBudget);
        }
        if input.alternatives.len().saturating_add(1) > input.config.planning.max_candidates {
            return Err(Unsupported::CandidateBudget);
        }
        admission::validate_admission(&input)?;
        if !input.resources.kv_reserved_through_scenario_end {
            return Err(Unsupported::UnknownKvCapacity);
        }
        if input.horizon == EvaluationHorizon::StaticRounds(0)
            || input.requests.iter().enumerate().any(|(index, request)| {
                !request.valid()
                    || input.requests[..index]
                        .iter()
                        .any(|prior| prior.id == request.id)
            })
        {
            return Err(Unsupported::InvalidState);
        }
        let started = Instant::now();
        let mut budget = QueryBudget {
            queries: 0,
            started,
            diagnostics,
            rank,
        };
        // Freeze full endpoint targets before any potentially partial static
        // traversal. A support gap must never shorten an alternative's goal.
        let endpoint_targets =
            (input.horizon == EvaluationHorizon::ScenarioCompletion).then(|| {
                input
                    .requests
                    .iter()
                    .map(|request| Progress {
                        prefill_tokens: request.prompt_tokens,
                        output_tokens: request.scenario_end_tokens,
                    })
                    .collect::<Vec<_>>()
            });
        let static_result = simulate(&input, input.static_action, None, &mut budget, 0)?;
        let window = match &static_result {
            StaticTrajectory::Complete(result) => StaticWindow::Exact(result.central_elapsed),
            StaticTrajectory::OutsideSupportPrefix(prefix) => {
                StaticWindow::AtLeast(prefix.central_elapsed)
            }
        };
        let targets = endpoint_targets.unwrap_or_else(|| {
            static_result
                .requests()
                .iter()
                .map(SimulatedRequest::progress)
                .collect()
        });
        let mut alternatives = Vec::with_capacity(input.alternatives.len());
        for (index, action) in input.alternatives.iter().enumerate() {
            let result = simulate(
                &input,
                action,
                match window {
                    StaticWindow::Exact(window) => Some((window, targets.as_slice())),
                    StaticWindow::AtLeast(_) => None,
                },
                &mut budget,
                index + 1,
            )
            .and_then(|result| match result {
                StaticTrajectory::Complete(result) => Ok(result),
                StaticTrajectory::OutsideSupportPrefix(prefix) => Err(prefix.unsupported),
            });
            if let Err(reason) = &result {
                if reason.is_hard_budget_failure() {
                    return Err(*reason);
                }
            }
            alternatives.push(result);
        }
        if matches!(static_result, StaticTrajectory::OutsideSupportPrefix(_)) {
            if let Some(reason) = alternatives
                .iter()
                .filter_map(|result| result.as_ref().err())
                .find(|reason| {
                    **reason
                        != Unsupported::Cost(CostUnknownReason::UnsupportedWithDetail(
                            CostUnsupportedDetail::OutsideObservedSupport,
                        ))
                })
            {
                return Err(*reason);
            }
        }
        Ok(Comparison {
            revision: input.snapshot.revision(),
            window,
            targets,
            static_result,
            alternatives,
        })
    }

    pub fn choose(
        comparison: &Comparison,
        static_risk: JointFeasibility,
        alternative_risks: &[JointFeasibility],
        paired: &[Option<PairedActionEvidence>],
    ) -> PolicyChoice {
        if alternative_risks.len() != comparison.alternatives.len()
            || paired.len() != comparison.alternatives.len()
        {
            return PolicyChoice::Unresolved;
        }
        let mut best: Option<(usize, Duration)> = None;
        for (index, result) in comparison.alternatives.iter().enumerate() {
            let Ok(result) = result else {
                continue;
            };
            if alternative_risks[index] != JointFeasibility::Feasible {
                continue;
            }
            if !comparison.partial_endpoints_covered(result) {
                continue;
            }
            if static_risk != JointFeasibility::Infeasible
                && !comparison.partial_window_covered(result)
            {
                continue;
            }
            if static_risk == JointFeasibility::Feasible {
                if result.has_window_debt(&comparison.targets) {
                    continue;
                }
                let Some(evidence) = paired[index] else {
                    continue;
                };
                let Some(static_result) = comparison.static_result.complete() else {
                    continue;
                };
                let Some(gain) = static_result
                    .target_reached_at
                    .checked_sub(result.target_reached_at)
                else {
                    continue;
                };
                if evidence
                    .uncertainty
                    .checked_add(evidence.control_cost)
                    .is_none_or(|cost| gain <= cost)
                {
                    continue;
                }
            } else if static_risk == JointFeasibility::Unresolved {
                continue;
            }
            if best.is_none_or(|(_, time)| result.target_reached_at < time) {
                best = Some((index, result.target_reached_at));
            }
        }
        if let Some((index, _)) = best {
            return PolicyChoice::Alternative(index);
        }
        if static_risk == JointFeasibility::Feasible
            && comparison.static_result.complete().is_some()
        {
            return PolicyChoice::Static;
        }
        if static_risk == JointFeasibility::Infeasible
            && comparison.static_result.complete().is_some()
            && comparison.alternatives.iter().all(Result::is_ok)
            && alternative_risks
                .iter()
                .all(|risk| *risk == JointFeasibility::Infeasible)
        {
            PolicyChoice::AllKnownInfeasible
        } else {
            PolicyChoice::Unresolved
        }
    }
}

struct QueryBudget<'a> {
    queries: usize,
    started: Instant,
    diagnostics: &'a diagnostics::Recorder,
    rank: Option<usize>,
}

fn simulate<Action, Adapter: TrajectoryAdapter<Action>>(
    input: &EvaluationInput<'_, Action, Adapter>,
    action: &Action,
    probe: Option<(Duration, &[Progress])>,
    budget: &mut QueryBudget<'_>,
    candidate: usize,
) -> Result<StaticTrajectory, Unsupported> {
    let failures = budget.diagnostics.failures();
    let result = simulate_recorded(input, action, probe, budget, candidate);
    if let Err(reason) = &result {
        budget
            .diagnostics
            .fallback_failure(failures, *reason, FailureOrigin::Simulation);
    } else if let Ok(StaticTrajectory::OutsideSupportPrefix(prefix)) = &result {
        budget.diagnostics.fallback_failure(
            failures,
            prefix.unsupported,
            FailureOrigin::Simulation,
        );
    }
    result
}

fn simulate_recorded<Action, Adapter: TrajectoryAdapter<Action>>(
    input: &EvaluationInput<'_, Action, Adapter>,
    action: &Action,
    probe: Option<(Duration, &[Progress])>,
    budget: &mut QueryBudget<'_>,
    candidate: usize,
) -> Result<StaticTrajectory, Unsupported> {
    let mut requests = input.requests.to_vec();
    let mut central = Duration::ZERO;
    let mut risk = Duration::ZERO;
    let mut window_progress = requests
        .iter()
        .map(SimulatedRequest::progress)
        .collect::<Vec<_>>();
    let mut reached = None;
    let query_start = budget.queries;
    let mut rounds = 0usize;
    let mut phase: Option<phase::PhaseState> = None;
    let mut cost_risk = CostRiskBudget::default();
    let itl_limit = Duration::try_from_secs_f64(input.config.itl_ms / 1000.0)
        .map_err(|_| Unsupported::InvalidState)?;
    loop {
        budget.diagnostics.point(
            EvaluationPoint {
                stage: EvaluationStage::Simulation,
                rank: budget.rank,
                candidate: Some(candidate),
                round: Some(rounds),
                group: None,
            },
            budget.queries,
        );
        if budget.queries >= input.config.planning.max_queries {
            budget.diagnostics.failure(
                Unsupported::QueryBudget,
                FailureOrigin::SimulationQueryBudget,
            );
            return Err(Unsupported::QueryBudget);
        }
        if budget.started.elapsed() >= Duration::from_micros(input.config.planning.max_elapsed_us) {
            budget
                .diagnostics
                .failure(Unsupported::TimeBudget, FailureOrigin::SimulationTime);
            return Err(Unsupported::TimeBudget);
        }
        let mut plan = PlanBuilder {
            groups: Vec::new(),
            participant_limit: input.config.planning.max_participants,
            query_limits: input.config.query.intersect(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    input.snapshot.query_limits()
                }))
                .map_err(|_| {
                    let reason = Unsupported::Cost(CostUnknownReason::UnsupportedDomain);
                    budget
                        .diagnostics
                        .failure(reason, FailureOrigin::ProviderPanic);
                    reason
                })?,
            ),
            execution: input.execution,
            active: input
                .adapter
                .admission_slots()
                .map(|slots| slots.active_mask(&requests))
                .transpose()?,
        };
        if rounds == 0 {
            let mut declaration = PhaseBuilder::new(
                &requests,
                input.config.planning.max_participants,
                input.config.planning.max_request_fact_bytes,
            );
            input
                .adapter
                .continuation_phase(action, &requests, &mut declaration)?;
            phase = declaration
                .finish()
                .map(|phase| phase::PhaseState::new(phase, &requests))
                .transpose()?;
            input.adapter.plan(Some(action), &requests, &mut plan)?;
        } else if let Some(phase) = &phase {
            input
                .adapter
                .plan_phase(action, phase.phase(), &requests, &mut plan)?;
        } else {
            input.adapter.plan(None, &requests, &mut plan)?;
        }
        if plan.groups.is_empty() {
            return Err(Unsupported::NoProgress);
        }
        for (group_index, group) in plan.groups.into_iter().enumerate() {
            budget.diagnostics.point(
                EvaluationPoint {
                    stage: EvaluationStage::Simulation,
                    rank: budget.rank,
                    candidate: Some(candidate),
                    round: Some(rounds),
                    group: Some(group_index),
                },
                budget.queries,
            );
            if budget.queries >= input.config.planning.max_queries {
                budget.diagnostics.failure(
                    Unsupported::QueryBudget,
                    FailureOrigin::SimulationQueryBudget,
                );
                return Err(Unsupported::QueryBudget);
            }
            for member in &group {
                validate_member(&requests, *member)?;
            }
            let work = group.iter().map(|member| member.work).collect::<Vec<_>>();
            budget.queries += 1;
            budget.diagnostics.simulation_query(budget.queries);
            let prediction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                input.snapshot.predict(
                    &CostQuery {
                        work: &work,
                        work_is_exact: true,
                        execution: CostExecution::SingleInvocation,
                    },
                    input.now,
                )
            }))
            .map_err(|_| {
                let reason = Unsupported::Cost(CostUnknownReason::UnsupportedDomain);
                budget
                    .diagnostics
                    .failure(reason, FailureOrigin::ProviderPanic);
                reason
            })?;
            let CostPrediction::Known {
                central_service_time,
                upper_service_time,
                observed_at,
                expires_at,
                samples,
                risk: group_risk,
                ..
            } = prediction
            else {
                let CostPrediction::Unknown(reason) = prediction else {
                    unreachable!()
                };
                if candidate == 0
                    && budget.rank.is_some()
                    && input.horizon == EvaluationHorizon::ScenarioCompletion
                    && reason
                        == CostUnknownReason::UnsupportedWithDetail(
                            CostUnsupportedDetail::OutsideObservedSupport,
                        )
                    && !central.is_zero()
                {
                    if budget.started.elapsed()
                        >= Duration::from_micros(input.config.planning.max_elapsed_us)
                    {
                        budget
                            .diagnostics
                            .failure(Unsupported::TimeBudget, FailureOrigin::PostQueryTime);
                        return Err(Unsupported::TimeBudget);
                    }
                    return Ok(StaticTrajectory::OutsideSupportPrefix(TrajectoryPrefix {
                        requests,
                        central_elapsed: central,
                        risk_elapsed: risk,
                        queries: budget.queries - query_start,
                        cost_risk,
                        unsupported: Unsupported::Cost(reason),
                    }));
                }
                return Err(Unsupported::Cost(reason));
            };
            if observed_at > input.now
                || expires_at <= input.now
                || samples == 0
                || central_service_time.is_zero()
                || upper_service_time.is_zero()
            {
                let reason = Unsupported::Cost(CostUnknownReason::UnsupportedWithDetail(
                    CostUnsupportedDetail::InvalidScenarioDistribution,
                ));
                budget
                    .diagnostics
                    .failure(reason, FailureOrigin::ScenarioContract);
                return Err(reason);
            }
            match group_risk {
                CostRisk::EngineeringMargin => cost_risk.union_failure_upper = 1.0,
                CostRisk::ConditionalScenario { dependence } => {
                    cost_risk.conditional_scenario = Some(dependence);
                }
                CostRisk::CalibratedQuantile { quantile, .. } => {
                    if !quantile.is_finite() || quantile <= 0.0 || quantile >= 1.0 {
                        let reason = Unsupported::Cost(CostUnknownReason::UnsupportedWithDetail(
                            CostUnsupportedDetail::InvalidScenarioDistribution,
                        ));
                        budget
                            .diagnostics
                            .failure(reason, FailureOrigin::ScenarioContract);
                        return Err(reason);
                    }
                    cost_risk.union_failure_upper =
                        (cost_risk.union_failure_upper + 1.0 - quantile).min(1.0);
                    cost_risk.empirical_quantiles = true;
                }
            }
            if budget.started.elapsed()
                >= Duration::from_micros(input.config.planning.max_elapsed_us)
            {
                budget
                    .diagnostics
                    .failure(Unsupported::TimeBudget, FailureOrigin::PostQueryTime);
                return Err(Unsupported::TimeBudget);
            }
            central = central
                .checked_add(central_service_time)
                .ok_or(Unsupported::Arithmetic)?;
            risk = risk
                .checked_add(upper_service_time)
                .ok_or(Unsupported::Arithmetic)?;
            for member in group {
                apply_member(
                    &mut requests[member.request],
                    member.work,
                    central,
                    risk,
                    itl_limit,
                )?;
            }
            if let Some((window, targets)) = probe {
                if central <= window {
                    window_progress = requests.iter().map(SimulatedRequest::progress).collect();
                }
                if reached.is_none()
                    && requests
                        .iter()
                        .zip(targets)
                        .all(|(request, target)| request.progress().covers(*target))
                {
                    reached = Some(central);
                }
                // Continue normal scheduling after early target attainment until
                // the common window has also been observed. No paused winners.
                if central >= window && reached.is_some() {
                    if let Some(phase) = &phase {
                        phase.progressed(&requests)?;
                    }
                    return Ok(StaticTrajectory::Complete(TrajectoryResult {
                        requests,
                        window_progress,
                        target_reached_at: reached.unwrap(),
                        central_elapsed: central,
                        risk_elapsed: risk,
                        queries: budget.queries - query_start,
                        cost_risk,
                    }));
                }
            }
        }
        if phase
            .as_mut()
            .map(|phase| phase.advance(&requests))
            .transpose()?
            .unwrap_or(false)
        {
            phase = None;
        }
        rounds += 1;
        if probe.is_none()
            && (matches!(input.horizon, EvaluationHorizon::StaticRounds(limit) if rounds >= limit)
                || requests.iter().all(SimulatedRequest::finished))
        {
            window_progress = requests.iter().map(SimulatedRequest::progress).collect();
            return Ok(StaticTrajectory::Complete(TrajectoryResult {
                requests,
                window_progress,
                target_reached_at: central,
                central_elapsed: central,
                risk_elapsed: risk,
                queries: budget.queries - query_start,
                cost_risk,
            }));
        }
        if requests.iter().all(SimulatedRequest::finished) {
            if let Some((_, targets)) = probe {
                if requests
                    .iter()
                    .zip(targets)
                    .all(|(request, target)| request.progress().covers(*target))
                {
                    window_progress = requests.iter().map(SimulatedRequest::progress).collect();
                    return Ok(StaticTrajectory::Complete(TrajectoryResult {
                        requests,
                        window_progress,
                        target_reached_at: reached.unwrap_or(central),
                        central_elapsed: central,
                        risk_elapsed: risk,
                        queries: budget.queries - query_start,
                        cost_risk,
                    }));
                }
            }
            return Err(Unsupported::NoProgress);
        }
    }
}

fn validate_member(requests: &[SimulatedRequest], member: GroupMember) -> Result<(), Unsupported> {
    let request = requests
        .get(member.request)
        .ok_or(Unsupported::InvalidPlan)?;
    if request.finished() {
        return Err(Unsupported::InvalidPlan);
    }
    let work = member.work;
    let valid = match work.kind {
        AttemptWorkKind::Prefill => {
            request.output_tokens == 0
                && work.context_tokens == request.prompt_tokens
                && work.requested.start() == request.prefill_tokens
                && work.requested.end() <= request.prompt_tokens
        }
        AttemptWorkKind::Decode => {
            request.prefill_tokens == request.prompt_tokens
                && request.output_tokens > 0
                && request
                    .prompt_tokens
                    .checked_add(request.output_tokens)
                    .is_some_and(|context| {
                        work.context_tokens == context
                            && work.requested.start() == context - 1
                            && work.requested.end() == context
                    })
        }
    };
    if valid {
        Ok(())
    } else {
        Err(Unsupported::InvalidPlan)
    }
}

fn apply_member(
    request: &mut SimulatedRequest,
    work: CostWork,
    central: Duration,
    risk: Duration,
    itl_limit: Duration,
) -> Result<(), Unsupported> {
    let emitted = match work.kind {
        AttemptWorkKind::Prefill => {
            request.prefill_tokens = work.requested.end();
            request.prefill_tokens == request.prompt_tokens
        }
        AttemptWorkKind::Decode => true,
    };
    if emitted {
        request.output_tokens = request
            .output_tokens
            .checked_add(1)
            .ok_or(Unsupported::Arithmetic)?;
        for (clock, elapsed) in [
            (&mut request.central_clock, central),
            (&mut request.risk_clock, risk),
        ] {
            let at = request
                .age
                .checked_add(elapsed)
                .ok_or(Unsupported::Arithmetic)?;
            if let Some(previous) = clock.last_output {
                let interval = at.checked_sub(previous).ok_or(Unsupported::InvalidState)?;
                clock.maximum_output_interval = clock.maximum_output_interval.max(interval);
                // Explicit future approximation: one emitted token produces
                // one visible update. Historical fields use delivered text
                // events; real client SSE coalescing is validated separately.
                clock.interval_count = clock
                    .interval_count
                    .checked_add(1)
                    .ok_or(Unsupported::Arithmetic)?;
                if interval > itl_limit {
                    clock.violating_interval_count = clock
                        .violating_interval_count
                        .checked_add(1)
                        .ok_or(Unsupported::Arithmetic)?;
                }
            }
            clock.first_output.get_or_insert(at);
            clock.last_output = Some(at);
            if request.output_tokens == request.scenario_end_tokens {
                clock.completed = Some(at);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
