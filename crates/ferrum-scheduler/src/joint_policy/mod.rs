//! Bounded, conditional action comparison. Execution commits are timing proxies;
//! these simulations do not certify client-visible SSE latency or throughput.

mod admission;
mod aggregation;
mod bounded;
mod decision_observation;
mod diagnostics;
mod groups;
mod phase;
mod risk;
mod scenarios;
pub use admission::*;
pub use aggregation::*;
pub(crate) use bounded::BoundedActionEvidence;
pub use bounded::{BoundedAccumulator, BoundedDecision, BoundedTiming, ServiceOpportunityClock};
pub use decision_observation::*;
pub use diagnostics::*;
use groups::GroupView;
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
    /// This candidate did not cover the finite static reference within its
    /// additional wave allowance. Other candidates remain independently valid.
    BoundedHorizonExhausted,
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

    /// Bounded control observes text independently of generated token progress.
    /// An empty decoded piece may leave no text even after several tokens;
    /// terminal projection may also publish several text spans for one token.
    fn valid_bounded(&self) -> bool {
        self.prompt_tokens > 0
            && self.prefill_tokens <= self.prompt_tokens
            && self.scenario_end_tokens > 0
            && self.scenario_end_tokens <= self.output_cap
            && self.output_tokens <= self.scenario_end_tokens
            && (self.output_tokens == 0 || self.prefill_tokens == self.prompt_tokens)
            && [self.central_clock, self.risk_clock].iter().all(|clock| {
                let ordered = match (clock.first_output, clock.last_output) {
                    (None, None) => {
                        clock.interval_count == 0
                            && clock.violating_interval_count == 0
                            && clock.maximum_output_interval.is_zero()
                    }
                    (Some(first), Some(last)) => {
                        first <= last
                            && last <= self.age
                            && clock.maximum_output_interval <= last - first
                            && (clock.interval_count > 0 || first == last)
                    }
                    _ => false,
                };
                ordered
                    && clock.violating_interval_count <= clock.interval_count
                    && clock.completed.is_none_or(|at| {
                        self.finished()
                            && at <= self.age
                            && clock.last_output.is_none_or(|last| last <= at)
                    })
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
    work: Vec<CostWork>,
    requests: Vec<usize>,
    group_ends: [usize; 2],
    members: Vec<GroupMember>,
    group_count: usize,
    participant_limit: usize,
    query_limits: CostQueryLimits,
    execution: CostExecution,
    active: Option<Vec<bool>>,
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

/// Crate-private opt-in for the concrete scheduler's token-only topology.
/// The same immutable adapter, action and complete endpoint vector must produce
/// the same ordered work independent of rank clocks. Public adapters do not
/// acquire this authority by implementing TrajectoryAdapter. Its captured
/// admission slot plan must also remain immutable: only completion of a request
/// can change the active membership between waves.
pub(crate) trait ReplayableTrajectory<Action>: TrajectoryAdapter<Action> {
    /// Read-only observation after canonical groups have passed the same core
    /// validations and the cache guard has been released. Concrete production
    /// adapters need no callback; paired tests use this to record exact work.
    fn record_replay(&self, _plan: &PlanBuilder) {}
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
    /// Static observes at most this many complete waves. Alternatives cover
    /// its per-request progress and elapsed window using natural continuation,
    /// with at most `planning.max_catchup_rounds` additional complete waves.
    /// A cohort that naturally finishes may stop sooner. An exhausted candidate
    /// remains unknown; a successful finite comparison is not a full-tail or
    /// client-visible SLO certificate.
    BoundedRounds(usize),
    /// Targets are the full, assigned endpoints of this scenario. This is not
    /// a latency SLO and does not relax the global query or elapsed budgets.
    /// Only a scenario evaluation with a known seed distribution may retain a
    /// later static support prefix; an unseeded evaluation still returns Err.
    ScenarioCompletion,
}

#[derive(Debug, Clone)]
pub struct TrajectoryResult {
    pub requests: Vec<SimulatedRequest>,
    /// Progress at the common static window when one is probed, otherwise at
    /// this trajectory's own final boundary (including BoundedRounds).
    pub window_progress: Vec<Progress>,
    /// Common-target attainment time when probed; without a probe this is the
    /// local horizon's elapsed time, not evidence of full endpoint attainment.
    pub target_reached_at: Duration,
    pub central_elapsed: Duration,
    pub risk_elapsed: Duration,
    pub queries: usize,
    pub cost_risk: CostRiskBudget,
    /// Present only for BoundedRounds. Request clocks remain the immutable
    /// observed text history; generated service opportunities live here.
    pub bounded_timing: Option<BoundedTiming>,
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
    /// The requested evaluation horizon completed. Finite horizons can retain
    /// unfinished requests; inspect their frontiers and completion clocks.
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
        Self::evaluate_recorded(input, &diagnostics, None, None)
    }

    fn evaluate_recorded<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
        diagnostics: &diagnostics::Recorder,
        rank: Option<usize>,
        mut trace: Option<&mut scenarios::TraceEvaluation<'_, '_>>,
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
        if matches!(
            input.horizon,
            EvaluationHorizon::StaticRounds(0) | EvaluationHorizon::BoundedRounds(0)
        ) || input.requests.iter().enumerate().any(|(index, request)| {
            !(if matches!(input.horizon, EvaluationHorizon::BoundedRounds(_)) {
                request.valid_bounded()
            } else {
                request.valid()
            }) || input.requests[..index]
                .iter()
                .any(|prior| prior.id == request.id)
        }) {
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
        let static_result = simulate(
            &input,
            input.static_action,
            None,
            &mut budget,
            0,
            trace.as_deref_mut(),
        )?;
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
                trace.as_deref_mut(),
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
    trace: Option<&mut scenarios::TraceEvaluation<'_, '_>>,
) -> Result<StaticTrajectory, Unsupported> {
    let failures = budget.diagnostics.failures();
    let result = simulate_recorded(input, action, probe, budget, candidate, trace);
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
    mut trace: Option<&mut scenarios::TraceEvaluation<'_, '_>>,
) -> Result<StaticTrajectory, Unsupported> {
    let bounded_wave_limit = match input.horizon {
        EvaluationHorizon::BoundedRounds(limit) => Some(if probe.is_some() {
            limit
                .checked_add(input.config.planning.max_catchup_rounds)
                .ok_or(Unsupported::Arithmetic)?
        } else {
            limit
        }),
        _ => None,
    };
    // All successful alternatives coexist with static until the comparison is
    // visited. Reserve their complete timing payload against the original
    // participant/fact limit, rather than checking each Vec in isolation.
    let mut bounded_timing = if matches!(input.horizon, EvaluationHorizon::BoundedRounds(_)) {
        let bytes = BoundedTiming::retained_bytes_bound(
            input.requests.len(),
            input.alternatives.len().saturating_add(1),
        )
        .ok_or(Unsupported::ParticipantBudget)?;
        if bytes > input.config.planning.max_request_fact_bytes {
            return Err(Unsupported::ParticipantBudget);
        }
        Some(BoundedTiming::new(input.requests)?)
    } else {
        None
    };
    let mut requests = input.requests.to_vec();
    let mut clock_coupling = ClockCoupling::for_requests(&requests);
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
    // Only the private concrete adapter path promises immutable slot topology.
    // Each rank/action simulation starts with an independently validated mask.
    let reuse_active_mask = trace.is_some();
    let mut active_mask_dirty = true;
    let mut cost_risk = CostRiskBudget::default();
    let itl_limit = Duration::try_from_secs_f64(input.config.itl_ms / 1000.0)
        .map_err(|_| Unsupported::InvalidState)?;
    let mut plan = PlanBuilder {
        work: Vec::new(),
        requests: Vec::new(),
        group_ends: [0; 2],
        members: Vec::new(),
        group_count: 0,
        participant_limit: input.config.planning.max_participants,
        query_limits: input.config.query,
        execution: input.execution,
        active: None,
    };
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
        plan.clear_groups();
        plan.query_limits = input.config.query.intersect(
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
        );
        if let Some(slots) = input.adapter.admission_slots() {
            if !reuse_active_mask || active_mask_dirty {
                slots.active_mask_into(&requests, plan.active.get_or_insert_with(Vec::new))?;
                active_mask_dirty = false;
            }
        } else {
            plan.active = None;
            active_mask_dirty = true;
        }
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
                .map(|phase| {
                    phase::PhaseState::new(phase, &requests, input.adapter.admission_slots())
                })
                .transpose()?;
        }
        if let Some(phase) = &mut phase {
            phase.begin_wave(&requests, plan.active.as_deref())?;
        }
        let phase_kind = scenarios::TracePhase::of(rounds, phase.as_ref().map(|p| p.phase()));
        let replayed = if let Some(trace) = trace.as_deref_mut() {
            trace.try_plan(candidate, rounds, phase_kind, &mut plan)?
        } else {
            false
        };
        if !replayed {
            if rounds == 0 {
                input.adapter.plan(Some(action), &requests, &mut plan)?;
            } else if let Some(phase) = &phase {
                input
                    .adapter
                    .plan_phase(action, phase.phase(), &requests, &mut plan)?;
            } else {
                input.adapter.plan(None, &requests, &mut plan)?;
            }
        }
        if plan.group_count == 0 {
            return Err(Unsupported::NoProgress);
        }
        let mut handles = [None; 2];
        for (group_index, group) in plan.groups().enumerate() {
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
            for member in group.iter() {
                validate_member(&requests, member)?;
            }
            budget.queries += 1;
            budget.diagnostics.simulation_query(budget.queries);
            let prediction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let query = CostQuery {
                    work: group.work,
                    work_is_exact: true,
                    execution: CostExecution::SingleInvocation,
                };
                if let Some(trace) = trace.as_deref_mut() {
                    let (prediction, handle) = trace.predict(
                        candidate,
                        rounds,
                        group_index,
                        phase_kind,
                        replayed,
                        &query,
                        input.now,
                    );
                    handles[group_index] = handle;
                    prediction
                } else {
                    input.snapshot.predict(&query, input.now)
                }
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
            clock_coupling.observe_elapsed(central, risk);
            for member in group.iter() {
                if let Some(timing) = bounded_timing.as_mut() {
                    timing.apply_member(
                        member.request,
                        &mut requests[member.request],
                        member.work,
                        central,
                        risk,
                        itl_limit,
                        clock_coupling,
                    )?;
                } else {
                    apply_member(
                        &mut requests[member.request],
                        member.work,
                        central,
                        risk,
                        itl_limit,
                        clock_coupling,
                    )?;
                }
                if reuse_active_mask && requests[member.request].finished() {
                    // Refresh only at the next wave boundary. A split group's
                    // completion cannot admit a waiter into this same wave.
                    active_mask_dirty = true;
                }
            }
            if let Some((window, targets)) = probe {
                if bounded_wave_limit.is_none() && central <= window {
                    for (progress, request) in window_progress.iter_mut().zip(&requests) {
                        *progress = request.progress();
                    }
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
                // Legacy Joint comparisons can stop at an invocation boundary.
                // Bounded comparisons must apply and validate the entire wave;
                // otherwise a Split prefill group could omit its decode group.
                if bounded_wave_limit.is_none() && central >= window && reached.is_some() {
                    if let Some(phase) = &phase {
                        phase.progressed(&requests)?;
                    }
                    // A complete wave is publishable only after all its actual
                    // groups applied and the original phase progress guard ran.
                    if group_index + 1 == plan.group_count {
                        if let Some(trace) = trace.as_deref_mut() {
                            trace.publish(
                                candidate,
                                rounds,
                                phase_kind,
                                plan.groups(),
                                handles,
                                replayed,
                            );
                        }
                    }
                    return Ok(StaticTrajectory::Complete(TrajectoryResult {
                        requests,
                        window_progress,
                        target_reached_at: reached.unwrap(),
                        central_elapsed: central,
                        risk_elapsed: risk,
                        queries: budget.queries - query_start,
                        cost_risk,
                        bounded_timing,
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
        if let Some(trace) = trace.as_deref_mut() {
            trace.publish(
                candidate,
                rounds,
                phase_kind,
                plan.groups(),
                handles,
                replayed,
            );
        }
        rounds += 1;
        if let (Some(limit), Some((window, _))) = (bounded_wave_limit, probe) {
            // Reached targets are never paused while waiting for the common
            // wall boundary. Natural endpoint completion is the only early
            // terminal exception; no idle time or future work is invented.
            if reached.is_some()
                && (central >= window || requests.iter().all(SimulatedRequest::finished))
            {
                for (progress, request) in window_progress.iter_mut().zip(&requests) {
                    *progress = request.progress();
                }
                return Ok(StaticTrajectory::Complete(TrajectoryResult {
                    requests,
                    window_progress,
                    target_reached_at: reached.unwrap(),
                    central_elapsed: central,
                    risk_elapsed: risk,
                    queries: budget.queries - query_start,
                    cost_risk,
                    bounded_timing,
                }));
            }
            if rounds >= limit {
                budget.diagnostics.failure(
                    Unsupported::BoundedHorizonExhausted,
                    FailureOrigin::Simulation,
                );
                return Err(Unsupported::BoundedHorizonExhausted);
            }
        }
        if probe.is_none()
            && (matches!(
                input.horizon,
                EvaluationHorizon::StaticRounds(limit) | EvaluationHorizon::BoundedRounds(limit)
                    if rounds >= limit
            ) || requests.iter().all(SimulatedRequest::finished))
        {
            for (progress, request) in window_progress.iter_mut().zip(&requests) {
                *progress = request.progress();
            }
            return Ok(StaticTrajectory::Complete(TrajectoryResult {
                requests,
                window_progress,
                target_reached_at: central,
                central_elapsed: central,
                risk_elapsed: risk,
                queries: budget.queries - query_start,
                cost_risk,
                bounded_timing,
            }));
        }
        if requests.iter().all(SimulatedRequest::finished) {
            if let Some((_, targets)) = probe {
                if requests
                    .iter()
                    .zip(targets)
                    .all(|(request, target)| request.progress().covers(*target))
                {
                    for (progress, request) in window_progress.iter_mut().zip(&requests) {
                        *progress = request.progress();
                    }
                    return Ok(StaticTrajectory::Complete(TrajectoryResult {
                        requests,
                        window_progress,
                        target_reached_at: reached.unwrap_or(central),
                        central_elapsed: central,
                        risk_elapsed: risk,
                        queries: budget.queries - query_start,
                        cost_risk,
                        bounded_timing,
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

/// A simulation-local proof. Unequal elapsed values permanently invalidate it,
/// even if later service times bring the cumulative elapsed values together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClockCoupling {
    Equal,
    Independent,
}

impl ClockCoupling {
    fn for_requests(requests: &[SimulatedRequest]) -> Self {
        if requests
            .iter()
            .all(|request| request.central_clock == request.risk_clock)
        {
            Self::Equal
        } else {
            Self::Independent
        }
    }

    fn observe_elapsed(&mut self, central: Duration, risk: Duration) {
        if central != risk {
            *self = Self::Independent;
        }
    }
}

fn apply_member(
    request: &mut SimulatedRequest,
    work: CostWork,
    central: Duration,
    risk: Duration,
    itl_limit: Duration,
    clock_coupling: ClockCoupling,
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
        let completed = request.output_tokens == request.scenario_end_tokens;
        apply_clock(
            &mut request.central_clock,
            request.age,
            central,
            itl_limit,
            completed,
        )?;
        if clock_coupling == ClockCoupling::Equal {
            // The second update has the same starting fields and arguments.
            // Copy only after the first update succeeds: its error/mutation
            // order remains identical to the independent-clock path.
            request.risk_clock = request.central_clock;
        } else {
            apply_clock(
                &mut request.risk_clock,
                request.age,
                risk,
                itl_limit,
                completed,
            )?;
        }
    }
    Ok(())
}

fn apply_clock(
    clock: &mut RequestClock,
    age: Duration,
    elapsed: Duration,
    itl_limit: Duration,
    completed: bool,
) -> Result<(), Unsupported> {
    let at = age.checked_add(elapsed).ok_or(Unsupported::Arithmetic)?;
    if let Some(previous) = clock.last_output {
        let interval = at.checked_sub(previous).ok_or(Unsupported::InvalidState)?;
        clock.maximum_output_interval = clock.maximum_output_interval.max(interval);
        // Explicit future approximation: one emitted token produces one
        // visible update. Historical fields retain delivered text events.
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
    if completed {
        clock.completed = Some(at);
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "tests/bounded_rounds.rs"]
mod bounded_rounds_tests;
