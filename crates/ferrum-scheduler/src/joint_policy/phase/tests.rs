use super::*;
use ferrum_execution::{execution_attempt::AttemptPrefixRestore, token_execution::ExecutionRange};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
enum Action {
    Ordinary,
    Phased,
}

struct Snapshot {
    at: Instant,
    calls: AtomicUsize,
    unknown_at: Option<usize>,
    groups: Mutex<Vec<AttemptWorkKind>>,
}

impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        self.groups.lock().unwrap().push(query.work[0].kind);
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.unknown_at == Some(call) {
            return CostPrediction::Unknown(CostUnknownReason::NoCoverage);
        }
        CostPrediction::Known {
            central_service_time: Duration::from_millis(1),
            upper_service_time: Duration::from_millis(1),
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ExactObservedWork,
            samples: 1,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
    fn cost_scenarios(&self, _: &CostQuery<'_>, _: Instant, _: usize) -> CostScenarioPrediction {
        CostScenarioPrediction::Known {
            central_service_time: Duration::from_millis(1),
            scenarios: (0..2)
                .map(|rank| CostScenario {
                    rank,
                    weight: 0.5,
                    service_time: Duration::from_micros(1000 + rank as u64),
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ExactObservedWork,
            samples: 2,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    First,
    Active,
    Common,
}

struct Adapter {
    stages: Mutex<Vec<(Stage, usize, usize)>>,
    declarations: AtomicUsize,
    stall: Option<Stage>,
}

impl Adapter {
    fn new(stall: Option<Stage>) -> Self {
        Self {
            stages: Mutex::new(Vec::new()),
            declarations: AtomicUsize::new(0),
            stall,
        }
    }
    fn wave(
        &self,
        stage: Stage,
        chunk: usize,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.stages.lock().unwrap().push((
            stage,
            requests[0].prefill_tokens,
            requests[1].output_tokens,
        ));
        for kind in [AttemptWorkKind::Prefill, AttemptWorkKind::Decode] {
            let members = requests
                .iter()
                .enumerate()
                .filter(|(_, request)| {
                    !request.finished()
                        && (request.output_tokens == 0) == (kind == AttemptWorkKind::Prefill)
                        && !(self.stall == Some(stage) && kind == AttemptWorkKind::Prefill)
                })
                .map(|(request, state)| {
                    let (range, context) = if kind == AttemptWorkKind::Prefill {
                        (
                            ExecutionRange::new(
                                state.prefill_tokens,
                                (state.prompt_tokens - state.prefill_tokens).min(chunk),
                            )
                            .unwrap(),
                            state.prompt_tokens,
                        )
                    } else {
                        let context = state.prompt_tokens + state.output_tokens;
                        (ExecutionRange::new(context - 1, 1).unwrap(), context)
                    };
                    GroupMember {
                        request,
                        work: CostWork {
                            kind,
                            requested: range,
                            context_tokens: context,
                            prefix_restore: AttemptPrefixRestore::NoneObserved,
                            internally_reused_tokens: None,
                        },
                    }
                })
                .collect::<Vec<_>>();
            if !members.is_empty() {
                plan.push_group(&members)?;
            }
        }
        Ok(())
    }
}

impl TrajectoryAdapter<Action> for Adapter {
    fn continuation_phase(
        &self,
        action: &Action,
        _: &[SimulatedRequest],
        phase: &mut PhaseBuilder<'_>,
    ) -> Result<(), Unsupported> {
        if matches!(action, Action::Phased) {
            self.declarations.fetch_add(1, Ordering::Relaxed);
            phase.prefill_target(0)?;
        }
        Ok(())
    }
    fn plan(
        &self,
        action: Option<&Action>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.wave(
            if action.is_some() {
                Stage::First
            } else {
                Stage::Common
            },
            if action.is_some() { 1 } else { 2 },
            requests,
            plan,
        )
    }
    fn plan_phase(
        &self,
        _: &Action,
        phase: &ContinuationPhase,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        assert_eq!(
            phase.prefill_targets(),
            &[PrefillTarget {
                request: 0,
                prompt_tokens: 3
            }]
        );
        self.wave(Stage::Active, 1, requests, plan)
    }
}

fn requests() -> [SimulatedRequest; 2] {
    let prefill = SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 3,
        prefill_tokens: 0,
        output_tokens: 0,
        output_cap: 3,
        scenario_end_tokens: 3,
        age: Duration::ZERO,
        central_clock: RequestClock::default(),
        risk_clock: RequestClock::default(),
    };
    let clock = RequestClock {
        first_output: Some(Duration::ZERO),
        last_output: Some(Duration::ZERO),
        ..Default::default()
    };
    let decode = SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 1,
        prefill_tokens: 1,
        output_tokens: 1,
        output_cap: 5,
        scenario_end_tokens: 5,
        age: Duration::ZERO,
        central_clock: clock,
        risk_clock: clock,
    };
    [prefill, decode]
}

fn snapshot() -> Snapshot {
    Snapshot {
        at: Instant::now(),
        calls: AtomicUsize::new(0),
        unknown_at: None,
        groups: Mutex::new(Vec::new()),
    }
}

fn config() -> SchedulerSloConfig {
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 2;
    config
}

fn input<'a, A: TrajectoryAdapter<Action>>(
    snapshot: &'a Snapshot,
    config: &'a SchedulerSloConfig,
    requests: &'a [SimulatedRequest],
    adapter: &'a A,
    action: &'a Action,
    alternatives: &'a [Action],
) -> EvaluationInput<'a, Action, A> {
    EvaluationInput {
        snapshot,
        now: snapshot.at,
        config,
        execution: CostExecution::SequentialPrefillThenDecode,
        requests,
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter,
        static_action: action,
        alternatives,
        horizon: EvaluationHorizon::ScenarioCompletion,
    }
}

#[test]
fn explicit_prefill_phase_ends_only_after_real_frontier_then_uses_common_continuation() {
    let snapshot = snapshot();
    let requests = requests();
    let adapter = Adapter::new(None);
    let comparison = SloPolicy::evaluate(input(
        &snapshot,
        &config(),
        &requests,
        &adapter,
        &Action::Phased,
        &[],
    ))
    .unwrap();
    assert!(comparison
        .static_result
        .complete()
        .unwrap()
        .requests
        .iter()
        .all(SimulatedRequest::finished));
    assert_eq!(
        *adapter.stages.lock().unwrap(),
        vec![
            (Stage::First, 0, 1),
            (Stage::Active, 1, 2),
            (Stage::Active, 2, 3),
            (Stage::Common, 3, 4),
            (Stage::Common, 3, 5),
        ]
    );
    // The final prefill fence does not discard the decode invocation already
    // planned for that split wave or switch its stage between groups.
    assert_eq!(
        &snapshot.groups.lock().unwrap()[..6],
        &[
            AttemptWorkKind::Prefill,
            AttemptWorkKind::Decode,
            AttemptWorkKind::Prefill,
            AttemptWorkKind::Decode,
            AttemptWorkKind::Prefill,
            AttemptWorkKind::Decode,
        ]
    );
    assert_eq!(requests[0].prefill_tokens, 0);
}

#[test]
fn explicit_prefill_phase_is_new_for_each_rank_and_candidate() {
    let snapshot = snapshot();
    let adapter = Adapter::new(None);
    let requests = requests();
    let mut visited = 0;
    let result = SloPolicy::evaluate_scenarios(
        input(
            &snapshot,
            &config(),
            &requests,
            &adapter,
            &Action::Phased,
            &[Action::Phased, Action::Phased],
        ),
        |_, _, comparison| {
            visited += 1;
            assert!(comparison.alternatives.iter().all(Result::is_ok));
        },
    )
    .unwrap();
    assert_eq!(result.ranks, 2);
    assert_eq!(visited, 2);
    assert_eq!(adapter.declarations.load(Ordering::Relaxed), 6);
    assert_eq!(
        adapter
            .stages
            .lock()
            .unwrap()
            .iter()
            .filter(|(stage, _, _)| *stage == Stage::Active)
            .count(),
        12
    );
}

#[test]
fn explicit_prefill_phase_requires_prefill_progress_even_when_decoders_advance() {
    for stalled in [Stage::First, Stage::Active] {
        let snapshot = snapshot();
        let adapter = Adapter::new(Some(stalled));
        let result = SloPolicy::evaluate(input(
            &snapshot,
            &config(),
            &requests(),
            &adapter,
            &Action::Phased,
            &[],
        ));
        assert_eq!(result.unwrap_err(), Unsupported::NoProgress);
    }
}

#[test]
fn explicit_prefill_phase_keeps_original_budget_and_unknown_exits() {
    for (queries, elapsed, expected) in [
        (1, 1_000_000, Unsupported::QueryBudget),
        (100, 0, Unsupported::TimeBudget),
    ] {
        let snapshot = snapshot();
        let adapter = Adapter::new(None);
        let mut config = config();
        config.planning.max_queries = queries;
        config.planning.max_elapsed_us = elapsed;
        let result = SloPolicy::evaluate(input(
            &snapshot,
            &config,
            &requests(),
            &adapter,
            &Action::Phased,
            &[],
        ));
        assert_eq!(result.unwrap_err(), expected);
        assert_eq!(
            snapshot.calls.load(Ordering::Relaxed),
            usize::from(elapsed > 0)
        );
    }
    let mut snapshot = snapshot();
    snapshot.unknown_at = Some(3);
    let result = SloPolicy::evaluate(input(
        &snapshot,
        &config(),
        &requests(),
        &Adapter::new(None),
        &Action::Phased,
        &[],
    ));
    assert_eq!(
        result.unwrap_err(),
        Unsupported::Cost(CostUnknownReason::NoCoverage)
    );
    assert_eq!(snapshot.calls.load(Ordering::Relaxed), 3);
}

#[test]
fn explicit_prefill_phase_builder_rejects_unbound_completed_duplicate_and_over_budget_targets() {
    let requests = requests();
    let bytes = ContinuationPhase::prefill_targets_bytes_bound(1).unwrap();
    for index in [1, requests.len()] {
        let mut builder = PhaseBuilder::new(&requests, 2, bytes * 2);
        assert_eq!(builder.prefill_target(index), Err(Unsupported::InvalidPlan));
        assert!(builder.finish().is_none());
    }
    let mut builder = PhaseBuilder::new(&requests, 2, bytes * 2);
    builder.prefill_target(0).unwrap();
    assert_eq!(builder.prefill_target(0), Err(Unsupported::InvalidPlan));
    assert_eq!(
        builder.finish().unwrap().prefill_targets(),
        &[PrefillTarget {
            request: 0,
            prompt_tokens: 3
        }]
    );
    for (count, bytes) in [(0, bytes), (1, bytes - 1)] {
        let mut builder = PhaseBuilder::new(&requests, count, bytes);
        assert_eq!(
            builder.prefill_target(0),
            Err(Unsupported::ParticipantBudget)
        );
        assert!(builder.finish().is_none());
    }
    assert!(ContinuationPhase::prefill_targets_bytes_bound(usize::MAX).is_none());
    let mut huge = requests;
    for request in &mut huge {
        request.prompt_tokens = usize::MAX;
        request.prefill_tokens = 0;
        request.output_tokens = 0;
    }
    let mut builder = PhaseBuilder::new(&huge, 2, bytes * 2);
    builder.prefill_target(0).unwrap();
    builder.prefill_target(1).unwrap();
    assert!(matches!(
        PhaseState::new(builder.finish().unwrap(), &huge),
        Err(Unsupported::Arithmetic)
    ));
}

#[test]
fn explicit_prefill_phase_defaults_preserve_first_wave_only_and_reject_unimplemented_active_phase()
{
    struct Original(Adapter);
    impl TrajectoryAdapter<Action> for Original {
        fn plan(
            &self,
            action: Option<&Action>,
            requests: &[SimulatedRequest],
            plan: &mut PlanBuilder,
        ) -> Result<(), Unsupported> {
            self.0.plan(action, requests, plan)
        }
    }
    let snapshot = snapshot();
    let original = Original(Adapter::new(None));
    SloPolicy::evaluate(input(
        &snapshot,
        &config(),
        &requests(),
        &original,
        &Action::Ordinary,
        &[],
    ))
    .unwrap();
    assert!(original
        .0
        .stages
        .lock()
        .unwrap()
        .iter()
        .skip(1)
        .all(|(stage, _, _)| *stage == Stage::Common));

    struct MissingPhase(Original);
    impl TrajectoryAdapter<Action> for MissingPhase {
        fn continuation_phase(
            &self,
            _: &Action,
            _: &[SimulatedRequest],
            phase: &mut PhaseBuilder<'_>,
        ) -> Result<(), Unsupported> {
            phase.prefill_target(0)
        }
        fn plan(
            &self,
            action: Option<&Action>,
            requests: &[SimulatedRequest],
            plan: &mut PlanBuilder,
        ) -> Result<(), Unsupported> {
            self.0.plan(action, requests, plan)
        }
    }
    let adapter = MissingPhase(Original(Adapter::new(None)));
    let result = SloPolicy::evaluate(input(
        &snapshot,
        &config(),
        &requests(),
        &adapter,
        &Action::Phased,
        &[],
    ));
    assert_eq!(result.unwrap_err(), Unsupported::InvalidPlan);
}
