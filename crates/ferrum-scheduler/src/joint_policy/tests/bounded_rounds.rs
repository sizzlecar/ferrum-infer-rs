use super::*;
#[path = "bounded_rounds/catchup.rs"]
mod catchup;
#[path = "bounded_rounds/clock_semantics.rs"]
mod clock_semantics;
use ferrum_execution::{execution_attempt::AttemptPrefixRestore, token_execution::ExecutionRange};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Snapshot {
    at: Instant,
    calls: Mutex<Vec<Vec<CostWork>>>,
    scenario_calls: AtomicUsize,
    slow_small_prefill: bool,
    unknown_call: Option<usize>,
}

impl Snapshot {
    fn new() -> Self {
        Self {
            at: Instant::now(),
            calls: Mutex::new(Vec::new()),
            scenario_calls: AtomicUsize::new(0),
            slow_small_prefill: false,
            unknown_call: None,
        }
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }

    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        let mut calls = self.calls.lock().unwrap();
        calls.push(query.work.to_vec());
        if self.unknown_call == Some(calls.len()) {
            return CostPrediction::Unknown(CostUnknownReason::NoCoverage);
        }
        let millis = if self.slow_small_prefill
            && query.work.iter().any(|work| {
                work.kind == AttemptWorkKind::Prefill && work.requested.as_range().len() <= 2
            }) {
            10
        } else {
            1
        };
        CostPrediction::Known {
            central_service_time: Duration::from_millis(millis),
            upper_service_time: Duration::from_millis(millis * 2),
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ExactObservedWork,
            samples: 2,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }

    fn cost_scenarios(&self, _: &CostQuery<'_>, _: Instant, _: usize) -> CostScenarioPrediction {
        self.scenario_calls.fetch_add(1, Ordering::Relaxed);
        CostScenarioPrediction::Known {
            central_service_time: Duration::from_millis(1),
            scenarios: vec![
                CostScenario {
                    rank: 0,
                    weight: 0.5,
                    service_time: Duration::from_millis(1),
                },
                CostScenario {
                    rank: 1,
                    weight: 0.5,
                    service_time: Duration::from_millis(2),
                },
            ],
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ExactObservedWork,
            samples: 2,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
}

#[derive(Clone, Copy)]
enum Action {
    Ordinary,
    SmallPrefillPhase,
}

struct Adapter {
    slots: Option<AdmissionSlotPlan>,
    split: bool,
    waves: AtomicUsize,
    phase_waves: AtomicUsize,
}

impl Adapter {
    fn new() -> Self {
        Self {
            slots: None,
            split: false,
            waves: AtomicUsize::new(0),
            phase_waves: AtomicUsize::new(0),
        }
    }

    fn wave(
        &self,
        chunk: usize,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.waves.fetch_add(1, Ordering::Relaxed);
        let active = plan.active_mask();
        let members = requests
            .iter()
            .enumerate()
            .filter(|(index, request)| {
                !request.finished() && active.is_none_or(|mask| mask[*index])
            })
            .map(|(index, request)| {
                let prefill = request.output_tokens == 0;
                let context = if prefill {
                    request.prompt_tokens
                } else {
                    request.prompt_tokens + request.output_tokens
                };
                GroupMember {
                    request: index,
                    work: CostWork {
                        kind: if prefill {
                            AttemptWorkKind::Prefill
                        } else {
                            AttemptWorkKind::Decode
                        },
                        requested: ExecutionRange::new(
                            if prefill {
                                request.prefill_tokens
                            } else {
                                context - 1
                            },
                            if prefill {
                                (request.prompt_tokens - request.prefill_tokens).min(chunk)
                            } else {
                                1
                            },
                        )
                        .unwrap(),
                        context_tokens: context,
                        prefix_restore: AttemptPrefixRestore::NoneObserved,
                        internally_reused_tokens: None,
                    },
                }
            })
            .collect::<Vec<_>>();
        if self.split {
            plan.push_group_kind(&members, AttemptWorkKind::Prefill)?;
            plan.push_group_kind(&members, AttemptWorkKind::Decode)
        } else {
            plan.push_group(&members)
        }
    }
}

impl TrajectoryAdapter<Action> for Adapter {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        self.slots.as_ref()
    }

    fn continuation_phase(
        &self,
        action: &Action,
        _: &[SimulatedRequest],
        phase: &mut PhaseBuilder<'_>,
    ) -> Result<(), Unsupported> {
        if matches!(action, Action::SmallPrefillPhase) {
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
            if matches!(action, Some(Action::SmallPrefillPhase)) {
                2
            } else {
                8
            },
            requests,
            plan,
        )
    }

    fn plan_phase(
        &self,
        action: &Action,
        phase: &ContinuationPhase,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        assert!(matches!(action, Action::SmallPrefillPhase));
        assert_eq!(
            phase.prefill_targets()[0].prompt_tokens,
            requests[0].prompt_tokens
        );
        self.phase_waves.fetch_add(1, Ordering::Relaxed);
        self.wave(2, requests, plan)
    }
}

fn prefill(prompt: usize, cap: usize) -> SimulatedRequest {
    SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: prompt,
        prefill_tokens: 0,
        output_tokens: 0,
        output_cap: cap,
        scenario_end_tokens: cap,
        age: Duration::from_millis(5),
        central_clock: RequestClock::default(),
        risk_clock: RequestClock::default(),
    }
}

fn decode(cap: usize) -> SimulatedRequest {
    let mut request = prefill(1, cap);
    request.prefill_tokens = 1;
    request.output_tokens = 1;
    request.central_clock.first_output = Some(request.age);
    request.central_clock.last_output = Some(request.age);
    request.risk_clock = request.central_clock;
    request
}

fn config() -> SchedulerSloConfig {
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config
}

fn input<'a>(
    snapshot: &'a Snapshot,
    config: &'a SchedulerSloConfig,
    requests: &'a [SimulatedRequest],
    adapter: &'a Adapter,
    alternatives: &'a [Action],
    horizon: EvaluationHorizon,
) -> EvaluationInput<'a, Action, Adapter> {
    EvaluationInput {
        snapshot,
        now: snapshot.at,
        config,
        execution: if adapter.split {
            CostExecution::SequentialPrefillThenDecode
        } else {
            CostExecution::SingleInvocation
        },
        requests,
        resources: ResourceSupport {
            waiting_requests: adapter
                .slots
                .as_ref()
                .map_or(0, AdmissionSlotPlan::waiting_requests),
            kv_reserved_through_scenario_end: true,
        },
        adapter,
        static_action: &Action::Ordinary,
        alternatives,
        horizon,
    }
}

#[test]
fn bounded_rounds_do_not_grow_with_prompt_or_output_endpoint() {
    for requests in [
        vec![prefill(32, 8)],
        vec![prefill(32_000, 8_192)],
        vec![decode(8)],
        vec![decode(8_192)],
    ] {
        let snapshot = Snapshot::new();
        let adapter = Adapter::new();
        let comparison = SloPolicy::evaluate(input(
            &snapshot,
            &config(),
            &requests,
            &adapter,
            &[],
            EvaluationHorizon::BoundedRounds(3),
        ))
        .unwrap();
        let result = comparison.static_result.complete().unwrap();
        assert_eq!(snapshot.call_count(), 3);
        assert_eq!(adapter.waves.load(Ordering::Relaxed), 3);
        assert_eq!(result.queries, 3);
        assert!(!result.has_completed_endpoints());
        assert_eq!(
            result.requests[0].scenario_end_tokens,
            requests[0].scenario_end_tokens
        );
        if requests[0].output_tokens == 0 {
            assert_eq!(result.requests[0].prefill_tokens, 24);
            assert_eq!(result.requests[0].output_tokens, 0);
            assert_eq!(result.requests[0].central_clock, requests[0].central_clock);
        } else {
            assert_eq!(result.requests[0].output_tokens, 4);
            assert_eq!(result.requests[0].central_clock, requests[0].central_clock);
            assert_eq!(
                result.bounded_timing.as_ref().unwrap().requests[0]
                    .central
                    .projection
                    .interval_count,
                3
            );
            assert_eq!(result.requests[0].central_clock.completed, None);
        }
    }
}

#[test]
fn bounded_rounds_cover_static_targets_with_a_finite_unfinished_phase() {
    let requests = [prefill(100, 64)];
    let alternatives = [Action::SmallPrefillPhase];
    for (horizon, expected_alternative_queries) in [
        (EvaluationHorizon::BoundedRounds(2), 8),
        // The legacy common-window contract still pursues static progress.
        (EvaluationHorizon::StaticRounds(2), 8),
    ] {
        let mut snapshot = Snapshot::new();
        snapshot.slow_small_prefill = true;
        let adapter = Adapter::new();
        let comparison = SloPolicy::evaluate(input(
            &snapshot,
            &config(),
            &requests,
            &adapter,
            &alternatives,
            horizon,
        ))
        .unwrap();
        let static_result = comparison.static_result.complete().unwrap();
        let alternative = comparison.alternatives[0].as_ref().unwrap();
        assert_eq!(static_result.requests[0].prefill_tokens, 16);
        assert_eq!(static_result.central_elapsed, Duration::from_millis(2));
        assert_eq!(alternative.queries, expected_alternative_queries);
        assert_eq!(snapshot.call_count(), 2 + expected_alternative_queries);
        assert_eq!(
            adapter.phase_waves.load(Ordering::Relaxed),
            expected_alternative_queries - 1
        );
        assert_eq!(
            alternative.requests[0].prefill_tokens,
            expected_alternative_queries * 2
        );
        assert_eq!(
            alternative.central_elapsed,
            Duration::from_millis(10 * expected_alternative_queries as u64)
        );
        assert!(!alternative.has_completed_endpoints());
        assert_eq!(alternative.requests[0].central_clock.completed, None);
        if matches!(horizon, EvaluationHorizon::BoundedRounds(_)) {
            assert_eq!(
                alternative.window_progress,
                [alternative.requests[0].progress()]
            );
            assert!(!alternative.has_window_debt(&comparison.targets));
        }
    }
}

#[test]
fn bounded_rounds_count_whole_split_waves_and_keep_waiters_and_clocks() {
    let requests = [prefill(32, 64), decode(64), prefill(16, 64)];
    let mut adapter = Adapter::new();
    adapter.split = true;
    adapter.slots = Some(
        AdmissionSlotPlan::new(
            &[
                InitialAdmission::Resident,
                InitialAdmission::Resident,
                InitialAdmission::Waiting { order: 0 },
            ],
            2,
            3,
        )
        .unwrap(),
    );
    let snapshot = Snapshot::new();
    let comparison = SloPolicy::evaluate(input(
        &snapshot,
        &config(),
        &requests,
        &adapter,
        &[],
        EvaluationHorizon::BoundedRounds(2),
    ))
    .unwrap();
    let result = comparison.static_result.complete().unwrap();
    assert_eq!(adapter.waves.load(Ordering::Relaxed), 2);
    assert_eq!(snapshot.call_count(), 4);
    assert_eq!(result.queries, 4);
    assert_eq!(result.central_elapsed, Duration::from_millis(4));
    assert_eq!(result.risk_elapsed, Duration::from_millis(8));
    assert_eq!(result.requests[0].prefill_tokens, 16);
    assert_eq!(result.requests[1].output_tokens, 3);
    assert_eq!(
        result.bounded_timing.as_ref().unwrap().requests[1]
            .central
            .projection
            .last_output,
        Some(Duration::from_millis(9))
    );
    assert_eq!(
        result.bounded_timing.as_ref().unwrap().requests[1]
            .risk
            .projection
            .last_output,
        Some(Duration::from_millis(13))
    );
    assert_eq!(result.requests[1].central_clock, requests[1].central_clock);
    assert_eq!(result.requests[1].risk_clock, requests[1].risk_clock);
    assert_eq!(
        result.bounded_timing.as_ref().unwrap().requests[1]
            .central
            .projection
            .interval_count,
        2
    );
    let waiting = &result.requests[2];
    assert_eq!(waiting.id, requests[2].id);
    assert_eq!(waiting.progress(), requests[2].progress());
    assert_eq!(waiting.age, requests[2].age);
    assert_eq!(waiting.central_clock, requests[2].central_clock);
    assert_eq!(waiting.risk_clock, requests[2].risk_clock);
    assert!(!waiting.finished());
    assert!(!result.has_completed_endpoints());
    assert_eq!(
        result.window_progress,
        result
            .requests
            .iter()
            .map(SimulatedRequest::progress)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        snapshot
            .calls
            .lock()
            .unwrap()
            .iter()
            .map(|group| group[0].kind)
            .collect::<Vec<_>>(),
        [
            AttemptWorkKind::Prefill,
            AttemptWorkKind::Decode,
            AttemptWorkKind::Prefill,
            AttemptWorkKind::Decode
        ]
    );
}

#[test]
fn bounded_rounds_stop_early_at_declared_endpoints_and_keep_query_guards() {
    let requests = [decode(3)];
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let comparison = SloPolicy::evaluate(input(
        &snapshot,
        &config(),
        &requests,
        &adapter,
        &[],
        EvaluationHorizon::BoundedRounds(8),
    ))
    .unwrap();
    let result = comparison.static_result.complete().unwrap();
    assert_eq!(snapshot.call_count(), 2);
    assert!(
        !result.has_completed_endpoints(),
        "a reached generated cap is not an observed text completion"
    );
    assert_eq!(result.requests[0].output_tokens, 3);
    assert_eq!(
        result.bounded_timing.as_ref().unwrap().requests[0]
            .central
            .projection
            .completed,
        Some(Duration::from_millis(7))
    );
    assert_eq!(result.requests[0].central_clock, requests[0].central_clock);

    let requests = [decode(64)];
    let snapshot = Snapshot::new();
    let mut limited = config();
    limited.planning.max_queries = 2;
    assert!(matches!(
        SloPolicy::evaluate(input(
            &snapshot,
            &limited,
            &requests,
            &adapter,
            &[],
            EvaluationHorizon::BoundedRounds(3)
        )),
        Err(Unsupported::QueryBudget)
    ));
    assert_eq!(snapshot.call_count(), 2);

    let mut snapshot = Snapshot::new();
    snapshot.unknown_call = Some(2);
    assert!(matches!(
        SloPolicy::evaluate(input(
            &snapshot,
            &config(),
            &requests,
            &adapter,
            &[],
            EvaluationHorizon::BoundedRounds(3)
        )),
        Err(Unsupported::Cost(CostUnknownReason::NoCoverage))
    ));
    assert_eq!(snapshot.call_count(), 2);
}

#[test]
fn bounded_rounds_zero_is_invalid_before_any_provider_query() {
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    assert!(matches!(
        SloPolicy::evaluate(input(
            &snapshot,
            &config(),
            &[decode(64)],
            &adapter,
            &[],
            EvaluationHorizon::BoundedRounds(0)
        )),
        Err(Unsupported::InvalidState)
    ));
    assert_eq!(snapshot.call_count(), 0);
    assert_eq!(adapter.waves.load(Ordering::Relaxed), 0);
}

#[test]
fn bounded_rounds_apply_independently_to_every_cost_rank_without_endpoint_certification() {
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let mut config = config();
    config.planning.max_cost_cache_bytes = 0;
    let requests = [decode(64)];
    let mut visited = 0;
    let result = SloPolicy::evaluate_scenarios_with_requests(
        input(
            &snapshot,
            &config,
            &requests,
            &adapter,
            &[Action::Ordinary],
            EvaluationHorizon::BoundedRounds(3),
        ),
        |rank, requests| {
            requests[0].scenario_end_tokens = if rank.index == 0 { 8 } else { 64 };
            Ok(())
        },
        |rank, comparison| {
            visited += 1;
            for result in [
                comparison.static_result.complete().unwrap(),
                comparison.alternatives[0].as_ref().unwrap(),
            ] {
                assert_eq!(result.queries, 3);
                assert_eq!(result.requests[0].output_tokens, 4);
                assert_eq!(
                    result.requests[0].scenario_end_tokens,
                    if rank.index == 0 { 8 } else { 64 }
                );
                assert_eq!(result.requests[0].central_clock.completed, None);
                assert!(!result.has_completed_endpoints());
            }
        },
    )
    .unwrap();
    assert_eq!(visited, 2);
    assert_eq!(result.ranks, 2);
    assert_eq!(result.queries, 1 + 2 * 2 * 3);
    assert_eq!(
        snapshot.scenario_calls.load(Ordering::Relaxed),
        result.queries
    );
    // One seed plan, then exactly three waves per candidate and cost rank.
    assert_eq!(adapter.waves.load(Ordering::Relaxed), 1 + 2 * 2 * 3);
}
