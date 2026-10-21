use super::*;
use ferrum_execution::execution_attempt::AttemptPrefixRestore;
use ferrum_execution::token_execution::ExecutionRange;
use ferrum_interfaces::request_observation::{StoppingForecast, StoppingMass};

struct Snapshot {
    at: Instant,
    split: bool,
}
impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        7
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        assert_eq!(query.execution, CostExecution::SingleInvocation);
        let millis = if self.split {
            if query.work[0].kind == AttemptWorkKind::Prefill {
                20
            } else {
                10
            }
        } else if query.work.len() == 2 {
            12
        } else if query
            .work
            .iter()
            .map(|work| work.context_tokens)
            .max()
            .unwrap()
            >= 6
        {
            22
        } else {
            24
        };
        CostPrediction::Known {
            central_service_time: Duration::from_millis(millis),
            upper_service_time: Duration::from_millis(millis),
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ExactObservedWork,
            samples: 3,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(10),
        }
    }
}

#[derive(Clone, Copy)]
enum Action {
    Full,
    ShortFirst,
}
struct Adapter;
impl TrajectoryAdapter<Action> for Adapter {
    fn plan(
        &self,
        action: Option<&Action>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        let count = if matches!(action, Some(Action::ShortFirst)) {
            requests.len() - 1
        } else {
            requests.len()
        };
        let group = requests
            .iter()
            .enumerate()
            .take(count)
            .filter(|(_, request)| !request.finished())
            .map(|(index, request)| GroupMember {
                request: index,
                work: decode(request),
            })
            .collect::<Vec<_>>();
        plan.push_group(&group)
    }
}

fn decode(request: &SimulatedRequest) -> CostWork {
    let context = request.prompt_tokens + request.output_tokens;
    CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(context - 1, 1).unwrap(),
        context_tokens: context,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    }
}

fn request() -> SimulatedRequest {
    let clock = RequestClock {
        first_output: Some(Duration::ZERO),
        last_output: Some(Duration::ZERO),
        ..Default::default()
    };
    SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 1,
        prefill_tokens: 1,
        output_tokens: 1,
        output_cap: 32,
        scenario_end_tokens: 32,
        age: Duration::ZERO,
        central_clock: clock,
        risk_clock: clock,
    }
}

fn config() -> SchedulerSloConfig {
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config
}

fn comparison() -> Comparison {
    SloPolicy::evaluate(EvaluationInput {
        snapshot: &Snapshot {
            at: Instant::now() - Duration::from_millis(1),
            split: false,
        },
        now: Instant::now(),
        config: &config(),
        execution: CostExecution::SingleInvocation,
        requests: &[request(), request(), request()],
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter: &Adapter,
        static_action: &Action::Full,
        alternatives: &[Action::ShortFirst],
        horizon: EvaluationHorizon::StaticRounds(4),
    })
    .unwrap()
}

#[test]
fn same_window_and_target_expose_the_twelve_96_vs_eleven_84_counterexample() {
    let comparison = comparison();
    let alternative = comparison.alternatives[0].as_ref().unwrap();
    assert_eq!(
        comparison.window,
        StaticWindow::Exact(Duration::from_millis(96))
    );
    assert_eq!(
        comparison
            .targets
            .iter()
            .map(|p| p.output_tokens - 1)
            .sum::<usize>(),
        12
    );
    assert_eq!(
        alternative
            .window_progress
            .iter()
            .map(|p| p.output_tokens - 1)
            .sum::<usize>(),
        11
    );
    assert_eq!(alternative.target_reached_at, Duration::from_millis(106));
    assert!(alternative.has_window_debt(&comparison.targets));
    assert_eq!(
        SloPolicy::choose(
            &comparison,
            JointFeasibility::Feasible,
            &[JointFeasibility::Feasible],
            &[Some(PairedActionEvidence {
                uncertainty: Duration::ZERO,
                control_cost: Duration::ZERO
            })]
        ),
        PolicyChoice::Static
    );
}

#[test]
fn per_request_debt_is_not_cancelled_by_another_requests_extra_token() {
    let mut comparison = comparison();
    let alternative = comparison.alternatives[0].as_mut().unwrap();
    alternative.window_progress[0].output_tokens += 1;
    alternative.target_reached_at = Duration::from_millis(80);
    assert_eq!(
        alternative
            .window_progress
            .iter()
            .map(|p| p.output_tokens)
            .sum::<usize>(),
        comparison
            .targets
            .iter()
            .map(|p| p.output_tokens)
            .sum::<usize>()
    );
    assert!(alternative.has_window_debt(&comparison.targets));
    assert_eq!(
        SloPolicy::choose(
            &comparison,
            JointFeasibility::Feasible,
            &[JointFeasibility::Feasible],
            &[Some(PairedActionEvidence {
                uncertainty: Duration::ZERO,
                control_cost: Duration::ZERO
            })]
        ),
        PolicyChoice::Static
    );
}

#[test]
fn feasible_static_requires_paired_evidence_but_infeasible_static_can_trade_throughput() {
    let mut comparison = comparison();
    let alternative = comparison.alternatives[0].as_mut().unwrap();
    alternative.window_progress = comparison.targets.clone();
    alternative.target_reached_at = Duration::from_millis(80);
    assert_eq!(
        SloPolicy::choose(
            &comparison,
            JointFeasibility::Feasible,
            &[JointFeasibility::Feasible],
            &[None]
        ),
        PolicyChoice::Static
    );
    let evidence = |uncertainty| {
        Some(PairedActionEvidence {
            uncertainty: Duration::from_millis(uncertainty),
            control_cost: Duration::from_millis(2),
        })
    };
    assert_eq!(
        SloPolicy::choose(
            &comparison,
            JointFeasibility::Feasible,
            &[JointFeasibility::Feasible],
            &[evidence(14)]
        ),
        PolicyChoice::Static
    );
    assert_eq!(
        SloPolicy::choose(
            &comparison,
            JointFeasibility::Feasible,
            &[JointFeasibility::Feasible],
            &[evidence(10)]
        ),
        PolicyChoice::Alternative(0)
    );
    let slow = comparison.alternatives[0].as_mut().unwrap();
    slow.window_progress[0].output_tokens -= 1;
    slow.target_reached_at = Duration::from_millis(106);
    assert_eq!(
        SloPolicy::choose(
            &comparison,
            JointFeasibility::Infeasible,
            &[JointFeasibility::Feasible],
            &[None]
        ),
        PolicyChoice::Alternative(0)
    );
}

struct SplitAdapter;
impl TrajectoryAdapter<Action> for SplitAdapter {
    fn plan(
        &self,
        _: Option<&Action>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        let prefill = GroupMember {
            request: 0,
            work: CostWork {
                kind: AttemptWorkKind::Prefill,
                requested: ExecutionRange::new(0, 1).unwrap(),
                context_tokens: 1,
                prefix_restore: AttemptPrefixRestore::NoneObserved,
                internally_reused_tokens: None,
            },
        };
        plan.push_group(&[prefill])?;
        plan.push_group(&[GroupMember {
            request: 1,
            work: decode(&requests[1]),
        }])
    }
}

#[test]
fn split_twenty_then_thirty_credits_prefill_at_a_twenty_five_ms_window() {
    let mut prefill = request();
    prefill.prefill_tokens = 0;
    prefill.output_tokens = 0;
    prefill.central_clock = RequestClock::default();
    prefill.risk_clock = RequestClock::default();
    let requests = [prefill, request()];
    let targets = [
        Progress {
            prefill_tokens: 1,
            output_tokens: 1,
        },
        Progress {
            prefill_tokens: 1,
            output_tokens: 2,
        },
    ];
    let snapshot = Snapshot {
        at: Instant::now(),
        split: true,
    };
    let config = config();
    let input = EvaluationInput {
        snapshot: &snapshot,
        now: snapshot.at,
        config: &config,
        execution: CostExecution::SequentialPrefillThenDecode,
        requests: &requests,
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter: &SplitAdapter,
        static_action: &Action::Full,
        alternatives: &[],
        horizon: EvaluationHorizon::StaticRounds(1),
    };
    let result = simulate(
        &input,
        &Action::Full,
        Some((Duration::from_millis(25), &targets)),
        &mut QueryBudget {
            queries: 0,
            started: Instant::now(),
            diagnostics: &diagnostics::Recorder::new(config.planning.max_queries),
            rank: None,
        },
        1,
    )
    .unwrap();
    let result = result.complete().unwrap();
    assert_eq!(result.window_progress[0].output_tokens, 1);
    assert_eq!(result.window_progress[1].output_tokens, 1);
    assert_eq!(result.target_reached_at, Duration::from_millis(30));
    assert_eq!(
        result.requests[0].central_clock.first_output,
        Some(Duration::from_millis(20))
    );
}

#[test]
fn missing_resources_or_exhausted_queries_cannot_report_gain() {
    let snapshot = Snapshot {
        at: Instant::now(),
        split: false,
    };
    let mut config = config();
    config.planning.max_queries = 4;
    let requests = [request(), request(), request()];
    let input = || EvaluationInput {
        snapshot: &snapshot,
        now: snapshot.at,
        config: &config,
        execution: CostExecution::SingleInvocation,
        requests: &requests,
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter: &Adapter,
        static_action: &Action::Full,
        alternatives: &[Action::ShortFirst],
        horizon: EvaluationHorizon::StaticRounds(4),
    };
    let result = SloPolicy::evaluate(input()).unwrap();
    assert!(matches!(
        result.alternatives[0],
        Err(Unsupported::QueryBudget)
    ));
    assert_eq!(
        SloPolicy::choose(
            &result,
            JointFeasibility::Infeasible,
            &[JointFeasibility::Feasible],
            &[None]
        ),
        PolicyChoice::Unresolved
    );
    let mut waiting = input();
    waiting.resources.waiting_requests = 1;
    assert!(matches!(
        SloPolicy::evaluate(waiting),
        Err(Unsupported::WaitingResources)
    ));
    let mut kv = input();
    kv.resources.kv_reserved_through_scenario_end = false;
    assert!(matches!(
        SloPolicy::evaluate(kv),
        Err(Unsupported::UnknownKvCapacity)
    ));
}

fn endpoint(tokens: usize, tpot: bool) -> EndpointRisk {
    EndpointRisk {
        generated_tokens: tokens,
        ttft_violation: false,
        tpot_violation: (tokens >= 2).then_some(tpot),
        itl_violations: 0,
        itl_events: tokens - 1,
        completed_by_deadline: true,
    }
}

#[test]
fn stopping_risk_keeps_eight_and_twenty_five_and_unknown_mass_conservative() {
    let forecast = StoppingForecast {
        stops: vec![
            StoppingMass {
                generated_tokens: 8,
                mass: 0.3,
            },
            StoppingMass {
                generated_tokens: 25,
                mass: 0.4,
            },
        ],
        known_survival_mass: 0.1,
        unresolved_mass: 0.2,
        tpot_undefined_mass: 0.0,
    };
    let risk = queue_risk(&[RequestRisk {
        forecast: &forecast,
        endpoints: &[endpoint(8, false), endpoint(25, true)],
        output_cap: 32,
        dependence: CostStoppingDependence::ConditionalEndpointScenario,
        product_success: Some(ProbabilityInterval {
            lower: 1.0,
            upper: 1.0,
        }),
    }])
    .unwrap();
    assert!((risk.tpot.unwrap().lower - 0.4).abs() < 1e-10);
    assert!((risk.tpot.unwrap().upper - 0.7).abs() < 1e-10);
    assert!((risk.completion_failure.upper - 0.3).abs() < 1e-10);
    assert!(queue_risk(&[RequestRisk {
        forecast: &forecast,
        endpoints: &[endpoint(8, false), endpoint(25, true)],
        output_cap: 32,
        dependence: CostStoppingDependence::Unspecified,
        product_success: Some(ProbabilityInterval {
            lower: 1.0,
            upper: 1.0
        }),
    }])
    .is_none());
}

#[test]
fn queue_denominator_includes_same_cohort_and_short_output_is_undefined() {
    let long = StoppingForecast {
        stops: vec![StoppingMass {
            generated_tokens: 8,
            mass: 1.0,
        }],
        known_survival_mass: 0.0,
        unresolved_mass: 0.0,
        tpot_undefined_mass: 0.0,
    };
    let short = StoppingForecast {
        stops: vec![StoppingMass {
            generated_tokens: 1,
            mass: 1.0,
        }],
        tpot_undefined_mass: 1.0,
        ..long.clone()
    };
    let mut bad = endpoint(8, true);
    bad.ttft_violation = true;
    let risk = queue_risk(&[
        RequestRisk {
            forecast: &long,
            endpoints: &[bad],
            output_cap: 8,
            dependence: CostStoppingDependence::ConditionalEndpointScenario,
            product_success: Some(ProbabilityInterval {
                lower: 1.0,
                upper: 1.0,
            }),
        },
        RequestRisk {
            forecast: &long,
            endpoints: &[endpoint(8, false)],
            output_cap: 8,
            dependence: CostStoppingDependence::ConditionalEndpointScenario,
            product_success: Some(ProbabilityInterval {
                lower: 1.0,
                upper: 1.0,
            }),
        },
        RequestRisk {
            forecast: &short,
            endpoints: &[endpoint(1, false)],
            output_cap: 1,
            dependence: CostStoppingDependence::ConditionalEndpointScenario,
            product_success: Some(ProbabilityInterval {
                lower: 1.0,
                upper: 1.0,
            }),
        },
    ])
    .unwrap();
    assert_eq!(risk.ttft.lower, 1.0 / 3.0);
    assert_eq!(risk.tpot.unwrap().lower, 0.5);
    assert_eq!(risk.tpot_undefined_mass, 1.0);
    assert_eq!(risk.request_count, 3);
    let short_only = queue_risk(&[RequestRisk {
        forecast: &short,
        endpoints: &[endpoint(1, false)],
        output_cap: 1,
        dependence: CostStoppingDependence::ConditionalEndpointScenario,
        product_success: Some(ProbabilityInterval {
            lower: 1.0,
            upper: 1.0,
        }),
    }])
    .unwrap();
    assert!(short_only.tpot.is_none());
    assert!(short_only.itl.is_none());
}

struct ScenarioSnapshot {
    original: Snapshot,
    change_weights: bool,
}

impl CostSnapshot for ScenarioSnapshot {
    fn revision(&self) -> u64 {
        self.original.revision()
    }
    fn query_limits(&self) -> CostQueryLimits {
        self.original.query_limits()
    }
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        self.original.predict(query, now)
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        max_scenarios: usize,
    ) -> CostScenarioPrediction {
        if max_scenarios < 2 {
            return CostScenarioPrediction::Unknown(CostUnknownReason::QueryLimit);
        }
        let CostPrediction::Known {
            central_service_time,
            support,
            samples,
            observed_at,
            expires_at,
            ..
        } = self.predict(query, now)
        else {
            unreachable!()
        };
        let first_weight =
            if self.change_weights && query.work.iter().any(|work| work.context_tokens > 2) {
                0.5
            } else {
                0.25
            };
        CostScenarioPrediction::Known {
            central_service_time,
            scenarios: vec![
                CostScenario {
                    rank: 0,
                    weight: first_weight,
                    service_time: central_service_time / 2,
                },
                CostScenario {
                    rank: 1,
                    weight: 1.0 - first_weight,
                    service_time: central_service_time * 2,
                },
            ],
            dependence: CostScenarioDependence::SharedResidualRank,
            support,
            samples,
            observed_at,
            expires_at,
        }
    }
}

#[test]
fn shared_rank_scenarios_keep_declared_dependence_and_a_global_resource_budget() {
    let mut snapshot = ScenarioSnapshot {
        original: Snapshot {
            at: Instant::now(),
            split: false,
        },
        change_weights: false,
    };
    let mut config = config();
    let requests = [request(), request(), request()];
    let mut outcomes = Vec::new();
    let mut run = |snapshot: &ScenarioSnapshot, config: &SchedulerSloConfig| {
        SloPolicy::evaluate_scenarios(
            EvaluationInput {
                snapshot,
                now: snapshot.original.at,
                config,
                execution: CostExecution::SingleInvocation,
                requests: &requests,
                resources: ResourceSupport {
                    waiting_requests: 0,
                    kv_reserved_through_scenario_end: true,
                },
                adapter: &Adapter,
                static_action: &Action::Full,
                alternatives: &[Action::ShortFirst],
                horizon: EvaluationHorizon::StaticRounds(4),
            },
            |rank, weight, comparison| {
                assert_eq!(
                    comparison
                        .static_result
                        .complete()
                        .unwrap()
                        .cost_risk
                        .conditional_scenario,
                    Some(CostScenarioDependence::SharedResidualRank)
                );
                outcomes.push((rank, weight, comparison.window));
            },
        )
    };
    let summary = run(&snapshot, &config).unwrap();
    assert_eq!(summary.ranks, 2);
    assert_eq!(
        summary.dependence,
        CostScenarioDependence::SharedResidualRank
    );
    snapshot.change_weights = true;
    assert!(run(&snapshot, &config).is_err());
    snapshot.change_weights = false;
    config.planning.max_queries = 5;
    assert!(run(&snapshot, &config).is_err());
    assert_eq!(
        outcomes[0],
        (0, 0.25, StaticWindow::Exact(Duration::from_millis(48)))
    );
    assert_eq!(
        outcomes[1],
        (1, 0.75, StaticWindow::Exact(Duration::from_millis(192)))
    );
}

#[test]
fn engineering_margin_does_not_invent_a_complete_scenario_distribution() {
    let snapshot = Snapshot {
        at: Instant::now(),
        split: false,
    };
    let mut called = false;
    let result = SloPolicy::evaluate_scenarios(
        EvaluationInput {
            snapshot: &snapshot,
            now: snapshot.at,
            config: &config(),
            execution: CostExecution::SingleInvocation,
            requests: &[request()],
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &Adapter,
            static_action: &Action::Full,
            alternatives: &[],
            horizon: EvaluationHorizon::StaticRounds(1),
        },
        |_, _, _| called = true,
    );
    assert!(matches!(
        result,
        Err(Unsupported::Cost(CostUnknownReason::NoCoverage))
    ));
    assert!(!called);
}

#[test]
fn deterministic_stop_cap_does_not_establish_product_success() {
    let forecast = StoppingForecast {
        stops: vec![StoppingMass {
            generated_tokens: 8,
            mass: 1.0,
        }],
        known_survival_mass: 0.0,
        unresolved_mass: 0.0,
        tpot_undefined_mass: 0.0,
    };
    let risk = queue_risk(&[RequestRisk {
        forecast: &forecast,
        endpoints: &[endpoint(8, false)],
        output_cap: 8,
        dependence: CostStoppingDependence::ConditionalEndpointScenario,
        product_success: None,
    }])
    .unwrap();
    assert_eq!(
        risk.completion_failure,
        ProbabilityInterval {
            lower: 0.0,
            upper: 1.0
        }
    );
    assert_eq!(risk.classify(&config()), JointFeasibility::Unresolved);
}

#[test]
fn per_rank_discrete_endpoints_complete_at_eight_and_twenty_five_without_cap_filling() {
    let snapshot = ScenarioSnapshot {
        original: Snapshot {
            at: Instant::now(),
            split: false,
        },
        change_weights: false,
    };
    let config = config();
    let requests = [request()];
    let input = || EvaluationInput {
        snapshot: &snapshot,
        now: snapshot.original.at,
        config: &config,
        execution: CostExecution::SingleInvocation,
        requests: &requests,
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter: &Adapter,
        static_action: &Action::Full,
        alternatives: &[],
        horizon: EvaluationHorizon::StaticRounds(100),
    };
    let mut endpoints = Vec::new();
    let complete = SloPolicy::evaluate_scenarios_with_requests(
        input(),
        |rank, requests| {
            requests[0].scenario_end_tokens = if rank.quantile_upper <= 0.25 { 8 } else { 25 };
            Ok(())
        },
        |rank, comparison| {
            endpoints.push((
                rank.weight,
                comparison.static_result.complete().unwrap().requests[0].output_tokens,
            ));
        },
    )
    .unwrap();
    assert_eq!(endpoints, vec![(0.25, 8), (0.75, 25)]);
    assert_eq!(complete.resolved_mass, 1.0);
    assert_eq!(complete.unresolved_mass, 0.0);
    endpoints.clear();
    let incomplete = SloPolicy::evaluate_scenarios_with_requests(
        input(),
        |rank, requests| {
            if rank.index == 1 {
                return Err(Unsupported::Cost(CostUnknownReason::NoCoverage));
            }
            requests[0].scenario_end_tokens = 8;
            Ok(())
        },
        |rank, comparison| {
            endpoints.push((
                rank.weight,
                comparison.static_result.complete().unwrap().requests[0].output_tokens,
            ))
        },
    )
    .unwrap();
    assert_eq!(endpoints, vec![(0.25, 8)]);
    assert_eq!(incomplete.resolved_mass, 0.25);
    assert_eq!(incomplete.unresolved_mass, 0.75);
    assert_eq!(incomplete.unresolved_ranks, 1);
    assert!(matches!(
        SloPolicy::evaluate_scenarios_with_requests(
            input(),
            |_, requests| {
                requests[0].output_tokens += 1;
                Ok(())
            },
            |_, _| {}
        ),
        Err(Unsupported::InvalidState)
    ));
}

#[test]
fn interval_risk_counts_historical_and_future_events_instead_of_using_only_maximum() {
    let snapshot = Snapshot {
        at: Instant::now(),
        split: false,
    };
    let mut config = config();
    config.itl_ms = 20.0;
    let mut request = request();
    request.output_tokens = 3;
    request.age = Duration::from_millis(100);
    request.central_clock = RequestClock {
        first_output: Some(Duration::ZERO),
        last_output: Some(request.age),
        maximum_output_interval: Duration::from_millis(99),
        interval_count: 2,
        violating_interval_count: 1,
        completed: None,
    };
    request.risk_clock = request.central_clock;
    let comparison = SloPolicy::evaluate(EvaluationInput {
        snapshot: &snapshot,
        now: snapshot.at,
        config: &config,
        execution: CostExecution::SingleInvocation,
        requests: &[request],
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter: &Adapter,
        static_action: &Action::Full,
        alternatives: &[],
        horizon: EvaluationHorizon::StaticRounds(1),
    })
    .unwrap();
    let clock = comparison.static_result.complete().unwrap().requests[0].central_clock;
    assert_eq!(clock.interval_count, 3);
    assert_eq!(clock.violating_interval_count, 2);
    assert_eq!(clock.maximum_output_interval, Duration::from_millis(99));
}
