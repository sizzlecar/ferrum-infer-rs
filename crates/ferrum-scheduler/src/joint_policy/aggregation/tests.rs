use super::*;

fn config() -> SchedulerSloConfig {
    "ttft:10,tpot:15,itl:15".parse().unwrap()
}

fn request(cap: usize) -> SimulatedRequest {
    SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 1,
        prefill_tokens: 0,
        output_tokens: 0,
        output_cap: cap,
        scenario_end_tokens: cap,
        age: Duration::ZERO,
        central_clock: RequestClock::default(),
        risk_clock: RequestClock::default(),
    }
}

fn complete(
    initial: &SimulatedRequest,
    first: u64,
    last: u64,
    bad_intervals: usize,
) -> SimulatedRequest {
    let mut request = initial.clone();
    request.prefill_tokens = request.prompt_tokens;
    request.output_tokens = request.scenario_end_tokens;
    request.central_clock = RequestClock {
        first_output: Some(Duration::from_millis(first)),
        last_output: Some(Duration::from_millis(last)),
        completed: Some(Duration::from_millis(last)),
        interval_count: request.output_tokens - 1,
        violating_interval_count: bad_intervals,
        ..Default::default()
    };
    request.risk_clock = request.central_clock;
    request
}

fn trajectory(requests: Vec<SimulatedRequest>, target_ms: u64) -> TrajectoryResult {
    TrajectoryResult {
        window_progress: requests.iter().map(SimulatedRequest::progress).collect(),
        requests,
        target_reached_at: Duration::from_millis(target_ms),
        central_elapsed: Duration::from_millis(target_ms),
        risk_elapsed: Duration::from_millis(target_ms),
        queries: 1,
        cost_risk: CostRiskBudget {
            conditional_scenario: Some(CostScenarioDependence::SharedResidualRank),
            ..Default::default()
        },
    }
}

fn comparison(
    static_result: TrajectoryResult,
    alternatives: Vec<Result<TrajectoryResult, Unsupported>>,
) -> Comparison {
    Comparison {
        revision: 1,
        window: StaticWindow::Exact(static_result.central_elapsed),
        targets: static_result.window_progress.clone(),
        static_result: StaticTrajectory::Complete(static_result),
        alternatives,
    }
}

fn rank(index: usize, count: usize) -> ScenarioRank {
    ScenarioRank {
        index,
        count,
        weight: 1.0 / count as f64,
        quantile_lower: index as f64 / count as f64,
        quantile_upper: (index + 1) as f64 / count as f64,
    }
}

fn summary(ranks: usize, unresolved_ranks: usize) -> ScenarioEvaluation {
    ScenarioEvaluation {
        cache_hits: 0,
        cache_retained_bytes: 0,
        valid_until: Instant::now() + Duration::from_secs(60),
        dependence: CostScenarioDependence::SharedResidualRank,
        ranks,
        queries: ranks,
        resolved_mass: (ranks - unresolved_ranks) as f64 / ranks as f64,
        unresolved_mass: unresolved_ranks as f64 / ranks as f64,
        unresolved_ranks,
        partial_static_ranks: 0,
    }
}

fn success(count: usize) -> Vec<Option<ProbabilityInterval>> {
    vec![
        Some(ProbabilityInterval {
            lower: 1.0,
            upper: 1.0
        });
        count
    ]
}

#[test]
fn aggregation_preserves_request_eligible_request_and_visible_event_denominators() {
    let requests = [request(1), request(5)];
    let mut short = complete(&requests[0], 2, 2, 0);
    // A terminal acknowledgement after the last text event does not inflate TPOT.
    short.central_clock.completed = Some(Duration::from_millis(500));
    let mut long = complete(&requests[1], 20, 60, 1);
    long.central_clock.completed = Some(Duration::from_millis(500));
    let comparison = comparison(trajectory(vec![short, long], 60), vec![]);
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 0, &config(), &success(2)).unwrap();
    aggregate.observe(rank(0, 1), &comparison).unwrap();
    let risk = aggregate.finish(summary(1, 0), &[]).unwrap().static_risk;
    assert_eq!(
        risk.ttft,
        ProbabilityInterval {
            lower: 0.5,
            upper: 0.5
        }
    );
    assert_eq!(
        risk.tpot,
        Some(ProbabilityInterval {
            lower: 0.0,
            upper: 0.0
        })
    );
    assert_eq!(
        risk.itl,
        Some(ProbabilityInterval {
            lower: 0.25,
            upper: 0.25
        })
    );
    assert_eq!(risk.tpot_undefined_mass, 1.0);
    assert_eq!(risk.request_count, 2);
}

#[test]
fn unresolved_rank_retains_its_weight_and_cap_event_bound() {
    let requests = [request(3)];
    let mut endpoint = complete(&requests[0], 20, 30, 0);
    endpoint.scenario_end_tokens = 2;
    endpoint.output_tokens = 2;
    endpoint.central_clock.interval_count = 1;
    let comparison = comparison(trajectory(vec![endpoint], 30), vec![]);
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 0, &config(), &success(1)).unwrap();
    aggregate.observe(rank(0, 2), &comparison).unwrap();
    let decision = aggregate.finish(summary(2, 1), &[]).unwrap();
    assert_eq!(
        decision.static_risk.ttft,
        ProbabilityInterval {
            lower: 0.5,
            upper: 1.0
        }
    );
    assert_eq!(
        decision.static_risk.tpot,
        Some(ProbabilityInterval {
            lower: 0.0,
            upper: 0.5
        })
    );
    assert_eq!(decision.static_risk.itl.unwrap().upper, 2.0 / 3.0);
    assert_eq!(
        decision.static_risk.completion_failure,
        ProbabilityInterval {
            lower: 0.0,
            upper: 0.5
        }
    );
    assert!(decision.static_target_time.is_none());
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
}

#[test]
fn unsupported_alternative_and_absent_product_evidence_cannot_become_feasible() {
    let requests = [request(2)];
    let comparison = comparison(
        trajectory(vec![complete(&requests[0], 1, 10, 0)], 10),
        vec![Err(Unsupported::Cost(CostUnknownReason::NoCoverage))],
    );
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 1, &config(), &success(1)).unwrap();
    aggregate.observe(rank(0, 1), &comparison).unwrap();
    let decision = aggregate.finish(summary(1, 0), &[None]).unwrap();
    assert_eq!(decision.choice, PolicyChoice::Static);
    assert_eq!(
        decision.alternative_risks[0].ttft,
        ProbabilityInterval {
            lower: 0.0,
            upper: 1.0
        }
    );
    assert!(decision.alternative_target_times[0].is_none());
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 1, &config(), &[None]).unwrap();
    aggregate.observe(rank(0, 1), &comparison).unwrap();
    let decision = aggregate.finish(summary(1, 0), &[None]).unwrap();
    assert_eq!(
        decision.static_risk.completion_failure,
        ProbabilityInterval {
            lower: 0.0,
            upper: 1.0
        }
    );
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
}

#[test]
fn feasible_static_gain_requires_independent_evidence_and_per_request_progress() {
    let requests = [request(2)];
    let mut targets = config();
    targets.tpot_ms = 100.0;
    targets.itl_ms = 100.0;
    let comparison = comparison(
        trajectory(vec![complete(&requests[0], 1, 100, 0)], 100),
        vec![Ok(trajectory(vec![complete(&requests[0], 1, 80, 0)], 80))],
    );
    let choose = |paired| {
        let mut aggregate =
            ScenarioRiskAccumulator::new(&requests, 1, &targets, &success(1)).unwrap();
        aggregate.observe(rank(0, 1), &comparison).unwrap();
        aggregate.finish(summary(1, 0), &[paired]).unwrap().choice
    };
    assert_eq!(choose(None), PolicyChoice::Static);
    let evidence = |uncertainty| {
        Some(PairedActionEvidence {
            uncertainty: Duration::from_millis(uncertainty),
            control_cost: Duration::from_millis(1),
        })
    };
    assert_eq!(choose(evidence(19)), PolicyChoice::Static);
    assert_eq!(choose(evidence(18)), PolicyChoice::Alternative(0));
    let mut debt_comparison = comparison;
    debt_comparison.alternatives[0]
        .as_mut()
        .unwrap()
        .window_progress[0]
        .output_tokens -= 1;
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 1, &targets, &success(1)).unwrap();
    aggregate.observe(rank(0, 1), &debt_comparison).unwrap();
    assert_eq!(
        aggregate
            .finish(summary(1, 0), &[evidence(0)])
            .unwrap()
            .choice,
        PolicyChoice::Static
    );
}

#[test]
fn infeasible_static_may_choose_feasible_slower_target_and_all_bad_is_distinct() {
    let requests = [request(3)];
    let bad = trajectory(vec![complete(&requests[0], 20, 30, 0)], 30);
    let mut good = trajectory(vec![complete(&requests[0], 1, 31, 0)], 31);
    good.window_progress[0].output_tokens = 1;
    let comparison = comparison(bad.clone(), vec![Ok(good), Ok(bad.clone())]);
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 2, &config(), &success(1)).unwrap();
    aggregate.observe(rank(0, 1), &comparison).unwrap();
    let decision = aggregate.finish(summary(1, 0), &[None, None]).unwrap();
    assert_eq!(decision.choice, PolicyChoice::Alternative(0));
    assert!(decision.any_window_debt[0]);
    let comparison = self::comparison(bad.clone(), vec![Ok(bad)]);
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 1, &config(), &success(1)).unwrap();
    aggregate.observe(rank(0, 1), &comparison).unwrap();
    assert_eq!(
        aggregate.finish(summary(1, 0), &[None]).unwrap().choice,
        PolicyChoice::AllKnownInfeasible
    );
}

#[test]
fn marginal_costs_and_incomplete_or_repeated_rank_visits_are_rejected() {
    let requests = [request(2)];
    let mut comparison = comparison(
        trajectory(vec![complete(&requests[0], 1, 10, 0)], 10),
        vec![],
    );
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 0, &config(), &success(1)).unwrap();
    let StaticTrajectory::Complete(result) = &mut comparison.static_result else {
        panic!("complete fixture")
    };
    result.cost_risk.conditional_scenario = None;
    assert_eq!(
        aggregate.observe(rank(0, 1), &comparison),
        Err(Unsupported::InvalidState)
    );
    let StaticTrajectory::Complete(result) = &mut comparison.static_result else {
        panic!("complete fixture")
    };
    result.cost_risk.conditional_scenario = Some(CostScenarioDependence::SharedResidualRank);
    aggregate.observe(rank(0, 2), &comparison).unwrap();
    assert_eq!(
        aggregate.observe(rank(0, 2), &comparison),
        Err(Unsupported::InvalidState)
    );
    assert!(matches!(
        aggregate.finish(summary(2, 0), &[]),
        Err(Unsupported::InvalidState)
    ));
}
