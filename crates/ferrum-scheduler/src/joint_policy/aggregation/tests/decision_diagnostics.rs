use super::*;

#[test]
fn partial_static_raw_feasible_is_not_reported_as_effectively_feasible() {
    let requests = [request(2)];
    let mut targets = config();
    targets.joint.tpot_quantile = 0.5;
    targets.joint.itl_quantile = 0.5;
    targets.joint.min_completion_rate = 0.5;
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 0, &targets, &success(1)).unwrap();
    aggregate
        .observe(
            rank(0, 2),
            &comparison(trajectory(vec![complete(&requests[0], 1, 2, 0)], 2), vec![]),
        )
        .unwrap();
    let mut prefix_request = requests[0].clone();
    prefix_request.prefill_tokens = 1;
    prefix_request.output_tokens = 1;
    prefix_request.central_clock.first_output = Some(Duration::from_millis(1));
    prefix_request.central_clock.last_output = Some(Duration::from_millis(1));
    let partial = Comparison {
        revision: 1,
        window: StaticWindow::AtLeast(Duration::from_millis(1)),
        targets: vec![Progress {
            prefill_tokens: 1,
            output_tokens: 2,
        }],
        static_result: StaticTrajectory::OutsideSupportPrefix(TrajectoryPrefix {
            requests: vec![prefix_request],
            central_elapsed: Duration::from_millis(1),
            risk_elapsed: Duration::from_millis(1),
            queries: 1,
            cost_risk: CostRiskBudget {
                conditional_scenario: Some(CostScenarioDependence::SharedResidualRank),
                ..Default::default()
            },
            unsupported: Unsupported::Cost(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport,
            )),
        }),
        alternatives: vec![],
    };
    aggregate.observe(rank(1, 2), &partial).unwrap();
    let mut evaluation = summary(2, 0);
    evaluation.partial_static_ranks = 1;
    let decision = aggregate.finish(evaluation, &[]).unwrap();
    assert_eq!(
        decision.static_risk.classify(&targets),
        JointFeasibility::Feasible
    );
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
    assert_eq!(decision.observation.static_raw, JointFeasibility::Feasible);
    assert_eq!(
        decision.observation.static_effective,
        JointFeasibility::Unresolved
    );
    let witness = decision.observation.unresolved.unwrap();
    assert_eq!(witness.partial_static_ranks, 1);
    assert_eq!(witness.static_action.unsupported_trajectory_mass, 0.5);
    assert_eq!(witness.static_action.target_time, None);
}

#[test]
fn zero_alternatives_preserve_undefined_metrics_and_product_success_uncertainty() {
    let requests = [request(1)];
    let comparison = comparison(trajectory(vec![complete(&requests[0], 1, 1, 0)], 1), vec![]);
    for (product_success, unavailable) in [
        (None, 1),
        (
            Some(ProbabilityInterval {
                lower: 0.0,
                upper: 1.0,
            }),
            0,
        ),
    ] {
        let mut aggregate =
            ScenarioRiskAccumulator::new(&requests, 0, &config(), &[product_success]).unwrap();
        aggregate.observe(rank(0, 1), &comparison).unwrap();
        let decision = aggregate.finish(summary(1, 0), &[]).unwrap();
        assert_eq!(decision.choice, PolicyChoice::Unresolved);
        let observed = decision.observation;
        assert_eq!(observed.choice, decision.choice);
        assert_eq!(observed.alternatives_total, 0);
        assert_eq!(observed.static_raw, JointFeasibility::Unresolved);
        assert_eq!(observed.static_effective, JointFeasibility::Unresolved);
        assert_eq!(observed.static_metrics.ttft, MetricClass::Feasible);
        assert_eq!(observed.static_metrics.tpot, MetricClass::Undefined);
        assert_eq!(observed.static_metrics.itl, MetricClass::Undefined);
        assert_eq!(
            observed.static_metrics.completion_failure,
            MetricClass::Unresolved
        );
        let witness = observed.unresolved.unwrap();
        assert_eq!(witness.revision, Some(1));
        assert_eq!(witness.endpoint_unresolved_mass, 0.0);
        assert_eq!(witness.product_success.unavailable_requests, unavailable);
        assert_eq!(witness.product_success.uncertain_requests, 1);
        assert_eq!(witness.product_success.success_lower_sum, 0.0);
        assert_eq!(witness.product_success.success_upper_sum, 1.0);
        assert_eq!(witness.static_action.unfinished_request_mass, 0.0);
        assert_eq!(witness.static_action.unsupported_trajectory_mass, 0.0);
        assert_eq!(witness.static_action.known_target_time_mass, 1.0);
        assert_eq!(
            witness.static_action.target_time,
            Some(Duration::from_millis(1))
        );
        assert_eq!(witness.static_action.denominators.tpot_undefined, 1.0);
        assert_eq!(witness.static_action.denominators.itl_known_events, 0.0);
        assert!(witness.alternatives.iter().all(Option::is_none));
        assert_eq!(witness.details_omitted, 0);
    }
}

#[test]
fn unsupported_trajectory_mass_is_distinct_from_unknown_product_success() {
    let requests = [request(2)];
    let mut late = complete(&requests[0], 1, 2, 0);
    // Timing did complete, but beyond the declared deadline: a known failure.
    late.central_clock.completed = Some(Duration::from_secs(121));
    let comparison = comparison(
        trajectory(vec![late], 121_000),
        vec![Err(Unsupported::Cost(CostUnknownReason::NoCoverage))],
    );
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 1, &config(), &success(1)).unwrap();
    aggregate.observe(rank(0, 1), &comparison).unwrap();
    let decision = aggregate.finish(summary(1, 0), &[None]).unwrap();
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
    assert_eq!(
        decision.observation.static_metrics.completion_failure,
        MetricClass::Infeasible
    );
    let witness = decision.observation.unresolved.unwrap();
    assert_eq!(witness.product_success.unavailable_requests, 0);
    assert_eq!(witness.product_success.uncertain_requests, 0);
    assert_eq!(witness.endpoint_unresolved_mass, 0.0);
    assert_eq!(witness.static_action.unfinished_request_mass, 0.0);
    assert_eq!(
        witness.static_action.risk.completion_failure,
        ProbabilityInterval {
            lower: 1.0,
            upper: 1.0
        }
    );
    let candidate = witness.alternatives[0].unwrap();
    assert_eq!(candidate.first_gate, Some(SelectionGate::RiskUnresolved));
    assert_eq!(candidate.unfinished_request_mass, 1.0);
    assert_eq!(candidate.unsupported_trajectory_mass, 1.0);
    assert_eq!(candidate.known_target_time_mass, 0.0);
    assert_eq!(candidate.target_time, None);
    assert_eq!(
        candidate.risk.completion_failure,
        ProbabilityInterval {
            lower: 0.0,
            upper: 1.0
        }
    );
}

#[test]
fn unresolved_endpoints_retain_weighted_request_and_event_denominators() {
    let requests = [request(3)];
    let comparison = comparison(
        trajectory(vec![complete(&requests[0], 1, 3, 0)], 3),
        vec![Err(Unsupported::Cost(CostUnknownReason::NoCoverage))],
    );
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 1, &config(), &success(1)).unwrap();
    aggregate.observe(rank(0, 2), &comparison).unwrap();
    let decision = aggregate.finish(summary(2, 1), &[None]).unwrap();
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
    let witness = decision.observation.unresolved.unwrap();
    assert_eq!(witness.endpoint_unresolved_mass, 0.5);
    assert_eq!(witness.product_success.uncertain_requests, 0);
    let observed = witness.static_action;
    assert_eq!(observed.unfinished_request_mass, 0.5);
    assert_eq!(observed.unsupported_trajectory_mass, 0.5);
    assert_eq!(observed.known_target_time_mass, 0.5);
    assert_eq!(observed.denominators.requests, 1);
    assert_eq!(observed.denominators.tpot_known_eligible, 0.5);
    assert_eq!(observed.denominators.tpot_unknown_eligible, 0.5);
    assert_eq!(observed.denominators.tpot_unknown_outcome, 0.0);
    assert_eq!(observed.denominators.itl_known_events, 1.0);
    assert_eq!(observed.denominators.itl_unknown_events, 1.0);
    let candidate = witness.alternatives[0].unwrap();
    assert_eq!(candidate.unfinished_request_mass, 1.0);
    assert_eq!(candidate.unsupported_trajectory_mass, 1.0);
    assert_eq!(candidate.denominators.tpot_known_eligible, 0.0);
    assert_eq!(candidate.denominators.tpot_unknown_eligible, 1.0);
    assert_eq!(candidate.denominators.itl_known_events, 0.0);
    assert_eq!(candidate.denominators.itl_unknown_events, 2.0);
}

#[test]
fn metric_threshold_equality_and_event_weighting_match_the_decision() {
    let requests = [request(1), request(5)];
    let comparison = comparison(
        trajectory(
            vec![
                complete(&requests[0], 2, 2, 0),
                complete(&requests[1], 20, 60, 1),
            ],
            60,
        ),
        vec![],
    );
    let mut targets = config();
    targets.joint.ttft_quantile = 0.5;
    targets.joint.itl_quantile = 0.75;
    // Missing product success keeps a witness available; it does not change
    // the actual TTFT/request and ITL/event failure denominators.
    let products = [None, success(1)[0]];
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 0, &targets, &products).unwrap();
    aggregate.observe(rank(0, 1), &comparison).unwrap();
    let decision = aggregate.finish(summary(1, 0), &[]).unwrap();
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
    assert_eq!(
        decision.observation.static_metrics.ttft,
        MetricClass::Feasible
    );
    assert_eq!(
        decision.observation.static_metrics.tpot,
        MetricClass::Feasible
    );
    assert_eq!(
        decision.observation.static_metrics.itl,
        MetricClass::Feasible
    );
    let action = decision.observation.unresolved.unwrap().static_action;
    assert_eq!(
        action.risk.ttft,
        ProbabilityInterval {
            lower: 0.5,
            upper: 0.5
        }
    );
    assert_eq!(
        action.risk.itl,
        Some(ProbabilityInterval {
            lower: 0.25,
            upper: 0.25
        })
    );
    assert_eq!(action.denominators.requests, 2);
    assert_eq!(action.denominators.tpot_known_eligible, 1.0);
    assert_eq!(action.denominators.tpot_undefined, 1.0);
    assert_eq!(action.denominators.itl_known_events, 4.0);
    assert_eq!(action.denominators.itl_unknown_events, 0.0);
}

#[test]
fn feasible_static_records_the_reached_paired_evidence_and_gain_gate() {
    let requests = [request(2)];
    let mut targets = config();
    targets.tpot_ms = 100.0;
    targets.itl_ms = 100.0;
    let comparison = comparison(
        trajectory(vec![complete(&requests[0], 1, 100, 0)], 100),
        vec![Ok(trajectory(vec![complete(&requests[0], 1, 80, 0)], 80))],
    );
    let evaluate = |paired| {
        let mut aggregate =
            ScenarioRiskAccumulator::new(&requests, 1, &targets, &success(1)).unwrap();
        aggregate.observe(rank(0, 1), &comparison).unwrap();
        aggregate.finish(summary(1, 0), &[paired]).unwrap()
    };
    let absent = evaluate(None);
    assert_eq!(absent.choice, PolicyChoice::Static);
    assert_eq!(absent.observation.candidates.joint.feasible, 1);
    assert_eq!(
        absent
            .observation
            .candidates
            .gates
            .static_time_or_paired_evidence_unavailable,
        1
    );
    assert!(absent.observation.unresolved.is_none());
    let equal = evaluate(Some(PairedActionEvidence {
        uncertainty: Duration::from_millis(19),
        control_cost: Duration::from_millis(1),
    }));
    assert_eq!(equal.choice, PolicyChoice::Static);
    assert_eq!(
        equal
            .observation
            .candidates
            .gates
            .gain_not_above_evidence_margin,
        1
    );
    let above = evaluate(Some(PairedActionEvidence {
        uncertainty: Duration::from_millis(18),
        control_cost: Duration::from_millis(1),
    }));
    assert_eq!(above.choice, PolicyChoice::Alternative(0));
    assert_eq!(above.observation.candidates.gates.eligible, 1);
}

#[test]
fn bounded_witness_omits_details_without_omitting_generic_candidates() {
    let requests = [request(2)];
    let comparison = comparison(
        trajectory(vec![complete(&requests[0], 1, 2, 0)], 2),
        (0..3)
            .map(|_| Err(Unsupported::Cost(CostUnknownReason::NoCoverage)))
            .collect(),
    );
    let mut aggregate = ScenarioRiskAccumulator::new(&requests, 3, &config(), &[None]).unwrap();
    aggregate.observe(rank(0, 1), &comparison).unwrap();
    let decision = aggregate.finish(summary(1, 0), &[None; 3]).unwrap();
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
    assert_eq!(decision.alternative_risks.len(), 3);
    assert_eq!(decision.observation.alternatives_total, 3);
    assert_eq!(decision.observation.candidates.joint.unresolved, 3);
    assert_eq!(decision.observation.candidates.gates.risk_unresolved, 3);
    let witness = decision.observation.unresolved.unwrap();
    assert_eq!(witness.alternatives_total, 3);
    assert!(witness.alternatives.iter().all(Option::is_some));
    assert_eq!(witness.details_omitted, 1);
}
