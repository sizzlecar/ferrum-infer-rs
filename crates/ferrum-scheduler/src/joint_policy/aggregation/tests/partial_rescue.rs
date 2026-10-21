use super::*;

fn rescue_config() -> SchedulerSloConfig {
    let mut config: SchedulerSloConfig = "ttft:90,tpot:200,itl:200".parse().unwrap();
    // One incomplete rank has weight 1/2. Its unknown completion/TPOT mass is
    // deliberately allowed by these targets, so only structural completion
    // proof can reject it; a strict risk threshold would hide that defect.
    config.joint.tpot_quantile = 0.5;
    config.joint.itl_quantile = 0.5;
    config.joint.min_completion_rate = 0.5;
    config
}

fn mixed_rank_rescue(alter: impl Fn(usize, &mut SimulatedRequest)) -> ScenarioPolicyDecision {
    let mut initial = request(2);
    initial.prompt_tokens = 2;
    let requests = [initial];
    let mut aggregate =
        ScenarioRiskAccumulator::new(&requests, 1, &rescue_config(), &success(1)).unwrap();
    for index in 0..2 {
        let mut candidate = complete(&requests[0], 40, 120, 0);
        alter(index, &mut candidate);
        candidate.risk_clock = candidate.central_clock;
        let mut candidate = trajectory(vec![candidate], 120);
        // A complete final trajectory can still have been behind at the old
        // static window. Known-infeasible rescue already permits this debt.
        candidate.window_progress[0].output_tokens = 1;
        let comparison = if index == 0 {
            comparison(
                trajectory(vec![complete(&requests[0], 100, 110, 0)], 110),
                vec![Ok(candidate)],
            )
        } else {
            let mut prefix = requests[0].clone();
            prefix.prefill_tokens = 2;
            prefix.output_tokens = 1;
            prefix.central_clock.first_output = Some(Duration::from_millis(100));
            prefix.central_clock.last_output = Some(Duration::from_millis(100));
            // At a partial window, the stored progress is final known
            // candidate progress; unlike the Exact case it cannot be a
            // partial-window checkpoint substituted for complete endpoints.
            candidate.window_progress = candidate
                .requests
                .iter()
                .map(SimulatedRequest::progress)
                .collect();
            Comparison {
                revision: 1,
                window: StaticWindow::AtLeast(Duration::from_millis(100)),
                targets: vec![Progress {
                    prefill_tokens: 2,
                    output_tokens: 2,
                }],
                static_result: StaticTrajectory::OutsideSupportPrefix(TrajectoryPrefix {
                    requests: vec![prefix],
                    central_elapsed: Duration::from_millis(100),
                    risk_elapsed: Duration::from_millis(100),
                    queries: 1,
                    cost_risk: CostRiskBudget {
                        conditional_scenario: Some(CostScenarioDependence::SharedResidualRank),
                        ..Default::default()
                    },
                    unsupported: Unsupported::Cost(CostUnknownReason::UnsupportedWithDetail(
                        CostUnsupportedDetail::OutsideObservedSupport,
                    )),
                }),
                alternatives: vec![Ok(candidate)],
            }
        };
        aggregate.observe(rank(index, 2), &comparison).unwrap();
    }
    let mut evaluation = summary(2, 0);
    evaluation.partial_static_ranks = 1;
    aggregate.finish(evaluation, &[None]).unwrap()
}

#[test]
fn mixed_exact_and_partial_ranks_allow_complete_rescue_after_the_prefix_bound() {
    let decision = mixed_rank_rescue(|_, _| {});
    assert_eq!(
        decision.observation.static_raw,
        JointFeasibility::Infeasible
    );
    assert_eq!(
        decision.alternative_risks[0].classify(&rescue_config()),
        JointFeasibility::Feasible
    );
    assert_eq!(decision.choice, PolicyChoice::Alternative(0));
    assert_eq!(decision.static_target_time, None);
    assert_eq!(
        decision.alternative_target_times[0],
        Some(Duration::from_millis(120))
    );
    assert!(decision.any_window_debt[0]);
    assert_eq!(decision.observation.candidates.gates.eligible, 1);
}

fn assert_structurally_incomplete_rescue_is_rejected(decision: ScenarioPolicyDecision) {
    assert_eq!(
        decision.observation.static_raw,
        JointFeasibility::Infeasible
    );
    assert_eq!(
        decision.alternative_risks[0].classify(&rescue_config()),
        JointFeasibility::Feasible,
        "loose risk targets must not mask the independent structural guard"
    );
    assert_eq!(
        decision.alternative_target_times[0],
        Some(Duration::from_millis(120))
    );
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
    assert_eq!(decision.observation.candidates.gates.eligible, 0);
    assert_eq!(
        decision
            .observation
            .candidates
            .gates
            .partial_window_unproven,
        1
    );
    assert_eq!(
        decision.observation.unresolved.unwrap().alternatives[0]
            .unwrap()
            .first_gate,
        Some(SelectionGate::PartialWindowUnproven)
    );
}

#[test]
fn mixed_rescue_requires_full_output_endpoints_in_both_exact_and_partial_ranks() {
    for incomplete_rank in 0..2 {
        let decision = mixed_rank_rescue(|index, request| {
            if index == incomplete_rank {
                request.output_tokens = 1;
                request.central_clock.last_output = request.central_clock.first_output;
                request.central_clock.completed = None;
                request.central_clock.interval_count = 0;
            }
        });
        assert_structurally_incomplete_rescue_is_rejected(decision);
    }
}

#[test]
fn endpoint_counts_without_a_completion_clock_cannot_authorize_mixed_rescue() {
    for incomplete_rank in 0..2 {
        let decision = mixed_rank_rescue(|index, request| {
            if index == incomplete_rank {
                request.central_clock.completed = None;
            }
        });
        assert_structurally_incomplete_rescue_is_rejected(decision);
    }
}

#[test]
fn output_and_completion_clock_do_not_substitute_for_required_prefill_progress() {
    for incomplete_rank in 0..2 {
        let decision = mixed_rank_rescue(|index, request| {
            if index == incomplete_rank {
                request.prefill_tokens = 1;
            }
        });
        assert_structurally_incomplete_rescue_is_rejected(decision);
    }
}
