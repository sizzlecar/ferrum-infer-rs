use super::*;

#[test]
fn complete_rescue_reports_an_eligible_action_without_an_unresolved_witness() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    let (decision, diagnostics) = evaluate_partial(&snapshot, &partial_config());
    let decision = decision.unwrap();
    assert_eq!(decision.choice, PolicyChoice::Alternative(0));
    assert_eq!(
        decision.alternative_target_times[0],
        Some(Duration::from_millis(30))
    );
    assert_eq!(diagnostics.rank_partial_static, 2);
    let observed = decision.observation;
    assert_eq!(observed.static_raw, JointFeasibility::Infeasible);
    assert_eq!(observed.static_effective, JointFeasibility::Infeasible);
    assert_eq!(observed.static_metrics.ttft, MetricClass::Infeasible);
    assert_eq!(observed.candidates.joint.feasible, 1);
    assert_eq!(observed.candidates.gates.eligible, 1);
    assert_eq!(observed.candidates.gates.partial_window_unproven, 0);
    assert!(observed.unresolved.is_none());
    // The seed distribution is cached; the three supported alternative groups
    // add three Known calls. Unknown static tails are not cached and are
    // queried once for each of the two ranks.
    assert_eq!(diagnostics.failure_count, 2);
    assert_eq!(diagnostics.provider_calls - diagnostics.failure_count, 4);
    assert_eq!(
        diagnostics.provider_calls,
        1 + diagnostics.simulation_queries - diagnostics.cache_hits
    );
}

#[test]
fn later_complete_rescue_keeps_window_fact_without_recording_a_selection_block() {
    let mut snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    snapshot.alternative_step_ms = 40;
    let mut config = partial_config();
    config.ttft_ms = 90.0;
    let (decision, _) = evaluate_partial(&snapshot, &config);
    let decision = decision.unwrap();
    assert_eq!(decision.choice, PolicyChoice::Alternative(0));
    assert_eq!(decision.observation.candidates.joint.feasible, 1);
    assert_eq!(
        decision
            .observation
            .candidates
            .gates
            .partial_window_unproven,
        0
    );
    assert_eq!(decision.observation.candidates.gates.eligible, 1);
    assert!(decision.observation.unresolved.is_none());
    assert_eq!(decision.static_target_time, None);
    assert_eq!(
        decision.alternative_target_times[0],
        Some(Duration::from_millis(120))
    );
    assert_eq!(decision.any_window_debt, vec![true]);
}

#[test]
fn feasible_candidate_does_not_hide_an_unresolved_static_action() {
    let mut snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    snapshot.static_first_ms = 10;
    snapshot.alternative_step_ms = 1;
    let (decision, _) = evaluate_partial(&snapshot, &partial_config());
    let decision = decision.unwrap();
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
    assert_eq!(
        decision.observation.static_raw,
        JointFeasibility::Unresolved
    );
    assert_eq!(
        decision.observation.static_effective,
        JointFeasibility::Unresolved
    );
    assert_eq!(decision.observation.candidates.joint.feasible, 1);
    assert_eq!(decision.observation.candidates.gates.static_unresolved, 1);
    let witness = decision.observation.unresolved.unwrap();
    assert_eq!(
        witness.static_action.metrics.completion_failure,
        MetricClass::Unresolved
    );
    assert_eq!(witness.static_action.unfinished_request_mass, 2.0);
    assert_eq!(witness.product_success.uncertain_requests, 0);
    let candidate = witness.alternatives[0].unwrap();
    assert_eq!(candidate.first_gate, Some(SelectionGate::StaticUnresolved));
    assert_eq!(candidate.joint, JointFeasibility::Feasible);
    assert_eq!(candidate.target_time, Some(Duration::from_millis(3)));
    assert_eq!(candidate.unfinished_request_mass, 0.0);
    assert!(!candidate.partial_window_unproven);
}

#[test]
fn partial_static_without_an_alternative_reports_no_rescue_action() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    let requests = partial_requests();
    let (decision, diagnostics) = evaluate_partial_with_actions(
        &snapshot,
        &partial_config(),
        &requests,
        EvaluationHorizon::ScenarioCompletion,
        &[],
        |_, _| Ok(()),
    );
    let decision = decision.unwrap();
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
    assert_eq!(diagnostics.rank_partial_static, 2);
    assert_eq!(decision.observation.alternatives_total, 0);
    assert_eq!(
        decision.observation.static_effective,
        JointFeasibility::Infeasible
    );
    assert_eq!(decision.observation.candidates.joint.feasible, 0);
    assert_eq!(decision.observation.candidates.joint.infeasible, 0);
    assert_eq!(decision.observation.candidates.joint.unresolved, 0);
    let witness = decision.observation.unresolved.unwrap();
    assert_eq!(witness.alternatives_total, 0);
    assert!(witness.alternatives.iter().all(Option::is_none));
    assert_eq!(witness.details_omitted, 0);
    // One Known seed plus an uncached static-tail rejection per rank.
    assert_eq!(diagnostics.failure_count, 2);
    assert_eq!(diagnostics.provider_calls - diagnostics.failure_count, 1);
    assert_eq!(
        diagnostics.provider_calls,
        1 + diagnostics.simulation_queries - diagnostics.cache_hits
    );
}
