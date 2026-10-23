use super::*;
use crate::cost_prediction::CostScenarioDependence;
use crate::joint_policy::{
    DecisionObservation, RequestClock, ScenarioEvaluation, ScenarioRiskAccumulator,
    SimulatedRequest,
};
use ferrum_types::RequestId;
use std::time::{Duration, Instant};

fn unresolved(alternatives: usize) -> DecisionObservation {
    let request = SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 1,
        prefill_tokens: 0,
        output_tokens: 0,
        output_cap: 2,
        scenario_end_tokens: 2,
        age: Duration::ZERO,
        central_clock: RequestClock::default(),
        risk_clock: RequestClock::default(),
    };
    let config = "ttft:10,tpot:10,itl:10".parse().unwrap();
    ScenarioRiskAccumulator::new(&[request], alternatives, &config, &[None])
        .unwrap()
        .finish(
            ScenarioEvaluation {
                valid_until: Instant::now() + Duration::from_secs(1),
                dependence: CostScenarioDependence::SharedResidualRank,
                ranks: 1,
                queries: 1,
                cache_hits: 0,
                cache_retained_bytes: 0,
                resolved_mass: 0.0,
                unresolved_mass: 1.0,
                unresolved_ranks: 1,
                partial_static_ranks: 0,
            },
            &vec![None; alternatives],
        )
        .unwrap()
        .observation
}

#[test]
fn decision_diagnostics_discarded_and_relabelled_results_keep_distinct_denominators() {
    let observation = unresolved(2);
    let mut totals = JointDecisionDiagnostics::default();
    totals.computed(ComparisonKind::Admission);
    totals.post_finish_time_rejected(ComparisonKind::Admission);
    assert_eq!(totals.admission.computed, 1);
    assert_eq!(totals.admission.returned, 0);
    assert_eq!(totals.admission.consumed, 0);
    assert!(totals.admission.latest_unresolved.is_none());

    totals.computed(ComparisonKind::Admission);
    totals.returned(ComparisonKind::Admission);
    totals.consume(
        ComparisonKind::Admission,
        DecisionConsumerOutcome::StoppingCoverage,
        &observation,
    );
    assert_eq!(totals.admission.outcomes.stopping_coverage, 1);
    assert_eq!(totals.admission.candidates.joint.unresolved, 2);
    assert_eq!(totals.admission.unresolved.decisions, 0);
    assert!(totals.admission.latest_unresolved.is_none());

    totals.computed(ComparisonKind::Dispatch);
    totals.returned(ComparisonKind::Dispatch);
    totals.consume(
        ComparisonKind::Dispatch,
        DecisionConsumerOutcome::FeasibilityUnresolved,
        &observation,
    );
    assert_eq!(totals.dispatch.unresolved.decisions, 1);
    assert_eq!(
        totals.dispatch.unresolved.candidates.gates.risk_unresolved,
        2
    );
    assert_eq!(
        totals
            .dispatch
            .unresolved
            .any_candidate_completion_unresolved,
        1
    );
    assert_eq!(
        totals
            .dispatch
            .latest_unresolved
            .unwrap()
            .alternatives_total,
        2
    );
    assert_eq!(totals.dispatch.outcomes.alternative, 0);
}

#[test]
fn decision_diagnostics_merge_preserves_bounded_last_evidence_and_named_health() {
    let mut retained = JointDecisionDiagnostics::default();
    let mut round = JointDecisionDiagnostics::default();
    let observation = unresolved(0);
    round.computed(ComparisonKind::Dispatch);
    round.returned(ComparisonKind::Dispatch);
    round.consume(
        ComparisonKind::Dispatch,
        DecisionConsumerOutcome::FeasibilityUnresolved,
        &observation,
    );
    retained.add(&round);
    assert_eq!(retained.dispatch.unresolved.no_alternatives, 1);
    assert_eq!(
        retained
            .dispatch
            .latest_unresolved
            .unwrap()
            .alternatives_total,
        0
    );
    let empty = JointDecisionDiagnostics::default();
    retained.add(&empty);
    assert!(retained.dispatch.latest_unresolved.is_some());
    assert_eq!(
        retained.retained_bytes_bound,
        std::mem::size_of::<JointDecisionDiagnostics>()
    );
    let json = serde_json::to_value(retained).unwrap();
    assert_eq!(json["dispatch"]["unresolved"]["decisions"], 1);
    assert_eq!(
        json["dispatch"]["latest_unresolved"]["static_effective"],
        "unresolved"
    );
    assert_eq!(json["dispatch"]["latest_unresolved"]["details_omitted"], 0);
    assert!(json["admission"]["latest_unresolved"].is_null());

    retained.dispatch.computed = u64::MAX;
    retained.add(&round);
    assert_eq!(retained.dispatch.computed, u64::MAX);
    assert_eq!(retained.dispatch.unresolved.decisions, 2);
}
