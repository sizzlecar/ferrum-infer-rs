//! Real frozen comparison -> cap commit -> fresh gate -> original executor probe.
use super::*;
use crate::implementations::continuous::joint_selection::AdmissionCapMode;

fn waiting_fixture() -> PhaseFixture {
    let mut fixture = PhaseFixture::with_admission(2, true);
    fixture.config.unknown_cost = ferrum_types::UnknownCostPolicy::Reject;
    fixture.scheduler.config.slo = Some(fixture.config);
    fixture
}

#[test]
fn admission_precommit_rescues_waiting_prefill_only_after_fresh_gate_and_real_probe() {
    let fixture = waiting_fixture();
    let mut round = fixture.round();
    let original = fixture
        .scheduler
        .waiting_queue
        .read()
        .iter()
        .next()
        .unwrap()
        .logical_work_frontier
        .clone();
    fixture
        .scheduler
        .prepare_joint_admission_gate(&hint(), &mut round, Instant::now());
    assert_eq!(round.statistics.joint_admission_prefill_phase_commits, 1);
    assert_eq!(
        round.statistics.joint_interventions, 0,
        "a policy commitment is not executed work"
    );
    assert_eq!(round.statistics.joint_prefill_phase_interventions, 0);
    assert_eq!(fixture.scheduler.waiting_count(), 1);
    assert_eq!(fixture.scheduler.active_count(), 1);
    {
        let waiting = fixture.scheduler.waiting_queue.read();
        let request = waiting.iter().next().unwrap();
        assert_eq!(request.joint_prefill_phase_cap.unwrap().limit.get(), 1);
        assert_eq!(request.logical_work_frontier, original);
    }
    let diagnostics = &round.statistics.joint_decision.admission;
    assert_eq!(diagnostics.computed, 2);
    assert_eq!(diagnostics.returned, 2);
    assert_eq!(diagnostics.consumed, 2);
    assert_eq!(diagnostics.outcomes.requires_action, 1);
    assert_eq!(diagnostics.outcomes.static_feasible, 1);
    let queries_before_probe = round.query_count;
    let mut scan = fixture.scheduler.begin_token_admission(1, None).unwrap();
    let candidate = scan
        .next(&mut |_| {}, Some(&mut round))
        .unwrap()
        .expect("fresh feasible gate permits a probe");
    assert_eq!(candidate.request().id, fixture.prefill);
    assert_eq!(
        fixture.scheduler.waiting_count(),
        1,
        "selection alone is not executor acceptance"
    );
    assert!(matches!(
        scan.apply(
            candidate,
            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                request_id: fixture.prefill.clone()
            })
        ),
        TokenAdmissionApply::Applied { accepted: Some(_) }
    ));
    assert!(scan.next(&mut |_| {}, Some(&mut round)).unwrap().is_none());
    scan.finish().unwrap();
    assert_eq!(round.query_count, queries_before_probe);
    assert_eq!(round.statistics.joint_decision.admission.consumed, 2);
    assert_eq!(round.statistics.joint_admission_prefill_phase_commits, 1);
    assert_eq!(fixture.scheduler.waiting_count(), 0);
    let batch = select(&fixture.scheduler, None).batch.unwrap();
    assert_eq!(tokens(&batch, &fixture.prefill), 1);
    assert_eq!(tokens(&batch, &fixture.decoder), 1);
}

#[test]
fn admission_precommit_fresh_comparison_uses_remaining_query_budget_without_rollback() {
    let fixture = waiting_fixture();
    // Disable memoization only in this budget fixture: cache hits consume the
    // per-rank simulation guard but not the original provider allowance. With
    // caching, provider charge alone need not permit even the first traversal.
    // Here every simulation call is charged, so the complete first traversal's
    // measured provider+forecast charge is an exact sufficient first-pass bound.
    let mut reference = fixture.round();
    reference.config.planning.max_cost_cache_bytes = 0;
    let selected = fixture
        .scheduler
        .evaluate_joint_admission(
            &hint(),
            &mut reference,
            Instant::now(),
            AdmissionCapMode::Propose,
            0,
        )
        .unwrap();
    assert!(selected.prefill_cap.is_some());
    let first_queries = reference.query_count;
    assert!(first_queries > 0);
    let evaluated = reference.statistics.joint_evaluation.admission;
    assert_eq!(evaluated.completed_traversals, 1);
    assert_eq!(evaluated.failed_traversals, 0);
    assert_eq!(evaluated.cache_hits, 0);
    assert_eq!(
        first_queries,
        usize::try_from(evaluated.provider_calls).unwrap()
            + 2 * reference.request_facts().unwrap().len(),
        "one stopping and one completion forecast per cohort member"
    );

    let mut round = fixture.round();
    round.config.planning.max_cost_cache_bytes = 0;
    round.config.planning.max_queries = first_queries;
    fixture
        .scheduler
        .prepare_joint_admission_gate(&hint(), &mut round, Instant::now());
    assert_eq!(round.statistics.joint_admission_prefill_phase_commits, 1);
    assert_eq!(
        round.query_count, first_queries,
        "the fresh pass cannot replenish exhausted queries"
    );
    assert_eq!(round.statistics.joint_comparisons, 2);
    let unsupported = serde_json::to_value(round.statistics.joint_unsupported).unwrap();
    assert_eq!(unsupported["query_budget"], 1);
    assert_eq!(
        round.statistics.joint_evaluation.admission.evaluations, 1,
        "the fresh pass rejects its empty query allowance before entering the evaluator"
    );
    assert_eq!(
        round.statistics.joint_evaluation.admission.provider_calls,
        evaluated.provider_calls
    );
    assert_eq!(
        round
            .statistics
            .joint_decision
            .admission
            .outcomes
            .static_feasible,
        0
    );
    assert_eq!(round.statistics.joint_interventions, 0);
    assert_eq!(fixture.scheduler.waiting_count(), 1);
    assert_eq!(
        fixture
            .scheduler
            .waiting_queue
            .read()
            .iter()
            .next()
            .unwrap()
            .joint_prefill_phase_cap
            .unwrap()
            .limit
            .get(),
        1
    );
    // Unknown=Reject is still authoritative. A durable cap does not turn the
    // failed second comparison into permission to execute or allocate.
    let mut scan = fixture.scheduler.begin_token_admission(1, None).unwrap();
    assert!(scan.next(&mut |_| {}, Some(&mut round)).unwrap().is_none());
    scan.finish().unwrap();
    assert_eq!(fixture.scheduler.active_count(), 1);
    assert!(fixture
        .scheduler
        .admission_failed_requests
        .read()
        .contains_key(&fixture.prefill));
}
