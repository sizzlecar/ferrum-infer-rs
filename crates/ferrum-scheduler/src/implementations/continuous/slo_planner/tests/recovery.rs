use super::*;

fn single_width(mut s: SchedulerSnapshot) -> SchedulerSnapshot {
    s.capabilities.decode_batch_sizes = vec![nz(1)];
    s.capabilities.max_wave_rows = nz(1);
    s.capabilities.native_mixed = false;
    s
}

fn deadline(mut row: RequestSchedulingView, at: u64) -> RequestSchedulingView {
    let last = row.timing.last_commit_at_ns.unwrap();
    row.timing.budgets.tpot_ns = n64(at - last);
    row.timing.budgets.itl_ns = n64(at - last);
    row
}

fn recovered(s: &SchedulerSnapshot, depth: usize, cost: u64) -> PlanningDecision {
    planner(depth).propose_recovery(
        s,
        Arc::new(PlanningObligationSet::capture(s, 100).unwrap()),
        &Model(|_: &WaveExecutionShape| Some(cost)),
        &TestResolver,
        None,
        &mut Clock(100),
    )
}

fn protected(
    decision: PlanningDecision,
) -> (
    SelectedWave,
    PlanningWitnessSummary,
    Arc<PlanningObligationSet>,
) {
    match decision {
        PlanningDecision::ProtectedWithinHorizon {
            first_wave,
            witness,
            protection,
            ..
        } => (first_wave, witness, protection),
        other => panic!("expected explicit forward protection, got {other:?}"),
    }
}

#[test]
fn historical_ttft_miss_preserves_both_forward_deadlines_and_failed_label() {
    let mut a = deadline(decode(1), 120);
    a.timing.budgets.ttft_ns = n64(80); // actual first=90 proves history, without trusting a flag
    let b = deadline(decode(2), 112);
    let s = single_width(snapshot(vec![a, b]));
    let (first, witness, scope) = protected(recovered(&s, 2, 8));
    assert_eq!(first.candidate.work[0].key, s.requests[1].key);
    assert_eq!(witness.waves, 2);
    assert!(scope.rows()[0].historical_violation);
    assert!(scope
        .rows()
        .iter()
        .all(|r| r.forward == ForwardObligation::Protected));
    assert!(scope.new_time_promises_closed());
    assert_eq!(s.requests[0].timing.first_commit_at_ns, Some(90));
    assert_eq!(first.protection.as_deref(), Some(scope.as_ref()));
    assert!(matches!(
        planner(2).propose(
            &s,
            &Model(|_: &WaveExecutionShape| Some(8)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::ProvenImpossibleUnderModel {
            reason: PlanningImpossibleReason::HistoricalViolation { .. },
            ..
        }
    ));
}

#[test]
fn expired_owner_stays_in_common_work_and_resource_accounting() {
    let mut a = decode(1);
    a.timing.first_commit_at_ns = Some(80);
    a.timing.last_commit_at_ns = Some(80);
    a = deadline(a, 90);
    let mut s = single_width(snapshot(vec![a, deadline(decode(2), 112)]));
    let scope = PlanningObligationSet::capture(&s, 100).unwrap();
    let candidates = candidates::enumerate(
        &s,
        &s.requests,
        100,
        16,
        Some(&scope),
        &TestResolver,
        &mut || Ok(()),
    )
    .unwrap();
    // Expired A remains an executable fairness candidate, but must not hide
    // the only feasible shared order B(108) -> A(116) behind A's old deadline.
    assert_eq!(candidates.waves[0].work[0].key, s.requests[1].key);
    assert!(candidates
        .waves
        .iter()
        .any(|wave| wave.work[0].key == s.requests[0].key));
    let (first, witness, scope) = protected(recovered(&s, 2, 8));
    assert_eq!(first.candidate.work[0].key, s.requests[1].key);
    assert_eq!(witness.predicted_output_tokens, 2);
    assert_eq!(scope.rows()[0].expired_deadline_ns, Some(90));
    s.capacity.available_kv_tokens = 1; // both real increments cannot fit
    assert!(matches!(
        recovered(&s, 2, 8),
        PlanningDecision::Unknown { .. }
    ));
}

#[test]
fn sticky_owner_with_earlier_live_deadline_is_not_automatically_deprioritized() {
    let mut a = deadline(decode(1), 107);
    a.timing.slo_failed = true;
    let s = single_width(snapshot(vec![a, deadline(decode(2), 116)]));
    let (first, _, scope) = protected(recovered(&s, 2, 4));
    assert_eq!(first.candidate.work[0].key, s.requests[0].key);
    assert_eq!(scope.rows()[0].forward, ForwardObligation::Protected);
}

#[test]
fn final_replay_cannot_demote_newly_late_healthy_peer() {
    use std::cell::Cell;
    let mut a = decode(1);
    a.timing.slo_failed = true;
    let s = single_width(snapshot(vec![a, deadline(decode(2), 112)]));
    let scope = Arc::new(PlanningObligationSet::capture(&s, 100).unwrap());
    let now = Cell::new(100);
    struct Advancing<'a>(&'a Cell<u64>);
    impl PlanningCostModel for Advancing<'_> {
        fn model_version(&self) -> u64 {
            7
        }
        fn predict(
            &self,
            _: &ExecutionFingerprint,
            _: &WaveExecutionShape,
            _: u64,
        ) -> Option<PlanningCost> {
            self.0.set(120);
            Some(PlanningCost {
                typical_ns: 4,
                planning_ns: 4,
                model_version: 7,
                valid_for_ns: 1000,
            })
        }
    }
    struct Now<'a>(&'a Cell<u64>);
    impl PlanningClock for Now<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    let result = planner(2).propose_recovery(
        &s,
        scope.clone(),
        &Advancing(&now),
        &TestResolver,
        None,
        &mut Now(&now),
    );
    assert!(matches!(result, PlanningDecision::Unknown { .. }));
    assert_eq!(scope.rows()[1].forward, ForwardObligation::Protected);
    assert!(!scope.rows()[1].historical_violation);
}

#[test]
fn due_service_is_first_not_perpetually_at_the_horizon_tail() {
    let mut a = decode(1);
    a.timing.slo_failed = true;
    a.recovery_service = RecoveryServiceDebt::new(nz(2));
    a.fairness_rank = 1000;
    let mut s = single_width(snapshot(vec![a, deadline(decode(2), 112)]));
    for bypass in 0..2 {
        let (first, _, _) = protected(recovered(&s, 2, 4));
        // Model a real submitted peer service; merely re-planning does not change debt.
        assert_eq!(s.requests[0].recovery_service.eligible_bypasses(), bypass);
        assert_eq!(
            first.candidate.work[0].key, s.requests[1].key,
            "urgent healthy service pushes late work to this witness's tail"
        );
        s.requests[0].recovery_service.bypass();
    }
    let scope = PlanningObligationSet::capture(&s, 100).unwrap();
    assert_eq!(scope.required_first_service(), Some(&s.requests[0].key));
    let (first, _, _) = protected(recovered(&s, 2, 4));
    assert_eq!(first.candidate.work[0].key, s.requests[0].key);
    assert!(
        s.requests[0].recovery_service.due(),
        "pure simulation must not repay live debt"
    );
}

#[test]
fn fairness_deadline_conflict_is_unknown_not_protected_or_admitted() {
    let mut a = decode(1);
    a.timing.slo_failed = true;
    a.recovery_service = RecoveryServiceDebt::new(nz(1));
    a.recovery_service.bypass();
    let s = single_width(snapshot(vec![a, deadline(decode(2), 112)]));
    assert!(matches!(
        recovered(&s, 2, 8),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::RecoveryConflict,
            ..
        }
    ));
}

#[test]
fn simulated_peer_service_matures_debt_inside_the_same_common_sequence() {
    let mut a = decode(1);
    a.timing.slo_failed = true;
    a.recovery_service = RecoveryServiceDebt::new(nz(1));
    let s = single_width(snapshot(vec![
        a,
        deadline(decode(2), 106),
        deadline(decode(3), 116),
    ]));
    let scope = PlanningObligationSet::capture(&s, 100).unwrap();
    let model = Model(|_: &WaveExecutionShape| Some(4));
    let initial =
        candidates::enumerate(
            &s,
            &s.requests,
            100,
            16,
            None,
            &TestResolver,
            &mut || Ok(()),
        )
        .unwrap();
    let b = initial
        .waves
        .iter()
        .find(|wave| wave.work[0].key == s.requests[1].key)
        .unwrap()
        .clone();
    let after_b = simulation::simulate(
        &s,
        &[b.clone()],
        &model,
        &TestResolver,
        None,
        &mut || Ok(()),
        100,
        true,
        Some(&scope),
    )
    .unwrap();
    assert!(after_b.requests[0].recovery_service.due());
    let next = candidates::enumerate(
        &s,
        &after_b.requests,
        104,
        16,
        None,
        &TestResolver,
        &mut || Ok(()),
    )
    .unwrap();
    let c = next
        .waves
        .iter()
        .find(|wave| wave.work[0].key == s.requests[2].key)
        .unwrap()
        .clone();
    assert!(matches!(
        simulation::simulate(
            &s,
            &[b, c],
            &model,
            &TestResolver,
            None,
            &mut || Ok(()),
            100,
            true,
            Some(&scope)
        ),
        Err(simulation::SimulationFailure::SequenceViolation)
    ));
    assert_eq!(s.requests[0].recovery_service.eligible_bypasses(), 0);
}

#[test]
fn expired_prefill_checkpoint_preserves_reference_and_net_recompute_credit() {
    let mut a = prefill(1);
    let RequestPhaseView::Prefill(ref mut progress) = a.phase else {
        unreachable!()
    };
    progress.logical_high_water = 4; // redoing 0..4 gets no new reference work
    progress.milestones = Arc::from([
        PrefillMilestone {
            at_ns: 90,
            required_reference_work_ns: 60,
        },
        PrefillMilestone {
            at_ns: 140,
            required_reference_work_ns: 100,
        },
    ]);
    let mut s = single_width(snapshot(vec![a]));
    s.capabilities.prefill_chunk_sizes = vec![n32(4)];
    let (first, witness, scope) = protected(recovered(&s, 2, 8));
    assert_eq!(scope.rows()[0].expired_control_milestones, vec![0]);
    assert!(!scope.rows()[0].historical_violation);
    assert_eq!(witness.net_prefill_reference_work_ns, 60);
    assert_eq!(witness.predicted_output_tokens, 1);
    assert!(matches!(
        first.candidate.work[0].action,
        WaveAction::Prefill { offset: 0, .. }
    ));
    let RequestPhaseView::Prefill(progress) = &s.requests[0].phase else {
        unreachable!()
    };
    assert_eq!(
        (progress.admitted_at_ns, progress.logical_high_water),
        (0, 4)
    );
}

#[test]
fn blocked_late_owner_is_retained_without_inventing_output_credit() {
    let mut a = decode(1);
    a.timing.first_commit_at_ns = Some(80);
    a.timing.last_commit_at_ns = Some(80);
    a = deadline(a, 90);
    a.readiness = RequestReadiness::OutputBlocked;
    a.output_credit.available_token_commands = 0;
    let s = single_width(snapshot(vec![a, deadline(decode(2), 112)]));
    let (_, witness, scope) = protected(recovered(&s, 2, 4));
    assert_eq!(scope.rows().len(), 2);
    assert_eq!(witness.predicted_output_tokens, 1);
    assert_eq!(witness.requests_with_obligations_beyond_horizon, 1);
}

#[test]
fn healthy_scope_retains_full_semantics_and_rejects_rebound_scope_or_budget_loss() {
    let s = snapshot(vec![decode(1)]);
    let scope = Arc::new(PlanningObligationSet::capture(&s, 100).unwrap());
    assert!(!scope.needs_recovery());
    assert!(matches!(
        recovered(&s, 1, 4),
        PlanningDecision::FeasibleWithinHorizon { .. }
    ));
    let mut changed = s.clone();
    changed.requests[0].key.incarnation += 1;
    assert!(matches!(
        planner(1).propose_recovery(
            &changed,
            scope,
            &Model(|_: &WaveExecutionShape| Some(4)),
            &TestResolver,
            None,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
    assert_eq!(
        PlanningObligationSet::capture_with_budget(&s, 100, &mut || Err(
            PlanningUnknownReason::ComputeBudgetExhausted
        )),
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    );
}
