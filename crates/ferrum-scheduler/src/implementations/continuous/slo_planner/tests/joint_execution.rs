//! Candidate for slo_planner/tests/joint_execution.rs.
//! Requires the new execution.rs traits/propose_with_execution entrypoint.
//! Synthetic wiring tests only; not backend resource/route validation.
use super::*;
mod fixture;
use fixture::*;

#[test]
fn immutable_execution_siblings_and_logical_rollout_share_their_own_frontier() {
    let s = snapshot(vec![decode(1), decode(2)]);
    let initial = s.clone();
    let pool = JointPool::new(3); // one persistent unit/row + one transient unit
    let settings = planner(2).settings;
    let session = execution::ExecutionSession::new(&pool, &settings);
    let root = session.begin(&s, &mut || Ok(())).unwrap();
    let left = project_decode(&s, root.as_ref(), &s.requests, &[0])
        .unwrap()
        .unwrap();
    let right = project_decode(&s, root.as_ref(), &s.requests, &[1])
        .unwrap()
        .unwrap();
    // Both rows still fit on the parent after both sibling projections.
    assert!(project_decode(&s, root.as_ref(), &s.requests, &[0, 1])
        .unwrap()
        .is_some());
    let mut after_left = s.requests.clone();
    after_left[0].context_tokens += 1;
    after_left[0].timing.committed_tokens += 1;
    after_left[0].timing.last_commit_at_ns = Some(105);
    let mut after_right = s.requests.clone();
    after_right[1].context_tokens += 1;
    after_right[1].timing.committed_tokens += 1;
    after_right[1].timing.last_commit_at_ns = Some(105);
    assert!(
        project_decode(&s, left.successor.as_ref(), &after_left, &[1])
            .unwrap()
            .is_some()
    );
    assert!(
        project_decode(&s, right.successor.as_ref(), &after_right, &[0])
            .unwrap()
            .is_some()
    );
    assert_eq!(s, initial);
    assert_eq!(pool.free, 3);

    // Actual planner advance supplies next logical time/credit while the
    // opaque successor supplies persistent resource growth, from one lineage.
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(3);
    request.timing.budgets.itl_ns = n64(20);
    request.timing.budgets.tpot_ns = n64(20);
    let mut s = snapshot(vec![request]);
    s.scope.horizon_end_ns = 130;
    let pool = JointPool::new(4);
    let (_, witness, _) = feasible(planner(2).propose_with_execution(
        &s,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &pool,
        &mut Clock(100),
    ));
    assert_eq!(witness.predicted_output_tokens, 2);
    let seen = pool.seen.borrow();
    let second = seen
        .iter()
        .find(|entry| {
            let r = &entry.requests[0];
            r.context_tokens == 11 && r.timing.committed_tokens == 2
        })
        .expect("the second wave must receive the first wave's actual logical successor");
    assert_eq!(second.remaining, 3);
    assert_eq!(second.requests[0].timing.first_commit_at_ns, Some(90));
    assert_eq!(second.requests[0].timing.last_commit_at_ns, Some(105));
    assert_eq!(
        second.requests[0].output_credit.available_token_commands,
        15
    );
    assert_eq!(s.requests[0].context_tokens, 10);
    assert_eq!(pool.free, 4);
}

#[test]
fn final_fresh_begin_detects_evidence_revocation_after_successful_search_projection() {
    let s = snapshot(vec![decode(1)]); // one remaining token, one logical choice
    let pool = JointPool::new(3);
    let model = Model(|_: &WaveExecutionShape| {
        pool.revoked.set(true); // first valid projection reached cost lookup
        Some(4)
    });
    let decision = planner(1).propose_with_execution(&s, &model, &pool, &mut Clock(100));
    assert!(
        matches!(
            decision,
            PlanningDecision::Unknown {
                reason: PlanningUnknownReason::UnknownResourceEvidence,
                ..
            }
        ),
        "revoked evidence cannot return a saved witness: {decision:?}"
    );
    assert!(!pool.seen.borrow().is_empty());
    assert!(
        pool.begin_revocations.borrow().contains(&true),
        "final validation must start from a fresh context, not a saved successor"
    );
}

#[test]
fn joint_callbacks_cannot_omit_or_swallow_a_failed_budget_poll() {
    let s = snapshot(vec![decode(1)]);
    for phase in [Phase::Begin, Phase::Project] {
        for fault in [BudgetFault::NoPoll, BudgetFault::SwallowPoll] {
            let mut pool = JointPool::new(3);
            pool.fault = Some((phase, fault));
            let decision = planner(1).propose_with_execution(
                &s,
                &Model(|_: &WaveExecutionShape| -> Option<u64> {
                    panic!("budget-invalid execution evidence must not query the model")
                }),
                &pool,
                &mut JointClock(&pool.now),
            );
            assert!(
                matches!(
                    decision,
                    PlanningDecision::Unknown {
                        reason: PlanningUnknownReason::ComputeBudgetExhausted,
                        ..
                    }
                ),
                "{phase:?}/{fault:?}: {decision:?}"
            );
            assert_eq!(pool.free, 3);
        }
    }
}

#[test]
fn returned_order_must_preserve_full_work_and_canonical_rows_must_match_it() {
    let mut s = snapshot(vec![decode(1), decode(2)]);
    s.capabilities.decode_batch_sizes = vec![nz(2)];
    s.requests[0].recurrent_state_bytes = 128;
    s.requests[1].recurrent_state_bytes = 256;
    let original: Vec<_> = s
        .requests
        .iter()
        .map(|r| CandidateWork {
            key: r.key.clone(),
            action: WaveAction::Decode,
        })
        .collect();
    let mut reverse = JointPool::new(3);
    reverse.returned = Returned::Reverse;
    let (selected, _, _) = feasible(planner(1).propose_with_execution(
        &s,
        &Model(|_: &WaveExecutionShape| Some(4)),
        &reverse,
        &mut Clock(100),
    ));
    assert_eq!(
        selected.candidate.work,
        original.iter().rev().cloned().collect::<Vec<_>>()
    );
    for bad in [
        Returned::Duplicate,
        Returned::ReplaceAction,
        Returned::ReplaceIncarnation,
        Returned::BadRow,
        Returned::BadRecurrent,
    ] {
        let mut pool = JointPool::new(3);
        pool.returned = bad;
        let decision = planner(1).propose_with_execution(
            &s,
            &Model(|_: &WaveExecutionShape| -> Option<u64> {
                panic!("invalid work/shape must be rejected before cost lookup")
            }),
            &pool,
            &mut Clock(100),
        );
        assert!(
            matches!(
                decision,
                PlanningDecision::Unknown {
                    reason: PlanningUnknownReason::InvalidShapeEvidence,
                    ..
                }
            ),
            "{bad:?}: {decision:?}"
        );
    }
}
