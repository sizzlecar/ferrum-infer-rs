//! Homogeneous copying preserves the same work and validation boundaries as
//! filtered split groups; an absent phase remains a no-op, even at capacity.

use super::*;
use ferrum_execution::execution_attempt::AttemptPrefixRestore;

#[test]
fn homogeneous_absent_phase_remains_noop_before_every_guard() {
    let mut plan = builder(CostExecution::SequentialPrefillThenDecode);
    let prefill = member(0, AttemptWorkKind::Prefill);
    let decode = member(1, AttemptWorkKind::Decode);
    plan.push_group(&[prefill]).unwrap();
    plan.push_group(&[decode]).unwrap();
    let before = contents(&plan);
    // Both group slots are occupied, the source is outside the active mask
    // and duplicated, and all declared budgets are exhausted. No member is
    // selected, so none of these guards was reachable in the old wrapper.
    plan.participant_limit = 0;
    plan.query_limits.max_participants = 0;
    plan.query_limits.max_query_bytes = 0;
    plan.active = Some(Vec::new());
    let ignored = member(usize::MAX, AttemptWorkKind::Decode);
    plan.push_group_kind(&[ignored, ignored], AttemptWorkKind::Prefill)
        .unwrap();
    plan.push_group_kind(&[], AttemptWorkKind::Decode).unwrap();
    assert_eq!(contents(&plan), before);
    assert_eq!(plan.push_group(&[]), Err(Unsupported::ParticipantBudget));
    assert_eq!(contents(&plan), before);
}

#[test]
fn homogeneous_copy_preserves_full_work_order_and_reused_group_boundaries() {
    let members = [3, 0, 2, 1].map(|index| GroupMember {
        request: index,
        work: CostWork {
            kind: AttemptWorkKind::Decode,
            requested: ExecutionRange::new(30 + index, 1).unwrap(),
            context_tokens: 31 + index,
            prefix_restore: if index % 2 == 0 {
                AttemptPrefixRestore::Restored { tokens: index }
            } else {
                AttemptPrefixRestore::NoneObserved
            },
            internally_reused_tokens: (index % 2 == 0).then_some(index),
        },
    });
    let expected = members
        .iter()
        .map(|m| (m.request, m.work))
        .collect::<Vec<_>>();
    for execution in [
        CostExecution::SingleInvocation,
        CostExecution::SequentialPrefillThenDecode,
    ] {
        let mut plan = builder(execution);
        plan.push_group_kind(&members, AttemptWorkKind::Decode)
            .unwrap();
        assert_eq!(contents(&plan), vec![expected.clone()]);
        plan.clear_groups();
        plan.push_group(&members[..1]).unwrap();
        assert_eq!(contents(&plan), vec![expected[..1].to_vec()]);
        plan.clear_groups();
        plan.push_group(&members).unwrap();
        assert_eq!(contents(&plan), vec![expected.clone()]);
    }
}

#[test]
fn homogeneous_fast_copy_does_not_skip_failure_precedence_or_commit_rejected_work() {
    let mut plan = builder(CostExecution::SequentialPrefillThenDecode);
    let prefill = member(0, AttemptWorkKind::Prefill);
    plan.push_group(&[prefill]).unwrap();
    let before = contents(&plan);
    let duplicate = member(0, AttemptWorkKind::Decode);
    let inactive = member(3, AttemptWorkKind::Decode);
    plan.active.as_mut().unwrap()[3] = false;
    plan.participant_limit = 2;
    assert_eq!(
        plan.push_group_kind(&[duplicate, inactive], AttemptWorkKind::Decode),
        Err(Unsupported::ParticipantBudget)
    );
    plan.participant_limit = 4;
    assert_eq!(
        plan.push_group_kind(&[duplicate, inactive], AttemptWorkKind::Decode),
        Err(Unsupported::InvalidPlan)
    );
    assert_eq!(
        plan.push_group_kind(&[inactive, duplicate], AttemptWorkKind::Decode),
        Err(Unsupported::WaitingResources)
    );
    assert_eq!(contents(&plan), before);
    let valid = member(1, AttemptWorkKind::Decode);
    plan.push_group_kind(&[valid], AttemptWorkKind::Decode)
        .unwrap();
    let mut expected = before;
    expected.push(vec![(valid.request, valid.work)]);
    assert_eq!(contents(&plan), expected);
}

#[test]
fn homogeneous_filter_counts_only_selected_members_and_keeps_split_order() {
    let mut plan = builder(CostExecution::SequentialPrefillThenDecode);
    plan.participant_limit = 2;
    plan.query_limits.max_participants = 1;
    plan.query_limits.max_query_bytes = CostQuery::projected_bytes(1).unwrap();
    let prefill = member(0, AttemptWorkKind::Prefill);
    let decode = member(1, AttemptWorkKind::Decode);
    let ignored = member(usize::MAX, AttemptWorkKind::Decode);
    plan.push_group_kind(&[ignored, prefill, ignored], AttemptWorkKind::Prefill)
        .unwrap();
    let before = contents(&plan);
    // A present second Prefill group still fails the Split phase order.
    assert_eq!(
        plan.push_group_kind(&[prefill], AttemptWorkKind::Prefill),
        Err(Unsupported::InvalidPlan)
    );
    assert_eq!(contents(&plan), before);
    plan.push_group_kind(&[prefill, decode, prefill], AttemptWorkKind::Decode)
        .unwrap();
    assert_eq!(
        contents(&plan),
        vec![vec![(0, prefill.work)], vec![(1, decode.work)]]
    );
}
