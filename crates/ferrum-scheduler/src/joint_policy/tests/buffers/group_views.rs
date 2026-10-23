//! Flat storage preserves the public builder's ordered work and rejection
//! boundaries. These fixtures exercise both original member and borrowed views.

use super::*;

fn push(plan: &mut PlanBuilder, members: &[GroupMember], view: bool) -> Result<(), Unsupported> {
    if view {
        let requests = members
            .iter()
            .map(|member| member.request)
            .collect::<Vec<_>>();
        let work = members.iter().map(|member| member.work).collect::<Vec<_>>();
        plan.push_group_view(GroupView {
            requests: &requests,
            work: &work,
        })
    } else {
        plan.push_group(members)
    }
}

#[test]
fn group_views_preserve_ordered_members_and_borrowed_prediction_work() {
    let mut prefill = [
        member(2, AttemptWorkKind::Prefill),
        member(0, AttemptWorkKind::Prefill),
    ];
    for (member, prompt) in prefill.iter_mut().zip([4, 2]) {
        member.work.requested = ExecutionRange::new(0, prompt).unwrap();
        member.work.context_tokens = prompt;
    }
    let mut decode = [
        member(3, AttemptWorkKind::Decode),
        member(1, AttemptWorkKind::Decode),
    ];
    for (member, context) in decode.iter_mut().zip([9, 7]) {
        member.work.requested = ExecutionRange::new(context - 1, 1).unwrap();
        member.work.context_tokens = context;
    }
    let mut original = builder(CostExecution::SequentialPrefillThenDecode);
    let mut viewed = builder(CostExecution::SequentialPrefillThenDecode);
    for members in [&prefill[..], &decode[..]] {
        push(&mut original, members, false).unwrap();
        push(&mut viewed, members, true).unwrap();
    }
    assert_eq!(contents(&viewed), contents(&original));
    let at = Instant::now();
    let snapshot = Snapshot { at, split: true };
    for (index, expected) in [&prefill[..], &decode[..]].into_iter().enumerate() {
        let group = viewed.group(index).unwrap();
        assert_eq!(group.len(), expected.len());
        assert_eq!(
            group
                .iter()
                .map(|m| (m.request, m.work))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|m| (m.request, m.work))
                .collect::<Vec<_>>()
        );
        // The original CostSnapshot receives the real contiguous group work,
        // including order-sensitive ranges, with no reconstructed query Vec.
        let query = CostQuery {
            work: group.work,
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        };
        assert_eq!(
            query.work,
            expected.iter().map(|m| m.work).collect::<Vec<_>>()
        );
        let CostPrediction::Known {
            central_service_time,
            ..
        } = snapshot.predict(&query, at)
        else {
            panic!("the original typed snapshot must see the accepted group")
        };
        assert_eq!(
            central_service_time,
            Duration::from_millis(if index == 0 { 20 } else { 10 })
        );
    }
    assert!(viewed.group(2).is_none());
    viewed.clear_groups();
    assert!(viewed.group(0).is_none());
    viewed.push_group(&decode[1..]).unwrap();
    assert_eq!(contents(&viewed), vec![vec![(1, decode[1].work)]]);
}

#[test]
fn rejected_second_view_preserves_first_group_and_allows_a_valid_retry() {
    for view in [false, true] {
        let mut plan = builder(CostExecution::SequentialPrefillThenDecode);
        let first = member(0, AttemptWorkKind::Prefill);
        push(&mut plan, &[first], view).unwrap();
        let before = contents(&plan);
        // The first member is valid; the later cross-group duplicate must not
        // append even that valid prefix or consume the second group boundary.
        assert_eq!(
            push(
                &mut plan,
                &[
                    member(1, AttemptWorkKind::Decode),
                    member(0, AttemptWorkKind::Decode)
                ],
                view
            ),
            Err(Unsupported::InvalidPlan)
        );
        assert_eq!(contents(&plan), before);
        assert!(plan.group(1).is_none());
        assert_eq!(
            push(&mut plan, &[member(1, AttemptWorkKind::Prefill)], view),
            Err(Unsupported::InvalidPlan),
            "a second prefill group is not a legal Split continuation"
        );
        assert_eq!(contents(&plan), before);
        let valid = [
            member(2, AttemptWorkKind::Decode),
            member(1, AttemptWorkKind::Decode),
        ];
        push(&mut plan, &valid, view).unwrap();
        assert_eq!(
            contents(&plan),
            vec![
                vec![(0, first.work)],
                vec![(2, valid[0].work), (1, valid[1].work)]
            ]
        );
    }
}

#[test]
fn group_views_keep_public_limit_and_rejection_precedence() {
    for view in [false, true] {
        let mut plan = builder(CostExecution::SingleInvocation);
        let first = member(0, AttemptWorkKind::Prefill);
        let duplicate = member(0, AttemptWorkKind::Decode);
        push(&mut plan, &[first], view).unwrap();
        let before = contents(&plan);
        plan.query_limits.max_query_bytes = CostQuery::projected_bytes(1).unwrap() - 1;
        plan.active.as_mut().unwrap()[0] = false;
        // Filtered absence is still a no-op even when a nonempty push would
        // fail byte limits, execution ordering, active membership and identity.
        plan.push_group_kind(&[first], AttemptWorkKind::Decode)
            .unwrap();
        assert_eq!(
            push(&mut plan, &[duplicate], view),
            Err(Unsupported::ParticipantBudget)
        );
        assert_eq!(contents(&plan), before);
        plan.query_limits = CostQueryLimits::default();
        assert_eq!(
            push(&mut plan, &[duplicate], view),
            Err(Unsupported::InvalidPlan)
        );
        plan.execution = CostExecution::SequentialPrefillThenDecode;
        assert_eq!(
            push(&mut plan, &[duplicate], view),
            Err(Unsupported::WaitingResources)
        );
        plan.active.as_mut().unwrap()[0] = true;
        assert_eq!(
            push(&mut plan, &[duplicate], view),
            Err(Unsupported::InvalidPlan)
        );
        assert_eq!(contents(&plan), before);
        let valid = member(1, AttemptWorkKind::Decode);
        push(&mut plan, &[valid], view).unwrap();
        assert_eq!(
            contents(&plan),
            vec![vec![(0, first.work)], vec![(1, valid.work)]]
        );
        // After both groups have been accepted, absent filtered work still
        // does not consume or reject a group; nonempty work sees the original
        // group-count guard before its other invalidity.
        plan.push_group_kind(&[first], AttemptWorkKind::Decode)
            .unwrap();
        assert_eq!(
            push(&mut plan, &[duplicate], view),
            Err(Unsupported::ParticipantBudget)
        );
    }
}

#[test]
fn mismatched_group_view_slices_do_not_truncate_or_mutate_accepted_work() {
    let mut plan = builder(CostExecution::SequentialPrefillThenDecode);
    plan.push_group(&[member(0, AttemptWorkKind::Prefill)])
        .unwrap();
    let before = contents(&plan);
    let work = [
        member(1, AttemptWorkKind::Decode).work,
        member(2, AttemptWorkKind::Decode).work,
    ];
    for (requests, work) in [(&[1usize][..], &work[..]), (&[1usize, 2][..], &work[..1])] {
        assert_eq!(
            plan.push_group_view(GroupView { requests, work }),
            Err(Unsupported::InvalidPlan)
        );
        assert_eq!(contents(&plan), before);
        assert!(plan.group(1).is_none());
    }
    plan.push_group_view(GroupView {
        requests: &[1, 2],
        work: &work,
    })
    .unwrap();
    assert_eq!(plan.group(1).unwrap().len(), 2);
}
