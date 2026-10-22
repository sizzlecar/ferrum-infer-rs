//! A machine-word fast path must preserve the unbounded index contract and
//! existing validation precedence, not impose the default participant limit.

use super::*;

fn wide_builder(execution: CostExecution) -> PlanBuilder {
    let width = usize::BITS as usize + 2;
    let mut plan = builder(execution);
    plan.participant_limit = width;
    plan.query_limits = CostQueryLimits {
        max_participants: width,
        max_query_bytes: CostQuery::projected_bytes(width).unwrap(),
    };
    plan.active = Some(vec![true; width]);
    plan
}

#[test]
fn membership_word_accepts_wider_cohorts_without_reordering_work() {
    let width = usize::BITS as usize + 2;
    let requests = (0..width).map(|_| request()).collect::<Vec<_>>();
    let wave = (0..width)
        .rev()
        .map(|index| GroupMember {
            request: index,
            work: decode(&requests[index]),
        })
        .collect::<Vec<_>>();
    let mut plan = wide_builder(CostExecution::SingleInvocation);
    plan.push_group(&wave).unwrap();
    assert_eq!(
        contents(&plan),
        vec![wave.iter().map(|m| (m.request, m.work)).collect::<Vec<_>>()]
    );
    for member in &plan.groups()[0] {
        validate_member(&requests, *member).unwrap();
    }
    plan.clear_groups();
    plan.push_group(&wave[..1]).unwrap();
    assert_eq!(contents(&plan), vec![vec![(wave[0].request, wave[0].work)]]);
}

#[test]
fn membership_word_preserves_unmasked_out_of_range_validation_boundary() {
    for index in [
        usize::BITS as usize - 1,
        usize::BITS as usize,
        usize::BITS as usize + 1,
        usize::MAX,
    ] {
        let mut plan = builder(CostExecution::SingleInvocation);
        // Neither a very large declared limit nor a very large index causes
        // allocation indexed by that value or creates an implicit hard cap.
        plan.participant_limit = usize::MAX;
        plan.query_limits = CostQueryLimits {
            max_participants: usize::MAX,
            max_query_bytes: usize::MAX,
        };
        plan.active = None;
        let item = member(index, AttemptWorkKind::Decode);
        plan.push_group(&[item]).unwrap();
        assert_eq!(
            validate_member(&[request()], item),
            Err(Unsupported::InvalidPlan)
        );
        plan.clear_groups();
        assert_eq!(
            plan.push_group(&[item, item]),
            Err(Unsupported::InvalidPlan)
        );
        assert!(plan.groups().is_empty());
        plan.active = Some(vec![true]);
        assert_eq!(
            plan.push_group(&[item, item]),
            Err(Unsupported::WaitingResources)
        );
        assert!(plan.groups().is_empty());
    }
}

#[test]
fn membership_word_keeps_budget_kind_and_per_member_error_precedence() {
    for index in [0, usize::BITS as usize] {
        let mut plan = wide_builder(CostExecution::SequentialPrefillThenDecode);
        let duplicate = member(index, AttemptWorkKind::Decode);
        let inactive_index = usize::BITS as usize + 1;
        let inactive = member(inactive_index, AttemptWorkKind::Decode);
        plan.active.as_mut().unwrap()[inactive_index] = false;

        // Count/byte bounds precede both kind and active/duplicate errors.
        plan.participant_limit = 1;
        assert_eq!(
            plan.push_group(&[duplicate, inactive]),
            Err(Unsupported::ParticipantBudget)
        );
        plan.participant_limit = usize::BITS as usize + 2;
        plan.query_limits.max_query_bytes = CostQuery::projected_bytes(1).unwrap();
        assert_eq!(
            plan.push_group(&[duplicate, inactive]),
            Err(Unsupported::ParticipantBudget)
        );
        plan.query_limits.max_query_bytes = usize::MAX;

        // Split-kind validation precedes membership checks.
        let mut prefill = duplicate;
        prefill.work.kind = AttemptWorkKind::Prefill;
        assert_eq!(
            plan.push_group(&[prefill, inactive]),
            Err(Unsupported::InvalidPlan)
        );

        // Validation remains in source member order, with active checked
        // before duplicate for each selected member, on both paths.
        assert_eq!(
            plan.push_group(&[duplicate, duplicate, inactive]),
            Err(Unsupported::InvalidPlan)
        );
        assert_eq!(
            plan.push_group(&[inactive, duplicate, duplicate]),
            Err(Unsupported::WaitingResources)
        );
        assert_eq!(
            plan.push_group(&[inactive, inactive]),
            Err(Unsupported::WaitingResources)
        );
        assert!(plan.groups().is_empty());
        plan.push_group(&[duplicate]).unwrap();
    }
}

#[test]
fn membership_word_filters_before_marking_and_failed_second_group_is_atomic() {
    let high = usize::BITS as usize;
    let mut plan = wide_builder(CostExecution::SequentialPrefillThenDecode);
    let prefills = [
        member(0, AttemptWorkKind::Prefill),
        member(high, AttemptWorkKind::Prefill),
    ];
    plan.push_group(&prefills).unwrap();
    let before = contents(&plan);
    for index in [0, high] {
        let duplicate = member(index, AttemptWorkKind::Decode);
        assert_eq!(
            plan.push_group_kind(
                &[duplicate, member(high + 1, AttemptWorkKind::Decode)],
                AttemptWorkKind::Decode
            ),
            Err(Unsupported::InvalidPlan)
        );
        assert_eq!(contents(&plan), before);
    }

    // A non-selected prefill cannot mark its request as seen in this Decode
    // group. The accepted earlier group's low/high members still participate.
    let low_valid = member(1, AttemptWorkKind::Decode);
    let valid = member(high + 1, AttemptWorkKind::Decode);
    let ignored = member(high + 1, AttemptWorkKind::Prefill);
    plan.push_group_kind(
        &[
            member(1, AttemptWorkKind::Prefill),
            low_valid,
            ignored,
            valid,
        ],
        AttemptWorkKind::Decode,
    )
    .unwrap();
    let mut expected = before;
    expected.push(vec![
        (low_valid.request, low_valid.work),
        (valid.request, valid.work),
    ]);
    assert_eq!(contents(&plan), expected);
    assert_eq!(
        plan.push_group(&[valid]),
        Err(Unsupported::ParticipantBudget)
    );

    plan.clear_groups();
    plan.push_group(&[valid]).unwrap();
    let accepted = contents(&plan);
    assert_eq!(plan.push_group(&prefills), Err(Unsupported::InvalidPlan));
    assert_eq!(contents(&plan), accepted);
}
