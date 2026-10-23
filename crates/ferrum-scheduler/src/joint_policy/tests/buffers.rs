use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

mod group_views;
mod homogeneous;
mod membership;

fn builder(execution: CostExecution) -> PlanBuilder {
    PlanBuilder {
        work: Vec::new(),
        requests: Vec::new(),
        group_ends: [0; 2],
        members: Vec::new(),
        group_count: 0,
        participant_limit: 4,
        query_limits: CostQueryLimits::default(),
        execution,
        active: Some(vec![true; 4]),
    }
}

fn member(request: usize, kind: AttemptWorkKind) -> GroupMember {
    let mut work = decode(&super::request());
    work.kind = kind;
    GroupMember { request, work }
}

fn contents(plan: &PlanBuilder) -> Vec<Vec<(usize, CostWork)>> {
    plan.groups()
        .map(|group| group.iter().map(|m| (m.request, m.work)).collect())
        .collect()
}

#[test]
fn reused_members_discard_failed_builds_without_rolling_back_accepted_groups() {
    let mut plan = builder(CostExecution::SequentialPrefillThenDecode);
    let prefill = member(0, AttemptWorkKind::Prefill);
    let decode = member(1, AttemptWorkKind::Decode);
    assert_eq!(
        plan.with_members(|members, plan| {
            members.extend([prefill, member(0, AttemptWorkKind::Decode)]);
            plan.push_group_kind(members, AttemptWorkKind::Prefill)?;
            plan.push_group_kind(members, AttemptWorkKind::Decode)
        }),
        Err(Unsupported::InvalidPlan)
    );
    // As before, rejection of the second split group leaves the accepted
    // first group intact. Scratch reuse does not make the wave transactional.
    assert_eq!(contents(&plan), vec![vec![(0, prefill.work)]]);
    plan.with_members(|members, plan| {
        assert!(members.is_empty());
        assert_eq!(plan.active_mask(), Some([true; 4].as_slice()));
        members.push(decode);
        plan.push_group(members)
    })
    .unwrap();
    assert_eq!(
        contents(&plan),
        vec![vec![(0, prefill.work)], vec![(1, decode.work)]]
    );
    plan.clear_groups();
    assert_eq!(
        plan.with_members(|members, _| {
            assert!(members.is_empty());
            members.push(prefill);
            Err(Unsupported::NoProgress)
        }),
        Err(Unsupported::NoProgress)
    );
    assert!(plan.groups().len() == 0);
    plan.with_members(|members, plan| {
        assert!(members.is_empty());
        members.push(decode);
        plan.push_group(members)
    })
    .unwrap();
    assert_eq!(contents(&plan), vec![vec![(1, decode.work)]]);
}

#[test]
fn filtered_groups_preserve_order_and_discard_prior_wave_members() {
    let members = [
        member(3, AttemptWorkKind::Decode),
        member(2, AttemptWorkKind::Prefill),
        member(1, AttemptWorkKind::Decode),
        member(0, AttemptWorkKind::Prefill),
    ];
    let mut filtered = builder(CostExecution::SequentialPrefillThenDecode);
    let mut slices = builder(CostExecution::SequentialPrefillThenDecode);
    for kind in [AttemptWorkKind::Prefill, AttemptWorkKind::Decode] {
        filtered.push_group_kind(&members, kind).unwrap();
        let original = members
            .iter()
            .copied()
            .filter(|m| m.work.kind == kind)
            .collect::<Vec<_>>();
        slices.push_group(&original).unwrap();
    }
    assert_eq!(contents(&filtered), contents(&slices));
    assert_eq!(
        contents(&filtered)
            .into_iter()
            .map(|group| group.into_iter().map(|m| m.0).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec![2, 0], vec![3, 1]]
    );

    filtered.clear_groups();
    filtered
        .push_group_kind(&members[..1], AttemptWorkKind::Prefill)
        .unwrap();
    assert!(filtered.groups().len() == 0);
    filtered
        .push_group_kind(&members[..1], AttemptWorkKind::Decode)
        .unwrap();
    assert_eq!(contents(&filtered), vec![vec![(3, members[0].work)]]);
    filtered.clear_groups();
    filtered.push_group(&members[1..2]).unwrap();
    assert_eq!(contents(&filtered), vec![vec![(2, members[1].work)]]);
}

#[test]
fn rejected_groups_leave_prior_work_and_resource_guards_intact() {
    let prefill = member(0, AttemptWorkKind::Prefill);
    let decode = member(1, AttemptWorkKind::Decode);
    let mut plan = builder(CostExecution::SequentialPrefillThenDecode);
    plan.push_group(&[prefill]).unwrap();
    let before = contents(&plan);
    let duplicate = member(0, AttemptWorkKind::Decode);
    assert_eq!(
        plan.push_group_kind(&[prefill, duplicate, decode], AttemptWorkKind::Decode),
        Err(Unsupported::InvalidPlan)
    );
    assert_eq!(
        plan.push_group_kind(&[prefill, decode, decode], AttemptWorkKind::Decode),
        Err(Unsupported::InvalidPlan)
    );
    assert_eq!(contents(&plan), before);
    assert_eq!(plan.push_group(&[duplicate]), Err(Unsupported::InvalidPlan));
    assert_eq!(
        plan.push_group(&[decode, decode]),
        Err(Unsupported::InvalidPlan)
    );
    assert_eq!(plan.push_group(&[prefill]), Err(Unsupported::InvalidPlan));
    assert_eq!(contents(&plan), before);
    plan.active.as_mut().unwrap()[1] = false;
    assert_eq!(
        plan.push_group_kind(&[prefill, decode], AttemptWorkKind::Decode),
        Err(Unsupported::WaitingResources)
    );
    assert_eq!(
        plan.push_group(&[decode]),
        Err(Unsupported::WaitingResources)
    );
    assert_eq!(contents(&plan), before);
    plan.active.as_mut().unwrap()[1] = true;
    plan.participant_limit = 1;
    assert_eq!(
        plan.push_group(&[decode]),
        Err(Unsupported::ParticipantBudget)
    );
    assert_eq!(contents(&plan), before);
    plan.participant_limit = 4;
    plan.push_group_kind(&[prefill, decode], AttemptWorkKind::Decode)
        .unwrap();
    assert_eq!(
        plan.push_group(&[member(2, AttemptWorkKind::Decode)]),
        Err(Unsupported::ParticipantBudget)
    );

    plan.clear_groups();
    plan.execution = CostExecution::SingleInvocation;
    plan.push_group(&[decode, prefill]).unwrap();
    let before = contents(&plan);
    assert_eq!(
        plan.push_group(&[member(2, AttemptWorkKind::Decode)]),
        Err(Unsupported::InvalidPlan)
    );
    assert_eq!(contents(&plan), before);
}

#[test]
fn reused_slot_mask_tracks_release_and_rejects_premature_waiter_progress() {
    let slots = AdmissionSlotPlan::new(
        &[
            InitialAdmission::Resident,
            InitialAdmission::Waiting { order: 0 },
        ],
        1,
        2,
    )
    .unwrap();
    let mut requests = vec![request(), request()];
    requests[1].prefill_tokens = 0;
    requests[1].output_tokens = 0;
    requests[1].central_clock = RequestClock::default();
    requests[1].risk_clock = RequestClock::default();
    let mut active = vec![true, true, true];
    slots.active_mask_into(&requests, &mut active).unwrap();
    assert_eq!(active, [true, false]);
    let frozen = active.clone();
    requests[0].output_tokens = requests[0].scenario_end_tokens;
    assert_eq!(
        frozen,
        [true, false],
        "a prior group's completion cannot change its wave mask"
    );
    slots.active_mask_into(&requests, &mut active).unwrap();
    assert_eq!(active, [false, true]);
    requests[0].output_tokens -= 1;
    slots.active_mask_into(&requests, &mut active).unwrap();
    assert_eq!(active, [true, false]);
    requests[1].prefill_tokens = 1;
    assert_eq!(
        slots.active_mask_into(&requests, &mut active),
        Err(Unsupported::InvalidState)
    );
    assert_eq!(
        slots.active_mask_into(&requests[..1], &mut active),
        Err(Unsupported::InvalidState)
    );
}

struct ChangingLimits {
    snapshot: Snapshot,
    calls: AtomicUsize,
    panic: bool,
}

impl CostSnapshot for ChangingLimits {
    fn revision(&self) -> u64 {
        7
    }
    fn query_limits(&self) -> CostQueryLimits {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        if call > 0 && self.panic {
            panic!("later wave query limits failed");
        }
        let mut limits = CostQueryLimits::default();
        if call > 0 {
            limits.max_participants = 1;
        }
        limits
    }
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        self.snapshot.predict(query, now)
    }
}

#[test]
fn reused_plan_still_refreshes_and_contains_provider_limits_each_wave() {
    for panic in [false, true] {
        let at = Instant::now();
        let snapshot = ChangingLimits {
            snapshot: Snapshot { at, split: false },
            calls: AtomicUsize::new(0),
            panic,
        };
        let result = SloPolicy::evaluate(EvaluationInput {
            snapshot: &snapshot,
            now: at,
            config: &config(),
            execution: CostExecution::SingleInvocation,
            requests: &[request(), request()],
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &Adapter,
            static_action: &Action::Full,
            alternatives: &[],
            horizon: EvaluationHorizon::StaticRounds(2),
        });
        assert_eq!(
            result.unwrap_err(),
            if panic {
                Unsupported::Cost(CostUnknownReason::UnsupportedDomain)
            } else {
                Unsupported::ParticipantBudget
            }
        );
        assert_eq!(snapshot.calls.load(Ordering::Relaxed), 2);
    }
}
