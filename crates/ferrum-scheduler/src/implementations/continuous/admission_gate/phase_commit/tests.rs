use super::super::super::joint_selection::PrefillCapTarget;
use super::*;
use std::num::NonZeroUsize;

// These fixtures test atomic policy publication, not a fabricated feasible
// decision. The separate admission_precommit tests obtain a real comparison.
fn pair() -> (
    ContinuousBatchScheduler,
    TimePlanningRound,
    Vec<Member>,
    PrefillCapProposal,
) {
    let (scheduler, round, _, _) = super::super::tests::fixture();
    let mut second = {
        let mut waiting = scheduler.waiting_queue.write();
        let ticket = waiting
            .iter()
            .next()
            .unwrap()
            .waiting_admission_ticket
            .unwrap();
        let first = waiting.request_mut(ticket).unwrap();
        first
            .inner
            .request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.into(), serde_json::json!(32));
        first.inner.request.clone()
    };
    second.id = RequestId::new();
    scheduler.submit_request(second).unwrap();
    let members = scheduler
        .waiting_queue
        .read()
        .iter()
        .map(|request| Member::capture(&scheduler, request).unwrap())
        .collect::<Vec<_>>();
    let proposal = PrefillCapProposal {
        targets: members
            .iter()
            .enumerate()
            .map(|(index, member)| PrefillCapTarget {
                request: index,
                key: member.key.clone(),
                phase: member.phase,
                generation: member.generation,
                prompt_tokens: member.prompt_tokens,
                prefilled_tokens: member.prefilled,
                prior_cap: member.phase_cap,
            })
            .collect(),
        limit: NonZeroUsize::new(1).unwrap(),
        packing_mode: PrefillPackingMode::Refill,
        valid_until: Instant::now() + Duration::from_secs(60),
    };
    (scheduler, round, members, proposal)
}

fn assert_no_cap(scheduler: &ContinuousBatchScheduler) {
    assert!(scheduler
        .waiting_queue
        .read()
        .iter()
        .all(|request| request.joint_prefill_phase_cap.is_none()));
    assert!(scheduler
        .prefill_queue
        .read()
        .iter()
        .all(|request| request.joint_prefill_phase_cap.is_none()));
}

#[test]
fn admission_phase_commit_revalidates_every_member_before_any_write() {
    for same_id_replacement in [false, true] {
        let (scheduler, _, members, proposal) = pair();
        let last = members.last().unwrap();
        if same_id_replacement {
            let replacement = scheduler
                .waiting_queue
                .read()
                .iter()
                .find(|request| last.key.matches(request))
                .unwrap()
                .inner
                .request
                .clone();
            assert!(scheduler.cancel_request(replacement.id.clone()).unwrap());
            scheduler.submit_request(replacement).unwrap();
        } else {
            // Generation is unchanged. The complete frontier must still catch
            // a newly scheduled, not-yet-committed work item.
            let mut waiting = scheduler.waiting_queue.write();
            waiting
                .request_mut(last.key.ticket())
                .unwrap()
                .logical_work_frontier
                .mark_scheduled(1);
        }
        assert!(!scheduler.commit_joint_admission_phase(&members, &proposal, proposal.valid_until));
        assert_no_cap(&scheduler);
    }
}

#[test]
fn admission_phase_commit_rejects_active_scan_expiry_and_changed_non_target() {
    let (scheduler, _, members, mut proposal) = pair();
    let scan = scheduler.begin_token_admission(1, None).unwrap();
    assert!(!scheduler.commit_joint_admission_phase(&members, &proposal, proposal.valid_until));
    assert_no_cap(&scheduler);
    scan.finish().unwrap();

    assert!(!scheduler.commit_joint_admission_phase(&members, &proposal, Instant::now()));
    assert_no_cap(&scheduler);
    proposal.targets.pop();
    let last = members.last().unwrap();
    scheduler
        .waiting_queue
        .write()
        .request_mut(last.key.ticket())
        .unwrap()
        .inner
        .request
        .priority = Priority::High;
    assert!(!scheduler.commit_joint_admission_phase(&members, &proposal, proposal.valid_until));
    assert_no_cap(&scheduler);
}

#[test]
fn admission_phase_commit_does_not_grant_admission_and_invalidates_old_cap_binding() {
    let (scheduler, mut round, members, proposal) = pair();
    round.admission_gate = Some(FrozenAdmissionGate {
        members: members.clone(),
        verdict: Verdict::RequiresAction,
        created_at: Instant::now(),
        apply_before: proposal.valid_until,
    });
    assert!(scheduler.commit_joint_admission_phase(&members, &proposal, proposal.valid_until));
    assert_eq!(scheduler.waiting_count(), 2);
    assert_eq!(scheduler.active_count(), 0);
    for request in scheduler.waiting_queue.read().iter() {
        let old = members
            .iter()
            .find(|member| member.key.matches(request))
            .unwrap();
        assert_eq!(request.phase, RequestPhase::Waiting);
        assert_eq!(request.logical_work_frontier, old.frontier);
        assert_eq!(request.joint_prefill_phase_cap.unwrap().limit.get(), 1);
    }
    scheduler.revalidate_joint_admission_gate(&scheduler.waiting_queue.read(), &mut round);
    assert!(
        round.admission_gate.is_none(),
        "old decision did not observe the committed caps"
    );
    assert!(!scheduler.commit_joint_admission_phase(&members, &proposal, proposal.valid_until));

    let current = scheduler
        .waiting_queue
        .read()
        .iter()
        .map(|request| Member::capture(&scheduler, request).unwrap())
        .collect::<Vec<_>>();
    let mut wider = proposal.clone();
    wider.limit = NonZeroUsize::new(2).unwrap();
    for (target, member) in wider.targets.iter_mut().zip(&current) {
        target.prior_cap = member.phase_cap;
    }
    assert!(!scheduler.commit_joint_admission_phase(&current, &wider, wider.valid_until));
    assert!(
        scheduler.waiting_queue.read().iter().all(|request| request
            .joint_prefill_phase_cap
            .unwrap()
            .limit
            .get()
            == 1),
        "a wider proposal cannot loosen an existing commitment"
    );
}
