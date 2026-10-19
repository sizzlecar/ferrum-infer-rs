//! Real scheduler transitions with controlled owner observations. Allocator
//! refcount arithmetic and device execution are qualified by their own tests.
use super::*;

fn active_requests(count: usize, decode: bool) -> (ContinuousBatchScheduler, Vec<RequestId>) {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let ids = (0..count)
        .map(|_| {
            let request = create_test_request_with_prompt_tokens(Priority::Normal, 4);
            let id = request.id.clone();
            enqueue_waiting(&scheduler, request);
            id
        })
        .collect::<Vec<_>>();
    let tick = scheduler
        .prepare_token_admission_observed(
            count,
            None,
            &mut |request| {
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                })
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(tick.admitted(), count);
    let first = scheduler
        .next_batch_with_token_admissions(BatchHint::simple(count), None, &mut |_| {}, None)
        .map(|selection| selection.batch)
        .unwrap()
        .expect("admitted prefill must be runnable");
    assert_eq!(first.requests.len(), count);
    if decode {
        for id in &ids {
            scheduler.mark_prefill_complete(id, 1);
        }
    }
    (scheduler, ids)
}

fn release_observation(
    domain: &AllocatorCapacityDomain,
    blocked: &RequestId,
    contributors: &[RequestId],
    pending: bool,
) -> ExecutionCapacityReleaseSnapshot {
    ExecutionCapacityReleaseSnapshot::default()
        .with_allocator(
            domain.clone(),
            blocked.clone(),
            contributors.to_vec(),
            pending,
        )
        .unwrap()
}

#[test]
fn allocator_capacity_pending_close_parks_prefill_and_decode_until_confirmed_release() {
    for decode in [false, true] {
        let (scheduler, ids) = active_requests(2, decode);
        let target = &ids[0];
        let domain = AllocatorCapacityDomain::new();
        // Logical cancellation transferred the physical owner to a close
        // observer; it is not the allocator's terminal release notification.
        assert!(scheduler.cancel_request(ids[1].clone()).unwrap());
        assert_eq!(scheduler.confirmed_capacity_release_epoch(), 0);
        let release = release_observation(&domain, target, &[], true);
        assert_eq!(
            scheduler
                .defer_for_allocator_capacity(target, domain.clone(), 0, &release)
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );
        assert!(scheduler.all_active_allocator_capacity_blocked());
        assert!(scheduler
            .passive_capacity_wait_condition()
            .unwrap()
            .is_none());
        let mut events = Vec::new();
        for _ in 0..2 {
            assert!(scheduler
                .next_batch_with_token_admissions(
                    BatchHint::simple(2),
                    None,
                    &mut |event| events.push(event),
                    None
                )
                .map(|selection| selection.batch)
                .unwrap()
                .is_none());
        }
        assert!(events.iter().any(|event| matches!(event,
            ExecutorAdmissionQueueObservation::AllocatorCapacitySkippedUnchanged {
                request_id, observed_release_epoch: 0, current_release_epoch: 0, ..
            } if request_id == target)));
        assert_eq!(scheduler.confirmed_capacity_release_epoch(), 0);
        scheduler.record_external_capacity_release();
        assert!(!scheduler.all_active_allocator_capacity_blocked());
        let resumed = scheduler
            .next_batch_with_token_admissions(
                BatchHint::simple(2),
                None,
                &mut |event| events.push(event),
                None,
            )
            .map(|selection| selection.batch)
            .unwrap()
            .unwrap();
        assert_eq!(resumed.requests.len(), 1);
        assert_eq!(&resumed.requests[0].request.id, target);
        assert!(events.iter().any(|event| matches!(event,
            ExecutorAdmissionQueueObservation::AllocatorCapacityResumed {
                request_id, observed_release_epoch: 0, current_release_epoch: 1, ..
            } if request_id == target)));
        assert_eq!(
            scheduler.trace_snapshot().pressure_active_episodes,
            1,
            "a release permits actual re-probe, not successful execution"
        );
    }
}

#[test]
fn allocator_capacity_release_during_execution_is_not_lost_at_deferral_installation() {
    let (scheduler, ids) = active_requests(2, true);
    let before_execution = scheduler.confirmed_capacity_release_epoch();
    let domain = AllocatorCapacityDomain::new();
    scheduler.record_external_capacity_release();
    let release = release_observation(&domain, &ids[0], std::slice::from_ref(&ids[1]), false);
    assert_eq!(
        scheduler
            .defer_for_allocator_capacity(&ids[0], domain, before_execution, &release)
            .unwrap(),
        ExecutionCapacityAction::Deferred { count: 1 }
    );
    assert!(!scheduler.all_active_allocator_capacity_blocked());
    let batch = scheduler
        .next_batch_with_token_admissions(BatchHint::simple(2), None, &mut |_| {}, None)
        .map(|selection| selection.batch)
        .unwrap()
        .unwrap();
    assert!(batch.requests.iter().any(|work| work.request.id == ids[0]));
}

#[test]
fn allocator_capacity_failed_close_invalidates_wait_without_claiming_release() {
    for decode in [false, true] {
        let (scheduler, ids) = active_requests(1, decode);
        let target = &ids[0];
        let domain = AllocatorCapacityDomain::new();
        let pending = release_observation(&domain, target, &[], true);
        assert_eq!(
            scheduler
                .defer_for_allocator_capacity(target, domain.clone(), 0, &pending)
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 },
        );
        assert!(scheduler.all_active_allocator_capacity_blocked());
        assert!(scheduler
            .next_batch_with_token_admissions(BatchHint::simple(1), None, &mut |_| {}, None)
            .map(|selection| selection.batch)
            .unwrap()
            .is_none());
        let journal = scheduler.pressure_transition_journal();

        // The external observer failed without terminal proof. Clearing the
        // stale retry predicate permits real execution, not a capacity grant.
        assert!(scheduler.invalidate_allocator_capacity_waits());
        assert!(!scheduler.invalidate_allocator_capacity_waits());
        assert_eq!(scheduler.confirmed_capacity_release_epoch(), 0);
        assert_eq!(scheduler.pressure_transition_journal(), journal);
        assert!(!scheduler.all_active_allocator_capacity_blocked());
        let batch = scheduler
            .next_batch_with_token_admissions(BatchHint::simple(1), None, &mut |_| {}, None)
            .map(|selection| selection.batch)
            .unwrap()
            .expect("an invalidated external wait must permit re-probe");
        assert_eq!(batch.requests.len(), 1);
        assert_eq!(&batch.requests[0].request.id, target);

        // The next exact observation has no release witness. The old pending
        // witness must not silently keep this request parked forever.
        let failed = release_observation(&domain, target, &[], false);
        assert!(matches!(
            scheduler.defer_for_allocator_capacity(target, domain, 0, &failed),
            Err(FerrumError::ResourceExhausted { .. }),
        ));
        assert_eq!(scheduler.confirmed_capacity_release_epoch(), 0);
        assert!(!scheduler.all_active_allocator_capacity_blocked());
    }
}

#[test]
fn allocator_capacity_invalidation_preserves_exact_fence_and_recompute_hold() {
    let (scheduler, ids) = active_requests(2, true);
    let owner = &ids[0];
    let victim = &ids[1];
    let domain = AllocatorCapacityDomain::new();
    for request in [victim, owner] {
        let pending = release_observation(&domain, request, &[], true);
        assert_eq!(
            scheduler
                .defer_for_allocator_capacity(request, domain.clone(), 0, &pending)
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 },
        );
    }

    // A new query replaces the external pending witness with the actual
    // blocked contributor. The same pressure episode must plan its release.
    let release = release_observation(&domain, owner, std::slice::from_ref(victim), false);
    let ExecutionCapacityAction::YieldPlanned { transaction } = scheduler
        .defer_for_allocator_capacity(owner, domain, 0, &release)
        .unwrap()
    else {
        panic!("a vanished pending witness must not suppress the exact release plan");
    };
    assert_eq!(transaction.progress_owner_id(), owner);
    assert_eq!(transaction.victim_request_id(), victim);
    let journal = scheduler.pressure_transition_journal();
    assert!(!scheduler.invalidate_allocator_capacity_waits());
    assert_eq!(scheduler.pressure_transition_journal(), journal);
    scheduler
        .arm_execution_capacity_yield(&transaction)
        .unwrap();
    let journal = scheduler.pressure_transition_journal();
    assert!(!scheduler.invalidate_allocator_capacity_waits());
    assert_eq!(scheduler.confirmed_capacity_release_epoch(), 0);
    assert_eq!(scheduler.pressure_transition_journal(), journal);
    assert_eq!(
        scheduler
            .pressure_coordinator
            .lock()
            .stats()
            .pending_release_fences,
        1
    );
    assert!(matches!(
        scheduler.pressure_coordinator.lock().hold_status(victim),
        PressureHoldStatus::Held { .. },
    ));
    assert!(scheduler
        .next_batch_with_token_admissions(BatchHint::simple(2), None, &mut |_| {}, None)
        .map(|selection| selection.batch)
        .unwrap()
        .is_none());

    // Only the original terminal fence may requeue the victim. A later
    // unrelated observation failure must still preserve its pressure hold.
    assert!(scheduler
        .complete_execution_capacity_yield(&transaction, 2, Some(0))
        .unwrap()
        .victim_requeued());
    scheduler.record_external_capacity_release_after_recompute();
    assert!(scheduler.invalidate_allocator_capacity_waits());
    assert_eq!(scheduler.confirmed_capacity_release_epoch(), 1);
    assert!(matches!(
        scheduler.pressure_coordinator.lock().hold_status(victim),
        PressureHoldStatus::Held { .. },
    ));
    let tick = scheduler
        .prepare_token_admission_observed(
            2,
            None,
            &mut |_| panic!("held contributor must not be readmitted"),
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(tick.admitted(), 0);
    let batch = scheduler
        .next_batch_with_token_admissions(BatchHint::simple(2), None, &mut |_| {}, None)
        .map(|selection| selection.batch)
        .unwrap()
        .unwrap();
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(&batch.requests[0].request.id, owner);
}

#[test]
fn allocator_capacity_without_a_contributor_fails_without_parking_or_cross_pool_aliasing() {
    let (scheduler, ids) = active_requests(1, true);
    let domain = AllocatorCapacityDomain::new();
    let empty = release_observation(&domain, &ids[0], &[], false);
    assert!(matches!(
        scheduler.defer_for_allocator_capacity(&ids[0], domain.clone(), 0, &empty),
        Err(FerrumError::ResourceExhausted { .. })
    ));
    let foreign = release_observation(&AllocatorCapacityDomain::new(), &ids[0], &[], true);
    assert!(scheduler
        .defer_for_allocator_capacity(&ids[0], domain.clone(), 0, &foreign)
        .is_err());
    let wrong_owner = release_observation(&domain, &RequestId::new(), &[], true);
    assert!(scheduler
        .defer_for_allocator_capacity(&ids[0], domain.clone(), 0, &wrong_owner)
        .is_err());
    assert!(ExecutionCapacityReleaseSnapshot::default()
        .with_allocator(domain.clone(), ids[0].clone(), vec![ids[0].clone()], false,)
        .is_err());
    let duplicate = RequestId::new();
    assert!(ExecutionCapacityReleaseSnapshot::default()
        .with_allocator(
            domain,
            ids[0].clone(),
            vec![duplicate.clone(), duplicate],
            false,
        )
        .is_err());
    assert!(!scheduler.all_active_allocator_capacity_blocked());
    assert_eq!(scheduler.trace_snapshot().pressure_active_episodes, 0);
    assert!(scheduler
        .next_batch_with_token_admissions(BatchHint::simple(1), None, &mut |_| {}, None)
        .map(|selection| selection.batch)
        .unwrap()
        .is_some());
}

#[test]
fn allocator_capacity_joint_contributors_yield_in_order_without_readmitting_the_first() {
    let (scheduler, ids) = active_requests(3, true);
    let owner = &ids[0];
    let domain = AllocatorCapacityDomain::new();
    // Both peers are blocked on the same allocator. Their individual reclaim
    // can be zero: the actual model query has proved their joint ref removal
    // sufficient and returns both as contributors for `owner`.
    for (blocked, releaser) in [(&ids[1], &ids[2]), (&ids[2], owner)] {
        let release = release_observation(&domain, blocked, std::slice::from_ref(releaser), false);
        assert_eq!(
            scheduler
                .defer_for_allocator_capacity(blocked, domain.clone(), 0, &release)
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );
    }
    let release = release_observation(&domain, owner, &ids[1..], false);
    let ExecutionCapacityAction::YieldPlanned { transaction: first } = scheduler
        .defer_for_allocator_capacity(owner, domain.clone(), 0, &release)
        .unwrap()
    else {
        panic!("all blocked contributors require a real release transaction");
    };
    assert_eq!(first.progress_owner_id(), owner);
    let first_victim = first.victim_request_id().clone();
    let remaining = ids[1..]
        .iter()
        .find(|id| **id != first_victim)
        .unwrap()
        .clone();
    scheduler.arm_execution_capacity_yield(&first).unwrap();
    assert_eq!(scheduler.confirmed_capacity_release_epoch(), 0);
    let completed = scheduler
        .complete_execution_capacity_yield(&first, 3, Some(0))
        .unwrap();
    assert!(completed.victim_requeued());
    scheduler.record_external_capacity_release_after_recompute();
    assert_eq!(scheduler.confirmed_capacity_release_epoch(), 1);
    assert!(matches!(
        scheduler
            .pressure_coordinator
            .lock()
            .hold_status(&first_victim),
        PressureHoldStatus::Held { .. }
    ));

    // Real cross-iteration selection must preserve the remaining contributor's
    // blocked state while only the accepted owner re-probes the released pool.
    let retry = scheduler
        .next_batch_with_token_admissions(BatchHint::simple(3), None, &mut |_| {}, None)
        .map(|selection| selection.batch)
        .unwrap()
        .unwrap();
    assert_eq!(retry.requests.len(), 1);
    assert_eq!(&retry.requests[0].request.id, owner);
    assert!(matches!(
        scheduler.decode_queue.read().requests[&remaining]
            .execution_capacity_deferral
            .as_ref(),
        Some(ExecutionCapacityDeferral::Allocator {
            release_epoch: 0,
            ..
        })
    ));

    // The first real close withdrew references but freed no block. The next
    // authoritative rejection observes the remaining contributor in that pool.
    let release = release_observation(&domain, owner, std::slice::from_ref(&remaining), false);
    let ExecutionCapacityAction::YieldPlanned {
        transaction: second,
    } = scheduler
        .defer_for_allocator_capacity(owner, domain, 1, &release)
        .unwrap()
    else {
        panic!("the remaining joint contributor must release before retry");
    };
    assert_eq!(second.episode_id(), first.episode_id());
    assert_eq!(second.progress_owner_id(), owner);
    assert_eq!(second.victim_request_id(), &remaining);
    assert!(matches!(
        scheduler
            .pressure_coordinator
            .lock()
            .hold_status(&first_victim),
        PressureHoldStatus::Held { .. }
    ));
    scheduler.arm_execution_capacity_yield(&second).unwrap();
    let completed = scheduler
        .complete_execution_capacity_yield(&second, 2, Some(1))
        .unwrap();
    assert!(completed.victim_requeued());
    scheduler.record_external_capacity_release_after_recompute();
    let tick = scheduler
        .prepare_token_admission_observed(
            3,
            None,
            &mut |_| panic!("yielded contributors must not consume the recovered capacity"),
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(tick.admitted(), 0);
    let batch = scheduler
        .next_batch_with_token_admissions(BatchHint::simple(3), None, &mut |_| {}, None)
        .map(|selection| selection.batch)
        .unwrap()
        .unwrap();
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(&batch.requests[0].request.id, owner);
    scheduler.update_decode_progress(owner, 1);
    assert_eq!(scheduler.confirmed_capacity_release_epoch(), 2);
    assert!(matches!(
        scheduler
            .pressure_coordinator
            .lock()
            .hold_status(&first_victim),
        PressureHoldStatus::Held { .. }
    ));
    assert!(scheduler.cancel_request(owner.clone()).unwrap());
    assert!(matches!(
        scheduler
            .pressure_coordinator
            .lock()
            .hold_status(&first_victim),
        PressureHoldStatus::Released { .. }
    ));
}

#[test]
fn allocator_capacity_owner_progress_reopens_peer_without_fabricating_another_release() {
    let (scheduler, ids) = active_requests(3, true);
    let owner = &ids[0];
    let domain = AllocatorCapacityDomain::new();
    for (blocked, releaser) in [(&ids[1], &ids[2]), (&ids[2], owner)] {
        let release = release_observation(&domain, blocked, std::slice::from_ref(releaser), false);
        assert!(matches!(
            scheduler
                .defer_for_allocator_capacity(blocked, domain.clone(), 0, &release)
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        ));
    }
    let release = release_observation(&domain, owner, &ids[1..], false);
    let ExecutionCapacityAction::YieldPlanned { transaction } = scheduler
        .defer_for_allocator_capacity(owner, domain.clone(), 0, &release)
        .unwrap()
    else {
        panic!("blocked contributors must yield to the accepted owner");
    };
    let remaining = ids[1..]
        .iter()
        .find(|id| *id != transaction.victim_request_id())
        .unwrap();
    scheduler
        .arm_execution_capacity_yield(&transaction)
        .unwrap();
    assert!(scheduler
        .complete_execution_capacity_yield(&transaction, 3, Some(1))
        .unwrap()
        .victim_requeued());
    scheduler.record_external_capacity_release_after_recompute();
    let mut observations = Vec::new();
    let retry = scheduler
        .next_batch_with_token_admissions(
            BatchHint::simple(3),
            None,
            &mut |event| observations.push(event),
            None,
        )
        .map(|selection| selection.batch)
        .unwrap()
        .unwrap();
    assert_eq!(retry.requests.len(), 1);
    assert_eq!(&retry.requests[0].request.id, owner);
    assert!(observations.iter().any(|event| matches!(event,
        ExecutorAdmissionQueueObservation::AllocatorCapacityWaitingForOwner {
            request_id, progress_owner_id, observed_release_epoch: 0, current_release_epoch: 1, ..
        } if request_id == remaining && progress_owner_id == owner)));

    // Committing a token, not rescheduling the same work, ends the owner's
    // exclusive re-probe. Keep the real release epoch at one.
    scheduler.update_decode_progress(owner, 2);
    let retry = scheduler
        .next_batch_with_token_admissions(BatchHint::simple(3), None, &mut |_| {}, None)
        .map(|selection| selection.batch)
        .unwrap()
        .unwrap();
    assert_eq!(retry.requests.len(), 2);
    assert!(retry
        .requests
        .iter()
        .any(|request| &request.request.id == remaining));
    assert_eq!(scheduler.confirmed_capacity_release_epoch(), 1);

    // A peer that still cannot fit waits for the live, progressed contributor;
    // its exclusion-based query cannot seize the existing episode's holds.
    let release = release_observation(&domain, remaining, std::slice::from_ref(owner), false);
    assert!(matches!(
        scheduler
            .defer_for_allocator_capacity(remaining, domain.clone(), 1, &release)
            .unwrap(),
        ExecutionCapacityAction::Deferred { count: 1 }
    ));
    assert_eq!(scheduler.confirmed_capacity_release_epoch(), 1);
    let release = release_observation(&domain, owner, std::slice::from_ref(remaining), false);
    let ExecutionCapacityAction::YieldPlanned { transaction: next } = scheduler
        .defer_for_allocator_capacity(owner, domain, 1, &release)
        .unwrap()
    else {
        panic!("the original owner still owns the next actual pressure decision");
    };
    assert_eq!(next.progress_owner_id(), owner);
    assert_eq!(next.episode_id(), transaction.episode_id());
}
