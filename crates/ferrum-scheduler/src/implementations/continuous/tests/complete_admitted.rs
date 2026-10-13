use super::*;

fn hint(tokens: usize) -> BatchHint {
    BatchHint {
        max_batch_size: 4,
        max_tokens: tokens,
        target_latency_ms: None,
        available_memory: None,
        resource_constraints: Default::default(),
    }
}

fn scheduled(batch: &BatchPlan, id: &RequestId) -> Option<usize> {
    batch
        .requests
        .iter()
        .find(|request| request.request.id == *id)
        .and_then(|request| request.tokens_to_process)
}

fn initial(
    policy: PrefillFirstPolicy,
    capacity: usize,
) -> (ContinuousBatchScheduler, Vec<RequestId>) {
    initial_with_budget(policy, capacity, None)
}

fn initial_with_budget(
    policy: PrefillFirstPolicy,
    capacity: usize,
    active_budget: Option<usize>,
) -> (ContinuousBatchScheduler, Vec<RequestId>) {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: capacity,
        prompt_token_estimate: true,
        prefill_first_until_active: Some(4),
        prefill_first_policy: policy,
        active_decode_prefill_token_budget: active_budget,
        ..SchedulerConfig::default()
    });
    let ids = [128, 128, 976, 128]
        .map(|tokens| {
            let request = create_test_request_with_prompt_tokens(Priority::Normal, tokens);
            let id = request.id.clone();
            enqueue_waiting(&scheduler, request);
            id
        })
        .to_vec();
    let first = scheduler.create_iteration_batch(hint(256)).unwrap();
    assert_eq!(scheduled(&first, &ids[0]), Some(128));
    assert_eq!(scheduled(&first, &ids[1]), Some(128));
    scheduler.mark_prefill_complete(&ids[0], 128);
    scheduler.mark_prefill_complete(&ids[1], 128);
    assert_eq!(scheduler.active_count(), 4);
    assert_eq!(scheduler.decoding_count(), 2);
    (scheduler, ids)
}

#[test]
fn prefill_first_complete_admitted_preserves_default_and_finishes_long_tail_with_live_budget() {
    for policy in [
        PrefillFirstPolicy::ActiveTarget,
        PrefillFirstPolicy::CompleteAdmitted,
    ] {
        let (scheduler, ids) = initial(policy, 4);
        let second = scheduler.create_iteration_batch(hint(512)).unwrap();
        if policy == PrefillFirstPolicy::ActiveTarget {
            assert_eq!(scheduled(&second, &ids[0]), Some(1));
            assert_eq!(scheduled(&second, &ids[1]), Some(1));
            assert_eq!(
                scheduled(&second, &ids[2]),
                Some(scheduler.default_active_decode_prefill_chunk())
            );
            continue;
        }
        assert_eq!(scheduled(&second, &ids[0]), None);
        assert_eq!(scheduled(&second, &ids[1]), None);
        assert_eq!(scheduled(&second, &ids[2]), Some(512));
        assert!(!scheduler.mark_prefill_chunk_processed(&ids[2], 976, 512));
        let third = scheduler.create_iteration_batch(hint(512)).unwrap();
        assert_eq!(scheduled(&third, &ids[2]), Some(464));
        assert_eq!(scheduled(&third, &ids[3]), Some(48));
        assert!(scheduler.mark_prefill_chunk_processed(&ids[2], 976, 464));
        assert!(!scheduler.mark_prefill_chunk_processed(&ids[3], 128, 48));
        let fourth = scheduler.create_iteration_batch(hint(512)).unwrap();
        assert_eq!(fourth.requests.len(), 1);
        assert_eq!(scheduled(&fourth, &ids[3]), Some(80));
        assert!(scheduler.mark_prefill_chunk_processed(&ids[3], 128, 80));
        let decode = scheduler.create_iteration_batch(hint(512)).unwrap();
        assert!(ids.iter().all(|id| scheduled(&decode, id) == Some(1)));
        assert!(!scheduler
            .fill_first_initial_cohort_armed
            .load(Ordering::Acquire));
    }
}

#[tokio::test]
async fn prefill_first_complete_admitted_late_arrival_does_not_extend_the_sealed_cohort() {
    let (scheduler, ids) = initial(PrefillFirstPolicy::CompleteAdmitted, 5);
    let second = scheduler.create_iteration_batch(hint(512)).unwrap();
    assert_eq!(scheduled(&second, &ids[2]), Some(512));
    assert!(!scheduler.mark_prefill_chunk_processed(&ids[2], 976, 512));
    assert!(scheduler.cancel(ids[3].clone()).await.unwrap());
    let late = create_test_request_with_prompt_tokens(Priority::Normal, 512);
    let late_id = scheduler.submit(late).await.unwrap();
    let tail = scheduler.create_iteration_batch(hint(1024)).unwrap();
    assert_eq!(scheduled(&tail, &ids[2]), Some(464));
    assert_eq!(scheduled(&tail, &late_id), None);
    assert_eq!(
        scheduler.request_index.read().get(&late_id),
        Some(&RequestPhase::Prefilling)
    );
    assert!(scheduler.mark_prefill_chunk_processed(&ids[2], 976, 464));
    let released = scheduler.create_iteration_batch(hint(1024)).unwrap();
    assert!(ids[..3]
        .iter()
        .all(|id| scheduled(&released, id) == Some(1)));
    assert!(!scheduler
        .fill_first_initial_cohort_armed
        .load(Ordering::Acquire));
}

#[tokio::test]
async fn prefill_first_complete_admitted_cancelled_cohort_drains_and_rearms_only_after_idle() {
    let (scheduler, ids) = initial(PrefillFirstPolicy::CompleteAdmitted, 4);
    scheduler.create_iteration_batch(hint(512)).unwrap();
    for id in &ids {
        assert!(scheduler.cancel(id.clone()).await.unwrap());
    }
    assert_eq!(scheduler.active_count(), 0);
    assert_eq!(scheduler.waiting_count(), 0);
    let mut next = Vec::new();
    for tokens in [128, 128, 976, 128] {
        next.push(
            scheduler
                .submit(create_test_request_with_prompt_tokens(
                    Priority::Normal,
                    tokens,
                ))
                .await
                .unwrap(),
        );
    }
    let first = scheduler.create_iteration_batch(hint(256)).unwrap();
    assert_eq!(scheduled(&first, &next[0]), Some(128));
    assert_eq!(scheduled(&first, &next[1]), Some(128));
    scheduler.mark_prefill_complete(&next[0], 128);
    scheduler.mark_prefill_complete(&next[1], 128);
    let cold = scheduler.create_iteration_batch(hint(512)).unwrap();
    assert_eq!(scheduled(&cold, &next[0]), None);
    assert_eq!(scheduled(&cold, &next[2]), Some(512));
    assert!(scheduler.cancel(next[2].clone()).await.unwrap());
    assert!(scheduler.cancel(next[3].clone()).await.unwrap());
    let drained = scheduler.create_iteration_batch(hint(512)).unwrap();
    assert_eq!(scheduled(&drained, &next[0]), Some(1));
    assert_eq!(scheduled(&drained, &next[1]), Some(1));
}

#[test]
fn prefill_first_complete_admitted_no_runnable_prefill_escapes_to_decode_and_disarms() {
    let (scheduler, ids) = initial(PrefillFirstPolicy::CompleteAdmitted, 4);
    scheduler.create_iteration_batch(hint(512)).unwrap();
    let readiness = scheduler.defer_for_execution_readiness(&ids[2..]).unwrap();
    assert_eq!(readiness.deferred_count(), 2);
    let fallback = scheduler.create_iteration_batch(hint(512)).unwrap();
    assert_eq!(fallback.requests.len(), 2);
    assert!(ids[..2]
        .iter()
        .all(|id| scheduled(&fallback, id) == Some(1)));
    assert!(!scheduler
        .fill_first_initial_cohort_armed
        .load(Ordering::Acquire));
    drop(readiness);
}

#[test]
fn prefill_first_complete_admitted_capacity_deferred_member_is_not_treated_as_complete() {
    let (scheduler, ids) = initial(PrefillFirstPolicy::CompleteAdmitted, 4);
    scheduler.create_iteration_batch(hint(512)).unwrap();
    assert!(scheduler.defer_prefill_to_waiting(&ids[2]));
    // The real deferred frontier remains tracked as Waiting, not completed.
    assert_eq!(
        scheduler.request_index.read().get(&ids[2]),
        Some(&RequestPhase::Waiting)
    );
    assert!(scheduler
        .initial_admitted_prefills_for_iteration(true)
        .unwrap()
        .contains(&ids[2]));
    let pressure = scheduler.create_iteration_batch(hint(512)).unwrap();
    assert!(ids[..2]
        .iter()
        .all(|id| scheduled(&pressure, id) == Some(1)));
    assert!(!scheduler
        .fill_first_initial_cohort_armed
        .load(Ordering::Acquire));
}

#[test]
fn prefill_first_complete_admitted_explicit_active_decode_budget_keeps_decode_priority() {
    let (scheduler, ids) = initial_with_budget(PrefillFirstPolicy::CompleteAdmitted, 4, Some(19));
    let budgeted = scheduler.create_iteration_batch(hint(512)).unwrap();
    assert!(ids[..2]
        .iter()
        .all(|id| scheduled(&budgeted, id) == Some(1)));
    let prefill_tokens = budgeted
        .requests
        .iter()
        .filter(|request| !ids[..2].contains(&request.request.id))
        .map(|request| request.tokens_to_process.unwrap())
        .sum::<usize>();
    assert!(prefill_tokens <= 19);
    assert!(!scheduler
        .fill_first_initial_cohort_armed
        .load(Ordering::Acquire));
}
