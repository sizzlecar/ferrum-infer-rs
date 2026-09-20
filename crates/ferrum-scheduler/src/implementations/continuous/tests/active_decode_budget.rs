use super::*;

fn hint(sequences: usize, tokens: usize) -> BatchHint {
    BatchHint {
        max_batch_size: sequences,
        max_tokens: tokens,
        target_latency_ms: None,
        available_memory: None,
        resource_constraints: Default::default(),
    }
}

fn waiting_prompt(scheduler: &ContinuousBatchScheduler, tokens: usize) -> RequestId {
    let request = create_test_request_with_prompt_tokens(Priority::Normal, tokens);
    let id = request.id.clone();
    enqueue_waiting(scheduler, request);
    id
}

fn budgeted_scheduler(budget: usize) -> ContinuousBatchScheduler {
    ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 8,
        active_decode_prefill_token_budget: Some(budget),
        ..SchedulerConfig::default()
    })
}

fn scheduled_tokens(batch: &BatchPlan, request_id: &RequestId) -> Option<usize> {
    batch
        .requests
        .iter()
        .find(|request| request.request.id == *request_id)
        .and_then(|request| request.tokens_to_process)
}

#[test]
fn active_decode_prefill_token_budget_caps_the_whole_iteration_with_request_caps() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 8,
        active_decode_prefill_chunk: Some(8),
        active_decode_prefill_token_budget: Some(19),
        ..SchedulerConfig::default()
    });
    let decode = activate_decode_requests(&scheduler, 1).remove(0);
    let prefills = (0..3)
        .map(|_| waiting_prompt(&scheduler, 100))
        .collect::<Vec<_>>();
    let batch = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&batch, &decode), Some(1));
    assert_eq!(
        prefills
            .iter()
            .map(|id| scheduled_tokens(&batch, id))
            .collect::<Vec<_>>(),
        vec![Some(8), Some(8), Some(3)],
        "the total budget must be shared, not multiplied by free sequence slots"
    );
    assert_eq!(batch.resource_requirements.gpu_memory, (1 + 19) * 16);
}

#[test]
fn active_decode_prefill_token_budget_applies_below_pressure_threshold_and_live_limits() {
    let scheduler = budgeted_scheduler(19);
    let decode = activate_decode_requests(&scheduler, 1).remove(0);
    let prefill = waiting_prompt(&scheduler, 100);
    assert!(scheduler.decode_pressure_prefill_cap_threshold(&hint(8, 256)) > 1);

    let batch = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&batch, &decode), Some(1));
    assert_eq!(scheduled_tokens(&batch, &prefill), Some(19));
    assert!(!scheduler.mark_prefill_chunk_processed(&prefill, 100, 19));
    scheduler.update_decode_progress(&decode, 1);

    let token_limited = scheduler.create_iteration_batch(hint(8, 12)).unwrap();
    assert_eq!(scheduled_tokens(&token_limited, &decode), Some(1));
    assert_eq!(scheduled_tokens(&token_limited, &prefill), Some(11));
    assert!(!scheduler.mark_prefill_chunk_processed(&prefill, 100, 11));
    scheduler.update_decode_progress(&decode, 2);

    let sequence_limited = scheduler.create_iteration_batch(hint(1, 256)).unwrap();
    assert_eq!(sequence_limited.requests.len(), 1);
    assert_eq!(scheduled_tokens(&sequence_limited, &decode), Some(1));
    assert_eq!(scheduled_tokens(&sequence_limited, &prefill), None);
}

#[test]
fn active_decode_prefill_token_budget_keeps_pure_prefill_elastic() {
    let scheduler = budgeted_scheduler(19);
    let prefills = (0..3)
        .map(|_| waiting_prompt(&scheduler, 50))
        .collect::<Vec<_>>();
    let batch = scheduler.create_iteration_batch(hint(8, 180)).unwrap();
    assert_eq!(batch.requests.len(), 3);
    for prefill in prefills {
        assert_eq!(scheduled_tokens(&batch, &prefill), Some(50));
    }
    assert_eq!(batch.resource_requirements.gpu_memory, 150 * 16);
}

#[test]
fn active_decode_prefill_token_budget_requires_a_runnable_decoder() {
    let scheduler = budgeted_scheduler(19);
    let decode = activate_decode_requests(&scheduler, 1).remove(0);
    let readiness = scheduler
        .defer_for_execution_readiness(std::slice::from_ref(&decode))
        .unwrap();
    let prefill = waiting_prompt(&scheduler, 100);

    let unblocked_prefill = scheduler.create_iteration_batch(hint(8, 80)).unwrap();
    assert_eq!(unblocked_prefill.requests.len(), 1);
    assert_eq!(scheduled_tokens(&unblocked_prefill, &prefill), Some(80));
    assert!(!scheduler.mark_prefill_chunk_processed(&prefill, 100, 80));

    assert!(readiness.wake().mark_ready());
    let with_decode = scheduler.create_iteration_batch(hint(8, 80)).unwrap();
    assert_eq!(scheduled_tokens(&with_decode, &decode), Some(1));
    assert_eq!(scheduled_tokens(&with_decode, &prefill), Some(19));
}

#[test]
fn active_decode_prefill_token_budget_protects_decode_before_fill_first_disarms() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 8,
        prefill_first_until_active: Some(4),
        active_decode_prefill_token_budget: Some(19),
        ..SchedulerConfig::default()
    });
    let cold = waiting_prompt(&scheduler, 100);
    let pure_prefill = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&pure_prefill, &cold), Some(100));
    scheduler.mark_prefill_complete(&cold, 100);
    assert!(scheduler
        .fill_first_initial_cohort_armed
        .load(Ordering::Acquire));
    let late = waiting_prompt(&scheduler, 100);
    assert_eq!(scheduler.active_count(), 1);
    assert_eq!(
        scheduler.fill_first_dynamic_admission_limit(&hint(8, 256), 4),
        0
    );

    let first_decode = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&first_decode, &cold), Some(1));
    assert_eq!(scheduled_tokens(&first_decode, &late), Some(19));
    assert!(!scheduler
        .fill_first_initial_cohort_armed
        .load(Ordering::Acquire));
}

#[test]
fn active_decode_prefill_token_budget_preserves_capacity_recompute_and_progress_feedback() {
    let scheduler = budgeted_scheduler(13);
    let recompute = waiting_prompt(&scheduler, 32);
    let incumbent = waiting_prompt(&scheduler, 32);
    let cold = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(cold.requests.len(), 2);
    scheduler.mark_prefill_complete(&recompute, 32);
    scheduler.mark_prefill_complete(&incumbent, 32);
    assert!(scheduler.defer_decode_to_waiting_for_capacity(&recompute, 2));

    let attempted = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&attempted, &incumbent), Some(1));
    assert_eq!(scheduled_tokens(&attempted, &recompute), Some(13));
    scheduler.defer_capacity_deferred_mixed_recompute_until_release();
    assert!(scheduler.defer_prefill_to_waiting(&recompute));
    let parked = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(parked.requests.len(), 1);
    assert_eq!(scheduled_tokens(&parked, &incumbent), Some(1));
    assert_eq!(scheduled_tokens(&parked, &recompute), None);

    scheduler.record_capacity_deferred_mixed_recompute_release_evidence();
    let resumed = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&resumed, &recompute), Some(13));
    assert!(!scheduler
        .mark_prefill_chunk_processed_with_capacity_feedback(&recompute, 32, 13, 5)
        .unwrap());
    let narrowed = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&narrowed, &recompute), Some(5));
    let frontier = narrowed
        .requests
        .iter()
        .find(|request| request.request.id == recompute)
        .unwrap();
    assert_eq!(frontier.tokens_processed, 5);
    assert!(!scheduler
        .mark_prefill_chunk_processed_with_capacity_feedback(&recompute, 32, 5, 5)
        .unwrap());
    let recovered = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&recovered, &recompute), Some(10));
    assert_eq!(scheduled_tokens(&recovered, &incumbent), Some(1));
}

#[test]
fn active_decode_prefill_token_budget_keeps_maintenance_yield_scoped_to_affected_requests() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 8,
        active_decode_prefill_chunk: Some(9),
        active_decode_prefill_token_budget: Some(17),
        ..SchedulerConfig::default()
    });
    let decode = activate_decode_requests(&scheduler, 1).remove(0);
    let maintained = waiting_prompt(&scheduler, 100);
    let peer = waiting_prompt(&scheduler, 100);
    let initial = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&initial, &maintained), Some(9));
    assert_eq!(scheduled_tokens(&initial, &peer), Some(8));
    scheduler
        .defer_retry_after_execution_maintenance_epoch(std::slice::from_ref(&maintained), 7)
        .unwrap();

    let yielded = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&yielded, &decode), Some(1));
    assert_eq!(scheduled_tokens(&yielded, &maintained), None);
    assert_eq!(scheduled_tokens(&yielded, &peer), Some(9));
    assert!(!scheduler.mark_prefill_chunk_processed(&peer, 100, 9));

    let resumed = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
    assert_eq!(scheduled_tokens(&resumed, &decode), Some(1));
    assert_eq!(scheduled_tokens(&resumed, &maintained), Some(9));
    assert_eq!(scheduled_tokens(&resumed, &peer), Some(8));
}

#[test]
fn active_decode_prefill_token_budget_unset_or_zero_keeps_existing_low_pressure_policy() {
    for budget in [None, Some(0)] {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            active_decode_prefill_token_budget: budget,
            ..SchedulerConfig::default()
        });
        let decode = activate_decode_requests(&scheduler, 1).remove(0);
        let prefill = waiting_prompt(&scheduler, 100);
        let batch = scheduler.create_iteration_batch(hint(8, 256)).unwrap();
        assert_eq!(scheduled_tokens(&batch, &decode), Some(1));
        assert_eq!(scheduled_tokens(&batch, &prefill), Some(100));
    }
}
