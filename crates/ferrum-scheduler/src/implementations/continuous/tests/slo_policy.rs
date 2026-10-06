use super::*;
use ferrum_types::SchedulerSloConfig;

fn scheduler(static_budget: Option<usize>) -> ContinuousBatchScheduler {
    ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 8,
        active_decode_prefill_token_budget: static_budget,
        slo: Some(SchedulerSloConfig {
            ttft_ms: 1000.0,
            tpot_ms: 100.0,
            itl_ms: 120.0,
        }),
        ..SchedulerConfig::default()
    })
}

fn hint() -> BatchHint {
    BatchHint {
        max_batch_size: 8,
        max_tokens: 256,
        target_latency_ms: None,
        available_memory: None,
        resource_constraints: Default::default(),
    }
}

fn waiting(scheduler: &ContinuousBatchScheduler, tokens: usize) -> RequestId {
    let request = create_test_request_with_prompt_tokens(Priority::Normal, tokens);
    let id = request.id.clone();
    enqueue_waiting(scheduler, request);
    id
}

fn train(scheduler: &ContinuousBatchScheduler) {
    scheduler.record_execution_step(100, 0, Duration::from_millis(100));
    scheduler.record_execution_step(0, 1, Duration::from_millis(20));
}

#[test]
fn slo_cold_start_falls_back_then_shares_dynamic_budget_and_records_actual_progress() {
    let scheduler = scheduler(None);
    activate_decode_requests(&scheduler, 1);
    let first = waiting(&scheduler, 100);
    waiting(&scheduler, 100);
    let cold = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(cold.resource_requirements.gpu_memory, 201 * 16);
    assert_eq!(
        scheduler.slo_snapshot().unwrap().prefill_budget_tokens,
        None
    );

    train(&scheduler);
    let adapted = scheduler.create_iteration_batch(hint()).unwrap();
    // 90 ms server allowance minus 20 ms decode = 70 aggregate prefill tokens.
    assert_eq!(adapted.resource_requirements.gpu_memory, 71 * 16);
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 0);
    scheduler.mark_prefill_chunk_processed(&first, 100, 70);
    scheduler.record_execution_step(70, 1, Duration::from_millis(160));
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 1);
    let slower = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(slower.resource_requirements.gpu_memory, 36 * 16);
    assert_eq!(
        scheduler.slo_snapshot().unwrap().prefill_budget_tokens,
        Some(35)
    );
}

#[test]
fn slo_respects_static_and_live_caps_and_keeps_pure_prefill_elastic() {
    let scheduler = scheduler(Some(13));
    train(&scheduler);
    let first = waiting(&scheduler, 100);
    let pure = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(pure.requests[0].tokens_to_process, Some(100));
    assert_eq!(
        scheduler.slo_snapshot().unwrap().prefill_budget_tokens,
        None
    );
    scheduler.mark_prefill_complete(&first, 100);
    waiting(&scheduler, 100);
    waiting(&scheduler, 100);
    let mixed = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(mixed.resource_requirements.gpu_memory, 14 * 16);
    let narrow = scheduler
        .create_iteration_batch(BatchHint {
            max_tokens: 7,
            ..hint()
        })
        .unwrap();
    assert_eq!(narrow.resource_requirements.gpu_memory, 7 * 16);
    assert_eq!(
        scheduler.slo_snapshot().unwrap().prefill_budget_tokens,
        Some(6)
    );
}

#[test]
fn slo_decode_overload_keeps_prefill_progress_and_no_progress_does_not_train() {
    let scheduler = scheduler(None);
    activate_decode_requests(&scheduler, 1);
    waiting(&scheduler, 100);
    train(&scheduler);
    scheduler.record_execution_step(0, 1, Duration::from_millis(150));
    let batch = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(batch.resource_requirements.gpu_memory, 2 * 16);
    let before = scheduler.slo_snapshot().unwrap();
    assert_eq!(before.decode_overload_steps, 1);
    assert_eq!(before.prefill_budget_tokens, Some(1));
    scheduler.record_execution_step(0, 0, Duration::from_millis(100));
    let after = scheduler.slo_snapshot().unwrap();
    assert_eq!(after.observed_steps, before.observed_steps);
    assert_eq!(after.adapted_prefill_steps, 0);
    // A stale failed/empty iteration cannot mark later work as adapted.
    scheduler.record_execution_step(1, 1, Duration::from_millis(151));
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 0);
}

#[test]
fn slo_ttft_urgency_prioritizes_the_request_with_least_remaining_slack() {
    let scheduler = scheduler(None);
    activate_decode_requests(&scheduler, 1);
    train(&scheduler);
    let short = waiting(&scheduler, 20);
    let urgent = waiting(&scheduler, 900);
    scheduler.promote_to_prefill_with_empty_retry(&short, None);
    scheduler.promote_to_prefill_with_empty_retry(&urgent, None);
    let now = chrono::Utc::now();
    for request in scheduler.prefill_queue.write().iter_mut() {
        // The younger long request is already close to its predicted deadline;
        // the older short request still has ample slack.
        let age_ms = if request.inner.request.id == short {
            200
        } else {
            100
        };
        request.inner.submitted_at = now - chrono::Duration::milliseconds(age_ms);
    }
    let batch = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(batch.requests[1].request.id, urgent);
    assert_eq!(batch.requests[1].tokens_to_process, Some(70));
    assert!(!batch
        .requests
        .iter()
        .any(|request| request.request.id == short));
}

#[tokio::test]
async fn slo_invalid_programmatic_targets_reject_requests() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        slo: Some(SchedulerSloConfig {
            ttft_ms: f64::NAN,
            tpot_ms: 100.0,
            itl_ms: 120.0,
        }),
        ..SchedulerConfig::default()
    });
    assert!(scheduler
        .submit(create_test_request(Priority::Normal))
        .await
        .is_err());
    assert_eq!(scheduler.waiting_count(), 0);
}
