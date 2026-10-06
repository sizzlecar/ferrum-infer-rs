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
    // t = 10.25 ms fixed + 9.5 ms/decode row + 1 ms/prefill token.
    scheduler.record_execution_step(100, 0, Duration::from_micros(110_250));
    scheduler.record_execution_step(50, 0, Duration::from_micros(60_250));
    // Pure-prefill observations alone cannot identify the decode coefficient.
    assert_eq!(scheduler.slo_snapshot().unwrap().prefill_ms_per_token, None);
    scheduler.record_execution_step(0, 1, Duration::from_micros(19_750));
}

#[test]
fn slo_cold_start_and_tpot_credit_preserve_budget_under_mixed_overhead() {
    let scheduler = scheduler(None);
    activate_decode_requests(&scheduler, 1);
    // Keep controller arithmetic independent of wall-clock elapsed time.
    for request in scheduler.decode_queue.write().requests.values_mut() {
        request.slo_output_progress = None;
    }
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
    // 90 ms allowance minus 19.75 ms decode permits 70 aggregate prefill tokens.
    assert_eq!(adapted.resource_requirements.gpu_memory, 71 * 16);
    assert!(!scheduler.slo_snapshot().unwrap().decode_target_infeasible);
    assert_eq!(
        scheduler.slo_snapshot().unwrap().budget_target_ms,
        Some(100.0)
    );
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 0);
    scheduler.mark_prefill_chunk_processed(&first, 100, 70);
    scheduler.record_execution_step(70, 1, Duration::from_micros(89_750));
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 1);
    let stable = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(stable.resource_requirements.gpu_memory, 71 * 16);
    assert_eq!(
        scheduler.slo_snapshot().unwrap().prefill_budget_tokens,
        Some(70)
    );

    let targets = scheduler.config.slo.unwrap();
    let mut controller = scheduler.slo_controller.as_ref().unwrap().lock();
    // Accumulated TPOT allowance permits a wave longer than one 100 ms TPOT,
    // while 120 ms ITL remains a per-wave ceiling: floor(108 - 19.75) = 88.
    assert_eq!(controller.budget(targets, 1, 256, Some(200.0)), Some(88));
    assert_eq!(controller.snapshot().budget_target_ms, Some(120.0));
    for tokens in [8, 2, 1] {
        controller.record(
            targets,
            tokens,
            1,
            Duration::from_micros(19_750 + tokens as u64 * 1000),
        );
        // Small mixed waves retain fixed overhead in a, not in c/token.
        assert!((controller.prefill_ms_per_token().unwrap() - 1.0).abs() < 1e-8);
        assert_eq!(controller.budget(targets, 1, 256, Some(200.0)), Some(88));
    }
    let model = controller.snapshot();
    assert!((model.fixed_step_ms.unwrap() - 10.25).abs() < 1e-8);
    assert!((model.decode_ms_per_sequence.unwrap() - 9.5).abs() < 1e-8);
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
    scheduler.record_execution_step(100, 0, Duration::from_micros(110_250));
    scheduler.record_execution_step(50, 0, Duration::from_micros(60_250));
    scheduler.record_execution_step(0, 1, Duration::from_millis(101));
    // Pure decode violates strict TPOT even though the wider ITL still fits.
    // Keep static prefill progress instead of switching to the wider target.
    let batch = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(batch.resource_requirements.gpu_memory, 101 * 16);
    let before = scheduler.slo_snapshot().unwrap();
    assert_eq!(before.decode_overload_steps, 1);
    assert!(before.decode_target_infeasible);
    assert_eq!(before.budget_target_ms, None);
    assert_eq!(before.prefill_budget_tokens, None);
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
    for request in scheduler.decode_queue.write().requests.values_mut() {
        request.slo_output_progress = None;
    }
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
