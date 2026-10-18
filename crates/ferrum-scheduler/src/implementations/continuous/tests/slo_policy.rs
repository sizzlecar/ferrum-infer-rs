use super::*;
use ferrum_types::{RequestIngressAnchor, SchedulerSloConfig};

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
fn slo_cold_start_and_itl_budget_preserve_fixed_mixed_overhead() {
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
    // 120 ms ITL minus 19.75 ms decode permits 100 aggregate prefill tokens.
    assert_eq!(adapted.resource_requirements.gpu_memory, 101 * 16);
    assert!(!scheduler.slo_snapshot().unwrap().decode_target_infeasible);
    assert_eq!(
        scheduler.slo_snapshot().unwrap().budget_target_ms,
        Some(120.0)
    );
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 0);
    scheduler.mark_prefill_chunk_processed(&first, 100, 100);
    scheduler.record_execution_step(100, 1, Duration::from_micros(119_750));
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 1);
    let stable = scheduler.create_iteration_batch(hint()).unwrap();
    // One additional decoder leaves floor(120 - 29.25) = 90 prefill tokens.
    assert_eq!(stable.resource_requirements.gpu_memory, 92 * 16);
    assert_eq!(
        scheduler.slo_snapshot().unwrap().prefill_budget_tokens,
        Some(90)
    );

    let targets = scheduler.config.slo.unwrap();
    let mut controller = scheduler.slo_controller.as_ref().unwrap().lock();
    // TPOT does not constrain individual waves or reject feasible ITL work.
    let tight_tpot = SchedulerSloConfig {
        tpot_ms: 1.0,
        ..targets
    };
    assert_eq!(controller.budget(tight_tpot, 1, 256, []), Some(100));
    assert_eq!(controller.snapshot().budget_target_ms, Some(120.0));
    let decode_ms = controller.snapshot().decode_step_ms.unwrap();
    let exact_itl = decode_ms + controller.prefill_ms_per_token().unwrap() * 100.0;
    assert_eq!(
        controller.budget(
            SchedulerSloConfig {
                itl_ms: exact_itl,
                ..tight_tpot
            },
            1,
            100,
            []
        ),
        None
    );
    assert_eq!(controller.snapshot().budget_target_ms, None);
    for tokens in [8, 2, 1] {
        controller.record(
            targets,
            tokens,
            1,
            Duration::from_micros(19_750 + tokens as u64 * 1000),
        );
        // Small mixed waves retain fixed overhead in a, not in c/token.
        assert!((controller.prefill_ms_per_token().unwrap() - 1.0).abs() < 1e-8);
        assert_eq!(controller.budget(targets, 1, 256, []), Some(100));
    }
    let model = controller.snapshot();
    assert!((model.fixed_step_ms.unwrap() - 10.25).abs() < 1e-8);
    assert!((model.decode_ms_per_sequence.unwrap() - 9.5).abs() < 1e-8);
    controller.record(tight_tpot, 0, 1, Duration::from_micros(19_750));
    assert_eq!(controller.snapshot().decode_overload_steps, 0);
}

#[test]
fn slo_unused_dynamic_budget_does_not_count_adaptation() {
    let scheduler = scheduler(None);
    train(&scheduler);
    activate_decode_requests(&scheduler, 1);
    waiting(&scheduler, 95);
    // 114.75 ms fits ITL, but would have been shrunk by the old 90% margin.
    let batch = scheduler.create_iteration_batch(hint()).unwrap();
    assert_eq!(batch.resource_requirements.gpu_memory, 96 * 16);
    scheduler.record_execution_step(95, 1, Duration::from_micros(114_750));
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 0);
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
    assert_eq!(
        scheduler.slo_snapshot().unwrap().prefill_budget_tokens,
        None
    );
    scheduler.record_execution_step(13, 1, Duration::from_micros(32_750));
    assert_eq!(scheduler.slo_snapshot().unwrap().adapted_prefill_steps, 0);
    let narrow = scheduler
        .create_iteration_batch(BatchHint {
            max_tokens: 7,
            ..hint()
        })
        .unwrap();
    assert_eq!(narrow.resource_requirements.gpu_memory, 7 * 16);
    assert_eq!(
        scheduler.slo_snapshot().unwrap().prefill_budget_tokens,
        None
    );
}

#[test]
fn slo_decode_overload_keeps_prefill_progress_and_no_progress_does_not_train() {
    let scheduler = scheduler(None);
    activate_decode_requests(&scheduler, 1);
    waiting(&scheduler, 100);
    scheduler.record_execution_step(100, 0, Duration::from_micros(110_250));
    scheduler.record_execution_step(50, 0, Duration::from_micros(60_250));
    scheduler.record_execution_step(0, 1, Duration::from_millis(121));
    // Pure decode already violates ITL, so retain static prefill progress.
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
fn slo_ttft_prioritizes_urgent_prefill_and_rejects_an_infeasible_dynamic_budget() {
    for wall_direction in [-1, 1] {
        let scheduler = scheduler(None);
        activate_decode_requests(&scheduler, 1);
        train(&scheduler);
        let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
        let short_request =
            create_test_request_with_prompt_tokens(Priority::Normal, 20).with_ingress(ingress);
        let urgent_request =
            create_test_request_with_prompt_tokens(Priority::Normal, 120).with_ingress(ingress);
        let short = short_request.id.clone();
        let urgent = urgent_request.id.clone();
        enqueue_waiting(&scheduler, short_request);
        enqueue_waiting(&scheduler, urgent_request);
        scheduler.promote_to_prefill_with_empty_retry(&short, None);
        scheduler.promote_to_prefill_with_empty_retry(&urgent, None);
        let now = ingress.instant() + Duration::from_millis(800);
        for request in scheduler.prefill_queue.write().iter_mut() {
            // Contradict the monotonic order with wall diagnostics in both
            // directions. Both requests have 200 ms left; only the long prompt
            // is urgent, and 120 + 19.75*ceil(120/100) fits the 180 ms allowance.
            let direction = if request.inner.request.id == short {
                wall_direction
            } else {
                -wall_direction
            };
            let wall = chrono::Utc::now() + chrono::Duration::days(direction * 3650);
            request.inner.submitted_at = wall;
            request.inner.started_at = Some(wall);
            request.inner.request.created_at = wall;
        }
        let batch = scheduler
            .create_iteration_batch_with_admission_at(hint(), WaitingAdmissionMode::Legacy, now)
            .unwrap()
            .unwrap();
        assert_eq!(batch.requests[1].request.id, urgent);
        assert_eq!(batch.requests[1].tokens_to_process, Some(100));
        assert!(!batch
            .requests
            .iter()
            .any(|request| request.request.id == short));
        assert!(!scheduler.slo_snapshot().unwrap().ttft_budget_infeasible);

        // With only 150 ms left, the same optimistic completion time cannot fit.
        // Retain the original static budget and report the actual fallback.
        let fallback = scheduler
            .create_iteration_batch_with_admission_at(
                hint(),
                WaitingAdmissionMode::Legacy,
                now + Duration::from_millis(50),
            )
            .unwrap()
            .unwrap();
        assert_eq!(fallback.requests[1].request.id, urgent);
        assert_eq!(fallback.requests[1].tokens_to_process, Some(120));
        let snapshot = scheduler.slo_snapshot().unwrap();
        assert!(snapshot.ttft_budget_infeasible);
        assert_eq!(snapshot.ttft_fallback_steps, 1);
        assert_eq!(snapshot.prefill_budget_tokens, None);
        assert_eq!(snapshot.budget_target_ms, None);
    }
}

#[test]
fn request_clock_recompute_after_output_has_no_ttft_priority() {
    let scheduler = scheduler(None);
    train(&scheduler);
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let recompute =
        create_test_request_with_prompt_tokens(Priority::Normal, 1000).with_ingress(ingress);
    let recompute_id = recompute.id.clone();
    scheduler.submit_request(recompute).unwrap();
    assert!(scheduler.promote_to_prefill_with_empty_retry(&recompute_id, None));
    scheduler.mark_prefill_complete(&recompute_id, 1000);
    scheduler.update_decode_progress(&recompute_id, 1);
    assert!(scheduler.defer_decode_to_waiting_for_capacity(&recompute_id, 1));
    assert!(scheduler.promote_to_prefill_with_empty_retry(&recompute_id, None));

    let fresh = create_test_request_with_prompt_tokens(Priority::Normal, 120).with_ingress(ingress);
    let fresh_id = fresh.id.clone();
    scheduler.submit_request(fresh).unwrap();
    assert!(scheduler.promote_to_prefill_with_empty_retry(&fresh_id, None));
    let mut queue = scheduler.prefill_queue.write();
    assert_eq!(queue.front().unwrap().inner.request.id, recompute_id);
    assert!(queue.front().unwrap().slo_first_token_emitted);
    scheduler.prioritize_ttft(&mut queue, ingress.instant() + Duration::from_millis(800));
    assert_eq!(queue.front().unwrap().inner.request.id, fresh_id);
    assert_eq!(queue.back().unwrap().inner.request.id, recompute_id);
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
