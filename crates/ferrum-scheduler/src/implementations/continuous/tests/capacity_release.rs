use super::*;
use ferrum_runtime::DeferredAction;

pub(super) fn response(request_id: &RequestId) -> InferenceResponse {
    InferenceResponse {
        request_id: request_id.clone(),
        text: String::new(),
        tokens: Vec::new(),
        finish_reason: ferrum_types::FinishReason::Length,
        usage: ferrum_types::TokenUsage::new(1, 0),
        latency_ms: 0,
        created_at: chrono::Utc::now(),
        metadata: Default::default(),
        api_response: None,
        execution_evidence: None,
    }
}

#[tokio::test]
async fn allocator_capacity_wait_has_no_native_epoch_and_requires_confirmed_release() {
    for cancelled in [false, true] {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            ..SchedulerConfig::default()
        });
        let active = create_test_request_with_prompt_tokens(Priority::Normal, 1);
        let active_id = active.id.clone();
        let waiting = create_test_request_with_prompt_tokens(Priority::Normal, 1);
        let waiting_id = waiting.id.clone();
        scheduler.submit(active).await.unwrap();
        scheduler.submit(waiting).await.unwrap();
        let prepared = scheduler
            .prepare_token_admission_observed(
                4,
                None,
                &mut |request| {
                    if request.id == active_id {
                        AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                            request_id: request.id.clone(),
                        })
                    } else {
                        AdmissionProbeOutcome::CapacityDeferred
                    }
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(prepared.admitted(), 1);
        assert_eq!(prepared.deferred(), 1);
        let batch = scheduler
            .next_batch_with_token_admissions(
                scheduler.prepare_token_batch_selection(BatchHint::simple(4)),
                None,
                &mut |_| {},
                None,
            )
            .map(|selection| selection.batch)
            .unwrap()
            .unwrap();
        assert_eq!(batch.requests.len(), 1);
        assert_eq!(batch.requests[0].request.id, active_id);
        if cancelled {
            assert!(scheduler.cancel_request(active_id.clone()).unwrap());
        } else {
            scheduler
                .complete_request(active_id.clone(), &response(&active_id))
                .unwrap();
        }
        // Logical termination, including an empty active queue, is not an
        // allocator release and cannot grant a fresh admission attempt.
        assert!(scheduler.waiting_for_confirmed_capacity_release());
        for _ in 0..2 {
            let tick = scheduler
                .prepare_token_admission_observed(
                    4,
                    None,
                    &mut |_| panic!("a held execution owner has not released capacity"),
                    &mut |_| {},
                )
                .unwrap();
            assert_eq!(tick.probed(), 0);
            assert_eq!(tick.skipped_unchanged(), 1);
        }
        assert!(scheduler
            .passive_capacity_wait_condition()
            .unwrap()
            .is_none());
        scheduler.record_external_capacity_release();
        assert!(!scheduler.waiting_for_confirmed_capacity_release());
        let admitted = scheduler
            .prepare_token_admission_observed(
                4,
                None,
                &mut |request| {
                    assert_eq!(request.id, waiting_id);
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    })
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(admitted.probed(), 1);
        assert_eq!(admitted.admitted(), 1);
    }
}

#[tokio::test]
async fn native_deferral_without_native_snapshot_is_a_contract_failure() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let request = create_test_request_with_prompt_tokens(Priority::Normal, 1);
    let id = request.id.clone();
    scheduler.submit(request).await.unwrap();
    let epochs = AdmissionWakeEpochs::new(NonZeroU64::new(19).unwrap(), 0, 0, 0);
    let availability =
        CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 1).unwrap();
    let condition = CapacityWaitCondition::from_observation(19, vec![availability]).unwrap();
    let tick = scheduler
        .prepare_token_admission_observed(
            1,
            None,
            &mut |_| {
                AdmissionProbeOutcome::Deferred(AdmissionDeferral::new(
                    DeferredAction::WaitForRelease,
                    epochs,
                    condition.clone(),
                ))
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(tick.faulted(), 1);
    let failures = scheduler.take_admission_failures();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].0, id);
    assert!(failures[0]
        .1
        .to_string()
        .contains("actual capacity snapshot"));
}

#[tokio::test]
async fn logical_terminal_waits_for_external_release_and_runtime_availability() {
    // Exercise both active queues and both logical terminal operations. The
    // execution owner remains held until the explicit notification below.
    for decoding in [false, true] {
        for cancel in [false, true] {
            let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
                max_running_requests: 4,
                ..SchedulerConfig::default()
            });
            let waiting = create_test_request_with_prompt_tokens(Priority::Normal, 1);
            let waiting_id = waiting.id.clone();
            let active = create_test_request_with_prompt_tokens(Priority::Normal, 1);
            let active_id = active.id.clone();
            scheduler.submit(waiting).await.unwrap();
            scheduler.submit(active).await.unwrap();

            let coordinator = NonZeroU64::new(19).unwrap();
            let wake = AdmissionWakeEpochs::new(coordinator, 0, 0, 0);
            let availability = [CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::ActiveSequenceSlots,
                1,
            )
            .unwrap()];
            let condition =
                CapacityWaitCondition::from_observation(19, availability.to_vec()).unwrap();
            let admitted = scheduler
                .next_batch_with_dynamic_admission(
                    BatchHint::simple(4),
                    AdmissionWakeSnapshot::new(wake, &availability),
                    &mut |request| {
                        if request.id == waiting_id {
                            AdmissionProbeOutcome::Deferred(AdmissionDeferral::new(
                                DeferredAction::WaitForRelease,
                                wake,
                                condition.clone(),
                            ))
                        } else {
                            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                                request_id: request.id.clone(),
                            })
                        }
                    },
                )
                .unwrap()
                .unwrap();
            assert_eq!(admitted.requests.len(), 1);
            assert_eq!(admitted.requests[0].request.id, active_id);
            if decoding {
                scheduler.mark_prefill_complete(&active_id, 1);
            }
            scheduler.record_capacity_defer_feedback(4);
            scheduler.record_decode_execution_capacity_pressure(4);
            let before = scheduler.trace_snapshot();
            assert_eq!(before.capacity_backpressure_admit_limit, Some(2));
            assert_eq!(before.decode_capacity_backpressure_admit_limit, Some(2));

            if cancel {
                assert!(scheduler.cancel(active_id.clone()).await.unwrap());
            } else {
                scheduler
                    .complete(active_id.clone(), &response(&active_id))
                    .await
                    .unwrap();
            }
            assert_eq!(scheduler.trace_phase(&active_id), None);
            let logical_terminal = scheduler.trace_snapshot();
            assert_eq!(logical_terminal.active_len, 0);
            assert_eq!(logical_terminal.waiting_queue_len, 1);
            assert_eq!(
                logical_terminal.capacity_release_epoch,
                before.capacity_release_epoch
            );
            assert_eq!(
                logical_terminal.capacity_mixed_recompute_epoch,
                before.capacity_mixed_recompute_epoch
            );
            assert_eq!(logical_terminal.capacity_backpressure_admit_limit, Some(2));
            assert_eq!(
                logical_terminal.decode_capacity_backpressure_admit_limit,
                Some(2)
            );
            assert!(!scheduler.record_decode_execution_capacity_success(2));

            // Empty active queues and repeated scheduling must not turn a
            // retained execution owner into capacity availability.
            for _ in 0..2 {
                let mut probes = 0;
                let batch = scheduler
                    .next_batch_with_dynamic_admission(
                        BatchHint::simple(4),
                        AdmissionWakeSnapshot::new(wake, &availability),
                        &mut |request| {
                            probes += 1;
                            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                                request_id: request.id.clone(),
                            })
                        },
                    )
                    .unwrap();
                assert!(batch.is_none());
                assert_eq!(probes, 0);
                assert_eq!(
                    scheduler.trace_snapshot().capacity_release_epoch,
                    before.capacity_release_epoch
                );
            }

            scheduler.record_external_capacity_release();
            let released = scheduler.trace_snapshot();
            assert_eq!(
                released.capacity_release_epoch,
                before.capacity_release_epoch + 1
            );
            assert_eq!(
                released.capacity_mixed_recompute_epoch,
                before.capacity_mixed_recompute_epoch + 1
            );
            assert_eq!(released.capacity_backpressure_admit_limit, None);
            assert_eq!(released.decode_capacity_backpressure_admit_limit, Some(2));
            assert!(scheduler.record_decode_execution_capacity_success(2));
            assert_eq!(
                scheduler
                    .trace_snapshot()
                    .decode_capacity_backpressure_admit_limit,
                Some(3)
            );

            // The scheduler notification cannot replace the execution owner's
            // authoritative availability snapshot for typed admission.
            assert!(scheduler
                .next_batch_with_dynamic_admission(
                    BatchHint::simple(4),
                    AdmissionWakeSnapshot::new(wake, &availability),
                    &mut |_| panic!("unchanged execution capacity must not be reprobed"),
                )
                .unwrap()
                .is_none());
            let released_wake = AdmissionWakeEpochs::new(coordinator, 1, 0, 0);
            let released_availability = [CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::ActiveSequenceSlots,
                2,
            )
            .unwrap()];
            let mut probes = 0;
            let batch = scheduler
                .next_batch_with_dynamic_admission(
                    BatchHint::simple(4),
                    AdmissionWakeSnapshot::new(released_wake, &released_availability),
                    &mut |request| {
                        probes += 1;
                        AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                            request_id: request.id.clone(),
                        })
                    },
                )
                .unwrap()
                .unwrap();
            assert_eq!(probes, 1);
            assert_eq!(batch.requests.len(), 1);
            assert_eq!(batch.requests[0].request.id, waiting_id);
        }
    }
}

#[tokio::test]
async fn logical_preemption_retains_allocation_without_reporting_physical_release() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 4,
        ..SchedulerConfig::default()
    });
    let request_id = activate_decode_requests(&scheduler, 1).remove(0);
    {
        let mut decode = scheduler.decode_queue.write();
        let allocation = &mut decode
            .requests
            .get_mut(&request_id)
            .unwrap()
            .inner
            .allocated_resources;
        allocation.gpu_memory = 4096;
        allocation.cpu_memory = 2048;
        allocation.recurrent_state_bytes = 1024;
        allocation.recurrent_state_slots = 1;
        allocation.compute_units = 1;
    }
    scheduler.record_capacity_defer_feedback(4);
    scheduler.record_decode_execution_capacity_pressure(4);
    let before = scheduler.trace_snapshot();
    let preempted = scheduler.preempt(request_id.clone()).await.unwrap();
    assert!(preempted.success);
    assert!(preempted.saved_state.is_some());
    assert!(preempted.freed_resources.kv_cache_blocks.is_empty());
    assert_eq!(preempted.freed_resources.gpu_memory, 0);
    assert_eq!(preempted.freed_resources.cpu_memory, 0);
    assert_eq!(preempted.freed_resources.recurrent_state_bytes, 0);
    assert_eq!(preempted.freed_resources.recurrent_state_slots, 0);
    assert_eq!(preempted.freed_resources.compute_units, 0);
    assert_eq!(
        scheduler.trace_phase(&request_id),
        Some(RequestPhase::Preempted)
    );
    let after = scheduler.trace_snapshot();
    assert_eq!(after.capacity_release_epoch, before.capacity_release_epoch);
    assert_eq!(
        after.capacity_mixed_recompute_epoch,
        before.capacity_mixed_recompute_epoch
    );
    assert_eq!(after.capacity_backpressure_admit_limit, Some(2));
    assert_eq!(after.decode_capacity_backpressure_admit_limit, Some(2));
    assert!(!scheduler.record_decode_execution_capacity_success(2));

    scheduler.resume(request_id.clone()).await.unwrap();
    assert_eq!(
        scheduler.trace_phase(&request_id),
        Some(RequestPhase::Decoding)
    );
    let decode = scheduler.decode_queue.read();
    let allocation = &decode.requests[&request_id].inner.allocated_resources;
    assert_eq!(allocation.gpu_memory, 4096);
    assert_eq!(allocation.cpu_memory, 2048);
    assert_eq!(allocation.recurrent_state_bytes, 1024);
    assert_eq!(allocation.recurrent_state_slots, 1);
    assert_eq!(allocation.compute_units, 1);
}

#[test]
fn confirmed_release_is_notified_after_recompute_requeue() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 4,
        ..SchedulerConfig::default()
    });
    let ids = activate_decode_requests(&scheduler, 2);
    let before = scheduler.trace_snapshot().capacity_release_epoch;
    assert!(scheduler.defer_decode_to_waiting_for_capacity(&ids[0], 2));
    assert_eq!(scheduler.trace_snapshot().capacity_release_epoch, before);
    assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 1);
    {
        let waiting = scheduler.waiting_queue.read();
        let request = waiting
            .iter()
            .find(|request| request.inner.request.id == ids[0])
            .unwrap();
        assert_eq!(request.capacity_deferred_until_release_epoch, before + 1);
    }
    scheduler.record_external_capacity_release();
    assert_eq!(
        scheduler.trace_snapshot().capacity_release_epoch,
        before + 1
    );
    assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 0);
    assert_eq!(scheduler.trace_phase(&ids[0]), Some(RequestPhase::Waiting));
    assert_eq!(scheduler.trace_phase(&ids[1]), Some(RequestPhase::Decoding));
}
