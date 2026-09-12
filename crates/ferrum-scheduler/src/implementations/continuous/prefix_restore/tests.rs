use super::*;
use crate::vnext::{AdmissionProbeOutcome, AdmissionWakeEpochs, AdmissionWakeSnapshot};
use crate::{BatchHint, Scheduler};
use ferrum_interfaces::model_executor::ExecutorPrefillAdmissionReceipt;
use ferrum_types::{InferenceRequest, SchedulerConfig, PROMPT_TOKENS_METADATA_KEY};
use std::num::NonZeroU64;
use std::sync::atomic::Ordering;

fn request() -> InferenceRequest {
    InferenceRequest::new("restore a real admitted prefill", "test-model")
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, serde_json::json!(128))
}

fn admit(scheduler: &ContinuousBatchScheduler, maximum: usize) {
    scheduler
        .prepare_dynamic_admission_observed(
            maximum,
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(NonZeroU64::new(1).unwrap(), 1, 1, 1),
                &[],
            ),
            &mut |request| {
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                })
            },
            &mut |_| {},
        )
        .unwrap();
}

fn snapshot(scheduler: &ContinuousBatchScheduler, id: &RequestId) -> ContinuousBatchRequest {
    scheduler
        .prefill_queue
        .read()
        .iter()
        .find(|request| request.inner.request.id == *id)
        .unwrap()
        .clone()
}

fn hint(tokens: usize) -> BatchHint {
    let mut hint = BatchHint::simple(3);
    hint.max_tokens = tokens;
    hint
}

#[tokio::test]
async fn prefix_restore_prepared_batch_cannot_admit_unrestored_waiting_work() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let hot = scheduler.submit(request()).await.unwrap();
    let waiting = scheduler.submit(request()).await.unwrap();
    admit(&scheduler, 1);
    let prepared = scheduler
        .prepare_prefix_restore(&hot, 0, 128)
        .unwrap()
        .unwrap();
    scheduler.commit_prefix_restored(prepared, 120).unwrap();

    let batch = scheduler
        .next_batch_with_prepared_admission_observed(
            hint(256),
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(NonZeroU64::new(1).unwrap(), 1, 1, 1),
                &[],
            ),
            &mut |_| {},
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].request.id, hot);
    assert_eq!(batch.requests[0].tokens_processed, 120);
    assert_eq!(batch.requests[0].tokens_to_process, Some(8));
    assert_eq!(scheduler.waiting_count(), 1);
    assert_eq!(scheduler.trace_phase(&waiting), Some(RequestPhase::Waiting));
}

#[tokio::test]
async fn prefix_restore_mixed_hot_and_cold_spend_only_actual_suffix_budget() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let hot = scheduler.submit(request()).await.unwrap();
    let cold = scheduler.submit(request()).await.unwrap();
    admit(&scheduler, 2);
    let before = snapshot(&scheduler, &hot);
    let trace_before = scheduler.trace_snapshot();
    let prepared = scheduler
        .prepare_prefix_restore(&hot, 0, 128)
        .unwrap()
        .unwrap();
    assert_eq!(prepared.request_id(), &hot);
    assert_eq!(prepared.expected_offset(), 0);
    assert_eq!(prepared.prompt_tokens(), 128);
    scheduler.commit_prefix_restored(prepared, 120).unwrap();

    let after = snapshot(&scheduler, &hot);
    assert_eq!(after.prefill_chunk_offset, 120);
    assert_eq!(after.inner.tokens_processed, 120);
    assert_eq!(
        after.logical_work_frontier.progress_generation(),
        before.logical_work_frontier.progress_generation()
    );
    assert_eq!(
        after.prefill_execution_chunk_ceiling,
        before.prefill_execution_chunk_ceiling
    );
    assert_eq!(after.prefill_time_ms, before.prefill_time_ms);
    assert_eq!(
        scheduler.trace_snapshot().capacity_release_epoch,
        trace_before.capacity_release_epoch
    );
    assert_eq!(
        scheduler.trace_snapshot().capacity_backpressure_admit_limit,
        trace_before.capacity_backpressure_admit_limit
    );
    assert_eq!(
        scheduler
            .metrics_tracker
            .iteration_count
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        scheduler
            .metrics_tracker
            .total_prefill_tokens
            .load(Ordering::Relaxed),
        0
    );

    let batch = scheduler.next_batch(hint(136)).await.unwrap();
    assert_eq!(batch.total_tokens(), 136);
    let hot_work = batch
        .requests
        .iter()
        .find(|work| work.request.id == hot)
        .unwrap();
    assert_eq!(hot_work.tokens_processed, 120);
    assert_eq!(hot_work.tokens_to_process, Some(8));
    let cold_work = batch
        .requests
        .iter()
        .find(|work| work.request.id == cold)
        .unwrap();
    assert_eq!(cold_work.tokens_processed, 0);
    assert_eq!(cold_work.tokens_to_process, Some(128));

    // A normal executor suffix commit advances work by eight, not by the
    // 120 imported tokens. Completion compute metrics retain that distinction.
    assert!(scheduler.mark_prefill_chunk_processed(&hot, 128, 8));
    let decode = scheduler.decode_queue.read();
    let hot_request = decode.requests.get(&hot).unwrap();
    assert_eq!(
        hot_request
            .logical_work_frontier
            .progress_generation()
            .get(),
        8
    );
    scheduler.metrics_tracker.record_completion(hot_request);
    assert_eq!(
        scheduler
            .metrics_tracker
            .total_prefill_tokens
            .load(Ordering::Relaxed),
        8
    );
}

#[tokio::test]
async fn prefix_restore_drop_releases_only_its_hold_without_progress() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let hot = scheduler.submit(request()).await.unwrap();
    let cold = scheduler.submit(request()).await.unwrap();
    admit(&scheduler, 2);
    let prepared = scheduler
        .prepare_prefix_restore(&hot, 0, 128)
        .unwrap()
        .unwrap();
    assert!(scheduler
        .prepare_prefix_restore(&hot, 0, 128)
        .unwrap()
        .is_none());
    let batch = scheduler.next_batch(hint(128)).await.unwrap();
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].request.id, cold);
    assert_eq!(snapshot(&scheduler, &hot).prefill_chunk_offset, 0);
    drop(prepared);
    assert!(!snapshot(&scheduler, &hot).prefix_restore.is_pending());
    let retry = scheduler
        .prepare_prefix_restore(&hot, 0, 128)
        .unwrap()
        .unwrap();
    drop(retry);
    let batch = scheduler.next_batch(hint(128)).await.unwrap();
    assert!(batch
        .requests
        .iter()
        .any(|work| work.request.id == hot && work.tokens_processed == 0));
}

#[tokio::test]
async fn prefix_restore_rejects_waiting_and_already_published_compute_work() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let id = scheduler.submit(request()).await.unwrap();
    assert!(scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .is_none());
    admit(&scheduler, 1);
    assert!(scheduler
        .prepare_prefix_restore(&id, 1, 128)
        .unwrap()
        .is_none());
    assert!(scheduler.prepare_prefix_restore(&id, 128, 128).is_err());
    assert!(scheduler.prepare_prefix_restore(&id, 0, 0).is_err());
    let batch = scheduler.next_batch(hint(64)).await.unwrap();
    assert_eq!(batch.total_tokens(), 64);
    assert!(scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .is_none());
    assert!(!scheduler.mark_prefill_chunk_processed(&id, 128, 64));
    let prepared = scheduler
        .prepare_prefix_restore(&id, 64, 128)
        .unwrap()
        .unwrap();
    scheduler.commit_prefix_restored(prepared, 96).unwrap();
    assert_eq!(
        snapshot(&scheduler, &id)
            .logical_work_frontier
            .progress_generation()
            .get(),
        64
    );
}

#[tokio::test]
async fn prefix_restore_rejects_stale_generation_without_partial_metadata_commit() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let id = scheduler.submit(request()).await.unwrap();
    admit(&scheduler, 1);
    let prepared = scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .unwrap();
    assert!(!scheduler.mark_prefill_chunk_processed(&id, 128, 8));
    let before = snapshot(&scheduler, &id);
    assert!(scheduler.commit_prefix_restored(prepared, 96).is_err());
    let after = snapshot(&scheduler, &id);
    assert_eq!(after.prefill_chunk_offset, before.prefill_chunk_offset);
    assert_eq!(after.logical_work_frontier, before.logical_work_frontier);
    assert_eq!(after.prefix_restore.restored_tokens(), 0);
    assert!(!after.prefix_restore.is_pending());
}

#[tokio::test]
async fn prefix_restore_zero_progress_requeue_changes_admission_incarnation() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let id = scheduler.submit(request()).await.unwrap();
    admit(&scheduler, 1);
    let before = snapshot(&scheduler, &id);
    let prepared = scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .unwrap();
    assert!(scheduler.defer_prefill_to_waiting(&id));
    assert!(scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .is_none());
    admit(&scheduler, 1);
    let after = snapshot(&scheduler, &id);
    assert_eq!(
        before.waiting_admission_ticket,
        after.waiting_admission_ticket
    );
    assert_eq!(
        before.logical_work_frontier.progress_generation(),
        after.logical_work_frontier.progress_generation()
    );
    assert_eq!(before.prefill_chunk_offset, after.prefill_chunk_offset);
    let current = scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .unwrap();
    assert!(scheduler.commit_prefix_restored(prepared, 96).is_err());
    assert!(snapshot(&scheduler, &id).prefix_restore.is_pending());
    assert_eq!(snapshot(&scheduler, &id).prefill_chunk_offset, 0);
    scheduler.commit_prefix_restored(current, 96).unwrap();
}

#[tokio::test]
async fn prefix_restore_cancel_and_replacement_cannot_commit_old_request() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let original = request();
    let id = scheduler.submit(original.clone()).await.unwrap();
    admit(&scheduler, 1);
    let prepared = scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .unwrap();
    assert!(scheduler.cancel(id.clone()).await.unwrap());
    assert!(scheduler.commit_prefix_restored(prepared, 96).is_err());
    scheduler.submit(original.clone()).await.unwrap();
    admit(&scheduler, 1);
    let prepared = scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .unwrap();
    assert!(scheduler.cancel(id.clone()).await.unwrap());
    scheduler.submit(original).await.unwrap();
    admit(&scheduler, 1);
    assert!(scheduler.commit_prefix_restored(prepared, 96).is_err());
    assert_eq!(snapshot(&scheduler, &id).prefill_chunk_offset, 0);
}

#[tokio::test]
async fn prefix_restore_duplicate_active_id_cannot_replace_prepared_admission() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let original = request();
    let id = scheduler.submit(original.clone()).await.unwrap();
    assert!(scheduler.submit(original.clone()).await.is_err());
    assert_eq!(scheduler.waiting_count(), 1);
    admit(&scheduler, 1);
    let before = snapshot(&scheduler, &id);
    let prepared = scheduler
        .prepare_prefix_restore(&id, 0, 128)
        .unwrap()
        .unwrap();
    assert!(scheduler.submit(original).await.is_err());
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(scheduler.prefilling_count(), 1);
    assert_eq!(
        snapshot(&scheduler, &id).waiting_admission_ticket,
        before.waiting_admission_ticket
    );
    scheduler.commit_prefix_restored(prepared, 96).unwrap();
    assert_eq!(snapshot(&scheduler, &id).prefill_chunk_offset, 96);
}

#[tokio::test]
async fn prefix_restore_foreign_scheduler_and_invalid_boundary_fail_closed() {
    let first = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let second = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let original = request();
    let id = first.submit(original.clone()).await.unwrap();
    second.submit(original).await.unwrap();
    admit(&first, 1);
    admit(&second, 1);
    let first_prepared = first.prepare_prefix_restore(&id, 0, 128).unwrap().unwrap();
    let second_prepared = second.prepare_prefix_restore(&id, 0, 128).unwrap().unwrap();
    assert!(second.commit_prefix_restored(first_prepared, 96).is_err());
    assert!(snapshot(&second, &id).prefix_restore.is_pending());
    second.commit_prefix_restored(second_prepared, 96).unwrap();
    for invalid in [0, 128, usize::MAX] {
        let before = snapshot(&first, &id);
        let prepared = first.prepare_prefix_restore(&id, 0, 128).unwrap().unwrap();
        assert!(first.commit_prefix_restored(prepared, invalid).is_err());
        let after = snapshot(&first, &id);
        assert_eq!(after.logical_work_frontier, before.logical_work_frontier);
        assert_eq!(after.prefill_chunk_offset, 0);
        assert_eq!(after.prefix_restore.restored_tokens(), 0);
        assert!(!after.prefix_restore.is_pending());
    }
    let foreign = PreparedPrefixRestore::new(id.clone(), 0, 128, ());
    assert!(first.commit_prefix_restored(foreign, 96).is_err());
}
