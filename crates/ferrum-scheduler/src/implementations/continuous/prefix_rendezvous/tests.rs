use super::*;

fn plan(boundary: usize) -> PrefixCapturePlan {
    PrefixCapturePlan {
        boundary,
        span: ferrum_interfaces::vnext::CheckpointTokenSpanConstraint::any_positive(),
    }
}

fn request() -> InferenceRequest {
    InferenceRequest::new("shared prefix with a distinct suffix", "test")
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, serde_json::json!(13))
}

fn wake() -> AdmissionWakeSnapshot<'static> {
    AdmissionWakeSnapshot::new(
        AdmissionWakeEpochs::new(NonZeroU64::new(1).unwrap(), 1, 1, 1),
        &[],
    )
}

fn admit(scheduler: &ContinuousBatchScheduler, maximum: usize) -> Vec<RequestId> {
    let mut probed = Vec::new();
    scheduler
        .prepare_dynamic_admission_observed(
            maximum,
            wake(),
            &mut |request| {
                probed.push(request.id.clone());
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                })
            },
            &mut |_| {},
        )
        .unwrap();
    probed
}

fn key(scheduler: &ContinuousBatchScheduler, id: &RequestId) -> PrefixRequestKey {
    scheduler
        .prefix_rendezvous_candidates()
        .into_iter()
        .find(|item| item.key.request_id() == id)
        .unwrap()
        .key
}

#[tokio::test]
async fn held_followers_do_not_probe_capacity_and_unrelated_work_advances() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        prefill_step_chunk: Some(8),
        prefill_first_until_active: Some(3),
        ..Default::default()
    });
    let source = scheduler.submit(request()).await.unwrap();
    let follower = scheduler.submit(request()).await.unwrap();
    let other = scheduler.submit(request()).await.unwrap();
    let source_key = key(&scheduler, &source);
    let follower_key = key(&scheduler, &follower);
    let hold = scheduler
        .hold_prefix_followers(&source_key, std::slice::from_ref(&follower_key), plan(7))
        .unwrap();
    assert_eq!(admit(&scheduler, 3), [source.clone(), other.clone()]);
    assert_eq!(
        scheduler.prefix_request_progress(&follower_key),
        Some((true, 0))
    );
    let mut hint = BatchHint::simple(3);
    hint.max_tokens = 32;
    let batch = scheduler
        .next_batch_with_prepared_admission_observed(hint, wake(), &mut |_| {})
        .unwrap()
        .unwrap();
    assert_eq!(batch.requests.len(), 2);
    assert_eq!(
        batch
            .requests
            .iter()
            .find(|request| request.request.id == source)
            .unwrap()
            .tokens_to_process,
        Some(7)
    );
    assert!(batch
        .requests
        .iter()
        .any(|request| request.request.id == other));
    hold.release();
    assert_eq!(admit(&scheduler, 1), [follower.clone()]);
    let prepared = scheduler
        .prepare_prefix_restore(&follower, 0, 13)
        .unwrap()
        .unwrap();
    scheduler.commit_prefix_restored(prepared, 7).unwrap();
    assert_eq!(
        scheduler.prefix_request_progress(&follower_key),
        Some((false, 7))
    );
}

#[tokio::test]
async fn unavailable_drops_hold_without_rearming_or_reusing_old_request_identity() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let source = scheduler.submit(request()).await.unwrap();
    let follower_request = request();
    let follower = scheduler.submit(follower_request.clone()).await.unwrap();
    let source_key = key(&scheduler, &source);
    let old_key = key(&scheduler, &follower);
    let hold = scheduler
        .hold_prefix_followers(&source_key, std::slice::from_ref(&old_key), plan(9))
        .unwrap();
    assert!(scheduler.cancel(source.clone()).await.unwrap());
    assert!(scheduler.prefix_request_progress(&source_key).is_none());
    drop(hold);
    assert_eq!(scheduler.prefix_held_waiting_count(), 0);
    assert!(!scheduler
        .prefix_rendezvous_candidates()
        .iter()
        .any(|candidate| candidate.key.request_id() == &follower));
    assert_eq!(admit(&scheduler, 1), [follower.clone()]);
    assert!(scheduler.cancel(follower.clone()).await.unwrap());
    scheduler.submit(follower_request).await.unwrap();
    let replacement = key(&scheduler, &follower);
    assert!(scheduler.prefix_request_progress(&old_key).is_none());
    assert_eq!(
        scheduler.prefix_request_progress(&replacement),
        Some((true, 0))
    );
}

#[tokio::test]
async fn prefix_hold_rejects_active_followers_and_higher_priority_without_partial_install() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let source = scheduler.submit(request()).await.unwrap();
    let mut urgent_request = request();
    urgent_request.priority = Priority::Critical;
    let urgent = scheduler.submit(urgent_request).await.unwrap();
    let source_key = key(&scheduler, &source);
    let urgent_key = key(&scheduler, &urgent);
    assert!(scheduler
        .hold_prefix_followers(&source_key, &[urgent_key], plan(7))
        .is_none());
    assert_eq!(scheduler.prefix_held_waiting_count(), 0);
    admit(&scheduler, 2);
    assert!(scheduler
        .hold_prefix_followers(
            &key(&scheduler, &source),
            &[key(&scheduler, &urgent)],
            plan(7)
        )
        .is_none());
    assert_eq!(scheduler.prefix_held_waiting_count(), 0);
}

#[tokio::test]
async fn prefix_capture_span_alignment_is_preserved_across_shorter_scheduler_chunks() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        prefill_step_chunk: Some(3),
        ..Default::default()
    });
    let source = scheduler.submit(request()).await.unwrap();
    let follower = scheduler.submit(request()).await.unwrap();
    let span = ferrum_interfaces::vnext::CheckpointTokenSpanConstraint::new(
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(2).unwrap(),
    )
    .unwrap();
    let hold = scheduler
        .hold_prefix_followers(
            &key(&scheduler, &source),
            &[key(&scheduler, &follower)],
            PrefixCapturePlan { boundary: 10, span },
        )
        .unwrap();
    admit(&scheduler, 2);
    let mut offset = 0;
    while offset < 10 {
        let mut hint = BatchHint::simple(2);
        hint.max_tokens = 3;
        let batch = scheduler
            .next_batch_with_prepared_admission_observed(hint, wake(), &mut |_| {})
            .unwrap()
            .unwrap();
        assert_eq!(batch.requests.len(), 1);
        let chunk = batch.requests[0].tokens_to_process.unwrap();
        assert!(span.permits(chunk as u64));
        assert!(offset + chunk <= 10);
        scheduler.mark_prefill_chunk_processed(&source, 13, chunk);
        offset += chunk;
    }
    assert!(hold.is_pending());
    hold.release();
    assert_eq!(admit(&scheduler, 1), [follower]);
}

#[tokio::test]
async fn mixed_decode_budget_below_legal_span_releases_prefix_wait_without_borrowing_tokens() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        prefill_step_chunk: Some(3),
        prefill_first_until_active: Some(3),
        prefix_rendezvous_max_wait_ms: NonZeroU64::new(10_000),
        ..Default::default()
    });
    let decode = scheduler.submit(request()).await.unwrap();
    admit(&scheduler, 1);
    scheduler.mark_prefill_complete(&decode, 13);
    let source = scheduler.submit(request()).await.unwrap();
    let follower = scheduler.submit(request()).await.unwrap();
    let span = ferrum_interfaces::vnext::CheckpointTokenSpanConstraint::new(
        NonZeroU64::new(4).unwrap(),
        NonZeroU64::new(4).unwrap(),
    )
    .unwrap();
    let hold = scheduler
        .hold_prefix_followers(
            &key(&scheduler, &source),
            &[key(&scheduler, &follower)],
            PrefixCapturePlan { boundary: 8, span },
        )
        .unwrap();
    admit(&scheduler, 2);
    let mut hint = BatchHint::simple(3);
    hint.max_tokens = 3;
    let batch = scheduler
        .next_batch_with_prepared_admission_observed(hint, wake(), &mut |_| {})
        .unwrap()
        .unwrap();
    assert!(batch
        .requests
        .iter()
        .any(|request| request.request.id == decode));
    assert!(batch
        .requests
        .iter()
        .any(|request| request.request.id == source));
    assert!(
        batch
            .requests
            .iter()
            .map(|request| request.tokens_to_process.unwrap_or(1))
            .sum::<usize>()
            <= 3
    );
    assert!(!hold.is_pending());
    assert_eq!(scheduler.prefix_held_waiting_count(), 0);
    assert_eq!(admit(&scheduler, 1), [follower]);
}
