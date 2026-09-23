use super::*;
use ferrum_types::ModelId;

async fn fixture() -> (ContinuousBatchScheduler, InferenceRequest) {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let mut request = InferenceRequest::new("prompt", ModelId::new("fixture"));
    request.sampling_params.max_tokens = 4;
    scheduler.submit(request.clone()).await.unwrap();
    assert!(scheduler.promote_to_prefill_with_empty_retry(&request.id, None));
    (scheduler, request)
}

fn frontier(s: &ContinuousBatchScheduler, id: &RequestId) -> LogicalWorkFrontier {
    s.decode_queue.read().requests[id]
        .logical_work_frontier
        .clone()
}

#[tokio::test]
async fn prefill_output_publication_does_not_compute_the_sampled_token_twice() {
    let (s, request) = fixture().await;
    let id = &request.id;
    let publication = s.prepare_prefill_output_publication(id, 4, 0).unwrap();
    s.mark_prefill_complete(id, 4);
    let before = frontier(&s, id);
    assert_eq!(
        s.publish_prefill_output_commit(&publication, 1).unwrap(),
        PrefillOutputPublicationOutcome::Published
    );
    let after = frontier(&s, id);
    assert_eq!(after.planning_counters(), (4, 4, 4, 1, None));
    assert_eq!(
        after.progress_generation().get(),
        before.progress_generation().get() + 1
    );
    assert_eq!(s.decode_queue.read().requests[id].decode_tokens, 0);
    assert_eq!(
        s.publish_prefill_output_commit(&publication, 1).unwrap(),
        PrefillOutputPublicationOutcome::AlreadyPublished
    );
    assert_eq!(frontier(&s, id), after);
    assert!(s.prepare_prefill_output_publication(id, 4, 1).is_err());
    s.update_decode_progress(id, 2);
    assert_eq!(frontier(&s, id).planning_counters(), (5, 5, 5, 2, None));
    assert!(s.publish_prefill_output_commit(&publication, 1).is_err());
}

#[tokio::test]
async fn prefill_output_publication_requires_the_complete_actual_boundary() {
    let (s, request) = fixture().await;
    let id = &request.id;
    assert!(!s.mark_prefill_chunk_processed(id, 4, 2));
    let publication = s.prepare_prefill_output_publication(id, 4, 0).unwrap();
    assert!(s.publish_prefill_output_commit(&publication, 1).is_err());
    assert_eq!(
        s.prefill_queue.read()[0]
            .logical_work_frontier
            .planning_counters(),
        (2, 2, 2, 0, None)
    );
    assert!(s
        .mark_prefill_chunk_processed_with_capacity_feedback(id, 4, 4, 2)
        .unwrap());
    assert!(s.publish_prefill_output_commit(&publication, 0).is_err());
    assert!(s.publish_prefill_output_commit(&publication, 2).is_err());
    s.publish_prefill_output_commit(&publication, 1).unwrap();
    assert_eq!(frontier(&s, id).planning_counters(), (4, 4, 4, 1, None));
}

#[tokio::test]
async fn prefill_output_publication_rejects_reused_id_and_another_scheduler() {
    let (s, request) = fixture().await;
    let publication = s
        .prepare_prefill_output_publication(&request.id, 4, 0)
        .unwrap();
    s.cancel(request.id.clone()).await.unwrap();
    s.submit(request.clone()).await.unwrap();
    assert!(s.promote_to_prefill_with_empty_retry(&request.id, None));
    s.mark_prefill_complete(&request.id, 4);
    let before = frontier(&s, &request.id);
    assert!(s.publish_prefill_output_commit(&publication, 1).is_err());
    assert_eq!(frontier(&s, &request.id), before);
    let other = ContinuousBatchScheduler::new(SchedulerConfig::default());
    other.submit(request.clone()).await.unwrap();
    assert!(other.promote_to_prefill_with_empty_retry(&request.id, None));
    other.mark_prefill_complete(&request.id, 4);
    assert!(other
        .publish_prefill_output_commit(&publication, 1)
        .is_err());
    assert_eq!(
        frontier(&other, &request.id).planning_counters(),
        (4, 4, 4, 0, None)
    );
}

#[tokio::test]
async fn prefill_output_publication_recompute_counts_only_new_sampling() {
    let (s, request) = fixture().await;
    let id = &request.id;
    let publication = s.prepare_prefill_output_publication(id, 4, 0).unwrap();
    s.mark_prefill_complete(id, 4);
    s.publish_prefill_output_commit(&publication, 1).unwrap();
    s.update_decode_progress(id, 2);
    let generation = frontier(&s, id).progress_generation();
    assert!(s.defer_decode_to_waiting_for_capacity(id, 1));
    assert!(s.promote_to_prefill_with_empty_retry(id, None));
    assert!(!s.mark_prefill_chunk_processed(id, 6, 3));
    let publication = s.prepare_prefill_output_publication(id, 6, 2).unwrap();
    s.mark_prefill_complete(id, 6);
    assert_eq!(frontier(&s, id).progress_generation(), generation);
    s.publish_prefill_output_commit(&publication, 3).unwrap();
    assert_eq!(frontier(&s, id).planning_counters(), (6, 6, 6, 3, None));
    assert_eq!(
        frontier(&s, id).progress_generation().get(),
        generation.get() + 1
    );
    s.update_decode_progress(id, 4);
    assert_eq!(frontier(&s, id).planning_counters(), (7, 7, 7, 4, None));
}

#[tokio::test]
async fn prefill_output_publication_legacy_completed_chunk_and_limits() {
    let (s, request) = fixture().await;
    let id = &request.id;
    assert!(s.mark_prefill_chunk_processed(id, 4, 4));
    assert!(s.prepare_prefill_output_publication(id, 3, 0).is_err());
    let publication = s.prepare_prefill_output_publication(id, 4, 0).unwrap();
    s.publish_prefill_output_commit(&publication, 1).unwrap();
    s.update_decode_progress(id, 4);
    assert!(s.defer_decode_to_waiting_for_capacity(id, 1));
    assert!(s.promote_to_prefill_with_empty_retry(id, None));
    assert!(s.prepare_prefill_output_publication(id, 8, 4).is_err());
}
