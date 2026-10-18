use super::*;

#[test]
fn request_clock_queue_segments_preserve_ingress_and_original_submission() {
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::HttpHandler);
    let mut request =
        ScheduledRequest::new(InferenceRequest::new("prompt", "test").with_ingress(ingress));
    let submitted = request.submitted_offset();
    let queue_entry = Instant::now() + Duration::from_secs(10);
    request.mark_queued_at(queue_entry);
    assert_eq!(request.queue_wait_at(queue_entry), Duration::ZERO);
    assert_eq!(request.processing_time_at(queue_entry), None);

    let first_start = queue_entry + Duration::from_secs(2);
    assert_eq!(request.mark_started_at(first_start), Duration::from_secs(2));
    let yield_at = first_start + Duration::from_secs(7);
    assert_eq!(request.queue_wait_at(yield_at), Duration::from_secs(2));
    assert_eq!(
        request.processing_time_at(yield_at),
        Some(Duration::from_secs(7))
    );

    request.requeue_at(yield_at);
    assert_eq!(request.processing_time_at(yield_at), None);
    assert_eq!(request.queue_wait_at(yield_at), Duration::ZERO);
    let next_start = yield_at + Duration::from_secs(3);
    assert_eq!(request.mark_started_at(next_start), Duration::from_secs(3));
    assert_eq!(request.ingress(), ingress);
    assert_eq!(request.submitted_offset(), submitted);
    assert_eq!(request.age_at(next_start), ingress.elapsed_at(next_start));
    assert_eq!(request.processing_time_at(next_start), Some(Duration::ZERO));
}

#[test]
fn request_clock_wall_changes_cannot_change_scheduling_durations() {
    let mut request = ScheduledRequest::new(InferenceRequest::new("prompt", "test"));
    assert_eq!(
        request.ingress().boundary(),
        RequestIngressBoundary::SchedulerApi
    );
    let queued = Instant::now();
    request.mark_queued_at(queued);
    request.mark_started_at(queued + Duration::from_secs(1));
    let now = queued + Duration::from_secs(4);
    let expected = (
        request.age_at(now),
        request.queue_wait_at(now),
        request.processing_time_at(now),
    );
    for seconds in [0, 7_258_118_400] {
        let wall = chrono::DateTime::from_timestamp(seconds, 0).unwrap();
        request.request.created_at = wall;
        request.submitted_at = wall;
        request.started_at = Some(wall);
        assert_eq!(
            (
                request.age_at(now),
                request.queue_wait_at(now),
                request.processing_time_at(now)
            ),
            expected
        );
    }
    // Public request metadata cannot replace the scheduler's original sample.
    request.request.ingress = None;
    assert_eq!(request.age_at(now), expected.0);
}
