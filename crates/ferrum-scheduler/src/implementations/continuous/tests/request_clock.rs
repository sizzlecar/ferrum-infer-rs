//! Scheduler queue transitions with real local ingress samples. Physical
//! executor release is supplied explicitly; these tests do not model devices.
use super::*;
use ferrum_types::RequestIngressAnchor;

fn waiting_clock(
    scheduler: &ContinuousBatchScheduler,
    request_id: &RequestId,
) -> (ScheduledRequest, WaitingAdmissionTicket) {
    let waiting = scheduler.waiting_queue.read();
    let request = waiting
        .iter()
        .find(|request| &request.inner.request.id == request_id)
        .unwrap();
    (
        request.inner.clone(),
        request.waiting_admission_ticket.unwrap(),
    )
}

fn admit(scheduler: &ContinuousBatchScheduler, request_id: &RequestId) {
    let receipt = scheduler
        .prepare_token_admission_observed(
            1,
            None,
            &mut |request| {
                assert_eq!(&request.id, request_id);
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                })
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(receipt.admitted(), 1);
}

fn prefill_clock(scheduler: &ContinuousBatchScheduler, id: &RequestId) -> ScheduledRequest {
    scheduler
        .prefill_queue
        .read()
        .iter()
        .find(|request| &request.inner.request.id == id)
        .unwrap()
        .inner
        .clone()
}

fn wait_us(request: &ScheduledRequest) -> u64 {
    request
        .queue_wait_at(Instant::now())
        .as_micros()
        .min(u64::MAX as u128) as u64
}

#[test]
fn request_clock_capacity_requeue_counts_only_each_waiting_segment() {
    for from_decode in [false, true] {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
        let mut request =
            create_test_request_with_prompt_tokens(Priority::Normal, 4).with_ingress(ingress);
        request.created_at = chrono::Utc::now() - chrono::Duration::days(3650);
        let id = request.id.clone();
        scheduler.submit_request(request).unwrap();
        let (first_waiting, ticket) = waiting_clock(&scheduler, &id);
        assert_eq!(first_waiting.ingress(), ingress);
        assert_eq!(first_waiting.request.ingress, Some(ingress));
        assert!(first_waiting.queue_wait_at(Instant::now()) <= first_waiting.age());
        admit(&scheduler, &id);
        let first_active = prefill_clock(&scheduler, &id);
        let first_wait_us = wait_us(&first_active);
        assert_eq!(
            scheduler.total_wait_time_us.load(Ordering::Relaxed),
            first_wait_us
        );

        // Only diagnostics move with wall time; this must not affect the next
        // promotion's queue metric or the original end-to-end request age.
        {
            let mut prefill = scheduler.prefill_queue.write();
            let request = prefill
                .iter_mut()
                .find(|request| request.inner.request.id == id)
                .unwrap();
            request.inner.submitted_at = chrono::Utc::now() + chrono::Duration::days(3650);
            request.inner.started_at = Some(chrono::Utc::now() - chrono::Duration::days(3650));
        }
        if from_decode {
            scheduler.mark_prefill_complete(&id, 4);
            scheduler.update_decode_progress(&id, 1);
        }
        let before_requeue = Instant::now();
        assert!(if from_decode {
            scheduler.defer_decode_to_waiting_for_capacity(&id, 1)
        } else {
            scheduler.defer_prefill_to_waiting(&id)
        });
        let after_requeue = Instant::now();
        let (requeued, requeued_ticket) = waiting_clock(&scheduler, &id);
        assert_eq!(requeued_ticket, ticket);
        assert_eq!(requeued.ingress(), ingress);
        assert_eq!(
            requeued.submitted_offset(),
            first_waiting.submitted_offset()
        );
        assert_eq!(requeued.processing_time_at(after_requeue), None);
        assert!(
            requeued.queue_wait_at(after_requeue) <= after_requeue.duration_since(before_requeue)
        );
        assert_eq!(
            requeued.age_at(after_requeue),
            ingress.elapsed_at(after_requeue)
        );
        assert_eq!(
            scheduler.total_wait_time_us.load(Ordering::Relaxed),
            first_wait_us
        );
        assert_eq!(scheduler.confirmed_capacity_release_epoch(), 0);

        // The physical owner notification is separate from logical requeue.
        scheduler.record_external_capacity_release_after_recompute();
        admit(&scheduler, &id);
        let second_active = prefill_clock(&scheduler, &id);
        assert_eq!(second_active.ingress(), ingress);
        assert_eq!(
            second_active.submitted_offset(),
            first_waiting.submitted_offset()
        );
        assert_eq!(
            scheduler.total_wait_time_us.load(Ordering::Relaxed),
            first_wait_us + wait_us(&second_active),
            "re-admission must not count the earlier active epoch as queue wait"
        );
        assert_eq!(scheduler.admitted_counter.load(Ordering::Relaxed), 2);
        assert!(scheduler.cancel_request(id).unwrap());
    }
}

#[tokio::test]
async fn request_clock_deferred_and_cancelled_admission_preserve_waiting_segment() {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let request = create_test_request(Priority::Normal);
    assert_eq!(request.ingress, None);
    let id = request.id.clone();
    scheduler.submit_request(request).unwrap();
    let (original, ticket) = waiting_clock(&scheduler, &id);
    assert_eq!(
        original.ingress().boundary(),
        RequestIngressBoundary::SchedulerApi
    );
    let check_at = Instant::now() + Duration::from_secs(1);
    let expected_wait = original.queue_wait_at(check_at);

    let mut round = scheduler.begin_token_admission(1, None).unwrap();
    let candidate = round.next(&mut |_| {}).unwrap().unwrap();
    assert_eq!(candidate.request().ingress, Some(original.ingress()));
    assert!(matches!(
        round.apply(candidate, AdmissionProbeOutcome::CapacityDeferred),
        TokenAdmissionApply::Applied { accepted: None }
    ));
    assert_eq!(round.finish().unwrap().deferred(), 1);
    let (deferred, same_ticket) = waiting_clock(&scheduler, &id);
    assert_eq!(same_ticket, ticket);
    assert_eq!(deferred.queue_wait_at(check_at), expected_wait);
    assert_eq!(deferred.submitted_offset(), original.submitted_offset());
    scheduler.record_external_capacity_release();

    // Cancellation of successive admission futures releases their in-flight
    // reservations, but neither one constitutes leaving the waiting queue.
    for _ in 0..2 {
        let (selected, selected_rx) = tokio::sync::oneshot::channel();
        let task_scheduler = Arc::clone(&scheduler);
        let task = tokio::spawn(async move {
            let mut round = task_scheduler.begin_token_admission(1, None).unwrap();
            let candidate = round.next(&mut |_| {}).unwrap().unwrap();
            selected.send(candidate.key().ordinal()).unwrap();
            std::future::pending::<()>().await;
            drop(candidate);
            drop(round);
        });
        let selected_ticket = tokio::time::timeout(Duration::from_secs(2), selected_rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(selected_ticket, ticket.get());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let (retained, retained_ticket) = waiting_clock(&scheduler, &id);
        assert_eq!(retained_ticket, ticket);
        assert_eq!(retained.ingress(), original.ingress());
        assert_eq!(retained.submitted_offset(), original.submitted_offset());
        assert_eq!(retained.queue_wait_at(check_at), expected_wait);
        assert_eq!(retained.processing_time_at(check_at), None);
    }
    assert_eq!(scheduler.confirmed_capacity_release_epoch(), 1);
    admit(&scheduler, &id);
    let active = prefill_clock(&scheduler, &id);
    assert_eq!(active.ingress(), original.ingress());
    assert_eq!(active.submitted_offset(), original.submitted_offset());
    assert_eq!(
        scheduler.total_wait_time_us.load(Ordering::Relaxed),
        wait_us(&active)
    );
    assert!(scheduler.cancel_request(id).unwrap());
}
