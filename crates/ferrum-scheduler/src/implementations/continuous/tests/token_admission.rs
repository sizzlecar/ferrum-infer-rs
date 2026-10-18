use super::*;

fn admitted(request_id: &RequestId) -> ExecutorAdmissionProbeOutcome {
    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
        request_id: request_id.clone(),
    })
}

fn accepted(result: TokenAdmissionApply) -> TokenAdmissionAccepted {
    match result {
        TokenAdmissionApply::Applied {
            accepted: Some(accepted),
        } => accepted,
        _ => panic!("exact live admission must be accepted"),
    }
}

#[tokio::test]
async fn token_admission_round_cancel_and_same_id_replacement_return_old_outcome() {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let request = create_test_request(Priority::Normal);
    let id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    let (selected, selected_rx) = tokio::sync::oneshot::channel();
    let (resume, resume_rx) = tokio::sync::oneshot::channel();
    let task_scheduler = Arc::clone(&scheduler);
    let task = tokio::spawn(async move {
        let mut round = task_scheduler.begin_token_admission(1, None).unwrap();
        let candidate = round.next(&mut |_| {}).unwrap().unwrap();
        let key = candidate.key().clone();
        selected.send(key.clone()).unwrap();
        resume_rx.await.unwrap();
        let outcome = admitted(&candidate.request().id);
        match round.apply(candidate, outcome) {
            TokenAdmissionApply::Unapplied {
                outcome: AdmissionProbeOutcome::Admitted(receipt),
                error: None,
            } => {
                assert_eq!(&receipt.request_id, key.request_id());
            }
            _ => panic!("old executor result must be returned intact after cancellation"),
        }
        let receipt = round.finish().unwrap();
        assert_eq!(receipt.admitted(), 0);
        key
    });
    let old_key = selected_rx.await.unwrap();
    // These operations must complete while the executor admission is awaiting.
    assert!(scheduler.cancel_request(id.clone()).unwrap());
    scheduler.submit(request).await.unwrap();
    resume.send(()).unwrap();
    assert_eq!(task.await.unwrap(), old_key);
    assert_eq!(scheduler.waiting_count(), 1);
    assert_eq!(scheduler.trace_phase(&id), Some(RequestPhase::Waiting));
    let mut round = scheduler.begin_token_admission(1, None).unwrap();
    let replacement = round.next(&mut |_| {}).unwrap().unwrap();
    assert_ne!(replacement.key(), &old_key);
    let new = accepted(round.apply(replacement, admitted(&id)));
    assert_eq!(round.finish().unwrap().admitted(), 1);
    assert!(scheduler.cancel_token_admission(&new).unwrap());
    assert_eq!(scheduler.trace_snapshot().capacity_release_epoch, 0);
}

#[tokio::test]
async fn token_admission_round_future_drop_restores_original_ticket_without_fake_deferral() {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let request = create_test_request(Priority::Normal);
    let id = request.id.clone();
    scheduler.submit(request).await.unwrap();
    let (selected, selected_rx) = tokio::sync::oneshot::channel();
    let task_scheduler = Arc::clone(&scheduler);
    let task = tokio::spawn(async move {
        let mut round = task_scheduler.begin_token_admission(1, None).unwrap();
        let candidate = round.next(&mut |_| {}).unwrap().unwrap();
        selected.send(candidate.key().clone()).unwrap();
        std::future::pending::<()>().await;
        drop(candidate);
        drop(round);
    });
    let key = selected_rx.await.unwrap();
    assert!(scheduler.begin_token_admission(1, None).is_err());
    assert!(scheduler.next_batch(BatchHint::simple(1)).await.is_none());
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(scheduler.waiting_count(), 1);
    assert_eq!(scheduler.trace_phase(&id), Some(RequestPhase::Waiting));
    assert!(!scheduler.waiting_for_confirmed_capacity_release());
    let mut round = scheduler.begin_token_admission(1, None).unwrap();
    let candidate = round.next(&mut |_| {}).unwrap().unwrap();
    assert_eq!(candidate.key(), &key);
    let accepted = accepted(round.apply(candidate, admitted(&id)));
    assert_eq!(round.finish().unwrap().admitted(), 1);
    assert!(scheduler.cancel_token_admission(&accepted).unwrap());
    assert_eq!(scheduler.trace_snapshot().capacity_release_epoch, 0);
}

#[tokio::test]
async fn token_admission_round_exact_accepted_rollback_cannot_cancel_replacement() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let request = create_test_request(Priority::Normal);
    let id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    let mut round = scheduler.begin_token_admission(1, None).unwrap();
    let candidate = round.next(&mut |_| {}).unwrap().unwrap();
    let original = accepted(round.apply(candidate, admitted(&id)));
    round.finish().unwrap();
    assert!(scheduler.cancel_token_admission(&original).unwrap());
    scheduler.submit(request).await.unwrap();
    let mut round = scheduler.begin_token_admission(1, None).unwrap();
    let candidate = round.next(&mut |_| {}).unwrap().unwrap();
    let replacement = accepted(round.apply(candidate, admitted(&id)));
    round.finish().unwrap();
    assert_ne!(original.key(), replacement.key());
    assert!(!scheduler.cancel_token_admission(&original).unwrap());
    assert_eq!(scheduler.trace_phase(&id), Some(RequestPhase::Prefilling));
    assert_eq!(scheduler.active_count(), 1);
    assert!(scheduler.cancel_token_admission(&replacement).unwrap());
    assert_eq!(scheduler.trace_snapshot().capacity_release_epoch, 0);
}

#[tokio::test]
async fn token_admission_round_preserves_priority_budget_and_defers_concurrent_insert_to_next_round(
) {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 2,
        ..SchedulerConfig::default()
    });
    let first = create_test_request(Priority::Normal);
    let first_id = first.id.clone();
    scheduler.submit(first).await.unwrap();
    let mut round = scheduler.begin_token_admission(2, None).unwrap();
    let candidate = round.next(&mut |_| {}).unwrap().unwrap();
    scheduler
        .update_priority(first_id.clone(), Priority::High)
        .await
        .unwrap();
    let later = create_test_request(Priority::Normal);
    let later_id = later.id.clone();
    scheduler.submit(later).await.unwrap();
    let first_owner = accepted(round.apply(candidate, admitted(&first_id)));
    assert!(round.next(&mut |_| {}).unwrap().is_none());
    assert_eq!(round.finish().unwrap().admitted(), 1);
    assert_eq!(
        scheduler.prefill_queue.read()[0].inner.request.priority,
        Priority::High
    );
    let mut second = scheduler.begin_token_admission(8, None).unwrap();
    let candidate = second.next(&mut |_| {}).unwrap().unwrap();
    assert_eq!(candidate.request().id, later_id);
    let later_owner = accepted(second.apply(candidate, admitted(&later_id)));
    assert!(second.next(&mut |_| {}).unwrap().is_none());
    assert_eq!(second.finish().unwrap().admitted(), 1);
    assert_eq!(scheduler.active_count(), 2);
    assert!(scheduler.cancel_token_admission(&first_owner).unwrap());
    assert!(scheduler.cancel_token_admission(&later_owner).unwrap());
}

#[tokio::test]
async fn token_admission_round_release_during_await_is_not_lost() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let request = create_test_request(Priority::Normal);
    let id = request.id.clone();
    scheduler.submit(request).await.unwrap();
    let mut round = scheduler.begin_token_admission(1, None).unwrap();
    let candidate = round.next(&mut |_| {}).unwrap().unwrap();
    scheduler.record_external_capacity_release();
    assert!(matches!(
        round.apply(candidate, AdmissionProbeOutcome::CapacityDeferred),
        TokenAdmissionApply::Applied { accepted: None }
    ));
    assert_eq!(round.finish().unwrap().deferred(), 1);
    assert!(!scheduler.waiting_for_confirmed_capacity_release());
    let mut retry = scheduler.begin_token_admission(1, None).unwrap();
    let candidate = retry.next(&mut |_| {}).unwrap().unwrap();
    let owner = accepted(retry.apply(candidate, admitted(&id)));
    retry.finish().unwrap();
    assert!(scheduler.cancel_token_admission(&owner).unwrap());
    assert_eq!(scheduler.trace_snapshot().capacity_release_epoch, 1);
}

#[tokio::test]
async fn token_admission_round_stale_round_cannot_consume_current_candidate() {
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let request = create_test_request(Priority::Normal);
    let id = request.id.clone();
    scheduler.submit(request).await.unwrap();
    let mut old = scheduler.begin_token_admission(1, None).unwrap();
    let stale = old.next(&mut |_| {}).unwrap().unwrap();
    drop(old);
    let mut current = scheduler.begin_token_admission(1, None).unwrap();
    let candidate = current.next(&mut |_| {}).unwrap().unwrap();
    match current.apply(stale, admitted(&id)) {
        TokenAdmissionApply::Unapplied {
            outcome: AdmissionProbeOutcome::Admitted(receipt),
            error: Some(_),
        } => {
            assert_eq!(receipt.request_id, id);
        }
        _ => panic!("another scan must not consume or misapply the current reservation"),
    }
    let owner = accepted(current.apply(candidate, admitted(&id)));
    assert_eq!(current.finish().unwrap().admitted(), 1);
    assert!(scheduler.cancel_token_admission(&owner).unwrap());
}

#[tokio::test]
async fn token_admission_round_preserves_exact_native_wake_and_skip_observation() {
    use ferrum_runtime::DeferredAction;
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
    let request = create_test_request(Priority::Normal);
    let id = request.id.clone();
    scheduler.submit(request).await.unwrap();
    let coordinator = NonZeroU64::new(31).unwrap();
    let first_epochs = AdmissionWakeEpochs::new(coordinator, 0, 0, 0);
    let availability =
        [
            CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 4)
                .unwrap(),
        ];
    let condition = CapacityWaitCondition::from_observation(31, availability.to_vec()).unwrap();
    let mut round = scheduler
        .begin_token_admission(
            1,
            Some(AdmissionWakeSnapshot::new(first_epochs, &availability)),
        )
        .unwrap();
    let candidate = round.next(&mut |_| {}).unwrap().unwrap();
    let key = candidate.key().clone();
    assert!(matches!(
        round.apply(
            candidate,
            AdmissionProbeOutcome::Deferred(AdmissionDeferral::new(
                DeferredAction::WaitForRelease,
                first_epochs,
                condition
            ),)
        ),
        TokenAdmissionApply::Applied { accepted: None }
    ));
    assert_eq!(round.finish().unwrap().deferred(), 1);

    let audit_only = AdmissionWakeEpochs::new(coordinator, 1, 0, 0);
    let mut observations = Vec::new();
    let mut unchanged = scheduler
        .begin_token_admission(
            1,
            Some(AdmissionWakeSnapshot::new(audit_only, &availability)),
        )
        .unwrap();
    assert!(unchanged
        .next(&mut |observation| observations.push(observation))
        .unwrap()
        .is_none());
    let receipt = unchanged.finish().unwrap();
    assert_eq!(receipt.probed(), 0);
    assert_eq!(receipt.skipped_unchanged(), 1);
    assert!(
        matches!(&observations[..], [ExecutorAdmissionQueueObservation::SkippedUnchanged {
        request_id, ticket, current, ..
    }] if request_id == &id && *ticket == key.ordinal() && *current == audit_only)
    );

    let changed =
        [
            CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 5)
                .unwrap(),
        ];
    let mut retry = scheduler
        .begin_token_admission(
            1,
            Some(AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(coordinator, 2, 0, 0),
                &changed,
            )),
        )
        .unwrap();
    let candidate = retry.next(&mut |_| {}).unwrap().unwrap();
    assert_eq!(candidate.key(), &key);
    let owner = accepted(retry.apply(candidate, admitted(&id)));
    assert_eq!(retry.finish().unwrap().admitted(), 1);
    assert!(scheduler.cancel_token_admission(&owner).unwrap());
}

#[tokio::test]
async fn token_admission_round_coexists_with_capacity_requeue_and_logical_completion() {
    // Exercise the actual queue transitions concurrently with admission. The
    // timeout is a deadlock bound, not a latency or throughput assertion.
    for transition in ["prefill_requeue", "decode_requeue", "prefill_complete"] {
        let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
        let active = create_test_request(Priority::Normal);
        let active_id = active.id.clone();
        scheduler.submit(active).await.unwrap();
        scheduler
            .prepare_token_admission_observed(
                1,
                None,
                &mut |request| admitted(&request.id),
                &mut |_| {},
            )
            .unwrap();
        if transition == "decode_requeue" {
            scheduler.mark_prefill_complete(&active_id, 1);
            assert_eq!(
                scheduler.trace_phase(&active_id),
                Some(RequestPhase::Decoding)
            );
        }
        let pending = create_test_request(Priority::Normal);
        let pending_id = pending.id.clone();
        scheduler.submit(pending).await.unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let (finished, completed) = std::sync::mpsc::channel();
        let admit_scheduler = Arc::clone(&scheduler);
        let admit_barrier = Arc::clone(&barrier);
        let admit_finished = finished.clone();
        let admitted_id = pending_id.clone();
        let admission = std::thread::spawn(move || {
            let mut round = admit_scheduler.begin_token_admission(1, None).unwrap();
            let candidate = round.next(&mut |_| {}).unwrap().unwrap();
            assert_eq!(candidate.request().id, admitted_id);
            // Select before racing the queue transitions, otherwise a request
            // requeued before this round starts is legitimately selected first.
            admit_barrier.wait();
            let owner = accepted(round.apply(candidate, admitted(&admitted_id)));
            round.finish().unwrap();
            admit_finished.send(()).unwrap();
            owner
        });
        let release_scheduler = Arc::clone(&scheduler);
        let closing_id = active_id.clone();
        let release = std::thread::spawn(move || {
            barrier.wait();
            match transition {
                "prefill_requeue" => {
                    assert!(release_scheduler.defer_prefill_to_waiting(&closing_id))
                }
                "decode_requeue" => {
                    assert!(release_scheduler.defer_decode_to_waiting_for_capacity(&closing_id, 1))
                }
                "prefill_complete" => {
                    let response = super::capacity_release::response(&closing_id);
                    release_scheduler
                        .complete_request(closing_id, &response)
                        .unwrap();
                }
                _ => unreachable!(),
            }
            finished.send(()).unwrap();
        });
        for _ in 0..2 {
            completed
                .recv_timeout(Duration::from_secs(5))
                .expect("queue transition deadlocked");
        }
        let owner = admission.join().unwrap();
        release.join().unwrap();
        assert_eq!(
            scheduler.trace_phase(&pending_id),
            Some(RequestPhase::Prefilling)
        );
        assert_eq!(
            scheduler.trace_phase(&active_id),
            if transition == "prefill_complete" {
                None
            } else {
                Some(RequestPhase::Waiting)
            }
        );
        assert_eq!(scheduler.trace_snapshot().capacity_release_epoch, 0);
        assert!(scheduler.cancel_token_admission(&owner).unwrap());
        if transition != "prefill_complete" {
            assert!(scheduler.cancel_request(active_id).unwrap());
        }
    }
}
