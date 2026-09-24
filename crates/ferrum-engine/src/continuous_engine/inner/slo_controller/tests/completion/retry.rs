//! Retry behavior through real capture/publication and durable dispatch entry.
use super::*;

async fn admitted_without_deadline(
    engine: &ContinuousBatchEngine,
) -> (RequestId, CreditedOutputSession) {
    let (id, session) = request(engine, 1, 2, None).await;
    engine.inner.sequences.write().get_mut(&id).unwrap().slo = None;
    admit(engine, 1).await;
    assert!(engine.inner.next_slo_violation_wake().is_none());
    // Request admission already woke its original consumer. The tests below
    // must progress without another request, output drain, or SLO deadline.
    while engine.inner.work_notify.notified().now_or_never().is_some() {}
    (id, session)
}

async fn retry_timer(engine: &ContinuousBatchEngine) {
    let backoff = Duration::from_millis(
        engine
            .inner
            .config
            .scheduler
            .slo
            .planner
            .retry_backoff_ms
            .get(),
    );
    let mut wake = Box::pin(engine.inner.wait_for_slo_controller_retry());
    assert!(wake.as_mut().now_or_never().is_none());
    tokio::time::advance(backoff / 2).await;
    assert!(wake.as_mut().now_or_never().is_none());
    // Tokio timers have millisecond resolution; this allowance tests their
    // documented granularity rather than a wall-clock performance threshold.
    tokio::time::advance(backoff - backoff / 2 + Duration::from_millis(1)).await;
    assert!(
        wake.as_mut().now_or_never().is_some(),
        "a transient controller failure did not arm an independent retry"
    );
}

async fn completes_after_retry(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    id: &RequestId,
) {
    assert!(engine.inner.work_notify.notified().now_or_never().is_none());
    retry_timer(engine).await;
    assert!(matches!(
        bounded(engine.inner.run_iteration()).await.unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[id].generated_tokens.len(), 1);
    assert_eq!(scheduled(engine), 0);
}

#[tokio::test(start_paused = true)]
async fn completion_publication_busy_retries_without_external_work() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, session) = admitted_without_deadline(&engine).await;
    let (start_tx, start_rx) = std::sync::mpsc::sync_channel(0);
    let (held_tx, held_rx) = std::sync::mpsc::sync_channel(0);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
    let inner = Arc::clone(&engine.inner);
    let holder = std::thread::spawn(move || {
        start_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        let _held = inner.dynamic_admission_availability.lock();
        held_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    *executor.after_resource_revalidation.lock() = Some(Box::new(move || {
        start_tx.send(()).unwrap();
        held_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    }));
    // Capture and real resource comparison succeed first. A different owner
    // then holds the actual publication mutex; no Busy result is fabricated.
    let attempted = engine
        .inner
        .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1));
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    assert!(matches!(attempted.unwrap(), SloIterationPlan::Idle));
    assert!(executor.after_resource_revalidation.lock().is_none());
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    assert!(engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .grant
        .is_none());
    completes_after_retry(&engine, &executor, &id).await;
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn completion_publication_epoch_change_recaptures_after_timer() {
    let (engine, scheduler, executor) = completion_fixture(1).await;
    let (id, session) = admitted_without_deadline(&engine).await;
    let before = scheduler.trace_snapshot().capacity_release_epoch;
    let changed = Arc::clone(&scheduler);
    *executor.after_resource_revalidation.lock() = Some(Box::new(move || {
        // The scheduler's real release protocol invalidates its complete queue
        // seal. It neither edits the selected request nor sends a work notify.
        changed.record_external_capacity_release();
    }));
    assert!(matches!(
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert_eq!(
        scheduler.trace_snapshot().capacity_release_epoch,
        before + 1
    );
    assert!(executor.after_resource_revalidation.lock().is_none());
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    assert_eq!(
        engine.inner.sequences.read()[&id].prefill_tokens_processed,
        0
    );
    completes_after_retry(&engine, &executor, &id).await;
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn completion_durable_zero_submit_notification_respects_retry_backoff() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, session) = admitted_without_deadline(&engine).await;
    executor.replan_before_encode.store(true, Ordering::Release);
    assert!(matches!(
        bounded(engine.inner.run_iteration()).await.unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    assert!(engine
        .inner
        .slo_controller
        .lock()
        .pending_execution
        .is_none());
    assert!(engine.inner.work_notify.notified().now_or_never().is_some());
    let observations = engine.inner.slo_controller.lock().observations;

    // The durable task and publication rollback really did wake the loop.
    // Even if the backend is now ready, that self-notification must not launch
    // another full attempt before the configured retry deadline.
    executor
        .replan_before_encode
        .store(false, Ordering::Release);
    assert!(matches!(
        bounded(engine.inner.run_iteration()).await.unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(
        engine.inner.slo_controller.lock().observations,
        observations
    );
    assert!(engine.inner.sequences.read()[&id]
        .generated_tokens
        .is_empty());
    retry_timer(&engine).await;
    assert!(matches!(
        bounded(engine.inner.run_iteration()).await.unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 2);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    assert_eq!(scheduled(&engine), 0);
    cleanup(engine, session).await;
}
