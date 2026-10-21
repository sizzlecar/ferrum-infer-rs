//! Product propagation/wake test. The slot diagnostic is injected by the test
//! envelope; cache storage and exact terminal identity remain the real CPU
//! model's. Model tests separately exercise the actual physical slot guard.
use super::*;

#[tokio::test]
async fn physical_sequence_slots_product_waits_for_original_end_not_logical_completion() {
    let mut fixture = fixture().await;
    *fixture.executor.slot_capacity.lock() = Some(1);
    let (admission, identity) = start_round(&mut fixture).await;
    fixture.resume.take().unwrap().send(()).unwrap();
    assert_eq!(admission.await.unwrap().unwrap().0, 1);
    let inner = Arc::clone(&fixture.engine.active_inner().unwrap());
    let mut waiter = InferenceRequest::new("wait for actual original owner", "tiny-llama-test");
    waiter.sampling_params.max_tokens = 4;
    let waiting_response = install_outer(
        &fixture.engine,
        waiter.clone(),
        vec![TokenId(3), TokenId(4)],
    )
    .await;
    assert_eq!(
        inner
            .prepare_dynamic_admission_round(1, None)
            .await
            .unwrap()
            .0,
        0
    );
    assert!(inner.scheduler.waiting_for_confirmed_capacity_release());
    assert_eq!(
        fixture.executor.admission_attempts.load(Ordering::SeqCst),
        2
    );
    let epoch = inner.scheduler.trace_snapshot().capacity_release_epoch;

    let (entered, started) = oneshot::channel();
    let (release, resume) = oneshot::channel();
    *fixture.executor.close_gate.lock() = Some((entered, resume));
    let completion_inner = Arc::clone(&inner);
    let first_id = fixture.request.id.clone();
    let completion = tokio::spawn(async move {
        completion_inner
            .complete_request_for_test(&first_id, FinishReason::Length)
            .await
    });
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), started)
            .await
            .unwrap()
            .unwrap(),
        identity
    );
    assert_eq!(inner.scheduler.active_count(), 0);
    assert!(identity.observed_end().is_none());
    assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
    assert_eq!(
        inner.scheduler.trace_snapshot().capacity_release_epoch,
        epoch
    );
    assert_eq!(
        inner
            .prepare_dynamic_admission_round(1, None)
            .await
            .unwrap()
            .0,
        0
    );
    assert_eq!(
        fixture.executor.admission_attempts.load(Ordering::SeqCst),
        2
    );

    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(5), completion)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    inner.execution_closes.join_observers().await;
    inner.drain_execution_closes().unwrap();
    assert!(identity.observed_end().is_some());
    assert_eq!(
        inner.scheduler.trace_snapshot().capacity_release_epoch,
        epoch + 1
    );
    assert_eq!(
        inner
            .prepare_dynamic_admission_round(1, None)
            .await
            .unwrap()
            .0,
        1
    );
    assert_eq!(
        fixture.executor.admission_attempts.load(Ordering::SeqCst),
        3
    );
    assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
    inner
        .complete_request_for_test(&waiter.id, FinishReason::Length)
        .await
        .unwrap();
    inner.execution_closes.join_observers().await;
    inner.drain_execution_closes().unwrap();
    assert!(fixture.model.lock().model.kv_caches.is_empty());
    assert!(fixture.response.await.unwrap().is_ok());
    assert!(waiting_response.await.unwrap().is_ok());
}
