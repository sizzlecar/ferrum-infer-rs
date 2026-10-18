//! Real CPU model and independent full-logits/KV oracle from the managed fixture.
use super::*;

#[tokio::test]
async fn output_slow_stream_pauses_only_its_request_and_shutdown_joins_delivery() {
    let mut fixture = fixture::Fixture::with_config(|config| {
        config.output.engine_queue_items = 1;
        config.scheduler.max_running_requests = 2;
        config.batching.max_batch_size = 2;
    });
    let request = fixture.request();
    let slow_id = request.id.clone();
    let stream = fixture.engine.infer_stream(request).await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    let mut healthy = Box::pin(fixture.engine.infer(fixture.request()));
    pending_once(healthy.as_mut()).await;
    fixture.release.send(()).unwrap();
    let healthy = tokio::time::timeout(Duration::from_secs(3), healthy)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(healthy.tokens.len(), 4);
    {
        let sequences = fixture.engine.inner.sequences.read();
        let slow = sequences
            .get(&slow_id)
            .expect("backpressure retains exact request");
        assert_eq!(
            slow.generated_tokens.len(),
            2,
            "one queued and one pending delta"
        );
        assert!(slow
            .execution_frontier
            .as_ref()
            .unwrap()
            .sequence()
            .observed_end()
            .is_none());
    }
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 6);
    assert_eq!(fixture.executor.maximum_in_flight.load(Ordering::SeqCst), 1);
    tokio::time::timeout(Duration::from_secs(3), fixture.engine.shutdown())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 2);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(fixture.engine.inner.output_budget.usage().sessions, 1);
    let chunks = stream.collect::<Vec<_>>().await;
    assert_eq!(
        chunks.iter().filter(|item| item.is_err()).count(),
        1,
        "shutdown preserves one bounded terminal after queued text"
    );
    assert!(chunks.last().unwrap().is_err());
    assert!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.as_ref().ok())
            .all(|chunk| chunk.chunk().finish_reason != Some(FinishReason::Length)),
        "shutdown cannot turn interrupted delivery into successful DONE"
    );
    drop(chunks);
    assert_eq!(fixture.engine.inner.output_budget.usage().sessions, 0);
    assert_eq!(fixture.engine.inner.output_budget.usage().queued_bytes, 0);
}

#[tokio::test]
async fn output_completed_unread_streams_retain_session_admission_until_drop() {
    let mut fixture = fixture::Fixture::with_config(|config| {
        config.output.max_sessions = 2;
        config.output.engine_queue_items = 4;
        config.scheduler.max_running_requests = 1;
        config.batching.max_batch_size = 1;
    });
    let mut first_request = fixture.request();
    let mut second_request = fixture.request();
    let mut third_request = fixture.request();
    let rejected_request = fixture.request();
    first_request.sampling_params.max_tokens = 1;
    second_request.sampling_params.max_tokens = 1;
    third_request.sampling_params.max_tokens = 1;
    let first = fixture.engine.infer_stream(first_request).await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    fixture.release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(3), fixture.executor.wait_for_closes(1))
        .await
        .unwrap();
    fixture.engine.inner.execution_closes.join_observers().await;
    fixture.engine.inner.output_tasks.join_current().await;
    fixture.engine.inner.drain_execution_closes().unwrap();
    assert!(fixture.engine.inner.sequences.read().is_empty());
    assert_eq!(fixture.engine.inner.output_budget.usage().sessions, 1);

    let second = fixture.engine.infer_stream(second_request).await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), fixture.executor.wait_for_closes(2))
        .await
        .unwrap();
    fixture.engine.inner.execution_closes.join_observers().await;
    fixture.engine.inner.output_tasks.join_current().await;
    fixture.engine.inner.drain_execution_closes().unwrap();
    assert_eq!(fixture.engine.inner.output_budget.usage().sessions, 2);
    let opened = fixture::slot_events(
        &fixture.engine,
        &fixture.trace.path().join("trace.jsonl"),
        ResourceAction::RequestOpen,
    );
    let error = fixture
        .engine
        .infer_stream(rejected_request)
        .await
        .err()
        .expect("unread streams exhaust session budget");
    assert!(matches!(error, FerrumError::ResourceExhausted { .. }));
    assert_eq!(fixture.executor.admissions.load(Ordering::SeqCst), 2);
    assert_eq!(
        fixture::slot_events(
            &fixture.engine,
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::RequestOpen
        ),
        opened
    );
    drop(first);
    assert_eq!(fixture.engine.inner.output_budget.usage().sessions, 1);
    let third = fixture.engine.infer_stream(third_request).await.unwrap();
    let chunks = tokio::time::timeout(Duration::from_secs(3), third.collect::<Vec<_>>())
        .await
        .unwrap();
    assert_eq!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.as_ref().ok())
            .filter(|chunk| chunk.chunk().token.is_some())
            .count(),
        1
    );
    assert_eq!(
        chunks
            .last()
            .unwrap()
            .as_ref()
            .unwrap()
            .chunk()
            .finish_reason,
        Some(FinishReason::Length)
    );
    drop(chunks);
    drop(second);
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 3);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert_eq!(fixture.engine.inner.output_budget.usage().sessions, 0);
    assert_eq!(fixture.engine.inner.output_budget.usage().queued_bytes, 0);
}

#[path = "output/resume.rs"]
mod resume;
