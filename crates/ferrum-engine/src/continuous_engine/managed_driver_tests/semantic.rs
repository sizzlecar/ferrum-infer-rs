//! Numerical work remains real; only the actual shared content credit is limited.
use super::*;

#[tokio::test]
async fn semantic_output_owner_outlives_completed_engine_until_last_clone_drop() {
    let mut fixture = fixture::Fixture::new();
    let budget = fixture.engine.inner.output_budget.clone();
    let mut request = fixture.request();
    request.evidence_request.capture_prompt_token_ids = true;
    let request_id = request.id.clone();
    let mut inference = Box::pin(fixture.engine.infer(request));
    pending_once(inference.as_mut()).await;
    tokio::time::timeout(Duration::from_secs(3), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    let prompt = fixture
        .engine
        .inner
        .sequences
        .get(&request_id)
        .unwrap()
        .lock()
        .as_ref()
        .unwrap()
        .input_tokens
        .clone();
    assert!(!prompt.is_empty());
    fixture.release.send(()).unwrap();
    let output = tokio::time::timeout(Duration::from_secs(3), inference)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.response().tokens.len(), 4);
    let evidence = output.response().execution_evidence.as_ref().unwrap();
    assert_eq!(evidence.prompt_token_ids, prompt);
    assert_eq!(output.response().usage.prompt_tokens, prompt.len());
    assert_eq!(
        output.response().usage.completion_tokens,
        output.response().tokens.len()
    );
    assert_eq!(evidence.output_token_ids, output.response().tokens);
    let cost = semantic_cost(output.response()).unwrap();
    fixture.engine.inner.output_tasks.join_current().await;
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 4);
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 1);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert_eq!(budget.usage().semantic_bytes, cost);
    assert_eq!(budget.usage().sessions, 1);
    let clone = output.clone();
    let inner = Arc::downgrade(&fixture.engine.inner);
    drop(output);
    drop(fixture.engine);
    assert!(inner.upgrade().is_none());
    assert_eq!(budget.usage().semantic_bytes, cost);
    assert_eq!(clone.response().tokens.len(), 4);
    drop(clone);
    assert_eq!(budget.usage().semantic_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}

#[tokio::test]
async fn semantic_quota_failure_keeps_real_progress_and_healthy_peer_without_replay() {
    let mut fixture = fixture::Fixture::with_config(|config| {
        config.scheduler.max_running_requests = 2;
        config.batching.max_batch_size = 2;
    });
    let request = fixture.request();
    let id = request.id.clone();
    let mut limited = Box::pin(fixture.engine.infer(request));
    pending_once(limited.as_mut()).await;
    tokio::time::timeout(Duration::from_secs(3), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    let session = fixture
        .engine
        .inner
        .sequences
        .get(&id)
        .unwrap()
        .lock()
        .as_ref()
        .unwrap()
        .output_semantic
        .output_session()
        .clone();
    // The original real model prefill has completed, but its product has not
    // reached Engine sampling. Three bytes cannot commit even one TokenId.
    let held = session
        .try_reserve_semantic(
            session.config().semantic_bytes_per_session - session.semantic_usage() - 3,
        )
        .unwrap();
    let mut healthy = Box::pin(fixture.engine.infer(fixture.request()));
    pending_once(healthy.as_mut()).await;
    fixture.release.send(()).unwrap();
    let (failed, healthy) = tokio::time::timeout(Duration::from_secs(3), async {
        tokio::join!(limited, healthy)
    })
    .await
    .unwrap();
    assert!(matches!(
        failed.unwrap_err(),
        FerrumError::ResourceExhausted { .. }
    ));
    let healthy = healthy.unwrap();
    assert_eq!(healthy.response().tokens.len(), 4);
    fixture.engine.inner.execution_closes.join_observers().await;
    fixture.engine.inner.output_tasks.join_current().await;
    fixture.engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        fixture.executor.checked_outputs.load(Ordering::SeqCst),
        5,
        "one failed-product prefill plus four independently verified peer outputs, no replay"
    );
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 2);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert!(fixture.engine.inner.sequences.is_empty());
    drop(held);
    drop(session);
    drop(healthy);
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.engine.inner.output_budget.usage().semantic_bytes, 0);
    assert_eq!(fixture.engine.inner.output_budget.usage().sessions, 0);
}
