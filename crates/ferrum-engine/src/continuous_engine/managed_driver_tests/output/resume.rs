//! Resume the original bounded stream after a peer has completed real CPU work.
use super::*;

#[tokio::test]
async fn output_slow_stream_resumes_exact_text_tokens_and_terminal_once() {
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
    {
        let original = fixture.engine.active_inner().unwrap();
        let sequences = &original.sequences;
        let slow_handle = sequences.get(&slow_id).unwrap();
        let slow_guard = slow_handle.lock();
        let slow = slow_guard.as_ref().unwrap();
        assert_eq!(slow.generated_tokens.len(), 2);
        assert_eq!(slow.generated_tokens, healthy.response().tokens[..2]);
    }
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 6);
    let chunks = tokio::time::timeout(Duration::from_secs(3), stream.collect::<Vec<_>>())
        .await
        .expect("consuming the original stream must wake its exact paused request")
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let text: String = chunks
        .iter()
        .map(|chunk| chunk.chunk().text.as_str())
        .collect();
    assert_eq!(text, healthy.response().text);
    let tokens: Vec<_> = chunks
        .iter()
        .filter_map(|chunk| chunk.chunk().token)
        .collect();
    let visible: Vec<_> = healthy
        .response()
        .tokens
        .iter()
        .copied()
        .filter(|token| {
            !fixture
                .engine
                .active_inner()
                .unwrap()
                .tokenizer
                .is_special_token(*token)
        })
        .collect();
    assert_eq!(tokens, visible);
    let terminals: Vec<_> = chunks
        .iter()
        .map(OutputChunk::chunk)
        .filter(|chunk| chunk.finish_reason.is_some())
        .collect();
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0].finish_reason, Some(FinishReason::Length));
    assert!(terminals[0].text.is_empty());
    assert_eq!(
        terminals[0].usage.as_ref().unwrap().prompt_tokens,
        healthy.response().usage.prompt_tokens
    );
    assert_eq!(terminals[0].usage.as_ref().unwrap().completion_tokens, 4);
    assert_eq!(
        chunks.last().unwrap().chunk().finish_reason,
        Some(FinishReason::Length)
    );
    drop(healthy);
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 8);
    assert_eq!(fixture.executor.maximum_in_flight.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 2);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert!(fixture.original_inner.upgrade().is_none());
    assert_eq!(fixture.output_budget.usage().sessions, 1);
    drop(chunks);
    assert_eq!(fixture.output_budget.usage().sessions, 0);
    assert_eq!(fixture.output_budget.usage().queued_bytes, 0);
}
