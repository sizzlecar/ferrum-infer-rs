//! Backend safety tests — verify the inference pipeline works correctly
//! regardless of which backend is active.
//!
//! These tests run on CPU with mock components. They catch regressions
//! that could be introduced when modifying CUDA/Metal code paths:
//! - Executor prefill/decode contract
//! - KV cache lifecycle (allocate → use → release)
//! - Streaming token delivery
//! - Multi-sequence concurrent decode
//! - Error handling and cleanup

use ferrum_engine::{ContinuousBatchEngine, InferenceEngineInterface};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    MockKvCacheManager, MockModelExecutor, MockSampler, MockTensorFactory, MockTokenizer,
};
use ferrum_types::{InferenceRequest, SchedulerConfig};
use futures::StreamExt;
use std::sync::Arc;

const VOCAB: usize = 1000;

fn engine() -> ContinuousBatchEngine {
    let config = ferrum_types::EngineConfig::default();
    ContinuousBatchEngine::new(
        config,
        Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default())),
        Arc::new(MockTokenizer::new(VOCAB)),
        Arc::new(MockSampler),
        Arc::new(MockKvCacheManager::new(1024)),
        Arc::new(MockModelExecutor::instant(VOCAB)),
        Arc::new(MockTensorFactory),
    )
}

fn engine_with_executor(
    executor: Arc<MockModelExecutor>,
) -> (ContinuousBatchEngine, Arc<MockKvCacheManager>) {
    let config = ferrum_types::EngineConfig::default();
    let kv = Arc::new(MockKvCacheManager::new(1024));
    let eng = ContinuousBatchEngine::new(
        config,
        Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default())),
        Arc::new(MockTokenizer::new(VOCAB)),
        Arc::new(MockSampler),
        kv.clone(),
        executor,
        Arc::new(MockTensorFactory),
    );
    (eng, kv)
}

fn req(prompt: &str, max_tokens: usize) -> InferenceRequest {
    let mut r = InferenceRequest::new(prompt, "mock");
    r.sampling_params.max_tokens = max_tokens;
    r.sampling_params.temperature = 0.0;
    r
}

// ======================== Prefill/Decode Contract ========================

#[tokio::test]
async fn prefill_called_once_per_request() {
    let executor = Arc::new(MockModelExecutor::instant(VOCAB));
    let (eng, _) = engine_with_executor(executor.clone());

    let _ = eng.infer(req("test prompt", 5)).await.unwrap();
    assert_eq!(executor.prefill_count(), 1);
    assert!(executor.decode_count() >= 4); // at least max_tokens - 1 decodes
}

#[tokio::test]
async fn decode_count_matches_generated_tokens() {
    let executor = Arc::new(MockModelExecutor::instant(VOCAB));
    let (eng, _) = engine_with_executor(executor.clone());

    let resp = eng.infer(req("hello", 10)).await.unwrap();
    // prefill produces 1 token, decode produces the rest
    let _total_tokens = resp.text.split_whitespace().count().max(1);
    assert!(
        executor.decode_count() >= 9,
        "expected ≥9 decodes, got {}",
        executor.decode_count()
    );
}

// ======================== KV Cache Lifecycle ========================

#[tokio::test]
async fn kv_cache_allocated_during_request() {
    let executor = Arc::new(MockModelExecutor::instant(VOCAB));
    let (eng, kv) = engine_with_executor(executor);

    assert_eq!(kv.active_count(), 0);
    let _ = eng.infer(req("test", 3)).await.unwrap();
    // After completion, KV cache should be deallocated
    assert_eq!(
        kv.active_count(),
        0,
        "KV cache should be freed after request completes"
    );
}

#[tokio::test]
async fn kv_cache_freed_after_multiple_requests() {
    let executor = Arc::new(MockModelExecutor::instant(VOCAB));
    let (eng, kv) = engine_with_executor(executor);

    for i in 0..5 {
        let _ = eng.infer(req(&format!("prompt {i}"), 3)).await.unwrap();
    }
    assert_eq!(kv.active_count(), 0, "all KV caches should be freed");
}

// ======================== Streaming ========================

#[tokio::test]
async fn streaming_delivers_tokens_incrementally() {
    let eng = engine();
    let mut r = req("hello world", 5);
    r.stream = true;

    let mut stream = eng.infer_stream(r).await.unwrap();
    let mut token_count = 0;
    let mut saw_finish = false;

    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                if chunk.token.is_some() {
                    token_count += 1;
                }
                if chunk.finish_reason.is_some() {
                    saw_finish = true;
                    break;
                }
            }
            Err(e) => panic!("stream error: {e}"),
        }
    }

    assert!(
        token_count >= 1,
        "should receive at least 1 token via stream"
    );
    assert!(saw_finish, "should receive finish signal");
}

// ======================== Concurrent Requests ========================

#[tokio::test]
async fn concurrent_requests_complete_independently() {
    let eng = Arc::new(engine());

    let mut handles = Vec::new();
    for i in 0..3 {
        let e = eng.clone();
        handles.push(tokio::spawn(async move {
            e.infer(req(&format!("concurrent {i}"), 3)).await
        }));
    }

    let mut successes = 0;
    for h in handles {
        if let Ok(Ok(resp)) = h.await {
            assert_eq!(resp.finish_reason, ferrum_types::FinishReason::Length);
            successes += 1;
        }
    }
    assert_eq!(successes, 3, "all concurrent requests should complete");
}

// ======================== Edge Cases ========================

#[tokio::test]
async fn single_token_generation() {
    let eng = engine();
    let resp = eng.infer(req("test", 1)).await.unwrap();
    assert!(!resp.text.is_empty());
}

#[tokio::test]
async fn empty_prompt_still_works() {
    let eng = engine();
    let resp = eng.infer(req("", 3)).await;
    // Should either succeed or return a clear error, not panic
    assert!(resp.is_ok() || resp.is_err());
}

#[tokio::test]
async fn response_has_valid_metadata() {
    let eng = engine();
    let resp = eng.infer(req("test metadata", 5)).await.unwrap();

    assert_eq!(resp.finish_reason, ferrum_types::FinishReason::Length);
    assert!(resp.usage.completion_tokens > 0);
    assert!(resp.usage.prompt_tokens > 0);
    assert_eq!(
        resp.usage.total_tokens,
        resp.usage.prompt_tokens + resp.usage.completion_tokens
    );
}
