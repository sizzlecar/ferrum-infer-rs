//! Integration tests for ContinuousBatchEngine using mock components.
//! Runs on any platform — no GPU required.

use ferrum_engine::{
    ContinuousBatchEngine, InferenceEngineInterface, Scheduler,
};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    MockKvCacheManager, MockModelExecutor, MockSampler, MockTensorFactory, MockTokenizer,
};
use ferrum_types::{InferenceRequest, SchedulerConfig};
use std::sync::Arc;
use std::time::Duration;

const VOCAB_SIZE: usize = 1000;

fn make_engine() -> ContinuousBatchEngine {
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let executor = Arc::new(MockModelExecutor::instant(VOCAB_SIZE));
    let tensor_factory = Arc::new(MockTensorFactory);

    ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    )
}

fn make_request(prompt: &str) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "mock-model");
    req.sampling_params.max_tokens = 5;
    req
}

#[tokio::test]
async fn single_request_completes() {
    let engine = make_engine();
    let request = make_request("Hello world");
    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.finish_reason, ferrum_types::FinishReason::Length);
    assert!(!response.text.is_empty());
}

#[tokio::test]
async fn multiple_requests_complete_sequentially() {
    let engine = make_engine();

    for i in 0..5 {
        let request = make_request(&format!("Request number {}", i));
        let response = engine.infer(request).await.unwrap();
        assert_eq!(response.finish_reason, ferrum_types::FinishReason::Length);
    }
}

#[tokio::test]
async fn streaming_produces_chunks() {
    use futures::StreamExt;

    let engine = make_engine();
    let request = make_request("Stream test");

    let stream = engine.infer_stream(request).await.unwrap();
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Should have at least one content chunk + final chunk
    assert!(!chunks.is_empty(), "Stream should produce chunks");

    // Last chunk should have finish_reason
    let last = chunks.last().unwrap();
    assert!(
        last.finish_reason.is_some(),
        "Final chunk should have finish_reason"
    );
}

#[tokio::test]
async fn concurrent_submit_tracked_by_scheduler() {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));

    // Submit 3 requests
    for i in 0..3 {
        let req = make_request(&format!("Request {}", i));
        scheduler.submit(req).await.unwrap();
    }

    let metrics = scheduler.metrics();
    assert_eq!(metrics.waiting_requests, 3);
}

#[tokio::test]
async fn kv_cache_allocated_and_deallocated() {
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let executor = Arc::new(MockModelExecutor::instant(VOCAB_SIZE));
    let tensor_factory = Arc::new(MockTensorFactory);

    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
    );

    // Before: no active caches
    assert_eq!(kv_cache.active_count(), 0);

    let request = make_request("KV cache test");
    let _response = engine.infer(request).await.unwrap();

    // After completion: cache should be deallocated
    assert_eq!(kv_cache.active_count(), 0);
}

#[tokio::test]
async fn mock_executor_tracks_operations() {
    let executor = Arc::new(MockModelExecutor::instant(VOCAB_SIZE));
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let tensor_factory = Arc::new(MockTensorFactory);

    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor.clone(),
        tensor_factory,
    );

    assert_eq!(executor.prefill_count(), 0);
    assert_eq!(executor.decode_count(), 0);

    let request = make_request("Track ops");
    let _response = engine.infer(request).await.unwrap();

    // Should have done 1 prefill and max_tokens-1 decodes (first token comes from prefill)
    assert_eq!(executor.prefill_count(), 1);
    assert_eq!(executor.decode_count(), 4); // max_tokens=5, first from prefill
}

#[tokio::test]
async fn engine_with_latency_still_completes() {
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let executor = Arc::new(MockModelExecutor::new(
        VOCAB_SIZE,
        Duration::from_millis(5),
        Duration::from_millis(2),
    ));
    let tensor_factory = Arc::new(MockTensorFactory);

    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    );

    let request = make_request("Latency test");
    let response = engine.infer(request).await.unwrap();
    assert_eq!(response.finish_reason, ferrum_types::FinishReason::Length);
    // With 5ms prefill + 4*2ms decode = ~13ms minimum
    assert!(response.latency_ms >= 10);
}
