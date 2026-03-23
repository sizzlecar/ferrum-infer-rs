//! Integration tests for ContinuousBatchEngine using mock components.
//! Runs on any platform — no GPU required.

use ferrum_engine::{ContinuousBatchEngine, InferenceEngineInterface, Scheduler};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    MockKvCacheManager, MockModelExecutor, MockSampler, MockTensorFactory, MockTokenizer,
};
use ferrum_types::{InferenceRequest, InferenceResponse, SchedulerConfig};
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
    req.sampling_params.temperature = 0.0; // Greedy for deterministic tests
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

// ────────────────────────────────────────────────────────────────────────────
// Continuous batching tests (concurrent requests)
// ────────────────────────────────────────────────────────────────────────────

fn make_engine_shared() -> Arc<ContinuousBatchEngine> {
    Arc::new(make_engine())
}

#[tokio::test]
async fn concurrent_requests_all_complete() {
    let engine = make_engine_shared();

    let mut handles = Vec::new();
    for i in 0..5 {
        let e = engine.clone();
        handles.push(tokio::spawn(async move {
            let req = make_request(&format!("Concurrent request {}", i));
            e.infer(req).await
        }));
    }

    let results: Vec<InferenceResponse> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap().unwrap())
        .collect();

    assert_eq!(results.len(), 5);
    for resp in &results {
        assert_eq!(resp.finish_reason, ferrum_types::FinishReason::Length);
        assert!(!resp.tokens.is_empty());
    }
}

#[tokio::test]
async fn concurrent_requests_deallocate_kv() {
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let executor = Arc::new(MockModelExecutor::instant(VOCAB_SIZE));
    let tensor_factory = Arc::new(MockTensorFactory);

    let engine = Arc::new(ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache.clone(),
        executor,
        tensor_factory,
    ));

    assert_eq!(kv_cache.active_count(), 0);

    let mut handles = Vec::new();
    for i in 0..3 {
        let e = engine.clone();
        handles.push(tokio::spawn(async move {
            let req = make_request(&format!("KV test {}", i));
            e.infer(req).await.unwrap()
        }));
    }

    futures::future::join_all(handles).await;

    // All KV caches should be deallocated after all requests complete
    assert_eq!(kv_cache.active_count(), 0, "All KV caches should be freed");
}

#[tokio::test]
async fn concurrent_streams_all_complete() {
    use futures::StreamExt;

    let engine = make_engine_shared();

    let mut handles = Vec::new();
    for i in 0..3 {
        let e = engine.clone();
        handles.push(tokio::spawn(async move {
            let req = make_request(&format!("Stream {}", i));
            let stream = e.infer_stream(req).await.unwrap();
            let chunks: Vec<_> = stream
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            chunks
        }));
    }

    let all_chunks: Vec<Vec<_>> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(all_chunks.len(), 3);
    for chunks in &all_chunks {
        assert!(!chunks.is_empty(), "Each stream should produce chunks");
        let last = chunks.last().unwrap();
        assert!(
            last.finish_reason.is_some(),
            "Final chunk should have finish_reason"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Prefix cache tests
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn prefix_cache_avoids_second_prefill() {
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

    // First request — cache miss, runs real prefill
    let req1 = make_request("Identical prompt for prefix cache");
    let resp1 = engine.infer(req1).await.unwrap();
    assert_eq!(executor.prefill_count(), 1);

    // Second request with same prompt — should hit prefix cache, no executor prefill
    let req2 = make_request("Identical prompt for prefix cache");
    let resp2 = engine.infer(req2).await.unwrap();
    assert_eq!(
        executor.prefill_count(),
        1,
        "Prefix cache should skip second prefill"
    );

    // Both should produce the same output (same prompt + deterministic sampling)
    assert_eq!(resp1.tokens, resp2.tokens);
    assert!(!resp1.tokens.is_empty());
}

#[tokio::test]
async fn json_mode_biases_first_token() {
    use ferrum_types::ResponseFormat;

    let engine = make_engine();

    // Without JSON mode: MockExecutor biases token 42, so greedy picks 42.
    let mut plain_req = make_request("Hello");
    plain_req.sampling_params.max_tokens = 1;
    let plain_resp = engine.infer(plain_req).await.unwrap();
    assert_eq!(
        plain_resp.tokens[0].get(),
        42,
        "Without JSON mode, greedy should pick token 42"
    );

    // With JSON mode: structural_bias (5.0) on token 123 (`{`) beats 1.0 on token 42.
    let mut json_req = make_request("Hello");
    json_req.sampling_params.max_tokens = 1;
    json_req.sampling_params.response_format = ResponseFormat::JsonObject;
    let json_resp = engine.infer(json_req).await.unwrap();
    assert_eq!(
        json_resp.tokens[0].get(),
        123,
        "JSON mode should bias first token to `{{` (token 123)"
    );
}
