//! Integration tests for ContinuousBatchEngine with PagedAttentionExecutor.
//! Verifies end-to-end paged KV cache usage: block allocation, K/V storage,
//! cross-block attention, and deallocation.

use ferrum_engine::{ContinuousBatchEngine, InferenceEngineInterface, KvCacheManager};
use ferrum_kv::managers::paged::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    MockSampler, MockTensorFactory, MockTokenizer, PagedAttentionExecutor, PagedExecutorConfig,
};
use ferrum_types::{InferenceRequest, InferenceResponse, SchedulerConfig};
use std::sync::Arc;

const VOCAB_SIZE: usize = 256;
const NUM_LAYERS: usize = 2;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 8;
const BLOCK_SIZE: usize = 4;

fn make_engine() -> (ContinuousBatchEngine, Arc<PagedKvCacheManager>) {
    let exec_config = PagedExecutorConfig {
        vocab_size: VOCAB_SIZE,
        num_layers: NUM_LAYERS,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_sequence_length: 512,
    };

    let kv_config = PagedKvCacheConfig {
        block_size: BLOCK_SIZE,
        max_gpu_blocks: 64,
        max_cpu_blocks: 0,
        enable_cow: false,
        enable_swapping: false,
        num_layers: NUM_LAYERS,
        num_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        enable_prefix_cache: false,
        ..Default::default()
    };

    let kv_manager =
        Arc::new(PagedKvCacheManager::new(ferrum_types::Device::CPU, kv_config).unwrap());
    let executor = Arc::new(PagedAttentionExecutor::new(exec_config, kv_manager.clone()));

    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let tensor_factory = Arc::new(MockTensorFactory);

    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_manager.clone(),
        executor,
        tensor_factory,
    );

    (engine, kv_manager)
}

fn make_request(prompt: &str) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "paged-test-model");
    req.sampling_params.max_tokens = 5;
    req
}

#[tokio::test]
async fn single_request_with_paged_kv() {
    let (engine, kv_manager) = make_engine();
    let request = make_request("Hello world");
    let response = engine.infer(request).await.unwrap();

    // Should complete with some tokens generated
    assert!(
        !response.text.is_empty() || !response.tokens.is_empty(),
        "Should produce output"
    );

    // KV cache should be fully deallocated after completion
    let stats = kv_manager.stats();
    assert_eq!(stats.active_caches, 0, "KV cache should be deallocated");
}

#[tokio::test]
async fn multiple_requests_sequential() {
    let (engine, kv_manager) = make_engine();

    for i in 0..3 {
        let request = make_request(&format!("Request {}", i));
        let response = engine.infer(request).await.unwrap();
        assert!(
            !response.tokens.is_empty(),
            "Request {} should produce tokens",
            i
        );
    }

    // All caches should be cleaned up
    let stats = kv_manager.stats();
    assert_eq!(stats.active_caches, 0);
}

#[tokio::test]
async fn paged_executor_tracks_operations() {
    let exec_config = PagedExecutorConfig {
        vocab_size: VOCAB_SIZE,
        num_layers: NUM_LAYERS,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_sequence_length: 512,
    };

    let kv_config = PagedKvCacheConfig {
        block_size: BLOCK_SIZE,
        max_gpu_blocks: 64,
        max_cpu_blocks: 0,
        enable_cow: false,
        enable_swapping: false,
        num_layers: NUM_LAYERS,
        num_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        enable_prefix_cache: false,
        ..Default::default()
    };

    let kv_manager =
        Arc::new(PagedKvCacheManager::new(ferrum_types::Device::CPU, kv_config).unwrap());
    let executor = Arc::new(PagedAttentionExecutor::new(exec_config, kv_manager.clone()));

    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let tensor_factory = Arc::new(MockTensorFactory);

    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_manager,
        executor.clone(),
        tensor_factory,
    );

    assert_eq!(executor.prefill_count(), 0);
    assert_eq!(executor.decode_count(), 0);

    let request = make_request("Track ops");
    let _response = engine.infer(request).await.unwrap();

    // Should have done 1 prefill and (max_tokens - 1) decodes
    assert_eq!(executor.prefill_count(), 1);
    assert_eq!(executor.decode_count(), 4); // max_tokens=5, first from prefill
}

#[tokio::test]
async fn streaming_with_paged_kv() {
    use futures::StreamExt;

    let (engine, _kv_manager) = make_engine();
    let request = make_request("Stream test");

    let stream = engine.infer_stream(request).await.unwrap();
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert!(!chunks.is_empty(), "Stream should produce chunks");

    let last = chunks.last().unwrap();
    assert!(
        last.finish_reason.is_some(),
        "Final chunk should have finish_reason"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Preemption tests
// ────────────────────────────────────────────────────────────────────────────

/// Build an engine with a very small block pool to force preemption.
fn make_tight_engine() -> (Arc<ContinuousBatchEngine>, Arc<PagedKvCacheManager>) {
    let exec_config = PagedExecutorConfig {
        vocab_size: VOCAB_SIZE,
        num_layers: NUM_LAYERS,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_sequence_length: 512,
    };

    // Very small pool: 6 blocks total, each request needs ~2 blocks
    let kv_config = PagedKvCacheConfig {
        block_size: BLOCK_SIZE,
        max_gpu_blocks: 6,
        max_cpu_blocks: 0,
        enable_cow: false,
        enable_swapping: false,
        num_layers: NUM_LAYERS,
        num_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        enable_prefix_cache: false,
        ..Default::default()
    };

    let kv_manager =
        Arc::new(PagedKvCacheManager::new(ferrum_types::Device::CPU, kv_config).unwrap());
    let executor = Arc::new(PagedAttentionExecutor::new(exec_config, kv_manager.clone()));

    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let tensor_factory = Arc::new(MockTensorFactory);

    let engine = Arc::new(ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_manager.clone(),
        executor,
        tensor_factory,
    ));

    (engine, kv_manager)
}

#[tokio::test]
async fn preemption_allows_all_requests_to_complete() {
    let (engine, kv_manager) = make_tight_engine();

    // Submit more requests than the block pool can handle simultaneously.
    // With 6 blocks and ~2 blocks per request, only ~3 can be active at once.
    // Submitting 5 forces preemption.
    let mut handles = Vec::new();
    for i in 0..5 {
        let e = engine.clone();
        handles.push(tokio::spawn(async move {
            let mut req = make_request(&format!("Preempt test {}", i));
            req.sampling_params.max_tokens = 3;
            e.infer(req).await
        }));
    }

    let results: Vec<InferenceResponse> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap().unwrap())
        .collect();

    assert_eq!(results.len(), 5, "All requests should complete");
    for resp in &results {
        assert!(!resp.tokens.is_empty(), "Each should generate tokens");
    }

    // All KV caches freed after completion
    let stats = kv_manager.stats();
    assert_eq!(stats.active_caches, 0, "All KV caches should be freed");
}
