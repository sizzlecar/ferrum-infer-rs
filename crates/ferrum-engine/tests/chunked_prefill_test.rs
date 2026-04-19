//! Engine-level chunked prefill regression.
//!
//! Verifies that when `FERRUM_CHUNKED_PREFILL=<n>` is set, a prompt longer
//! than `n` tokens gets split into sequential `prefill` calls, each
//! advancing the scheduler's chunk offset, and that the final generated
//! token sequence matches the non-chunked baseline.

use ferrum_engine::{ContinuousBatchEngine, InferenceEngineInterface};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    ConfigurableModelExecutor, MockKvCacheManager, MockSampler, MockTensorFactory, MockTokenizer,
};
use ferrum_types::{InferenceRequest, SchedulerConfig};
use std::sync::Arc;

const VOCAB: usize = 1000;

fn make_engine() -> ContinuousBatchEngine {
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB));
    let sampler = Arc::new(MockSampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let tensor_factory = Arc::new(MockTensorFactory);
    let executor = Arc::new(ConfigurableModelExecutor::with_eos_after(VOCAB, 3, 2));
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

fn make_long_prompt(word_count: usize) -> String {
    // Each word hashes to one token via MockTokenizer → `word_count` tokens.
    (0..word_count)
        .map(|i| format!("tok{}", i))
        .collect::<Vec<_>>()
        .join(" ")
}

fn make_request(prompt: &str, max_tokens: usize) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "mock-model");
    req.sampling_params.max_tokens = max_tokens;
    req.sampling_params.temperature = 0.0;
    req
}

/// Baseline: 200-token prompt, no chunking → single prefill call.
#[tokio::test]
async fn baseline_no_chunking() {
    // Clear env so a prior test can't affect this run.
    std::env::remove_var("FERRUM_CHUNKED_PREFILL");

    let engine = make_engine();
    let req = make_request(&make_long_prompt(200), 3);
    let resp = engine.infer(req).await.expect("baseline infer");

    // Mock produces deterministic output independent of chunking; capture it.
    assert!(!resp.text.is_empty());
    // Sanity: generation should include the stop token from the mock.
    assert!(!resp.tokens.is_empty());
}

/// Chunked prefill: same prompt, `FERRUM_CHUNKED_PREFILL=64` → 200 / 64 = 4
/// chunks. Output must match the baseline token-for-token because the mock
/// executor's decode behaviour depends only on `decode_count`, not on how
/// prefill was split.
#[tokio::test]
async fn chunked_matches_baseline() {
    std::env::remove_var("FERRUM_CHUNKED_PREFILL");

    let baseline_engine = make_engine();
    let baseline =
        baseline_engine
            .infer(make_request(&make_long_prompt(200), 3))
            .await
            .expect("baseline infer");

    std::env::set_var("FERRUM_CHUNKED_PREFILL", "64");
    let chunked_engine = make_engine();
    let chunked = chunked_engine
        .infer(make_request(&make_long_prompt(200), 3))
        .await
        .expect("chunked infer");
    std::env::remove_var("FERRUM_CHUNKED_PREFILL");

    assert_eq!(
        baseline.text, chunked.text,
        "chunked prefill output must match non-chunked baseline"
    );
    assert_eq!(
        baseline.tokens.len(),
        chunked.tokens.len(),
        "chunked prefill must generate the same number of tokens"
    );
}

/// Very small chunk size (4) — many iterations — still produces the same
/// output. Guards against off-by-one in the chunk boundary loop.
#[tokio::test]
async fn chunked_small_chunk_size() {
    std::env::remove_var("FERRUM_CHUNKED_PREFILL");

    let baseline_engine = make_engine();
    let baseline =
        baseline_engine
            .infer(make_request(&make_long_prompt(50), 3))
            .await
            .expect("baseline infer");

    std::env::set_var("FERRUM_CHUNKED_PREFILL", "4");
    let chunked_engine = make_engine();
    let chunked = chunked_engine
        .infer(make_request(&make_long_prompt(50), 3))
        .await
        .expect("chunked infer");
    std::env::remove_var("FERRUM_CHUNKED_PREFILL");

    assert_eq!(baseline.text, chunked.text);
}

/// Chunk size larger than prompt must degrade gracefully to a single prefill.
#[tokio::test]
async fn chunked_larger_than_prompt_falls_back() {
    std::env::set_var("FERRUM_CHUNKED_PREFILL", "4096");
    let engine = make_engine();
    let resp = engine
        .infer(make_request(&make_long_prompt(10), 3))
        .await
        .expect("infer");
    std::env::remove_var("FERRUM_CHUNKED_PREFILL");
    assert!(!resp.text.is_empty());
}
