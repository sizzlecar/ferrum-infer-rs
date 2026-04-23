//! Prefix cache engine integration: verify hits on repeat prompts,
//! misses on partial-prefix inputs (safety fall-through), and opt-in
//! CUDA gating via FERRUM_PREFIX_CACHE.
//!
//! Uses the same mock-executor pattern as other correctness tests — no
//! GPU required.

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
    // Cycle through [100, 200, 300]; EOS after 3 decode steps.
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

fn make_request(prompt: &str, max_tokens: usize) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "mock-model");
    req.sampling_params.max_tokens = max_tokens;
    req.sampling_params.temperature = 0.0;
    req
}

/// Two identical prompts in a row should hit the prefix cache on the
/// second call (tokenization produces the same TokenId vec → exact match).
#[tokio::test]
async fn prefix_cache_hit_on_repeat_prompt() {
    let engine = make_engine();

    // First request populates the cache.
    let req1 = make_request("hello world hello world", 3);
    let _resp1 = engine.infer(req1).await.expect("first infer");
    let hits_before = engine.prefix_cache_hits();

    // Second identical request should hit.
    let req2 = make_request("hello world hello world", 3);
    let _resp2 = engine.infer(req2).await.expect("second infer");
    let hits_after = engine.prefix_cache_hits();

    assert!(
        hits_after > hits_before,
        "expected hit on repeat prompt, hits: before={hits_before} after={hits_after}"
    );
}

/// A request whose tokens are a SUPERSET of a stored prefix must NOT be
/// served from cache (until true incremental-prefill lands). The
/// engine's exact-match gate should force a full prefill — stats must
/// show a miss here.
#[tokio::test]
async fn prefix_cache_partial_match_falls_through() {
    let engine = make_engine();

    // Seed the cache with a short prompt.
    let seed = make_request("aaa bbb", 3);
    let _ = engine.infer(seed).await.expect("seed infer");
    let hits_after_seed = engine.prefix_cache_hits();

    // Longer prompt starting with the same tokens — partial match, must miss.
    let longer = make_request("aaa bbb ccc ddd", 3);
    let _ = engine.infer(longer).await.expect("longer infer");
    let hits_after_longer = engine.prefix_cache_hits();

    assert_eq!(
        hits_after_longer, hits_after_seed,
        "partial-prefix request must not count as cache hit (no incremental prefill yet)"
    );
}

/// Fresh engine with no prior cache must miss on any request.
#[tokio::test]
async fn prefix_cache_cold_start_miss() {
    let engine = make_engine();
    let req = make_request("cold start", 2);
    let _ = engine.infer(req).await.expect("infer");
    assert_eq!(engine.prefix_cache_hits(), 0, "cold start should miss");
}

/// Stats accessor returns sensible counters after mixed traffic.
#[tokio::test]
async fn prefix_cache_stats_accounting() {
    let engine = make_engine();

    // Miss
    let _ = engine.infer(make_request("zzz yyy", 2)).await;
    // Hit
    let _ = engine.infer(make_request("zzz yyy", 2)).await;

    let stats = engine.prefix_cache_stats();
    assert!(stats.hits >= 1, "expected >=1 hit, got {}", stats.hits);
    assert!(stats.misses >= 1, "expected >=1 miss, got {}", stats.misses);
    assert!(stats.active_prefixes >= 1);
}
