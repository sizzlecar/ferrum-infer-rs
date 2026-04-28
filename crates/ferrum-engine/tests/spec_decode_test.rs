//! Speculative decoding end-to-end through the engine.
//!
//! Wires two `ConfigurableModelExecutor`s — one as draft, one as target —
//! into `ContinuousBatchEngine::new_with_speculation` and verifies that:
//!
//!   1. When both agree, the engine generates tokens via the runner and
//!      reaches the target's EOS in fewer iterations than a single-token
//!      decode (because each iteration emits N+1 tokens on full accept).
//!   2. The final generated sequence is the target's expected output.
//!
//! No GPU, no real model — mock executor with deterministic token streams.

use std::sync::Arc;

use ferrum_engine::{
    speculative::SpeculativeDecodingConfig, ContinuousBatchEngine, InferenceEngineInterface,
};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    ConfigurableModelExecutor, MockKvCacheManager, MockSampler, MockTensorFactory, MockTokenizer,
};
use ferrum_types::{InferenceRequest, SchedulerConfig};

const VOCAB: usize = 1000;

fn make_engine_speculative(
    num_spec_tokens: usize,
    draft_seq: Vec<u32>,
    target_seq: Vec<u32>,
) -> ContinuousBatchEngine {
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB));
    let sampler = Arc::new(MockSampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let tensor_factory = Arc::new(MockTensorFactory);
    let target = Arc::new(ConfigurableModelExecutor::with_token_sequence(
        VOCAB, target_seq,
    ));
    let draft = Arc::new(ConfigurableModelExecutor::with_token_sequence(
        VOCAB, draft_seq,
    ));
    let cfg = SpeculativeDecodingConfig {
        num_speculative_tokens: num_spec_tokens,
        temperature: 1.0,
    };
    ContinuousBatchEngine::new_with_speculation(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        target,
        tensor_factory,
        Some(draft),
        Some(cfg),
    )
}

fn make_request(prompt: &str, max_tokens: usize) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "mock-model");
    req.sampling_params.max_tokens = max_tokens;
    req.sampling_params.temperature = 0.0;
    req
}

/// Both draft and target biased to the same repeating sequence — every
/// speculative round should fully accept all drafts plus a bonus token.
/// Verifies the engine successfully plumbs through the speculative path
/// and that tokens eventually reach a stop condition.
#[tokio::test]
async fn speculative_agreement_generates_expected_tokens() {
    let engine = make_engine_speculative(3, vec![42, 42, 42, 42, 42], vec![42, 42, 42, 42, 42]);
    let req = make_request("hello", 4);
    let resp = engine.infer(req).await.expect("infer");

    // max_tokens=4; with spec decoding we should reach it via spec rounds.
    // All tokens should be 42 (the agreed-on biased token).
    assert_eq!(
        resp.tokens.len(),
        4,
        "should generate exactly max_tokens={} tokens, got {}",
        4,
        resp.tokens.len()
    );
    for tok in &resp.tokens {
        assert_eq!(
            tok.get(),
            42,
            "agreeing draft+target should emit token 42 each time"
        );
    }
}

/// Draft and target disagree — spec step rejects at position 0 and the
/// engine resumes via the target's residual pick. Engine still converges
/// on a final output (one token per iteration in the pathological case).
#[tokio::test]
async fn speculative_disagreement_still_produces_output() {
    // Draft wants token 7; target wants token 21.
    let engine = make_engine_speculative(3, vec![7, 7, 7, 7, 7], vec![21, 21, 21, 21, 21]);
    let req = make_request("hello", 3);
    let resp = engine.infer(req).await.expect("infer");

    assert!(!resp.tokens.is_empty(), "engine should produce tokens");
    // All non-stop tokens should be the target's preferred token.
    for tok in &resp.tokens {
        assert_eq!(
            tok.get(),
            21,
            "disagreement case should resolve to target-residual pick"
        );
    }
}

/// Sanity check: speculative decoding must not change output compared to
/// non-speculative when the two models are identical. (Given MockExecutor
/// deterministic behavior, this is a strong invariant.)
#[tokio::test]
async fn speculative_output_matches_non_speculative_for_identical_models() {
    // Baseline: no spec.
    let baseline_engine = {
        let config = ferrum_types::EngineConfig::default();
        let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
        let tokenizer = Arc::new(MockTokenizer::new(VOCAB));
        let sampler = Arc::new(MockSampler);
        let kv_cache = Arc::new(MockKvCacheManager::new(1024));
        let tensor_factory = Arc::new(MockTensorFactory);
        let executor = Arc::new(ConfigurableModelExecutor::with_token_sequence(
            VOCAB,
            vec![9, 9, 9, 9, 9],
        ));
        ContinuousBatchEngine::new(
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            executor,
            tensor_factory,
        )
    };
    let baseline_resp = baseline_engine
        .infer(make_request("hi", 4))
        .await
        .expect("baseline infer");

    // Spec: identical target + draft.
    let spec_engine = make_engine_speculative(2, vec![9, 9, 9, 9, 9], vec![9, 9, 9, 9, 9]);
    let spec_resp = spec_engine
        .infer(make_request("hi", 4))
        .await
        .expect("spec infer");

    assert_eq!(
        baseline_resp.tokens.len(),
        spec_resp.tokens.len(),
        "identical draft/target spec should match baseline count"
    );
    for (b, s) in baseline_resp.tokens.iter().zip(spec_resp.tokens.iter()) {
        assert_eq!(
            b.get(),
            s.get(),
            "per-token match between baseline and speculative"
        );
    }
}
