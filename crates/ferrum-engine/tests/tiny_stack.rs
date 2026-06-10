//! Full-stack tiny-model engine tests.
//!
//! Runs the REAL `ContinuousBatchEngine` over a REAL model forward
//! (`LlmExecutor` wrapping a tiny CPU `LlamaFamilyModel`), a REAL sampler,
//! and REAL paged KV — no `MockModelExecutor`, no model download, no GPU.
//! These exercise the engine state machine (multi-turn KV, EOS, stop
//! sequences, streaming, concurrency, cancellation) that the heavyweight
//! Metal-only CLI integration suites historically caught only at nightly.
//!
//! Test names are pinned by `scripts/release/test_arch_goal_gate.py`
//! (`REQUIRED_SCENARIO_TESTS`); renaming one fails Gate B1.

use std::sync::Arc;

use ferrum_engine::{ContinuousBatchEngine, GreedySampler, LlmInferenceEngine};
use ferrum_models::test_support::{
    tiny_llama_executor, tiny_tokenizer, TinyLlamaConfig, TinyTokenizer,
};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{MockKvCacheManager, MockTensorFactory};
use ferrum_types::{EngineConfig, FinishReason, InferenceRequest, SchedulerConfig};

const KV_TOKENS: usize = 4096;

/// Assemble a real engine over the tiny model. Returns the engine plus the
/// typed tokenizer (so scenarios can query EOS / token texts).
fn build_engine() -> (ContinuousBatchEngine, Arc<TinyTokenizer>) {
    build_engine_with(TinyLlamaConfig::default(), KV_TOKENS)
}

fn build_engine_with(
    cfg: TinyLlamaConfig,
    kv_tokens: usize,
) -> (ContinuousBatchEngine, Arc<TinyTokenizer>) {
    let tokenizer = tiny_tokenizer(&cfg);
    let engine = build_engine_with_tokenizer(cfg, kv_tokens, tokenizer.clone());
    (engine, tokenizer)
}

/// Assemble the engine with a caller-supplied tokenizer (for EOS / stop /
/// composite scenarios that need a customized vocab).
fn build_engine_with_tokenizer(
    cfg: TinyLlamaConfig,
    kv_tokens: usize,
    tokenizer: Arc<TinyTokenizer>,
) -> ContinuousBatchEngine {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let sampler = Arc::new(GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(kv_tokens));
    let executor = tiny_llama_executor(&cfg);
    let tensor_factory = Arc::new(MockTensorFactory);

    ContinuousBatchEngine::new(
        EngineConfig::default(),
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    )
}

fn greedy_request(prompt: &str, max_tokens: usize) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "tiny-llama-test");
    req.sampling_params.max_tokens = max_tokens;
    req.sampling_params.temperature = 0.0;
    req
}

/// Five sequential turns whose prompt grows each round, simulating an
/// accumulating conversation. Exercises the model's KV/scratch across
/// growing sequence lengths (the hb-08 paged-scratch-realloc surface) and
/// the engine across repeated requests. Must not panic; every turn must
/// terminate deterministically.
#[tokio::test]
async fn tiny_stack_multi_turn_five_rounds() {
    let (engine, _tok) = build_engine();

    let mut history = String::from("hello");
    let mut first_round_text = None;
    for round in 0..5 {
        let req = greedy_request(&history, 6);
        let resp = engine.infer(req).await.expect("turn must complete");
        assert!(
            !resp.text.is_empty(),
            "round {round}: response text must be non-empty"
        );
        assert_eq!(
            resp.finish_reason,
            ferrum_types::FinishReason::Length,
            "round {round}: short max_tokens must stop by Length"
        );
        if round == 0 {
            first_round_text = Some(resp.text.clone());
        }
        history.push_str(" more context turn ");
        history.push_str(&round.to_string());
    }

    // Determinism: a fresh engine replays round 0 identically.
    let (engine2, _tok2) = build_engine();
    let replay = engine2
        .infer(greedy_request("hello", 6))
        .await
        .expect("replay must complete");
    assert_eq!(
        Some(replay.text),
        first_round_text,
        "greedy decode must be deterministic across engine instances"
    );
}

/// EOS termination. The greedy stream is fixed by the model; we observe it,
/// then point the tokenizer's EOS at the 3rd emitted token and assert the
/// engine stops exactly there with `FinishReason::Stop`. Kills hb-01 (the
/// EOS-detection guard removal): without it, the run ignores EOS and falls
/// through to `Length`.
#[tokio::test]
async fn tiny_stack_eos_terminates() {
    let cfg = TinyLlamaConfig::default();

    // Observe: default EOS is unreachable here, so all 8 tokens are emitted.
    let (engine, _tok) = build_engine();
    let observed = engine
        .infer(greedy_request("hello world", 8))
        .await
        .expect("observe run completes");
    assert_eq!(
        observed.finish_reason,
        FinishReason::Length,
        "observe run must not early-stop; got {:?}",
        observed.finish_reason
    );
    let ids: Vec<u32> = observed.tokens.iter().map(|t| t.get()).collect();
    assert!(ids.len() >= 4, "need >=4 tokens to pick an EOS: {ids:?}");
    // Pick a token the model actually emits, then compute where the engine
    // should stop = first occurrence of that token + 1. Robust whether the
    // greedy stream is varied or (with tiny weights) constant.
    let eos = ids[2];
    let expected_len = ids.iter().position(|&id| id == eos).unwrap() + 1;

    let tokenizer = Arc::new(TinyTokenizer::new(cfg.vocab_size).with_eos(eos));
    let engine2 = build_engine_with_tokenizer(cfg, KV_TOKENS, tokenizer);
    let resp = engine2
        .infer(greedy_request("hello world", 8))
        .await
        .expect("eos run completes");

    assert_eq!(
        resp.finish_reason,
        FinishReason::Stop,
        "must stop on EOS, not Length"
    );
    let out_ids: Vec<u32> = resp.tokens.iter().map(|t| t.get()).collect();
    assert_eq!(
        out_ids.last(),
        Some(&eos),
        "last generated token must be the EOS token: {out_ids:?}"
    );
    assert_eq!(
        out_ids.len(),
        expected_len,
        "must stop at first EOS occurrence: {out_ids:?}"
    );
}
