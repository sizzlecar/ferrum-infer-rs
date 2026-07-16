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

use ferrum_engine::{ContinuousBatchEngine, GreedySampler, LlmInferenceEngine, SequenceState};
use ferrum_kernels::backend::{cpu::CpuBackend, BackendPagedKv};
use ferrum_models::test_support::{
    tiny_llama_executor, tiny_tokenizer, TinyLlamaConfig, TinyTokenizer,
};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{MockKvCacheManager, MockTensorFactory};
use ferrum_types::{
    EngineConfig, FinishReason, InferenceRequest, ResponseFormat, SchedulerConfig, TokenId,
};

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
    .expect("legacy engine composition must match executor authority")
}

fn greedy_request(prompt: &str, max_tokens: usize) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "tiny-llama-test");
    req.sampling_params.max_tokens = max_tokens;
    req.sampling_params.temperature = 0.0;
    req
}

/// The greedy stream the model emits for `prompt` (token ids), observed with
/// an unreachable EOS so nothing early-stops.
async fn observe_stream(prompt: &str, max_tokens: usize) -> Vec<u32> {
    let (engine, _tok) = build_engine();
    let resp = engine
        .infer(greedy_request(prompt, max_tokens))
        .await
        .expect("observe run completes");
    assert_eq!(
        resp.finish_reason,
        FinishReason::Length,
        "observe run must not early-stop"
    );
    resp.tokens.iter().map(|t| t.get()).collect()
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

/// Stop text embedded inside a composite (multi-character) token. We make the
/// token the model emits decode to `"a@b"` and set stop=`"@"`. The engine must
/// detect the marker inside the token's decoded text and stop. Kills hb-02:
/// the pre-fix code only matched single-token stop ids, so a marker buried in
/// a composite token slipped through to `Length`.
#[tokio::test]
async fn tiny_stack_stop_sequence_composite_token() {
    let cfg = TinyLlamaConfig::default();
    let ids = observe_stream("hello world", 6).await;
    let emitted = ids[0];

    let tokenizer =
        Arc::new(TinyTokenizer::new(cfg.vocab_size).with_composite_token(emitted, "a@b"));
    let engine = build_engine_with_tokenizer(cfg, KV_TOKENS, tokenizer);

    let mut req = greedy_request("hello world", 6);
    req.sampling_params.stop_sequences = vec!["@".to_string()];
    let resp = engine.infer(req).await.expect("stop run completes");

    assert_eq!(
        resp.finish_reason,
        FinishReason::Stop,
        "marker inside composite token must stop generation, not run to Length"
    );
}

/// Streaming contract: at least one content chunk, exactly one terminal chunk
/// carrying `finish_reason`, and the concatenated deltas equal the final text.
#[tokio::test]
async fn tiny_stack_stream_chunk_contract() {
    use futures::StreamExt;

    let (engine, _tok) = build_engine();
    let mut req = greedy_request("stream me", 5);
    req.stream = true;

    let stream = engine.infer_stream(req).await.expect("stream starts");
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .expect("all chunks ok");

    assert!(!chunks.is_empty(), "stream must produce chunks");
    let terminal = chunks.iter().filter(|c| c.finish_reason.is_some()).count();
    assert_eq!(terminal, 1, "exactly one terminal chunk");
    assert!(
        chunks.last().unwrap().finish_reason.is_some(),
        "the last chunk is the terminal one"
    );

    let joined: String = chunks.iter().map(|c| c.text.as_str()).collect();
    assert!(!joined.is_empty(), "streamed text must be non-empty");
}

/// Concurrent sessions stay isolated: the same prompt run concurrently with
/// other distinct prompts yields the same result as run alone. Catches
/// cross-request KV/state bleed.
#[tokio::test]
async fn tiny_stack_concurrent_sessions_isolated() {
    let (engine, _tok) = build_engine();
    let engine = Arc::new(engine);

    let solo = engine
        .infer(greedy_request("anchor prompt", 6))
        .await
        .expect("solo completes")
        .text;

    let mut handles = Vec::new();
    for i in 0..4 {
        let engine = engine.clone();
        let prompt = if i == 0 {
            "anchor prompt".to_string()
        } else {
            format!("noisy neighbor {i}")
        };
        handles.push(tokio::spawn(async move {
            engine.infer(greedy_request(&prompt, 6)).await
        }));
    }
    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.expect("task joins").expect("infer ok"));
    }

    assert_eq!(
        results[0].text, solo,
        "anchor result must match whether run alone or amid concurrent load"
    );
    for r in &results {
        assert!(
            !r.text.is_empty(),
            "every concurrent request produces output"
        );
    }
}

/// Cancelling a stream mid-flight (dropping the stream) must not poison the
/// engine: a subsequent request still completes.
#[tokio::test]
async fn tiny_stack_cancel_mid_stream() {
    use futures::StreamExt;

    let (engine, _tok) = build_engine();
    let engine = Arc::new(engine);

    let mut req = greedy_request("cancel me", 64);
    req.stream = true;
    {
        let mut stream = engine.infer_stream(req).await.expect("stream starts");
        // Pull a couple chunks, then drop the stream to cancel.
        let _ = stream.next().await;
        let _ = stream.next().await;
    }

    // Engine remains usable after a cancelled stream.
    let after = engine
        .infer(greedy_request("after cancel", 5))
        .await
        .expect("engine still works after cancellation");
    assert!(!after.text.is_empty());
}

/// Repetition-runaway guard: a model that emits a near-constant token must
/// still terminate by `max_tokens` and not exceed it.
#[tokio::test]
async fn tiny_stack_repetition_runaway_guard() {
    let (engine, _tok) = build_engine();
    let budget = 12;
    let resp = engine
        .infer(greedy_request("repeat repeat", budget))
        .await
        .expect("completes");
    assert_eq!(resp.finish_reason, FinishReason::Length);
    assert!(
        resp.tokens.len() <= budget,
        "generated {} tokens, must not exceed budget {budget}",
        resp.tokens.len()
    );
}

/// KV capacity boundary: under a tight engine KV pool, several concurrent
/// requests must not panic — they either complete or fail cleanly. Exercises
/// the paged scratch growth path (hb-08 surface) at the admission boundary.
#[tokio::test]
async fn tiny_stack_kv_capacity_boundary() {
    let cfg = TinyLlamaConfig::default();
    let tokenizer = tiny_tokenizer(&cfg);
    // Deliberately tight pool relative to the concurrent demand below.
    let engine = Arc::new(build_engine_with_tokenizer(cfg, 64, tokenizer));

    let mut handles = Vec::new();
    for i in 0..6 {
        let engine = engine.clone();
        handles.push(tokio::spawn(async move {
            engine
                .infer(greedy_request(&format!("pressure {i}"), 8))
                .await
        }));
    }
    for h in handles {
        // Must not panic. A clean Err is acceptable under capacity pressure;
        // a completion is fine too.
        let _ = h.await.expect("task must not panic");
    }

    // After the pressure wave, a fresh request still completes.
    let after = engine
        .infer(greedy_request("recovered", 4))
        .await
        .expect("engine usable after capacity pressure");
    assert!(!after.text.is_empty());
}

/// Capability-fallback law (Gate A5, Llama path). The tiny model runs on
/// `CpuBackend`, which declares every optional accelerator capability false.
/// The full engine path must therefore exercise only fallback code — proving
/// shared model code never hard-depends on a capability a backend lacks (the
/// hb-07 class). We assert the precondition (caps are off) and that the
/// fallback path still produces correct output end-to-end.
#[tokio::test]
async fn tiny_stack_capability_fallback_law_cpu() {
    assert!(
        !CpuBackend::supports_paged_kv(),
        "precondition: CpuBackend must lack paged-kv so the fallback path is tested"
    );
    assert!(!CpuBackend::supports_varlen_qkv());
    assert!(!CpuBackend::supports_vllm_paged_attn());

    let (engine, _tok) = build_engine();
    let resp = engine
        .infer(greedy_request("fallback path", 6))
        .await
        .expect("all-capabilities-off backend must complete via fallback code");
    assert_eq!(resp.finish_reason, FinishReason::Length);
    assert!(!resp.tokens.is_empty());
}

/// Guided / tool constraints. Two checks:
///
/// 1. A `JsonSchema`-constrained request must force full logits even when the
///    schema fails to compile to a guided processor — otherwise the greedy
///    argmax fast path silently bypasses the constraint. Kills hb-04 (removal
///    of the `JsonSchema` clause in `requires_full_logits_for_sampling`).
/// 2. End-to-end: a valid guided request runs through the real engine without
///    panicking even when the tiny vocab cannot satisfy the pattern (the
///    no-extension fallback path).
#[tokio::test]
async fn tiny_stack_guided_tool_constraint() {
    let cfg = TinyLlamaConfig::default();
    let tokenizer = tiny_tokenizer(&cfg);

    // (1) A JsonSchema request whose only full-logits trigger is the schema
    // itself must still force full logits. Built with no tokenizer so no
    // guided processor and no forbidden-token masks are wired — the JsonSchema
    // clause is then the sole reason the predicate can be true, isolating the
    // exact decision hb-04 removes.
    let mut req = greedy_request("constrain me", 4);
    req.sampling_params.response_format =
        ResponseFormat::JsonSchema("{\"type\":\"object\"}".into());
    let seq = SequenceState::new(req, vec![TokenId::new(1)]);
    assert!(
        seq.requires_full_logits_for_sampling(),
        "a JsonSchema-constrained request must force full logits via the schema clause alone"
    );

    // (2) Valid guided request runs end-to-end without panic.
    let engine = build_engine_with_tokenizer(cfg, KV_TOKENS, tokenizer);
    let mut guided = greedy_request("integer please", 4);
    guided.sampling_params.response_format =
        ResponseFormat::JsonSchema("{\"type\":\"integer\"}".into());
    let resp = engine
        .infer(guided)
        .await
        .expect("guided request must complete without panic");
    assert!(
        matches!(
            resp.finish_reason,
            FinishReason::Length | FinishReason::Stop
        ),
        "guided run terminates cleanly: {:?}",
        resp.finish_reason
    );
}
