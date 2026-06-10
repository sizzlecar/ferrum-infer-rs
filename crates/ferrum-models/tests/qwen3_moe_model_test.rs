//! `Qwen3MoeModel<B>` integration smoke test: load from a synthesized
//! Qwen3-MoE-shaped GGUF, run a small prefill + decode sequence on CPU,
//! verify the model state machine is coherent (no NaN logits, no panics
//! across multi-layer dispatch, KV cache grows by one each decode step).
//!
//! This is intentionally architectural — not a numerical-parity test
//! against a reference implementation. Numerical correctness of the MoE
//! primitive is covered by `moe_dispatch_test.rs`. What's new here is
//! the wiring between attention + router + expert dispatch + LM head.

use std::io::{Cursor, Write};
use std::sync::Arc;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_models::common::DecoderOnlyLLM;
use ferrum_models::models::Qwen3MoeModel;
use ferrum_models::moe_config::Qwen3MoeConfig;
use ferrum_quantization::gguf::{GgufFile, GgufLoader};

// Tiny MoE shape — enough exercise for the full forward path.
const VOCAB: usize = 16;
const HIDDEN: usize = 8;
const NUM_HEADS: usize = 2;
const HEAD_DIM: usize = 4; // hidden / heads
const NUM_KV_HEADS: usize = 2;
const NUM_LAYERS: usize = 2;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const EXPERT_FFN: usize = 8;

fn ramp_2d(rows: usize, cols: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let n = rows * cols;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.001).collect();
    let t = Tensor::from_vec(raw, (rows, cols), &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

fn ramp_3d(d0: usize, d1: usize, d2: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let n = d0 * d1 * d2;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.001).collect();
    let t = Tensor::from_vec(raw, (d0, d1, d2), &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

fn ramp_1d(n: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.01).collect();
    let t = Tensor::from_vec(raw, n, &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

/// Build a Qwen3-MoE-shaped GGUF with all the metadata `Qwen3MoeConfig::from_gguf`
/// reads, plus per-layer attention + MoE tensors.
fn build_synth_moe_gguf() -> tempfile::NamedTempFile {
    // Top-level
    let token_embd = ramp_2d(VOCAB, HIDDEN, 0.0);
    let output_norm = ramp_1d(HIDDEN, 0.5);
    let output = ramp_2d(VOCAB, HIDDEN, 1.0);

    // Per-layer tensors: attention dense + MoE-specific.
    // We construct two layers worth of (attn_norm, attn_q/k/v, attn_output,
    // ffn_norm, router, gate_exps, up_exps, down_exps).
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    tensors.push(("token_embd.weight".into(), token_embd));
    tensors.push(("output_norm.weight".into(), output_norm));
    tensors.push(("output.weight".into(), output));

    for li in 0..NUM_LAYERS {
        let base = (li + 1) as f32;
        let attn_norm = ramp_1d(HIDDEN, 0.6 * base);
        let attn_q = ramp_2d(NUM_HEADS * HEAD_DIM, HIDDEN, 2.0 * base);
        let attn_k = ramp_2d(NUM_KV_HEADS * HEAD_DIM, HIDDEN, 3.0 * base);
        let attn_v = ramp_2d(NUM_KV_HEADS * HEAD_DIM, HIDDEN, 4.0 * base);
        let attn_output = ramp_2d(HIDDEN, NUM_HEADS * HEAD_DIM, 5.0 * base);
        let ffn_norm = ramp_1d(HIDDEN, 0.7 * base);

        let router = ramp_2d(NUM_EXPERTS, HIDDEN, 8.0 * base);
        let gate_exps = ramp_3d(NUM_EXPERTS, EXPERT_FFN, HIDDEN, 9.0 * base);
        let up_exps = ramp_3d(NUM_EXPERTS, EXPERT_FFN, HIDDEN, 10.0 * base);
        let down_exps = ramp_3d(NUM_EXPERTS, HIDDEN, EXPERT_FFN, 11.0 * base);

        tensors.push((format!("blk.{li}.attn_norm.weight"), attn_norm));
        tensors.push((format!("blk.{li}.attn_q.weight"), attn_q));
        tensors.push((format!("blk.{li}.attn_k.weight"), attn_k));
        tensors.push((format!("blk.{li}.attn_v.weight"), attn_v));
        tensors.push((format!("blk.{li}.attn_output.weight"), attn_output));
        tensors.push((format!("blk.{li}.ffn_norm.weight"), ffn_norm));
        tensors.push((format!("blk.{li}.ffn_gate_inp.weight"), router));
        tensors.push((format!("blk.{li}.ffn_gate_exps.weight"), gate_exps));
        tensors.push((format!("blk.{li}.ffn_up_exps.weight"), up_exps));
        tensors.push((format!("blk.{li}.ffn_down_exps.weight"), down_exps));
    }

    // Metadata — `Qwen3MoeConfig::from_gguf` keys.
    let arch = Value::String("qwen3moe".to_string());
    let block_count = Value::U32(NUM_LAYERS as u32);
    let embed_len = Value::U32(HIDDEN as u32);
    let head_cnt = Value::U32(NUM_HEADS as u32);
    let head_cnt_kv = Value::U32(NUM_KV_HEADS as u32);
    let rms_eps = Value::F32(1e-5);
    let ctx_len = Value::U32(64);
    let rope_theta = Value::F32(10_000.0);
    let vocab_sz = Value::U32(VOCAB as u32);
    let expert_count = Value::U32(NUM_EXPERTS as u32);
    let expert_used = Value::U32(TOP_K as u32);
    let expert_ffn = Value::U32(EXPERT_FFN as u32);
    let norm_topk = Value::Bool(true);

    let metadata: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch),
        ("qwen3moe.block_count", &block_count),
        ("qwen3moe.embedding_length", &embed_len),
        ("qwen3moe.attention.head_count", &head_cnt),
        ("qwen3moe.attention.head_count_kv", &head_cnt_kv),
        ("qwen3moe.attention.layer_norm_rms_epsilon", &rms_eps),
        ("qwen3moe.context_length", &ctx_len),
        ("qwen3moe.rope.freq_base", &rope_theta),
        ("qwen3moe.vocab_size", &vocab_sz),
        ("qwen3moe.expert_count", &expert_count),
        ("qwen3moe.expert_used_count", &expert_used),
        ("qwen3moe.expert_feed_forward_length", &expert_ffn),
        ("qwen3moe.expert_norm_topk_prob", &norm_topk),
    ];
    let tensors_view: Vec<(&str, &QTensor)> =
        tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        gguf_file::write(&mut cursor, &metadata, &tensors_view).unwrap();
    }
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();
    tmp
}

/// Building blocks of `Qwen3MoeModel::new` plumb through to construction.
/// Loads + asserts dimensions are coherent before any forward call.
fn build_model_for_test() -> Qwen3MoeModel<CpuBackend> {
    let tmp = build_synth_moe_gguf();
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = Qwen3MoeConfig::from_gguf(&gguf).unwrap();
    let gguf_arc = Arc::new(gguf);
    let loader = GgufLoader::<CpuBackend>::from_file(gguf_arc.clone());
    Qwen3MoeModel::<CpuBackend>::new(cfg, &loader, &gguf_arc).unwrap()
}

#[test]
fn model_loads_from_synth_gguf() {
    let model = build_model_for_test();
    assert_eq!(model.cfg.base.num_layers, NUM_LAYERS);
    assert_eq!(model.cfg.base.hidden_size, HIDDEN);
    assert_eq!(model.cfg.num_experts, NUM_EXPERTS);
    assert_eq!(model.cfg.num_experts_per_tok, TOP_K);
    assert_eq!(model.cfg.expert_intermediate_size, EXPERT_FFN);
    // One MoE state per layer + matching attention layers.
    assert_eq!(model.attn_layers.len(), NUM_LAYERS);
    assert_eq!(model.moe_layers.len(), NUM_LAYERS);
    // Each MoE layer has num_experts experts.
    for l in &model.moe_layers {
        assert_eq!(l.experts.num_experts(), NUM_EXPERTS);
    }
    // Router shape sanity.
    let r0 = &model.moe_layers[0].router;
    assert_eq!(r0.in_features(), HIDDEN);
    assert_eq!(r0.out_features(), NUM_EXPERTS);
}

#[test]
fn prefill_returns_finite_logits() {
    let mut model = build_model_for_test();
    // 3-token prompt, all valid token ids.
    let tokens: Vec<u32> = vec![1, 2, 3];
    let logits = model.prefill("test", &tokens);
    assert_eq!(logits.len(), VOCAB);
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "non-finite logit at vocab index {i}: {v}");
    }
}

#[test]
fn decode_advances_kv_cache_one_per_step() {
    let mut model = build_model_for_test();
    let tokens: Vec<u32> = vec![1, 2];
    let _ = model.prefill("seq1", &tokens);
    // After 2-token prefill, kv cache len should be 2.
    let kv_len_after_prefill = model
        .kv_caches
        .get("seq1")
        .and_then(|layers| layers.first())
        .map(|c| c.len)
        .unwrap();
    assert_eq!(kv_len_after_prefill, tokens.len());

    // Decode 3 tokens. Each step should grow the cache by one.
    for step in 0..3 {
        let pos = (kv_len_after_prefill + step) as u32;
        let logits = model.decode("seq1", 5 + step as u32, pos);
        assert_eq!(logits.len(), VOCAB);
        let new_len = model
            .kv_caches
            .get("seq1")
            .and_then(|layers| layers.first())
            .map(|c| c.len)
            .unwrap();
        assert_eq!(
            new_len,
            kv_len_after_prefill + step + 1,
            "kv cache length wrong at decode step {step}"
        );
    }
}

#[test]
fn release_and_reset_clear_kv_caches() {
    let mut model = build_model_for_test();
    let _ = model.prefill("a", &[1, 2]);
    let _ = model.prefill("b", &[3, 4]);
    assert!(model.kv_caches.contains_key("a"));
    assert!(model.kv_caches.contains_key("b"));

    model.release("a");
    assert!(!model.kv_caches.contains_key("a"));
    assert!(model.kv_caches.contains_key("b"));

    model.reset();
    assert!(model.kv_caches.is_empty());
}

// ──────────────────────────────────────────────────────────────────────
// Phase 3 (CUDA Graph capture) smoke tests
//
// FERRUM_MOE_GRAPH=1 opts into capture in `decode_batch_internal`. The
// CPU backend has no graph machinery (begin/end/replay return Err or
// Ok(false) defaults), so the wrapper must degrade gracefully:
//   * decode_batch returns the same finite logits as without the flag
//   * `batched_graph_failed` flips to true after the first attempt
//   * subsequent calls stay eager (no repeated error spam)
//
// Real capture validation requires a CUDA pod — these tests only
// cover the state-machine + fall-back path.
//
// All these tests set FERRUM_MOE_BATCH_THRESHOLD=2 so m=2 hits
// decode_batch_internal (default 8 would route to the per-token
// decode() loop instead, which has no graph wrapper).
// ──────────────────────────────────────────────────────────────────────

/// Serializes all Phase 3 tests on a shared mutex so their env-var
/// set/restore doesn't race with each other (cargo test runs N tests
/// in parallel by default).
static MOE_GRAPH_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Set both env vars needed for the Phase 3 tests. Returns a guard
/// that restores the previous values on drop.
struct MoeGraphEnvGuard {
    prev_graph: Option<String>,
    prev_threshold: Option<String>,
    prev_vllm_moe: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl MoeGraphEnvGuard {
    fn set(graph_value: &str) -> Self {
        // Acquire the global lock FIRST — only one Phase 3 test mutates
        // env vars at a time. Other tests in this file are unaffected.
        let lock = MOE_GRAPH_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev_graph = std::env::var("FERRUM_MOE_GRAPH").ok();
        let prev_threshold = std::env::var("FERRUM_MOE_BATCH_THRESHOLD").ok();
        let prev_vllm_moe = std::env::var("FERRUM_VLLM_MOE").ok();
        // SAFETY: the lock above serializes env-var access across tests.
        unsafe {
            std::env::set_var("FERRUM_MOE_GRAPH", graph_value);
            std::env::set_var("FERRUM_MOE_BATCH_THRESHOLD", "2");
            if graph_value == "1" {
                std::env::set_var("FERRUM_VLLM_MOE", "1");
            } else {
                std::env::remove_var("FERRUM_VLLM_MOE");
            }
        }
        // The model now resolves runtime knobs from the process-wide snapshot
        // installed at the composition root, not std::env directly. Mirror the
        // just-set env into it so each graph_value constructs with these knobs.
        ferrum_types::install_runtime_snapshot(
            ferrum_types::RuntimeConfigSnapshot::capture_current(),
        );
        Self {
            prev_graph,
            prev_threshold,
            prev_vllm_moe,
            _lock: lock,
        }
    }
}

impl Drop for MoeGraphEnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref v) = self.prev_graph {
                std::env::set_var("FERRUM_MOE_GRAPH", v);
            } else {
                std::env::remove_var("FERRUM_MOE_GRAPH");
            }
            if let Some(ref v) = self.prev_threshold {
                std::env::set_var("FERRUM_MOE_BATCH_THRESHOLD", v);
            } else {
                std::env::remove_var("FERRUM_MOE_BATCH_THRESHOLD");
            }
            if let Some(ref v) = self.prev_vllm_moe {
                std::env::set_var("FERRUM_VLLM_MOE", v);
            } else {
                std::env::remove_var("FERRUM_VLLM_MOE");
            }
        }
        ferrum_types::install_runtime_snapshot(
            ferrum_types::RuntimeConfigSnapshot::capture_current(),
        );
    }
}

/// Run two prefills + a batched decode step, returning the m logit vectors.
fn run_batched_decode_step(model: &mut Qwen3MoeModel<CpuBackend>) -> Vec<Vec<f32>> {
    let _ = model.prefill("seq_a", &[1, 2]);
    let _ = model.prefill("seq_b", &[3, 4]);
    use ferrum_models::common::DecoderOnlyLLM;
    // m=2 + threshold=2 → exercises decode_batch_internal
    // (m == 1 short-circuits, threshold=8 default routes to per-token).
    model.decode_batch(&[("seq_a".to_string(), 5, 2), ("seq_b".to_string(), 6, 2)])
}

/// Capture-disabled run must produce finite logits of the expected shape.
#[test]
fn moe_graph_disabled_baseline() {
    let _guard = MoeGraphEnvGuard::set("0");
    let mut model = build_model_for_test();
    let out = run_batched_decode_step(&mut model);
    assert_eq!(out.len(), 2);
    for row in &out {
        assert_eq!(row.len(), VOCAB);
        for v in row {
            assert!(v.is_finite(), "non-finite logit under FERRUM_MOE_GRAPH=0");
        }
    }
    // Disabled path leaves graph state untouched (graph_enabled=false →
    // none of the warmup/capture branches fire).
    assert!(!model.batched_graph_failed());
    assert_eq!(model.batched_graph_warmup(), 0);
    assert!(model.batched_graph_keys_seen().is_empty());
}

/// Capture-enabled run on CpuBackend (no graph support) must produce
/// the SAME logits as the disabled run (graceful degradation), and the
/// warmup counter must bump on each batched call.
#[test]
fn moe_graph_enabled_degrades_gracefully_on_cpu() {
    // Baseline (graph flag = 0).
    let baseline_out = {
        let _guard = MoeGraphEnvGuard::set("0");
        let mut model = build_model_for_test();
        run_batched_decode_step(&mut model)
    };

    // Same call with graph flag = 1.
    let _guard = MoeGraphEnvGuard::set("1");
    let mut model = build_model_for_test();
    let out = run_batched_decode_step(&mut model);

    // Output shape + numerical match (CPU has no real capture so the
    // wrapped path is identical to eager — any drift would be a bug).
    assert_eq!(out.len(), baseline_out.len());
    for (row, base_row) in out.iter().zip(baseline_out.iter()) {
        assert_eq!(row.len(), base_row.len());
        for (v, b) in row.iter().zip(base_row.iter()) {
            assert!(v.is_finite(), "non-finite logit under FERRUM_MOE_GRAPH=1");
            assert!(
                (v - b).abs() < 1e-5,
                "logit drift between FERRUM_MOE_GRAPH=0 and =1: {v} vs {b}"
            );
        }
    }
    // Warmup advanced once for this single batched call.
    assert_eq!(
        model.batched_graph_warmup(),
        1,
        "warmup counter should bump per batched decode call on the eager path"
    );
    // No graph captured (warmup<3 → no capture attempt; CPU defaults
    // for replay_graph return Ok(false) → no failure flag either).
    assert!(model.batched_graph_keys_seen().is_empty());
}

/// After 3+ batched calls under FERRUM_MOE_GRAPH=1, capture is attempted.
/// On CPU `begin_graph_capture` returns Err(unsupported) → batched_graph_failed
/// becomes true. The 5th call must skip the capture attempt and stay
/// eager (no panic, no repeated error spam, correct logits).
#[test]
fn moe_graph_capture_failure_recovers_to_eager() {
    let _guard = MoeGraphEnvGuard::set("1");
    let mut model = build_model_for_test();
    use ferrum_models::common::DecoderOnlyLLM;

    let _ = model.prefill("seq_a", &[1, 2]);
    let _ = model.prefill("seq_b", &[3, 4]);

    // Run 5 batched decode steps. The 4th call (warmup>=3) should attempt
    // begin_graph_capture, which fails on CpuBackend → batched_graph_failed
    // flips. Call 5 stays eager via the !failed guard.
    let mut last_out: Vec<Vec<f32>> = Vec::new();
    for step in 0..5 {
        let pos = (2 + step) as u32;
        let out = model.decode_batch(&[
            ("seq_a".to_string(), 5 + step as u32, pos),
            ("seq_b".to_string(), 6 + step as u32, pos),
        ]);
        for row in &out {
            for v in row {
                assert!(v.is_finite(), "non-finite logit at step {step}");
            }
        }
        last_out = out;
    }

    // Warmup bumps once per non-replay call. All 5 calls are eager
    // (CPU has no replay), so counter = 5.
    assert_eq!(
        model.batched_graph_warmup(),
        5,
        "warmup counter bumps once per eager call (no replay short-circuit)"
    );
    // batched_graph_failed flipped when begin_graph_capture errored on
    // CPU (some time at or after call 4 where warmup>=3 first hits).
    assert!(
        model.batched_graph_failed(),
        "CPU has no graph machinery — failure flag must be set after first capture attempt"
    );
    // No keys inserted because end_capture never succeeded.
    assert!(model.batched_graph_keys_seen().is_empty());
    // Last output must still be finite + correct shape.
    assert_eq!(last_out.len(), 2);
    assert_eq!(last_out[0].len(), VOCAB);
}

/// `reset()` clears the graph state — a subsequent decode_batch starts
/// a fresh capture cycle.
#[test]
fn moe_graph_state_cleared_on_reset() {
    let _guard = MoeGraphEnvGuard::set("1");
    let mut model = build_model_for_test();
    let _ = model.prefill("a", &[1, 2]);
    let _ = model.prefill("b", &[3, 4]);
    // One batched call to bump warmup counter.
    use ferrum_models::common::DecoderOnlyLLM;
    let _ = model.decode_batch(&[("a".to_string(), 5, 2), ("b".to_string(), 6, 2)]);
    assert!(model.batched_graph_warmup() > 0);

    model.reset();
    assert_eq!(model.batched_graph_warmup(), 0);
    assert!(!model.batched_graph_failed());
    assert!(model.batched_graph_keys_seen().is_empty());
    assert!(model.kv_caches.is_empty());
}
