//! Phase 2B integration tests: synthesize a Qwen-MoE-shaped GGUF (one
//! decoder layer with router + 3 stacked-expert tensors) and verify
//! `GgufLoader<B>` translates the new ferrum-side names correctly,
//! reads the tensors at full size, and rejects loading 3-D stacked
//! expert tensors as `Linear<B>` (which require rank-2).
//!
//! The runtime path that actually USES these tensors lands in Phase 2C/2D.
//! This PR just proves the loader can materialise them.

use std::io::{Cursor, Write};
use std::sync::Arc;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_quantization::gguf::{GgufFile, GgufLoader};
use ferrum_quantization::WeightLoader;

// Toy MoE dimensions: 4 experts, hidden 4, expert FFN 8.
// Router: [E=4, hidden=4] = 16 elements
// gate_exps / up_exps: [E=4, ffn=8, hidden=4] = 128 elements
// down_exps: [E=4, hidden=4, ffn=8] = 128 elements
const HIDDEN: usize = 4;
const NUM_EXPERTS: usize = 4;
const EXPERT_FFN: usize = 8;
const VOCAB: usize = 8;

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

/// Build a minimal Qwen3-MoE-shaped GGUF: top-level (embed/output),
/// one layer's attention (dense, same as dense GGUFs), and the four
/// MoE-specific tensors.
fn build_moe_gguf() -> tempfile::NamedTempFile {
    let token_embd = ramp_2d(VOCAB, HIDDEN, 0.0);
    let output_norm = ramp_1d(HIDDEN, 0.5);
    let output = ramp_2d(VOCAB, HIDDEN, 1.0);

    let attn_norm = ramp_1d(HIDDEN, 0.6);
    let attn_q = ramp_2d(HIDDEN, HIDDEN, 2.0);
    let attn_k = ramp_2d(HIDDEN, HIDDEN, 3.0);
    let attn_v = ramp_2d(HIDDEN, HIDDEN, 4.0);
    let attn_output = ramp_2d(HIDDEN, HIDDEN, 5.0);
    let ffn_norm = ramp_1d(HIDDEN, 0.7);

    // The four MoE-specific tensors
    let router = ramp_2d(NUM_EXPERTS, HIDDEN, 8.0); // [E, hidden]
    let gate_exps = ramp_3d(NUM_EXPERTS, EXPERT_FFN, HIDDEN, 9.0); // [E, ffn, hidden]
    let up_exps = ramp_3d(NUM_EXPERTS, EXPERT_FFN, HIDDEN, 10.0); // [E, ffn, hidden]
    let down_exps = ramp_3d(NUM_EXPERTS, HIDDEN, EXPERT_FFN, 11.0); // [E, hidden, ffn]

    let arch_v = Value::String("qwen3moe".to_string());
    let metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch_v)];
    let tensors: Vec<(&str, &QTensor)> = vec![
        ("token_embd.weight", &token_embd),
        ("output_norm.weight", &output_norm),
        ("output.weight", &output),
        ("blk.0.attn_norm.weight", &attn_norm),
        ("blk.0.attn_q.weight", &attn_q),
        ("blk.0.attn_k.weight", &attn_k),
        ("blk.0.attn_v.weight", &attn_v),
        ("blk.0.attn_output.weight", &attn_output),
        ("blk.0.ffn_norm.weight", &ffn_norm),
        ("blk.0.ffn_gate_inp.weight", &router),
        ("blk.0.ffn_gate_exps.weight", &gate_exps),
        ("blk.0.ffn_up_exps.weight", &up_exps),
        ("blk.0.ffn_down_exps.weight", &down_exps),
    ];

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        gguf_file::write(&mut cursor, &metadata, &tensors).unwrap();
    }
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();
    tmp
}

fn build_loader(tmp: &tempfile::NamedTempFile) -> GgufLoader<CpuBackend> {
    let gguf = GgufFile::open(tmp.path()).unwrap();
    GgufLoader::<CpuBackend>::from_file(Arc::new(gguf))
}

#[test]
fn moe_tensors_are_discoverable_by_ferrum_names() {
    let tmp = build_moe_gguf();
    let loader = build_loader(&tmp);

    assert!(loader.has_tensor("model.layers.0.mlp.router.weight"));
    assert!(loader.has_tensor("model.layers.0.mlp.gate_exps.weight"));
    assert!(loader.has_tensor("model.layers.0.mlp.up_exps.weight"));
    assert!(loader.has_tensor("model.layers.0.mlp.down_exps.weight"));

    // HF-style per-expert names should NOT resolve (we deliberately
    // didn't map them — see names.rs design notes).
    assert!(!loader.has_tensor("model.layers.0.mlp.experts.0.gate_proj.weight"));
}

#[test]
fn router_load_tensor_returns_full_buffer() {
    let tmp = build_moe_gguf();
    let loader = build_loader(&tmp);

    let router = loader
        .load_tensor("model.layers.0.mlp.router.weight")
        .unwrap();
    // [num_experts=4, hidden=4] = 16 elements
    assert_eq!(router.len(), NUM_EXPERTS * HIDDEN);

    // Spot-check the ramp: base 8.0 step 0.001, first elem = 8.0
    assert!((router[0] - 8.0).abs() < 1e-6);
    let last_idx = NUM_EXPERTS * HIDDEN - 1;
    let expected_last = 8.0 + (last_idx as f32) * 0.001;
    assert!(
        (router[last_idx] - expected_last).abs() < 1e-5,
        "last router element: got {}, expected {expected_last}",
        router[last_idx]
    );
}

#[test]
fn stacked_expert_tensors_load_at_full_3d_size() {
    let tmp = build_moe_gguf();
    let loader = build_loader(&tmp);

    let gate = loader
        .load_tensor("model.layers.0.mlp.gate_exps.weight")
        .unwrap();
    assert_eq!(gate.len(), NUM_EXPERTS * EXPERT_FFN * HIDDEN);

    let up = loader
        .load_tensor("model.layers.0.mlp.up_exps.weight")
        .unwrap();
    assert_eq!(up.len(), NUM_EXPERTS * EXPERT_FFN * HIDDEN);

    let down = loader
        .load_tensor("model.layers.0.mlp.down_exps.weight")
        .unwrap();
    assert_eq!(down.len(), NUM_EXPERTS * HIDDEN * EXPERT_FFN);

    // The runtime in Phase 2C/2D will slice these per-expert; the loader's
    // job is just to materialise them at full size in row-major order.
    // Spot-check ramp consistency (gate base=9.0):
    assert!((gate[0] - 9.0).abs() < 1e-6);
}

#[test]
fn router_load_linear_succeeds_as_2d() {
    // The router is 2-D so it can go through GgufLinear if a caller wants
    // a `Box<dyn Linear<B>>` (e.g. if the router is reused across layers
    // some day). This isn't the primary path — most MoE runtimes use
    // load_tensor for the router and fold its forward into the dispatch
    // kernel — but supporting it costs nothing and keeps symmetry with
    // dense projections.
    let tmp = build_moe_gguf();
    let loader = build_loader(&tmp);

    let router_linear = loader.load_linear("model.layers.0.mlp.router").unwrap();
    assert_eq!(router_linear.in_features(), HIDDEN);
    assert_eq!(router_linear.out_features(), NUM_EXPERTS);
}

#[test]
fn stacked_expert_load_linear_rejects_rank_3() {
    // Stacked expert tensors are 3-D and must NOT be wrapped in
    // GgufLinear<B> — the rank check catches that and returns Err with
    // a clear message about the dimensionality.
    let tmp = build_moe_gguf();
    let loader = build_loader(&tmp);

    let result = loader.load_linear("model.layers.0.mlp.gate_exps");
    assert!(result.is_err(), "3-D stacked expert tensor must reject");
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("2-D"),
        "error message mentions rank constraint: {err}"
    );
}

#[test]
fn dense_tensors_still_loadable_alongside_moe() {
    // A regression check: adding MoE mappings shouldn't have broken any
    // dense names. The fixture has both attention (dense) and MoE
    // tensors in the same layer; both should resolve.
    let tmp = build_moe_gguf();
    let loader = build_loader(&tmp);

    assert!(loader.has_tensor("model.layers.0.self_attn.q_proj.weight"));
    assert!(loader.has_tensor("model.layers.0.input_layernorm.weight"));
    assert!(loader.has_tensor("model.layers.0.post_attention_layernorm.weight"));
    assert!(loader.has_tensor("model.embed_tokens.weight"));
    assert!(loader.has_tensor("lm_head.weight"));
}
