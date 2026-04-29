//! `Qwen3MoeLayer::load_from_gguf` + `forward_cpu` end-to-end:
//! synthesise a complete one-layer Qwen3-MoE GGUF (router + three
//! stacked-expert tensors + arch metadata) and verify a single forward
//! pass produces finite output of the right shape.

use std::io::{Cursor, Write};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_models::moe::Qwen3MoeLayer;
use ferrum_models::moe_config::Qwen3MoeConfig;
use ferrum_quantization::gguf::GgufFile;

fn ramp_3d(d0: usize, d1: usize, d2: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let n = d0 * d1 * d2;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.001).collect();
    let t = Tensor::from_vec(raw, (d0, d1, d2), &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

fn ramp_2d(rows: usize, cols: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let n = rows * cols;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.001).collect();
    let t = Tensor::from_vec(raw, (rows, cols), &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

/// Build a single-layer Qwen3-MoE GGUF with the four MoE-specific tensors
/// (router + gate_exps + up_exps + down_exps). Returns the tempfile.
fn build_one_layer_moe_gguf(
    n_experts: usize,
    hidden: usize,
    ffn: usize,
) -> tempfile::NamedTempFile {
    // Router weights (`[num_experts, hidden]`).
    let router = ramp_2d(n_experts, hidden, 0.05);
    // Stacked experts.
    let gate_exps = ramp_3d(n_experts, ffn, hidden, 0.1);
    let up_exps = ramp_3d(n_experts, ffn, hidden, 0.2);
    let down_exps = ramp_3d(n_experts, hidden, ffn, 0.3);

    let arch_v = Value::String("qwen3moe".to_string());
    let metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch_v)];
    let tensors: Vec<(&str, &QTensor)> = vec![
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

/// Toy MoE config matching the GGUF shapes above.
fn toy_config(n_experts: usize, hidden: usize, ffn: usize, top_k: usize) -> Qwen3MoeConfig {
    use ferrum_models::models::llama_family::LlamaFamilyConfig;
    let base = LlamaFamilyConfig {
        hidden_size: hidden,
        intermediate_size: ffn, // mirrored from per-expert
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        num_layers: 1,
        vocab_size: 8,
        max_seq_len: 32,
        rms_norm_eps: 1.0e-6,
        rope_theta: 1.0e6,
        has_qk_norm: true,
        sliding_window: 0,
    };
    Qwen3MoeConfig::from_base(base, n_experts, top_k, ffn, true)
}

#[test]
fn loads_layer_from_synthesized_moe_gguf() {
    let n_experts = 4;
    let hidden = 4;
    let ffn = 8;
    let cfg = toy_config(n_experts, hidden, ffn, 2);
    let tmp = build_one_layer_moe_gguf(n_experts, hidden, ffn);
    let gguf = GgufFile::open(tmp.path()).unwrap();

    let layer = Qwen3MoeLayer::<CpuBackend>::load_from_gguf(&gguf, 0, &cfg).unwrap();

    // Router shape sanity
    assert_eq!(layer.router.in_features(), hidden);
    assert_eq!(layer.router.out_features(), n_experts);
    // Expert stack
    assert_eq!(layer.num_experts, n_experts);
    assert_eq!(layer.experts.num_experts(), n_experts);
    // Routing config carried through
    assert_eq!(layer.top_k, 2);
    assert!(layer.norm_topk_prob);
}

#[test]
fn forward_cpu_produces_finite_output_of_correct_shape() {
    let n_experts = 4;
    let hidden = 4;
    let ffn = 6;
    let top_k = 2;
    let cfg = toy_config(n_experts, hidden, ffn, top_k);
    let tmp = build_one_layer_moe_gguf(n_experts, hidden, ffn);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let layer = Qwen3MoeLayer::<CpuBackend>::load_from_gguf(&gguf, 0, &cfg).unwrap();

    // Two tokens, deliberate non-trivial inputs.
    let x: Vec<f32> = vec![0.5, -0.25, 0.1, 0.0, 0.7, 0.3, -0.4, 0.2];
    let mut out = Vec::new();
    layer.forward_cpu(&x, 2, &mut out).unwrap();

    assert_eq!(out.len(), 2 * hidden);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "out[{i}] = {v} is not finite");
    }
}

#[test]
fn forward_cpu_rejects_wrong_input_size() {
    let n_experts = 4;
    let hidden = 4;
    let ffn = 6;
    let cfg = toy_config(n_experts, hidden, ffn, 2);
    let tmp = build_one_layer_moe_gguf(n_experts, hidden, ffn);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let layer = Qwen3MoeLayer::<CpuBackend>::load_from_gguf(&gguf, 0, &cfg).unwrap();

    let x = vec![0.0_f32; 7]; // wrong size
    let mut out = Vec::new();
    let result = layer.forward_cpu(&x, 2, &mut out);
    assert!(result.is_err());
}

#[test]
fn missing_router_tensor_returns_clear_error() {
    // Synthesize a GGUF that has only the expert tensors but no router.
    let n_experts = 4;
    let hidden = 4;
    let ffn = 6;
    let gate_exps = ramp_3d(n_experts, ffn, hidden, 0.1);
    let up_exps = ramp_3d(n_experts, ffn, hidden, 0.2);
    let down_exps = ramp_3d(n_experts, hidden, ffn, 0.3);

    let arch_v = Value::String("qwen3moe".to_string());
    let metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch_v)];
    let tensors: Vec<(&str, &QTensor)> = vec![
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

    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = toy_config(n_experts, hidden, ffn, 2);
    let result = Qwen3MoeLayer::<CpuBackend>::load_from_gguf(&gguf, 0, &cfg);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("router") && err.contains("ffn_gate_inp"),
        "error mentions router tensor: {err}"
    );
}

#[test]
fn config_dimension_mismatch_is_caught() {
    // Build a GGUF with hidden=4, but pass a config that thinks hidden=8.
    let n_experts = 2;
    let gguf_hidden = 4;
    let cfg_hidden = 8; // wrong on purpose
    let ffn = 4;

    let tmp = build_one_layer_moe_gguf(n_experts, gguf_hidden, ffn);
    let gguf = GgufFile::open(tmp.path()).unwrap();

    // Router in the GGUF has shape [num_experts=2, hidden=4]. Config
    // says hidden=8. Construction should fail at the in_features check.
    let cfg = toy_config(n_experts, cfg_hidden, ffn, 1);
    let result = Qwen3MoeLayer::<CpuBackend>::load_from_gguf(&gguf, 0, &cfg);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    // Could fire from either the ExpertStack size check or the router
    // in_features check — both are valid evidence of "config doesn't
    // match GGUF shapes".
    assert!(
        err.contains("mismatch") || err.contains("in_features"),
        "expected dimension mismatch error, got: {err}"
    );
}

#[test]
fn top_k_one_with_strong_router_picks_dominant_expert() {
    // Make the router strongly favour a particular expert, top_k=1, and
    // verify forward output equals that expert's standalone MLP output.
    let n_experts = 2;
    let hidden = 2;
    let ffn = 2;

    // Router: [num_experts, hidden]. To force expert 1 to win for input
    // x=[1, 0], make row 0 produce -10 and row 1 produce +10 logit.
    // Row 0 = [logit_of_expert_0_for_basis_0, ...]. Skip — easier to
    // hand-craft after we know how routing computes:
    //   logits = router_weight @ x  (shape [num_experts])
    // For x=[1,0]: logits = [router[0,0], router[1,0]]
    // Want logits = [-10, +10]: router_weight = [[-10, 0], [10, 0]]
    let router_q = QTensor::quantize(
        &Tensor::from_vec(
            vec![-10.0_f32, 0.0, 10.0, 0.0],
            (n_experts, hidden),
            &Device::Cpu,
        )
        .unwrap(),
        GgmlDType::F32,
    )
    .unwrap();

    // Expert 0: identity * 0 (zero output). Expert 1: identity * 7 (out = 7*silu(x)*x).
    // Use ramp_2d_const for the gate/up/down stacks but this is a 3-D
    // stacked tensor — easier path: hand-build flat data and pack as 3-D.
    let device = Device::Cpu;
    // gate_exps shape [E=2, ffn=2, hidden=2]
    // Expert 0 gate: zeros. Expert 1 gate: 7 * I_2.
    let mut gate_data = vec![0.0_f32; n_experts * ffn * hidden];
    // Expert 1 gate (offset = 1 * ffn * hidden = 4): identity*7
    // Index layout intentional: `[expert_offset + ffn_idx * hidden + hidden_idx]`.
    // The first row uses ffn_idx=0, hidden_idx=0 → offset 0; second uses (1,1) → 3.
    gate_data[4] = 7.0; // [expert=1, ffn=0, hidden=0]
    gate_data[4 + 1 * 2 + 1] = 7.0; // [expert=1, ffn=1, hidden=1]
    let gate_t = Tensor::from_vec(gate_data, (n_experts, ffn, hidden), &device).unwrap();
    let gate_qt = QTensor::quantize(&gate_t, GgmlDType::F32).unwrap();

    // Same for up_exps
    let mut up_data = vec![0.0_f32; n_experts * ffn * hidden];
    up_data[4] = 7.0;
    up_data[4 + 1 * 2 + 1] = 7.0;
    let up_t = Tensor::from_vec(up_data, (n_experts, ffn, hidden), &device).unwrap();
    let up_qt = QTensor::quantize(&up_t, GgmlDType::F32).unwrap();

    // down_exps shape [E=2, hidden=2, ffn=2]
    let mut down_data = vec![0.0_f32; n_experts * hidden * ffn];
    down_data[4] = 1.0; // expert 1 down identity, [expert=1, hidden=0, ffn=0]
    down_data[4 + 1 * 2 + 1] = 1.0;
    let down_t = Tensor::from_vec(down_data, (n_experts, hidden, ffn), &device).unwrap();
    let down_qt = QTensor::quantize(&down_t, GgmlDType::F32).unwrap();

    // Build GGUF
    let arch_v = Value::String("qwen3moe".to_string());
    let metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch_v)];
    let tensors: Vec<(&str, &QTensor)> = vec![
        ("blk.0.ffn_gate_inp.weight", &router_q),
        ("blk.0.ffn_gate_exps.weight", &gate_qt),
        ("blk.0.ffn_up_exps.weight", &up_qt),
        ("blk.0.ffn_down_exps.weight", &down_qt),
    ];
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        gguf_file::write(&mut cursor, &metadata, &tensors).unwrap();
    }
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();

    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = toy_config(n_experts, hidden, ffn, 1); // top_k = 1
    let layer = Qwen3MoeLayer::<CpuBackend>::load_from_gguf(&gguf, 0, &cfg).unwrap();

    let x = vec![1.0_f32, 0.0];
    let mut out = Vec::new();
    layer.forward_cpu(&x, 1, &mut out).unwrap();

    // Router with x=[1,0] → logits = [-10, +10], softmax → ~[0, 1]
    // top_k=1 → expert 1 picked with weight 1.0
    // Expert 1: gate=[7*1, 7*0]=[7, 0], up=[7, 0]
    // silu(7)*7 ≈ 6.9936 * 7 = 48.95 (silu(7) = 7 * sigmoid(7) ≈ 7 * 0.99909)
    // silu*up = [48.95, 0]
    // down identity → [48.95, 0]
    let silu_7 = 7.0_f32 * (1.0 / (1.0 + (-7.0_f32).exp()));
    let expected_0 = silu_7 * 7.0;
    assert!(
        (out[0] - expected_0).abs() < 0.01,
        "out[0]: expected {expected_0}, got {}",
        out[0]
    );
    assert!(out[1].abs() < 1e-3, "out[1] should be ~0, got {}", out[1]);
}
