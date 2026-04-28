//! Phase 1C integration tests: synthesize a GGUF that covers every tensor
//! shape a Llama-family decoder layer asks for, build a `GgufLoader<B>`
//! over it, and exercise the `WeightLoader<B>` API the model actually
//! uses (load_tensor / load_linear / qkv fusion / gate_up fusion / bias).
//!
//! All tests run on `CpuBackend` — no GGUF model file required.

use std::io::{Cursor, Write};
use std::sync::Arc;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_quantization::gguf::{GgufFile, GgufLoader};
use ferrum_quantization::WeightLoader;

/// Build a tensor of given shape filled with a deterministic ramp so we can
/// later verify which sub-tensor ended up where in a fused weight.
fn ramp_tensor(rows: usize, cols: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let n = rows * cols;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.001).collect();
    let t = Tensor::from_vec(raw, (rows, cols), &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

fn vec_tensor(n: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.01).collect();
    let t = Tensor::from_vec(raw, n, &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

/// Build a GGUF tempfile containing the full set of Llama-family layer
/// tensors (one decoder layer + embed/output) — minimal but exercising
/// every name-mapping and fusion path.
fn build_full_layer_gguf() -> tempfile::NamedTempFile {
    // Toy dimensions: hidden=4, head_dim=4 (1 head), ffn=8.
    // q/k/v all 4x4 — combined qkv = 12x4.
    // gate/up both 8x4 — combined gate_up = 16x4.
    let token_embd = ramp_tensor(8, 4, 0.0); // vocab=8, hidden=4
    let output_norm = vec_tensor(4, 0.5);
    let output = ramp_tensor(8, 4, 1.0); // lm_head, vocab=8, hidden=4
    let attn_norm = vec_tensor(4, 0.6);
    let attn_q = ramp_tensor(4, 4, 2.0);
    let attn_k = ramp_tensor(4, 4, 3.0);
    let attn_v = ramp_tensor(4, 4, 4.0);
    let attn_output = ramp_tensor(4, 4, 5.0);
    let ffn_norm = vec_tensor(4, 0.7);
    let ffn_gate = ramp_tensor(8, 4, 6.0);
    let ffn_up = ramp_tensor(8, 4, 7.0);
    let ffn_down = ramp_tensor(4, 8, 8.0);

    let arch_v = Value::String("qwen3".to_string());
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
        ("blk.0.ffn_gate.weight", &ffn_gate),
        ("blk.0.ffn_up.weight", &ffn_up),
        ("blk.0.ffn_down.weight", &ffn_down),
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
fn has_tensor_translates_ferrum_names() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    assert!(loader.has_tensor("model.embed_tokens.weight"));
    assert!(loader.has_tensor("model.norm.weight"));
    assert!(loader.has_tensor("lm_head.weight"));
    assert!(loader.has_tensor("model.layers.0.input_layernorm.weight"));
    assert!(loader.has_tensor("model.layers.0.self_attn.q_proj.weight"));
    assert!(loader.has_tensor("model.layers.0.mlp.down_proj.weight"));

    // Unknown ferrum name → false (no GGUF mapping)
    assert!(!loader.has_tensor("totally_made_up.weight"));
    // Mapped, but not in this file (no second layer)
    assert!(!loader.has_tensor("model.layers.1.self_attn.q_proj.weight"));
}

#[test]
fn load_tensor_returns_dequantized_values() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    // attn_norm was vec_tensor(4, 0.6) → [0.60, 0.61, 0.62, 0.63]
    let raw = loader
        .load_tensor("model.layers.0.input_layernorm.weight")
        .unwrap();
    assert_eq!(raw.len(), 4);
    let expected: Vec<f32> = (0..4).map(|i| 0.6 + (i as f32) * 0.01).collect();
    assert_eq!(raw, expected);

    // embed table 8x4 = 32 elems, base 0.0 step 0.001
    let embed = loader.load_tensor("model.embed_tokens.weight").unwrap();
    assert_eq!(embed.len(), 32);
    assert!((embed[0] - 0.0).abs() < 1e-6);
    assert!((embed[31] - 0.031).abs() < 1e-5);
}

#[test]
fn load_tensor_unknown_name_errors() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let err = loader
        .load_tensor("model.layers.99.self_attn.q_proj.weight")
        .unwrap_err()
        .to_string();
    assert!(err.contains("not present"), "error mentions missing: {err}");

    let err = loader
        .load_tensor("nonsense.thing")
        .unwrap_err()
        .to_string();
    assert!(err.contains("unrecognised") || err.contains("no GGUF mapping"));
}

#[test]
fn load_linear_direct_path() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let o = loader
        .load_linear("model.layers.0.self_attn.o_proj")
        .unwrap();
    assert_eq!(o.in_features(), 4);
    assert_eq!(o.out_features(), 4);

    // lm_head is 8x4 (vocab=8, hidden=4)
    let lm = loader.load_linear("lm_head").unwrap();
    assert_eq!(lm.in_features(), 4);
    assert_eq!(lm.out_features(), 8);

    let down = loader.load_linear("model.layers.0.mlp.down_proj").unwrap();
    assert_eq!(down.in_features(), 8);
    assert_eq!(down.out_features(), 4);
}

#[test]
fn load_linear_fuses_qkv() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let qkv = loader
        .load_linear("model.layers.0.self_attn.qkv_proj")
        .unwrap();
    // Each part is 4x4; fused = 12x4
    assert_eq!(qkv.in_features(), 4);
    assert_eq!(qkv.out_features(), 12);

    // Forward verification: feed [1,1,1,1] (all-ones), expect each output
    // row = sum of that row's weights. Construct expected from the ramp.
    use ferrum_kernels::backend::Backend;
    let input: Vec<f32> = vec![1.0; 4];
    let mut out: Vec<f32> = vec![0.0; 12];
    let mut ctx = <CpuBackend as Backend>::new_context();
    qkv.forward(&mut ctx, &input, &mut out, 1);

    // q ramp base 2.0, k base 3.0, v base 4.0; each row of 4 has step 0.001.
    // row r within q (r=0..3) sums to 4*(2.0 + 0.001*(4*r)) + 0.001*(0+1+2+3) =
    //   for ramp(rows=4, cols=4, base=B), row r sum = 4*B + 0.001*(4*r*4 + 0+1+2+3)
    // Simpler: just compute expected via the same ramp logic.
    let mut expected = Vec::with_capacity(12);
    for &(rows_so_far, base) in &[(0usize, 2.0_f32), (4, 3.0), (8, 4.0)] {
        for r in 0..4 {
            // row r in this ramp_tensor(4,4,base): elements = base + 0.001*(r*4 + c) for c in 0..4
            let row_sum: f32 = (0..4)
                .map(|c| base + 0.001 * ((r * 4 + c) as f32))
                .sum::<f32>();
            expected.push(row_sum);
            let _ = rows_so_far; // unused; just here to clarify ordering
        }
    }
    for (i, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "qkv row {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn load_linear_fuses_gate_up() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let gu = loader
        .load_linear("model.layers.0.mlp.gate_up_proj")
        .unwrap();
    // gate (8x4) + up (8x4) → 16x4
    assert_eq!(gu.in_features(), 4);
    assert_eq!(gu.out_features(), 16);
}

#[test]
fn load_linear_unknown_returns_err() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let result = loader.load_linear("model.layers.0.unknown_proj");
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("could not load Linear"),
        "error explains failure: {err}"
    );
}

#[test]
fn open_path_constructor_works() {
    let tmp = build_full_layer_gguf();
    let loader = GgufLoader::<CpuBackend>::open(tmp.path()).unwrap();
    assert!(loader.has_tensor("model.embed_tokens.weight"));
    assert_eq!(loader.gguf().architecture().unwrap(), "qwen3");
}

#[test]
fn quant_config_returns_none_in_phase_1c() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);
    assert!(loader.quant_config().is_none());
}
