//! `LlamaFamilyModel<MetalF16Backend>` correctness vs CPU/f32 baseline.
//!
//! Uses the real `NativeSafetensorsLoader` so weights actually flow through
//! `Backend::from_weight_bytes` — the whole point of MetalF16Backend is that
//! path stores weights directly as fp16 without a transient f32 Vec.
//!
//! Tolerance is looser than `qwen3_model_parity_test` because f16 storage
//! sacrifices ~13 bits of weight mantissa. Argmax must still agree for a
//! simple prompt; logits correlation should stay >= 0.995.
//!
//! Run: cargo test -p ferrum-models --features metal --release \
//!          --test qwen3_f16_parity_test -- --ignored --nocapture

#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::metal_f16::MetalF16Backend;
use ferrum_models::common::DecoderOnlyLLM;
use ferrum_models::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};
use ferrum_quantization::NativeSafetensorsLoader;

fn qwen3_path() -> Option<std::path::PathBuf> {
    let p = dirs::home_dir()?.join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    std::fs::read_dir(&p).ok()?.find_map(|e| {
        let e = e.ok()?;
        e.path().join("config.json").exists().then(|| e.path())
    })
}

fn load_model_def(mp: &std::path::Path) -> ferrum_models::definition::ModelDefinition {
    let cj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(mp.join("config.json")).unwrap()).unwrap();
    ferrum_models::definition::ModelDefinition {
        architecture: ferrum_models::registry::Architecture::Qwen3,
        hidden_size: cj["hidden_size"].as_u64().unwrap() as usize,
        intermediate_size: cj["intermediate_size"].as_u64().unwrap() as usize,
        vocab_size: cj["vocab_size"].as_u64().unwrap() as usize,
        num_hidden_layers: cj["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: cj["num_attention_heads"].as_u64().unwrap() as usize,
        num_key_value_heads: cj
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        max_position_embeddings: cj["max_position_embeddings"].as_u64().unwrap() as usize,
        rope_theta: cj.get("rope_theta").and_then(|v| v.as_f64()),
        extra_params: cj.clone(),
        ..Default::default()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-10)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
#[ignore]
fn qwen3model_cpu_vs_metalf16() {
    let mp = qwen3_path().expect("Qwen3-0.6B not in HF cache");
    let def = load_model_def(&mp);
    let qcfg = LlamaFamilyConfig::qwen3_from_def(&def);

    let cpu_loader = NativeSafetensorsLoader::<CpuBackend>::open(&mp).unwrap();
    let mut cpu_model = LlamaFamilyModel::<CpuBackend>::new(qcfg.clone(), &cpu_loader).unwrap();
    drop(cpu_loader);

    // Probe the size-threshold dispatch:
    //   - tiny (< 1 M elems) → stays F32 so norm/bias shaders still see f32
    //   - big  (≥ 1 M elems) → promoted to F16 for memory savings
    {
        use ferrum_kernels::backend::{Backend, SrcDtype};
        let small = MetalF16Backend::from_weight_bytes(&vec![0u8; 8], SrcDtype::BF16);
        assert!(!small.is_f16(), "tiny probe should stay F32 (under threshold)");
        let big_bytes = vec![0u8; 2 * 1_048_576 * 2]; // 2 M bf16 elements
        let big = MetalF16Backend::from_weight_bytes(&big_bytes, SrcDtype::BF16);
        assert!(big.is_f16(), "big probe should promote to F16");
        eprintln!(
            "✔ size-threshold dispatch: tiny→F32, big→F16 (big.is_f16={})",
            big.is_f16()
        );
    }

    let f16_loader = NativeSafetensorsLoader::<MetalF16Backend>::open(&mp).unwrap();
    let mut f16_model =
        LlamaFamilyModel::<MetalF16Backend>::new(qcfg, &f16_loader).unwrap();
    drop(f16_loader);

    let prompt: Vec<u32> = vec![872, 111, 248, 104715, 0, 56568, 53481, 5048];
    eprintln!("\n=== Prefill {} tokens ===", prompt.len());

    let c_logits = cpu_model.prefill("t", &prompt);
    let m_logits = f16_model.prefill("t", &prompt);
    let (ca, ma) = (argmax(&c_logits), argmax(&m_logits));
    let cos = cosine(&c_logits, &m_logits);
    let mad = max_abs_diff(&c_logits, &m_logits);
    eprintln!(
        "prefill  CPU argmax={ca} ({:.4})  F16 argmax={ma} ({:.4})  cos={cos:.6}  max_diff={mad:.4}",
        c_logits[ca], m_logits[ma]
    );
    assert_eq!(ca, ma, "prefill argmax mismatch");
    assert!(
        cos > 0.995,
        "prefill cosine too low (f16 tol): {cos}"
    );

    let mut pos = prompt.len() as u32;
    let mut tok = ca as u32;
    for step in 0..5 {
        let c = cpu_model.decode("t", tok, pos);
        let m = f16_model.decode("t", tok, pos);
        let (ca, ma) = (argmax(&c), argmax(&m));
        let cos = cosine(&c, &m);
        let mad = max_abs_diff(&c, &m);
        eprintln!(
            "decode {step} pos={pos} tok={tok}  CPU argmax={ca}  F16 argmax={ma}  cos={cos:.6}  max_diff={mad:.4}",
        );
        assert_eq!(ca, ma, "decode step {step} argmax mismatch");
        assert!(cos > 0.99, "decode step {step} cosine too low (f16 tol): {cos}");
        tok = ca as u32;
        pos += 1;
    }
    eprintln!("✅ LlamaFamilyModel CpuF32 ↔ MetalF16 parity pass");
}
