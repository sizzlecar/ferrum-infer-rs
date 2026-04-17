//! Metal backend parity — multi-step regression guard.
//!
//! Runs real Qwen3-0.6B weights through ModelRunner<CpuBackend> and
//! ModelRunner<MetalBackend>, comparing logits at every step (prefill + 5 decodes).
//! Must stay green across MetalBackend refactors.
//!
//! Run: cargo test -p ferrum-models --features metal --release --test metal_parity_test -- --ignored --nocapture

#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::metal::MetalBackend;
use ferrum_kernels::backend::runner::{convert_weights_to_metal, ModelRunner};

fn qwen3_path() -> Option<std::path::PathBuf> {
    let p = dirs::home_dir()?.join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    std::fs::read_dir(&p).ok()?.find_map(|e| {
        let e = e.ok()?;
        e.path().join("config.json").exists().then(|| e.path())
    })
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

fn argmax(v: &[f32]) -> (usize, f32) {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Prefill + 5-step decode — catches accumulating drift across KV cache.
#[test]
#[ignore]
fn prefill_decode_parity() {
    let mp = qwen3_path().expect("Qwen3-0.6B not in HF cache");
    let cj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(mp.join("config.json")).unwrap()).unwrap();
    let def = ferrum_models::definition::ModelDefinition {
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
        ..Default::default()
    };

    let cfg = ferrum_models::model_config::qwen3_config(&def);
    let loader = ferrum_models::SafeTensorsLoader::new(mp.to_str().unwrap());
    let vb = loader
        .load_varbuilder(&candle_core::Device::Cpu, candle_core::DType::F32)
        .unwrap();
    let cpu_w = ferrum_models::model_config::weight_loader::load_model_weights(&vb, &cfg).unwrap();
    let metal_w = convert_weights_to_metal(&cpu_w);

    let mut cpu_r = ModelRunner::<CpuBackend>::new(cfg.clone(), cpu_w);
    let mut mtl_r = ModelRunner::<MetalBackend>::new(cfg, metal_w);

    // ── Prefill ────────────────────────────────────────────────────────
    let prompt: Vec<u32> = vec![872, 111, 248, 104715, 0, 56568, 53481, 5048]; // "你好!有什么"-ish tokens
    eprintln!("\n=== Prefill {} tokens ===", prompt.len());

    let cpu_logits = cpu_r.prefill("t", &prompt);
    let mtl_logits = mtl_r.prefill("t", &prompt);

    let (ca, cv) = argmax(&cpu_logits);
    let (ma, mv) = argmax(&mtl_logits);
    let cos = cosine(&cpu_logits, &mtl_logits);
    let mad = max_abs_diff(&cpu_logits, &mtl_logits);
    eprintln!(
        "prefill  CPU argmax={ca} ({cv:.4})  Metal argmax={ma} ({mv:.4})  cos={cos:.6}  max_diff={mad:.4}",
    );
    assert_eq!(ca, ma, "prefill argmax mismatch");
    assert!(cos > 0.9999, "prefill cosine too low: {cos}");

    // ── Decode 5 steps, chain on CPU argmax for determinism ────────────
    let mut pos = prompt.len() as u32;
    let mut tok = ca as u32;
    for step in 0..5 {
        let cpu_logits = cpu_r.decode("t", tok, pos);
        let mtl_logits = mtl_r.decode("t", tok, pos);
        let (ca, cv) = argmax(&cpu_logits);
        let (ma, mv) = argmax(&mtl_logits);
        let cos = cosine(&cpu_logits, &mtl_logits);
        let mad = max_abs_diff(&cpu_logits, &mtl_logits);
        eprintln!(
            "decode {step} pos={pos} tok={tok}  CPU argmax={ca} ({cv:.4})  Metal argmax={ma} ({mv:.4})  cos={cos:.6}  max_diff={mad:.4}",
        );
        assert_eq!(ca, ma, "decode step {step} argmax mismatch");
        assert!(cos > 0.9999, "decode step {step} cosine too low: {cos}");
        tok = ca as u32;
        pos += 1;
    }
    eprintln!("✅ prefill + 5 decode steps parity pass");
}
