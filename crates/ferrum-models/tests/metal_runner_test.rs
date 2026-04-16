//! Metal ModelRunner vs CPU reference — real model weights.
//! Run: cargo test -p ferrum-models --features metal --test metal_runner_test -- --ignored --nocapture
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

#[test]
#[ignore]
fn metal_vs_cpu_decode() {
    let mp = qwen3_path().expect("Qwen3-0.6B not found");
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

    eprintln!("Running CPU decode...");
    let cpu_logits = cpu_r.decode("t", 1, 0);
    eprintln!("Running Metal decode...");
    let mtl_logits = mtl_r.decode("t", 1, 0);

    let ca = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let ma = mtl_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("CPU argmax={} logit={:.4}", ca.0, ca.1);
    eprintln!("MTL argmax={} logit={:.4}", ma.0, ma.1);

    let mut dot = 0.0f64;
    let mut nc = 0.0f64;
    let mut nr = 0.0f64;
    for (c, r) in cpu_logits.iter().zip(&mtl_logits) {
        dot += *c as f64 * *r as f64;
        nc += (*c as f64).powi(2);
        nr += (*r as f64).powi(2);
    }
    let cos = dot / (nc.sqrt() * nr.sqrt() + 1e-10);
    eprintln!("Cosine: {cos:.6}");

    if ca.0 == ma.0 {
        eprintln!("✅ argmax match");
    } else {
        eprintln!("❌ argmax mismatch");
    }
    assert!(cos > 0.99, "Metal vs CPU cosine too low: {cos}");
}
