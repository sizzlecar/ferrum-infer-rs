//! `Qwen3MoeModel::new_safetensors` end-to-end smoke test on a real
//! Qwen3-MoE GPTQ-INT4 checkpoint (Qwen3-30B-A3B and friends).
//!
//! Validates the stacked GPTQ Marlin loader path:
//!   1. Load completes without OOM (proxy for: stacked tile per layer
//!      fits in VRAM and Marlin repack succeeds).
//!   2. Prefill produces finite logits (proxy for: layout matches the
//!      offset GEMM expectations).
//!   3. Multi-step decode produces finite logits and advances KV cache.
//!
//! Run on a CUDA pod (4090 / Blackwell):
//!
//!   FERRUM_M3_PATH=/workspace/models/M3 \
//!     cargo test -p ferrum-models --features cuda --release \
//!       --test qwen3_moe_safetensors_smoke -- --ignored --nocapture
//!
//! `FERRUM_M3_PATH` defaults to `/workspace/models/M3`; override for
//! local-cache layouts.

#![cfg(feature = "cuda")]

use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_models::common::DecoderOnlyLLM;
use ferrum_models::models::Qwen3MoeModel;
use ferrum_models::moe_config::Qwen3MoeConfig;
use ferrum_quantization::NativeSafetensorsLoader;

fn m3_path() -> std::path::PathBuf {
    std::env::var("FERRUM_M3_PATH")
        .unwrap_or_else(|_| "/workspace/models/M3".to_string())
        .into()
}

fn load_model() -> Qwen3MoeModel<CudaBackend> {
    let path = m3_path();
    eprintln!("Loading M3 from {}", path.display());

    // CUDA's MoE prefill path defaults to paged_decode_attention which
    // isn't implemented (only paged_varlen_attn is). Force paged-KV off
    // so prefill takes the flash_attention path. Re-enable explicitly
    // when the engine wires the varlen path through MoE forward.
    if std::env::var("FERRUM_METAL_PAGED_KV").is_err() {
        std::env::set_var("FERRUM_METAL_PAGED_KV", "0");
    }

    // ConfigManager::load_from_path is async — spin a tiny runtime to keep
    // the test sync-friendly.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let model_def = rt.block_on(async {
        let mut cm = ferrum_models::ConfigManager::new();
        cm.load_from_path(&path)
            .await
            .expect("ConfigManager::load_from_path")
    });

    let cfg = Qwen3MoeConfig::from_def(&model_def).expect("Qwen3MoeConfig::from_def");
    let loader =
        NativeSafetensorsLoader::<CudaBackend>::open(&path).expect("NativeSafetensorsLoader::open");

    let t0 = std::time::Instant::now();
    let model =
        Qwen3MoeModel::<CudaBackend>::new_safetensors(cfg, &loader).expect("M3 safetensors load");
    let dt = t0.elapsed();
    eprintln!(
        "M3 load complete in {:.2}s ({} layers, {} experts/layer)",
        dt.as_secs_f64(),
        model.cfg.base.num_layers,
        model.cfg.num_experts,
    );
    model
}

#[test]
#[ignore]
fn m3_load_finishes() {
    let _ = load_model();
}

#[test]
#[ignore]
fn m3_prefill_finite_logits() {
    let mut model = load_model();
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    let t0 = std::time::Instant::now();
    let logits = model.prefill("smoke", &tokens);
    let dt = t0.elapsed();
    eprintln!(
        "M3 prefill: {} tokens in {:.2}ms ({} logits)",
        tokens.len(),
        dt.as_secs_f64() * 1000.0,
        logits.len()
    );
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for v in &logits {
        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        }
    }
    assert_eq!(nan_count, 0, "{} NaN logits", nan_count);
    assert_eq!(inf_count, 0, "{} Inf logits", inf_count);
}

#[test]
#[ignore]
fn m3_decode_advances_kv() {
    let mut model = load_model();
    let prompt: Vec<u32> = vec![1, 2, 3];
    let _ = model.prefill("seq1", &prompt);

    // Decode 5 steps; each step's logits must be finite.
    for step in 0..5 {
        let pos = (prompt.len() + step) as u32;
        let logits = model.decode("seq1", 5 + step as u32, pos);
        let bad = logits.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(bad, 0, "step {step}: {bad} non-finite logits");
    }
}
