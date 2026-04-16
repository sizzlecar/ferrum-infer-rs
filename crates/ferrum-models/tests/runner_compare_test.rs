//! Compare ModelRunner output vs Candle (old path) for the same input.
//! Requires Qwen3-0.6B downloaded.
//!
//! Run: cargo test -p ferrum-models --test runner_compare_test -- --ignored --nocapture

use candle_core::{DType, Device as CandleDevice, Tensor};

fn qwen3_model_path() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let path = home.join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    for entry in std::fs::read_dir(&path).ok()? {
        let entry = entry.ok()?;
        if entry.path().join("config.json").exists() {
            return Some(entry.path());
        }
    }
    None
}

#[test]
#[ignore]
fn compare_runner_vs_candle() {
    let model_path = match qwen3_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Model not found");
            return;
        }
    };

    let loader = ferrum_models::SafeTensorsLoader::new(model_path.to_str().unwrap());

    // --- Old path: Candle Qwen3ModelWrapper ---
    let mut config_manager = ferrum_models::ConfigManager::new();
    let model_def = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(config_manager.load_from_path(&model_path))
        .unwrap();

    let vb_candle = loader
        .load_varbuilder(&CandleDevice::Cpu, DType::F32)
        .unwrap();
    let qwen3 = ferrum_models::Qwen3ModelWrapper::from_varbuilder(
        vb_candle,
        &model_def,
        CandleDevice::Cpu,
        DType::F32,
    )
    .unwrap();

    // Forward token_id=1 at pos=0
    let input = Tensor::new(&[1u32], &CandleDevice::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap();
    let candle_logits = qwen3.forward_decode(&input, 0, "compare-test").unwrap();
    let candle_logits_flat = candle_logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap();
    let candle_logits_vec: Vec<f32> = candle_logits_flat.to_vec1().unwrap();

    let candle_argmax = candle_logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    eprintln!(
        "Candle: argmax={}, max_logit={:.4}",
        candle_argmax.0, candle_argmax.1
    );

    // --- New path: ModelRunner ---
    let vb_runner = loader
        .load_varbuilder(&CandleDevice::Cpu, DType::F32)
        .unwrap();
    let cfg = ferrum_models::model_config::qwen3_config(&model_def);
    let weights =
        ferrum_models::model_config::weight_loader::load_model_weights(&vb_runner, &cfg).unwrap();

    let mut runner = ferrum_kernels::backend::runner::ModelRunner::<
        ferrum_kernels::backend::cpu::CpuBackend,
    >::new(cfg.clone(), weights);

    let runner_logits = runner.decode("test", 1, 0);

    let runner_argmax = runner_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    eprintln!(
        "Runner: argmax={}, max_logit={:.4}",
        runner_argmax.0, runner_argmax.1
    );

    // Compare top-10 tokens
    let mut candle_top: Vec<(usize, f32)> = candle_logits_vec.iter().copied().enumerate().collect();
    candle_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut runner_top: Vec<(usize, f32)> = runner_logits.iter().copied().enumerate().collect();
    runner_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\nCandle top-5:");
    for (id, logit) in &candle_top[..5] {
        eprintln!("  token={id}, logit={logit:.4}");
    }
    eprintln!("\nRunner top-5:");
    for (id, logit) in &runner_top[..5] {
        eprintln!("  token={id}, logit={logit:.4}");
    }

    // Check if argmax matches
    if candle_argmax.0 == runner_argmax.0 {
        eprintln!("\n✅ argmax MATCH: {}", candle_argmax.0);
    } else {
        eprintln!(
            "\n❌ argmax MISMATCH: candle={} vs runner={}",
            candle_argmax.0, runner_argmax.0
        );
    }

    // Compute L2 distance and cosine similarity for diagnostics
    let mut dot = 0.0f64;
    let mut norm_c = 0.0f64;
    let mut norm_r = 0.0f64;
    for (c, r) in candle_logits_vec.iter().zip(&runner_logits) {
        dot += *c as f64 * *r as f64;
        norm_c += (*c as f64) * (*c as f64);
        norm_r += (*r as f64) * (*r as f64);
    }
    let cosine = dot / (norm_c.sqrt() * norm_r.sqrt() + 1e-10);
    eprintln!("Cosine similarity: {cosine:.6}");
    eprintln!(
        "Candle logit range: [{:.4}, {:.4}]",
        candle_logits_vec
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        candle_logits_vec
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    eprintln!(
        "Runner logit range: [{:.4}, {:.4}]",
        runner_logits.iter().cloned().fold(f32::INFINITY, f32::min),
        runner_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
}
