//! Debug: compare prefill+decode between Candle and ModelRunner.
//! Run: cargo test -p ferrum-models --test debug_prefill_decode_test -- --ignored --nocapture

use candle_core::{DType, Device, Tensor};

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
fn compare_prefill_then_decode() {
    let model_path = qwen3_model_path().unwrap();
    let loader = ferrum_models::SafeTensorsLoader::new(model_path.to_str().unwrap());

    let mut cm = ferrum_models::ConfigManager::new();
    let model_def = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(cm.load_from_path(&model_path))
        .unwrap();

    // Use a simple 3-token prompt
    let prompt_tokens = [1u32, 2, 3];

    // === Candle path ===
    let vb1 = loader.load_varbuilder(&Device::Cpu, DType::F32).unwrap();
    let qwen3 =
        ferrum_models::Qwen3ModelWrapper::from_varbuilder(vb1, &model_def, Device::Cpu, DType::F32)
            .unwrap();

    // Candle prefill: process all 3 tokens
    let input_t = Tensor::new(&prompt_tokens[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap();
    let candle_prefill = qwen3.forward_prefill(&input_t, "test").unwrap();
    let candle_prefill_logits: Vec<f32> = candle_prefill.flatten_all().unwrap().to_vec1().unwrap();

    // Candle prefill may return [1, vocab] (last token only) or [1, seq, vocab]
    let vocab = model_def.vocab_size;
    let candle_last_logits = if candle_prefill_logits.len() == vocab {
        &candle_prefill_logits[..]
    } else {
        let last_offset = candle_prefill_logits.len() - vocab;
        &candle_prefill_logits[last_offset..]
    };
    let c_argmax = candle_last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!(
        "Candle prefill last-token argmax={}, logit={:.4}",
        c_argmax.0, c_argmax.1
    );

    // Candle decode: next token (use argmax from prefill)
    let next_token = c_argmax.0 as u32;
    let decode_input = Tensor::new(&[next_token], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap();
    let candle_decode = qwen3
        .forward_decode(&decode_input, prompt_tokens.len(), "test")
        .unwrap();
    let candle_decode_logits: Vec<f32> = candle_decode.flatten_all().unwrap().to_vec1().unwrap();
    let c_decode_argmax = candle_decode_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!(
        "Candle decode argmax={}, logit={:.4}",
        c_decode_argmax.0, c_decode_argmax.1
    );

    // === ModelRunner path ===
    let cfg = ferrum_models::model_config::qwen3_config(&model_def);
    let vb2 = loader.load_varbuilder(&Device::Cpu, DType::F32).unwrap();
    let weights =
        ferrum_models::model_config::weight_loader::load_model_weights(&vb2, &cfg).unwrap();
    let mut runner = ferrum_kernels::backend::runner::ModelRunner::<
        ferrum_kernels::backend::cpu::CpuBackend,
    >::new(cfg.clone(), weights);

    // Runner prefill
    let runner_prefill_logits = runner.prefill("test", &prompt_tokens);
    let r_argmax = runner_prefill_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!(
        "Runner prefill argmax={}, logit={:.4}",
        r_argmax.0, r_argmax.1
    );

    // Compare prefill
    if c_argmax.0 == r_argmax.0 {
        eprintln!("✅ Prefill argmax MATCH: {}", c_argmax.0);
    } else {
        eprintln!(
            "❌ Prefill argmax MISMATCH: candle={} runner={}",
            c_argmax.0, r_argmax.0
        );
    }

    // Runner decode
    let runner_decode_logits = runner.decode("test", next_token, prompt_tokens.len() as u32);
    let r_decode_argmax = runner_decode_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!(
        "Runner decode argmax={}, logit={:.4}",
        r_decode_argmax.0, r_decode_argmax.1
    );

    if c_decode_argmax.0 == r_decode_argmax.0 {
        eprintln!("✅ Decode argmax MATCH: {}", c_decode_argmax.0);
    } else {
        eprintln!(
            "❌ Decode argmax MISMATCH: candle={} runner={}",
            c_decode_argmax.0, r_decode_argmax.0
        );
        // Cosine
        let mut dot = 0.0f64;
        let mut nc = 0.0f64;
        let mut nr = 0.0f64;
        for (c, r) in candle_decode_logits.iter().zip(&runner_decode_logits) {
            dot += *c as f64 * *r as f64;
            nc += (*c as f64).powi(2);
            nr += (*r as f64).powi(2);
        }
        eprintln!(
            "Decode cosine: {:.6}",
            dot / (nc.sqrt() * nr.sqrt() + 1e-10)
        );
    }
}
