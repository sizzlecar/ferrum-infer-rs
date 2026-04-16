//! Debug: compare layer 0 intermediate values between Candle and ModelRunner.
//! Run: cargo test -p ferrum-models --test debug_layer0_test -- --ignored --nocapture

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
fn debug_embedding_and_norm() {
    let model_path = qwen3_model_path().unwrap();
    let loader = ferrum_models::SafeTensorsLoader::new(model_path.to_str().unwrap());
    let vb = loader.load_varbuilder(&Device::Cpu, DType::F32).unwrap();

    // Token ID = 1
    let token_id = 1u32;

    // === Candle: get embedding for token 1 ===
    let embed_w = vb
        .get_unchecked("model.embed_tokens.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap(); // [vocab, hidden]
    let embed_candle = embed_w.get(token_id as usize).unwrap(); // [hidden]
    let embed_vec: Vec<f32> = embed_candle.to_vec1().unwrap();
    eprintln!("Candle embed[0..8]: {:?}", &embed_vec[..8]);

    // === Runner: get embedding ===
    let mut config_manager = ferrum_models::ConfigManager::new();
    let model_def = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(config_manager.load_from_path(&model_path))
        .unwrap();
    let cfg = ferrum_models::model_config::qwen3_config(&model_def);

    let vb2 = loader.load_varbuilder(&Device::Cpu, DType::F32).unwrap();
    let weights =
        ferrum_models::model_config::weight_loader::load_model_weights(&vb2, &cfg).unwrap();

    // Runner embedding: weights.embed is flat [vocab * hidden]
    let h = cfg.hidden_size;
    let runner_embed = &weights.embed[token_id as usize * h..(token_id as usize + 1) * h];
    eprintln!("Runner embed[0..8]: {:?}", &runner_embed[..8]);

    // Compare
    let max_diff: f32 = embed_vec
        .iter()
        .zip(runner_embed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Embedding max diff: {max_diff}");
    assert!(max_diff < 1e-6, "Embeddings should match exactly");

    // === Layer 0 input norm ===
    let norm_w = vb
        .get_unchecked("model.layers.0.input_layernorm.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let norm_w_vec: Vec<f32> = norm_w.to_vec1().unwrap();

    // Candle RMS norm
    let eps = 1e-6f32;
    let mut sum_sq = 0.0f32;
    for &v in &embed_vec {
        sum_sq += v * v;
    }
    let rms = (sum_sq / h as f32 + eps).sqrt();
    let candle_normed: Vec<f32> = embed_vec
        .iter()
        .zip(&norm_w_vec)
        .map(|(x, w)| x / rms * w)
        .collect();
    eprintln!("Candle norm[0..8]: {:?}", &candle_normed[..8]);

    // Runner RMS norm
    let mut runner_normed = vec![0.0f32; h];
    ferrum_kernels::backend::cpu::CpuBackend::rms_norm(
        &runner_embed.to_vec(),
        &weights.layers[0].input_ln_w,
        cfg.rms_norm_eps,
        &mut runner_normed,
        1,
        h,
    );
    eprintln!("Runner norm[0..8]: {:?}", &runner_normed[..8]);

    let norm_diff: f32 = candle_normed
        .iter()
        .zip(&runner_normed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Norm max diff: {norm_diff}");

    // === QKV projection ===
    // Candle: separate q_proj, k_proj, v_proj
    let q_w = vb
        .get_unchecked("model.layers.0.self_attn.q_proj.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let candle_q = embed_candle
        .unsqueeze(0)
        .unwrap()
        .matmul(&q_w.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let candle_q_vec: Vec<f32> = candle_q.to_vec1().unwrap();

    // Wait - Candle does matmul on normed input, not raw embedding
    let normed_tensor = Tensor::new(&candle_normed[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let candle_q2 = normed_tensor
        .matmul(&q_w.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let candle_q2_vec: Vec<f32> = candle_q2.to_vec1().unwrap();
    eprintln!("Candle Q[0..8] (from normed): {:?}", &candle_q2_vec[..8]);

    // Runner QKV
    use ferrum_kernels::backend::Backend;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let qkv_dim = q_dim + 2 * kv_dim;
    let mut qkv_out = vec![0.0f32; qkv_dim];
    ferrum_kernels::backend::cpu::CpuBackend::gemm(
        &runner_normed,
        &weights.layers[0].qkv_proj_w,
        &mut qkv_out,
        1,
        qkv_dim,
        h,
    );
    eprintln!("Runner Q[0..8] (from QKV): {:?}", &qkv_out[..8]);

    let q_diff: f32 = candle_q2_vec
        .iter()
        .zip(&qkv_out[..q_dim])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Q projection max diff: {q_diff}");
    if q_diff < 0.01 {
        eprintln!("✅ Q projection matches");
    } else {
        eprintln!("❌ Q projection MISMATCH (diff={q_diff})");
    }

    // === Run full layer 0 through Candle ===
    let mut config_manager2 = ferrum_models::ConfigManager::new();
    let model_def2 = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(config_manager2.load_from_path(&model_path))
        .unwrap();
    let vb3 = loader.load_varbuilder(&Device::Cpu, DType::F32).unwrap();
    let qwen3 = ferrum_models::Qwen3ModelWrapper::from_varbuilder(
        vb3,
        &model_def2,
        Device::Cpu,
        DType::F32,
    )
    .unwrap();

    // Candle forward_decode: token=1, pos=0
    let input_t = Tensor::new(&[1u32], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap();
    let candle_out = qwen3.forward_decode(&input_t, 0, "debug").unwrap();
    let candle_logits: Vec<f32> = candle_out.flatten_all().unwrap().to_vec1().unwrap();
    eprintln!("Candle full output[0..8]: {:?}", &candle_logits[..8]);

    // === Run full through ModelRunner (just 1 layer for comparison) ===
    // Actually, let's run the full model to match
    let vb4 = loader.load_varbuilder(&Device::Cpu, DType::F32).unwrap();
    let weights2 =
        ferrum_models::model_config::weight_loader::load_model_weights(&vb4, &cfg).unwrap();
    let mut runner = ferrum_kernels::backend::runner::ModelRunner::<
        ferrum_kernels::backend::cpu::CpuBackend,
    >::new(cfg.clone(), weights2);
    let runner_logits = runner.decode("debug", 1, 0);
    eprintln!("Runner full output[0..8]: {:?}", &runner_logits[..8]);

    // Quick cosine
    let mut dot = 0.0f64;
    let mut nc = 0.0f64;
    let mut nr = 0.0f64;
    for (c, r) in candle_logits.iter().zip(&runner_logits) {
        dot += *c as f64 * *r as f64;
        nc += (*c as f64).powi(2);
        nr += (*r as f64).powi(2);
    }
    let cosine = dot / (nc.sqrt() * nr.sqrt() + 1e-10);
    eprintln!("Full model cosine: {cosine:.6}");
}
