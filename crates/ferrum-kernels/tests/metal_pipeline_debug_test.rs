#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::metal::MetalBackend;
use ferrum_kernels::backend::Backend;

fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// rms_norm then gemm — does Metal output match CPU?
#[test]
fn test_norm_then_gemm() {
    let dim = 1024;
    let n = 4096;
    let x = pseudo_random(1, dim);
    let w = pseudo_random(2, dim);
    let b = pseudo_random(3, n * dim);
    let eps = 1e-6f32;

    // CPU: norm → gemm
    let mut cpu_norm = CpuBackend::alloc(dim);
    CpuBackend::rms_norm(&mut (), &x, &w, eps, &mut cpu_norm, 1, dim);
    let mut cpu_out = CpuBackend::alloc(n);
    CpuBackend::gemm(&mut (), &cpu_norm, &b, &mut cpu_out, 1, n, dim);

    // Metal: norm → gemm
    let x_m = MetalBackend::from_slice(&x);
    let w_m = MetalBackend::from_slice(&w);
    let b_m = MetalBackend::from_slice(&b);
    let mut norm_m = MetalBackend::alloc(dim);
    let mut out_m = MetalBackend::alloc(n);

    let mut ctx = MetalBackend::new_context();
    MetalBackend::rms_norm(&mut ctx, &x_m, &w_m, eps, &mut norm_m, 1, dim);
    MetalBackend::gemm(&mut ctx, &norm_m, &b_m, &mut out_m, 1, n, dim);
    MetalBackend::sync(&mut ctx);

    let mv = MetalBackend::to_vec(&out_m, n);
    let max_diff = cpu_out
        .iter()
        .zip(&mv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_max = cpu_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("norm→gemm: max_diff={max_diff:.2e} cpu_max={cpu_max:.2e}");
    eprintln!("CPU[0..5]: {:?}", &cpu_out[..5]);
    eprintln!("MTL[0..5]: {:?}", &mv[..5]);
    assert!(max_diff / cpu_max < 1e-3, "norm→gemm mismatch");
}

/// Full 1-layer decode: compare Metal pipeline vs CPU
#[test]
fn test_one_layer_decode_metal_vs_cpu() {
    use ferrum_kernels::backend::runner::ModelRunner;
    use ferrum_kernels::backend::{
        AttnConfig, AttnType, KvCache, LayerScratch, LayerWeights, MlpType, ModelWeights,
        RopeConfig, TransformerConfig,
    };

    let cfg = TransformerConfig {
        num_layers: 1,
        hidden_size: 64,
        intermediate_size: 128,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        vocab_size: 100,
        max_seq_len: 32,
        rms_norm_eps: 1e-5,
        rope: RopeConfig {
            theta: 10000.0,
            head_dim: 16,
            max_seq_len: 32,
        },
        has_qk_norm: false,
        attn_type: AttnType::Gqa,
        mlp_type: MlpType::SwiGlu,
    };
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let qkv_dim = q_dim + 2 * kv_dim;

    let mut seed = 100u64;
    let mut rng = |n: usize| -> Vec<f32> {
        seed += 1;
        pseudo_random(seed, n)
    };

    let cpu_weights = ModelWeights::<CpuBackend> {
        embed: rng(vocab * h),
        layers: vec![LayerWeights {
            input_ln_w: vec![1.0; h],
            qkv_proj_w: rng(qkv_dim * h),
            o_proj_w: rng(h * q_dim),
            post_ln_w: vec![1.0; h],
            gate_up_proj_w: rng(2 * im * h),
            down_proj_w: rng(h * im),
            q_norm_w: None,
            k_norm_w: None,
        }],
        final_norm_w: vec![1.0; h],
        lm_head_w: rng(vocab * h),
    };

    let metal_weights = ModelWeights::<MetalBackend> {
        embed: MetalBackend::from_slice(&cpu_weights.embed),
        layers: vec![LayerWeights {
            input_ln_w: MetalBackend::from_slice(&cpu_weights.layers[0].input_ln_w),
            qkv_proj_w: MetalBackend::from_slice(&cpu_weights.layers[0].qkv_proj_w),
            o_proj_w: MetalBackend::from_slice(&cpu_weights.layers[0].o_proj_w),
            post_ln_w: MetalBackend::from_slice(&cpu_weights.layers[0].post_ln_w),
            gate_up_proj_w: MetalBackend::from_slice(&cpu_weights.layers[0].gate_up_proj_w),
            down_proj_w: MetalBackend::from_slice(&cpu_weights.layers[0].down_proj_w),
            q_norm_w: None,
            k_norm_w: None,
        }],
        final_norm_w: MetalBackend::from_slice(&cpu_weights.final_norm_w),
        lm_head_w: MetalBackend::from_slice(&cpu_weights.lm_head_w),
    };

    let mut cpu_runner = ModelRunner::<CpuBackend>::new(cfg.clone(), cpu_weights);
    let mut metal_runner = ModelRunner::<MetalBackend>::new(cfg, metal_weights);

    let cpu_logits = cpu_runner.decode("t", 5, 0);
    let metal_logits = metal_runner.decode("t", 5, 0);

    let max_diff = cpu_logits
        .iter()
        .zip(&metal_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_max = cpu_logits.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!(
        "1-layer decode: max_diff={max_diff:.2e} cpu_max={cpu_max:.2e} rel={:.2e}",
        max_diff / cpu_max
    );
    eprintln!(
        "CPU argmax={}",
        cpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );
    eprintln!(
        "MTL argmax={}",
        metal_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );
    assert!(max_diff / cpu_max < 1e-2, "1-layer Metal vs CPU mismatch");
}

/// Test with real model weights — 1 decode step
#[test]
#[ignore] // needs Qwen3-0.6B downloaded
fn test_real_model_metal_vs_cpu() {
    use ferrum_kernels::backend::runner::{convert_weights_to_metal, ModelRunner};

    let home = dirs::home_dir().unwrap();
    let path = home.join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let model_path = std::fs::read_dir(&path)
        .ok()
        .and_then(|mut d| {
            d.find(|e| {
                e.as_ref()
                    .ok()
                    .map_or(false, |e| e.path().join("config.json").exists())
            })
        })
        .map(|e| e.unwrap().path());
    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("skip");
            return;
        }
    };

    let loader = ferrum_models::SafeTensorsLoader::new(model_path.to_str().unwrap());
    let cj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_path.join("config.json")).unwrap())
            .unwrap();
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

    let vb = loader
        .load_varbuilder(&candle_core::Device::Cpu, candle_core::DType::F32)
        .unwrap();
    let cpu_w = ferrum_models::model_config::weight_loader::load_model_weights(&vb, &cfg).unwrap();
    let metal_w = convert_weights_to_metal(&cpu_w);

    let mut cpu_r = ModelRunner::<CpuBackend>::new(cfg.clone(), cpu_w);
    let mut mtl_r = ModelRunner::<MetalBackend>::new(cfg, metal_w);

    let cpu_logits = cpu_r.decode("t", 1, 0);
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
    assert!(cos > 0.99, "Metal vs CPU cosine too low: {cos}");
}
