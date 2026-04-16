//! Debug: trace single layer forward step by step.
//! Run: cargo test -p ferrum-models --test debug_single_layer_test -- --ignored --nocapture

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::{AttnConfig, Backend};

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
fn trace_layer0_step_by_step() {
    let model_path = qwen3_model_path().unwrap();
    let loader = ferrum_models::SafeTensorsLoader::new(model_path.to_str().unwrap());
    let vb = loader.load_varbuilder(&Device::Cpu, DType::F32).unwrap();

    let mut cm = ferrum_models::ConfigManager::new();
    let model_def = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(cm.load_from_path(&model_path))
        .unwrap();
    let cfg = ferrum_models::model_config::qwen3_config(&model_def);
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let eps = cfg.rms_norm_eps;

    // Get embedding for token 1
    let embed_w = vb
        .get_unchecked("model.embed_tokens.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let embed = embed_w.get(1usize).unwrap();
    let embed_vec: Vec<f32> = embed.to_vec1().unwrap();

    // Load layer 0 weights individually via VarBuilder
    let q_w = vb
        .get_unchecked("model.layers.0.self_attn.q_proj.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let k_w = vb
        .get_unchecked("model.layers.0.self_attn.k_proj.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let v_w = vb
        .get_unchecked("model.layers.0.self_attn.v_proj.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let o_w = vb
        .get_unchecked("model.layers.0.self_attn.o_proj.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let q_norm_w = vb
        .get_unchecked("model.layers.0.self_attn.q_norm.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let k_norm_w = vb
        .get_unchecked("model.layers.0.self_attn.k_norm.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let input_ln_w = vb
        .get_unchecked("model.layers.0.input_layernorm.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Step 1: RMS norm
    let input_ln_vec: Vec<f32> = input_ln_w.to_vec1().unwrap();
    let mut norm_out = vec![0.0f32; h];
    CpuBackend::rms_norm(&mut (), &embed_vec, &input_ln_vec, eps, &mut norm_out, 1, h);

    // Step 2: Q, K, V projections (separate, like Candle)
    let normed_t = Tensor::new(&norm_out[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let candle_q = normed_t
        .matmul(&q_w.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let candle_k = normed_t
        .matmul(&k_w.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let candle_v = normed_t
        .matmul(&v_w.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();

    let cq: Vec<f32> = candle_q.to_vec1().unwrap();
    let ck: Vec<f32> = candle_k.to_vec1().unwrap();
    let cv: Vec<f32> = candle_v.to_vec1().unwrap();

    // Step 3: QK norm (per-head RMS norm)
    let q_norm_vec: Vec<f32> = q_norm_w.to_vec1().unwrap();
    let k_norm_vec: Vec<f32> = k_norm_w.to_vec1().unwrap();

    // Candle does: reshape [1, 1, nh, hd] → per-head norm
    let mut cq_normed = cq.clone();
    let mut ck_normed = ck.clone();
    for head in 0..nh {
        let off = head * hd;
        let q_slice = &cq[off..off + hd];
        let mut sum_sq = 0.0f32;
        for &v in q_slice {
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq / hd as f32 + eps).sqrt();
        for i in 0..hd {
            cq_normed[off + i] = q_slice[i] * inv * q_norm_vec[i];
        }
    }
    for head in 0..nkv {
        let off = head * hd;
        let k_slice = &ck[off..off + hd];
        let mut sum_sq = 0.0f32;
        for &v in k_slice {
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq / hd as f32 + eps).sqrt();
        for i in 0..hd {
            ck_normed[off + i] = k_slice[i] * inv * k_norm_vec[i];
        }
    }
    eprintln!("After QK-norm Q[0..8]: {:?}", &cq_normed[..8]);
    eprintln!("After QK-norm K[0..8]: {:?}", &ck_normed[..8]);

    // Step 4: RoPE at position 0
    // At pos=0, cos=1 sin=0 for all freqs → no rotation
    eprintln!(
        "After RoPE Q[0..8] (pos=0, no change): {:?}",
        &cq_normed[..8]
    );

    // Step 5: Attention (single token → output = V weighted)
    // With 1 KV token, softmax(Q·K^T * scale) = [1.0], so output = V
    let attn_cfg = AttnConfig {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        causal: false,
        scale: 1.0 / (hd as f32).sqrt(),
    };

    // Q: [nh, hd], K: [nkv, 1, hd] → reshape for cpu_attention [1, nh, 1, hd]
    let mut attn_out = vec![0.0f32; q_dim];
    CpuBackend::decode_attention(
        &mut (),
        &cq_normed,
        &ck_normed,
        &cv,
        &mut attn_out,
        1,
        &attn_cfg,
    );
    eprintln!("Attn out[0..8]: {:?}", &attn_out[..8]);
    // With 1 KV token, attention output should be just V (broadcast to all Q heads)
    eprintln!("V[0..8] (expected for 1-token attention): {:?}", &cv[..8]);

    // Step 6: O projection
    let attn_t = Tensor::new(&attn_out[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let o_out = attn_t
        .matmul(&o_w.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let o_vec: Vec<f32> = o_out.to_vec1().unwrap();
    eprintln!("O proj out[0..8]: {:?}", &o_vec[..8]);

    // Step 7: Residual add
    let mut residual: Vec<f32> = embed_vec.iter().zip(&o_vec).map(|(a, b)| a + b).collect();
    eprintln!("After attn residual[0..8]: {:?}", &residual[..8]);

    // === Now compare with Candle full layer 0 ===
    // Run Candle through layer 0 only (need to access internal)
    // Instead, let's compare the residual after layer 0 from full model run
    // === Run ModelRunner with 1-layer model for comparison ===
    let vb2 = loader.load_varbuilder(&Device::Cpu, DType::F32).unwrap();
    let mut cfg1 = cfg.clone();
    cfg1.num_layers = 1; // Only 1 layer

    let weights1 =
        ferrum_models::model_config::weight_loader::load_model_weights(&vb2, &cfg1).unwrap();
    let mut runner =
        ferrum_kernels::backend::runner::ModelRunner::<CpuBackend>::new(cfg1.clone(), weights1);

    // Decode with same token
    let runner_logits = runner.decode("test", 1, 0);

    // Manual path: residual after layer 0 → final norm → lm_head
    let post_ln_w = vb
        .get_unchecked("model.layers.0.post_attention_layernorm.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let post_ln_vec: Vec<f32> = post_ln_w.to_vec1().unwrap();

    // Post-attn norm
    let mut post_norm = vec![0.0f32; h];
    CpuBackend::rms_norm(&mut (), &residual, &post_ln_vec, eps, &mut post_norm, 1, h);

    // MLP
    let gate_w = vb
        .get_unchecked("model.layers.0.mlp.gate_proj.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let up_w_t = vb
        .get_unchecked("model.layers.0.mlp.up_proj.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let down_w = vb
        .get_unchecked("model.layers.0.mlp.down_proj.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let im = cfg.intermediate_size;

    let pn_t = Tensor::new(&post_norm[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let gate_out = pn_t
        .matmul(&gate_w.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let up_out = pn_t
        .matmul(&up_w_t.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let gv: Vec<f32> = gate_out.to_vec1().unwrap();
    let uv: Vec<f32> = up_out.to_vec1().unwrap();

    let silu_out: Vec<f32> = gv
        .iter()
        .zip(&uv)
        .map(|(g, u)| {
            let s = g / (1.0 + (-g).exp());
            s * u
        })
        .collect();

    let silu_t = Tensor::new(&silu_out[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mlp_out = silu_t
        .matmul(&down_w.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let mlp_vec: Vec<f32> = mlp_out.to_vec1().unwrap();

    // Final residual
    let final_residual: Vec<f32> = residual.iter().zip(&mlp_vec).map(|(a, b)| a + b).collect();

    // Final norm
    let final_norm_w = vb
        .get_unchecked("model.norm.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let fnw: Vec<f32> = final_norm_w.to_vec1().unwrap();
    let mut final_normed = vec![0.0f32; h];
    CpuBackend::rms_norm(&mut (), &final_residual, &fnw, eps, &mut final_normed, 1, h);

    // LM head
    let lm_head = vb
        .get_unchecked("lm_head.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let fn_t = Tensor::new(&final_normed[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let manual_logits = fn_t
        .matmul(&lm_head.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    let manual_logits_vec: Vec<f32> = manual_logits.to_vec1().unwrap();

    // Compare manual vs runner (both 1-layer)
    let mut dot = 0.0f64;
    let mut nm = 0.0f64;
    let mut nr = 0.0f64;
    for (m, r) in manual_logits_vec.iter().zip(&runner_logits) {
        dot += *m as f64 * *r as f64;
        nm += (*m as f64).powi(2);
        nr += (*r as f64).powi(2);
    }
    let cosine = dot / (nm.sqrt() * nr.sqrt() + 1e-10);

    let m_argmax = manual_logits_vec
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let r_argmax = runner_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("\n1-layer comparison:");
    eprintln!("Manual argmax={}, logit={:.4}", m_argmax.0, m_argmax.1);
    eprintln!("Runner argmax={}, logit={:.4}", r_argmax.0, r_argmax.1);
    eprintln!("Cosine: {cosine:.6}");

    if cosine > 0.99 {
        eprintln!("✅ 1-layer outputs match");
    } else {
        eprintln!("❌ 1-layer outputs MISMATCH (cosine={cosine:.6})");
        eprintln!("Manual logits[0..5]: {:?}", &manual_logits_vec[..5]);
        eprintln!("Runner logits[0..5]: {:?}", &runner_logits[..5]);
    }
}
