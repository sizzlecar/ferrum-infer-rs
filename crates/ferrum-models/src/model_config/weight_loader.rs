//! Weight loading bridge: Candle VarBuilder → ModelWeights<CpuBackend>.
//!
//! This is the ONLY place Candle touches inference data structures.
//! Weights are extracted as f32 vecs, then wrapped in Backend::from_slice().

use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::{Backend, LayerWeights, ModelWeights, TransformerConfig};
use ferrum_types::{FerrumError, Result};

/// Load transformer weights from safetensors into ModelWeights<CpuBackend>.
///
/// Handles both fused (Qwen3: `qkv_proj`) and separate (Llama: `q_proj`+`k_proj`+`v_proj`)
/// weight formats automatically.
pub fn load_model_weights(
    vb: &VarBuilder,
    cfg: &TransformerConfig,
) -> Result<ModelWeights<CpuBackend>> {
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;

    // Embedding
    let embed = get_f32(vb, "model.embed_tokens.weight")?;

    // Per-layer weights
    let mut layers = Vec::with_capacity(cfg.num_layers);
    for li in 0..cfg.num_layers {
        let prefix = format!("model.layers.{li}");

        // Input norm
        let input_ln_w = get_f32(vb, &format!("{prefix}.input_layernorm.weight"))?;

        // QKV projection: try fused first, then separate
        let qkv_proj_w = if let Ok(fused) = get_f32(vb, &format!("{prefix}.self_attn.qkv_proj.weight")) {
            fused
        } else {
            // Separate Q, K, V → fuse into [q_dim + 2*kv_dim, hidden]
            let q = get_tensor(vb, &format!("{prefix}.self_attn.q_proj.weight"))?;
            let k = get_tensor(vb, &format!("{prefix}.self_attn.k_proj.weight"))?;
            let v = get_tensor(vb, &format!("{prefix}.self_attn.v_proj.weight"))?;
            let fused = Tensor::cat(&[&q, &k, &v], 0)
                .map_err(|e| FerrumError::model(format!("QKV cat: {e}")))?;
            tensor_to_f32(&fused)?
        };

        // O projection
        let o_proj_w = get_f32(vb, &format!("{prefix}.self_attn.o_proj.weight"))?;

        // Post-attention norm
        let post_ln_w = get_f32(vb, &format!("{prefix}.post_attention_layernorm.weight"))?;

        // Gate/Up MLP: try fused first, then separate
        let gate_up_proj_w =
            if let Ok(fused) = get_f32(vb, &format!("{prefix}.mlp.gate_up_proj.weight")) {
                fused
            } else {
                let gate = get_tensor(vb, &format!("{prefix}.mlp.gate_proj.weight"))?;
                let up = get_tensor(vb, &format!("{prefix}.mlp.up_proj.weight"))?;
                let fused = Tensor::cat(&[&gate, &up], 0)
                    .map_err(|e| FerrumError::model(format!("gate_up cat: {e}")))?;
                tensor_to_f32(&fused)?
            };

        // Down projection
        let down_proj_w = get_f32(vb, &format!("{prefix}.mlp.down_proj.weight"))?;

        // Optional QK norm weights
        let q_norm_w = get_f32(vb, &format!("{prefix}.self_attn.q_norm.weight")).ok();
        let k_norm_w = get_f32(vb, &format!("{prefix}.self_attn.k_norm.weight")).ok();

        layers.push(LayerWeights {
            input_ln_w: CpuBackend::from_slice(&input_ln_w),
            qkv_proj_w: CpuBackend::from_slice(&qkv_proj_w),
            o_proj_w: CpuBackend::from_slice(&o_proj_w),
            post_ln_w: CpuBackend::from_slice(&post_ln_w),
            gate_up_proj_w: CpuBackend::from_slice(&gate_up_proj_w),
            down_proj_w: CpuBackend::from_slice(&down_proj_w),
            q_norm_w: q_norm_w.map(|w| CpuBackend::from_slice(&w)),
            k_norm_w: k_norm_w.map(|w| CpuBackend::from_slice(&w)),
        });
    }

    // Final norm
    let final_norm_w = get_f32(vb, "model.norm.weight")?;

    // LM head: try dedicated lm_head, fallback to tied embed
    let lm_head_w = get_f32(vb, "lm_head.weight")
        .unwrap_or_else(|_| embed.clone());

    Ok(ModelWeights {
        embed: CpuBackend::from_slice(&embed),
        layers,
        final_norm_w: CpuBackend::from_slice(&final_norm_w),
        lm_head_w: CpuBackend::from_slice(&lm_head_w),
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn get_tensor(vb: &VarBuilder, name: &str) -> Result<Tensor> {
    vb.get_with_hints(&[], name, candle_nn::Init::Const(0.0))
        .or_else(|_| {
            // VarBuilder sometimes needs shape hints — try without
            vb.pp("").get_with_hints(&[], name, candle_nn::Init::Const(0.0))
        })
        .map_err(|e| FerrumError::model(format!("weight '{name}': {e}")))
}

fn get_f32(vb: &VarBuilder, name: &str) -> Result<Vec<f32>> {
    let t = get_tensor(vb, name)?;
    tensor_to_f32(&t)
}

fn tensor_to_f32(t: &Tensor) -> Result<Vec<f32>> {
    let t = t.to_dtype(DType::F32)
        .map_err(|e| FerrumError::model(format!("to_f32: {e}")))?;
    let t = t.flatten_all()
        .map_err(|e| FerrumError::model(format!("flatten: {e}")))?;
    t.to_vec1::<f32>()
        .map_err(|e| FerrumError::model(format!("to_vec: {e}")))
}
