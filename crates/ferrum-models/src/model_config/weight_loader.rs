//! Weight loading bridge: Candle VarBuilder → ModelWeights<CpuBackend>.
//!
//! This is the ONLY place Candle touches inference data structures.
//! Weights are extracted as f32 vecs, then wrapped in Backend::from_slice().

use candle_core::{DType, Tensor};
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
    let embed = get_f32(vb, "model.embed_tokens")?;

    // Per-layer weights
    let mut layers = Vec::with_capacity(cfg.num_layers);
    for li in 0..cfg.num_layers {
        let prefix = format!("model.layers.{li}");

        // Input norm
        let input_ln_w = get_f32(vb, &format!("{prefix}.input_layernorm"))?;

        // QKV projection: try fused first, then separate
        let qkv_proj_w = if let Ok(fused) = get_f32(vb, &format!("{prefix}.self_attn.qkv_proj")) {
            fused
        } else {
            // Separate Q, K, V → fuse into [q_dim + 2*kv_dim, hidden]
            let fused = get_tensor_cat(
                vb,
                &[
                    &format!("{prefix}.self_attn.q_proj"),
                    &format!("{prefix}.self_attn.k_proj"),
                    &format!("{prefix}.self_attn.v_proj"),
                ],
            )?;
            tensor_to_f32(&fused)?
        };

        // O projection
        let o_proj_w = get_f32(vb, &format!("{prefix}.self_attn.o_proj"))?;

        // Post-attention norm
        let post_ln_w = get_f32(vb, &format!("{prefix}.post_attention_layernorm"))?;

        // Gate/Up MLP: try fused first, then separate
        let gate_up_proj_w = if let Ok(fused) = get_f32(vb, &format!("{prefix}.mlp.gate_up_proj")) {
            fused
        } else {
            let fused = get_tensor_cat(
                vb,
                &[
                    &format!("{prefix}.mlp.gate_proj"),
                    &format!("{prefix}.mlp.up_proj"),
                ],
            )?;
            tensor_to_f32(&fused)?
        };

        // Down projection
        let down_proj_w = get_f32(vb, &format!("{prefix}.mlp.down_proj"))?;

        // Optional QK norm weights
        let q_norm_w = get_f32(vb, &format!("{prefix}.self_attn.q_norm")).ok();
        let k_norm_w = get_f32(vb, &format!("{prefix}.self_attn.k_norm")).ok();

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
    let final_norm_w = get_f32(vb, "model.norm")?;

    // LM head: try dedicated lm_head, fallback to tied embed
    let lm_head_w = get_f32(vb, "lm_head").unwrap_or_else(|_| embed.clone());

    Ok(ModelWeights {
        embed: CpuBackend::from_slice(&embed),
        layers,
        final_norm_w: CpuBackend::from_slice(&final_norm_w),
        lm_head_w: CpuBackend::from_slice(&lm_head_w),
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Navigate VarBuilder by dotted path and get the "weight" tensor.
///
/// Uses safetensors metadata to infer shape (no shape hint needed).
fn get_weight(vb: &VarBuilder, dotted_path: &str) -> Result<Tensor> {
    // VarBuilder.get() requires shape, but we don't always know it.
    // Instead, access the underlying safetensors data directly.
    // The full key in safetensors is "{dotted_path}.weight".
    let key = format!("{dotted_path}.weight");
    vb.get_unchecked(&key)
        .map_err(|e| FerrumError::model(format!("weight '{key}': {e}")))
}

fn get_f32(vb: &VarBuilder, dotted_path: &str) -> Result<Vec<f32>> {
    let t = get_weight(vb, dotted_path)?;
    tensor_to_f32(&t)
}

fn get_tensor_cat(vb: &VarBuilder, paths: &[&str]) -> Result<Tensor> {
    let tensors: Vec<Tensor> = paths
        .iter()
        .map(|p| get_weight(vb, p))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Tensor> = tensors.iter().collect();
    Tensor::cat(&refs, 0).map_err(|e| FerrumError::model(format!("cat: {e}")))
}

fn tensor_to_f32(t: &Tensor) -> Result<Vec<f32>> {
    let t = t
        .to_dtype(DType::F32)
        .map_err(|e| FerrumError::model(format!("to_f32: {e}")))?;
    let t = t
        .flatten_all()
        .map_err(|e| FerrumError::model(format!("flatten: {e}")))?;
    t.to_vec1::<f32>()
        .map_err(|e| FerrumError::model(format!("to_vec: {e}")))
}
