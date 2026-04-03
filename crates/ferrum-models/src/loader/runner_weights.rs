//! Generic CUDA decode runner weight loader.
//!
//! Loads transformer weights from safetensors, fuses separate Q/K/V → QKV
//! and gate/up → gate_up, then uploads to a CUDA stream. Architecture-agnostic:
//! works for Llama, Qwen2, Mistral, and any model with the standard naming.

#[cfg(feature = "cuda")]
use candle_core::{DType, Device as CandleDevice, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::VarBuilder;
#[cfg(feature = "cuda")]
use ferrum_cuda_kernels::{
    decode_buffers::ModelDims,
    weight_store::{GpuWeight, LayerWeights, LinearWeight, TransformerGpuWeights},
};
#[cfg(feature = "cuda")]
use ferrum_types::{FerrumError, Result};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Architecture-specific weight naming and structure.
#[cfg(feature = "cuda")]
pub struct WeightConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    /// Whether the model has Q/K normalization (Qwen3 yes, Llama/Qwen2 no)
    pub has_qk_norm: bool,
    /// Whether QKV projection is already fused into one weight
    pub qkv_fused: bool,
    /// Whether gate+up MLP projection is already fused
    pub gate_up_fused: bool,
}

/// Load transformer weights from safetensors for the CUDA decode runner.
///
/// Handles both fused (Qwen3) and separate (Llama/Qwen2) weight formats.
/// Returns `TransformerGpuWeights` ready for `CudaDecodeRunner`.
#[cfg(feature = "cuda")]
pub fn load_runner_weights(
    vb: &VarBuilder,
    cfg: &WeightConfig,
    device: &CandleDevice,
) -> Result<(
    TransformerGpuWeights,
    ModelDims,
    Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
)> {
    use candle_core::cuda_backend::CudaDevice;

    let cuda_device = device
        .as_cuda_device()
        .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;

    // Sync candle stream, create runner stream
    let candle_stream = cuda_device.cuda_stream();
    candle_stream
        .synchronize()
        .map_err(|e| FerrumError::model(format!("candle stream sync: {e}")))?;
    let rs = candle_stream
        .context()
        .new_stream()
        .map_err(|e| FerrumError::model(format!("new_stream: {e}")))?;

    // Embed tokens
    let embed_t = vb
        .get(
            (cfg.vocab_size, cfg.hidden_size),
            "model.embed_tokens.weight",
        )
        .map_err(|e| FerrumError::model(format!("embed: {e}")))?;
    let embed_table = GpuWeight::from_tensor(&embed_t, &rs)
        .map_err(|e| FerrumError::model(format!("embed: {e}")))?;

    // Per-layer weights
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;

    for li in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{li}");

        // Input layer norm
        let ln_w = vb
            .get(cfg.hidden_size, &format!("{prefix}.input_layernorm.weight"))
            .map_err(|e| FerrumError::model(format!("input_ln L{li}: {e}")))?;
        let input_ln_w = GpuWeight::from_tensor(&ln_w, &rs)
            .map_err(|e| FerrumError::model(format!("input_ln: {e}")))?;

        // QKV projection — fuse if separate
        let qkv_tensor = if cfg.qkv_fused {
            vb.get(
                (q_dim + 2 * kv_dim, cfg.hidden_size),
                &format!("{prefix}.self_attn.qkv_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("qkv L{li}: {e}")))?
        } else {
            let q = vb
                .get(
                    (q_dim, cfg.hidden_size),
                    &format!("{prefix}.self_attn.q_proj.weight"),
                )
                .map_err(|e| FerrumError::model(format!("q L{li}: {e}")))?;
            let k = vb
                .get(
                    (kv_dim, cfg.hidden_size),
                    &format!("{prefix}.self_attn.k_proj.weight"),
                )
                .map_err(|e| FerrumError::model(format!("k L{li}: {e}")))?;
            let v = vb
                .get(
                    (kv_dim, cfg.hidden_size),
                    &format!("{prefix}.self_attn.v_proj.weight"),
                )
                .map_err(|e| FerrumError::model(format!("v L{li}: {e}")))?;
            Tensor::cat(&[&q, &k, &v], 0)
                .map_err(|e| FerrumError::model(format!("qkv cat L{li}: {e}")))?
        };
        let qkv_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(&qkv_tensor, &rs)
                .map_err(|e| FerrumError::model(format!("qkv: {e}")))?,
        );

        // Q/K norms (Qwen3 only)
        let q_norm_w = if cfg.has_qk_norm {
            let t = vb
                .get(cfg.head_dim, &format!("{prefix}.self_attn.q_norm.weight"))
                .map_err(|e| FerrumError::model(format!("q_norm L{li}: {e}")))?;
            Some(
                GpuWeight::from_tensor(&t, &rs)
                    .map_err(|e| FerrumError::model(format!("q_norm: {e}")))?,
            )
        } else {
            None
        };
        let k_norm_w = if cfg.has_qk_norm {
            let t = vb
                .get(cfg.head_dim, &format!("{prefix}.self_attn.k_norm.weight"))
                .map_err(|e| FerrumError::model(format!("k_norm L{li}: {e}")))?;
            Some(
                GpuWeight::from_tensor(&t, &rs)
                    .map_err(|e| FerrumError::model(format!("k_norm: {e}")))?,
            )
        } else {
            None
        };

        // O projection
        let o_t = vb
            .get(
                (cfg.hidden_size, q_dim),
                &format!("{prefix}.self_attn.o_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("o L{li}: {e}")))?;
        let o_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(&o_t, &rs).map_err(|e| FerrumError::model(format!("o: {e}")))?,
        );

        // Post-attention layer norm
        let pln_t = vb
            .get(
                cfg.hidden_size,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )
            .map_err(|e| FerrumError::model(format!("post_ln L{li}: {e}")))?;
        let post_ln_w = GpuWeight::from_tensor(&pln_t, &rs)
            .map_err(|e| FerrumError::model(format!("post_ln: {e}")))?;

        // MLP gate+up — fuse if separate
        let gate_up_tensor = if cfg.gate_up_fused {
            vb.get(
                (2 * cfg.intermediate_size, cfg.hidden_size),
                &format!("{prefix}.mlp.gate_up_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("gate_up L{li}: {e}")))?
        } else {
            let gate = vb
                .get(
                    (cfg.intermediate_size, cfg.hidden_size),
                    &format!("{prefix}.mlp.gate_proj.weight"),
                )
                .map_err(|e| FerrumError::model(format!("gate L{li}: {e}")))?;
            let up = vb
                .get(
                    (cfg.intermediate_size, cfg.hidden_size),
                    &format!("{prefix}.mlp.up_proj.weight"),
                )
                .map_err(|e| FerrumError::model(format!("up L{li}: {e}")))?;
            Tensor::cat(&[&gate, &up], 0)
                .map_err(|e| FerrumError::model(format!("gate_up cat L{li}: {e}")))?
        };
        let gate_up_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(&gate_up_tensor, &rs)
                .map_err(|e| FerrumError::model(format!("gate_up: {e}")))?,
        );

        // Down projection
        let down_t = vb
            .get(
                (cfg.hidden_size, cfg.intermediate_size),
                &format!("{prefix}.mlp.down_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("down L{li}: {e}")))?;
        let down_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(&down_t, &rs)
                .map_err(|e| FerrumError::model(format!("down: {e}")))?,
        );

        layers.push(LayerWeights {
            input_ln_w,
            qkv_w,
            q_norm_w,
            k_norm_w,
            o_w,
            post_ln_w,
            gate_up_w,
            down_w,
        });
    }

    // Final norm
    let fn_t = vb
        .get(cfg.hidden_size, "model.norm.weight")
        .map_err(|e| FerrumError::model(format!("final_norm: {e}")))?;
    let final_norm_w = GpuWeight::from_tensor(&fn_t, &rs)
        .map_err(|e| FerrumError::model(format!("final_norm: {e}")))?;

    // LM head (or tied to embed_tokens)
    let lm_t = vb
        .get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
        .or_else(|_| {
            vb.get(
                (cfg.vocab_size, cfg.hidden_size),
                "model.embed_tokens.weight",
            )
        })
        .map_err(|e| FerrumError::model(format!("lm_head: {e}")))?;
    let lm_head_w = LinearWeight::Fp16(
        GpuWeight::from_tensor(&lm_t, &rs)
            .map_err(|e| FerrumError::model(format!("lm_head: {e}")))?,
    );

    // RoPE cos/sin tables — compute from config
    let (rope_cos, rope_sin) = compute_rope_tables(cfg, device, &rs)?;

    let weights = TransformerGpuWeights {
        embed_table,
        layers,
        final_norm_w,
        lm_head_w,
        rope_cos,
        rope_sin,
    };

    let dims = ModelDims {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_attention_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        vocab_size: cfg.vocab_size,
        num_layers: cfg.num_hidden_layers,
        max_seq_len: cfg.max_seq_len,
        quantized: false,
        max_batch_size: std::env::var("FERRUM_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
    };

    rs.synchronize()
        .map_err(|e| FerrumError::model(format!("stream sync: {e}")))?;

    Ok((weights, dims, rs))
}

/// Compute RoPE tables for TP (public wrapper).
#[cfg(feature = "cuda")]
pub fn compute_rope_tables_for_tp(
    cfg: &super::tp_weight_loader::TpWeightConfig,
    device: &CandleDevice,
    stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<(GpuWeight, GpuWeight)> {
    let w = WeightConfig {
        num_hidden_layers: cfg.num_hidden_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_attention_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        vocab_size: cfg.vocab_size,
        max_seq_len: cfg.max_seq_len,
        rope_theta: cfg.rope_theta,
        has_qk_norm: cfg.has_qk_norm,
        qkv_fused: false,
        gate_up_fused: false,
    };
    compute_rope_tables(&w, device, stream)
}

/// Compute RoPE cos/sin tables from config parameters.
#[cfg(feature = "cuda")]
fn compute_rope_tables(
    cfg: &WeightConfig,
    device: &CandleDevice,
    stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<(GpuWeight, GpuWeight)> {
    let half_dim = cfg.head_dim / 2;
    let max_len = cfg.max_seq_len;

    // inv_freq = 1.0 / (theta ^ (2i / head_dim)) for i in 0..head_dim/2
    let mut inv_freq = vec![0f32; half_dim];
    for i in 0..half_dim {
        inv_freq[i] = 1.0 / (cfg.rope_theta as f32).powf(2.0 * i as f32 / cfg.head_dim as f32);
    }

    // cos/sin table: [max_len, half_dim] as f16
    let total = max_len * half_dim;
    let mut cos_data = vec![half::f16::ZERO; total];
    let mut sin_data = vec![half::f16::ZERO; total];

    for pos in 0..max_len {
        for i in 0..half_dim {
            let angle = pos as f32 * inv_freq[i];
            cos_data[pos * half_dim + i] = half::f16::from_f32(angle.cos());
            sin_data[pos * half_dim + i] = half::f16::from_f32(angle.sin());
        }
    }

    // Upload to GPU
    let cos_slice = stream
        .clone_htod(&cos_data)
        .map_err(|e| FerrumError::model(format!("rope cos upload: {e}")))?;
    let sin_slice = stream
        .clone_htod(&sin_data)
        .map_err(|e| FerrumError::model(format!("rope sin upload: {e}")))?;

    Ok((
        GpuWeight {
            slice: cos_slice,
            len: total,
        },
        GpuWeight {
            slice: sin_slice,
            len: total,
        },
    ))
}
