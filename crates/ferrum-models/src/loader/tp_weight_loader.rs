//! Tensor-parallel weight loading.
//!
//! Loads transformer weights from safetensors and shards them across
//! multiple GPUs following the Megatron-LM pattern:
//! - Column-parallel: QKV, gate, up (split output dim)
//! - Row-parallel: O, down (split input dim)
//! - Replicated: norms, embeddings, lm_head, RoPE
//!
//! Feature-gated: only available with `tensor-parallel` feature.

#[cfg(feature = "tensor-parallel")]
use candle_core::{DType, Device as CandleDevice, Tensor};
#[cfg(feature = "tensor-parallel")]
use candle_nn::VarBuilder;
#[cfg(feature = "tensor-parallel")]
use ferrum_cuda_kernels::{
    decode_buffers::ModelDims,
    weight_store::{GpuWeight, LayerWeights, LinearWeight, TransformerGpuWeights},
};
#[cfg(feature = "tensor-parallel")]
use ferrum_types::{FerrumError, Result};
#[cfg(feature = "tensor-parallel")]
use std::sync::Arc;

/// Tensor parallel weight config — extends base WeightConfig with TP params.
#[cfg(feature = "tensor-parallel")]
#[derive(Clone)]
pub struct TpWeightConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    pub has_qk_norm: bool,
    /// TP world size
    pub tp_size: usize,
    /// This rank's index (0..tp_size)
    pub rank: usize,
}

/// Load sharded weights for one TP rank.
///
/// Returns TransformerGpuWeights with:
/// - QKV: only this rank's heads (q_dim/tp + 2*kv_dim/tp)
/// - O: only this rank's input slice (hidden, q_dim/tp)
/// - gate_up: only this rank's intermediate slice (2*inter/tp, hidden)
/// - down: only this rank's input slice (hidden, inter/tp)
/// - Norms, embed, lm_head, RoPE: full (replicated)
#[cfg(feature = "tensor-parallel")]
pub fn load_sharded_weights(
    vb: &VarBuilder,
    cfg: &TpWeightConfig,
    device: &CandleDevice,
) -> Result<(
    TransformerGpuWeights,
    ModelDims,
    Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
)> {
    let cuda_device = device
        .as_cuda_device()
        .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;
    let candle_stream = cuda_device.cuda_stream();
    candle_stream
        .synchronize()
        .map_err(|e| FerrumError::model(format!("sync: {e}")))?;

    // Helper: move tensor to target device (cross-GPU transfer for rank > 0)
    let to_dev = |t: &Tensor| -> Result<Tensor> {
        t.to_device(device)
            .map_err(|e| FerrumError::model(format!("to_device: {e}")))
    };
    let rs = candle_stream
        .context()
        .new_stream()
        .map_err(|e| FerrumError::model(format!("new_stream: {e}")))?;

    let tp = cfg.tp_size;
    let rank = cfg.rank;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let q_dim_shard = q_dim / tp;
    let kv_dim_shard = kv_dim / tp;
    let inter_shard = cfg.intermediate_size / tp;

    // Embed — replicated
    let embed_t = vb
        .get(
            (cfg.vocab_size, cfg.hidden_size),
            "model.embed_tokens.weight",
        )
        .map_err(|e| FerrumError::model(format!("embed: {e}")))?;
    let embed_t = to_dev(&embed_t)?;
    let embed_table = GpuWeight::from_tensor(&embed_t, &rs)
        .map_err(|e| FerrumError::model(format!("embed: {e}")))?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for li in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{li}");

        // Input norm — replicated
        let ln_w = vb
            .get(cfg.hidden_size, &format!("{prefix}.input_layernorm.weight"))
            .map_err(|e| FerrumError::model(format!("input_ln L{li}: {e}")))?;
        let ln_w = to_dev(&ln_w)?;
        let input_ln_w = GpuWeight::from_tensor(&ln_w, &rs)
            .map_err(|e| FerrumError::model(format!("input_ln: {e}")))?;

        // QKV — column-parallel (split output dim = rows for [out, in] weight)
        let q_full = vb
            .get(
                (q_dim, cfg.hidden_size),
                &format!("{prefix}.self_attn.q_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("q L{li}: {e}")))?;
        let k_full = vb
            .get(
                (kv_dim, cfg.hidden_size),
                &format!("{prefix}.self_attn.k_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("k L{li}: {e}")))?;
        let v_full = vb
            .get(
                (kv_dim, cfg.hidden_size),
                &format!("{prefix}.self_attn.v_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("v L{li}: {e}")))?;

        // Shard: take rank's slice of rows
        let q_shard = q_full
            .narrow(0, rank * q_dim_shard, q_dim_shard)
            .map_err(|e| FerrumError::model(format!("q shard L{li}: {e}")))?;
        let k_shard = k_full
            .narrow(0, rank * kv_dim_shard, kv_dim_shard)
            .map_err(|e| FerrumError::model(format!("k shard L{li}: {e}")))?;
        let v_shard = v_full
            .narrow(0, rank * kv_dim_shard, kv_dim_shard)
            .map_err(|e| FerrumError::model(format!("v shard L{li}: {e}")))?;
        let qkv_shard = Tensor::cat(&[&q_shard, &k_shard, &v_shard], 0)
            .map_err(|e| FerrumError::model(format!("qkv cat L{li}: {e}")))?;

        let qkv_shard = to_dev(&qkv_shard)?;
        let qkv_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(&qkv_shard, &rs)
                .map_err(|e| FerrumError::model(format!("qkv: {e}")))?,
        );

        // Q/K norms — replicated (if present)
        let q_norm_w = if cfg.has_qk_norm {
            let t = vb
                .get(cfg.head_dim, &format!("{prefix}.self_attn.q_norm.weight"))
                .map_err(|e| FerrumError::model(format!("q_norm L{li}: {e}")))?;
            let t = to_dev(&t)?;
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
            let t = to_dev(&t)?;
            Some(
                GpuWeight::from_tensor(&t, &rs)
                    .map_err(|e| FerrumError::model(format!("k_norm: {e}")))?,
            )
        } else {
            None
        };

        // O projection — row-parallel (split input dim = columns for [out, in] weight)
        let o_full = vb
            .get(
                (cfg.hidden_size, q_dim),
                &format!("{prefix}.self_attn.o_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("o L{li}: {e}")))?;
        let o_shard = o_full
            .narrow(1, rank * q_dim_shard, q_dim_shard)
            .map_err(|e| FerrumError::model(format!("o shard L{li}: {e}")))?
            .contiguous()
            .map_err(|e| FerrumError::model(format!("o contiguous L{li}: {e}")))?;
        let o_shard = to_dev(&o_shard)?;
        let o_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(&o_shard, &rs)
                .map_err(|e| FerrumError::model(format!("o: {e}")))?,
        );

        // Post-attn norm — replicated
        let pln_t = vb
            .get(
                cfg.hidden_size,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )
            .map_err(|e| FerrumError::model(format!("post_ln L{li}: {e}")))?;
        let pln_t = to_dev(&pln_t)?;
        let post_ln_w = GpuWeight::from_tensor(&pln_t, &rs)
            .map_err(|e| FerrumError::model(format!("post_ln: {e}")))?;

        // MLP gate+up — column-parallel (split output dim)
        let gate_full = vb
            .get(
                (cfg.intermediate_size, cfg.hidden_size),
                &format!("{prefix}.mlp.gate_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("gate L{li}: {e}")))?;
        let up_full = vb
            .get(
                (cfg.intermediate_size, cfg.hidden_size),
                &format!("{prefix}.mlp.up_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("up L{li}: {e}")))?;
        let gate_shard = gate_full
            .narrow(0, rank * inter_shard, inter_shard)
            .map_err(|e| FerrumError::model(format!("gate shard L{li}: {e}")))?;
        let up_shard = up_full
            .narrow(0, rank * inter_shard, inter_shard)
            .map_err(|e| FerrumError::model(format!("up shard L{li}: {e}")))?;
        let gate_up_shard = Tensor::cat(&[&gate_shard, &up_shard], 0)
            .map_err(|e| FerrumError::model(format!("gate_up cat L{li}: {e}")))?;
        let gate_up_shard = to_dev(&gate_up_shard)?;
        let gate_up_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(&gate_up_shard, &rs)
                .map_err(|e| FerrumError::model(format!("gate_up: {e}")))?,
        );

        // MLP down — row-parallel (split input dim)
        let down_full = vb
            .get(
                (cfg.hidden_size, cfg.intermediate_size),
                &format!("{prefix}.mlp.down_proj.weight"),
            )
            .map_err(|e| FerrumError::model(format!("down L{li}: {e}")))?;
        let down_shard = down_full
            .narrow(1, rank * inter_shard, inter_shard)
            .map_err(|e| FerrumError::model(format!("down shard L{li}: {e}")))?
            .contiguous()
            .map_err(|e| FerrumError::model(format!("down contiguous L{li}: {e}")))?;
        let down_shard = to_dev(&down_shard)?;
        let down_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(&down_shard, &rs)
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

    // Final norm — replicated
    let fn_t = vb
        .get(cfg.hidden_size, "model.norm.weight")
        .map_err(|e| FerrumError::model(format!("final_norm: {e}")))?;
    let fn_t = to_dev(&fn_t)?;
    let final_norm_w = GpuWeight::from_tensor(&fn_t, &rs)
        .map_err(|e| FerrumError::model(format!("final_norm: {e}")))?;

    // LM head — replicated (or tied to embed_tokens)
    let lm_t = vb
        .get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
        .or_else(|_| {
            vb.get(
                (cfg.vocab_size, cfg.hidden_size),
                "model.embed_tokens.weight",
            )
        })
        .map_err(|e| FerrumError::model(format!("lm_head: {e}")))?;
    let lm_t = to_dev(&lm_t)?;
    let lm_head_w = LinearWeight::Fp16(
        GpuWeight::from_tensor(&lm_t, &rs)
            .map_err(|e| FerrumError::model(format!("lm_head: {e}")))?,
    );

    // RoPE — compute and replicate
    let (rope_cos, rope_sin) = super::runner_weights::compute_rope_tables_for_tp(cfg, device, &rs)?;

    let weights = TransformerGpuWeights {
        embed_table,
        layers,
        final_norm_w,
        lm_head_w,
        rope_cos,
        rope_sin,
    };

    // Sharded dims for this rank's runner
    let dims = ModelDims {
        hidden_size: cfg.hidden_size,   // full (needed for replicated norms)
        intermediate_size: inter_shard, // sharded
        num_attention_heads: cfg.num_attention_heads / tp, // sharded
        num_kv_heads: cfg.num_kv_heads / tp, // sharded
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
        .map_err(|e| FerrumError::model(format!("sync: {e}")))?;

    Ok((weights, dims, rs))
}
