//! Pre-allocated GPU buffer pool for decode forward pass.
//!
//! All intermediate tensors for a single decode step are allocated once at
//! runner creation. Addresses never change, making CUDA Graph capture possible.
//!
//! Buffers are sized for `max_batch_size` tokens (default 1).
//! Batch decode uses the same buffers with `m = batch` for GEMMs.

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Model dimensions needed to compute buffer sizes.
#[derive(Debug, Clone)]
pub struct ModelDims {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    /// Whether model uses INT4 quantized weights.
    pub quantized: bool,
    /// Maximum batch size for decode. Buffers are allocated for this many tokens.
    /// Default: 1 (single-token decode).
    pub max_batch_size: usize,
}

/// Pre-allocated buffers for decode step(s).
///
/// All batch-scalable buffers are `[B * dim]` where B = `max_batch_size`.
/// Single-token decode (B=1) uses the same layout.
/// The decode runner writes to these buffers in-place each step.
pub struct DecodeBuffers {
    // ---- Batch-scaled buffers: [B * dim] ----
    /// After embedding lookup: [B * hidden_size]
    pub embed_out: CudaSlice<half::f16>,

    /// After input layer norm: [B * hidden_size]
    pub norm_out: CudaSlice<half::f16>,

    /// After fused QKV projection: [B * (q_dim + 2 * kv_dim)]
    pub qkv_out: CudaSlice<half::f16>,

    /// Q after norm+RoPE: [B * num_q_heads * head_dim]
    pub q_rotated: CudaSlice<half::f16>,

    /// K after norm+RoPE: [B * num_kv_heads * head_dim]
    pub k_rotated: CudaSlice<half::f16>,

    /// RoPE temp buffer for Q: [B * num_q_heads * head_dim]
    pub rope_q_temp: CudaSlice<half::f16>,

    /// RoPE temp buffer for K: [B * num_kv_heads * head_dim]
    pub rope_k_temp: CudaSlice<half::f16>,

    /// After attention: [B * num_q_heads * head_dim]
    pub attn_out: CudaSlice<half::f16>,

    /// After O projection: [B * hidden_size]
    pub o_proj_out: CudaSlice<half::f16>,

    /// Residual accumulator (double-buffer A): [B * hidden_size]
    pub residual: CudaSlice<half::f16>,

    /// After fused_add_rms_norm (normalized): [B * hidden_size]
    pub post_norm_out: CudaSlice<half::f16>,

    /// Residual accumulator (double-buffer B): [B * hidden_size]
    pub post_norm_residual: CudaSlice<half::f16>,

    /// After fused gate+up projection: [B * 2 * intermediate_size]
    pub gate_up_out: CudaSlice<half::f16>,

    /// After fused SiLU*mul: [B * intermediate_size]
    pub mlp_act: CudaSlice<half::f16>,

    /// After down projection: [B * hidden_size]
    pub down_out: CudaSlice<half::f16>,

    /// After final norm: [B * hidden_size]
    pub final_norm_out: CudaSlice<half::f16>,

    /// LM head output logits: [B * vocab_size]
    pub logits: CudaSlice<half::f16>,

    // ---- INT4 dequant scratch buffer ----
    // Temporary FP16 buffer for dequantized weights (sized for largest layer).
    // Only allocated when dims.quantized is true.
    /// Dequant temp: [max_weight_k * max_weight_n] fp16
    pub temp_dequant: Option<CudaSlice<half::f16>>,

    // ---- Single-token scratch for per-item ops in batch mode ----
    /// Per-item attention output scratch: [q_dim]
    /// Used when batch > 1 to avoid overwriting other items' attn results.
    pub scratch_attn: CudaSlice<half::f16>,

    // ---- Batched attention helper buffers ----
    /// Device pointer array for K caches: [max_batch_size] u64
    pub batched_k_ptrs: CudaSlice<u64>,
    /// Device pointer array for V caches: [max_batch_size] u64
    pub batched_v_ptrs: CudaSlice<u64>,
    /// Valid KV lengths per batch item: [max_batch_size] i32
    pub batched_kv_lens: CudaSlice<i32>,
    /// Positions per batch item (for batched RoPE): [max_batch_size] i32
    pub batched_positions: CudaSlice<i32>,

    // ---- Flash Decode partial buffers (f32, batch-scaled for batched flash decode) ----
    /// Partial V accumulation: [B * num_q_heads * MAX_SPLITS * head_dim]
    pub flash_partial_out: CudaSlice<f32>,

    /// Per-split max score: [B * num_q_heads * MAX_SPLITS]
    pub flash_partial_m: CudaSlice<f32>,

    /// Per-split exp sum: [B * num_q_heads * MAX_SPLITS]
    pub flash_partial_l: CudaSlice<f32>,

    /// Model dimensions (for reference)
    pub dims: ModelDims,
}

impl DecodeBuffers {
    /// Maximum number of KV splits for flash decoding.
    pub const MAX_SPLITS: usize = 32;

    /// Allocate all decode buffers on the given stream.
    ///
    /// Batch-scalable buffers are sized for `dims.max_batch_size` tokens.
    pub fn new(
        dims: ModelDims,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let b = dims.max_batch_size.max(1);
        let q_dim = dims.num_attention_heads * dims.head_dim;
        let kv_dim = dims.num_kv_heads * dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        Ok(Self {
            embed_out: unsafe { stream.alloc::<half::f16>(b * dims.hidden_size)? },
            norm_out: unsafe { stream.alloc::<half::f16>(b * dims.hidden_size)? },
            qkv_out: unsafe { stream.alloc::<half::f16>(b * qkv_dim)? },
            q_rotated: unsafe { stream.alloc::<half::f16>(b * q_dim)? },
            k_rotated: unsafe { stream.alloc::<half::f16>(b * kv_dim)? },
            rope_q_temp: unsafe { stream.alloc::<half::f16>(b * q_dim)? },
            rope_k_temp: unsafe { stream.alloc::<half::f16>(b * kv_dim)? },
            attn_out: unsafe { stream.alloc::<half::f16>(b * q_dim)? },
            o_proj_out: unsafe { stream.alloc::<half::f16>(b * dims.hidden_size)? },
            residual: unsafe { stream.alloc::<half::f16>(b * dims.hidden_size)? },
            post_norm_out: unsafe { stream.alloc::<half::f16>(b * dims.hidden_size)? },
            post_norm_residual: unsafe { stream.alloc::<half::f16>(b * dims.hidden_size)? },
            gate_up_out: unsafe { stream.alloc::<half::f16>(b * 2 * dims.intermediate_size)? },
            mlp_act: unsafe { stream.alloc::<half::f16>(b * dims.intermediate_size)? },
            down_out: unsafe { stream.alloc::<half::f16>(b * dims.hidden_size)? },
            final_norm_out: unsafe { stream.alloc::<half::f16>(b * dims.hidden_size)? },
            logits: unsafe { stream.alloc::<half::f16>(b * dims.vocab_size)? },
            temp_dequant: if dims.quantized {
                // Largest weight is gate_up_proj: [hidden_size, 2*intermediate_size]
                let max_k = dims.hidden_size;
                let max_n = 2 * dims.intermediate_size;
                Some(unsafe { stream.alloc::<half::f16>(max_k * max_n)? })
            } else {
                None
            },
            scratch_attn: unsafe { stream.alloc::<half::f16>(q_dim)? },
            batched_k_ptrs: unsafe { stream.alloc::<u64>(b)? },
            batched_v_ptrs: unsafe { stream.alloc::<u64>(b)? },
            batched_kv_lens: unsafe { stream.alloc::<i32>(b)? },
            batched_positions: unsafe { stream.alloc::<i32>(b)? },
            flash_partial_out: unsafe {
                stream
                    .alloc::<f32>(b * dims.num_attention_heads * Self::MAX_SPLITS * dims.head_dim)?
            },
            flash_partial_m: unsafe {
                stream.alloc::<f32>(b * dims.num_attention_heads * Self::MAX_SPLITS)?
            },
            flash_partial_l: unsafe {
                stream.alloc::<f32>(b * dims.num_attention_heads * Self::MAX_SPLITS)?
            },
            dims,
        })
    }

    /// Total GPU memory used by all buffers (in bytes).
    pub fn memory_bytes(&self) -> usize {
        let b = self.dims.max_batch_size.max(1);
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        // Batch-scaled fp16 buffers
        let batch_fp16 = b
            * (self.dims.hidden_size * 8 // embed, norm, o_proj, residual, post_norm, post_norm_res, down, final_norm
                + qkv_dim
                + q_dim * 3  // q_rotated, rope_q_temp, attn_out
                + kv_dim * 2 // k_rotated, rope_k_temp
                + 2 * self.dims.intermediate_size // gate_up
                + self.dims.intermediate_size     // mlp_act
                + self.dims.vocab_size); // logits

        // Single-token scratch
        let scratch_fp16 = q_dim; // scratch_attn

        // Flash decode partials (f32, batch-scaled)
        let flash_f32 =
            b * (q_dim * Self::MAX_SPLITS + self.dims.num_attention_heads * Self::MAX_SPLITS * 2);

        (batch_fp16 + scratch_fp16) * std::mem::size_of::<half::f16>()
            + flash_f32 * std::mem::size_of::<f32>()
    }
}
