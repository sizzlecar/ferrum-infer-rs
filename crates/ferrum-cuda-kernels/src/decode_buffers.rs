//! Pre-allocated GPU buffer pool for decode forward pass.
//!
//! All intermediate tensors for a single decode step are allocated once at
//! runner creation. Addresses never change, making CUDA Graph capture possible.
//!
//! Decode always processes batch=1, seq_len=1, so all shapes are fixed.

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
}

/// Pre-allocated buffers for one decode step.
///
/// All buffers are [1, dim] (batch=1, seq=1) unless noted otherwise.
/// The decode runner writes to these buffers in-place each step.
pub struct DecodeBuffers {
    /// After embedding lookup: [hidden_size]
    pub embed_out: CudaSlice<half::f16>,

    /// After input layer norm: [hidden_size]
    pub norm_out: CudaSlice<half::f16>,

    /// After fused QKV projection: [q_dim + 2 * kv_dim]
    pub qkv_out: CudaSlice<half::f16>,

    /// Q after norm+RoPE: [num_q_heads * head_dim]
    pub q_rotated: CudaSlice<half::f16>,

    /// K after norm+RoPE: [num_kv_heads * head_dim]
    pub k_rotated: CudaSlice<half::f16>,

    /// RoPE temp buffer for Q output: [num_q_heads * head_dim]
    /// (can't do in-place RoPE, need separate input/output buffers)
    pub rope_q_temp: CudaSlice<half::f16>,

    /// RoPE temp buffer for K output: [num_kv_heads * head_dim]
    pub rope_k_temp: CudaSlice<half::f16>,

    /// After attention: [num_q_heads * head_dim]
    pub attn_out: CudaSlice<half::f16>,

    /// After O projection: [hidden_size]
    pub o_proj_out: CudaSlice<half::f16>,

    /// Residual accumulator: [hidden_size]
    pub residual: CudaSlice<half::f16>,

    /// After fused_add_rms_norm (normalized): [hidden_size]
    pub post_norm_out: CudaSlice<half::f16>,

    /// After fused_add_rms_norm (residual updated): [hidden_size]
    pub post_norm_residual: CudaSlice<half::f16>,

    /// After fused gate+up projection: [2 * intermediate_size]
    pub gate_up_out: CudaSlice<half::f16>,

    /// After fused SiLU*mul: [intermediate_size]
    pub mlp_act: CudaSlice<half::f16>,

    /// After down projection: [hidden_size]
    pub down_out: CudaSlice<half::f16>,

    /// After final norm: [hidden_size]
    pub final_norm_out: CudaSlice<half::f16>,

    /// LM head output logits: [vocab_size]
    pub logits: CudaSlice<half::f16>,

    // ---- Flash Decode partial buffers (f32 for numerical precision) ----

    /// Partial V accumulation: [num_q_heads * MAX_SPLITS * head_dim]
    pub flash_partial_out: CudaSlice<f32>,

    /// Per-split max score: [num_q_heads * MAX_SPLITS]
    pub flash_partial_m: CudaSlice<f32>,

    /// Per-split exp sum: [num_q_heads * MAX_SPLITS]
    pub flash_partial_l: CudaSlice<f32>,

    /// Model dimensions (for reference)
    pub dims: ModelDims,
}

impl DecodeBuffers {
    /// Maximum number of KV splits for flash decoding.
    pub const MAX_SPLITS: usize = 32;

    /// Allocate all decode buffers on the given stream.
    pub fn new(
        dims: ModelDims,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let q_dim = dims.num_attention_heads * dims.head_dim;
        let kv_dim = dims.num_kv_heads * dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        Ok(Self {
            embed_out: unsafe { stream.alloc::<half::f16>(dims.hidden_size)? },
            norm_out: unsafe { stream.alloc::<half::f16>(dims.hidden_size)? },
            qkv_out: unsafe { stream.alloc::<half::f16>(qkv_dim)? },
            q_rotated: unsafe { stream.alloc::<half::f16>(q_dim)? },
            k_rotated: unsafe { stream.alloc::<half::f16>(kv_dim)? },
            rope_q_temp: unsafe { stream.alloc::<half::f16>(q_dim)? },
            rope_k_temp: unsafe { stream.alloc::<half::f16>(kv_dim)? },
            attn_out: unsafe { stream.alloc::<half::f16>(q_dim)? },
            o_proj_out: unsafe { stream.alloc::<half::f16>(dims.hidden_size)? },
            residual: unsafe { stream.alloc::<half::f16>(dims.hidden_size)? },
            post_norm_out: unsafe { stream.alloc::<half::f16>(dims.hidden_size)? },
            post_norm_residual: unsafe { stream.alloc::<half::f16>(dims.hidden_size)? },
            gate_up_out: unsafe { stream.alloc::<half::f16>(2 * dims.intermediate_size)? },
            mlp_act: unsafe { stream.alloc::<half::f16>(dims.intermediate_size)? },
            down_out: unsafe { stream.alloc::<half::f16>(dims.hidden_size)? },
            final_norm_out: unsafe { stream.alloc::<half::f16>(dims.hidden_size)? },
            logits: unsafe { stream.alloc::<half::f16>(dims.vocab_size)? },
            flash_partial_out: unsafe {
                stream.alloc::<f32>(
                    dims.num_attention_heads * Self::MAX_SPLITS * dims.head_dim,
                )?
            },
            flash_partial_m: unsafe {
                stream.alloc::<f32>(dims.num_attention_heads * Self::MAX_SPLITS)?
            },
            flash_partial_l: unsafe {
                stream.alloc::<f32>(dims.num_attention_heads * Self::MAX_SPLITS)?
            },
            dims,
        })
    }

    /// Total GPU memory used by all buffers (in bytes).
    pub fn memory_bytes(&self) -> usize {
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;

        let fp16_elems = self.dims.hidden_size * 7 // embed, norm, o_proj, residual, post_norm, post_norm_res, down, final_norm
            + qkv_dim
            + q_dim * 2  // q_rotated, attn_out
            + kv_dim     // k_rotated
            + 2 * self.dims.intermediate_size  // gate_up
            + self.dims.intermediate_size      // mlp_act
            + self.dims.hidden_size            // final_norm_out
            + self.dims.vocab_size; // logits

        let flash_f32_elems = q_dim * Self::MAX_SPLITS  // partial_out (q_dim = num_q_heads * head_dim)
            + self.dims.num_attention_heads * Self::MAX_SPLITS * 2; // partial_m + partial_l

        fp16_elems * std::mem::size_of::<half::f16>()
            + flash_f32_elems * std::mem::size_of::<f32>()
    }
}
