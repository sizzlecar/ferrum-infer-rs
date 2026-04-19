//! CUDA fused transformer layer — single CUDA stream, zero CPU-GPU sync mid-layer.
//!
//! Mirrors `metal/transformer.rs` but uses:
//! - cuBLAS GEMM for Q/K/V projections, O projection, MLP
//! - Custom CUDA kernels for RMSNorm, RoPE, SiLU, residual add
//! - Custom flash attention (full sequence + decode)
//!
//! All ops on a single CUDA stream. KV cache managed on GPU.

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// GPU-resident layer weights.
pub struct CudaLayerWeights {
    pub input_ln_w: CudaSlice<f32>,
    pub q_proj_w: CudaSlice<f32>,
    pub k_proj_w: CudaSlice<f32>,
    pub v_proj_w: CudaSlice<f32>,
    pub o_proj_w: CudaSlice<f32>,
    pub q_norm_w: CudaSlice<f32>,
    pub k_norm_w: CudaSlice<f32>,
    pub has_qk_norm: bool,
    pub post_ln_w: CudaSlice<f32>,
    pub gate_proj_w: CudaSlice<f32>,
    pub up_proj_w: CudaSlice<f32>,
    pub down_proj_w: CudaSlice<f32>,
    pub attn_scale: Option<CudaSlice<f32>>,
    pub mlp_scale: Option<CudaSlice<f32>>,
}

/// GPU KV cache.
pub struct CudaKvCache {
    pub k_buf: CudaSlice<f32>, // [nkv, max_len, hd]
    pub v_buf: CudaSlice<f32>,
    pub len: usize,
    pub max_len: usize,
}

impl CudaKvCache {
    pub fn new(dev: &Arc<CudaDevice>, nkv: usize, hd: usize, max_len: usize) -> Self {
        let size = nkv * max_len * hd;
        Self {
            k_buf: dev.alloc_zeros::<f32>(size).unwrap(),
            v_buf: dev.alloc_zeros::<f32>(size).unwrap(),
            len: 0,
            max_len,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

/// Pre-allocated scratch buffers for one layer forward.
pub struct CudaLayerScratch {
    pub ln_out: CudaSlice<f32>,
    pub q_buf: CudaSlice<f32>,
    pub k_buf: CudaSlice<f32>,
    pub v_buf: CudaSlice<f32>,
    pub attn_out: CudaSlice<f32>,
    pub o_out: CudaSlice<f32>,
    pub hidden: CudaSlice<f32>,
    pub post_ln: CudaSlice<f32>,
    pub gate_buf: CudaSlice<f32>,
    pub up_buf: CudaSlice<f32>,
    pub silu_out: CudaSlice<f32>,
    pub mlp_out: CudaSlice<f32>,
    pub output: CudaSlice<f32>,
}

impl CudaLayerScratch {
    pub fn new(
        dev: &Arc<CudaDevice>,
        tokens: usize,
        h: usize,
        im: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) -> Self {
        Self {
            ln_out: dev.alloc_zeros::<f32>(tokens * h).unwrap(),
            q_buf: dev.alloc_zeros::<f32>(tokens * nh * hd).unwrap(),
            k_buf: dev.alloc_zeros::<f32>(tokens * nkv * hd).unwrap(),
            v_buf: dev.alloc_zeros::<f32>(tokens * nkv * hd).unwrap(),
            attn_out: dev.alloc_zeros::<f32>(tokens * nh * hd).unwrap(),
            o_out: dev.alloc_zeros::<f32>(tokens * h).unwrap(),
            hidden: dev.alloc_zeros::<f32>(tokens * h).unwrap(),
            post_ln: dev.alloc_zeros::<f32>(tokens * h).unwrap(),
            gate_buf: dev.alloc_zeros::<f32>(tokens * im).unwrap(),
            up_buf: dev.alloc_zeros::<f32>(tokens * im).unwrap(),
            silu_out: dev.alloc_zeros::<f32>(tokens * im).unwrap(),
            mlp_out: dev.alloc_zeros::<f32>(tokens * h).unwrap(),
            output: dev.alloc_zeros::<f32>(tokens * h).unwrap(),
        }
    }
}

/// CUDA transformer config (mirrors MetalTransformerConfig).
pub struct CudaTransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
}

/// Holds CUDA device state for the fused transformer.
pub struct CudaTransformerState {
    pub dev: Arc<CudaDevice>,
    pub blas: Arc<CudaBlas>,
    pub weights: Vec<CudaLayerWeights>,
    pub kv: Vec<CudaKvCache>,
    pub cos_buf: CudaSlice<f32>,
    pub sin_buf: CudaSlice<f32>,
    pub cfg: CudaTransformerConfig,
    pub scratch: Option<CudaLayerScratch>,
    pub max_scratch_tokens: usize,
    pub norm_w_buf: CudaSlice<f32>,
}

// TODO: Implement cuda_layer_forward() using:
// - cuBLAS sgemm for Q/K/V/O/gate/up/down projections
// - ferrum-kernels rms_norm_f32 for layer norms
// - ferrum-kernels rope_f32 for RoPE
// - ferrum-kernels fused_silu_mul_f32 for SiLU gate
// - ferrum-kernels residual_add_f32 for residual connections
// - ferrum-kernels flash_attn_full_f32 for attention
// - ferrum-kernels softmax_f32 (used internally by attention)
//
// All ops on a single CUDA stream. KV cache append is a device-to-device copy.
// Output written to scratch.output. Caller reads after all layers.
