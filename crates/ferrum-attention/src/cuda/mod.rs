//! CUDA backend for FusedTransformer.
//!
//! Parallel implementation to `metal/` — uses cuBLAS for GEMM and custom
//! CUDA kernels (from ferrum-kernels) for element-wise ops.

pub mod transformer;

use crate::AttentionParams;

/// Check if CUDA device is available.
pub fn is_available() -> bool {
    cudarc::driver::CudaDevice::new(0).is_ok()
}

/// Run fused attention on CUDA (full sequence, not decode-only).
pub fn fused_attention(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], params: &AttentionParams) {
    // TODO: implement using flash_attn_full kernel
    // For now, fall back to CPU
    crate::cpu::fused_attention(q, k, v, out, params);
}
