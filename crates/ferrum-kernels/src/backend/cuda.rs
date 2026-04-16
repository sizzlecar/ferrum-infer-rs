//! CUDA backend using cuBLAS + custom PTX kernels.
//!
//! Buffer type: `CudaSlice<f16>` — all operations on a single CUDA stream.
//! Wraps the existing kernel launch functions from ferrum-kernels.
//!
//! NOTE: This file compiles only with `feature = "cuda"`.
//! Full implementation requires CUDA toolkit; will be tested on GPU machine.

#![allow(unused_variables, dead_code)]

use super::{AttnConfig, Backend};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DeviceRepr};
use half::f16;
use std::sync::Arc;

/// CUDA backend state: device + stream + cuBLAS handle.
///
/// Created once per inference session. All kernel launches go through
/// the same stream for implicit synchronization.
pub struct CudaState {
    pub device: Arc<CudaDevice>,
    pub stream: Arc<CudaStream>,
}

pub struct CudaBackend;

impl Backend for CudaBackend {
    type Buffer = CudaSlice<f16>;

    fn gemm(
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        // TODO: cuBLAS hgemm via cublas::linear_f16
        // Uses the same cuBLAS wrapper as CudaDecodeRunner::linear()
        todo!("CudaBackend::gemm — wire cuBLAS hgemm")
    }

    fn rms_norm(
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        // TODO: launch rms_norm_f16 kernel from PTX
        todo!("CudaBackend::rms_norm — launch rms_norm_f16")
    }

    fn fused_add_rms_norm(
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        // TODO: launch fused_add_rms_norm_f16 kernel
        todo!("CudaBackend::fused_add_rms_norm")
    }

    fn rope(
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        positions: &[u32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        // TODO: launch rope_f16 kernel
        // Upload positions to GPU, then dispatch
        todo!("CudaBackend::rope — launch rope_f16")
    }

    fn decode_attention(
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        out: &mut Self::Buffer,
        kv_len: usize,
        cfg: &AttnConfig,
    ) {
        // TODO: dispatch to decode_attention_f16 or flash_decode based on kv_len
        // if kv_len <= 256: launch decode_attention_f16
        // else: launch flash_decode_attn_f16 + flash_decode_reduce_f16
        todo!("CudaBackend::decode_attention")
    }

    fn flash_attention(
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        q_len: usize,
        kv_len: usize,
        pos_offset: usize,
        cfg: &AttnConfig,
    ) {
        // TODO: launch flash_attn_full_f16 kernel (Phase 5: upgrade .cu to F16)
        todo!("CudaBackend::flash_attention — needs flash_attn_full F16")
    }

    fn silu_mul(gate: &Self::Buffer, up: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        // TODO: launch fused_silu_mul_f16 kernel
        todo!("CudaBackend::silu_mul")
    }

    fn add(a: &Self::Buffer, b: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        // TODO: launch residual_add_f16 kernel
        todo!("CudaBackend::add")
    }

    fn copy(src: &Self::Buffer, dst: &mut Self::Buffer, len: usize) {
        // TODO: cudarc stream.memcpy_dtod
        todo!("CudaBackend::copy")
    }

    fn embedding_lookup(table: &Self::Buffer, ids: &[u32], out: &mut Self::Buffer, dim: usize) {
        // TODO: launch embedding_lookup_f16 kernel (already exists in .cu)
        todo!("CudaBackend::embedding_lookup")
    }

    fn alloc(len: usize) -> Self::Buffer {
        // TODO: CudaDevice::alloc_zeros(len)
        todo!("CudaBackend::alloc")
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        // TODO: dtoh + f16→f32 conversion
        todo!("CudaBackend::to_vec")
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        // TODO: f32→f16 conversion + htod
        todo!("CudaBackend::from_slice")
    }
}
