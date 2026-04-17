//! CUDA backend skeleton.
//!
//! Buffer type: `CudaSlice<f16>` — all operations on a single CUDA stream.
//! Only method signatures match the Backend trait at the moment; the bodies
//! are `todo!()` placeholders. Real kernel launches land in Phase E on a GPU
//! machine — CI only has to prove `cargo check --features cuda` still passes.
//!
//! NOTE: This file compiles only with `feature = "cuda"`.

#![allow(unused_variables, dead_code)]

use super::{AttnConfig, Backend};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use half::f16;
use std::sync::Arc;

/// CUDA backend state: device + stream.
///
/// Created once per inference session. All kernel launches go through
/// the same stream for implicit synchronization. NCCL multi-GPU support
/// (for Tensor Parallel) will add rank/world_size here in Phase E.
pub struct CudaState {
    pub device: Arc<CudaDevice>,
    pub stream: Arc<CudaStream>,
}

pub struct CudaBackend;

impl Backend for CudaBackend {
    type Buffer = CudaSlice<f16>;
    type Context = CudaState;

    fn new_context() -> Self::Context {
        todo!("CudaBackend::new_context — init device + stream")
    }

    fn sync(_ctx: &mut Self::Context) {
        todo!("CudaBackend::sync — stream.synchronize()")
    }

    fn gemm(
        _ctx: &mut Self::Context,
        _a: &Self::Buffer,
        _b: &Self::Buffer,
        _out: &mut Self::Buffer,
        _m: usize,
        _n: usize,
        _k: usize,
    ) {
        todo!("CudaBackend::gemm — wire cuBLAS hgemm")
    }

    fn rms_norm(
        _ctx: &mut Self::Context,
        _x: &Self::Buffer,
        _w: &Self::Buffer,
        _eps: f32,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _dim: usize,
    ) {
        todo!("CudaBackend::rms_norm")
    }

    fn fused_add_rms_norm(
        _ctx: &mut Self::Context,
        _residual: &mut Self::Buffer,
        _x: &Self::Buffer,
        _w: &Self::Buffer,
        _eps: f32,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _dim: usize,
    ) {
        todo!("CudaBackend::fused_add_rms_norm")
    }

    fn flash_attention(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k: &Self::Buffer,
        _v: &Self::Buffer,
        _out: &mut Self::Buffer,
        _batch: usize,
        _q_len: usize,
        _kv_len: usize,
        _pos_offset: usize,
        _cfg: &AttnConfig,
    ) {
        todo!("CudaBackend::flash_attention — flash_attn_full F16 kernel")
    }

    fn copy_slice(
        _ctx: &mut Self::Context,
        _src: &Self::Buffer,
        _src_offset: usize,
        _dst: &mut Self::Buffer,
        _dst_offset: usize,
        _len: usize,
    ) {
        todo!("CudaBackend::copy_slice — cudaMemcpyDeviceToDevice")
    }

    fn embedding_lookup(
        _ctx: &mut Self::Context,
        _table: &Self::Buffer,
        _ids: &[u32],
        _out: &mut Self::Buffer,
        _dim: usize,
    ) {
        todo!("CudaBackend::embedding_lookup")
    }

    fn split_qkv(
        _ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _q: &mut Self::Buffer,
        _k: &mut Self::Buffer,
        _v: &mut Self::Buffer,
        _tokens: usize,
        _q_dim: usize,
        _kv_dim: usize,
    ) {
        todo!("CudaBackend::split_qkv")
    }

    fn fused_silu_mul_split(
        _ctx: &mut Self::Context,
        _gate_up: &Self::Buffer,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _im: usize,
    ) {
        todo!("CudaBackend::fused_silu_mul_split")
    }

    fn qk_norm_rope(
        _ctx: &mut Self::Context,
        _input: &Self::Buffer,
        _norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _output: &mut Self::Buffer,
        _tokens: usize,
        _heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _mode: i32,
    ) {
        todo!("CudaBackend::qk_norm_rope")
    }

    fn kv_cache_append_head_major(
        _ctx: &mut Self::Context,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _cache_len: usize,
        _cache_capacity: usize,
        _new_k_head_major: &Self::Buffer,
        _new_v_head_major: &Self::Buffer,
        _new_tokens: usize,
        _nkv: usize,
        _hd: usize,
    ) {
        todo!("CudaBackend::kv_cache_append_head_major")
    }

    fn transpose_head_to_token(
        _ctx: &mut Self::Context,
        _src: &Self::Buffer,
        _dst: &mut Self::Buffer,
        _tokens: usize,
        _heads: usize,
        _dim: usize,
    ) {
        todo!("CudaBackend::transpose_head_to_token")
    }

    fn add_inplace(
        _ctx: &mut Self::Context,
        _residual: &mut Self::Buffer,
        _x: &Self::Buffer,
        _len: usize,
    ) {
        todo!("CudaBackend::add_inplace")
    }

    fn alloc(_len: usize) -> Self::Buffer {
        todo!("CudaBackend::alloc — CudaDevice::alloc_zeros")
    }

    fn to_vec(_buf: &Self::Buffer, _len: usize) -> Vec<f32> {
        todo!("CudaBackend::to_vec — dtoh + f16→f32")
    }

    fn from_slice(_data: &[f32]) -> Self::Buffer {
        todo!("CudaBackend::from_slice — f32→f16 + htod")
    }
}
