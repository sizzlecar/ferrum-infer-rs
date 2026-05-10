//! INT8 KV cache kernels — Dim 5 (KV cache precision).
//!
//! Two launchers:
//!   - [`launch_int8_paged_decode_attention`]: read INT8 K/V (dequantized
//!     on the fly via per-token per-kv-head FP16 scales), compute paged
//!     decode attention. Mirrors [`crate::cuda_decode::launch_paged_decode_attention`]
//!     for the FP16 path.
//!   - [`launch_int8_kv_cache_append`]: take FP16 K/V tokens, compute
//!     per-token symmetric scale `s = max(|x|)/127`, write INT8 + scale
//!     into the paged pool.
//!
//! Storage layout:
//!   - K/V pool : `int8_t [num_blocks * block_size * num_kv_heads * head_dim]`
//!   - scales   : `__half  [num_blocks * block_size * num_kv_heads]`
//!   - block_table : `i32 [max_blocks_per_seq]`
//!
//! These launchers operate on plain cudarc primitives (no candle Tensor)
//! so they can be unit-tested independently of the `Backend` trait. Trait
//! integration (`BackendKvDtype<KvInt8>` for `CudaBackend`) lands in a
//! follow-up PR alongside model wire-up.

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::ptx;

/// Launch the INT8 paged decode attention kernel.
///
/// All dimensions match the FP16 path; the only difference is that
/// `k_pool` / `v_pool` are `int8_t` and `k_scales_pool` / `v_scales_pool`
/// carry per-token (per-kv-head) FP16 scales.
#[allow(clippy::too_many_arguments)]
pub fn launch_int8_paged_decode_attention(
    ctx: &Arc<CudaContext>,
    q: &CudaSlice<half::f16>,
    k_pool: &CudaSlice<i8>,
    v_pool: &CudaSlice<i8>,
    k_scales_pool: &CudaSlice<half::f16>,
    v_scales_pool: &CudaSlice<half::f16>,
    block_table: &CudaSlice<i32>,
    output: &mut CudaSlice<half::f16>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    valid_kv_len: usize,
    block_size: usize,
    scale: f32,
) -> std::result::Result<(), String> {
    let stream = ctx.default_stream();
    let func = stream
        .context()
        .load_module(ptx::INT8_PAGED_DECODE_ATTENTION.into())
        .map_err(|e| format!("load int8_paged_decode_attention module: {e}"))?
        .load_function("paged_decode_attention_int8")
        .map_err(|e| format!("load paged_decode_attention_int8 func: {e}"))?;

    let nq = num_q_heads as i32;
    let nkv = num_kv_heads as i32;
    let hd = head_dim as i32;
    let kvl = valid_kv_len as i32;
    let bs = block_size as i32;

    let mut b = stream.launch_builder(&func);
    b.arg(q);
    b.arg(k_pool);
    b.arg(v_pool);
    b.arg(k_scales_pool);
    b.arg(v_scales_pool);
    b.arg(block_table);
    b.arg(output);
    b.arg(&nq);
    b.arg(&nkv);
    b.arg(&hd);
    b.arg(&kvl);
    b.arg(&bs);
    b.arg(&scale);

    let shared_bytes = (valid_kv_len as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim: (num_q_heads as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    unsafe { b.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("int8_paged_decode_attention launch: {e}"))
}

/// Launch the INT8 KV cache append kernel.
///
/// Reads FP16 K/V at `[num_tokens, num_kv_heads, head_dim]` (token-major)
/// and writes INT8 + per-(token, head) FP16 scale to the paged pool at
/// the slot indices given by `slot_mapping[t]` (a flat KV position =
/// physical_block * block_size + slot).
#[allow(clippy::too_many_arguments)]
pub fn launch_int8_kv_cache_append(
    ctx: &Arc<CudaContext>,
    k_in: &CudaSlice<half::f16>,
    v_in: &CudaSlice<half::f16>,
    k_out_pool: &mut CudaSlice<i8>,
    v_out_pool: &mut CudaSlice<i8>,
    k_scales_pool: &mut CudaSlice<half::f16>,
    v_scales_pool: &mut CudaSlice<half::f16>,
    slot_mapping: &CudaSlice<i32>,
    num_tokens: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> std::result::Result<(), String> {
    if head_dim > 256 {
        return Err(format!(
            "int8_kv_cache_append: head_dim {head_dim} > 256 (kernel uses one thread per element)"
        ));
    }
    let stream = ctx.default_stream();
    let func = stream
        .context()
        .load_module(ptx::INT8_PAGED_DECODE_ATTENTION.into())
        .map_err(|e| format!("load int8_paged_decode_attention module: {e}"))?
        .load_function("int8_kv_cache_append")
        .map_err(|e| format!("load int8_kv_cache_append func: {e}"))?;

    let nkv = num_kv_heads as i32;
    let hd = head_dim as i32;
    let nt = num_tokens as i32;

    let mut b = stream.launch_builder(&func);
    b.arg(k_in);
    b.arg(v_in);
    b.arg(&mut *k_out_pool);
    b.arg(&mut *v_out_pool);
    b.arg(&mut *k_scales_pool);
    b.arg(&mut *v_scales_pool);
    b.arg(slot_mapping);
    b.arg(&nkv);
    b.arg(&hd);
    b.arg(&nt);

    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, num_kv_heads as u32, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("int8_kv_cache_append launch: {e}"))
}
