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

use cudarc::driver::{CudaContext, CudaSlice, CudaView, LaunchConfig, PushKernelArg};
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
    block_table: &CudaView<'_, i32>,
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

/// Launch the fused split-QKV → norm → RoPE → quantize → INT8 paged
/// write kernel. One launch replaces the 4-kernel chain
/// (`split_qkv` + `qk_norm_rope` ×3 + `int8_kv_cache_append`) used by
/// the decode hot path. Q lands in FP16 head-major scratch; K/V are
/// quantized per-(token, kv_head) with FP16 scales and written into
/// the paged INT8 pool addressed by `block_table` + `cache_len_before`.
#[allow(clippy::too_many_arguments)]
pub fn launch_split_qkv_norm_rope_into_int8_paged_cache(
    ctx: &Arc<CudaContext>,
    qkv: &CudaSlice<half::f16>,
    q_norm_w: &CudaSlice<half::f16>,
    k_norm_w: &CudaSlice<half::f16>,
    cos_tab: &CudaSlice<half::f16>,
    sin_tab: &CudaSlice<half::f16>,
    q_out: &mut CudaSlice<half::f16>,
    cache_k: &mut CudaSlice<i8>,
    cache_v: &mut CudaSlice<i8>,
    cache_k_scales: &mut CudaSlice<half::f16>,
    cache_v_scales: &mut CudaSlice<half::f16>,
    block_table: &CudaView<'_, i32>,
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    eps: f32,
    qk_mode: i32,
    cache_len_before: usize,
    block_size: usize,
) -> std::result::Result<(), String> {
    if head_dim > 128 {
        return Err(format!(
            "split_qkv_norm_rope_into_int8_paged_cache: head_dim {head_dim} > 128 \
             (kernel uses 4-element register array per lane)"
        ));
    }
    let stream = ctx.default_stream();
    let func = stream
        .context()
        .load_module(ptx::SPLIT_QKV_NORM_ROPE_INTO_INT8_PAGED_CACHE.into())
        .map_err(|e| format!("load split_qkv_norm_rope_into_int8_paged_cache module: {e}"))?
        .load_function("split_qkv_norm_rope_into_int8_paged_cache_f16")
        .map_err(|e| format!("load fn: {e}"))?;

    let tokens_i32 = tokens as i32;
    let q_heads_i32 = q_heads as i32;
    let kv_heads_i32 = kv_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let pos_offset_i32 = pos_offset as i32;
    let qk_mode_i32 = qk_mode;
    let cache_len_i32 = cache_len_before as i32;
    let block_size_i32 = block_size as i32;

    let mut b = stream.launch_builder(&func);
    b.arg(qkv);
    b.arg(q_norm_w);
    b.arg(k_norm_w);
    b.arg(cos_tab);
    b.arg(sin_tab);
    b.arg(&mut *q_out);
    b.arg(&mut *cache_k);
    b.arg(&mut *cache_v);
    b.arg(&mut *cache_k_scales);
    b.arg(&mut *cache_v_scales);
    b.arg(block_table);
    b.arg(&tokens_i32);
    b.arg(&q_heads_i32);
    b.arg(&kv_heads_i32);
    b.arg(&head_dim_i32);
    b.arg(&pos_offset_i32);
    b.arg(&eps);
    b.arg(&qk_mode_i32);
    b.arg(&cache_len_i32);
    b.arg(&block_size_i32);

    let total_heads = q_heads + 2 * kv_heads;
    let cfg = LaunchConfig {
        grid_dim: (tokens as u32, total_heads as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("split_qkv_norm_rope_into_int8_paged_cache launch: {e}"))
}
