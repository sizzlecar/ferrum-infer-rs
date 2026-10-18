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
//! These launchers operate on plain cudarc primitives (no candle Tensor).
//! They enqueue on the caller's stream so preceding writes and the caller's
//! completion observation cover the same ordered work.

use cudarc::driver::{CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::backend::KvTokenLayout;
use crate::ptx;

/// Launch causal INT8 paged attention for the last `query_rows` KV positions.
/// Q and output have the explicitly selected token/head layout. Row r may
/// attend through `valid_kv_len - query_rows + r`, including that position.
///
/// All dimensions match the FP16 path; the only difference is that
/// `k_pool` / `v_pool` are `int8_t` and `k_scales_pool` / `v_scales_pool`
/// carry per-token (per-kv-head) FP16 scales.
#[allow(clippy::too_many_arguments)]
pub fn launch_int8_paged_decode_attention(
    stream: &Arc<CudaStream>,
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
    query_rows: usize,
    query_layout: KvTokenLayout,
    block_size: usize,
    scale: f32,
) -> std::result::Result<(), String> {
    if num_q_heads == 0
        || num_kv_heads == 0
        || num_q_heads % num_kv_heads != 0
        || head_dim == 0
        || head_dim > 256
        || block_size == 0
        || query_rows == 0
        || query_rows > valid_kv_len
        || query_rows > u16::MAX as usize
        || !scale.is_finite()
    {
        return Err("invalid INT8 causal attention dimensions".into());
    }
    let q_elements = query_rows
        .checked_mul(num_q_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or("INT8 query dimensions overflow")?;
    if q_elements > i32::MAX as usize
        || q.len() < q_elements
        || output.len() < q_elements
        || valid_kv_len > i32::MAX as usize
        || num_q_heads > i32::MAX as usize
        || block_table.len() < valid_kv_len.div_ceil(block_size)
    {
        return Err("INT8 query/table buffers do not cover the declared shape".into());
    }
    let (token_stride, head_stride) = query_layout
        .strides(query_rows, num_q_heads, head_dim)
        .ok_or("INT8 query strides overflow")?;
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
    let qr = query_rows as i32;
    let qt = token_stride as i32;
    let qh = head_stride as i32;

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
    b.arg(&qr);
    b.arg(&qt);
    b.arg(&qh);
    b.arg(&bs);
    b.arg(&scale);

    // Dynamic shmem: 3 × valid_kv_len × sizeof(float)
    //   s_scores   (Q·K^T → softmax → folded weight)
    //   s_k_scales (preloaded for Step 1)
    //   s_v_scales (preloaded for Step 3, folded into s_scores)
    let shared_bytes = u32::try_from(
        valid_kv_len
            .checked_mul(12)
            .ok_or("INT8 shared memory overflow")?,
    )
    .map_err(|_| "INT8 shared memory overflow")?;
    let cfg = LaunchConfig {
        grid_dim: (num_q_heads as u32, query_rows as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    unsafe { b.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("int8_paged_decode_attention launch: {e}"))
}

/// Launch the INT8 KV cache append kernel.
///
/// Reads FP16 K/V in the explicitly selected token/head layout
/// and writes INT8 + per-(token, head) FP16 scale to the paged pool at
/// the slot indices given by `slot_mapping[t]` (a flat KV position =
/// physical_block * block_size + slot).
#[allow(clippy::too_many_arguments)]
pub fn launch_int8_kv_cache_append(
    stream: &Arc<CudaStream>,
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
    input_layout: KvTokenLayout,
) -> std::result::Result<(), String> {
    if head_dim == 0 || head_dim > 256 || num_tokens == 0 || num_kv_heads == 0 {
        return Err("invalid INT8 append dimensions (positive heads/tokens and head_dim in 1..=256 required)".into());
    }
    let elements = num_tokens
        .checked_mul(num_kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or("INT8 append dimensions overflow")?;
    if elements > i32::MAX as usize
        || k_in.len() < elements
        || v_in.len() < elements
        || slot_mapping.len() < num_tokens
        || num_kv_heads > u16::MAX as usize
    {
        return Err("INT8 append input buffers do not cover the declared shape".into());
    }
    let (token_stride, head_stride) = input_layout
        .strides(num_tokens, num_kv_heads, head_dim)
        .ok_or("INT8 append strides overflow")?;
    let func = stream
        .context()
        .load_module(ptx::INT8_PAGED_DECODE_ATTENTION.into())
        .map_err(|e| format!("load int8_paged_decode_attention module: {e}"))?
        .load_function("int8_kv_cache_append")
        .map_err(|e| format!("load int8_kv_cache_append func: {e}"))?;

    let nkv = num_kv_heads as i32;
    let hd = head_dim as i32;
    let nt = num_tokens as i32;
    let ts = token_stride as i32;
    let hs = head_stride as i32;

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
    b.arg(&ts);
    b.arg(&hs);

    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, num_kv_heads as u32, 1),
        block_dim: (head_dim.div_ceil(32) as u32 * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("int8_kv_cache_append launch: {e}"))
}
