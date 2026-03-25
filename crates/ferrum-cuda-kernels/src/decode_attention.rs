//! Single-query decode attention kernel with GQA support.
//!
//! More efficient than FlashAttention for decode (seq_len=1) because
//! FlashAttention's tile-based approach has overhead for tiny queries.
//!
//! Each block handles one query head:
//!   1. Compute Q·K^T scores for all kv positions
//!   2. Softmax with numerical stability
//!   3. Compute scores·V to produce output

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::ptx;

const MODULE_NAME: &str = "decode_attention";

/// Single-query attention for decode phase.
///
/// - `q`:       [num_q_heads, head_dim] on CUDA (single token)
/// - `k_cache`: [num_kv_heads, max_kv_len, head_dim] on CUDA
/// - `v_cache`: [num_kv_heads, max_kv_len, head_dim] on CUDA
/// - `num_q_heads`, `num_kv_heads`, `head_dim`: model dimensions
/// - `max_kv_len`:   total allocated KV cache length (buffer size)
/// - `valid_kv_len`: number of valid KV positions (for masking)
/// - `scale`:        1.0 / sqrt(head_dim)
///
/// Returns: [num_q_heads, head_dim] attention output.
pub fn decode_attention(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_kv_len: usize,
    valid_kv_len: usize,
    scale: f32,
) -> candle_core::Result<Tensor> {
    let dtype = q.dtype();

    if dtype != DType::F16 {
        candle_core::bail!("decode_attention: only F16 supported, got {dtype:?}");
    }

    let cuda_dev = q.device().as_cuda_device()?;
    let func = cuda_dev.get_or_load_custom_func(
        "decode_attention_f16",
        MODULE_NAME,
        ptx::DECODE_ATTENTION,
    )?;

    // One block per query head, threads cooperate over kv positions
    let block_size = 256u32;
    let grid_size = num_q_heads as u32;
    // Shared memory for attention scores: one float per kv position
    let shared_mem = (max_kv_len as u32) * 4; // f32 per position

    let num_q_heads_i32 = num_q_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let max_kv_len_i32 = max_kv_len as i32;
    let valid_kv_len_i32 = valid_kv_len as i32;

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k_cache.storage_and_layout();
    let (v_s, v_l) = v_cache.storage_and_layout();

    let q_cuda = match &*q_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("q must be on CUDA"),
    };
    let k_cuda = match &*k_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("k_cache must be on CUDA"),
    };
    let v_cuda = match &*v_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("v_cache must be on CUDA"),
    };

    let q_in = q_cuda.as_cuda_slice::<half::f16>()?;
    let k_in = k_cuda.as_cuda_slice::<half::f16>()?;
    let v_in = v_cuda.as_cuda_slice::<half::f16>()?;
    let out = unsafe { cuda_dev.alloc::<half::f16>(num_q_heads * head_dim)? };

    let q_in = q_in.slice(q_l.start_offset()..);
    let k_in = k_in.slice(k_l.start_offset()..);
    let v_in = v_in.slice(v_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&q_in);
    builder.arg(&k_in);
    builder.arg(&v_in);
    builder.arg(&out);
    builder.arg(&num_q_heads_i32);
    builder.arg(&num_kv_heads_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&max_kv_len_i32);
    builder.arg(&valid_kv_len_i32);
    builder.arg(&scale);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("decode_attention kernel launch: {e}")))?;

    drop(q_s);
    drop(k_s);
    drop(v_s);

    let output_storage = CudaStorage::wrap_cuda_slice(out, cuda_dev.clone());
    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        (num_q_heads, head_dim),
        BackpropOp::none(),
        false,
    ))
}
