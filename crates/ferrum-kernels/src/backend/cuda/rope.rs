//! Fused RoPE (Rotary Position Embedding) for Q and K.
//!
//! Applies rotary embedding to both query and key in a single kernel launch.
//!
//! RoPE rotation for pair (x0, x1):
//!   x0_out = x0 * cos - x1 * sin
//!   x1_out = x1 * cos + x0 * sin

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::ptx;

const MODULE_NAME: &str = "rope";

/// Fused RoPE for Q and K tensors.
///
/// - `q`:   [num_q_heads, head_dim] on CUDA (single token, flattened from [1, num_q_heads, 1, head_dim])
/// - `k`:   [num_k_heads, head_dim] on CUDA
/// - `cos`: [head_dim/2] on CUDA — cos table row for current position
/// - `sin`: [head_dim/2] on CUDA — sin table row for current position
///
/// Returns: `(q_rotated, k_rotated)` with same shapes.
pub fn rope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    num_q_heads: usize,
    num_k_heads: usize,
    head_dim: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    let dtype = q.dtype();

    let func_name = match dtype {
        DType::F16 => "rope_f16",
        DType::F32 => "rope_f32",
        _ => candle_core::bail!("rope: unsupported dtype {dtype:?}"),
    };

    let cuda_dev = q.device().as_cuda_device()?;
    let func = cuda_dev.get_or_load_custom_func(func_name, MODULE_NAME, ptx::ROPE)?;

    let total_heads = num_q_heads + num_k_heads;
    let half_dim = head_dim / 2;
    let block_size = half_dim.min(1024) as u32;
    let grid_size = total_heads as u32;
    let num_q_heads_i32 = num_q_heads as i32;
    let num_k_heads_i32 = num_k_heads as i32;
    let head_dim_i32 = head_dim as i32;

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();

    let q_cuda = match &*q_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("q must be on CUDA"),
    };
    let k_cuda = match &*k_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("k must be on CUDA"),
    };
    let cos_cuda = match &*cos_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("cos must be on CUDA"),
    };
    let sin_cuda = match &*sin_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("sin must be on CUDA"),
    };

    let (q_out_storage, k_out_storage) = match dtype {
        DType::F16 => {
            let q_in = q_cuda.as_cuda_slice::<half::f16>()?;
            let k_in = k_cuda.as_cuda_slice::<half::f16>()?;
            let cos_in = cos_cuda.as_cuda_slice::<half::f16>()?;
            let sin_in = sin_cuda.as_cuda_slice::<half::f16>()?;

            let q_out = unsafe { cuda_dev.alloc::<half::f16>(num_q_heads * head_dim)? };
            let k_out = unsafe { cuda_dev.alloc::<half::f16>(num_k_heads * head_dim)? };

            let q_in = q_in.slice(q_l.start_offset()..);
            let k_in = k_in.slice(k_l.start_offset()..);
            let cos_in = cos_in.slice(cos_l.start_offset()..);
            let sin_in = sin_in.slice(sin_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&q_in);
            builder.arg(&k_in);
            builder.arg(&cos_in);
            builder.arg(&sin_in);
            builder.arg(&q_out);
            builder.arg(&k_out);
            builder.arg(&num_q_heads_i32);
            builder.arg(&num_k_heads_i32);
            builder.arg(&head_dim_i32);

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("rope kernel launch: {e}")))?;

            (
                CudaStorage::wrap_cuda_slice(q_out, cuda_dev.clone()),
                CudaStorage::wrap_cuda_slice(k_out, cuda_dev.clone()),
            )
        }
        DType::F32 => {
            let q_in = q_cuda.as_cuda_slice::<f32>()?;
            let k_in = k_cuda.as_cuda_slice::<f32>()?;
            let cos_in = cos_cuda.as_cuda_slice::<f32>()?;
            let sin_in = sin_cuda.as_cuda_slice::<f32>()?;

            let q_out = unsafe { cuda_dev.alloc::<f32>(num_q_heads * head_dim)? };
            let k_out = unsafe { cuda_dev.alloc::<f32>(num_k_heads * head_dim)? };

            let q_in = q_in.slice(q_l.start_offset()..);
            let k_in = k_in.slice(k_l.start_offset()..);
            let cos_in = cos_in.slice(cos_l.start_offset()..);
            let sin_in = sin_in.slice(sin_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&q_in);
            builder.arg(&k_in);
            builder.arg(&cos_in);
            builder.arg(&sin_in);
            builder.arg(&q_out);
            builder.arg(&k_out);
            builder.arg(&num_q_heads_i32);
            builder.arg(&num_k_heads_i32);
            builder.arg(&head_dim_i32);

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("rope kernel launch: {e}")))?;

            (
                CudaStorage::wrap_cuda_slice(q_out, cuda_dev.clone()),
                CudaStorage::wrap_cuda_slice(k_out, cuda_dev.clone()),
            )
        }
        _ => unreachable!(),
    };

    drop(q_s);
    drop(k_s);
    drop(cos_s);
    drop(sin_s);

    let q_shape = q.shape().clone();
    let k_shape = k.shape().clone();
    let q_rotated = Tensor::from_storage(
        Storage::Cuda(q_out_storage),
        q_shape,
        BackpropOp::none(),
        false,
    );
    let k_rotated = Tensor::from_storage(
        Storage::Cuda(k_out_storage),
        k_shape,
        BackpropOp::none(),
        false,
    );
    Ok((q_rotated, k_rotated))
}
