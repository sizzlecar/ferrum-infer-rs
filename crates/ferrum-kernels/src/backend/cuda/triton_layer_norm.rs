//! Triton-rs-compiled LayerNorm.
//!
//! No native Rust wrapper exists yet (only the .cu kernel
//! `kernels/layer_norm.cu`), so this is a fresh tensor-level entry point
//! that follows the same convention as the other triton wrappers.
//!
//! Triton 3.6 ABI (9 args = 7 user + 2 implicit):
//!   user args:    (x, gamma, beta, output, dim, inv_dim, eps)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! Computes:
//!   mean = mean(x[r, :])
//!   var  = mean((x[r, :] - mean)^2)
//!   out[r, c] = (x[r, c] - mean) / sqrt(var + eps) * gamma[c] + beta[c]
//!
//! One program per row, block dim from JSON. Hidden size capped at the
//! BLOCK_SIZE baked into the PTX (1024 for the f32 build).
//!
//! Only compiled with `--features cuda,triton-kernels`.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_meta::parse_meta;
use crate::triton_ptx;

const MODULE_NAME: &str = "triton_layer_norm";

/// Triton-compiled LayerNorm. F32 only.
///
/// - `x`:      [..., dim] on CUDA
/// - `gamma`:  [dim] on CUDA
/// - `beta`:   [dim] on CUDA
/// - `eps`:    normalization epsilon
///
/// Returns: tensor with same shape as `x`.
pub fn layer_norm_triton(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let dims = x.dims();
    let dim = *dims.last().unwrap();
    let num_rows = x.elem_count() / dim;

    if dtype != DType::F32 {
        candle_core::bail!(
            "triton layer_norm: only F32 currently has a triton-rs port (got {dtype:?})"
        );
    }

    let meta = parse_meta(triton_ptx::layer_norm_f32::META)?;

    let cuda_dev = x.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func = cuda_dev.get_or_load_custom_func(
        kernel_name,
        MODULE_NAME,
        triton_ptx::layer_norm_f32::PTX,
    )?;

    let grid_size = num_rows as u32;
    let block_size = (meta.num_warps * 32) as u32;
    let dim_i32 = dim as i32;
    let inv_dim: f32 = 1.0 / dim as f32;
    let elem_count = num_rows * dim;

    let global_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.profile_scratch_size.max(1))?;

    let (x_s, x_l) = x.storage_and_layout();
    let (gamma_s, gamma_l) = gamma.storage_and_layout();
    let (beta_s, beta_l) = beta.storage_and_layout();

    let x_cuda = match &*x_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("x must be on CUDA"),
    };
    let gamma_cuda = match &*gamma_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("gamma must be on CUDA"),
    };
    let beta_cuda = match &*beta_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("beta must be on CUDA"),
    };

    let xs = x_cuda.as_cuda_slice::<f32>()?;
    let g = gamma_cuda.as_cuda_slice::<f32>()?;
    let bt = beta_cuda.as_cuda_slice::<f32>()?;
    let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

    let xs = xs.slice(x_l.start_offset()..);
    let g = g.slice(gamma_l.start_offset()..);
    let bt = bt.slice(beta_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&xs);
    builder.arg(&g);
    builder.arg(&bt);
    builder.arg(&out);
    builder.arg(&dim_i32);
    builder.arg(&inv_dim);
    builder.arg(&eps);
    builder.arg(&global_scratch);
    builder.arg(&profile_scratch);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: meta.shared_mem as u32,
    };
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("triton layer_norm launch: {e}")))?;

    let output_storage = CudaStorage::wrap_cuda_slice(out, cuda_dev.clone());

    drop(x_s);
    drop(gamma_s);
    drop(beta_s);

    let shape = x.shape().clone();
    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape,
        BackpropOp::none(),
        false,
    ))
}
