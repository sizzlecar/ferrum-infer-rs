//! Triton-rs-compiled fused residual-add + RMS norm — drop-in alternative
//! to `fused_add_rms_norm.rs` (F32 only).
//!
//! Triton 3.6 ABI (10 args = 8 user + 2 implicit):
//!   user args:    (input, residual, weight, output, residual_out,
//!                  hidden_size, inv_n, eps)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! The native .cu wrapper passes only `eps`; the triton version also
//! consumes a precomputed `inv_n = 1 / hidden_size`.
//!
//! Only compiled with `--features cuda,triton-kernels`.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_meta::parse_meta;
use crate::triton_ptx;

const MODULE_NAME: &str = "triton_fused_add_rms_norm";

/// Triton-compiled fused: `residual += input; output = rms_norm(residual)`.
/// Returns `(normalized_output, updated_residual)`. F32 only.
pub fn fused_add_rms_norm_triton(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> candle_core::Result<(Tensor, Tensor)> {
    let dtype = input.dtype();
    let (num_tokens, hidden_size) = input.dims2()?;

    if dtype != DType::F32 {
        candle_core::bail!(
            "triton fused_add_rms_norm: only F32 currently has a triton-rs port (got {dtype:?})"
        );
    }

    let meta = parse_meta(triton_ptx::fused_add_rms_norm_f32::META)?;

    let cuda_dev = input.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func = cuda_dev.get_or_load_custom_func(
        kernel_name,
        MODULE_NAME,
        triton_ptx::fused_add_rms_norm_f32::PTX,
    )?;

    // One program per row, block dim from JSON.
    let grid_size = num_tokens as u32;
    let block_size = (meta.num_warps * 32) as u32;
    let hidden_i32 = hidden_size as i32;
    let inv_n: f32 = 1.0 / hidden_size as f32;
    let elem_count = num_tokens * hidden_size;

    let global_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.profile_scratch_size.max(1))?;

    let (input_s, input_l) = input.storage_and_layout();
    let (residual_s, residual_l) = residual.storage_and_layout();
    let (weight_s, weight_l) = weight.storage_and_layout();

    let input_cuda = match &*input_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("input must be on CUDA"),
    };
    let residual_cuda = match &*residual_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("residual must be on CUDA"),
    };
    let weight_cuda = match &*weight_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("weight must be on CUDA"),
    };

    let inp = input_cuda.as_cuda_slice::<f32>()?;
    let res = residual_cuda.as_cuda_slice::<f32>()?;
    let w = weight_cuda.as_cuda_slice::<f32>()?;

    let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };
    let res_out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

    let inp = inp.slice(input_l.start_offset()..);
    let res = res.slice(residual_l.start_offset()..);
    let w = w.slice(weight_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&inp);
    builder.arg(&res);
    builder.arg(&w);
    builder.arg(&out);
    builder.arg(&res_out);
    builder.arg(&hidden_i32);
    builder.arg(&inv_n);
    builder.arg(&eps);
    builder.arg(&global_scratch);
    builder.arg(&profile_scratch);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: meta.shared_mem as u32,
    };
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("triton fused_add_rms_norm launch: {e}")))?;

    let output_storage = CudaStorage::wrap_cuda_slice(out, cuda_dev.clone());
    let residual_out_storage = CudaStorage::wrap_cuda_slice(res_out, cuda_dev.clone());

    drop(input_s);
    drop(residual_s);
    drop(weight_s);

    let shape = input.shape().clone();
    let normalized = Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape.clone(),
        BackpropOp::none(),
        false,
    );
    let residual_updated = Tensor::from_storage(
        Storage::Cuda(residual_out_storage),
        shape,
        BackpropOp::none(),
        false,
    );
    Ok((normalized, residual_updated))
}
