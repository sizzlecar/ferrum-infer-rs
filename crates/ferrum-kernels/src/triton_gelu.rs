//! Triton-rs-compiled GELU (erf-based). F32 only.
//!
//! Triton 3.6 ABI (5 args = 3 user + 2 implicit):
//!   user args:    (x, output, len)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! Computes: `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.
//!
//! No native Rust wrapper exists yet (only the .cu kernel
//! `kernels/gelu.cu`).
//!
//! Only compiled with `--features cuda,triton-kernels`.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_meta::parse_meta;
use crate::triton_ptx;

const MODULE_NAME: &str = "triton_gelu";

/// Triton-compiled GELU (erf flavor). F32 only.
pub fn gelu_triton(x: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let shape = x.shape().clone();
    let elem_count = shape.elem_count();

    if dtype != DType::F32 {
        candle_core::bail!("triton gelu: only F32 currently has a triton-rs port (got {dtype:?})");
    }

    let meta = parse_meta(triton_ptx::gelu_f32::META)?;

    let cuda_dev = x.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func =
        cuda_dev.get_or_load_custom_func(kernel_name, MODULE_NAME, triton_ptx::gelu_f32::PTX)?;

    let block_size = (meta.num_warps * 32) as u32;
    let block_elems: u32 = 1024;
    let grid_size = ((elem_count as u32) + block_elems - 1) / block_elems;
    let n = elem_count as i32;

    let global_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.profile_scratch_size.max(1))?;

    let (x_s, x_l) = x.storage_and_layout();

    let x_cuda = match &*x_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("x must be on CUDA"),
    };

    let xs = x_cuda.as_cuda_slice::<f32>()?;
    let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

    let xs = xs.slice(x_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&xs);
    builder.arg(&out);
    builder.arg(&n);
    builder.arg(&global_scratch);
    builder.arg(&profile_scratch);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: meta.shared_mem as u32,
    };
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("triton gelu launch: {e}")))?;

    let output_storage = CudaStorage::wrap_cuda_slice(out, cuda_dev.clone());

    drop(x_s);

    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape,
        BackpropOp::none(),
        false,
    ))
}
