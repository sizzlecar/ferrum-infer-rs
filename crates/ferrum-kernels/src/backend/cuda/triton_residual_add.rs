//! Triton-rs-compiled residual add — drop-in alternative to `residual_add.rs`.
//!
//! Loads PTX produced by triton-rs's `compile_mlir` from `triton_ptx/`,
//! launches via cudarc with the Triton 3.6 ABI:
//!
//!   user args:    (a, b, output, n)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! Caller signature is identical to `residual_add::residual_add` (F32 only)
//! so the two paths are interchangeable. The element-wise equivalence test
//! lives in `tests/triton_residual_add_eq.rs`.
//!
//! Only compiled with `--features cuda,triton-kernels`. Without the
//! feature, this module is empty.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_meta::parse_meta;
use crate::triton_ptx;

const MODULE_NAME: &str = "triton_residual_add";

/// Triton-compiled element-wise add: `output = a + b`. Same semantics as
/// [`crate::residual_add`], different code path. F32 only.
pub fn residual_add_triton(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = a.dtype();
    let shape = a.shape().clone();
    let elem_count = shape.elem_count();

    if dtype != DType::F32 {
        candle_core::bail!(
            "triton residual_add: only F32 currently has a triton-rs port (got {dtype:?})"
        );
    }

    let meta = parse_meta(triton_ptx::residual_add_f32::META)?;

    let cuda_dev = a.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func = cuda_dev.get_or_load_custom_func(
        kernel_name,
        MODULE_NAME,
        triton_ptx::residual_add_f32::PTX,
    )?;

    // Triton kernel uses BLOCK_SIZE=1024 (one program covers up to 1024 elems).
    // We mirror the .cu launch shape: grid covers full elem_count, block dim
    // is num_warps * 32 from JSON.
    let block_size = (meta.num_warps * 32) as u32;
    // BLOCK_SIZE constant baked into PTX is 1024 (typical for triton element-
    // wise kernels). Each program handles 1024 elements.
    let block_elems: u32 = 1024;
    let grid_size = ((elem_count as u32) + block_elems - 1) / block_elems;
    let n = elem_count as i32;

    let global_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.profile_scratch_size.max(1))?;

    let (a_s, a_l) = a.storage_and_layout();
    let (b_s, b_l) = b.storage_and_layout();

    let a_cuda = match &*a_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("a must be on CUDA"),
    };
    let b_cuda = match &*b_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("b must be on CUDA"),
    };

    let a_in = a_cuda.as_cuda_slice::<f32>()?;
    let b_in = b_cuda.as_cuda_slice::<f32>()?;
    let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

    let a_in = a_in.slice(a_l.start_offset()..);
    let b_in = b_in.slice(b_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&a_in);
    builder.arg(&b_in);
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
        .map_err(|e| candle_core::Error::Msg(format!("triton residual_add launch: {e}")))?;

    let output_storage = CudaStorage::wrap_cuda_slice(out, cuda_dev.clone());

    drop(a_s);
    drop(b_s);

    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape,
        BackpropOp::none(),
        false,
    ))
}
