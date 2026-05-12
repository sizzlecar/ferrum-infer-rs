//! Triton-rs-compiled in-place residual add: `a += b` (F32 only).
//!
//! No native .cu wrapper exists for the F32 in-place flavor (the .cu file
//! only ships an `_f16` variant), so this exposes a tensor-level entry
//! point used by the equivalence test.
//!
//! Triton 3.6 ABI:
//!   user args:    (a, b, n)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! Only compiled with `--features cuda,triton-kernels`.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_meta::parse_meta;
use crate::triton_ptx;

const MODULE_NAME: &str = "triton_residual_add_inplace";

/// Triton-compiled in-place add: `a[i] += b[i]`. Mutates `a` directly.
/// `a` and `b` must be the same shape on CUDA, F32. There is no native
/// equivalent for F32 — this exists for benchmarking + correctness testing
/// against `residual_add` (out-of-place).
pub fn residual_add_inplace_triton(a: &Tensor, b: &Tensor) -> candle_core::Result<()> {
    let dtype = a.dtype();
    let elem_count = a.shape().elem_count();

    if dtype != DType::F32 {
        candle_core::bail!(
            "triton residual_add_inplace: only F32 currently has a triton-rs port (got {dtype:?})"
        );
    }

    let meta = parse_meta(triton_ptx::residual_add_inplace_f32::META)?;

    let cuda_dev = a.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func = cuda_dev.get_or_load_custom_func(
        kernel_name,
        MODULE_NAME,
        triton_ptx::residual_add_inplace_f32::PTX,
    )?;

    let block_size = (meta.num_warps * 32) as u32;
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

    let a_in = a_in.slice(a_l.start_offset()..);
    let b_in = b_in.slice(b_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&a_in);
    builder.arg(&b_in);
    builder.arg(&n);
    builder.arg(&global_scratch);
    builder.arg(&profile_scratch);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: meta.shared_mem as u32,
    };
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("triton residual_add_inplace launch: {e}")))?;

    drop(a_s);
    drop(b_s);

    Ok(())
}
