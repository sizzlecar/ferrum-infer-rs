//! Triton-rs-compiled broadcast bias add (in-place). F32 only.
//!
//! Triton 3.6 ABI (6 args = 4 user + 2 implicit):
//!   user args:    (data, bias, rows, cols)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! Computes per-row: `data[r, c] += bias[c]`. Mutates `data` directly.
//!
//! No native Rust wrapper exists yet (only the .cu kernel
//! `kernels/add_bias.cu`).
//!
//! One program per row, block dim from JSON. Cols capped at the BLOCK_SIZE
//! baked into the PTX (1024 for the f32 build).
//!
//! Only compiled with `--features cuda,triton-kernels`.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_meta::parse_meta;
use crate::triton_ptx;

const MODULE_NAME: &str = "triton_add_bias";

/// Triton-compiled in-place broadcast bias add: `data[r, c] += bias[c]`.
/// `data` is mutated in place. F32 only.
pub fn add_bias_triton(data: &Tensor, bias: &Tensor) -> candle_core::Result<()> {
    let dtype = data.dtype();
    let dims = data.dims();
    let cols = *dims.last().unwrap();
    let rows = data.elem_count() / cols;

    if dtype != DType::F32 {
        candle_core::bail!(
            "triton add_bias: only F32 currently has a triton-rs port (got {dtype:?})"
        );
    }

    let meta = parse_meta(triton_ptx::add_bias_f32::META)?;

    let cuda_dev = data.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func = cuda_dev.get_or_load_custom_func(
        kernel_name,
        MODULE_NAME,
        triton_ptx::add_bias_f32::PTX,
    )?;

    let grid_size = rows as u32;
    let block_size = (meta.num_warps * 32) as u32;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;

    let global_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.profile_scratch_size.max(1))?;

    let (data_s, data_l) = data.storage_and_layout();
    let (bias_s, bias_l) = bias.storage_and_layout();

    let data_cuda = match &*data_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("data must be on CUDA"),
    };
    let bias_cuda = match &*bias_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("bias must be on CUDA"),
    };

    let d = data_cuda.as_cuda_slice::<f32>()?;
    let b = bias_cuda.as_cuda_slice::<f32>()?;

    let d = d.slice(data_l.start_offset()..);
    let b = b.slice(bias_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&d);
    builder.arg(&b);
    builder.arg(&rows_i32);
    builder.arg(&cols_i32);
    builder.arg(&global_scratch);
    builder.arg(&profile_scratch);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: meta.shared_mem as u32,
    };
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("triton add_bias launch: {e}")))?;

    drop(data_s);
    drop(bias_s);

    Ok(())
}
