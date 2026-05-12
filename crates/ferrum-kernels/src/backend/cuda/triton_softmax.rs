//! Triton-rs-compiled softmax over the last dimension. F32 only.
//!
//! Triton 3.6 ABI (6 args = 4 user + 2 implicit):
//!   user args:    (input, output, rows, cols)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! Computes per-row: `out[r, c] = exp(x[r, c] - max(x[r, :])) /
//! sum_c exp(x[r, c] - max(x[r, :]))`.
//!
//! One program per row, block dim from JSON. Cols capped at the BLOCK_SIZE
//! baked into the PTX (1024 for the f32 build).
//!
//! No native Rust wrapper exists yet (only the .cu kernel
//! `kernels/softmax.cu`).
//!
//! Only compiled with `--features cuda,triton-kernels`.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_meta::parse_meta;
use crate::triton_ptx;

const MODULE_NAME: &str = "triton_softmax";

/// Triton-compiled softmax along the last dimension. F32 only.
pub fn softmax_triton(input: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = input.dtype();
    let dims = input.dims();
    let cols = *dims.last().unwrap();
    let rows = input.elem_count() / cols;

    if dtype != DType::F32 {
        candle_core::bail!(
            "triton softmax: only F32 currently has a triton-rs port (got {dtype:?})"
        );
    }

    let meta = parse_meta(triton_ptx::softmax_f32::META)?;

    let cuda_dev = input.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func =
        cuda_dev.get_or_load_custom_func(kernel_name, MODULE_NAME, triton_ptx::softmax_f32::PTX)?;

    let grid_size = rows as u32;
    let block_size = (meta.num_warps * 32) as u32;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;
    let elem_count = rows * cols;

    let global_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.profile_scratch_size.max(1))?;

    let (input_s, input_l) = input.storage_and_layout();

    let input_cuda = match &*input_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("input must be on CUDA"),
    };

    let inp = input_cuda.as_cuda_slice::<f32>()?;
    let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

    let inp = inp.slice(input_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&inp);
    builder.arg(&out);
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
        .map_err(|e| candle_core::Error::Msg(format!("triton softmax launch: {e}")))?;

    let output_storage = CudaStorage::wrap_cuda_slice(out, cuda_dev.clone());

    drop(input_s);

    let shape = input.shape().clone();
    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape,
        BackpropOp::none(),
        false,
    ))
}
