//! Triton-rs-compiled fused SiLU + multiply — drop-in alternative to
//! `fused_silu_mul.rs` (F32 only).
//!
//! Triton 3.6 ABI:
//!   user args:    (gate, up, output, n)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! Only compiled with `--features cuda,triton-kernels`.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_meta::parse_meta;
use crate::triton_ptx;

const MODULE_NAME: &str = "triton_fused_silu_mul";

/// Triton-compiled `output = silu(gate) * up`. Same semantics as
/// [`crate::fused_silu_mul`], different code path. F32 only.
pub fn fused_silu_mul_triton(gate: &Tensor, up: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = gate.dtype();
    let shape = gate.shape().clone();
    let elem_count = shape.elem_count();

    if dtype != DType::F32 {
        candle_core::bail!(
            "triton fused_silu_mul: only F32 currently has a triton-rs port (got {dtype:?})"
        );
    }

    let meta = parse_meta(triton_ptx::fused_silu_mul_f32::META)?;

    let cuda_dev = gate.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func = cuda_dev.get_or_load_custom_func(
        kernel_name,
        MODULE_NAME,
        triton_ptx::fused_silu_mul_f32::PTX,
    )?;

    let block_size = (meta.num_warps * 32) as u32;
    let block_elems: u32 = 1024;
    let grid_size = ((elem_count as u32) + block_elems - 1) / block_elems;
    let n = elem_count as i32;

    let global_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.profile_scratch_size.max(1))?;

    let (gate_s, gate_l) = gate.storage_and_layout();
    let (up_s, up_l) = up.storage_and_layout();

    let gate_cuda = match &*gate_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("gate must be on CUDA"),
    };
    let up_cuda = match &*up_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("up must be on CUDA"),
    };

    let g = gate_cuda.as_cuda_slice::<f32>()?;
    let u = up_cuda.as_cuda_slice::<f32>()?;
    let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

    let g = g.slice(gate_l.start_offset()..);
    let u = u.slice(up_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&g);
    builder.arg(&u);
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
        .map_err(|e| candle_core::Error::Msg(format!("triton fused_silu_mul launch: {e}")))?;

    let output_storage = CudaStorage::wrap_cuda_slice(out, cuda_dev.clone());

    drop(gate_s);
    drop(up_s);

    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape,
        BackpropOp::none(),
        false,
    ))
}
