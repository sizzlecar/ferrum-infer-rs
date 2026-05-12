//! Triton-rs-compiled RMS norm — drop-in alternative to `rms_norm.rs`.
//!
//! Loads PTX produced by triton-rs's `compile_mlir` from `triton_ptx/`,
//! launches via cudarc with the Triton 3.6 ABI:
//!
//!   user args:    (input, weight, output, row_size, inv_n, eps)
//!   implicit:     (global_scratch, profile_scratch)
//!
//! Caller signature is identical to `rms_norm::rms_norm` so the two paths
//! are interchangeable; the test in `tests/triton_rms_norm_eq.rs` runs
//! both on the same input and asserts element-wise equivalence.
//!
//! Only compiled with `--features cuda,triton-kernels`. Without the
//! feature, this module is empty.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::triton_ptx;

const MODULE_NAME: &str = "triton_rms_norm";

/// Triton-compiled RMS normalization. Same semantics as
/// [`crate::rms_norm`], different code path.
pub fn rms_norm_triton(input: &Tensor, weight: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    let dtype = input.dtype();
    let dims = input.dims();
    let row_size = *dims.last().unwrap();
    let num_rows = input.elem_count() / row_size;

    if dtype != DType::F32 {
        candle_core::bail!(
            "triton rms_norm: only F32 currently has a triton-rs port (got {dtype:?})"
        );
    }

    // Parse metadata (kernel name + scratch sizes + shared mem) once.
    let meta = parse_meta(triton_ptx::rms_norm_f32::META)?;

    let cuda_dev = input.device().as_cuda_device()?;
    let kernel_name: &'static str = Box::leak(meta.name.into_boxed_str());
    let func = cuda_dev.get_or_load_custom_func(
        kernel_name,
        MODULE_NAME,
        triton_ptx::rms_norm_f32::PTX,
    )?;

    // Triton kernels launch one program per row; block dim is num_warps * 32.
    let grid_size = num_rows as u32;
    let block_size = (meta.num_warps * 32) as u32;
    let row_size_i32 = row_size as i32;
    let inv_n: f32 = 1.0 / row_size as f32;
    let elem_count = num_rows * row_size;

    // Implicit scratch buffers (Triton 3.6 ABI). Empty alloc is invalid;
    // bump to 1 byte when the kernel doesn't use scratch.
    let global_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        cuda_dev.alloc_zeros::<u8>(meta.profile_scratch_size.max(1))?;

    let (input_s, input_l) = input.storage_and_layout();
    let (weight_s, weight_l) = weight.storage_and_layout();

    let input_cuda = match &*input_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("input must be on CUDA"),
    };
    let weight_cuda = match &*weight_s {
        Storage::Cuda(cs) => cs,
        _ => candle_core::bail!("weight must be on CUDA"),
    };

    let inp = input_cuda.as_cuda_slice::<f32>()?;
    let w = weight_cuda.as_cuda_slice::<f32>()?;
    let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

    let inp = inp.slice(input_l.start_offset()..);
    let w = w.slice(weight_l.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&inp);
    builder.arg(&w);
    builder.arg(&out);
    builder.arg(&row_size_i32);
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
        .map_err(|e| candle_core::Error::Msg(format!("triton rms_norm launch: {e}")))?;

    let output_storage = CudaStorage::wrap_cuda_slice(out, cuda_dev.clone());

    drop(input_s);
    drop(weight_s);

    let shape = input.shape().clone();
    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape,
        BackpropOp::none(),
        false,
    ))
}

struct Meta {
    name: String,
    num_warps: u32,
    shared_mem: u64,
    global_scratch_size: usize,
    profile_scratch_size: usize,
}

fn parse_meta(json: &str) -> candle_core::Result<Meta> {
    fn pick<'a>(s: &'a str, key: &str) -> Option<&'a str> {
        let needle = format!("\"{key}\":");
        let start = s.find(&needle)? + needle.len();
        let rest = s[start..].trim_start();
        if rest.starts_with('"') {
            let inner = &rest[1..];
            let end = inner.find('"')?;
            Some(&inner[..end])
        } else {
            let end = rest
                .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
                .unwrap_or(rest.len());
            Some(rest[..end].trim())
        }
    }
    let parse_u = |k: &str| -> u64 {
        pick(json, k)
            .and_then(|v| v.trim_matches('"').parse::<u64>().ok())
            .unwrap_or(0)
    };
    let name = pick(json, "name")
        .ok_or_else(|| candle_core::Error::Msg(format!("triton meta: missing name in {json}")))?
        .to_string();
    Ok(Meta {
        name,
        num_warps: parse_u("num_warps") as u32,
        shared_mem: parse_u("shared_mem"),
        global_scratch_size: parse_u("global_scratch_size") as usize,
        profile_scratch_size: parse_u("profile_scratch_size") as usize,
    })
}
