//! Fused residual-add + RMS normalization.
//!
//! Replaces 2 separate kernel launches (add + rms_norm) with 1:
//!   residual_out = input + residual
//!   output       = rms_norm(residual_out, weight, eps)
//!
//! Memory bandwidth saved: reads input+residual once instead of twice
//! (once for add, once for norm).

use candle_core::{BackpropOp, CudaStorage, DType, Storage, Tensor};
use std::sync::OnceLock;

const CUDA_SRC: &str = include_str!("../kernels/fused_add_rms_norm.cu");
const MODULE_NAME: &str = "fused_add_rms_norm";

static COMPILED_PTX: OnceLock<cudarc::nvrtc::Ptx> = OnceLock::new();

fn get_ptx() -> candle_core::Result<&'static cudarc::nvrtc::Ptx> {
    COMPILED_PTX
        .get_or_try_init(|| {
            let opts = cudarc::nvrtc::CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            };
            cudarc::nvrtc::safe::compile_ptx_with_opts(CUDA_SRC, opts)
                .map_err(|e| candle_core::Error::Msg(format!("nvrtc compile: {e}")))
        })
        .map_err(|e| candle_core::Error::Msg(format!("PTX init: {e}")))
}

/// Fused: residual_out = input + residual; output = rms_norm(residual_out, weight, eps)
///
/// - `input`:    [num_tokens, hidden_size] on CUDA
/// - `residual`: [num_tokens, hidden_size] on CUDA
/// - `weight`:   [hidden_size] on CUDA
/// - `eps`:      normalization epsilon
///
/// Returns: `(normalized_output, updated_residual)` both [num_tokens, hidden_size]
pub fn fused_add_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> candle_core::Result<(Tensor, Tensor)> {
    let dtype = input.dtype();
    let (num_tokens, hidden_size) = input.dims2()?;

    let func_name = match dtype {
        DType::F16 => "fused_add_rms_norm_f16",
        DType::F32 => "fused_add_rms_norm_f32",
        _ => candle_core::bail!("fused_add_rms_norm: unsupported dtype {dtype:?}"),
    };

    let ptx = get_ptx()?;
    let cuda_dev = input.device().as_cuda_device()?;
    let func = cuda_dev.get_or_load_custom_func(func_name, MODULE_NAME, ptx)?;

    let block_size = hidden_size.min(1024) as u32;
    let grid_size = num_tokens as u32;
    let hidden_i32 = hidden_size as i32;
    let elem_count = num_tokens * hidden_size;

    let (input_s, input_l) = input.storage_and_layout();
    let (residual_s, residual_l) = residual.storage_and_layout();
    let (weight_s, weight_l) = weight.storage_and_layout();

    let (output_storage, residual_out_storage) = match dtype {
        DType::F16 => {
            let inp = input_s.as_cuda_slice::<half::f16>()?;
            let res = residual_s.as_cuda_slice::<half::f16>()?;
            let w = weight_s.as_cuda_slice::<half::f16>()?;

            let out = unsafe { cuda_dev.alloc::<half::f16>(elem_count)? };
            let res_out = unsafe { cuda_dev.alloc::<half::f16>(elem_count)? };

            let inp = inp.slice(input_l.start_offset()..);
            let res = res.slice(residual_l.start_offset()..);
            let w = w.slice(weight_l.start_offset()..);

            let stream = cuda_dev.cuda_stream();
            let mut builder = stream.launch_builder(&func);
            builder.arg(&inp);
            builder.arg(&res);
            builder.arg(&w);
            builder.arg(&out);
            builder.arg(&res_out);
            builder.arg(&hidden_i32);
            builder.arg(&eps);

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("kernel launch: {e}")))?;

            (
                CudaStorage::wrap_cuda_slice(out, cuda_dev.clone()),
                CudaStorage::wrap_cuda_slice(res_out, cuda_dev.clone()),
            )
        }
        DType::F32 => {
            let inp = input_s.as_cuda_slice::<f32>()?;
            let res = residual_s.as_cuda_slice::<f32>()?;
            let w = weight_s.as_cuda_slice::<f32>()?;

            let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };
            let res_out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

            let inp = inp.slice(input_l.start_offset()..);
            let res = res.slice(residual_l.start_offset()..);
            let w = w.slice(weight_l.start_offset()..);

            let stream = cuda_dev.cuda_stream();
            let mut builder = stream.launch_builder(&func);
            builder.arg(&inp);
            builder.arg(&res);
            builder.arg(&w);
            builder.arg(&out);
            builder.arg(&res_out);
            builder.arg(&hidden_i32);
            builder.arg(&eps);

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("kernel launch: {e}")))?;

            (
                CudaStorage::wrap_cuda_slice(out, cuda_dev.clone()),
                CudaStorage::wrap_cuda_slice(res_out, cuda_dev.clone()),
            )
        }
        _ => unreachable!(),
    };

    // Drop storage guards before creating output tensors
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
