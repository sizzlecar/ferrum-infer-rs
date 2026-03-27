//! Standalone RMS normalization kernel.
//!
//! output = input / sqrt(mean(input^2) + eps) * weight
//!
//! Works for both [num_tokens, hidden_size] and [num_heads, head_dim] layouts.

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::ptx;

const MODULE_NAME: &str = "rms_norm";

/// RMS normalization on candle Tensors.
///
/// - `input`:  [num_rows, row_size] on CUDA
/// - `weight`: [row_size] on CUDA
/// - `eps`:    normalization epsilon
///
/// Returns: normalized tensor with same shape as input.
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    let dtype = input.dtype();
    let dims = input.dims();
    let row_size = *dims.last().unwrap();
    let num_rows = input.elem_count() / row_size;

    let func_name = match dtype {
        DType::F16 => "rms_norm_f16",
        DType::F32 => "rms_norm_f32",
        _ => candle_core::bail!("rms_norm: unsupported dtype {dtype:?}"),
    };

    let cuda_dev = input.device().as_cuda_device()?;
    let func = cuda_dev.get_or_load_custom_func(func_name, MODULE_NAME, ptx::RMS_NORM)?;

    let block_size = row_size.min(1024) as u32;
    let grid_size = num_rows as u32;
    let row_size_i32 = row_size as i32;
    let elem_count = num_rows * row_size;

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

    let output_storage = match dtype {
        DType::F16 => {
            let inp = input_cuda.as_cuda_slice::<half::f16>()?;
            let w = weight_cuda.as_cuda_slice::<half::f16>()?;
            let out = unsafe { cuda_dev.alloc::<half::f16>(elem_count)? };

            let inp = inp.slice(input_l.start_offset()..);
            let w = w.slice(weight_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&inp);
            builder.arg(&w);
            builder.arg(&out);
            builder.arg(&row_size_i32);
            builder.arg(&eps);

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("rms_norm kernel launch: {e}")))?;

            CudaStorage::wrap_cuda_slice(out, cuda_dev.clone())
        }
        DType::F32 => {
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
            builder.arg(&eps);

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("rms_norm kernel launch: {e}")))?;

            CudaStorage::wrap_cuda_slice(out, cuda_dev.clone())
        }
        _ => unreachable!(),
    };

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
