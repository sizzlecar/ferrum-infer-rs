//! Element-wise residual add: output = a + b

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;

use crate::ptx;

const MODULE_NAME: &str = "residual_add";

/// Element-wise add of two tensors with same shape.
///
/// Returns: result tensor with same shape.
pub fn residual_add(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = a.dtype();
    let shape = a.shape().clone();
    let elem_count = shape.elem_count();

    let func_name = match dtype {
        DType::F16 => "residual_add_f16",
        DType::F32 => "residual_add_f32",
        _ => candle_core::bail!("residual_add: unsupported dtype {dtype:?}"),
    };

    let cuda_dev = a.device().as_cuda_device()?;
    let func = cuda_dev.get_or_load_custom_func(func_name, MODULE_NAME, ptx::RESIDUAL_ADD)?;

    let block_size = 256u32;
    let grid_size = (elem_count as u32 + block_size - 1) / block_size;
    let n = elem_count as i32;

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

    let output_storage = match dtype {
        DType::F16 => {
            let a_in = a_cuda.as_cuda_slice::<half::f16>()?;
            let b_in = b_cuda.as_cuda_slice::<half::f16>()?;
            let out = unsafe { cuda_dev.alloc::<half::f16>(elem_count)? };

            let a_in = a_in.slice(a_l.start_offset()..);
            let b_in = b_in.slice(b_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&a_in);
            builder.arg(&b_in);
            builder.arg(&out);
            builder.arg(&n);

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("residual_add launch: {e}")))?;

            CudaStorage::wrap_cuda_slice(out, cuda_dev.clone())
        }
        DType::F32 => {
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

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("residual_add launch: {e}")))?;

            CudaStorage::wrap_cuda_slice(out, cuda_dev.clone())
        }
        _ => unreachable!(),
    };

    drop(a_s);
    drop(b_s);

    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape,
        BackpropOp::none(),
        false,
    ))
}
