//! Fused SiLU activation + elementwise multiply.
//!
//! Replaces 2 kernel launches with 1:
//!   output = silu(gate) * up
//!
//! Used in gated MLP: after fused gate+up matmul, split and apply silu+mul.

use candle_core::cuda_backend::CudaStorage;
use candle_core::{op::BackpropOp, DType, Storage, Tensor};
use cudarc::driver::PushKernelArg;
use std::sync::OnceLock;

const CUDA_SRC: &str = include_str!("../kernels/fused_silu_mul.cu");
const MODULE_NAME: &str = "fused_silu_mul";

static COMPILED_PTX: OnceLock<String> = OnceLock::new();

fn get_ptx() -> &'static str {
    COMPILED_PTX.get_or_init(|| {
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        cudarc::nvrtc::safe::compile_ptx_with_opts(CUDA_SRC, opts)
            .expect("Failed to compile fused_silu_mul CUDA kernel")
            .to_src()
    })
}

/// Fused: output = silu(gate) * up
///
/// Both `gate` and `up` must have the same shape and be contiguous on CUDA.
///
/// Returns: result tensor with same shape
pub fn fused_silu_mul(gate: &Tensor, up: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = gate.dtype();
    let shape = gate.shape().clone();
    let elem_count = shape.elem_count();

    let func_name = match dtype {
        DType::F16 => "fused_silu_mul_f16",
        DType::F32 => "fused_silu_mul_f32",
        _ => candle_core::bail!("fused_silu_mul: unsupported dtype {dtype:?}"),
    };

    let cuda_dev = gate.device().as_cuda_device()?;
    let func = cuda_dev.get_or_load_custom_func(func_name, MODULE_NAME, get_ptx())?;

    let block_size = 256u32;
    let grid_size = (elem_count as u32 + block_size - 1) / block_size;
    let n = elem_count as i32;

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

    let output_storage = match dtype {
        DType::F16 => {
            let g = gate_cuda.as_cuda_slice::<half::f16>()?;
            let u = up_cuda.as_cuda_slice::<half::f16>()?;
            let out = unsafe { cuda_dev.alloc::<half::f16>(elem_count)? };

            let g = g.slice(gate_l.start_offset()..);
            let u = u.slice(up_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&g);
            builder.arg(&u);
            builder.arg(&out);
            builder.arg(&n);

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("kernel launch: {e}")))?;

            CudaStorage::wrap_cuda_slice(out, cuda_dev.clone())
        }
        DType::F32 => {
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

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("kernel launch: {e}")))?;

            CudaStorage::wrap_cuda_slice(out, cuda_dev.clone())
        }
        _ => unreachable!(),
    };

    drop(gate_s);
    drop(up_s);

    Ok(Tensor::from_storage(
        Storage::Cuda(output_storage),
        shape,
        BackpropOp::none(),
        false,
    ))
}
