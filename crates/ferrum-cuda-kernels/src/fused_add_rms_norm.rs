//! Fused residual-add + RMS normalization.
//!
//! Replaces 3 separate kernel launches with 1:
//!   residual = input + residual   (in-place)
//!   output   = rms_norm(residual, weight, eps)
//!
//! Memory reads: input + residual + weight (once each)
//! Memory writes: residual + output
//! vs. separate ops that read/write intermediate results multiple times.

use crate::kernel_loader::KernelStore;
use candle_core::{BackpropOp, CudaStorage, DType, Storage, Tensor};
use tracing::debug;

const CUDA_SRC: &str = include_str!("../kernels/fused_add_rms_norm.cu");
const MODULE_NAME: &str = "fused_add_rms_norm";
const FUNC_NAMES: &[&str] = &["fused_add_rms_norm_f16", "fused_add_rms_norm_f32"];

/// Handle for the fused add + RMS norm kernel.
pub struct FusedAddRmsNorm<'a> {
    store: &'a KernelStore,
}

impl<'a> FusedAddRmsNorm<'a> {
    pub fn new(store: &'a KernelStore) -> candle_core::Result<Self> {
        store.ensure_loaded(MODULE_NAME, CUDA_SRC, FUNC_NAMES)?;
        Ok(Self { store })
    }

    /// Fused: residual += input; output = rms_norm(residual, weight, eps)
    ///
    /// - `input`:    [num_tokens, hidden_size] on CUDA
    /// - `residual`: [num_tokens, hidden_size] on CUDA — **modified in-place**
    /// - `weight`:   [hidden_size] on CUDA
    /// - `eps`:      normalization epsilon
    ///
    /// Returns: normalized output [num_tokens, hidden_size]
    pub fn forward(
        &self,
        input: &Tensor,
        residual: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> candle_core::Result<Tensor> {
        let dtype = input.dtype();
        let (num_tokens, hidden_size) = input.dims2()?;
        debug!("fused_add_rms_norm: tokens={num_tokens}, hidden={hidden_size}, dtype={dtype:?}");

        let func_name = match dtype {
            DType::F16 => "fused_add_rms_norm_f16",
            DType::F32 => "fused_add_rms_norm_f32",
            _ => candle_core::bail!("fused_add_rms_norm: unsupported dtype {dtype:?}"),
        };

        let cuda_dev = input.device().as_cuda_device()?;
        let func = cuda_dev.get_or_load_custom_func(func_name, MODULE_NAME, &[])?;

        let block_size = hidden_size.min(1024) as u32;
        let grid_size = num_tokens as u32;
        let hidden_i32 = hidden_size as i32;

        let (input_s, input_l) = input.storage_and_layout();
        let (residual_s, residual_l) = residual.storage_and_layout();
        let (weight_s, weight_l) = weight.storage_and_layout();

        let input_cuda = input_s.as_cuda_slice::<half::f16>()?;
        let residual_cuda = residual_s.as_cuda_slice::<half::f16>()?;
        let weight_cuda = weight_s.as_cuda_slice::<half::f16>()?;

        let output_storage = match dtype {
            DType::F16 => {
                let elem_count = num_tokens * hidden_size;
                let out = unsafe { cuda_dev.alloc::<half::f16>(elem_count)? };

                let inp = input_cuda.slice(input_l.start_offset()..);
                let res = residual_cuda.slice(residual_l.start_offset()..);
                let w = weight_cuda.slice(weight_l.start_offset()..);

                let stream = cuda_dev.cuda_stream();
                let mut builder = stream.launch_builder(&func);
                builder.arg(&inp);
                builder.arg(&res);
                builder.arg(&w);
                builder.arg(&out);
                builder.arg(&hidden_i32);
                builder.arg(&eps);

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
                let input_cuda = input_s.as_cuda_slice::<f32>()?;
                let residual_cuda = residual_s.as_cuda_slice::<f32>()?;
                let weight_cuda = weight_s.as_cuda_slice::<f32>()?;

                let elem_count = num_tokens * hidden_size;
                let out = unsafe { cuda_dev.alloc::<f32>(elem_count)? };

                let inp = input_cuda.slice(input_l.start_offset()..);
                let res = residual_cuda.slice(residual_l.start_offset()..);
                let w = weight_cuda.slice(weight_l.start_offset()..);

                let stream = cuda_dev.cuda_stream();
                let mut builder = stream.launch_builder(&func);
                builder.arg(&inp);
                builder.arg(&res);
                builder.arg(&w);
                builder.arg(&out);
                builder.arg(&hidden_i32);
                builder.arg(&eps);

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

        // Drop storage guards before creating output tensor
        drop(input_s);
        drop(residual_s);
        drop(weight_s);

        Ok(Tensor::from_storage(
            Storage::Cuda(output_storage),
            input.shape().clone(),
            BackpropOp::none(),
            false,
        ))
    }
}
