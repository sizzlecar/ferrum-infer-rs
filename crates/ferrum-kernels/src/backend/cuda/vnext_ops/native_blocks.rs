//! Kernels consume original compressed GGUF bytes with bounded register scratch.
//! Provider registration and model closure are separate from these primitives.
use std::sync::Arc;

use super::super::vnext_runtime::CudaDeviceRuntimeError;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

pub(super) mod weights;

#[derive(Clone)]
pub(super) struct CudaNativeBlockKernels {
    pub linear_f16: CudaFunction,
    pub linear_f32: CudaFunction,
    pub embedding_f16: CudaFunction,
    pub embedding_f32: CudaFunction,
    #[cfg(test)]
    decode: CudaFunction,
}

impl CudaNativeBlockKernels {
    pub(super) fn load(context: &Arc<CudaContext>) -> Result<Self, CudaDeviceRuntimeError> {
        let module = context
            .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("native GGUF module load", error))?;
        let load = |name| {
            module
                .load_function(name)
                .map_err(|error| CudaDeviceRuntimeError::driver("native GGUF function load", error))
        };
        Ok(Self {
            linear_f16: load("vnext_gguf_linear_f16")?,
            linear_f32: load("vnext_gguf_linear_f32")?,
            embedding_f16: load("vnext_gguf_embedding_f16")?,
            embedding_f32: load("vnext_gguf_embedding_f32")?,
            #[cfg(test)]
            decode: load("vnext_gguf_decode")?,
        })
    }

    pub(super) fn linear(
        &self,
        stream: &CudaStream,
        input: u64,
        weight: u64,
        output: u64,
        part: &weights::MatrixPart,
        rows: u32,
        output_stride: u32,
        activation: ferrum_interfaces::vnext::ElementType,
    ) -> Result<(), CudaDeviceRuntimeError> {
        use ferrum_interfaces::vnext::ElementType;
        if rows == 0
            || rows > u16::MAX as u32
            || part
                .output_offset
                .checked_add(part.rows)
                .is_none_or(|end| end > output_stride)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA native linear launch extent is invalid",
            ));
        }
        let kernel = match activation {
            ElementType::F16 => &self.linear_f16,
            ElementType::F32 => &self.linear_f32,
            _ => {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA native linear activation dtype is unsupported",
                ))
            }
        };
        let [format, values, bytes] = part.format.parameters();
        let parameters = [
            rows,
            part.columns,
            part.rows,
            output_stride,
            part.output_offset,
            format,
            values,
            bytes,
        ];
        let mut launch = stream.launch_builder(kernel);
        launch.arg(&input).arg(&weight).arg(&output);
        for parameter in &parameters {
            launch.arg(parameter);
        }
        // SAFETY: The provider retains exact row-complete matrix and activation
        // regions. One complete warp writes each guarded output column.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (part.rows.div_ceil(4), rows, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("native matrix linear launch", error))
    }
}

#[cfg(test)]
mod tests;
