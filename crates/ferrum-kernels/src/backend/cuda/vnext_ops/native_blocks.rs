//! Kernels consume original compressed GGUF bytes with bounded register scratch.
//! Provider registration and model closure are separate from these primitives.
use std::sync::Arc;

use super::super::vnext_runtime::CudaDeviceRuntimeError;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

pub(super) mod weights;

// Must match the bounded row tile instantiated by vnext_gguf_linear_tiled_*.
const LINEAR_ROW_TILE: u32 = 8;

#[derive(Clone)]
pub(super) struct CudaNativeBlockKernels {
    pub linear_f16: CudaFunction,
    pub linear_f32: CudaFunction,
    linear_tiled_f16: CudaFunction,
    linear_tiled_f32: CudaFunction,
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
            linear_tiled_f16: load("vnext_gguf_linear_tiled_f16")?,
            linear_tiled_f32: load("vnext_gguf_linear_tiled_f32")?,
            embedding_f16: load("vnext_gguf_embedding_f16")?,
            embedding_f32: load("vnext_gguf_embedding_f32")?,
            #[cfg(test)]
            decode: load("vnext_gguf_decode")?,
        })
    }

    pub(super) fn embedding(
        &self,
        stream: &CudaStream,
        tokens: u64,
        weight: u64,
        output: u64,
        part: &weights::MatrixPart,
        count: u32,
        activation: ferrum_interfaces::vnext::ElementType,
    ) -> Result<(), CudaDeviceRuntimeError> {
        use ferrum_interfaces::vnext::ElementType;
        let elements = embedding_elements(part, count)?;
        let function = match activation {
            ElementType::F16 => &self.embedding_f16,
            ElementType::F32 => &self.embedding_f32,
            _ => {
                return Err(CudaDeviceRuntimeError::contract(
                    "unsupported embedding dtype",
                ))
            }
        };
        let [format, values, bytes] = part.format.parameters();
        let parameters = [count, part.columns, part.rows, format, values, bytes];
        let mut launch = stream.launch_builder(function);
        launch.arg(&tokens).arg(&weight).arg(&output);
        for parameter in &parameters {
            launch.arg(parameter);
        }
        // SAFETY: The provider retains the complete table and exact token and
        // output spans. The kernel guards the element count and invalid IDs.
        unsafe { launch.launch(LaunchConfig::for_num_elems(elements)) }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("native embedding launch", error))
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
        let row_tile = if rows > 1 { LINEAR_ROW_TILE } else { 1 };
        let kernel = match (activation, row_tile > 1) {
            (ElementType::F16, false) => &self.linear_f16,
            (ElementType::F32, false) => &self.linear_f32,
            (ElementType::F16, true) => &self.linear_tiled_f16,
            (ElementType::F32, true) => &self.linear_tiled_f32,
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
        // regions. One complete warp writes each guarded output column for
        // every row in its tile; the final partial tile guards all row accesses.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (part.rows.div_ceil(4), rows.div_ceil(row_tile), 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("native matrix linear launch", error))
    }
}

pub(super) fn embedding_elements(
    part: &weights::MatrixPart,
    count: u32,
) -> Result<u32, CudaDeviceRuntimeError> {
    let [_, values, _] = part.format.parameters();
    if part.output_offset != 0 || part.rows == 0 || part.columns % values != 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "native embedding requires a complete row-aligned vocabulary table",
        ));
    }
    count
        .checked_mul(part.columns)
        .filter(|&n| n != 0)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("native embedding launch extent overflows"))
}

#[cfg(test)]
mod tests;
