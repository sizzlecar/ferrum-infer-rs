//! Kernels consume original compressed GGUF bytes with bounded register scratch.
//! Provider registration and model closure are separate from these primitives.
use std::sync::Arc;

use super::super::vnext_runtime::{CudaBufferRegion, CudaDeviceRuntimeError};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

pub(super) mod hadamard;
mod linear;
pub(super) use linear::TransformedLinearSelection;
mod q8_predecode;
pub(super) mod q8act;
pub(super) mod upstream_linear;
pub(super) mod weights;

// Must match the bounded row tile instantiated by vnext_gguf_linear_tiled_*.
const LINEAR_ROW_TILE: u32 = 8;

// Small Q8_0 grids benefit from independent rows. Keep the measured lower-K
// boundary and a conservative four-block-per-SM limit; wider grids retain
// generic T8 reuse. u64 covers the complete u32 launch-shape product.
fn q8_0_f16_row_tile(rows: u32, inputs: u32, outputs: u32, multiprocessors: u32) -> u32 {
    let tiled_grid =
        u64::from(outputs).div_ceil(4) * u64::from(rows).div_ceil(u64::from(LINEAR_ROW_TILE));
    if rows == 1 || (inputs >= 1024 && tiled_grid < 4 * u64::from(multiprocessors)) {
        1
    } else {
        LINEAR_ROW_TILE
    }
}

#[derive(Clone)]
pub(super) struct CudaNativeBlockKernels {
    multiprocessors: u32,
    q8_predecode_context: Arc<CudaContext>,
    pub linear_f16: CudaFunction,
    pub linear_f32: CudaFunction,
    linear_tiled_f16: CudaFunction,
    linear_tiled_f32: CudaFunction,
    linear_f32_f16: CudaFunction,
    linear_tiled_f32_f16: CudaFunction,
    linear_q4k_f16: CudaFunction,
    linear_q4k_tiled_f16: CudaFunction,
    linear_q8_0_f16: CudaFunction,
    q8_0_predecode_f32: CudaFunction,
    linear_q8_0_predecoded_f32_f16: CudaFunction,
    #[cfg(test)]
    linear_q8_0_tiled_f16: CudaFunction,
    linear_q5k_f16: CudaFunction,
    linear_q5k_tiled_f16: CudaFunction,
    linear_iq4xs_f16: CudaFunction,
    linear_iq4xs_tiled_f16: CudaFunction,
    #[cfg(test)]
    linear_iq4xs_constant_f16: CudaFunction,
    #[cfg(test)]
    linear_iq4xs_constant_tiled_f16: CudaFunction,
    #[cfg(test)]
    iq4xs_register_decode: CudaFunction,
    linear_q6k_f16: CudaFunction,
    linear_q6k_tiled_f16: CudaFunction,
    #[cfg(test)]
    linear_q6k_test_t4_f16: CudaFunction,
    linear_q6k_f32: CudaFunction,
    linear_q6k_tiled_f32: CudaFunction,
    hadamard: hadamard::CudaHadamardKernels,
    pub embedding_f16: CudaFunction,
    pub embedding_f32: CudaFunction,
    #[cfg(test)]
    decode: CudaFunction,
}

impl CudaNativeBlockKernels {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn transformed_embedding(
        &self,
        stream: &CudaStream,
        tokens: u64,
        weight: u64,
        output: u64,
        part: &weights::MatrixPart,
        count: u32,
        activation: ferrum_runtime::ElementType,
        signs: u64,
        scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        use ferrum_runtime::{ElementType, HadamardApplication};
        if let Some(spec) = &part.transform {
            if !matches!(spec.application, HadamardApplication::AfterEmbeddingLookup)
                || scratch == 0
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "native embedding requires an inverse transform and planned F32 scratch",
                ));
            }
            self.embedding(
                stream,
                tokens,
                weight,
                scratch,
                part,
                count,
                ElementType::F32,
            )?;
            self.hadamard.launch(
                stream,
                scratch,
                output,
                signs,
                count,
                part.columns,
                ElementType::F32,
                activation,
                spec,
            )
        } else {
            self.embedding(stream, tokens, weight, output, part, count, activation)
        }
    }
    pub(super) fn load(context: &Arc<CudaContext>) -> Result<Self, CudaDeviceRuntimeError> {
        use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
        let multiprocessors = context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|error| CudaDeviceRuntimeError::driver("native GGUF SM count", error))?;
        let multiprocessors = u32::try_from(multiprocessors)
            .ok()
            .filter(|&count| count > 0)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("native GGUF SM count is invalid"))?;
        let module = context
            .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("native GGUF module load", error))?;
        let load = |name| {
            module
                .load_function(name)
                .map_err(|error| CudaDeviceRuntimeError::driver("native GGUF function load", error))
        };
        Ok(Self {
            multiprocessors,
            q8_predecode_context: context.clone(),
            linear_f16: load("vnext_gguf_linear_f16")?,
            linear_f32: load("vnext_gguf_linear_f32")?,
            linear_tiled_f16: load("vnext_gguf_linear_tiled_f16")?,
            linear_tiled_f32: load("vnext_gguf_linear_tiled_f32")?,
            linear_f32_f16: load("vnext_gguf_linear_f32_f16")?,
            linear_tiled_f32_f16: load("vnext_gguf_linear_tiled_f32_f16")?,
            linear_q4k_f16: load("vnext_gguf_linear_q4k_f16")?,
            linear_q4k_tiled_f16: load("vnext_gguf_linear_q4k_tiled_f16")?,
            linear_q8_0_f16: load("vnext_gguf_linear_q8_0_f16")?,
            q8_0_predecode_f32: load("vnext_gguf_q8_0_predecode_f32")?,
            linear_q8_0_predecoded_f32_f16: load("vnext_gguf_linear_q8_0_predecoded_f32_f16")?,
            #[cfg(test)]
            linear_q8_0_tiled_f16: load("vnext_gguf_linear_q8_0_tiled_f16")?,
            linear_q5k_f16: load("vnext_gguf_linear_q5k_f16")?,
            linear_q5k_tiled_f16: load("vnext_gguf_linear_q5k_tiled_f16")?,
            linear_iq4xs_f16: load("vnext_gguf_linear_iq4xs_f16")?,
            linear_iq4xs_tiled_f16: load("vnext_gguf_linear_iq4xs_tiled_f16")?,
            #[cfg(test)]
            linear_iq4xs_constant_f16: load("vnext_gguf_linear_iq4xs_constant_f16")?,
            #[cfg(test)]
            linear_iq4xs_constant_tiled_f16: load("vnext_gguf_linear_iq4xs_constant_tiled_f16")?,
            #[cfg(test)]
            iq4xs_register_decode: load("vnext_gguf_iq4xs_register_decode")?,
            linear_q6k_f16: load("vnext_gguf_linear_q6k_f16")?,
            linear_q6k_tiled_f16: load("vnext_gguf_linear_q6k_tiled_f16")?,
            #[cfg(test)]
            linear_q6k_test_t4_f16: load("vnext_gguf_linear_q6k_test_t4_f16")?,
            linear_q6k_f32: load("vnext_gguf_linear_q6k_f32")?,
            linear_q6k_tiled_f32: load("vnext_gguf_linear_q6k_tiled_f32")?,
            hadamard: hadamard::CudaHadamardKernels::load(&module)?,
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
        activation: ferrum_runtime::ElementType,
    ) -> Result<(), CudaDeviceRuntimeError> {
        use ferrum_runtime::ElementType;
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
        activation: ferrum_runtime::ElementType,
    ) -> Result<(), CudaDeviceRuntimeError> {
        self.linear_with_precision(
            stream,
            input,
            weight,
            output,
            part,
            rows,
            output_stride,
            activation,
            activation,
        )
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
