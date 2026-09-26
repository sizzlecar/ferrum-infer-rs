//! Kernels consume original compressed GGUF bytes with bounded register scratch.
//! Provider registration and model closure are separate from these primitives.
use std::sync::Arc;

use super::super::vnext_runtime::CudaDeviceRuntimeError;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

pub(super) mod embedding;
pub(super) mod hadamard;
mod linear_launch;
pub(super) mod q8_f32scale;
pub(super) mod q8_pair;
pub(super) mod selected;
pub(super) mod stream_mmq;
pub(super) mod weights;

// Must match the bounded row tile instantiated by vnext_gguf_linear_tiled_*.
const LINEAR_ROW_TILE: u32 = 8;

#[derive(Clone)]
pub(super) struct CudaNativeBlockKernels {
    pub linear_f16: CudaFunction,
    pub linear_f32: CudaFunction,
    linear_tiled_f16: CudaFunction,
    linear_q8_pair_tiled_f16: CudaFunction,
    #[cfg(test)]
    q8_pair_disabled: bool,
    linear_tiled_f32: CudaFunction,
    linear_f32_f16: CudaFunction,
    linear_tiled_f32_f16: CudaFunction,
    linear_q4k_f16: CudaFunction,
    linear_q4k_tiled_f16: CudaFunction,
    linear_q5k_f16: CudaFunction,
    linear_q5k_tiled_f16: CudaFunction,
    linear_q6k_f16: CudaFunction,
    linear_q6k_tiled_f16: CudaFunction,
    linear_q6k_f32: CudaFunction,
    linear_q6k_tiled_f32: CudaFunction,
    linear_q6k_f32_f16: CudaFunction,
    linear_q6k_tiled_f32_f16: CudaFunction,
    gemm_q4k_f16: CudaFunction,
    gemm_q5k_f16: CudaFunction,
    gemm_q6k_f16: CudaFunction,
    hadamard: hadamard::CudaHadamardKernels,
    pub embedding_f16: CudaFunction,
    pub embedding_f32: CudaFunction,
    #[cfg(test)]
    decode: CudaFunction,
}

impl CudaNativeBlockKernels {
    /// Retain the exact old CUDA exports as the independent test control while
    /// exercising all production callers, strides and state transitions.
    #[cfg(test)]
    pub(super) fn with_generic_q5k_control(&self) -> Self {
        let mut control = self.clone();
        control.linear_q5k_f16 = self.linear_f16.clone();
        control.linear_q5k_tiled_f16 = self.linear_tiled_f16.clone();
        control
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn transformed_linear(
        &self,
        stream: &CudaStream,
        input: u64,
        weight: u64,
        output: u64,
        part: &weights::MatrixPart,
        rows: u32,
        output_stride: u32,
        activation: ferrum_interfaces::vnext::ElementType,
        signs: u64,
        scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        use ferrum_interfaces::vnext::{ElementType, HadamardApplication};
        let (input, input_type) = if let Some(spec) = &part.transform {
            if !matches!(spec.application, HadamardApplication::BeforeMatmul { .. }) || scratch == 0
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "native linear requires a forward transform and planned F32 scratch",
                ));
            }
            self.hadamard.launch(
                stream,
                input,
                scratch,
                signs,
                rows,
                part.columns,
                activation,
                ElementType::F32,
                spec,
            )?;
            (scratch, ElementType::F32)
        } else {
            (input, activation)
        };
        self.linear_with_precision(
            stream,
            input,
            weight,
            output,
            part,
            rows,
            output_stride,
            input_type,
            activation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn transformed_embedding(
        &self,
        stream: &CudaStream,
        tokens: u64,
        weight: u64,
        output: u64,
        part: &weights::MatrixPart,
        count: u32,
        activation: ferrum_interfaces::vnext::ElementType,
        signs: u64,
        scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        use ferrum_interfaces::vnext::{ElementType, HadamardApplication};
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
            linear_q8_pair_tiled_f16: load("vnext_gguf_linear_q8_pair_tiled_f16")?,
            #[cfg(test)]
            q8_pair_disabled: false,
            linear_tiled_f32: load("vnext_gguf_linear_tiled_f32")?,
            linear_f32_f16: load("vnext_gguf_linear_f32_f16")?,
            linear_tiled_f32_f16: load("vnext_gguf_linear_tiled_f32_f16")?,
            linear_q4k_f16: load("vnext_gguf_linear_q4k_f16")?,
            linear_q4k_tiled_f16: load("vnext_gguf_linear_q4k_tiled_f16")?,
            linear_q5k_f16: load("vnext_gguf_linear_q5k_f16")?,
            linear_q5k_tiled_f16: load("vnext_gguf_linear_q5k_tiled_f16")?,
            linear_q6k_f16: load("vnext_gguf_linear_q6k_f16")?,
            linear_q6k_tiled_f16: load("vnext_gguf_linear_q6k_tiled_f16")?,
            linear_q6k_f32: load("vnext_gguf_linear_q6k_f32")?,
            linear_q6k_tiled_f32: load("vnext_gguf_linear_q6k_tiled_f32")?,
            linear_q6k_f32_f16: load("vnext_gguf_linear_q6k_f32_f16")?,
            linear_q6k_tiled_f32_f16: load("vnext_gguf_linear_q6k_tiled_f32_f16")?,
            gemm_q4k_f16: load("vnext_gguf_gemm_q4k_f16")?,
            gemm_q5k_f16: load("vnext_gguf_gemm_q5k_f16")?,
            gemm_q6k_f16: load("vnext_gguf_gemm_q6k_f16")?,
            hadamard: hadamard::CudaHadamardKernels::load(&module)?,
            embedding_f16: load(embedding::F16_ENTRY)?,
            embedding_f32: load(embedding::F32_ENTRY)?,
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
        let selected = embedding::lookup_plan(part, count, activation)?;
        let function = match selected.entry {
            embedding::F16_ENTRY => &self.embedding_f16,
            embedding::F32_ENTRY => &self.embedding_f32,
            _ => unreachable!("checked installed embedding entry"),
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&tokens).arg(&weight).arg(&output);
        for parameter in &selected.parameters {
            launch.arg(parameter);
        }
        // SAFETY: The provider retains the complete table and exact token and
        // output spans. The kernel guards the element count and invalid IDs.
        unsafe { launch.launch(selected.config) }
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn linear_with_precision(
        &self,
        stream: &CudaStream,
        input: u64,
        weight: u64,
        output: u64,
        part: &weights::MatrixPart,
        rows: u32,
        output_stride: u32,
        input_type: ferrum_interfaces::vnext::ElementType,
        output_type: ferrum_interfaces::vnext::ElementType,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let selected = linear_launch::select(part, rows, output_stride, input_type, output_type)?;
        let kernel = selected.kernel.function(self);
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
        // regions. Both launch geometries guard partial row/column tiles. The
        // shared kernel's fixed 16x16 block cooperatively initializes all of
        // its bounded shared storage before any thread consumes it.
        unsafe { launch.launch(selected.config) }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("native matrix linear launch", error))
    }
}

fn use_shared_gemm(
    part: &weights::MatrixPart,
    rows: u32,
    input_type: ferrum_interfaces::vnext::ElementType,
    output_type: ferrum_interfaces::vnext::ElementType,
) -> bool {
    use crate::gguf_blocks::GgufBlockFormat;
    use ferrum_interfaces::vnext::ElementType;
    if input_type != ElementType::F16 || output_type != ElementType::F16 {
        return false;
    }
    // Preserve the previously qualified large-row path, including its small-K
    // and partitioned-output cases. The format match at launch remains final.
    if rows >= 128 && part.rows >= 128 {
        return true;
    }
    if rows < 32 || part.transform.is_some() {
        return false;
    }
    match part.format {
        weights::MatrixFormat::Block(GgufBlockFormat::Q5K) => {
            (part.columns >= 4096 && part.rows >= 2560)
                || (part.columns >= 2560 && part.rows >= 8192)
        }
        weights::MatrixFormat::Block(GgufBlockFormat::Q6K) => {
            part.columns >= 4096 && part.rows >= 4096
        }
        _ => false,
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
