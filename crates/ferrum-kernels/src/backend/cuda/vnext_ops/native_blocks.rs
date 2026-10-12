//! Kernels consume original compressed GGUF bytes with bounded register scratch.
//! Provider registration and model closure are separate from these primitives.
use std::sync::Arc;

use super::super::vnext_runtime::CudaDeviceRuntimeError;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

pub(super) mod hadamard;
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

fn q8_0_pair_f16_eligible(
    parts: [&weights::MatrixPart; 2],
    rows: u32,
    output_stride: u32,
    multiprocessors: u32,
) -> Result<bool, CudaDeviceRuntimeError> {
    let q8 = weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q8_0);
    if parts
        .iter()
        .any(|part| part.format != q8 || part.transform.is_some() || part.signs_region.is_some())
        || parts[0].columns != parts[1].columns
    {
        return Ok(false);
    }
    if rows == 0
        || parts[0].columns == 0
        || parts[0].columns % 32 != 0
        || parts.iter().any(|part| part.rows == 0)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA native linear pair launch extent is invalid",
        ));
    }
    // Larger row batches use the existing chunked single-leaf path. Keep the
    // caller's signed shape domain too; unsupported extents never launch here.
    if rows > u16::MAX as u32
        || parts[0].columns > i32::MAX as u32
        || output_stride > i32::MAX as u32
    {
        return Ok(false);
    }
    // Select each original leaf independently. Combining their output widths
    // must not change a former T1 leaf into a different arithmetic schedule.
    if parts
        .iter()
        .any(|part| q8_0_f16_row_tile(rows, part.columns, part.rows, multiprocessors) != 1)
    {
        return Ok(false);
    }
    let [Some(end0), Some(end1)] = parts.map(|part| part.output_offset.checked_add(part.rows))
    else {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA native linear pair output extent overflows",
        ));
    };
    if end0 > output_stride
        || end1 > output_stride
        || !(end0 <= parts[1].output_offset || end1 <= parts[0].output_offset)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA native linear pair output windows are invalid",
        ));
    }
    Ok(true)
}

#[derive(Clone)]
pub(super) struct CudaNativeBlockKernels {
    multiprocessors: u32,
    pub linear_f16: CudaFunction,
    pub linear_f32: CudaFunction,
    linear_tiled_f16: CudaFunction,
    linear_tiled_f32: CudaFunction,
    linear_f32_f16: CudaFunction,
    linear_tiled_f32_f16: CudaFunction,
    linear_q4k_f16: CudaFunction,
    linear_q4k_tiled_f16: CudaFunction,
    linear_q8_0_f16: CudaFunction,
    linear_q8_0_pair_f16: CudaFunction,
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
    pub(super) fn try_linear_q8_0_pair_f16(
        &self,
        stream: &CudaStream,
        input: u64,
        weights: [u64; 2],
        output: u64,
        parts: [&weights::MatrixPart; 2],
        rows: u32,
        output_stride: u32,
    ) -> Result<bool, CudaDeviceRuntimeError> {
        if !q8_0_pair_f16_eligible(parts, rows, output_stride, self.multiprocessors)? {
            return Ok(false);
        }
        let parameters = [
            rows,
            parts[0].columns,
            parts[0].rows,
            parts[1].rows,
            output_stride,
            parts[0].output_offset,
            parts[1].output_offset,
        ];
        let mut launch = stream.launch_builder(&self.linear_q8_0_pair_f16);
        launch
            .arg(&input)
            .arg(&weights[0])
            .arg(&weights[1])
            .arg(&output);
        for parameter in &parameters {
            launch.arg(parameter);
        }
        // SAFETY: The caller retains the independently validated original
        // weight/input/output regions. Checked disjoint windows remain in the
        // existing stride; each z plane uses exactly one original T1 leaf.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (parts[0].rows.max(parts[1].rows).div_ceil(4), rows, 2),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| true)
        .map_err(|error| CudaDeviceRuntimeError::driver("native Q8_0 pair linear launch", error))
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
            linear_f16: load("vnext_gguf_linear_f16")?,
            linear_f32: load("vnext_gguf_linear_f32")?,
            linear_tiled_f16: load("vnext_gguf_linear_tiled_f16")?,
            linear_tiled_f32: load("vnext_gguf_linear_tiled_f32")?,
            linear_f32_f16: load("vnext_gguf_linear_f32_f16")?,
            linear_tiled_f32_f16: load("vnext_gguf_linear_tiled_f32_f16")?,
            linear_q4k_f16: load("vnext_gguf_linear_q4k_f16")?,
            linear_q4k_tiled_f16: load("vnext_gguf_linear_q4k_tiled_f16")?,
            linear_q8_0_f16: load("vnext_gguf_linear_q8_0_f16")?,
            linear_q8_0_pair_f16: load("vnext_gguf_linear_q8_0_pair_f16")?,
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
        let q4k =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q4K);
        let q8_0 =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q8_0);
        let q5k =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q5K);
        let iq4xs =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Iq4Xs);
        let q6k =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q6K);
        let row_tile = if q8_0 && input_type == ElementType::F16 && output_type == ElementType::F16
        {
            q8_0_f16_row_tile(rows, part.columns, part.rows, self.multiprocessors)
        } else if rows > 1 {
            LINEAR_ROW_TILE
        } else {
            1
        };
        let kernel = match (input_type, output_type, row_tile > 1) {
            (ElementType::F16, ElementType::F16, false) if q4k => &self.linear_q4k_f16,
            (ElementType::F16, ElementType::F16, true) if q4k => &self.linear_q4k_tiled_f16,
            (ElementType::F16, ElementType::F16, false) if q8_0 => &self.linear_q8_0_f16,
            (ElementType::F16, ElementType::F16, false) if q5k => &self.linear_q5k_f16,
            (ElementType::F16, ElementType::F16, true) if q5k => &self.linear_q5k_tiled_f16,
            (ElementType::F16, ElementType::F16, false) if iq4xs => &self.linear_iq4xs_f16,
            (ElementType::F16, ElementType::F16, true) if iq4xs => &self.linear_iq4xs_tiled_f16,
            (ElementType::F16, ElementType::F16, false) if q6k => &self.linear_q6k_f16,
            (ElementType::F16, ElementType::F16, true) if q6k => &self.linear_q6k_tiled_f16,
            (ElementType::F32, ElementType::F32, false) if q6k => &self.linear_q6k_f32,
            (ElementType::F32, ElementType::F32, true) if q6k => &self.linear_q6k_tiled_f32,
            (ElementType::F16, ElementType::F16, false) => &self.linear_f16,
            (ElementType::F32, ElementType::F32, false) => &self.linear_f32,
            (ElementType::F16, ElementType::F16, true) => &self.linear_tiled_f16,
            (ElementType::F32, ElementType::F32, true) => &self.linear_tiled_f32,
            (ElementType::F32, ElementType::F16, false) => &self.linear_f32_f16,
            (ElementType::F32, ElementType::F16, true) => &self.linear_tiled_f32_f16,
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
