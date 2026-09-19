use std::ffi::c_void;
use std::fmt::Write;

use ferrum_interfaces::vnext::ElementType;

use metal::{
    CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, FunctionConstantValues,
    MTLDataType,
};

use crate::backend::metal::vnext_runtime::MetalDeviceRuntimeError;
use crate::gguf_blocks::{GgufBlockFormat, IQ3_S_GRID, IQ4_NL_VALUES};

pub(super) const FINGERPRINT_SOURCE: &str = concat!(
    include_str!("native_blocks.rs"),
    include_str!("native_blocks.metal"),
    include_str!("group_dot.metal"),
    include_str!("../../../gguf_blocks/iq3s_grid.rs"),
    include_str!("../../../gguf_blocks/iq4nl_values.rs"),
    include_str!("../../../gguf_blocks/mod.rs"),
);

const M64_THREADGROUP_BYTES: u64 = 16384;
const M64_THREADS: u64 = 256;

pub(super) fn pq2_full_tiles_supported(
    row_tile: u32,
    rows: u32,
    in_features: u32,
    out_features: u32,
) -> bool {
    matches!(row_tile, 32 | 64)
        && rows > 0
        && rows.is_multiple_of(row_tile)
        && in_features > 0
        && in_features.is_multiple_of(128)
        && out_features > 0
        && out_features.is_multiple_of(64)
}

pub(super) fn pq2_full_tiles_vector_input_supported(
    rows: u32,
    in_features: u32,
    out_features: u32,
    input_region_offset_bytes: u64,
    input_offset_bytes: u64,
    input_region_bytes: u64,
) -> bool {
    if !pq2_full_tiles_supported(64, rows, in_features, out_features) {
        return false;
    }
    let Some(start) = input_region_offset_bytes.checked_add(input_offset_bytes) else {
        return false;
    };
    let Some(bytes) = u64::from(rows)
        .checked_mul(u64::from(in_features))
        .and_then(|elements| elements.checked_mul(4))
    else {
        return false;
    };
    // Metal buffer bases are aligned; eligibility depends on the complete
    // bound span's start, including a region's base and any workspace offset.
    // K128 divisibility also keeps every row and K32 iteration float4-aligned.
    start.is_multiple_of(16)
        && start.checked_add(bytes).is_some()
        && input_offset_bytes
            .checked_add(bytes)
            .is_some_and(|end| end <= input_region_bytes)
}

pub(super) fn supports_m64_threadgroup(
    execution_width: u64,
    maximum_threads: u64,
    static_bytes: u64,
    maximum_bytes: u64,
) -> bool {
    execution_width == 32
        && maximum_threads >= M64_THREADS
        && static_bytes
            .checked_add(M64_THREADGROUP_BYTES)
            .is_some_and(|required| required <= maximum_bytes)
}

#[repr(C)]
pub(super) struct NativeBlockParams {
    format: u32,
    values: u32,
    bytes: u32,
}

impl From<GgufBlockFormat> for NativeBlockParams {
    fn from(format: GgufBlockFormat) -> Self {
        Self {
            format: format.ggml_type_id(),
            values: format.block_values() as u32,
            bytes: format.block_bytes() as u32,
        }
    }
}

struct NativeGemvPipelines {
    f16: ComputePipelineState,
    f32: ComputePipelineState,
    f32_f16: ComputePipelineState,
}

struct NativeSharedPipelines {
    formats: Vec<(GgufBlockFormat, ElementType, [ComputePipelineState; 3])>,
}

impl NativeSharedPipelines {
    fn new(device: &Device, library: &metal::LibraryRef) -> Result<Self, MetalDeviceRuntimeError> {
        let batch = |format: GgufBlockFormat, suffix, rows| {
            format_pipeline(
                device,
                library,
                &format!("vnext_native_shared_linear_{suffix}_b{rows}"),
                format.ggml_type_id(),
            )
        };
        // Standard Q4/Q5/Q6/Q8 F16 linears retain their dedicated kernels.
        // Q5 F32 is the one standard format using native_linear.
        let mut formats = Vec::new();
        for format in [
            GgufBlockFormat::Q3K,
            GgufBlockFormat::Iq3S,
            GgufBlockFormat::Iq4Nl,
            GgufBlockFormat::Iq4Xs,
            GgufBlockFormat::Q5K,
            GgufBlockFormat::Pq2_0,
        ] {
            for (dtype, suffix) in [(ElementType::F16, "f16"), (ElementType::F32, "f32")] {
                if format == GgufBlockFormat::Q5K && dtype == ElementType::F16 {
                    continue;
                }
                // The former IQ4_XS F16 shared PSOs remain numerical controls
                // in tests. Production uses grouped dots for this same scope.
                #[cfg(not(test))]
                if format == GgufBlockFormat::Iq4Xs && dtype == ElementType::F16 {
                    continue;
                }
                formats.push((
                    format,
                    dtype,
                    [
                        batch(format, suffix, 2)?,
                        batch(format, suffix, 3)?,
                        batch(format, suffix, 4)?,
                    ],
                ));
            }
        }
        Ok(Self { formats })
    }

    fn pipeline(
        &self,
        format: GgufBlockFormat,
        rows: u32,
        activation_type: ElementType,
    ) -> Option<&ComputePipelineState> {
        let index = rows.checked_sub(2)? as usize;
        let (_, _, batches) = self
            .formats
            .iter()
            .find(|(candidate, dtype, _)| *candidate == format && *dtype == activation_type)?;
        batches.get(index)
    }
}

#[cfg(test)]
pub(super) struct NativePipelineInitialization {
    pub(super) library_ns: u64,
    pub(super) specialized_gemv_ns: u64,
    pub(super) shared_gemv_ns: u64,
    pub(super) generic_gemv_ns: u64,
    pub(super) specialized_gemm_ns: u64,
}

pub(super) struct MetalNativeBlockPipelines {
    q3_k: NativeGemvPipelines,
    q4_k: NativeGemvPipelines,
    q5_k: NativeGemvPipelines,
    q6_k: NativeGemvPipelines,
    q8_0: NativeGemvPipelines,
    iq3_s: NativeGemvPipelines,
    iq4_nl: NativeGemvPipelines,
    iq4_xs: NativeGemvPipelines,
    pq2_0: NativeGemvPipelines,
    pub(super) pq2_linear_f32: ComputePipelineState,
    pub(super) pq2_linear_f32_f16: ComputePipelineState,
    shared: NativeSharedPipelines,
    iq4xs_group_dot_f16: [ComputePipelineState; 4],
    pub(super) gemm_f16_f32: ComputePipelineState,
    pub(super) gemm_input_f32_output_f16: ComputePipelineState,
    pub(super) pq2_gemm_input_f32_output_f16: ComputePipelineState,
    pub(super) pq2_gemm_input_f32_output_f16_m64: Option<ComputePipelineState>,
    #[cfg(test)]
    pub(super) pq2_gemm_input_f32_output_f16_full_tiles: Option<ComputePipelineState>,
    pub(super) pq2_gemm_input_f32_output_f16_m64_full_tiles: Option<ComputePipelineState>,
    pub(super) pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input:
        Option<ComputePipelineState>,
    pub(super) iq4xs_gemm_f16_f32: ComputePipelineState,
    pub(super) iq4xs_gemm_f16_f32_m64: Option<ComputePipelineState>,
    #[cfg(test)]
    pub(super) gemm_f16_f32_m64: Option<ComputePipelineState>,
    #[cfg(test)]
    pub(super) generic_linear_f16: ComputePipelineState,
    #[cfg(test)]
    pub(super) generic_linear_f32: ComputePipelineState,
    #[cfg(test)]
    pub(super) initialization: NativePipelineInitialization,
    #[cfg(test)]
    pub(super) decode: ComputePipelineState,
}

impl MetalNativeBlockPipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let mut shader = String::from(
            "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\nconstant uint iq3_s_grid[512] = {\n",
        );
        for value in IQ3_S_GRID {
            write!(&mut shader, "0x{value:08x},").expect("writing a String cannot fail");
        }
        shader.push_str("\n};\nconstant char iq4_nl_values[16] = {");
        for value in IQ4_NL_VALUES {
            write!(&mut shader, "{value},").expect("writing a String cannot fail");
        }
        shader.push_str("};\n");
        shader.push_str(include_str!("native_blocks.metal"));
        shader.push_str(include_str!("group_dot.metal"));
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        #[cfg(test)]
        let library_started = std::time::Instant::now();
        let library = device
            .new_library_with_source(&shader, &options)
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "compile native GGUF block kernels: {error}"
                ))
            })?;
        #[cfg(test)]
        let library_ns = library_started.elapsed().as_nanos() as u64;
        let pipeline = |name: &str| {
            let function = library
                .get_function(name, None)
                .map_err(MetalDeviceRuntimeError::contract)?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(MetalDeviceRuntimeError::contract)
        };
        let gemv_pipeline =
            |name: &str, format: u32| format_pipeline(device, &library, name, format);
        let specialized = |format: GgufBlockFormat| {
            Ok::<_, MetalDeviceRuntimeError>(NativeGemvPipelines {
                f16: gemv_pipeline("vnext_native_block_linear_f16", format.ggml_type_id())?,
                f32: gemv_pipeline("vnext_native_block_linear_f32", format.ggml_type_id())?,
                f32_f16: gemv_pipeline("vnext_native_block_linear_f32_f16", format.ggml_type_id())?,
            })
        };
        // Compile the bounded format set during registry construction. Dispatch
        // only selects an already-owned PSO; it never compiles or locks a cache.
        #[cfg(test)]
        let specialized_started = std::time::Instant::now();
        let q3_k = specialized(GgufBlockFormat::Q3K)?;
        let q4_k = specialized(GgufBlockFormat::Q4K)?;
        let q5_k = specialized(GgufBlockFormat::Q5K)?;
        let q6_k = specialized(GgufBlockFormat::Q6K)?;
        let q8_0 = specialized(GgufBlockFormat::Q8_0)?;
        let iq3_s = specialized(GgufBlockFormat::Iq3S)?;
        let iq4_nl = specialized(GgufBlockFormat::Iq4Nl)?;
        let iq4_xs = specialized(GgufBlockFormat::Iq4Xs)?;
        let pq2_0 = specialized(GgufBlockFormat::Pq2_0)?;
        #[cfg(test)]
        let specialized_gemv_ns = specialized_started.elapsed().as_nanos() as u64;
        #[cfg(test)]
        let shared_started = std::time::Instant::now();
        let shared = NativeSharedPipelines::new(device, &library)?;
        #[cfg(test)]
        let shared_gemv_ns = shared_started.elapsed().as_nanos() as u64;
        // Zero is not a GGUF type supported by this decoder. It preserves the
        // runtime-format control only in conformance and performance tests.
        #[cfg(test)]
        let generic_started = std::time::Instant::now();
        #[cfg(test)]
        let generic_linear_f16 = gemv_pipeline("vnext_native_block_linear_f16", 0)?;
        #[cfg(test)]
        let generic_linear_f32 = gemv_pipeline("vnext_native_block_linear_f32", 0)?;
        #[cfg(test)]
        let generic_gemv_ns = generic_started.elapsed().as_nanos() as u64;
        // M64 is optional. Devices unable to create or execute this larger
        // threadgroup retain M32; no lazy compilation occurs during dispatch.
        #[cfg(test)]
        let gemm_f16_f32_m64 =
            optional_m64_pipeline(device, || pipeline("vnext_native_block_gemm_f16_f32_m64"));
        #[cfg(test)]
        let specialized_gemm_started = std::time::Instant::now();
        let pq2_gemm_input_f32_output_f16 = gemm_format_pipeline(
            device,
            &library,
            "vnext_native_block_gemm_input_f32_output_f16_specialized",
            GgufBlockFormat::Pq2_0,
        )?;
        let pq2_gemm_input_f32_output_f16_m64 = optional_m64_pipeline(device, || {
            gemm_format_pipeline(
                device,
                &library,
                "vnext_native_block_gemm_input_f32_output_f16_m64_specialized",
                GgufBlockFormat::Pq2_0,
            )
        });
        #[cfg(test)]
        let pq2_gemm_input_f32_output_f16_full_tiles = optional_m32_pipeline(device, || {
            gemm_format_pipeline(
                device,
                &library,
                "vnext_native_block_gemm_input_f32_output_f16_full_tiles_specialized",
                GgufBlockFormat::Pq2_0,
            )
        });
        let pq2_gemm_input_f32_output_f16_m64_full_tiles = optional_m64_pipeline(device, || {
            gemm_format_pipeline(
                device,
                &library,
                "vnext_native_block_gemm_input_f32_output_f16_m64_full_tiles_specialized",
                GgufBlockFormat::Pq2_0,
            )
        });
        let pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input = optional_m64_pipeline(
            device,
            || {
                gemm_format_pipeline(
                    device,
                    &library,
                    "vnext_native_block_gemm_input_f32_output_f16_m64_full_tiles_vector_input_specialized",
                    GgufBlockFormat::Pq2_0,
                )
            },
        );
        let iq4xs_gemm_f16_f32 = gemm_format_pipeline(
            device,
            &library,
            "vnext_native_block_gemm_f16_f32_specialized",
            GgufBlockFormat::Iq4Xs,
        )?;
        let iq4xs_gemm_f16_f32_m64 = optional_m64_pipeline(device, || {
            gemm_format_pipeline(
                device,
                &library,
                "vnext_native_block_gemm_f16_f32_m64_specialized",
                GgufBlockFormat::Iq4Xs,
            )
        });
        #[cfg(test)]
        let specialized_gemm_ns = specialized_gemm_started.elapsed().as_nanos() as u64;
        Ok(Self {
            q3_k,
            q4_k,
            q5_k,
            q6_k,
            q8_0,
            iq3_s,
            iq4_nl,
            iq4_xs,
            pq2_0,
            pq2_linear_f32: pipeline("vnext_pq2_linear_f32")?,
            pq2_linear_f32_f16: pipeline("vnext_pq2_linear_f32_f16")?,
            shared,
            iq4xs_group_dot_f16: [
                pipeline("vnext_iq4_group_dot_b1")?,
                pipeline("vnext_iq4_group_dot_b2")?,
                pipeline("vnext_iq4_group_dot_b3")?,
                pipeline("vnext_iq4_group_dot_b4")?,
            ],
            gemm_f16_f32: pipeline("vnext_native_block_gemm_f16_f32")?,
            gemm_input_f32_output_f16: pipeline("vnext_native_block_gemm_input_f32_output_f16")?,
            pq2_gemm_input_f32_output_f16,
            pq2_gemm_input_f32_output_f16_m64,
            #[cfg(test)]
            pq2_gemm_input_f32_output_f16_full_tiles,
            pq2_gemm_input_f32_output_f16_m64_full_tiles,
            pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input,
            iq4xs_gemm_f16_f32,
            iq4xs_gemm_f16_f32_m64,
            #[cfg(test)]
            gemm_f16_f32_m64,
            #[cfg(test)]
            generic_linear_f16,
            #[cfg(test)]
            generic_linear_f32,
            #[cfg(test)]
            initialization: NativePipelineInitialization {
                library_ns,
                specialized_gemv_ns,
                shared_gemv_ns,
                generic_gemv_ns,
                specialized_gemm_ns,
            },
            #[cfg(test)]
            decode: pipeline("vnext_native_block_decode")?,
        })
    }

    fn gemv(&self, format: GgufBlockFormat) -> &NativeGemvPipelines {
        match format {
            GgufBlockFormat::Q3K => &self.q3_k,
            GgufBlockFormat::Q4K => &self.q4_k,
            GgufBlockFormat::Q5K => &self.q5_k,
            GgufBlockFormat::Q6K => &self.q6_k,
            GgufBlockFormat::Q8_0 => &self.q8_0,
            GgufBlockFormat::Iq3S => &self.iq3_s,
            GgufBlockFormat::Iq4Nl => &self.iq4_nl,
            GgufBlockFormat::Iq4Xs => &self.iq4_xs,
            GgufBlockFormat::Pq2_0 => &self.pq2_0,
        }
    }

    pub(super) fn linear_f16(&self, format: GgufBlockFormat) -> &ComputePipelineState {
        &self.gemv(format).f16
    }

    pub(super) fn linear_f32(&self, format: GgufBlockFormat) -> &ComputePipelineState {
        &self.gemv(format).f32
    }

    pub(super) fn linear_f32_f16(&self, format: GgufBlockFormat) -> &ComputePipelineState {
        &self.gemv(format).f32_f16
    }

    pub(super) fn iq4xs_group_dot(&self, rows: u32) -> Option<&ComputePipelineState> {
        self.iq4xs_group_dot_f16.get(rows.checked_sub(1)? as usize)
    }

    pub(super) fn shared_linear(
        &self,
        format: GgufBlockFormat,
        rows: u32,
        activation_type: ElementType,
    ) -> Option<&ComputePipelineState> {
        self.shared.pipeline(format, rows, activation_type)
    }
}

#[cfg(test)]
fn optional_m32_pipeline(
    device: &Device,
    create: impl FnOnce() -> Result<ComputePipelineState, MetalDeviceRuntimeError>,
) -> Option<ComputePipelineState> {
    if device.max_threads_per_threadgroup().width < 128
        || device.max_threadgroup_memory_length() < 12288
    {
        return None;
    }
    create().ok().filter(|pipeline| {
        pipeline.thread_execution_width() == 32
            && pipeline.max_total_threads_per_threadgroup() >= 128
            && pipeline
                .static_threadgroup_memory_length()
                .checked_add(12288)
                .is_some_and(|bytes| bytes <= device.max_threadgroup_memory_length())
    })
}

fn optional_m64_pipeline(
    device: &Device,
    create: impl FnOnce() -> Result<ComputePipelineState, MetalDeviceRuntimeError>,
) -> Option<ComputePipelineState> {
    if device.max_threads_per_threadgroup().width < M64_THREADS
        || device.max_threadgroup_memory_length() < M64_THREADGROUP_BYTES
    {
        return None;
    }
    create().ok().filter(|pipeline| {
        supports_m64_threadgroup(
            pipeline.thread_execution_width(),
            pipeline.max_total_threads_per_threadgroup(),
            pipeline.static_threadgroup_memory_length(),
            device.max_threadgroup_memory_length(),
        )
    })
}

fn gemm_format_pipeline(
    device: &Device,
    library: &metal::LibraryRef,
    name: &str,
    format: GgufBlockFormat,
) -> Result<ComputePipelineState, MetalDeviceRuntimeError> {
    let constants = FunctionConstantValues::new();
    // The format and block geometry must match the native buffer ABI exactly.
    let block = NativeBlockParams::from(format);
    for (index, value) in [(1, block.format), (2, block.values), (3, block.bytes)] {
        constants.set_constant_value_at_index(
            &value as *const u32 as *const c_void,
            MTLDataType::UInt,
            index,
        );
    }
    let function = library
        .get_function(name, Some(constants))
        .map_err(|error| {
            MetalDeviceRuntimeError::contract(format!(
                "specialize native GGUF GEMM `{name}` {format:?}: {error}"
            ))
        })?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalDeviceRuntimeError::contract)
}

fn format_pipeline(
    device: &Device,
    library: &metal::LibraryRef,
    name: &str,
    format: u32,
) -> Result<ComputePipelineState, MetalDeviceRuntimeError> {
    let constants = FunctionConstantValues::new();
    constants.set_constant_value_at_index(
        &format as *const u32 as *const c_void,
        MTLDataType::UInt,
        0,
    );
    let function = library
        .get_function(name, Some(constants))
        .map_err(|error| {
            MetalDeviceRuntimeError::contract(format!(
                "specialize native GGUF GEMV `{name}` format={format}: {error}"
            ))
        })?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalDeviceRuntimeError::contract)
}

pub(super) fn bind_native_block(
    encoder: &ComputeCommandEncoderRef,
    format: GgufBlockFormat,
    index: u64,
) {
    let params = NativeBlockParams::from(format);
    encoder.set_bytes(
        index,
        std::mem::size_of::<NativeBlockParams>() as u64,
        &params as *const _ as *const c_void,
    );
}

pub(super) fn dispatch_m64_grid(encoder: &ComputeCommandEncoderRef, rows: u32, out_features: u32) {
    // X[64][32] + W[32][64] floats, reused as the full Y[64][64] tile.
    encoder.set_threadgroup_memory_length(0, M64_THREADGROUP_BYTES);
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(
            u64::from(rows).div_ceil(64),
            u64::from(out_features).div_ceil(64),
            1,
        ),
        metal::MTLSize::new(M64_THREADS, 1, 1),
    );
}

#[cfg(test)]
mod tests;
