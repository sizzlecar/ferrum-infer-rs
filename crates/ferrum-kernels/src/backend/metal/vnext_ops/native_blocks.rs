use std::ffi::c_void;
use std::fmt::Write;

use metal::{
    CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, FunctionConstantValues,
    MTLDataType,
};

use crate::backend::metal::vnext_runtime::MetalDeviceRuntimeError;
use crate::gguf_blocks::{GgufBlockFormat, IQ3_S_GRID, IQ4_NL_VALUES};

pub(super) const FINGERPRINT_SOURCE: &str = concat!(
    include_str!("native_blocks.rs"),
    include_str!("native_blocks.metal"),
    include_str!("../../../gguf_blocks/iq3s_grid.rs"),
    include_str!("../../../gguf_blocks/iq4nl_values.rs"),
    include_str!("../../../gguf_blocks/mod.rs"),
);

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
}

#[cfg(test)]
pub(super) struct NativePipelineInitialization {
    pub(super) library_ns: u64,
    pub(super) specialized_gemv_ns: u64,
    pub(super) generic_gemv_ns: u64,
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
    pub(super) gemm_f16_f32: ComputePipelineState,
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
        let gemv_pipeline = |name: &str, format: u32| {
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
        };
        let specialized = |format: GgufBlockFormat| {
            Ok::<_, MetalDeviceRuntimeError>(NativeGemvPipelines {
                f16: gemv_pipeline("vnext_native_block_linear_f16", format.ggml_type_id())?,
                f32: gemv_pipeline("vnext_native_block_linear_f32", format.ggml_type_id())?,
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
        #[cfg(test)]
        let specialized_gemv_ns = specialized_started.elapsed().as_nanos() as u64;
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
        Ok(Self {
            q3_k,
            q4_k,
            q5_k,
            q6_k,
            q8_0,
            iq3_s,
            iq4_nl,
            iq4_xs,
            gemm_f16_f32: pipeline("vnext_native_block_gemm_f16_f32")?,
            #[cfg(test)]
            generic_linear_f16,
            #[cfg(test)]
            generic_linear_f32,
            #[cfg(test)]
            initialization: NativePipelineInitialization {
                library_ns,
                specialized_gemv_ns,
                generic_gemv_ns,
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
        }
    }

    pub(super) fn linear_f16(&self, format: GgufBlockFormat) -> &ComputePipelineState {
        &self.gemv(format).f16
    }

    pub(super) fn linear_f32(&self, format: GgufBlockFormat) -> &ComputePipelineState {
        &self.gemv(format).f32
    }
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

#[cfg(test)]
mod tests;
