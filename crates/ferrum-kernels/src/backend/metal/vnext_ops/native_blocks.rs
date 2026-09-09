use std::ffi::c_void;
use std::fmt::Write;

use metal::{CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device};

use crate::backend::metal::vnext_runtime::MetalDeviceRuntimeError;
use crate::gguf_blocks::{GgufBlockFormat, IQ3_S_GRID, IQ4_NL_VALUES};

pub(super) const FINGERPRINT_SOURCE: &str = concat!(
    include_str!("native_blocks.rs"),
    include_str!("native_blocks.metal"),
    include_str!("../../../gguf_blocks/iq3s_grid.rs"),
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

pub(super) struct MetalNativeBlockPipelines {
    pub(super) linear_f16: ComputePipelineState,
    pub(super) linear_f32: ComputePipelineState,
    #[cfg(test)]
    pub(super) decode: ComputePipelineState,
}

impl MetalNativeBlockPipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let mut shader = String::from(
            "#include <metal_stdlib>\nusing namespace metal;\nconstant uint iq3_s_grid[512] = {\n",
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
        let library = device
            .new_library_with_source(&shader, &options)
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "compile native GGUF block kernels: {error}"
                ))
            })?;
        let pipeline = |name: &str| {
            let function = library
                .get_function(name, None)
                .map_err(MetalDeviceRuntimeError::contract)?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(MetalDeviceRuntimeError::contract)
        };
        Ok(Self {
            linear_f16: pipeline("vnext_native_block_linear_f16")?,
            linear_f32: pipeline("vnext_native_block_linear_f32")?,
            #[cfg(test)]
            decode: pipeline("vnext_native_block_decode")?,
        })
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
