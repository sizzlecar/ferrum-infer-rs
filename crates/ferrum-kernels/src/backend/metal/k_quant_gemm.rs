//! Shared tiled quantized GEMM pipelines for F16 prefill execution.

#![cfg(all(target_os = "macos", feature = "metal"))]

use metal::{CompileOptions, ComputePipelineState, Device};

pub(crate) const SHADER_SOURCE: &str = include_str!("k_quant_gemm.metal");

pub(crate) struct MetalKQuantGemmPipelines {
    pub(crate) q4_k: ComputePipelineState,
    pub(crate) q5_k: ComputePipelineState,
    pub(crate) q6_k: ComputePipelineState,
    pub(crate) q8_0: ComputePipelineState,
}

impl MetalKQuantGemmPipelines {
    pub(crate) fn new(device: &Device) -> Result<Self, String> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|error| format!("compile shared quantized GEMM library: {error}"))?;
        let pipeline = |name: &str| {
            let function = library
                .get_function(name, None)
                .map_err(|error| format!("load shared quantized GEMM kernel `{name}`: {error}"))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| format!("build shared quantized GEMM kernel `{name}`: {error}"))
        };
        Ok(Self {
            q4_k: pipeline("gemm_f16a_q4kw_tiled")?,
            q5_k: pipeline("gemm_f16a_q5kw_tiled")?,
            q6_k: pipeline("gemm_f16a_q6kw_tiled")?,
            q8_0: pipeline("gemm_f16a_q8_0w_tiled")?,
        })
    }
}
