//! Shared tiled quantized GEMM pipelines for F16 prefill execution.

#![cfg(all(target_os = "macos", feature = "metal"))]

use metal::{CompileOptions, ComputePipelineState, Device};

pub(crate) const SHADER_SOURCE: &str = concat!(
    include_str!("k_quant_gemm.metal"),
    "\n",
    include_str!("k_quant_gemm_m8.metal"),
);

pub(crate) struct MetalKQuantGemmPipelines {
    pub(crate) q4_k: ComputePipelineState,
    pub(crate) q5_k: ComputePipelineState,
    pub(crate) q6_k: ComputePipelineState,
    pub(crate) q4_k_m8: ComputePipelineState,
    pub(crate) q5_k_m8: ComputePipelineState,
    pub(crate) q6_k_m8: ComputePipelineState,
    pub(crate) q8_0: ComputePipelineState,
    pub(crate) stage_q4_k: ComputePipelineState,
    pub(crate) stage_q5_k: ComputePipelineState,
    pub(crate) stage_q6_k: ComputePipelineState,
    pub(crate) staged_f16: ComputePipelineState,
}

impl MetalKQuantGemmPipelines {
    /// Independent test PSO; production construction and selection are unchanged.
    #[cfg(test)]
    pub(crate) fn staged_contiguous_store_for_test(
        device: &Device,
    ) -> Result<ComputePipelineState, String> {
        let source = format!("#define FERRUM_TEST_STAGED_CONTIGUOUS_STORE 1\n{SHADER_SOURCE}");
        let library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(|error| format!("compile staged contiguous-store test library: {error}"))?;
        let function = library
            .get_function("gemm_f16a_f16w_tiled_contiguous_store", None)
            .map_err(|error| format!("load staged contiguous-store test kernel: {error}"))?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("build staged contiguous-store test pipeline: {error}"))
    }

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
            q4_k_m8: pipeline("gemm_f16a_q4kw_m8")?,
            q5_k_m8: pipeline("gemm_f16a_q5kw_m8")?,
            q6_k_m8: pipeline("gemm_f16a_q6kw_m8")?,
            q8_0: pipeline("gemm_f16a_q8_0w_tiled")?,
            stage_q4_k: pipeline("stage_q4k_f16")?,
            stage_q5_k: pipeline("stage_q5k_f16")?,
            stage_q6_k: pipeline("stage_q6k_f16")?,
            staged_f16: pipeline("gemm_f16a_f16w_tiled")?,
        })
    }
}
