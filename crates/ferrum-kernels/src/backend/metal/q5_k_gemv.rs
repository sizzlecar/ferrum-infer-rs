//! Shared Q5_K cooperative GEMV for native Metal execution plans.

#![cfg(all(target_os = "macos", feature = "metal"))]

use metal::{CompileOptions, ComputePipelineState, Device};

const SHADER_SRC: &str = include_str!("q5_k_gemv.metal");
pub(crate) const F16_BATCHED_KERNEL_NAME: &str = "gemv_f16a_q5kw_v2_batched";

pub(crate) fn new_f16_batched_pipeline(device: &Device) -> Result<ComputePipelineState, String> {
    let library = device
        .new_library_with_source(SHADER_SRC, &CompileOptions::new())
        .map_err(|error| format!("compile shared Q5_K GEMV library: {error}"))?;
    let function = library
        .get_function(F16_BATCHED_KERNEL_NAME, None)
        .map_err(|error| format!("load shared Q5_K GEMV kernel: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("build shared Q5_K GEMV pipeline: {error}"))
}
