#![cfg(feature = "cuda")]
#[path = "support/rms_modulation_component.rs"]
mod fixture;
#[path = "support/rms_modulation_qualification.rs"]
mod qualification;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use std::sync::Arc;
#[test]
#[ignore = "requires configured CUDA host; bounded-domain numerical qualification"]
fn fused_rms_modulation_preserves_f32_residual_and_close_ownership() {
    let r = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 16,
            opaque_memory_profile: "rms-modulation-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(row_selected_rms_modulation_bf16_contract().unwrap()),
            Box::new(row_selected_residual_rms_modulation_bf16_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers(
            &r,
            &[
                CudaTensorOperation::RowSelectedRmsModulation,
                CudaTensorOperation::RowSelectedResidualRmsModulation,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    for (rows, width) in [(3, 2), (3, 5376), (3, 8191), (67, 7)] {
        for residual in [false, true] {
            qualification::qualify(
                r.clone(),
                &registry,
                fixture::Config {
                    rows,
                    width,
                    residual,
                },
            );
        }
    }
}
