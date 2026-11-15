#![cfg(feature = "cuda")]
#[path = "support/packed_swiglu_component.rs"]
mod fixture;
#[path = "support/packed_swiglu_qualification.rs"]
mod qualification;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use std::sync::Arc;

#[test]
#[ignore = "requires configured CUDA host; reports bounded-domain numerical qualification"]
fn packed_swiglu_matches_declared_rounding_and_retires_managed_owners() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 16,
            opaque_memory_profile: "packed-swiglu-numerical-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(packed_swiglu_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::PackedSwiGlu]).unwrap(),
    )
    .unwrap();
    for profile in [
        ferrum_runtime::PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound,
        ferrum_runtime::PackedSwiGluBf16Profile::SiluBf16ThenMultiply,
    ] {
        for (rows, width) in [(2, 17), (3, 129)] {
            qualification::qualify(
                runtime.clone(),
                &registry,
                fixture::Config {
                    rows,
                    width,
                    profile,
                },
            );
        }
    }
}
