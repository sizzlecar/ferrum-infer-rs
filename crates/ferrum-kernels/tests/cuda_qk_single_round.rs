#![cfg(feature = "cuda")]
#[path = "support/qk_single_round_component.rs"]
mod fixture;
#[path = "support/qk_single_round_qualification.rs"]
mod qualification;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use std::sync::Arc;
#[test]
#[ignore = "requires configured CUDA host; bounded-domain numerical qualification"]
fn qk_single_round_actual_heads_tail_and_retained_owners() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 16,
            opaque_memory_profile: "qk-single-round-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(
            head_rms_partial_rope_single_round_bf16_contract().unwrap(),
        )],
        cuda_vnext_tensor_providers(
            &runtime,
            &[CudaTensorOperation::HeadRmsPartialRopeSingleRound],
        )
        .unwrap(),
    )
    .unwrap();
    for (head, rot, heads) in [(2, 2, 2), (4, 2, 2), (128, 96, 56), (127, 96, 3)] {
        qualification::qualify(
            runtime.clone(),
            &registry,
            fixture::Config {
                rows: 3,
                width: head * heads,
                head,
                rot,
            },
        );
    }
}
