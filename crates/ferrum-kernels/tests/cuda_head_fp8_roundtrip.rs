#![cfg(feature = "cuda")]
#[path = "support/head_fp8_roundtrip_component.rs"]
mod fixture;
#[path = "support/head_fp8_roundtrip_qualification.rs"]
mod qualification;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use std::sync::Arc;

#[test]
#[ignore = "requires configured CUDA host; reports bounded-domain numerical qualification"]
fn head_fp8_roundtrip_matches_e4m3fn_zero_scale_roundtrip_and_retires_managed_owners() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 16,
            opaque_memory_profile: "head-fp8-roundtrip-numerical-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(head_fp8_roundtrip_bf16_contract().unwrap()),
            Box::new(indexed_head_fp8_roundtrip_bf16_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::HeadFp8RoundtripBf16,
                CudaTensorOperation::IndexedHeadFp8RoundtripBf16,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    // Warp tails, head-width transition, multiple batches, and a grid-stride
    // tail beyond 65,535 CTAs. The scalar-width case bounds the large oracle.
    for (batches, heads, rows, width) in [
        (2, 2, 3, 7),
        (2, 2, 3, 128),
        (2, 2, 3, 129),
        (1, 1, 5, 31),
        (1, 1, 5, 32),
        (1, 1, 5, 33),
        (1, 1, 5, 256),
        (1, 1, 5, 257),
        (1, 1, 262145, 1),
    ] {
        qualification::qualify(
            runtime.clone(),
            &registry,
            fixture::Config {
                batches,
                heads,
                rows,
                width,
                selected_rows: None,
            },
        );
    }
    for (width, selected_rows) in [(7, 11), (128, 11), (257, 11), (1, 262145)] {
        qualification::qualify(
            runtime.clone(),
            &registry,
            fixture::Config {
                batches: 2,
                heads: 2,
                rows: 7,
                width,
                selected_rows: Some(selected_rows),
            },
        );
    }
}
