#![cfg(feature = "cuda")]
//! Managed CUDA selected-row arithmetic and ownership; no throughput or
//! upstream binary parity is inferred from this fixture.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
#[path = "support/row_scatter_f32_component.rs"]
mod scatter;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use fixture::id;
use std::sync::Arc;

#[test]
#[ignore = "requires configured CUDA host with admitted F32 selected-row fixture capacity"]
fn f32_visual_rows_execute_without_half_rounding_and_retain_their_original_owners() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "row-scatter-f32-fixture-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(row_scatter_replace_f32_contract().unwrap()),
            Box::new(row_scatter_add_f32_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::RowScatterReplaceF32,
                CudaTensorOperation::RowScatterAddF32,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.row_scatter_f32.fixture"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(runtime.descriptor().clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    scatter::exercise(runtime, &registry, &catalog, &materializers);
}
