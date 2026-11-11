#![cfg(feature = "cuda")]
#[path = "support/row_selected_bf16_component.rs"]
mod fixture;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use std::sync::{Arc, Mutex};
static GPU: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires configured CUDA and pinned native operator set; actual compact versus expanded BF16 graphs"]
fn compact_bf16_routing_matches_expanded_graphs_and_preserves_request_isolation() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "unqualified-compact-bf16-routing-fixture-v1".into(),
        })
        .unwrap(),
    );
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(row_selected_modulation_bf16_contract().unwrap()),
        Box::new(row_selected_addcmul_bf16_contract().unwrap()),
        Box::new(modulation_scale_bf16_contract().unwrap()),
        Box::new(plain_residual_add_bf16_contract().unwrap()),
        Box::new(addcmul_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(
        contracts,
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::RowSelectedModulationBf16,
                CudaTensorOperation::RowSelectedAddcmulBf16,
                CudaTensorOperation::ModulationScale,
                CudaTensorOperation::PlainResidualAdd,
                CudaTensorOperation::Addcmul,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        fixture::id("provider.engine.compact.bf16.fixture"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(descriptor.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    fixture::run(runtime, &registry, &catalog, &materializers);
}
