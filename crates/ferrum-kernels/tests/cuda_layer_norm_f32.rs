#![cfg(feature = "cuda")]
#[path = "support/layer_norm_f32_component.rs"]
mod fixture;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use std::sync::{Arc, Mutex};
static GPU: Mutex<()> = Mutex::new(());
#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn stored_f32_layer_norm_executes_two_pass_stats_without_half_or_fused_affine() {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "unqualified-layer-norm-f32-fixture-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(layer_norm_f32_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::LayerNormF32]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        fixture::id("provider.engine.layer_norm_f32.fixture"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    fixture::run(runtime, &registry, &catalog, &materializers);
}
