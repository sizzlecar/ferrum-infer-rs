#![cfg(feature = "cuda")]
#[path = "support/gqa_f32_nonpower_component.rs"]
mod fixture;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use std::sync::{Arc, Mutex};
static GPU: Mutex<()> = Mutex::new(());
#[test]
#[ignore = "requires configured CUDA/native operator set; actual F32 D72 and nonpower compact GQA"]
fn f32_d72_and_nonpower_heads_execute_original_compact_kv_without_half_rounding() {
    let _guard = GPU
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
            opaque_memory_profile: "unqualified-gqa-f32-general-width-fixture-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(structured_gqa_attention_f32_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::StructuredGqaAttentionF32])
            .unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        fixture::id("provider.engine.gqa_f32.general"),
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
