#![cfg(feature = "cuda")]
#[path = "support/dynamic_preparation_qualification.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::cuda::{vnext_ops::*, vnext_runtime::*};
use ferrum_runtime::*;
use std::sync::Arc;
#[test]
#[ignore = "requires CUDA; actual zero-submit quota refusal, B1 retry and B2 copy, no performance claim"]
fn cuda_dynamic_preparation_refuses_b2_quota_then_runs_original_b1_and_released_b2() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 8,
            opaque_memory_profile: "dynamic-preparation-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(LeadingBatchBf16Kind::Merge.contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::MergeLeadingBatch]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        fixture::id("provider.engine.dynamic_prepare"),
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
    checks::run(
        runtime,
        &registry,
        &catalog,
        &materializers,
        1 << 30,
        None,
        None,
    );
}
