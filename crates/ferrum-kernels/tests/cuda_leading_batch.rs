#![cfg(feature = "cuda")]
#[path = "support/leading_batch_qualification.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use std::sync::Arc;

#[test]
#[ignore = "requires CUDA host; actual rank bridges, all BF16 encodings and owner retirement"]
fn leading_batch_managed_gpu_copy_preserves_axes_bits_and_owners() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 16,
            opaque_memory_profile: "leading-batch-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(LeadingBatchBf16Kind::Merge.contract().unwrap()),
            Box::new(LeadingBatchBf16Kind::Split.contract().unwrap()),
            Box::new(LeadingBatchBf16Kind::Merge.view_contract().unwrap()),
            Box::new(LeadingBatchBf16Kind::Split.view_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::MergeLeadingBatch,
                CudaTensorOperation::SplitLeadingBatch,
                CudaTensorOperation::MergeLeadingBatchView,
                CudaTensorOperation::SplitLeadingBatchView,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        fixture::id("provider.engine.leading_batch"),
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
    checks::run(runtime, &registry, &catalog, &materializers);
}
