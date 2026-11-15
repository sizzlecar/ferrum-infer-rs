use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
#[test]
#[ignore = "requires CUDA: shared key-centering fixture covers staged BF16 rounding, padded rows and owner retirement"]
fn key_center_cuda_matches_independent_mean_and_closes_both_outputs() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0").unwrap()).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 16,
            opaque_memory_profile: "key-center-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(row_center_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::RowCenterBf16]).unwrap(),
    )
    .unwrap();
    let m = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.key_center").unwrap(),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = m
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    qualify(runtime, &registry, &catalog, &m);
}
