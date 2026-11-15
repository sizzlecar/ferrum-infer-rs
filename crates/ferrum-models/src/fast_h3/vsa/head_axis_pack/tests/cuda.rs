use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
#[test]
#[ignore = "requires CUDA: shared layout fixture covers all BF16 bits, independent RNE boundaries and owner retirement"]
fn head_axis_pack_cuda_preserves_bits_and_narrows_once_before_untile_or_expand() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0").unwrap()).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 16,
            opaque_memory_profile: "head-axis-pack-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        HeadAxisPackKind::ALL
            .into_iter()
            .map(|kind| Box::new(kind.contract().unwrap()) as Box<dyn OperationContract>)
            .collect(),
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::PackHeadAxisBf16,
                CudaTensorOperation::PackHeadAxisF32ToBf16,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let m = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.head_axis_pack").unwrap(),
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
