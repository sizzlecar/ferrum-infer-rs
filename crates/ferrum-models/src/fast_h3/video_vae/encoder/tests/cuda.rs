use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_tensor_providers_with_cudnn_selection,
        CudaCudnnOperation, CudaTensorOperation,
    },
    vnext_runtime::{CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
#[test]
#[ignore = "requires exclusive CUDA and explicit cuDNN library/dependencies"]
fn full_causal_cuda_encoder_matches_independent_f64_volume_and_future_isolation() {
    let (runtime, registry, catalog) = composition(1 << 30);
    qualify(
        runtime,
        &registry,
        &catalog,
        FastH3VideoEncoderProfile::Cuda,
    );
}
fn composition(
    device_bytes: u64,
) -> (
    Arc<CudaDeviceRuntime>,
    OperationRuntimeRegistry<CudaDeviceRuntime>,
    CapabilityCatalog,
) {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0").unwrap()).unwrap(),
            usable_memory_bytes: device_bytes,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 256,
            opaque_memory_profile: "unqualified-causal-video-encoder-fixture-v1".into(),
        })
        .unwrap(),
    );
    let config = CudaCudnnConfig {
        library_path: std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY")
            .expect("explicit library")
            .into(),
        dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default(),
        opaque_memory_bytes: 256 << 20,
        workspace_limit_bytes: 64 << 20,
        opaque_memory_profile: "fixture.video_encoder.f32.fma.v1".into(),
    };
    let registry = OperationRuntimeRegistry::new(
        [
            F32CnnOp::Conv3d,
            F32CnnOp::FrameGroupNormSilu,
            F32CnnOp::VolumeResidualAdd,
        ]
        .into_iter()
        .map(|op| Box::new(op.contract().unwrap()) as Box<dyn OperationContract>)
        .collect(),
        cuda_vnext_tensor_providers_with_cudnn_selection(
            &runtime,
            &config,
            None,
            &[CudaCudnnOperation::Conv3dF32],
            &[
                CudaTensorOperation::FrameGroupNormSiluNcthwF32,
                CudaTensorOperation::PlainResidualAddNcthwF32,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.video_encoder.fixture").unwrap(),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(d.clone(), vec![engine])
        .unwrap();
    (runtime, registry, catalog)
}

mod released;
