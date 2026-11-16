use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_cnn_f32_providers_with_segmented_sdpa,
        CudaTensorOperation as Op,
    },
    vnext_runtime::{
        CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig, CudnnSdpaPlanPolicy,
    },
};

#[test]
#[ignore = "requires exclusive CUDA and explicit cuDNN library/dependencies"]
fn cuda_video_vision_preserves_temporal_isolation_main_and_three_deepstack_outputs() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 256,
            opaque_memory_profile: "unqualified-qwen-video-fixture-v1".into(),
        })
        .unwrap(),
    );
    let cudnn = CudaCudnnConfig {
        library_path: std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY")
            .expect("explicit cuDNN library")
            .into(),
        dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default(),
        opaque_memory_bytes: 256 << 20,
        workspace_limit_bytes: 64 << 20,
        opaque_memory_profile: "fixture.qwen_video.segmented_sdpa.v1".into(),
    };
    let providers = cuda_vnext_cnn_f32_providers_with_segmented_sdpa(
        &runtime,
        &cudnn,
        CudnnSdpaPlanPolicy {
            maximum_plans: 4,
            maximum_live_bindings: 12,
            opaque_memory_bytes: 64 << 20,
            workspace_limit_bytes: 64 << 20,
        },
        &[
            Op::AffineLinear,
            Op::PatchProjection,
            Op::AffineLayerNorm,
            Op::GeluErf,
            Op::GeluTanh,
            Op::HalfRotary,
            Op::ReshapeCopy,
            Op::SegmentedAttention,
            Op::RowGather,
            Op::PlainResidualAdd,
        ],
    )
    .unwrap();
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv2d_f32_contract().unwrap()),
            Box::new(affine_linear_bf16_contract().unwrap()),
            Box::new(patch_projection_bf16_contract().unwrap()),
            Box::new(affine_layer_norm_bf16_contract().unwrap()),
            Box::new(gelu_erf_bf16_contract().unwrap()),
            Box::new(gelu_tanh_bf16_contract().unwrap()),
            Box::new(half_rotary_bf16_contract().unwrap()),
            Box::new(reshape_copy_bf16_contract().unwrap()),
            Box::new(segmented_attention_bf16_contract().unwrap()),
            Box::new(row_gather_bf16_contract().unwrap()),
            Box::new(plain_residual_add_bf16_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.qwen_video.fixture"),
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
    qualify(
        runtime,
        &registry,
        &catalog,
        &materializers,
        Qwen3VLVisionFamily::CUDA_PROFILE_ID,
        72,
    );
}
