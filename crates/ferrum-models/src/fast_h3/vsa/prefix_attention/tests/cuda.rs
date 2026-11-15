use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_tensor_providers_with_cudnn_cross_attention,
        CudaTensorOperation,
    },
    vnext_runtime::{
        CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig, CudnnSdpaPlanPolicy,
    },
};
#[test]
#[ignore = "requires CUDA and explicit FERRUM_TEST_CUDNN_LIBRARY; <=2 BF16 ULP tiny FP64/table oracle, shared map/managed retirement, no upstream bit parity"]
fn prefix_cuda_centers_roundtrips_and_compacts_valid_keys_in_one_graph() {
    // Test artifact location only; production callers pass the typed config.
    // Missing evidence never turns into a passing/skipped GPU qualification.
    let library_path = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY")
        .expect("supply trusted absolute cuDNN library path")
        .into();
    let dependency_libraries = std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
        .map(|s| std::env::split_paths(&s).collect())
        .unwrap_or_default();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0").unwrap()).unwrap(),
            usable_memory_bytes: 2 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 32,
            opaque_memory_profile: "prefix-attention-qualification".into(),
        })
        .unwrap(),
    );
    let cudnn = CudaCudnnConfig {
        library_path,
        dependency_libraries,
        opaque_memory_bytes: 512 << 20,
        workspace_limit_bytes: 256 << 20,
        opaque_memory_profile: "prefix-attention-cudnn9-declared.v1".into(),
    };
    let policy = CudnnSdpaPlanPolicy {
        maximum_plans: 2,
        maximum_live_bindings: 8,
        opaque_memory_bytes: 128 << 20,
        workspace_limit_bytes: 256 << 20,
    };
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(row_center_bf16_contract().unwrap()),
        Box::new(head_fp8_roundtrip_bf16_contract().unwrap()),
        Box::new(indexed_head_fp8_roundtrip_bf16_contract().unwrap()),
        Box::new(row_gather_bf16_contract().unwrap()),
        Box::new(row_index_gather_or_zero_bf16_contract().unwrap()),
        Box::new(LeadingBatchBf16Kind::Merge.view_contract().unwrap()),
        Box::new(LeadingBatchBf16Kind::Split.view_contract().unwrap()),
        Box::new(cross_attention_fused_bf16_contract().unwrap()),
        Box::new(structured_attention_fused_bf16_contract().unwrap()),
        Box::new(conv2d_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(
        contracts,
        cuda_vnext_tensor_providers_with_cudnn_cross_attention(
            &runtime,
            &cudnn,
            policy,
            &[
                CudaTensorOperation::RowCenterBf16,
                CudaTensorOperation::HeadFp8RoundtripBf16,
                CudaTensorOperation::IndexedHeadFp8RoundtripBf16,
                CudaTensorOperation::RowGather,
                CudaTensorOperation::RowIndexGather,
                CudaTensorOperation::MergeLeadingBatchView,
                CudaTensorOperation::SplitLeadingBatchView,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let m = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let e = EngineProviderDescriptor::new(
        id("provider.engine.prefix").unwrap(),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = m
        .augment_catalog(registry.capability_catalog(d.clone(), vec![e]).unwrap())
        .unwrap();
    qualify(
        runtime,
        &registry,
        &catalog,
        &m,
        FastH3PrefixAttentionArithmetic::VendorFused,
    );
}
