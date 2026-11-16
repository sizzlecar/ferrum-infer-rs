//! Shared test setup for the core and its actual enclosing DiT block.
use super::*;
use ferrum_kernels::{
    backend::cuda::{vnext_ops::*, vnext_runtime::*},
    native_ops::sage_vsa_fp4::{CheckedVsaRegions, SageVsaFp4Config},
};
use std::num::NonZeroUsize;
fn ident<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).unwrap()
}
pub(in crate::fast_h3::package::tests) fn create(
    regions: Arc<CheckedVsaRegions>,
    extra_operations: &[CudaTensorOperation],
    extra_contracts: Vec<Box<dyn OperationContract>>,
) -> (
    Arc<CudaDeviceRuntime>,
    Arc<OperationRuntimeRegistry<CudaDeviceRuntime>>,
) {
    let runtime = runtime();
    let factory = SharedProviders::new(runtime.clone(), extra_operations);
    let registry = factory.registry(regions, extra_contracts);
    (runtime, registry)
}

pub(in crate::fast_h3::package::tests) fn runtime() -> Arc<CudaDeviceRuntime> {
    runtime_with_command_budget(128)
}

pub(in crate::fast_h3::package::tests) fn runtime_with_command_budget(
    maximum_commands_per_submission: usize,
) -> Arc<CudaDeviceRuntime> {
    let mut device = cuda_tensor_runtime_config(0, ident("device.cuda.0")).unwrap();
    cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device).unwrap();
    cuda_tensor_runtime_enable_sage_vsa_fp4_regions(&mut device).unwrap();
    cuda_tensor_runtime_enable_sage_vsa_fp4_quantize(&mut device).unwrap();
    Arc::new(
        CudaDeviceRuntime::new_tensor_eager_with_options(
            CudaTensorRuntimeConfig {
                device,
                usable_memory_bytes: 3 << 30,
                bootstrap_memory_bytes: 256 << 20,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 1,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission,
                opaque_memory_profile: "attention-core-qualification".into(),
            },
            CudaTensorRuntimeOptions {
                pinned_weight_upload: Some(CudaPinnedWeightUploadConfig {
                    minimum_allocation_bytes: 0,
                    maximum_allocation_bytes: NonZeroUsize::new(16 << 20).unwrap(),
                    maximum_registered_bytes: NonZeroUsize::new(32 << 20).unwrap(),
                }),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

/// Same runtime, native modules, cuDNN cache and provider objects. Only the
/// retained checked region provider varies with each cohort's B dimension.
pub(in crate::fast_h3::package::tests) struct SharedProviders {
    runtime: Arc<CudaDeviceRuntime>,
    providers: Vec<Arc<dyn OperationProvider<CudaDeviceRuntime>>>,
    native: SageVsaFp4Config,
}
impl SharedProviders {
    pub(in crate::fast_h3::package::tests) fn new(
        runtime: Arc<CudaDeviceRuntime>,
        extra_operations: &[CudaTensorOperation],
    ) -> Self {
        let mut selected = vec![
            CudaTensorOperation::BatchedMatmulF32,
            CudaTensorOperation::ScaleRank4F32,
            CudaTensorOperation::RowSoftmaxRank4F32,
            CudaTensorOperation::RowCenterBf16,
            CudaTensorOperation::HeadFp8RoundtripBf16,
            CudaTensorOperation::IndexedHeadFp8RoundtripBf16,
            CudaTensorOperation::RowIndexGather,
            CudaTensorOperation::MergeLeadingBatchView,
            CudaTensorOperation::SplitLeadingBatchView,
            CudaTensorOperation::BatchedRowConcat,
            CudaTensorOperation::PackHeadAxisBf16,
            CudaTensorOperation::PackHeadAxisF32ToBf16,
            CudaTensorOperation::AddcmulFma,
        ];
        selected.extend_from_slice(extra_operations);
        let cudnn = CudaCudnnConfig {
            library_path: std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY")
                .expect("explicit trusted cuDNN library path")
                .into(),
            dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
                .map(|s| std::env::split_paths(&s).collect())
                .unwrap_or_default(),
            opaque_memory_bytes: 512 << 20,
            workspace_limit_bytes: 256 << 20,
            opaque_memory_profile: "attention-core-cudnn9-qualification".into(),
        };
        let mut providers = cuda_vnext_tensor_providers_with_cudnn_cross_attention(
            &runtime,
            &cudnn,
            CudnnSdpaPlanPolicy {
                maximum_plans: 2,
                maximum_live_bindings: 8,
                opaque_memory_bytes: 128 << 20,
                workspace_limit_bytes: 256 << 20,
            },
            &selected,
        )
        .unwrap();
        let native = SageVsaFp4Config {
            opaque_memory_bytes: 64 << 20,
            opaque_memory_profile: "attention-core-shared-sage-module".into(),
        };
        providers
            .push(cuda_vnext_sparse_fp4_qkv_quantize_bf16_provider(&runtime, &native).unwrap());
        providers.extend(
            cuda_vnext_nvfp4_bf16_providers(
                &runtime,
                &CutlassNvfp4Config {
                    opaque_memory_bytes: 64 << 20,
                    opaque_memory_profile: "attention-core-shared-nvfp4-module".into(),
                },
            )
            .unwrap(),
        );
        Self {
            runtime,
            providers: providers.into_iter().map(Arc::from).collect(),
            native,
        }
    }

    pub(in crate::fast_h3::package::tests) fn registry(
        &self,
        regions: Arc<CheckedVsaRegions>,
        extra_contracts: Vec<Box<dyn OperationContract>>,
    ) -> Arc<OperationRuntimeRegistry<CudaDeviceRuntime>> {
        let mut providers = self.providers.clone();
        providers.push(Arc::from(
            cuda_vnext_sparse_fp4_attention_region_topk_provider(
                &self.runtime,
                &self.native,
                regions,
            )
            .unwrap(),
        ));
        let mut contracts: Vec<Box<dyn OperationContract>> = vec![
            Box::new(batched_matmul_f32_contract().unwrap()),
            Box::new(scale_rank4_f32_contract().unwrap()),
            Box::new(row_softmax_rank4_f32_contract().unwrap()),
            Box::new(row_center_bf16_contract().unwrap()),
            Box::new(head_fp8_roundtrip_bf16_contract().unwrap()),
            Box::new(indexed_head_fp8_roundtrip_bf16_contract().unwrap()),
            Box::new(row_index_gather_or_zero_bf16_contract().unwrap()),
            Box::new(LeadingBatchBf16Kind::Merge.view_contract().unwrap()),
            Box::new(LeadingBatchBf16Kind::Split.view_contract().unwrap()),
            Box::new(batched_row_concat_bf16_contract().unwrap()),
            Box::new(HeadAxisPackKind::Bf16.contract().unwrap()),
            Box::new(HeadAxisPackKind::F32ToBf16.contract().unwrap()),
            Box::new(addcmul_fma_bf16_contract().unwrap()),
            Box::new(sparse_fp4_qkv_quantize_bf16_contract().unwrap()),
            Box::new(sparse_fp4_attention_region_topk_contract().unwrap()),
            Box::new(nvfp4_quantize_bf16_contract().unwrap()),
            Box::new(nvfp4_matmul_bf16_contract().unwrap()),
            Box::new(cross_attention_fused_bf16_contract().unwrap()),
            Box::new(structured_attention_fused_bf16_contract().unwrap()),
            Box::new(conv2d_bf16_contract().unwrap()),
        ];
        contracts.extend(extra_contracts);
        Arc::new(OperationRuntimeRegistry::from_shared_providers(contracts, providers).unwrap())
    }
}
