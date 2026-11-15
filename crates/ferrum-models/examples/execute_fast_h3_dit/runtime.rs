use super::*;
use ferrum_kernels::{
    backend::cuda::{vnext_ops::*, vnext_runtime::*},
    native_ops::sage_vsa_fp4::{CheckedVsaRegions, SageVsaFp4Config},
};
use std::num::NonZeroUsize;
pub(super) struct Context {
    pub runtime: Arc<CudaDeviceRuntime>,
    pub registry: Arc<OperationRuntimeRegistry<CudaDeviceRuntime>>,
    pub catalog: CapabilityCatalog,
    pub policy: TensorComponentPolicy,
    pub materializers: WeightMaterializerRegistry,
}
pub(super) fn create(a: &Config, regions: Arc<CheckedVsaRegions>) -> Result<Arc<Context>, Failure> {
    let mut device = cuda_tensor_runtime_config(a.device, id(format!("device.cuda.{}", a.device)))
        .map_err(|e| error("descriptor", e))?;
    cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device)
        .map_err(|e| error("NVFP4 capability", e))?;
    cuda_tensor_runtime_enable_sage_vsa_fp4_regions(&mut device)
        .map_err(|e| error("Sage capability", e))?;
    cuda_tensor_runtime_enable_sage_vsa_fp4_quantize(&mut device)
        .map_err(|e| error("Sage quant capability", e))?;
    let registered = a
        .staging_bytes
        .checked_mul(2)
        .ok_or_else(|| error("staging", "registration capacity overflow"))?;
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager_with_options(
            CudaTensorRuntimeConfig {
                device,
                usable_memory_bytes: a.device_bytes,
                bootstrap_memory_bytes: a.bootstrap_bytes,
                stream_memory_bytes: a.stream_bytes,
                maximum_streams: 1,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: a.maximum_plan_nodes,
                opaque_memory_profile: "fast-h3-dit-backbone-v1".into(),
            },
            CudaTensorRuntimeOptions {
                pinned_weight_upload: Some(CudaPinnedWeightUploadConfig {
                    minimum_allocation_bytes: 0,
                    maximum_allocation_bytes: NonZeroUsize::new(a.staging_bytes)
                        .ok_or_else(|| error("staging", "zero capacity"))?,
                    maximum_registered_bytes: NonZeroUsize::new(registered)
                        .ok_or_else(|| error("staging", "zero capacity"))?,
                }),
                ..Default::default()
            },
        )
        .map_err(|e| error("runtime", e))?,
    );
    let selected = [
        CudaTensorOperation::BatchedMatmulF32,
        CudaTensorOperation::ScaleRank4F32,
        CudaTensorOperation::RowSoftmaxRank4F32,
        CudaTensorOperation::RowCenterBf16,
        CudaTensorOperation::HeadFp8RoundtripBf16,
        CudaTensorOperation::RowIndexGather,
        CudaTensorOperation::MergeLeadingBatchView,
        CudaTensorOperation::SplitLeadingBatchView,
        CudaTensorOperation::BatchedRowConcat,
        CudaTensorOperation::PackHeadAxisBf16,
        CudaTensorOperation::PackHeadAxisF32ToBf16,
        CudaTensorOperation::AddcmulFma,
        CudaTensorOperation::RowSelectedRmsModulation,
        CudaTensorOperation::RowSelectedResidualRmsModulation,
        CudaTensorOperation::RowSelectedAddcmulFmaBf16,
        CudaTensorOperation::HeadRmsPartialRopeSingleRound,
        CudaTensorOperation::MaskedTileMean,
        CudaTensorOperation::PackedSwiGlu,
        CudaTensorOperation::AffineLinearF32,
        CudaTensorOperation::SiluF32,
        CudaTensorOperation::CastF32ToBf16,
        CudaTensorOperation::AffineLinear,
        CudaTensorOperation::HeadRmsNorm,
        CudaTensorOperation::SegmentedAttention,
        CudaTensorOperation::RowGather,
        CudaTensorOperation::SiluMul,
        CudaTensorOperation::PlainResidualAdd,
        CudaTensorOperation::RowSelectedModulationBf16,
        CudaTensorOperation::CastBf16ToF32,
    ];
    let cudnn = CudaCudnnConfig {
        library_path: a.cudnn_library.clone(),
        dependency_libraries: a.cudnn_dependencies.clone(),
        opaque_memory_bytes: a.cudnn_opaque_bytes,
        workspace_limit_bytes: a.workspace_bytes,
        opaque_memory_profile: "fast-h3-dit-cudnn-v1".into(),
    };
    let mut providers = cuda_vnext_tensor_providers_with_cudnn_cross_attention(
        &runtime,
        &cudnn,
        CudnnSdpaPlanPolicy {
            maximum_plans: a.maximum_layers,
            maximum_live_bindings: a.maximum_layers,
            opaque_memory_bytes: a.cudnn_plan_bytes,
            workspace_limit_bytes: a.workspace_bytes,
        },
        &selected,
    )
    .map_err(|e| Failure::retain("cuDNN providers", e, runtime.clone()))?;
    let sage = SageVsaFp4Config {
        opaque_memory_bytes: a.native_bytes,
        opaque_memory_profile: "fast-h3-dit-sage-v1".into(),
    };
    providers.push(
        cuda_vnext_dense_bf16_provider(&runtime)
            .map_err(|e| Failure::retain("dense provider", e, runtime.clone()))?,
    );
    providers.push(
        cuda_vnext_sparse_fp4_qkv_quantize_bf16_provider(&runtime, &sage)
            .map_err(|e| Failure::retain("Sage quant provider", e, runtime.clone()))?,
    );
    providers.push(
        cuda_vnext_sparse_fp4_attention_region_topk_provider(&runtime, &sage, regions)
            .map_err(|e| Failure::retain("Sage region provider", e, runtime.clone()))?,
    );
    providers.extend(
        cuda_vnext_nvfp4_bf16_providers(
            &runtime,
            &CutlassNvfp4Config {
                opaque_memory_bytes: a.native_bytes,
                opaque_memory_profile: "fast-h3-dit-nvfp4-v1".into(),
            },
        )
        .map_err(|e| Failure::retain("NVFP4 providers", e, runtime.clone()))?,
    );
    let contracts = (|| -> Result<Vec<Box<dyn OperationContract>>, VNextError> {
        Ok(vec![
            Box::new(batched_matmul_f32_contract()?),
            Box::new(scale_rank4_f32_contract()?),
            Box::new(row_softmax_rank4_f32_contract()?),
            Box::new(row_center_bf16_contract()?),
            Box::new(head_fp8_roundtrip_bf16_contract()?),
            Box::new(row_index_gather_or_zero_bf16_contract()?),
            Box::new(LeadingBatchBf16Kind::Merge.view_contract()?),
            Box::new(LeadingBatchBf16Kind::Split.view_contract()?),
            Box::new(batched_row_concat_bf16_contract()?),
            Box::new(HeadAxisPackKind::Bf16.contract()?),
            Box::new(HeadAxisPackKind::F32ToBf16.contract()?),
            Box::new(addcmul_fma_bf16_contract()?),
            Box::new(sparse_fp4_qkv_quantize_bf16_contract()?),
            Box::new(sparse_fp4_attention_region_topk_contract()?),
            Box::new(nvfp4_quantize_bf16_contract()?),
            Box::new(nvfp4_matmul_bf16_contract()?),
            Box::new(cross_attention_fused_bf16_contract()?),
            Box::new(structured_attention_fused_bf16_contract()?),
            Box::new(conv2d_bf16_contract()?),
            Box::new(row_selected_rms_modulation_bf16_contract()?),
            Box::new(row_selected_residual_rms_modulation_bf16_contract()?),
            Box::new(row_selected_addcmul_fma_bf16_contract()?),
            Box::new(head_rms_partial_rope_single_round_bf16_contract()?),
            Box::new(masked_tile_mean_bf16_f32_contract()?),
            Box::new(packed_swiglu_bf16_contract()?),
            Box::new(affine_linear_f32_contract()?),
            Box::new(silu_f32_contract()?),
            Box::new(cast_f32_to_bf16_contract()?),
            Box::new(affine_linear_bf16_contract()?),
            Box::new(dense_linear_bf16_contract()?),
            Box::new(head_rms_norm_bf16_contract()?),
            Box::new(segmented_attention_bf16_contract()?),
            Box::new(row_gather_bf16_contract()?),
            Box::new(silu_mul_bf16_contract()?),
            Box::new(plain_residual_add_bf16_contract()?),
            Box::new(row_selected_modulation_bf16_contract()?),
            Box::new(cast_bf16_to_f32_contract()?),
        ])
    })()
    .map_err(|e| Failure::retain("contracts", e, runtime.clone()))?;
    let registry = Arc::new(
        OperationRuntimeRegistry::new(contracts, providers)
            .map_err(|e| Failure::retain("registry", e, runtime.clone()))?,
    );
    let materializers =
        WeightMaterializerRegistry::identity_only().map_err(|e| error("materializers", e))?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.fast_h3.dit"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .map_err(|e| error("engine", e))?;
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .map_err(|e| error("catalog", e))?,
        )
        .map_err(|e| error("catalog", e))?;
    let policy = TensorComponentPolicy::new(
        a.device_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .map_err(|e| error("policy", e))?;
    Ok(Arc::new(Context {
        runtime,
        registry,
        catalog,
        policy,
        materializers,
    }))
}
