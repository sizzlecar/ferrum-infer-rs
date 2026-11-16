use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::*,
    vnext_runtime::{CudaCudnnConfig, CudaTensorRuntimeConfig, CudnnSdpaPlanPolicy},
};
pub(super) struct Bundle {
    pub runtime: Arc<CudaDeviceRuntime>,
    pub registry: Arc<OperationRuntimeRegistry<CudaDeviceRuntime>>,
    pub catalog: CapabilityCatalog,
    pub materializers: WeightMaterializerRegistry,
}
pub(super) fn build(c: &Configuration, layers: usize, nodes: usize) -> anyhow::Result<Bundle> {
    let mut device =
        cuda_tensor_runtime_config(c.device, id(&format!("device.cuda.{}", c.device)))?;
    cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device)?;
    let runtime = Arc::new(CudaDeviceRuntime::new_tensor_eager(
        CudaTensorRuntimeConfig {
            device,
            usable_memory_bytes: c.device_bytes,
            bootstrap_memory_bytes: c.bootstrap_bytes,
            stream_memory_bytes: c.stream_bytes,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: nodes,
            opaque_memory_profile: "fast_h3.full_video_decoder.v1".into(),
        },
    )?);
    let mut providers = cuda_vnext_nvfp4_bf16_providers(
        &runtime,
        &CutlassNvfp4Config {
            opaque_memory_bytes: c.nvfp4_opaque_bytes,
            opaque_memory_profile: "fast_h3.video_decoder.nvfp4.v1".into(),
        },
    )?;
    providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
    // This factory replaces selected segmented attention with the existing
    // cuDNN SDPA owner/cache. There is no materialized N×N score fallback.
    providers.extend(cuda_vnext_cnn_f32_providers_with_segmented_sdpa(
        &runtime,
        &CudaCudnnConfig {
            library_path: c.cudnn_library.clone(),
            dependency_libraries: c.cudnn_dependencies.clone(),
            opaque_memory_bytes: c.cudnn_opaque_bytes,
            workspace_limit_bytes: c.cudnn_workspace_bytes,
            opaque_memory_profile: "fast_h3.video_decoder.cudnn9.v1".into(),
        },
        CudnnSdpaPlanPolicy {
            maximum_plans: 2,
            maximum_live_bindings: layers
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("binding limit overflow"))?,
            opaque_memory_bytes: c.sdpa_cache_bytes,
            workspace_limit_bytes: c.cudnn_workspace_bytes,
        },
        &[
            CudaTensorOperation::HeadRmsNormF32,
            CudaTensorOperation::HeadRmsNorm,
            CudaTensorOperation::CastF32ToBf16,
            CudaTensorOperation::CastBf16ToF32,
            CudaTensorOperation::ChannelBiasAddBf16,
            CudaTensorOperation::PartialHalfSplitRope,
            CudaTensorOperation::SegmentedAttention,
            CudaTensorOperation::ChannelScaledResidualSeparateF32,
            CudaTensorOperation::RowGather,
            CudaTensorOperation::PackedSwiGlu,
            CudaTensorOperation::AffineLinear,
            CudaTensorOperation::BatchedSuffixAppendF32,
            CudaTensorOperation::LayerNormF32,
        ],
    )?);
    let registry = Arc::new(OperationRuntimeRegistry::new(
        vec![
            Box::new(head_rms_norm_f32_contract()?),
            Box::new(head_rms_norm_bf16_contract()?),
            Box::new(cast_f32_to_bf16_contract()?),
            Box::new(cast_bf16_to_f32_contract()?),
            Box::new(channel_bias_add_bf16_contract()?),
            Box::new(partial_half_split_rope_bf16_contract()?),
            Box::new(segmented_attention_bf16_contract()?),
            Box::new(channel_scaled_residual_separate_f32_contract()?),
            Box::new(row_gather_bf16_contract()?),
            Box::new(packed_swiglu_bf16_contract()?),
            Box::new(nvfp4_quantize_bf16_contract()?),
            Box::new(nvfp4_matmul_bf16_contract()?),
            Box::new(dense_linear_bf16_contract()?),
            Box::new(affine_linear_bf16_contract()?),
            Box::new(batched_suffix_append_f32_contract()?),
            Box::new(layer_norm_f32_contract()?),
            Box::new(conv2d_f32_contract()?),
        ],
        providers,
    )?);
    let materializers = WeightMaterializerRegistry::identity_only()?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.fast_h3.video_decoder"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )?;
    let catalog =
        materializers.augment_catalog(registry.capability_catalog(d.clone(), vec![engine])?)?;
    Ok(Bundle {
        runtime,
        registry,
        catalog,
        materializers,
    })
}
