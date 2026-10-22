//! CUDA providers for backend-neutral vNext operation contracts.

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_runtime::{
    causal_paged_attention_contract, constant_scale_contract, dense_geglu_tanh_contract,
    dense_linear_contract, dense_swiglu_contract, gated_delta_recurrent_attention_contract,
    gpt_oss_causal_paged_attention_contract, hybrid_vnorm_causal_paged_attention_contract,
    last_token_dense_linear_contract, last_token_masked_argmax_contract,
    last_token_masked_argmax_f32_contract, logit_softcap_contract, residual_add_contract,
    residual_add_f32_f16_contract, rms_norm_contract, rms_norm_f32_contract,
    rms_norm_f32_to_f16_contract, token_embedding_contract, AttributeId,
    BatchedOperationInvocation, CapabilityCatalog, CapabilityId, CheckpointBoundaryConstraint,
    CheckpointCompletedInputCapture, CheckpointInputDependency, CheckpointPartitionNumerics,
    ContractVersion, DeviceBatchingForm, DeviceId, DeviceReusableExecutionTopologyFingerprint,
    DeviceRuntime, DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageRequirement,
    DynamicStorageView, ElementType, EncodedDeviceOperation, EngineProviderDescriptor,
    OperationContract, OperationFailure, OperationInvocation, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, OperationRuntimeRegistry, PreparedModelFamily, ProfilePhase,
    ProviderCheckpointCapability, ProviderCheckpointContract, ProviderId,
    ProviderStorageBindingRequirement, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, QuantizationFormatId,
    ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, VNextError, WeightFormatId,
    WeightMaterializerId, WeightMaterializerRegistry, WeightMaterializerSelection,
    CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID, CONSTANT_SCALE_F16_CAPABILITY_ID,
    DENSE_GEGLU_TANH_F16_CAPABILITY_ID, DENSE_LINEAR_F16_CAPABILITY_ID,
    DENSE_SWIGLU_F16_CAPABILITY_ID, DEVICE_NATIVE_ADAPTIVE_ATTENTION_CAPABILITY_ID,
    DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID, DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
    GPT_OSS_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
    HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID, IDENTITY_WEIGHT_MATERIALIZER_ID,
    LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID, LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
    LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID, LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID,
    LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID, LOGIT_SOFTCAP_F16_CAPABILITY_ID,
    RESIDUAL_ADD_F16_CAPABILITY_ID, RESIDUAL_ADD_F32_F16_CAPABILITY_ID, RMS_NORM_F16_CAPABILITY_ID,
    RMS_NORM_F32_CAPABILITY_ID, RMS_NORM_F32_TO_F16_CAPABILITY_ID,
    TOKEN_EMBEDDING_F16_CAPABILITY_ID, TOKEN_EMBEDDING_OPERATION_ID,
};
#[cfg(feature = "vllm-moe-marlin")]
use ferrum_runtime::{
    gpt_oss_routed_clamped_swiglu_moe_contract, routed_shared_swiglu_moe_contract,
    routed_swiglu_moe_contract, GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID,
    ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID, ROUTED_SWIGLU_MOE_F16_CAPABILITY_ID,
};
use ferrum_types::{
    AttentionExecutionPolicy, NativeOperatorBackend, NativeOperatorProviderCatalog,
    ProgramBindingUploadStrategy,
};
use sha2::{Digest, Sha256};

use super::vnext_replay::CudaCommandReplayKeyBuilder;
use super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeConfig, CudaDeviceRuntimeError,
};

#[cfg(test)]
mod last_token_linear_tests;
mod native_blocks;
mod native_io;
mod q6_head;
pub use q6_head::CudaQ6MmqF32LastTokenProvider;
mod selection;
mod tensor_dense_bf16;
mod tensor_silu;
mod transformer;
use ferrum_runtime::{
    causal_paged_attention_f32_master_contract, causal_paged_attention_f32_master_int8_kv_contract,
    causal_paged_attention_int8_kv_contract, gated_delta_recurrent_attention_f32_master_contract,
    last_token_dense_linear_f32_contract, token_embedding_f32_master_contract,
    CAUSAL_PAGED_ATTENTION_F32_MASTER_CAPABILITY_ID,
    CAUSAL_PAGED_ATTENTION_F32_MASTER_INT8_KV_CAPABILITY_ID,
    CAUSAL_PAGED_ATTENTION_INT8_KV_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID,
    LAST_TOKEN_DENSE_LINEAR_F32_CAPABILITY_ID, TOKEN_EMBEDDING_F32_MASTER_CAPABILITY_ID,
};
use native_io::TokenPrecision;
use selection::{
    argmax_dispatches, ArgmaxArguments, ArgmaxBatchArguments, ArgmaxFunctions, ArgmaxPrecision,
    ARGMAX_BATCH_ROWS,
};
pub use tensor_dense_bf16::{
    NUMERICAL_PROFILE_ID as CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID,
    NUMERICAL_PROFILE_IDENTITY as CUDA_DENSE_BF16_NUMERICAL_PROFILE_IDENTITY,
};
pub use tensor_silu::{
    CudaPhiloxDeviceGeometry, CudaPhiloxLaunchGeometry, CudaTensorOperation,
    ATTENTION_NUMERICAL_IDENTITY as CUDA_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_IDENTITY,
    ATTENTION_NUMERICAL_PROFILE_ID as CUDA_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_ID,
    BATCHED_ROW_CONCAT_PROFILE_ID as CUDA_BATCHED_ROW_CONCAT_BF16_NUMERICAL_PROFILE_ID,
    BUNDLE_NUMERICAL_IDENTITY as CUDA_POINTWISE_BF16_NUMERICAL_PROFILE_IDENTITY,
    CHANNEL_L2_IMAGE_PROFILE_ID as CUDA_CHANNEL_L2_NORM_IMAGE_BF16_NUMERICAL_PROFILE_ID,
    CHANNEL_L2_PROFILE_ID as CUDA_CHANNEL_L2_NORM_BF16_NUMERICAL_PROFILE_ID,
    CONV2D_PROFILE_ID as CUDA_CONV2D_BF16_NUMERICAL_PROFILE_ID,
    FUSED_ATTENTION_NUMERICAL_IDENTITY as CUDA_STRUCTURED_ATTENTION_FUSED_BF16_NUMERICAL_PROFILE_IDENTITY,
    FUSED_ATTENTION_NUMERICAL_PROFILE_ID as CUDA_STRUCTURED_ATTENTION_FUSED_BF16_NUMERICAL_PROFILE_ID,
    GATED_RESIDUAL_NUMERICAL_PROFILE_ID as CUDA_GATED_RESIDUAL_BF16_NUMERICAL_PROFILE_ID,
    GELU_TANH_PROFILE_ID as CUDA_GELU_TANH_BF16_NUMERICAL_PROFILE_ID,
    GQA_NUMERICAL_IDENTITY as CUDA_STRUCTURED_GQA_ATTENTION_BF16_NUMERICAL_PROFILE_IDENTITY,
    GQA_NUMERICAL_PROFILE_ID as CUDA_STRUCTURED_GQA_ATTENTION_BF16_NUMERICAL_PROFILE_ID,
    HALF_SPLIT_ROPE_NUMERICAL_IDENTITY as CUDA_HALF_SPLIT_ROPE_BF16_NUMERICAL_PROFILE_IDENTITY,
    HALF_SPLIT_ROPE_NUMERICAL_PROFILE_ID as CUDA_HALF_SPLIT_ROPE_BF16_NUMERICAL_PROFILE_ID,
    HEAD_RMS_NORM_PROFILE_ID as CUDA_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_ID,
    LAYER_NORM_PROFILE_ID as CUDA_LAYER_NORM_BF16_NUMERICAL_PROFILE_ID,
    MODULATION_NUMERICAL_PROFILE_ID as CUDA_MODULATION_SCALE_BF16_NUMERICAL_PROFILE_ID,
    NORM_NUMERICAL_IDENTITY as CUDA_NORMALIZATION_BF16_NUMERICAL_PROFILE_IDENTITY,
    NUMERICAL_PROFILE_ID as CUDA_SILU_BF16_NUMERICAL_PROFILE_ID,
    NUMERICAL_PROFILE_IDENTITY as CUDA_SILU_BF16_NUMERICAL_PROFILE_IDENTITY,
    PHILOX_BUILD_IDENTITY_SHA256 as CUDA_PHILOX_BUILD_IDENTITY_SHA256,
    PHILOX_NORMAL_IDENTITY as CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_IDENTITY,
    PHILOX_NORMAL_PROFILE_ID as CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID,
    PLAIN_RESIDUAL_ADD_NUMERICAL_PROFILE_ID as CUDA_PLAIN_RESIDUAL_ADD_BF16_NUMERICAL_PROFILE_ID,
    RESIDUAL_ADD_NCHW_PROFILE_ID as CUDA_PLAIN_RESIDUAL_ADD_NCHW_BF16_NUMERICAL_PROFILE_ID,
    ROPE_NUMERICAL_IDENTITY as CUDA_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_IDENTITY,
    ROPE_NUMERICAL_PROFILE_ID as CUDA_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID,
    ROW_GATHER_PROFILE_ID as CUDA_ROW_GATHER_BF16_NUMERICAL_PROFILE_ID,
    SILU_MUL_NUMERICAL_PROFILE_ID as CUDA_SILU_MUL_BF16_NUMERICAL_PROFILE_ID,
    SILU_NCHW_PROFILE_ID as CUDA_SILU_NCHW_BF16_NUMERICAL_PROFILE_ID,
    SPATIAL_ATTENTION_PROFILE_ID as CUDA_SPATIAL_ATTENTION_BF16_NUMERICAL_PROFILE_ID,
    SPATIAL_GATHER_PROFILE_ID as CUDA_SPATIAL_UPSAMPLE_GATHER_BF16_NUMERICAL_PROFILE_ID,
    TANH_NUMERICAL_PROFILE_ID as CUDA_TANH_BF16_NUMERICAL_PROFILE_ID,
    TEXT_PROJECTION_NUMERICAL_IDENTITY as CUDA_TEXT_PROJECTION_BF16_NUMERICAL_PROFILE_IDENTITY,
    TRANSPORT_NUMERICAL_IDENTITY as CUDA_TRANSPORT_BF16_NUMERICAL_PROFILE_IDENTITY,
    ZERO_CENTERED_RMS_NORM_PROFILE_ID as CUDA_ZERO_CENTERED_RMS_NORM_BF16_NUMERICAL_PROFILE_ID,
};

const TOKEN_EMBEDDING_PROVIDER_ID: &str = "provider.cuda.token_embedding.f16";
const TOKEN_EMBEDDING_ESTIMATOR_ID: &str = "resource-estimator.cuda.token_embedding.f16";
const LAST_TOKEN_DENSE_LINEAR_PROVIDER_ID: &str =
    "provider.cuda.last_token_dense_linear.f16.cublas";
const LAST_TOKEN_DENSE_LINEAR_ESTIMATOR_ID: &str =
    "resource-estimator.cuda.last_token_dense_linear.f16.cublas";
const LAST_TOKEN_MASKED_ARGMAX_PROVIDER_ID: &str = "provider.cuda.last_token_masked_argmax.f16";
const LAST_TOKEN_MASKED_ARGMAX_ESTIMATOR_ID: &str =
    "resource-estimator.cuda.last_token_masked_argmax.f16";
const CUDA_ENGINE_PROVIDER_ID: &str = "provider.engine.cuda.vnext";
const DENSE_SAFETENSORS_FORMAT_ID: &str = "weight-format.safetensors.dense";
const BLOCK_FP8_SAFETENSORS_FORMAT_ID: &str =
    "weight-format.safetensors.fp8-e4m3-block-grid-inverse-scale";
const BLOCK_FP8_SOURCE_QUANTIZATION_FORMAT_ID: &str =
    "quantization.safetensors.fp8-e4m3-block-grid-inverse-scale";
const EMBEDDING_FUNCTION_NAME: &str = "vnext_embedding_lookup_f16";
const MASKED_ARGMAX_PRESERVING_LOGITS_FUNCTION_NAME: &str =
    "last_token_masked_argmax_preserving_logits_f16";
const VALUE_ALIGNMENT_BYTES: u64 = 16;
const THREADS_PER_BLOCK: u32 = 256;
const MAXIMUM_TOKENS_PER_LAUNCH: u64 = u16::MAX as u64;
pub(super) const VNEXT_KV_PAGE_BYTES: u64 = 64 * 1024;

pub(crate) fn encode_segment_bindings(
    patch: ferrum_runtime::PreparedSegmentBindingPatch<'_, CudaDeviceBuffer>,
) -> Result<
    Option<Vec<ferrum_runtime::EncodedSegmentBindingNode<CudaDeviceCommand>>>,
    CudaDeviceRuntimeError,
> {
    transformer::segment_bindings::encode(patch)
}

pub(crate) fn compile_segment_binding_program(
    declarations: &[(u32, &ferrum_runtime::SegmentBindingDeclaration)],
) -> Result<Option<Arc<dyn std::any::Any + Send + Sync>>, CudaDeviceRuntimeError> {
    transformer::segment_bindings::compile_program(declarations)
}

pub(crate) fn compile_resident_decode_parts(
    program: &(dyn std::any::Any + Send + Sync),
    wave: &ferrum_runtime::PreparedPhysicalSegmentWave<'_, CudaDeviceBuffer>,
) -> Result<Option<super::vnext_runtime::ResidentColdParts>, CudaDeviceRuntimeError> {
    transformer::segment_bindings::compile_resident_program(program, wave)
}

pub(crate) fn encode_compiled_segment_bindings(
    program: &(dyn std::any::Any + Send + Sync),
    frame: &mut ferrum_runtime::PreparedSegmentBindingFrame<'_, CudaDeviceBuffer>,
) -> Result<
    (
        Vec<ferrum_runtime::EncodedSegmentBindingNode<CudaDeviceCommand>>,
        super::vnext_runtime::CudaBindingProgramSink,
        Arc<super::vnext_runtime::CudaSegmentOwnerTable>,
    ),
    CudaDeviceRuntimeError,
> {
    transformer::segment_bindings::encode_program(program, frame)
}

pub(crate) fn encode_physical_segment_bindings(
    program: &(dyn std::any::Any + Send + Sync),
    wave: &mut ferrum_runtime::PreparedPhysicalSegmentWave<'_, CudaDeviceBuffer>,
) -> Result<
    (
        Vec<ferrum_runtime::EncodedSegmentBindingNode<CudaDeviceCommand>>,
        super::vnext_runtime::CudaBindingProgramSink,
        Arc<super::vnext_runtime::CudaSegmentOwnerTable>,
    ),
    CudaDeviceRuntimeError,
> {
    transformer::segment_bindings::encode_physical_program(program, wave)
}

/// Typed CUDA runtime input for the currently installed vNext provider bundle.
pub fn cuda_vnext_runtime_config(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
) -> Result<CudaDeviceRuntimeConfig, VNextError> {
    let q6_head_fingerprint = q6_head::fingerprint();
    let fingerprint_parts: Vec<&[u8]> = vec![
        include_str!("vnext_runtime.rs").as_bytes(),
        q6_head_fingerprint.as_bytes(),
        include_str!("vnext_runtime/binding_transfers.rs").as_bytes(),
        include_str!("vnext_runtime/binding_scatter.rs").as_bytes(),
        include_str!("vnext_runtime/segment_oracle.rs").as_bytes(),
        include_str!("vnext_runtime/segment_oracle/packet.rs").as_bytes(),
        include_str!("vnext_runtime/segment_owners.rs").as_bytes(),
        include_str!("vnext_runtime/sequence_physical.rs").as_bytes(),
        include_str!("vnext_runtime/terminal_readback.rs").as_bytes(),
        include_str!("vnext_runtime/resident_program.rs").as_bytes(),
        include_str!("vnext_ops/transformer/segment_bindings/resident.rs").as_bytes(),
        include_str!("vnext_ops/transformer/segment_bindings/physical.rs").as_bytes(),
        include_str!("../program_binding_rows.rs").as_bytes(),
        include_str!("../program_binding_upload.rs").as_bytes(),
        include_str!("vnext_replay.rs").as_bytes(),
        include_str!("../reusable_execution.rs").as_bytes(),
        include_str!("../reusable_execution/warmup.rs").as_bytes(),
        include_str!("vnext_ops.rs").as_bytes(),
        include_str!("vnext_ops/selection.rs").as_bytes(),
        include_str!("vnext_ops/transformer.rs").as_bytes(),
        include_bytes!("vnext_ops/transformer/segment_bindings.rs"),
        include_bytes!("vnext_ops/transformer/attention/segment_bindings.rs"),
        include_bytes!("vnext_ops/transformer/causal_attention/segment_bindings.rs"),
        include_str!("vnext_ops/transformer/precision.rs").as_bytes(),
        include_str!("vnext_ops/transformer/native_linear.rs").as_bytes(),
        include_str!("vnext_ops/transformer/native_matrix.rs").as_bytes(),
        include_str!("vnext_ops/transformer/native_swiglu.rs").as_bytes(),
        include_bytes!("vnext_ops/transformer/gguf_f16_projection.rs"),
        include_bytes!("vnext_ops/transformer/rn_fragment_swiglu.rs"),
        include_bytes!("vnext_ops/transformer/q8act_swiglu.rs"),
        include_bytes!("vnext_ops/transformer/q8act_attention.rs"),
        include_bytes!("vnext_ops/native_blocks/q8act.rs"),
        include_bytes!("vnext_ops/transformer/rn_fragment_swiglu/plan.rs"),
        include_bytes!("vnext_ops/transformer/rn_fragment_swiglu/weights.rs"),
        include_bytes!("vnext_ops/transformer/rn_fragment_swiglu/execution.rs"),
        include_str!("vnext_ops/native_blocks.rs").as_bytes(),
        include_str!("vnext_ops/native_blocks/hadamard.rs").as_bytes(),
        include_str!("vnext_ops/native_io.rs").as_bytes(),
        include_str!("vnext_ops/native_blocks/weights.rs").as_bytes(),
        include_str!("vnext_ops/transformer/attention.rs").as_bytes(),
        include_str!("vnext_ops/transformer/attention/precision.rs").as_bytes(),
        include_str!("vnext_ops/transformer/attention/native_projection.rs").as_bytes(),
        include_str!("vnext_ops/transformer/causal_attention.rs").as_bytes(),
        include_str!("vnext_ops/transformer/causal_attention/precision.rs").as_bytes(),
        include_str!("vnext_ops/transformer/gpt_oss_attention.rs").as_bytes(),
        crate::ptx::EMBEDDING_LOOKUP.as_bytes(),
        crate::ptx::GATHER_COLUMNS.as_bytes(),
        crate::ptx::ARGMAX_ROWS.as_bytes(),
        crate::ptx::RMS_NORM.as_bytes(),
        crate::ptx::FUSED_SILU_MUL.as_bytes(),
        crate::ptx::RESIDUAL_ADD.as_bytes(),
        crate::ptx::SANDWICH_NORM.as_bytes(),
        crate::ptx::LINEAR_ATTENTION.as_bytes(),
        crate::ptx::GATED_DELTA_RULE.as_bytes(),
        crate::ptx::VNEXT_CAUSAL_ATTENTION.as_bytes(),
        crate::ptx::VNEXT_GGUF.as_bytes(),
        crate::ptx::GPT_OSS_ATTENTION.as_bytes(),
    ];
    let fingerprint_parts = {
        let mut fingerprint_parts = fingerprint_parts;
        fingerprint_parts.extend([
            include_bytes!("vnext_ops/transformer/upstream_swiglu.rs").as_slice(),
            include_bytes!("vnext_ops/transformer/upstream_swiglu/encode.rs").as_slice(),
            include_bytes!("vnext_ops/transformer/q8act_attention/upstream.rs").as_slice(),
            include_bytes!("vnext_ops/native_blocks/upstream_linear.rs").as_slice(),
            include_bytes!("vnext_ops/native_blocks/upstream_linear/native_plan.rs").as_slice(),
            include_bytes!("../../native_ops/upstream_q6_f16_linear.rs").as_slice(),
            include_bytes!("../../native_ops/upstream_q6_f16_linear/ffi.rs").as_slice(),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../ferrum-native-ops/src/upstream_q6_f16_linear.rs"
            ))
            .as_slice(),
            include_bytes!("vnext_ops/native_blocks/upstream_linear/preparation.rs").as_slice(),
            include_bytes!("vnext_ops/native_blocks/upstream_linear/weight_validation.rs")
                .as_slice(),
            include_bytes!("../../native_ops/upstream_linear.rs").as_slice(),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../ferrum-native-ops/src/upstream_extra_linear/prefill.rs"
            ))
            .as_slice(),
        ]);
        fingerprint_parts
    };
    #[cfg(feature = "vllm-moe-marlin")]
    let fingerprint_parts = {
        let mut fingerprint_parts = fingerprint_parts;
        fingerprint_parts.extend([
            include_str!("vnext_ops/transformer/moe.rs").as_bytes(),
            include_str!("vnext_ops/transformer/moe_launch.rs").as_bytes(),
            include_str!("vnext_ops/transformer/moe_routed.rs").as_bytes(),
            include_str!("vnext_ops/transformer/moe_weights.rs").as_bytes(),
            include_str!("vnext_ops/transformer/moe_workspace.rs").as_bytes(),
            include_str!("vnext_ops/transformer/gpt_oss_moe.rs").as_bytes(),
            crate::ptx::MOE_ROUTER.as_bytes(),
            crate::ptx::MOE_ALIGN_BLOCK_SIZE_PAIR_IDS.as_bytes(),
            crate::ptx::MOE_COMBINE.as_bytes(),
            crate::ptx::GPT_OSS_MOE.as_bytes(),
        ]);
        fingerprint_parts
    };
    #[cfg(feature = "vllm-marlin")]
    let fingerprint_parts = {
        let mut fingerprint_parts = fingerprint_parts;
        fingerprint_parts.extend([
            include_str!("../../marlin_fp8_materializer.rs").as_bytes(),
            include_str!("../../mxfp4_marlin_materializer.rs").as_bytes(),
            crate::ptx::MXFP4_MARLIN_PREPARE.as_bytes(),
        ]);
        fingerprint_parts
    };
    let capabilities = cuda_vnext_capabilities()?;
    let attention_execution_policy = requested_attention_policy
        .resolve(capabilities.iter().any(|capability| {
            capability.as_str() == DEVICE_NATIVE_ADAPTIVE_ATTENTION_CAPABILITY_ID
        }))
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    Ok(CudaDeviceRuntimeConfig {
        ordinal,
        device_id,
        attention_execution_policy,
        runtime_implementation_fingerprint: implementation_fingerprint(&fingerprint_parts),
        capabilities,
        dynamic_storage_profiles: BTreeSet::from([
            DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )?,
            DynamicStorageProfile::new(
                DynamicStorageAllocator::FixedBlockArena {
                    block_bytes: VNEXT_KV_PAGE_BYTES,
                },
                DynamicStorageView::PagedRegions {
                    block_bytes: VNEXT_KV_PAGE_BYTES,
                },
            )?,
        ]),
    })
}

/// Minimal capability/configuration for the bounded, eager tensor constructor.
/// This does not construct a device or choose opaque memory allowances; pass
/// it as `CudaTensorRuntimeConfig::device` to `new_tensor_eager`.
pub fn cuda_tensor_runtime_config(
    ordinal: usize,
    device_id: DeviceId,
) -> Result<CudaDeviceRuntimeConfig, VNextError> {
    let mut config =
        cuda_vnext_runtime_config(ordinal, device_id, AttentionExecutionPolicy::Portable)?;
    let numerical = tensor_silu::numerical_fingerprint();
    let dense_numerical = tensor_dense_bf16::source_fingerprint();
    config.runtime_implementation_fingerprint = implementation_fingerprint(&[
        config.runtime_implementation_fingerprint.as_bytes(),
        numerical.as_bytes(),
        dense_numerical.as_bytes(),
        include_bytes!("vnext_runtime/tensor_io.rs"),
        include_bytes!("vnext_runtime/tensor_eager.rs"),
        include_bytes!("vnext_runtime/cudnn.rs"),
        include_bytes!("vnext_runtime/cudnn/ffi.rs"),
        include_bytes!("vnext_runtime/cudnn/sdpa.rs"),
        include_bytes!("vnext_runtime/cudnn/sdpa/native.rs"),
    ]);
    config.capabilities = BTreeSet::from([
        CapabilityId::new(ferrum_runtime::SILU_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::PHILOX_NORMAL_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::SILU_MUL_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::TANH_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::MODULATION_SCALE_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::GATED_RESIDUAL_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::DENSE_LINEAR_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::LAYER_NORM_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::HEAD_RMS_NORM_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::ADJACENT_ROPE_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::STRUCTURED_ATTENTION_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::STRUCTURED_ATTENTION_FUSED_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::GELU_TANH_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::ROW_GATHER_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::HALF_SPLIT_ROPE_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::CONV2D_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::CHANNEL_L2_NORM_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::SPATIAL_ATTENTION_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::SILU_NCHW_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID)?,
    ]);
    config.dynamic_storage_profiles = BTreeSet::from([DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )?]);
    Ok(config)
}

/// The fixed-tensor SiLU provider, without token kernels/materializers or
/// graph registration. Build one registry from this and `silu_bf16_contract`.
pub fn cuda_vnext_silu_bf16_provider(
    runtime: &CudaDeviceRuntime,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    let mut providers = cuda_vnext_tensor_providers(runtime, &[CudaTensorOperation::Silu])?;
    providers
        .pop()
        .ok_or_else(|| CudaDeviceRuntimeError::contract("SiLU selection produced no provider"))
}

/// Standalone explicit-state RNG provider. Device SM/thread limits become part
/// of this provider's fingerprint. It acquires the same single module permit
/// as other tensor factories. To share that module with other stages, select
/// `CudaTensorOperation::PhiloxNormal` in the common tensor provider factory
/// instead of combining standalone factories in one live runtime.
/// Fresh offset-zero output has NOT been qualified against deployed Torch.
pub fn cuda_vnext_philox_normal_bf16_provider(
    runtime: &CudaDeviceRuntime,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    let mut providers = cuda_vnext_tensor_providers(runtime, &[CudaTensorOperation::PhiloxNormal])?;
    providers
        .pop()
        .ok_or_else(|| CudaDeviceRuntimeError::contract("Philox selection produced no provider"))
}

/// Assemble an explicitly selected subset from one shared native module and
/// capacity permit. Empty/duplicate selections and missing runtime capabilities
/// are rejected before native loading. Selection order is preserved. Register
/// each selected operation's standard contract; Dense is supplied separately.
/// Selecting `PhiloxNormal` queries launch geometry from this existing runtime
/// and shares the same module owner with all selected component operations.
/// New normalization/activation profiles require their own target evidence.
pub fn cuda_vnext_tensor_providers(
    runtime: &CudaDeviceRuntime,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    tensor_silu::selected_providers(runtime, selected)
}

/// Registers one BF16 Conv2D provider and the selected native tensor providers.
/// All native kernels share one module permit; the cuDNN handle/descriptors
/// acquire a separate explicit opaque capacity claim. An empty native selection
/// is valid for a convolution-only plan. No token or capture path is enabled.
/// The selection may include `PhiloxNormal`, so noise and later image stages
/// can retain one combined registry without a second native module load.
pub fn cuda_vnext_tensor_providers_with_cudnn(
    runtime: &CudaDeviceRuntime,
    config: &super::vnext_runtime::CudaCudnnConfig,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    tensor_silu::providers_with_cudnn(runtime, config, selected)
}

/// Opt-in segmented cuDNN SDPA alongside the selected native operations and
/// Conv2D, sharing one module permit and one separately charged cuDNN owner.
/// The explicit finite cache quota is additional to the cuDNN owner's opaque
/// allowance. Register `structured_attention_fused_bf16_contract` as well as
/// the selected contracts; legacy attention remains available independently.
pub fn cuda_vnext_tensor_providers_with_cudnn_sdpa(
    runtime: &CudaDeviceRuntime,
    config: &super::vnext_runtime::CudaCudnnConfig,
    policy: super::vnext_runtime::CudnnSdpaPlanPolicy,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    tensor_silu::providers_with_cudnn_sdpa(runtime, config, policy, selected)
}

/// Builds the five independent BF16 pointwise providers from one native
/// module load and one shared capacity permit. Register their corresponding
/// standard contracts alongside them. Do not combine this bundle with the
/// standalone SiLU factory in the same runtime: only one bundle may be live.
/// An encoded command retains the bundle after its provider is dropped.
pub fn cuda_vnext_tensor_pointwise_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    tensor_silu::providers(runtime)
}

/// Builds five BF16 pointwise and two normalization providers sharing one
/// native module/capacity permit. Register all seven corresponding contracts.
/// LayerNorm and HeadRmsNorm support power-of-two reduction widths <=4096;
/// their estimators reject unsupported geometry before execution. No global
/// normalization workspace is allocated. Do not combine this factory with
/// another module-bundle factory on the same runtime.
pub fn cuda_vnext_tensor_bf16_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    tensor_silu::component_providers(runtime)
}

/// Builds all eight fixed tensor providers (five pointwise, two normalization,
/// adjacent RoPE) sharing one module and capacity permit. Dense remains a
/// separate provider using the existing lane-owned cuBLAS handle. Register
/// the corresponding contracts or retain the required subset of providers.
/// The five- and seven-provider factory interfaces retain their former sets.
pub fn cuda_vnext_tensor_component_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    tensor_silu::all_component_providers(runtime)
}

/// Builds nine providers: the existing eight tensor component providers plus
/// bounded, nonpaged structured BF16 attention, sharing one native module and
/// capacity permit. Attention passes its mask by value (2312 bytes; the entire
/// pointer/scalar/mask parameter list is below 4 KiB). Its declared shared
/// storage is 4*D dynamic bytes plus 36 static bytes, with no global score or
/// workspace buffer. This is not a measurement of total native device usage.
/// Register the corresponding contracts or retain only the required providers.
/// Dense remains separate; the five/seven/eight-provider factory sets persist.
/// This baseline implementation makes no optimized-throughput claim.
pub fn cuda_vnext_tensor_attention_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    tensor_silu::attention_providers(runtime)
}

/// Source, precise build options, and compiled PTX identity. This identifies
/// an implementation; qualification still requires running on the target GPU.
pub fn cuda_silu_bf16_implementation_fingerprint() -> String {
    tensor_silu::numerical_fingerprint()
}

/// Fixed-tensor BF16 Dense with an explicit FP32 cuBLAS profile and zero
/// user workspace. A zero-workspace library failure is returned without an
/// implicit-workspace fallback. The existing token provider is unchanged.
pub fn cuda_vnext_dense_bf16_provider(
    runtime: &CudaDeviceRuntime,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    Ok(Box::new(
        tensor_dense_bf16::CudaTensorDenseBf16Provider::new(runtime)?,
    ))
}

/// Loaded cuBLAS version for validation provenance. NVIDIA supports this
/// query with a null handle; it creates no extra cuBLAS handle or workspace.
pub fn cuda_dense_bf16_cublas_version() -> Result<i32, CudaDeviceRuntimeError> {
    tensor_dense_bf16::cublas_version()
}

pub(crate) fn rn_fragment_mma_compiled() -> bool {
    transformer::compiled_mma_target(crate::ptx::VNEXT_GGUF)
}

pub(crate) fn q8act_g32_compiled() -> bool {
    native_blocks::q8act::compiled(crate::ptx::VNEXT_GGUF)
}

pub(crate) fn q8act_g32_profile_compiled(profile: ferrum_runtime::Q8ActSwiGluProfile) -> bool {
    native_blocks::q8act::compiled_for_profile(crate::ptx::VNEXT_GGUF, profile)
}

const UPSTREAM_MARKER_V2_EXPORTS: &[&str] = &[
    "ferrum_upstream_mmq_plan_v1",
    "ferrum_upstream_mmq_pack_v1",
    "ferrum_upstream_mmq_dot_v1",
    "ferrum_upstream_mmq_cast_v1",
    "ferrum_upstream_mmq_pack_v2",
    "ferrum_upstream_mmq_cast_v2",
    "ferrum_upstream_mmq_check_weights_v2",
    "ferrum_upstream_mmvq_plan_v1",
    "ferrum_upstream_mmvq_pack_v1",
    "ferrum_upstream_mmvq_dot_v1",
    "ferrum_upstream_mmvq_cast_v1",
    "ferrum_upstream_mmvq_pack_v2",
    "ferrum_upstream_mmvq_cast_v2",
    "ferrum_upstream_mmvq_check_weights_v2",
];

fn upstream_marker_v2_exports_present(exports: &[String]) -> bool {
    UPSTREAM_MARKER_V2_EXPORTS
        .iter()
        .all(|required| exports.iter().any(|export| export == required))
}

/// The build's locked and binary-validated inventory is the only source of
/// linked symbol availability. Exact live provider/G03 identity is additionally
/// checked by CudaVNextComposition::create before exposing execution.
fn upstream_profile_exports_present(exports: &[String], prefill: bool) -> bool {
    upstream_marker_v2_exports_present(exports)
        && (!prefill
            || exports
                .iter()
                .any(|export| export == "ferrum_upstream_mmq_prefill_plan_v1"))
}

/// The production pair-of-query-heads ABI must be present in the validated archive.
pub fn causal_decode_kv_reuse_compiled() -> bool {
    cfg!(feature = "cuda-causal-gqa2")
        && crate::native_ops::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator == "ferrum.cuda.vllm_paged_attention_v2"
                    && artifact.backend == NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && artifact.exports.iter().any(|name| {
                        name == "ferrum_vnext_vllm_paged_attention_v2_f16_h256_b16_gqa2_indexed_v1"
                    })
            })
}

pub(crate) fn q6_mmq_f32_compiled() -> bool {
    cfg!(feature = "cuda-upstream-q6-f32-linear")
        && crate::native_ops::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator
                    == ferrum_native_ops::upstream_q6_f32_linear::UPSTREAM_Q6_F32_LINEAR_OPERATOR
                    && artifact.backend == NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && ["plan", "pack", "dot", "check_weights", "publish"]
                        .iter()
                        .all(|stage| {
                            artifact
                                .exports
                                .iter()
                                .any(|name| name == &format!("ferrum_upstream_q6_f32_{stage}_v1"))
                        })
            })
}

pub(crate) fn upstream_q6_f16_profile_compiled() -> bool {
    native_blocks::upstream_linear::q6_f16_compiled()
}

pub(crate) fn upstream_marker_v2_compiled() -> bool {
    upstream_profile_compiled(false)
}
pub(crate) fn upstream_profile_compiled(prefill: bool) -> bool {
    cfg!(feature = "cuda-upstream-linear")
        && crate::native_ops::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator == "ferrum.cuda.upstream_linear"
                    && artifact.backend == NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && upstream_profile_exports_present(&artifact.exports, prefill)
            })
}

fn q4_shared_weight_decode_exports_present(exports: &[String]) -> bool {
    upstream_marker_v2_exports_present(exports)
        && exports
            .iter()
            .any(|name| name == "ferrum_upstream_mmvq_q4_shared_weight_decode_dot_v1")
}

/// Whether the opt-in Q4_K M8 dot and its original MarkerV2 contract are linked.
pub fn q4_shared_weight_decode_compiled() -> bool {
    cfg!(feature = "cuda-q4-shared-weight-decode")
        && upstream_marker_v2_compiled()
        && crate::native_ops::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator == "ferrum.cuda.upstream_linear"
                    && artifact.backend == NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && q4_shared_weight_decode_exports_present(&artifact.exports)
            })
}

fn q4_shared_weight_decode_m4_m8_exports_present(exports: &[String]) -> bool {
    q4_shared_weight_decode_exports_present(exports)
        && exports
            .iter()
            .any(|name| name == "ferrum_upstream_mmvq_q4_shared_weight_decode_m4_dot_v1")
}

/// Whether both opt-in Q4_K M4 and M8 dots share the linked MarkerV2 contract.
pub fn q4_shared_weight_decode_m4_m8_compiled() -> bool {
    cfg!(feature = "cuda-q4-shared-weight-decode")
        && upstream_marker_v2_compiled()
        && crate::native_ops::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator == "ferrum.cuda.upstream_linear"
                    && artifact.backend == NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && q4_shared_weight_decode_m4_m8_exports_present(&artifact.exports)
            })
}

fn q5_shared_weight_decode_exports_present(exports: &[String]) -> bool {
    upstream_marker_v2_exports_present(exports)
        && exports
            .iter()
            .any(|name| name == "ferrum_upstream_mmvq_q5_shared_weight_decode_dot_v1")
}

/// Whether the opt-in Q5_K M8 dot and its original MarkerV2 contract are linked.
pub fn q5_shared_weight_decode_compiled() -> bool {
    cfg!(feature = "cuda-q5-shared-weight-decode")
        && upstream_marker_v2_compiled()
        && crate::native_ops::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator == "ferrum.cuda.upstream_linear"
                    && artifact.backend == NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && q5_shared_weight_decode_exports_present(&artifact.exports)
            })
}

fn q5_shared_weight_decode_m4_m8_exports_present(exports: &[String]) -> bool {
    q5_shared_weight_decode_exports_present(exports)
        && exports
            .iter()
            .any(|name| name == "ferrum_upstream_mmvq_q5_shared_weight_decode_m4_dot_v1")
}

/// Whether both opt-in Q5_K M4 and M8 dots share the linked MarkerV2 contract.
pub fn q5_shared_weight_decode_m4_m8_compiled() -> bool {
    cfg!(feature = "cuda-q5-shared-weight-decode")
        && upstream_marker_v2_compiled()
        && crate::native_ops::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator == "ferrum.cuda.upstream_linear"
                    && artifact.backend == NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && q5_shared_weight_decode_m4_m8_exports_present(&artifact.exports)
            })
}

fn upstream_extra_exports_present(exports: &[String]) -> bool {
    ["mmq", "mmvq"].iter().all(|algorithm| {
        [
            "plan_v1",
            "pack_v2",
            "dot_v1",
            "cast_v2",
            "check_weights_v2",
        ]
        .iter()
        .all(|stage| {
            let required = format!("ferrum_upstream_extra_{algorithm}_{stage}");
            exports.iter().any(|export| export == &required)
        })
    })
}

pub(crate) fn upstream_extra_compiled() -> bool {
    upstream_extra_profile_compiled(false)
}

fn upstream_extra_profile_exports_present(exports: &[String], prefill: bool) -> bool {
    upstream_extra_exports_present(exports)
        && (!prefill
            || exports
                .iter()
                .any(|export| export == "ferrum_upstream_extra_mmq_prefill_plan_v2"))
}

pub(crate) fn upstream_extra_profile_compiled(prefill: bool) -> bool {
    cfg!(feature = "cuda-upstream-extra-linear")
        && crate::native_ops::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator == "ferrum.cuda.upstream_extra_linear"
                    && artifact.backend == NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && upstream_extra_profile_exports_present(&artifact.exports, prefill)
            })
}

#[cfg(test)]
mod upstream_marker_registration_tests {
    use super::{
        upstream_marker_v2_exports_present, upstream_profile_exports_present,
        UPSTREAM_MARKER_V2_EXPORTS,
    };

    #[cfg(feature = "cuda-q4-shared-weight-decode")]
    #[test]
    fn shared_weight_decode_requires_additive_dot_and_complete_original_marker_contract() {
        let baseline = UPSTREAM_MARKER_V2_EXPORTS
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        assert!(upstream_marker_v2_exports_present(&baseline));
        assert!(!super::q4_shared_weight_decode_exports_present(&baseline));
        let mut shared = baseline;
        shared.push("ferrum_upstream_mmvq_q4_shared_weight_decode_dot_v1".into());
        assert!(super::q4_shared_weight_decode_exports_present(&shared));
        assert!(!super::q5_shared_weight_decode_exports_present(&shared));
        for missing in &shared {
            let incomplete = shared
                .iter()
                .filter(|name| *name != missing)
                .cloned()
                .collect::<Vec<_>>();
            assert!(!super::q4_shared_weight_decode_exports_present(&incomplete));
        }
    }

    #[cfg(feature = "cuda-q5-shared-weight-decode")]
    #[test]
    fn shared_q5_weight_decode_requires_additive_dot_and_complete_original_marker_contract() {
        let baseline = UPSTREAM_MARKER_V2_EXPORTS
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        assert!(upstream_marker_v2_exports_present(&baseline));
        assert!(!super::q5_shared_weight_decode_exports_present(&baseline));
        let mut shared = baseline;
        shared.push("ferrum_upstream_mmvq_q5_shared_weight_decode_dot_v1".into());
        assert!(super::q5_shared_weight_decode_exports_present(&shared));
        assert!(!super::q4_shared_weight_decode_exports_present(&shared));
        for missing in &shared {
            let incomplete = shared
                .iter()
                .filter(|name| *name != missing)
                .cloned()
                .collect::<Vec<_>>();
            assert!(!super::q5_shared_weight_decode_exports_present(&incomplete));
        }
    }

    #[test]
    fn shared_m4_m8_requires_both_dots_and_complete_original_marker_contract() {
        for (old_check, combined_check, format) in [
            (
                super::q4_shared_weight_decode_exports_present as fn(&[String]) -> bool,
                super::q4_shared_weight_decode_m4_m8_exports_present as fn(&[String]) -> bool,
                "q4",
            ),
            (
                super::q5_shared_weight_decode_exports_present as fn(&[String]) -> bool,
                super::q5_shared_weight_decode_m4_m8_exports_present as fn(&[String]) -> bool,
                "q5",
            ),
        ] {
            let mut exports = UPSTREAM_MARKER_V2_EXPORTS
                .iter()
                .map(|name| (*name).to_owned())
                .collect::<Vec<_>>();
            assert!(!combined_check(&exports));
            exports.push(format!(
                "ferrum_upstream_mmvq_{format}_shared_weight_decode_dot_v1"
            ));
            assert!(old_check(&exports));
            assert!(!combined_check(&exports));
            exports.push(format!(
                "ferrum_upstream_mmvq_{format}_shared_weight_decode_m4_dot_v1"
            ));
            assert!(old_check(&exports));
            assert!(combined_check(&exports));
            for missing in &exports {
                let incomplete = exports
                    .iter()
                    .filter(|name| *name != missing)
                    .cloned()
                    .collect::<Vec<_>>();
                assert!(!combined_check(&incomplete), "accepted without {missing}");
            }
        }
    }

    #[test]
    fn extra_marker_capability_requires_its_own_complete_symbol_family() {
        let exports = ["mmq", "mmvq"]
            .into_iter()
            .flat_map(|algorithm| {
                [
                    "plan_v1",
                    "pack_v2",
                    "dot_v1",
                    "cast_v2",
                    "check_weights_v2",
                ]
                .map(move |stage| format!("ferrum_upstream_extra_{algorithm}_{stage}"))
            })
            .collect::<Vec<_>>();
        assert!(super::upstream_extra_exports_present(&exports));
        assert!(super::upstream_extra_profile_exports_present(
            &exports, false
        ));
        assert!(!super::upstream_extra_profile_exports_present(
            &exports, true
        ));
        let mut prefill = exports.clone();
        // The historical v1 symbol was inert and is not large-row eligibility.
        prefill.push("ferrum_upstream_extra_mmq_prefill_plan_v1".into());
        assert!(!super::upstream_extra_profile_exports_present(
            &prefill, true
        ));
        prefill.push("ferrum_upstream_extra_mmq_prefill_plan_v2".into());
        assert!(super::upstream_extra_profile_exports_present(
            &prefill, true
        ));
        for missing in &exports {
            let incomplete = exports
                .iter()
                .filter(|name| *name != missing)
                .cloned()
                .collect::<Vec<_>>();
            assert!(!super::upstream_extra_exports_present(&incomplete));
            let mut incomplete_prefill = incomplete;
            incomplete_prefill.push("ferrum_upstream_extra_mmq_prefill_plan_v2".into());
            assert!(!super::upstream_extra_profile_exports_present(
                &incomplete_prefill,
                true
            ));
        }
        let base = exports
            .iter()
            .map(|name| name.replace("upstream_extra_", "upstream_"))
            .collect::<Vec<_>>();
        assert!(!super::upstream_extra_exports_present(&base));
    }

    #[test]
    fn upstream_marker_capability_rejects_missing_v1_or_device_marker_exports() {
        let exports = UPSTREAM_MARKER_V2_EXPORTS
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        assert!(upstream_marker_v2_exports_present(&exports));
        assert!(upstream_profile_exports_present(&exports, false));
        assert!(
            !upstream_profile_exports_present(&exports, true),
            "old artifact cannot qualify prefill"
        );
        let mut expanded = exports.clone();
        expanded.push("ferrum_upstream_mmq_prefill_plan_v1".to_owned());
        assert!(upstream_profile_exports_present(&expanded, true));
        for missing in UPSTREAM_MARKER_V2_EXPORTS {
            let incomplete = exports
                .iter()
                .filter(|name| name.as_str() != *missing)
                .cloned()
                .collect::<Vec<_>>();
            assert!(!upstream_marker_v2_exports_present(&incomplete));
        }
        assert!(!upstream_marker_v2_exports_present(&[
            "ferrum_upstream_mmq_plan_v1".to_owned(),
            "ferrum_upstream_mmq_plan_v1".to_owned(),
        ]));
    }
}

pub fn cuda_vnext_capabilities() -> Result<BTreeSet<CapabilityId>, VNextError> {
    let capabilities = [
        TOKEN_EMBEDDING_F16_CAPABILITY_ID,
        TOKEN_EMBEDDING_F32_MASTER_CAPABILITY_ID,
        LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
        LAST_TOKEN_DENSE_LINEAR_F32_CAPABILITY_ID,
        LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID,
        LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID,
        RMS_NORM_F16_CAPABILITY_ID,
        RMS_NORM_F32_TO_F16_CAPABILITY_ID,
        RMS_NORM_F32_CAPABILITY_ID,
        DENSE_LINEAR_F16_CAPABILITY_ID,
        DENSE_SWIGLU_F16_CAPABILITY_ID,
        ferrum_runtime::DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_CAPABILITY_ID,
        crate::gguf_rn_fragment_materializer::GGUF_RN_FRAGMENT_CAPABILITY_ID,
        ferrum_runtime::GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID,
        ferrum_runtime::CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID,
        DENSE_GEGLU_TANH_F16_CAPABILITY_ID,
        CONSTANT_SCALE_F16_CAPABILITY_ID,
        LOGIT_SOFTCAP_F16_CAPABILITY_ID,
        RESIDUAL_ADD_F16_CAPABILITY_ID,
        RESIDUAL_ADD_F32_F16_CAPABILITY_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID,
        CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_CAPABILITY_ID,
        CAUSAL_PAGED_ATTENTION_INT8_KV_CAPABILITY_ID,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_INT8_KV_CAPABILITY_ID,
        HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
        GPT_OSS_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
        DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
        DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID,
    ]
    .into_iter()
    .map(CapabilityId::new)
    .collect::<Result<BTreeSet<_>, _>>()?;
    #[cfg(feature = "vllm-moe-marlin")]
    let capabilities = {
        let mut capabilities = capabilities;
        capabilities.insert(CapabilityId::new(
            ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID,
        )?);
        capabilities.insert(CapabilityId::new(ROUTED_SWIGLU_MOE_F16_CAPABILITY_ID)?);
        capabilities.insert(CapabilityId::new(
            GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID,
        )?);
        capabilities
    };
    #[cfg(feature = "vllm-marlin")]
    let capabilities = {
        let mut capabilities = capabilities;
        capabilities.insert(CapabilityId::new(
            crate::marlin_fp8_materializer::MARLIN_FP8_CAPABILITY_ID,
        )?);
        capabilities.insert(CapabilityId::new(transformer::GPTQ_MARLIN_CAPABILITY_ID)?);
        capabilities.insert(CapabilityId::new(
            transformer::COMPRESSED_TENSORS_MARLIN_CAPABILITY_ID,
        )?);
        capabilities.insert(CapabilityId::new(
            transformer::COMPRESSED_TENSORS_MARLIN_SYMMETRIC_CAPABILITY_ID,
        )?);
        capabilities
    };
    #[cfg(feature = "vllm-paged-attn-v2")]
    let capabilities = {
        let mut capabilities = capabilities;
        capabilities.insert(CapabilityId::new(
            DEVICE_NATIVE_ADAPTIVE_ATTENTION_CAPABILITY_ID,
        )?);
        capabilities
    };
    let mut capabilities = capabilities;
    if q6_mmq_f32_compiled() {
        capabilities.insert(CapabilityId::new(
            ferrum_runtime::LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID,
        )?);
    }
    for profile in [
        ferrum_runtime::Q8ActSwiGluProfile::Iq4Xs,
        ferrum_runtime::Q8ActSwiGluProfile::Q4KQ5KIq4Xs,
    ] {
        if q8act_g32_profile_compiled(profile) {
            capabilities.insert(CapabilityId::new(profile.capability_id())?);
        }
    }
    if q8act_g32_profile_compiled(ferrum_runtime::Q8ActSwiGluProfile::Q4KQ5KIq4Xs) {
        for profile in [
            ferrum_runtime::Q8ActAttentionProfile::GatedDelta,
            ferrum_runtime::Q8ActAttentionProfile::Causal,
        ] {
            capabilities.insert(CapabilityId::new(profile.capability_id())?);
        }
    }
    if upstream_marker_v2_compiled() {
        for profile in [
            ferrum_runtime::UpstreamMarkerV2Profile::SwiGlu,
            ferrum_runtime::UpstreamMarkerV2Profile::GatedDelta,
            ferrum_runtime::UpstreamMarkerV2Profile::Causal,
            ferrum_runtime::UpstreamMarkerV2Profile::SwiGluPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::SwiGluG32MmqPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::CausalPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::CausalG32MmqPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraPrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraLargePrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraAllRows,
            ferrum_runtime::UpstreamMarkerV2Profile::SwiGluQ6F16,
            ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
            ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaQ6F16,
            ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraLargePrefill,
            ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraAllRows,
            ferrum_runtime::UpstreamMarkerV2Profile::CausalQ6F16,
        ] {
            if upstream_profile_compiled(profile.prefill() || profile.hybrid())
                && (!profile.extra() || upstream_extra_profile_compiled(profile.extra_prefill()))
                && (!profile.q6_f16() || upstream_q6_f16_profile_compiled())
                && (!profile.hybrid()
                    || q8act_g32_profile_compiled(ferrum_runtime::Q8ActSwiGluProfile::Q4KQ5KIq4Xs))
            {
                capabilities.insert(CapabilityId::new(profile.capability_id())?);
            }
        }
    }
    if !rn_fragment_mma_compiled() {
        capabilities.retain(|capability| {
            capability.as_str()
                != ferrum_runtime::DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_CAPABILITY_ID
        });
    }
    Ok(capabilities)
}

pub fn cuda_weight_materializer_selection(
    family: &PreparedModelFamily,
) -> Result<WeightMaterializerSelection, VNextError> {
    if crate::gguf_rn_fragment_materializer::requests_gguf_rn_fragment_materialization(family) {
        let selection =
            crate::gguf_rn_fragment_materializer::gguf_rn_fragment_materializer_selection(family)?;
        let inventory = crate::gguf_rn_fragment_materializer::gguf_rn_fragment_inventory(family)?;
        tracing::info!(
            source_schema_fingerprint = %inventory.source_schema_fingerprint,
            execution_schema_fingerprint = %inventory.execution_schema_fingerprint,
            source_consumed_bytes = inventory.source_consumed_bytes,
            converted_f16_bytes = inventory.converted_f16_bytes,
            packed_fragment_bytes = inventory.packed_fragment_bytes,
            retained_consumed_bytes = inventory.retained_consumed_bytes,
            unique_consumed_execution_bytes = inventory.unique_consumed_execution_bytes,
            includes_placement_alignment = inventory.includes_placement_alignment,
            "Explicit GGUF RN-F16 dense and fragment inventory (not Peak VRAM)"
        );
        return Ok(selection);
    }
    let block_fp8_weight_format = WeightFormatId::new(BLOCK_FP8_SAFETENSORS_FORMAT_ID)?;
    let block_fp8_quantization =
        QuantizationFormatId::new(BLOCK_FP8_SOURCE_QUANTIZATION_FORMAT_ID)?;
    let source_schema = family.stored_weight_schema()?;
    let quantization_formats = source_schema.quantization_formats();
    let has_block_fp8_quantization = quantization_formats.contains(&block_fp8_quantization);
    let has_block_fp8_weight_format = source_schema.format_id == block_fp8_weight_format;
    let mxfp4_weight_format = WeightFormatId::new(
        crate::mxfp4_marlin_materializer::GPT_OSS_MXFP4_SOURCE_WEIGHT_FORMAT_ID,
    )?;
    let mxfp4_quantization = QuantizationFormatId::new(
        crate::mxfp4_marlin_materializer::GPT_OSS_MXFP4_SOURCE_QUANTIZATION_FORMAT_ID,
    )?;
    let has_mxfp4_quantization = quantization_formats.contains(&mxfp4_quantization);
    let has_mxfp4_weight_format = source_schema.format_id == mxfp4_weight_format;

    if has_block_fp8_weight_format != has_block_fp8_quantization
        || (has_block_fp8_weight_format
            && quantization_formats != BTreeSet::from([block_fp8_quantization]))
    {
        return Err(VNextError::InvalidExecutionPlan {
            reason: "CUDA block-FP8 source format and typed quantization schema disagree"
                .to_owned(),
        });
    }
    if has_mxfp4_weight_format != has_mxfp4_quantization
        || (has_mxfp4_weight_format && quantization_formats != BTreeSet::from([mxfp4_quantization]))
    {
        return Err(VNextError::InvalidExecutionPlan {
            reason: "CUDA GPT-OSS MXFP4 source format and typed quantization schema disagree"
                .to_owned(),
        });
    }
    if has_block_fp8_weight_format && has_mxfp4_weight_format {
        return Err(VNextError::InvalidExecutionPlan {
            reason: "CUDA source schema cannot require block-FP8 and GPT-OSS MXFP4 materializers together"
                .to_owned(),
        });
    }

    if !has_block_fp8_weight_format && !has_mxfp4_weight_format {
        return Ok(WeightMaterializerSelection::exact(
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID)?,
        ));
    }

    if has_mxfp4_weight_format {
        #[cfg(feature = "vllm-moe-marlin")]
        {
            return Ok(WeightMaterializerSelection::exact(
                WeightMaterializerId::new(
                    crate::mxfp4_marlin_materializer::GPT_OSS_MXFP4_TO_MARLIN_WEIGHT_MATERIALIZER_ID,
                )?,
            ));
        }
        #[cfg(not(feature = "vllm-moe-marlin"))]
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason:
                    "CUDA GPT-OSS MXFP4 source requires the compiled vllm-moe-marlin materializer"
                        .to_owned(),
            });
        }
    }

    #[cfg(feature = "vllm-marlin")]
    {
        Ok(WeightMaterializerSelection::exact(
            WeightMaterializerId::new(
                crate::marlin_fp8_materializer::BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID,
            )?,
        ))
    }
    #[cfg(not(feature = "vllm-marlin"))]
    {
        Err(VNextError::InvalidExecutionPlan {
            reason: "CUDA block-FP8 source requires the compiled vllm-marlin materializer"
                .to_owned(),
        })
    }
}

#[cfg(all(test, feature = "vllm-marlin"))]
mod block_fp8_exact_materializer_tests {
    use super::*;

    #[test]
    fn block_fp8_selection_requires_the_live_exact_materializer() {
        let materializer =
            crate::marlin_fp8_materializer::block_fp8_to_marlin_fp8_weight_materializer()
                .expect("live block-FP8 materializer");
        assert_eq!(
            materializer.descriptor().fidelity(),
            ferrum_runtime::WeightMaterializationFidelity::Exact
        );
        let selection = WeightMaterializerSelection::exact(
            WeightMaterializerId::new(
                crate::marlin_fp8_materializer::BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID,
            )
            .expect("block-FP8 materializer id"),
        );
        assert!(!selection.has_numeric_quality_artifact());
    }
}

fn cuda_operation_contracts(
    attention_policy: AttentionExecutionPolicy,
) -> Result<Vec<Box<dyn OperationContract>>, CudaDeviceRuntimeError> {
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(token_embedding_contract().map_err(contract_error)?),
        Box::new(token_embedding_f32_master_contract().map_err(contract_error)?),
        Box::new(last_token_dense_linear_contract().map_err(contract_error)?),
        Box::new(last_token_dense_linear_f32_contract().map_err(contract_error)?),
        Box::new(last_token_masked_argmax_contract().map_err(contract_error)?),
        Box::new(last_token_masked_argmax_f32_contract().map_err(contract_error)?),
        Box::new(rms_norm_contract().map_err(contract_error)?),
        Box::new(rms_norm_f32_to_f16_contract().map_err(contract_error)?),
        Box::new(rms_norm_f32_contract().map_err(contract_error)?),
        Box::new(dense_linear_contract().map_err(contract_error)?),
        Box::new(ferrum_runtime::gated_delta_recurrent_attention_f32_master_gguf_f16_projections_contract().map_err(contract_error)?),
        Box::new(ferrum_runtime::causal_paged_attention_f32_master_gguf_f16_projections_contract().map_err(contract_error)?),
        Box::new(dense_swiglu_contract().map_err(contract_error)?),
        Box::new(dense_geglu_tanh_contract().map_err(contract_error)?),
        Box::new(constant_scale_contract().map_err(contract_error)?),
        Box::new(logit_softcap_contract().map_err(contract_error)?),
        Box::new(residual_add_contract().map_err(contract_error)?),
        Box::new(residual_add_f32_f16_contract().map_err(contract_error)?),
        Box::new(gated_delta_recurrent_attention_contract().map_err(contract_error)?),
        Box::new(gated_delta_recurrent_attention_f32_master_contract().map_err(contract_error)?),
        Box::new(causal_paged_attention_contract().map_err(contract_error)?),
        Box::new(causal_paged_attention_f32_master_contract().map_err(contract_error)?),
        Box::new(hybrid_vnorm_causal_paged_attention_contract().map_err(contract_error)?),
        Box::new(gpt_oss_causal_paged_attention_contract().map_err(contract_error)?),
    ];
    // A catalog requires a real provider for every operation contract. Native
    // F16 compositions must not advertise the portable-only INT8 operations.
    let contracts = if attention_policy == AttentionExecutionPolicy::Portable {
        let mut contracts = contracts;
        contracts.push(Box::new(
            causal_paged_attention_int8_kv_contract().map_err(contract_error)?,
        ));
        contracts.push(Box::new(
            causal_paged_attention_f32_master_int8_kv_contract().map_err(contract_error)?,
        ));
        contracts
    } else {
        contracts
    };
    #[cfg(feature = "vllm-moe-marlin")]
    let contracts = {
        let mut contracts = contracts;
        contracts.push(Box::new(
            routed_shared_swiglu_moe_contract().map_err(contract_error)?,
        ));
        contracts.push(Box::new(
            routed_swiglu_moe_contract().map_err(contract_error)?,
        ));
        contracts.push(Box::new(
            gpt_oss_routed_clamped_swiglu_moe_contract().map_err(contract_error)?,
        ));
        contracts
    };
    Ok(contracts)
}

/// Build the exact composition root used for both planning and dispatch.
pub fn cuda_vnext_operation_registry(
    runtime: &CudaDeviceRuntime,
) -> Result<OperationRuntimeRegistry<CudaDeviceRuntime>, CudaDeviceRuntimeError> {
    let mut contracts = cuda_operation_contracts(runtime.attention_execution_policy())?;
    let providers: Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>> = vec![
        Box::new(CudaTokenEmbeddingProvider::new(runtime)?),
        Box::new(CudaTokenEmbeddingProvider::new_f32(runtime)?),
        Box::new(CudaLastTokenDenseLinearProvider::new(runtime)?),
        Box::new(CudaLastTokenDenseLinearProvider::new_f32(runtime)?),
        Box::new(CudaLastTokenMaskedArgmaxProvider::new(runtime)?),
        Box::new(CudaLastTokenMaskedArgmaxProvider::new_f32(runtime)?),
        Box::new(transformer::CudaRmsNormProvider::new(runtime)?),
        Box::new(transformer::CudaRmsNormProvider::new_f32_to_f16(runtime)?),
        Box::new(transformer::CudaRmsNormProvider::new_f32(runtime)?),
        Box::new(transformer::CudaDenseLinearProvider::new(runtime)?),
        Box::new(transformer::CudaDenseSwiGluProvider::new(runtime)?),
        Box::new(transformer::CudaGatedDeltaRecurrentAttentionProvider::new_f32_master_gguf_f16_projections(runtime)?),
        Box::new(transformer::CudaCausalPagedAttentionProvider::new_f32_master_gguf_f16_projections(runtime, runtime.attention_execution_policy())?),
        Box::new(transformer::CudaDenseGeGluTanhProvider::new(runtime)?),
        Box::new(transformer::CudaConstantScaleProvider::new(runtime)?),
        Box::new(transformer::CudaLogitSoftcapProvider::new(runtime)?),
        Box::new(transformer::CudaResidualAddProvider::new(runtime)?),
        Box::new(transformer::CudaResidualAddProvider::new_f32_f16(runtime)?),
        Box::new(transformer::CudaGatedDeltaRecurrentAttentionProvider::new(
            runtime,
        )?),
        Box::new(transformer::CudaGatedDeltaRecurrentAttentionProvider::new_f32_master(runtime)?),
        Box::new(transformer::CudaCausalPagedAttentionProvider::new(
            runtime,
            runtime.attention_execution_policy(),
        )?),
        Box::new(
            transformer::CudaCausalPagedAttentionProvider::new_f32_master(
                runtime,
                runtime.attention_execution_policy(),
            )?,
        ),
        Box::new(transformer::CudaCausalPagedAttentionProvider::new_gemma4(
            runtime,
            runtime.attention_execution_policy(),
            &hybrid_vnorm_causal_paged_attention_contract().map_err(contract_error)?,
            HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
        )?),
        Box::new(transformer::CudaGptOssCausalPagedAttentionProvider::new(
            runtime,
        )?),
    ];
    // A native-adaptive runtime has no INT8 native ABI. Composition selects a
    // portable runtime for an automatic INT8 request; explicit native requests
    // therefore cannot find an INT8 provider in this registry.
    let providers = if runtime.attention_execution_policy() == AttentionExecutionPolicy::Portable {
        let mut providers = providers;
        providers.push(Box::new(
            transformer::CudaCausalPagedAttentionProvider::new_int8_kv(
                runtime,
                AttentionExecutionPolicy::Portable,
            )?,
        ));
        providers.push(Box::new(
            transformer::CudaCausalPagedAttentionProvider::new_f32_master_int8_kv(
                runtime,
                AttentionExecutionPolicy::Portable,
            )?,
        ));
        providers
    } else {
        providers
    };
    #[cfg(feature = "vllm-marlin")]
    let providers = {
        let mut providers = providers;
        providers.push(Box::new(
            transformer::CudaMarlinFp8DenseLinearProvider::new(runtime)?,
        ));
        providers
    };
    #[cfg(feature = "vllm-moe-marlin")]
    let providers = {
        let mut providers = providers;
        providers.push(Box::new(
            transformer::CudaRoutedSharedSwiGluMoeProvider::new(runtime)?,
        ));
        providers.push(Box::new(
            transformer::CudaRoutedSharedSwiGluMoeProvider::new_marlin_fp8(runtime)?,
        ));
        providers.push(Box::new(transformer::CudaRoutedSwiGluMoeProvider::new(
            runtime,
        )?));
        providers.push(Box::new(
            transformer::CudaGptOssRoutedClampedSwiGluMoeProvider::new(runtime)?,
        ));
        providers
    };
    let mut providers = providers;
    if runtime
        .descriptor()
        .capabilities
        .iter()
        .any(|cap| cap.as_str() == ferrum_runtime::LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID)
    {
        contracts.push(Box::new(
            ferrum_runtime::last_token_dense_linear_q6_mmq_f32_contract()
                .map_err(contract_error)?,
        ));
        providers.push(Box::new(CudaQ6MmqF32LastTokenProvider::new(runtime)?));
    }
    for profile in [
        ferrum_runtime::UpstreamMarkerV2Profile::SwiGlu,
        ferrum_runtime::UpstreamMarkerV2Profile::GatedDelta,
        ferrum_runtime::UpstreamMarkerV2Profile::Causal,
        ferrum_runtime::UpstreamMarkerV2Profile::SwiGluPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::SwiGluG32MmqPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::CausalPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::CausalG32MmqPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraPrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraLargePrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraAllRows,
        ferrum_runtime::UpstreamMarkerV2Profile::SwiGluQ6F16,
        ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
        ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaQ6F16,
        ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraLargePrefill,
        ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraAllRows,
        ferrum_runtime::UpstreamMarkerV2Profile::CausalQ6F16,
    ] {
        if runtime
            .descriptor()
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == profile.capability_id())
        {
            contracts.push(Box::new(profile.contract().map_err(contract_error)?));
            match profile {
                ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaQ6F16 => {
                    providers.push(Box::new(transformer::CudaGatedDeltaRecurrentAttentionProvider::new_upstream_q6_f16(runtime)?))
                }
                ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraAllRows => {
                    providers.push(Box::new(
                        transformer::CudaGatedDeltaRecurrentAttentionProvider::new_upstream_extra_all_rows(runtime)?,
                    ))
                }
                ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill => {
                    providers.push(Box::new(
                        transformer::CudaGatedDeltaRecurrentAttentionProvider::new_upstream_extra_prefill(runtime)?,
                    ))
                }
                ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaExtraPrefill => {
                    providers.push(Box::new(
                        transformer::CudaGatedDeltaRecurrentAttentionProvider::new_upstream_extra(
                            runtime,
                        )?,
                    ))
                }
                ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill => {
                    providers.push(Box::new(
                        transformer::CudaGatedDeltaRecurrentAttentionProvider::new_g32_mmq_prefill(
                            runtime,
                        )?,
                    ))
                }
                ferrum_runtime::UpstreamMarkerV2Profile::SwiGlu
                | ferrum_runtime::UpstreamMarkerV2Profile::SwiGluPrefill
                | ferrum_runtime::UpstreamMarkerV2Profile::SwiGluG32MmqPrefill
                | ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraPrefill
                | ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraLargePrefill
                | ferrum_runtime::UpstreamMarkerV2Profile::SwiGluExtraAllRows
                | ferrum_runtime::UpstreamMarkerV2Profile::SwiGluQ6F16 => {
                    providers.push(Box::new(transformer::CudaUpstreamSwiGluProvider::new(
                        runtime, profile,
                    )?))
                }
                ferrum_runtime::UpstreamMarkerV2Profile::GatedDelta
                | ferrum_runtime::UpstreamMarkerV2Profile::GatedDeltaPrefill => providers
                    .push(Box::new(
                        transformer::CudaGatedDeltaRecurrentAttentionProvider::new_upstream(
                            runtime,
                            profile.prefill(),
                        )?,
                    )),
                ferrum_runtime::UpstreamMarkerV2Profile::Causal
                | ferrum_runtime::UpstreamMarkerV2Profile::CausalPrefill
                | ferrum_runtime::UpstreamMarkerV2Profile::CausalG32MmqPrefill
                | ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraPrefill
                | ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraLargePrefill
                | ferrum_runtime::UpstreamMarkerV2Profile::CausalExtraAllRows
                | ferrum_runtime::UpstreamMarkerV2Profile::CausalQ6F16 => {
                    providers.push(Box::new(
                        transformer::CudaCausalPagedAttentionProvider::new_upstream(
                            runtime,
                            profile,
                            runtime.attention_execution_policy(),
                        )?,
                    ))
                }
            }
        }
    }
    for profile in [
        ferrum_runtime::Q8ActAttentionProfile::GatedDelta,
        ferrum_runtime::Q8ActAttentionProfile::Causal,
    ] {
        if runtime
            .descriptor()
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == profile.capability_id())
        {
            contracts.push(Box::new(profile.contract().map_err(contract_error)?));
            match profile {
                ferrum_runtime::Q8ActAttentionProfile::GatedDelta => providers.push(Box::new(
                    transformer::CudaGatedDeltaRecurrentAttentionProvider::new_q8act(runtime)?,
                )),
                ferrum_runtime::Q8ActAttentionProfile::Causal => providers.push(Box::new(
                    transformer::CudaCausalPagedAttentionProvider::new_q8act(
                        runtime,
                        runtime.attention_execution_policy(),
                    )?,
                )),
            }
        }
    }
    for profile in [
        ferrum_runtime::Q8ActSwiGluProfile::Iq4Xs,
        ferrum_runtime::Q8ActSwiGluProfile::Q4KQ5KIq4Xs,
    ] {
        if runtime
            .descriptor()
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == profile.capability_id())
        {
            contracts.push(Box::new(profile.contract().map_err(contract_error)?));
            providers.push(Box::new(transformer::CudaQ8ActSwiGluProvider::new(
                runtime, profile,
            )?));
        }
    }
    if runtime.descriptor().capabilities.iter().any(|capability| {
        capability.as_str()
            == ferrum_runtime::DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_CAPABILITY_ID
    }) {
        contracts.push(Box::new(
            ferrum_runtime::dense_swiglu_gguf_rn_f16_fragment_m1to8_contract()
                .map_err(contract_error)?,
        ));
        providers.push(Box::new(transformer::CudaRnFragmentSwiGluProvider::new(
            runtime,
        )?));
    }
    OperationRuntimeRegistry::new(contracts, providers).map_err(contract_error)
}

/// One CUDA composition root shared by planning, provisioning, and dispatch.
/// The capability catalog is derived from the retained registry objects rather
/// than reconstructed from a second descriptor list.
pub struct CudaVNextComposition {
    runtime: Arc<CudaDeviceRuntime>,
    registry: OperationRuntimeRegistry<CudaDeviceRuntime>,
    weight_materializers: WeightMaterializerRegistry,
    catalog: CapabilityCatalog,
}

#[derive(Clone, Copy)]
enum CudaCompositionPurpose {
    Execution,
    NativeCatalogDeclaration,
}

impl CudaVNextComposition {
    fn prepare(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        purpose: CudaCompositionPurpose,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
        causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
        causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
        q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
        execution_context_limit: Option<usize>,
        q8_projection_weight_preparation: ferrum_types::Q8ProjectionWeightPreparation,
        causal_decode_kv_reuse: ferrum_types::CausalDecodeKvReuse,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        // Auto must start with compressed storage. Product startup may create
        // a separate immutable predecoded composition after fitting this plan.
        let q8_projection_weight_preparation = match q8_projection_weight_preparation {
            ferrum_types::Q8ProjectionWeightPreparation::Auto => {
                ferrum_types::Q8ProjectionWeightPreparation::Compressed
            }
            explicit => explicit,
        };
        if causal_decode_attention_grouping
            == ferrum_types::CausalDecodeAttentionGrouping::FixedExecutionCapacity
            && execution_context_limit == Some(0)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "fixed execution capacity requires a positive execution context limit",
            ));
        }
        if matches!(
            q4_mmvq_weight_decode,
            ferrum_types::Q4MmvqWeightDecode::SharedM8
                | ferrum_types::Q4MmvqWeightDecode::SharedM4M8
        ) && (!cfg!(feature = "cuda-q4-shared-weight-decode")
            || (matches!(purpose, CudaCompositionPurpose::Execution)
                && !match q4_mmvq_weight_decode {
                    ferrum_types::Q4MmvqWeightDecode::Auto
                    | ferrum_types::Q4MmvqWeightDecode::Baseline => true,
                    ferrum_types::Q4MmvqWeightDecode::SharedM8 => {
                        q4_shared_weight_decode_compiled()
                    }
                    ferrum_types::Q4MmvqWeightDecode::SharedM4M8 => {
                        q4_shared_weight_decode_m4_m8_compiled()
                    }
                }))
        {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "{} Q4 weight decode requires its compiled upstream native exports",
                q4_mmvq_weight_decode.as_runtime_value()
            )));
        }
        if matches!(
            q5_mmvq_weight_decode,
            ferrum_types::Q5MmvqWeightDecode::SharedM8
                | ferrum_types::Q5MmvqWeightDecode::SharedM4M8
        ) && (!cfg!(feature = "cuda-q5-shared-weight-decode")
            || (matches!(purpose, CudaCompositionPurpose::Execution)
                && !match q5_mmvq_weight_decode {
                    ferrum_types::Q5MmvqWeightDecode::Auto
                    | ferrum_types::Q5MmvqWeightDecode::Baseline => true,
                    ferrum_types::Q5MmvqWeightDecode::SharedM8 => {
                        q5_shared_weight_decode_compiled()
                    }
                    ferrum_types::Q5MmvqWeightDecode::SharedM4M8 => {
                        q5_shared_weight_decode_m4_m8_compiled()
                    }
                }))
        {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "{} Q5 weight decode requires its compiled upstream native exports",
                q5_mmvq_weight_decode.as_runtime_value()
            )));
        }
        if matches!(
            causal_decode_attention_grouping,
            ferrum_types::CausalDecodeAttentionGrouping::Indexed
                | ferrum_types::CausalDecodeAttentionGrouping::FixedCapacity
                | ferrum_types::CausalDecodeAttentionGrouping::FixedExecutionCapacity
        ) && !cfg!(feature = "cuda-indexed-causal-attention")
        {
            return Err(CudaDeviceRuntimeError::contract(
                "indexed causal decode attention grouping requires compiled indexed native attention",
            ));
        }
        if matches!(
            causal_decode_kv_reuse,
            ferrum_types::CausalDecodeKvReuse::PairedQueryHeads
                | ferrum_types::CausalDecodeKvReuse::PairedQueryHeadsUnifiedV2
        ) {
            if causal_decode_attention_grouping
                != ferrum_types::CausalDecodeAttentionGrouping::FixedExecutionCapacity
                || causal_decode_preparation != ferrum_types::CausalDecodePreparationMode::Packed
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "paired query heads require packed fixed-execution-capacity attention",
                ));
            }
            if !cfg!(feature = "cuda-causal-gqa2")
                || (matches!(purpose, CudaCompositionPurpose::Execution)
                    && !causal_decode_kv_reuse_compiled())
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "paired query heads require the compiled production GQA2 native export",
                ));
            }
        }
        let config = cuda_vnext_runtime_config(ordinal, device_id, requested_attention_policy)
            .map_err(contract_error)?;
        if matches!(
            causal_decode_attention_grouping,
            ferrum_types::CausalDecodeAttentionGrouping::Indexed
                | ferrum_types::CausalDecodeAttentionGrouping::FixedCapacity
                | ferrum_types::CausalDecodeAttentionGrouping::FixedExecutionCapacity
        ) && causal_decode_preparation != ferrum_types::CausalDecodePreparationMode::Packed
        {
            return Err(CudaDeviceRuntimeError::contract(
                "indexed causal decode attention grouping requires packed causal decode preparation",
            ));
        }
        if causal_decode_preparation == ferrum_types::CausalDecodePreparationMode::Packed
            && (!cfg!(feature = "vllm-paged-attn-v2")
                || config.attention_execution_policy != AttentionExecutionPolicy::NativeAdaptive)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "packed causal decode preparation requires compiled native-adaptive paged attention",
            ));
        }
        let runtime = Arc::new(
            match purpose {
                CudaCompositionPurpose::Execution => {
                    CudaDeviceRuntime::new_with_program_binding_upload_strategy(
                        config,
                        upload_strategy,
                    )?
                }
                CudaCompositionPurpose::NativeCatalogDeclaration => {
                    CudaDeviceRuntime::new_for_native_catalog(config)?
                }
            }
            .with_segment_binding_oracle(oracle)
            .with_segment_binding_owner_view_mode(owner_view_mode)
            .with_packed_decode_prepare_gate(
                causal_decode_preparation == ferrum_types::CausalDecodePreparationMode::Packed,
            )
            .with_causal_decode_attention_grouping(causal_decode_attention_grouping)
            .with_q4_mmvq_weight_decode(q4_mmvq_weight_decode)
            .with_q5_mmvq_weight_decode(q5_mmvq_weight_decode)
            .with_q8_projection_weight_preparation(q8_projection_weight_preparation)
            .with_causal_decode_kv_reuse(causal_decode_kv_reuse)
            .with_execution_context_limit(execution_context_limit),
        );
        let registry = cuda_vnext_operation_registry(&runtime)?;
        #[allow(unused_mut)]
        let mut weight_materializers = vec![
            crate::gguf_rn_fragment_materializer::gguf_rn_fragment_materializer()
                .map_err(contract_error)?,
        ];
        #[cfg(feature = "vllm-marlin")]
        weight_materializers.extend([
            crate::marlin_fp8_materializer::marlin_fp8_weight_materializer()
                .map_err(contract_error)?,
            crate::marlin_fp8_materializer::block_fp8_to_marlin_fp8_weight_materializer()
                .map_err(contract_error)?,
        ]);
        #[cfg(feature = "vllm-moe-marlin")]
        weight_materializers.push(
            crate::mxfp4_marlin_materializer::gpt_oss_mxfp4_to_marlin_weight_materializer()
                .map_err(contract_error)?,
        );
        let weight_materializers =
            WeightMaterializerRegistry::new(weight_materializers).map_err(contract_error)?;
        let engine = EngineProviderDescriptor::new(
            ProviderId::new(CUDA_ENGINE_PROVIDER_ID).map_err(contract_error)?,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[
                include_str!("vnext_ops.rs").as_bytes(),
                include_str!("vnext_runtime.rs").as_bytes(),
                q6_head::fingerprint().as_bytes(),
                include_str!("vnext_runtime/binding_transfers.rs").as_bytes(),
                include_str!("vnext_runtime/segment_oracle.rs").as_bytes(),
                include_str!("vnext_runtime/segment_oracle/packet.rs").as_bytes(),
                include_str!("vnext_runtime/segment_owners.rs").as_bytes(),
                include_str!("../program_binding_rows.rs").as_bytes(),
                include_str!("../program_binding_upload.rs").as_bytes(),
                CUDA_ENGINE_PROVIDER_ID.as_bytes(),
            ]),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )
        .map_err(contract_error)?;
        let catalog = registry
            .capability_catalog(runtime.descriptor().clone(), vec![engine])
            .map_err(contract_error)?;
        let catalog = weight_materializers
            .augment_catalog(catalog)
            .map_err(contract_error)?;
        Ok(Self {
            runtime,
            registry,
            weight_materializers,
            catalog,
        })
    }

    pub fn create(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_program_binding_upload_strategy(
            ordinal,
            device_id,
            requested_attention_policy,
            ProgramBindingUploadStrategy::Sparse,
        )
    }

    pub fn create_with_program_binding_upload_strategy(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_segment_binding_oracle(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            ferrum_runtime::SegmentBindingOracleMode::Disabled,
        )
    }

    /// Explicit test/diagnostic factory; normal product factories keep Disabled.
    pub fn create_with_segment_binding_oracle(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_segment_binding_owner_view_mode(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            oracle,
            ferrum_runtime::SegmentBindingOwnerViewMode::Legacy,
        )
    }

    /// Explicit representation diagnostic; existing factories remain Legacy.
    pub fn create_with_segment_binding_owner_view_mode(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_causal_decode_preparation_mode(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            oracle,
            owner_view_mode,
            ferrum_types::CausalDecodePreparationMode::PerParticipant,
        )
    }

    /// Select preparation before constructing the actual provider registry and catalog.
    pub fn create_with_causal_decode_preparation_mode(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
        causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_causal_decode_attention_grouping(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            oracle,
            owner_view_mode,
            causal_decode_preparation,
            ferrum_types::CausalDecodeAttentionGrouping::Contiguous,
        )
    }

    /// Select grouping before constructing the provider and native catalog.
    #[allow(clippy::too_many_arguments)]
    pub fn create_with_causal_decode_attention_grouping(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
        causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
        causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_q4_mmvq_weight_decode(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            oracle,
            owner_view_mode,
            causal_decode_preparation,
            causal_decode_attention_grouping,
            ferrum_types::Q4MmvqWeightDecode::Baseline,
        )
    }

    /// Select the bit-preserving native dot variant before constructing the registry.
    #[allow(clippy::too_many_arguments)]
    pub fn create_with_q4_mmvq_weight_decode(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
        causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
        causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_mmvq_weight_decode(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            oracle,
            owner_view_mode,
            causal_decode_preparation,
            causal_decode_attention_grouping,
            q4_mmvq_weight_decode,
            ferrum_types::Q5MmvqWeightDecode::Baseline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_with_mmvq_weight_decode(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
        causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
        causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
        q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_execution_context_limit(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            oracle,
            owner_view_mode,
            causal_decode_preparation,
            causal_decode_attention_grouping,
            q4_mmvq_weight_decode,
            q5_mmvq_weight_decode,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_with_execution_context_limit(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
        causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
        causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
        q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
        execution_context_limit: Option<usize>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_q8_projection_weight_preparation(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            oracle,
            owner_view_mode,
            causal_decode_preparation,
            causal_decode_attention_grouping,
            q4_mmvq_weight_decode,
            q5_mmvq_weight_decode,
            execution_context_limit,
            ferrum_types::Q8ProjectionWeightPreparation::Compressed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_with_q8_projection_weight_preparation(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
        causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
        causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
        q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
        execution_context_limit: Option<usize>,
        q8_projection_weight_preparation: ferrum_types::Q8ProjectionWeightPreparation,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::create_with_causal_decode_kv_reuse(
            ordinal,
            device_id,
            requested_attention_policy,
            upload_strategy,
            oracle,
            owner_view_mode,
            causal_decode_preparation,
            causal_decode_attention_grouping,
            q4_mmvq_weight_decode,
            q5_mmvq_weight_decode,
            execution_context_limit,
            q8_projection_weight_preparation,
            ferrum_types::CausalDecodeKvReuse::None,
        )
    }

    pub fn create_with_causal_decode_kv_reuse(
        ordinal: usize,
        device_id: DeviceId,
        requested_attention_policy: AttentionExecutionPolicy,
        upload_strategy: ProgramBindingUploadStrategy,
        oracle: ferrum_runtime::SegmentBindingOracleMode,
        owner_view_mode: ferrum_runtime::SegmentBindingOwnerViewMode,
        causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
        causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
        q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
        execution_context_limit: Option<usize>,
        q8_projection_weight_preparation: ferrum_types::Q8ProjectionWeightPreparation,
        causal_decode_kv_reuse: ferrum_types::CausalDecodeKvReuse,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let composition = Self::prepare(
            ordinal,
            device_id,
            requested_attention_policy,
            CudaCompositionPurpose::Execution,
            upload_strategy,
            oracle,
            owner_view_mode,
            causal_decode_preparation,
            causal_decode_attention_grouping,
            q4_mmvq_weight_decode,
            q5_mmvq_weight_decode,
            execution_context_limit,
            q8_projection_weight_preparation,
            causal_decode_kv_reuse,
        )?;
        composition.validate_compiled_native_operators()?;
        Ok(composition)
    }

    fn validate_compiled_native_operators(&self) -> Result<(), CudaDeviceRuntimeError> {
        let catalog = &self.catalog;
        let native_provider_catalog = catalog
            .native_operator_provider_catalog(NativeOperatorBackend::Cuda)
            .map_err(contract_error)?;
        crate::native_ops::validate_compiled_native_operator_provider_catalog(
            &native_provider_catalog,
            crate::native_ops::compiled_native_operator_artifacts(),
        )
        .map_err(CudaDeviceRuntimeError::contract)
    }

    pub fn runtime(&self) -> &Arc<CudaDeviceRuntime> {
        &self.runtime
    }

    pub fn registry(&self) -> &OperationRuntimeRegistry<CudaDeviceRuntime> {
        &self.registry
    }

    pub fn catalog(&self) -> &CapabilityCatalog {
        &self.catalog
    }

    pub fn into_parts(
        self,
    ) -> (
        Arc<CudaDeviceRuntime>,
        OperationRuntimeRegistry<CudaDeviceRuntime>,
        WeightMaterializerRegistry,
        CapabilityCatalog,
    ) {
        (
            self.runtime,
            self.registry,
            self.weight_materializers,
            self.catalog,
        )
    }
}

/// Immutable catalog input used to rebuild native artifacts after provider
/// identity changes. It deliberately exposes no executable runtime.
pub struct CudaNativeOperatorCatalogInput {
    provider_catalog: NativeOperatorProviderCatalog,
    capability_catalog: CapabilityCatalog,
}

impl CudaNativeOperatorCatalogInput {
    pub fn provider_catalog(&self) -> &NativeOperatorProviderCatalog {
        &self.provider_catalog
    }

    pub fn capability_catalog(&self) -> &CapabilityCatalog {
        &self.capability_catalog
    }

    pub fn into_parts(self) -> (NativeOperatorProviderCatalog, CapabilityCatalog) {
        (self.provider_catalog, self.capability_catalog)
    }
}

fn cuda_native_operator_catalog_input_from_composition(
    composition: CudaVNextComposition,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    let provider_catalog = composition
        .catalog
        .native_operator_provider_catalog(NativeOperatorBackend::Cuda)
        .map_err(contract_error)?;
    Ok(CudaNativeOperatorCatalogInput {
        provider_catalog,
        capability_catalog: composition.catalog,
    })
}

/// Capture the exact provider identities and validate the installed native
/// artifact set without exposing an executable family-less composition.
pub fn cuda_validated_native_operator_catalog_input(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_validated_native_operator_catalog_input_with_causal_decode_preparation_mode(
        ordinal,
        device_id,
        requested_attention_policy,
        ferrum_types::CausalDecodePreparationMode::PerParticipant,
    )
}

/// Export the live catalog for the explicitly selected preparation mode.
pub fn cuda_validated_native_operator_catalog_input_with_causal_decode_preparation_mode(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_validated_native_operator_catalog_input_with_causal_decode_attention_grouping(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        ferrum_types::CausalDecodeAttentionGrouping::Contiguous,
    )
}

/// Export the live catalog for the selected preparation and grouping policies.
pub fn cuda_validated_native_operator_catalog_input_with_causal_decode_attention_grouping(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_validated_native_operator_catalog_input_with_q4_mmvq_weight_decode(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        ferrum_types::Q4MmvqWeightDecode::Baseline,
    )
}

pub fn cuda_validated_native_operator_catalog_input_with_q4_mmvq_weight_decode(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_validated_native_operator_catalog_input_with_mmvq_weight_decode(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        ferrum_types::Q5MmvqWeightDecode::Baseline,
    )
}

pub fn cuda_validated_native_operator_catalog_input_with_mmvq_weight_decode(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_validated_native_operator_catalog_input_with_execution_context_limit(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        q5_mmvq_weight_decode,
        None,
    )
}

pub fn cuda_validated_native_operator_catalog_input_with_execution_context_limit(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    execution_context_limit: Option<usize>,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_validated_native_operator_catalog_input_with_q8_projection_weight_preparation(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        q5_mmvq_weight_decode,
        execution_context_limit,
        ferrum_types::Q8ProjectionWeightPreparation::Compressed,
    )
}

pub fn cuda_validated_native_operator_catalog_input_with_q8_projection_weight_preparation(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    execution_context_limit: Option<usize>,
    q8_projection_weight_preparation: ferrum_types::Q8ProjectionWeightPreparation,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_validated_native_operator_catalog_input_with_causal_decode_kv_reuse(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        q5_mmvq_weight_decode,
        execution_context_limit,
        q8_projection_weight_preparation,
        ferrum_types::CausalDecodeKvReuse::None,
    )
}

pub fn cuda_validated_native_operator_catalog_input_with_causal_decode_kv_reuse(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    execution_context_limit: Option<usize>,
    q8_projection_weight_preparation: ferrum_types::Q8ProjectionWeightPreparation,
    causal_decode_kv_reuse: ferrum_types::CausalDecodeKvReuse,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    let composition = CudaVNextComposition::prepare(
        ordinal,
        device_id,
        requested_attention_policy,
        CudaCompositionPurpose::Execution,
        ProgramBindingUploadStrategy::Sparse,
        ferrum_runtime::SegmentBindingOracleMode::Disabled,
        ferrum_runtime::SegmentBindingOwnerViewMode::Legacy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        q5_mmvq_weight_decode,
        execution_context_limit,
        q8_projection_weight_preparation,
        causal_decode_kv_reuse,
    )?;
    composition.validate_compiled_native_operators()?;
    cuda_native_operator_catalog_input_from_composition(composition)
}

/// Capture the exact provider identities needed to package a new native
/// artifact set. Product composition uses [`CudaVNextComposition::create`]
/// before family/profile selection and cannot bypass compiled-artifact validation.
/// This output declares actual source-constructed provider contracts; it does
/// not establish linked-artifact availability or numerical/GPU qualification.
/// Native FFI planning and launches are never called on this catalog-only path.
pub fn cuda_native_operator_catalog_input(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_native_operator_catalog_input_with_causal_decode_preparation_mode(
        ordinal,
        device_id,
        requested_attention_policy,
        ferrum_types::CausalDecodePreparationMode::PerParticipant,
    )
}

/// Declare source-constructed identities for an explicit mode without native execution.
pub fn cuda_native_operator_catalog_input_with_causal_decode_preparation_mode(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_native_operator_catalog_input_with_causal_decode_attention_grouping(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        ferrum_types::CausalDecodeAttentionGrouping::Contiguous,
    )
}

/// Declare source identities for grouping without bypassing artifact validation at execution.
pub fn cuda_native_operator_catalog_input_with_causal_decode_attention_grouping(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_native_operator_catalog_input_with_q4_mmvq_weight_decode(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        ferrum_types::Q4MmvqWeightDecode::Baseline,
    )
}

pub fn cuda_native_operator_catalog_input_with_q4_mmvq_weight_decode(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_native_operator_catalog_input_with_mmvq_weight_decode(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        ferrum_types::Q5MmvqWeightDecode::Baseline,
    )
}

pub fn cuda_native_operator_catalog_input_with_mmvq_weight_decode(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_native_operator_catalog_input_with_execution_context_limit(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        q5_mmvq_weight_decode,
        None,
    )
}

pub fn cuda_native_operator_catalog_input_with_execution_context_limit(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    execution_context_limit: Option<usize>,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_native_operator_catalog_input_with_q8_projection_weight_preparation(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        q5_mmvq_weight_decode,
        execution_context_limit,
        ferrum_types::Q8ProjectionWeightPreparation::Compressed,
    )
}

pub fn cuda_native_operator_catalog_input_with_q8_projection_weight_preparation(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    execution_context_limit: Option<usize>,
    q8_projection_weight_preparation: ferrum_types::Q8ProjectionWeightPreparation,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    cuda_native_operator_catalog_input_with_causal_decode_kv_reuse(
        ordinal,
        device_id,
        requested_attention_policy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        q5_mmvq_weight_decode,
        execution_context_limit,
        q8_projection_weight_preparation,
        ferrum_types::CausalDecodeKvReuse::None,
    )
}

pub fn cuda_native_operator_catalog_input_with_causal_decode_kv_reuse(
    ordinal: usize,
    device_id: DeviceId,
    requested_attention_policy: AttentionExecutionPolicy,
    causal_decode_preparation: ferrum_types::CausalDecodePreparationMode,
    causal_decode_attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    execution_context_limit: Option<usize>,
    q8_projection_weight_preparation: ferrum_types::Q8ProjectionWeightPreparation,
    causal_decode_kv_reuse: ferrum_types::CausalDecodeKvReuse,
) -> Result<CudaNativeOperatorCatalogInput, CudaDeviceRuntimeError> {
    let composition = CudaVNextComposition::prepare(
        ordinal,
        device_id,
        requested_attention_policy,
        CudaCompositionPurpose::NativeCatalogDeclaration,
        ProgramBindingUploadStrategy::Sparse,
        ferrum_runtime::SegmentBindingOracleMode::Disabled,
        ferrum_runtime::SegmentBindingOwnerViewMode::Legacy,
        causal_decode_preparation,
        causal_decode_attention_grouping,
        q4_mmvq_weight_decode,
        q5_mmvq_weight_decode,
        execution_context_limit,
        q8_projection_weight_preparation,
        causal_decode_kv_reuse,
    )?;
    cuda_native_operator_catalog_input_from_composition(composition)
}

pub struct CudaTokenEmbeddingProvider {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
    native: native_blocks::CudaNativeBlockKernels,
    precision: TokenPrecision,
}

impl CudaTokenEmbeddingProvider {
    pub fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_precision(runtime, TokenPrecision::F16)
    }

    pub fn new_f32(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_precision(runtime, TokenPrecision::F32)
    }

    fn with_precision(
        runtime: &CudaDeviceRuntime,
        precision: TokenPrecision,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let (contract, provider, capability, estimator) = match precision {
            TokenPrecision::F16 => (
                token_embedding_contract(),
                TOKEN_EMBEDDING_PROVIDER_ID,
                TOKEN_EMBEDDING_F16_CAPABILITY_ID,
                TOKEN_EMBEDDING_ESTIMATOR_ID,
            ),
            TokenPrecision::F32 => (
                token_embedding_f32_master_contract(),
                "provider.cuda.token_embedding.f32-master",
                TOKEN_EMBEDDING_F32_MASTER_CAPABILITY_ID,
                "resource-estimator.cuda.token_embedding.f32-master",
            ),
        };
        let contract = contract.map_err(contract_error)?;
        // Lookup reads immutable weights and the exact immediate token IDs;
        // it has no sequence state or partition-dependent accumulation. Its
        // completed output can accompany either KV state's retained prefix.
        let descriptor =
            native_io::descriptor(runtime, &contract, provider, capability, estimator)?
                .with_checkpoint_capability(ProviderCheckpointCapability::CompletedBoundary(
                    ProviderCheckpointContract::new(
                        CheckpointInputDependency::ExactTokenPrefix,
                        CheckpointBoundaryConstraint::any_positive(),
                        CheckpointPartitionNumerics::CapturedExecutionContinuation,
                    )
                    .with_completed_input_capture(CheckpointCompletedInputCapture::Supported),
                ));
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::EMBEDDING_LOOKUP.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("embedding module load", error))?;
        let function = module
            .load_function(EMBEDDING_FUNCTION_NAME)
            .map_err(|error| CudaDeviceRuntimeError::driver("embedding function load", error))?;
        Ok(Self {
            descriptor,
            function,
            native: native_blocks::CudaNativeBlockKernels::load(runtime.context())?,
            precision,
        })
    }
}

impl OperationResourceEstimator for CudaTokenEmbeddingProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != self.precision.embedding_operation()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "CUDA token embedding estimator received another operation".to_owned(),
            });
        }
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            native_blocks::hadamard::token_workspace(request.values())?,
            None,
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaTokenEmbeddingProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        reusable_token_topology(
            &request,
            b"ferrum.cuda.token-embedding.reusable-topology.v2\0",
        )
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        let encoded =
            if self.precision == TokenPrecision::F32 || native_io::requires_native(&invocation) {
                native_io::encode_embedding(
                    self.descriptor.provider_implementation_fingerprint(),
                    &self.native,
                    self.precision,
                    invocation,
                )
            } else {
                encode_token_embedding(
                    &self.function,
                    self.descriptor.provider_implementation_fingerprint(),
                    invocation,
                )
            };
        encoded
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "cuda.token_embedding.encode", message))
    }
}

pub struct CudaLastTokenDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
    native: native_blocks::CudaNativeBlockKernels,
    precision: TokenPrecision,
}

impl CudaLastTokenDenseLinearProvider {
    pub fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_precision(runtime, TokenPrecision::F16)
    }

    pub fn new_f32(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_precision(runtime, TokenPrecision::F32)
    }

    fn with_precision(
        runtime: &CudaDeviceRuntime,
        precision: TokenPrecision,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let (contract, provider, capability, estimator) = match precision {
            TokenPrecision::F16 => (
                last_token_dense_linear_contract(),
                LAST_TOKEN_DENSE_LINEAR_PROVIDER_ID,
                LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
                LAST_TOKEN_DENSE_LINEAR_ESTIMATOR_ID,
            ),
            TokenPrecision::F32 => (
                last_token_dense_linear_f32_contract(),
                "provider.cuda.last_token_dense_linear.f32.native",
                LAST_TOKEN_DENSE_LINEAR_F32_CAPABILITY_ID,
                "resource-estimator.cuda.last_token_dense_linear.f32.native",
            ),
        };
        let contract = contract.map_err(contract_error)?;
        let descriptor =
            native_io::descriptor(runtime, &contract, provider, capability, estimator)?;
        Ok(Self {
            descriptor,
            native: native_blocks::CudaNativeBlockKernels::load(runtime.context())?,
            precision,
        })
    }
}

impl OperationResourceEstimator for CudaLastTokenDenseLinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        transformer::ensure_estimator_request(
            &self.descriptor,
            &request,
            self.precision.projection_operation(),
        )?;
        let scratch = if last_token_uses_dense_f16(self.precision, request.values()) {
            let hidden = unsigned_attribute(request.attributes(), "hidden_size")
                .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
            Some(last_token_gather_workspace(hidden)?)
        } else {
            native_blocks::hadamard::token_workspace(request.values())?
        };
        Ok(transformer::estimate(
            &self.descriptor,
            request.input_fingerprint(),
            scratch,
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaLastTokenDenseLinearProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        // A dense F16 batch may gather even when its semantic spans are unit
        // rows: physical input windows can still have gaps. Capture therefore
        // requires the declared scratch address as well as every value address.
        if last_token_uses_dense_f16(self.precision, request.bindings())
            && request.work_shape().participant_token_ranges().len() > 1
            && request.scratch_reusable_address_scope()?.is_none()
        {
            return Ok(ReusableExecutionTopology::EagerBoundary);
        }
        reusable_token_topology(
            &request,
            b"ferrum.cuda.last-token-linear.reusable-topology.v3\0",
        )
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        let encoded =
            if self.precision == TokenPrecision::F32 || native_io::requires_native(&invocation) {
                native_io::encode_projection(
                    self.descriptor.provider_implementation_fingerprint(),
                    &self.native,
                    self.precision,
                    invocation,
                )
            } else {
                encode_last_token_dense_linear(
                    self.descriptor.provider_implementation_fingerprint(),
                    invocation,
                )
            };
        encoded
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| {
                provider_failure(identity, "cuda.last_token_dense_linear.encode", message)
            })
    }
}

pub struct CudaLastTokenMaskedArgmaxProvider {
    descriptor: OperationProviderDescriptor,
    functions: ArgmaxFunctions,
    precision: ArgmaxPrecision,
}

impl CudaLastTokenMaskedArgmaxProvider {
    pub fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_precision(runtime, ArgmaxPrecision::F16)
    }

    pub fn new_f32(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_precision(runtime, ArgmaxPrecision::F32)
    }

    fn with_precision(
        runtime: &CudaDeviceRuntime,
        precision: ArgmaxPrecision,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = precision.contract().map_err(contract_error)?;
        let descriptor = transformer::weightless_provider_descriptor(
            runtime,
            &contract,
            precision.provider(),
            precision.capability(),
            precision.estimator(),
            transformer::contiguous_bindings(5),
            implementation_fingerprint(&[
                include_str!("vnext_ops.rs").as_bytes(),
                include_str!("vnext_ops/selection.rs").as_bytes(),
                crate::ptx::ARGMAX_ROWS.as_bytes(),
                precision.kernel().as_bytes(),
            ]),
        )?;
        let functions = ArgmaxFunctions::load(runtime.context(), precision)?;
        Ok(Self {
            descriptor,
            functions,
            precision,
        })
    }
}

impl OperationResourceEstimator for CudaLastTokenMaskedArgmaxProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        transformer::ensure_estimator_request(
            &self.descriptor,
            &request,
            self.precision.operation(),
        )?;
        let vocabulary_size = unsigned_attribute(request.attributes(), "vocab_size")
            .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
        let scratch_bytes = masked_argmax_scratch_stride(vocabulary_size, self.precision.element())
            .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::actual_sequences(scratch_bytes)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(transformer::estimate(
            &self.descriptor,
            request.input_fingerprint(),
            Some(scratch),
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaLastTokenMaskedArgmaxProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        transformer::static_contiguous_reusable_topology(
            &request,
            5,
            &[transformer::CapturedProviderWorkspace::Scratch],
        )
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_last_token_masked_argmax(
            &self.functions,
            self.descriptor.provider_implementation_fingerprint(),
            self.precision,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            provider_failure(identity, "cuda.last_token_masked_argmax.encode", message)
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct MaskedArgmaxLaunch {
    first_region: usize,
    scratch_offset_bytes: u64,
    vocabulary_size: i32,
    repetition_capacity: i32,
}

fn encode_last_token_masked_argmax(
    functions: &ArgmaxFunctions,
    provider_fingerprint: &str,
    precision: ArgmaxPrecision,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.operation().id.as_str() != precision.operation()
        || invocation.participants().is_empty()
    {
        return Err("CUDA masked argmax received another or empty operation".to_owned());
    }

    let first_vocabulary_size =
        unsigned_attribute(invocation.participants()[0].attributes(), "vocab_size")?;
    let scratch_stride = masked_argmax_scratch_stride(first_vocabulary_size, precision.element())?;
    let required_scratch_bytes = scratch_stride
        .checked_mul(invocation.participants().len() as u64)
        .ok_or_else(|| "CUDA masked argmax scratch size overflows".to_owned())?;
    let mut regions = Vec::with_capacity(invocation.participants().len() * 6 + 1);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant_index, participant) in invocation.participants().iter().enumerate() {
        let logits = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let valid_mask = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let repetition_token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 2)?;
        let repetition_offsets = binding(participant.bindings(), ResolvedValueRole::Input, 3)?;
        let repetition_penalty = binding(participant.bindings(), ResolvedValueRole::Input, 4)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        if vocabulary_size != first_vocabulary_size {
            return Err("CUDA masked argmax participants disagree on vocabulary size".to_owned());
        }
        let repetition_capacity = validate_masked_argmax_signature(
            logits,
            valid_mask,
            repetition_token_ids,
            repetition_offsets,
            repetition_penalty,
            output,
            vocabulary_size,
            precision.element(),
        )?;

        let first_region = regions.len();
        regions.push(contiguous_region(participant, logits, precision.element())?);
        regions.push(contiguous_region(participant, valid_mask, ElementType::U8)?);
        regions.push(contiguous_region(
            participant,
            repetition_token_ids,
            ElementType::U32,
        )?);
        regions.push(contiguous_region(
            participant,
            repetition_offsets,
            ElementType::U32,
        )?);
        regions.push(contiguous_region(
            participant,
            repetition_penalty,
            ElementType::F32,
        )?);
        regions.push(contiguous_region(participant, output, ElementType::U32)?);
        launches.push(MaskedArgmaxLaunch {
            first_region,
            scratch_offset_bytes: scratch_stride
                .checked_mul(participant_index as u64)
                .ok_or_else(|| "CUDA masked argmax scratch offset overflows".to_owned())?,
            vocabulary_size: i32::try_from(vocabulary_size)
                .map_err(|_| "masked argmax vocabulary exceeds i32".to_owned())?,
            repetition_capacity,
        });
    }
    let scratch_region = regions.len();
    regions.push(transformer::shared_scratch_region(
        &invocation,
        required_scratch_bytes,
    )?);

    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "masked argmax participant count exceeds u32".to_owned())?;
    let dispatches_per_participant = argmax_dispatches(launches[0].vocabulary_size);
    let parallel = dispatches_per_participant == 2;
    let batched = parallel
        && (2..=ARGMAX_BATCH_ROWS).contains(&launches.len())
        && masked_argmax_batch_ranges_are_independent(
            &regions
                .iter()
                .map(|region| (region.device_ptr(), region.length_bytes()))
                .collect::<Vec<_>>(),
            launches.iter().map(|launch| launch.first_region + 5),
            scratch_region,
        );
    let mut replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_last_token_masked_argmax")
            .u64(dispatches_per_participant)
            .u64(u64::from(batched))
            .u64(launches.len() as u64);
    for launch in &launches {
        replay_key = replay_key
            .u64(launch.first_region as u64)
            .u64(launch.scratch_offset_bytes)
            .i32(launch.vocabulary_size)
            .i32(launch.repetition_capacity);
    }
    let functions = functions.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_last_token_masked_argmax",
        regions,
        replay_key.finish(),
        move |stream, regions| {
            let arguments =
                |launch: &MaskedArgmaxLaunch| -> Result<ArgmaxArguments, CudaDeviceRuntimeError> {
                    let logits = regions[launch.first_region].device_ptr();
                    let valid_mask = regions[launch.first_region + 1].device_ptr();
                    let repetition_token_ids = regions[launch.first_region + 2].device_ptr();
                    let repetition_offsets = regions[launch.first_region + 3].device_ptr();
                    let repetition_penalty = regions[launch.first_region + 4].device_ptr();
                    let output = regions[launch.first_region + 5].device_ptr();
                    let scratch = regions[scratch_region]
                        .device_ptr()
                        .checked_add(launch.scratch_offset_bytes)
                        .ok_or_else(|| {
                            CudaDeviceRuntimeError::contract(
                                "vNext masked argmax scratch pointer overflows",
                            )
                        })?;
                    Ok(ArgmaxArguments {
                        logits,
                        scratch,
                        valid_mask,
                        repetition_offsets,
                        repetition_token_ids,
                        repetition_penalty,
                        output,
                        vocabulary_size: launch.vocabulary_size,
                        repetition_capacity: launch.repetition_capacity,
                    })
                };
            if batched {
                let mut batch_rows = [ArgmaxArguments::default(); ARGMAX_BATCH_ROWS];
                for (index, launch) in launches.iter().enumerate() {
                    batch_rows[index] = arguments(launch)?;
                }
                let parameters = ArgmaxBatchArguments::new(&batch_rows[..launches.len()])
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("invalid batched argmax rows")
                    })?;
                functions.launch_batched(stream, &parameters, participant_count)?;
            } else {
                for launch in &launches {
                    functions.launch(stream, arguments(launch)?, parallel)?;
                }
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            if batched {
                DeviceBatchingForm::Packed
            } else if participant_count == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participant_count,
            u64::from(participant_count),
            if batched {
                2
            } else {
                u64::from(participant_count) * dispatches_per_participant
            },
            0,
        )
    })
    .map_err(|error| error.to_string())
}

// The participant-loop path permits ordered aliases. Parallel rows may share
// read-only policy/logits, but outputs and scratch must remain independent.
fn masked_argmax_batch_ranges_are_independent(
    ranges: &[(u64, u64)],
    output_indices: impl Iterator<Item = usize>,
    scratch_index: usize,
) -> bool {
    output_indices
        .chain(std::iter::once(scratch_index))
        .all(|write_index| {
            let Some(&(write_start, write_length)) = ranges.get(write_index) else {
                return false;
            };
            let Some(write_end) = write_start.checked_add(write_length) else {
                return false;
            };
            write_length != 0
                && ranges.iter().enumerate().all(|(index, &(start, length))| {
                    let Some(end) = start.checked_add(length) else {
                        return false;
                    };
                    length != 0
                        && (index == write_index || write_end <= start || end <= write_start)
                })
        })
}

#[cfg(test)]
mod masked_argmax_batch_alias_tests {
    use super::masked_argmax_batch_ranges_are_independent;

    #[test]
    fn parallel_rows_share_reads_but_fall_back_for_ordered_writes() {
        let independent = [(100, 8), (100, 8), (108, 4), (112, 4), (116, 256)];
        assert!(masked_argmax_batch_ranges_are_independent(
            &independent,
            [2, 3].into_iter(),
            4,
        ));
        for (index, alias) in [(3, (108, 4)), (2, (104, 4)), (4, (100, 256))] {
            let mut ranges = independent;
            ranges[index] = alias;
            assert!(!masked_argmax_batch_ranges_are_independent(
                &ranges,
                [2, 3].into_iter(),
                4,
            ));
        }
        let mut overflowing = independent;
        overflowing[0] = (u64::MAX, 8);
        assert!(!masked_argmax_batch_ranges_are_independent(
            &overflowing,
            [2, 3].into_iter(),
            4,
        ));
    }
}

fn masked_argmax_scratch_stride(
    vocabulary_size: u64,
    element_type: ElementType,
) -> Result<u64, String> {
    let bytes = vocabulary_size
        .checked_mul(element_type.size_bytes())
        .ok_or_else(|| "CUDA masked argmax scratch size overflows".to_owned())?;
    bytes
        .checked_add(VALUE_ALIGNMENT_BYTES - 1)
        .map(|value| value & !(VALUE_ALIGNMENT_BYTES - 1))
        .filter(|value| *value != 0)
        .ok_or_else(|| "CUDA masked argmax scratch alignment overflows".to_owned())
}

fn validate_masked_argmax_signature(
    logits: &ResolvedValueBinding,
    valid_mask: &ResolvedValueBinding,
    repetition_token_ids: &ResolvedValueBinding,
    repetition_offsets: &ResolvedValueBinding,
    repetition_penalty: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    vocabulary_size: u64,
    logits_type: ElementType,
) -> Result<i32, String> {
    let contiguous = |binding: &ResolvedValueBinding| {
        matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
    };
    if logits.tensor().element_type() != logits_type
        || valid_mask.tensor().element_type() != ElementType::U8
        || repetition_token_ids.tensor().element_type() != ElementType::U32
        || repetition_offsets.tensor().element_type() != ElementType::U32
        || repetition_penalty.tensor().element_type() != ElementType::F32
        || output.tensor().element_type() != ElementType::U32
        || logits.tensor().dimensions() != [1, vocabulary_size]
        || valid_mask.tensor().dimensions() != [vocabulary_size]
        || repetition_token_ids.tensor().dimensions().len() != 1
        || repetition_token_ids.tensor().dimensions()[0] == 0
        || repetition_offsets.tensor().dimensions() != [2]
        || repetition_penalty.tensor().dimensions() != [1]
        || output.tensor().dimensions() != [1]
        || !contiguous(logits)
        || !contiguous(valid_mask)
        || !contiguous(repetition_token_ids)
        || !contiguous(repetition_offsets)
        || !contiguous(repetition_penalty)
        || !contiguous(output)
    {
        return Err("CUDA masked argmax invocation differs from its resolved signature".to_owned());
    }
    i32::try_from(repetition_token_ids.tensor().dimensions()[0])
        .map_err(|_| "CUDA masked argmax repetition capacity exceeds i32".to_owned())
}

#[derive(Debug, Clone, Copy)]
struct LastTokenDenseLinearLaunch {
    input_region: usize,
    output_region: usize,
    rows: i32,
}

#[derive(Debug, Clone, Copy)]
struct LastTokenF16Row {
    input_pointer: u64,
    input_bytes: u64,
    output_pointer: u64,
    output_bytes: u64,
}

fn last_token_uses_dense_f16(precision: TokenPrecision, values: &[ResolvedValueBinding]) -> bool {
    precision == TokenPrecision::F16
        && !values.iter().any(|value| {
            value.role() == ResolvedValueRole::Input
                && value.ordinal() == 1
                && value.weight().is_some_and(|weight| {
                    !matches!(
                        weight.physical_layout(),
                        ferrum_runtime::PhysicalWeightLayout::Dense { .. }
                    )
                })
        })
}

fn last_token_gather_workspace(hidden: u64) -> Result<ProviderWorkspaceRequirement, VNextError> {
    let bytes = hidden
        .checked_mul(2)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| VNextError::InvalidExecutionPlan {
            reason: "CUDA last-token gather row extent is zero or overflowing".into(),
        })?;
    ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::actual_sequences(bytes)?,
        VALUE_ALIGNMENT_BYTES,
        ProviderWorkspaceScope::Invocation,
        ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
        DynamicStorageRequirement::contiguous(),
    )
}

#[derive(Clone, Copy, Debug)]
struct LastTokenByteSpan {
    pointer: u64,
    bytes: u64,
}

fn last_token_access_range(
    span: LastTokenByteSpan,
    bytes: u64,
) -> Result<std::ops::Range<u64>, String> {
    if span.pointer == 0
        || span.pointer % 2 != 0
        || bytes == 0
        || span.bytes < bytes
        || span.pointer.checked_add(span.bytes).is_none()
    {
        return Err("CUDA last-token matrix has an invalid physical extent".into());
    }
    Ok(span.pointer
        ..span
            .pointer
            .checked_add(bytes)
            .ok_or("CUDA last-token address overflows")?)
}

fn last_token_ranges_overlap(a: &std::ops::Range<u64>, b: &std::ops::Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

/// Validate actual reads/writes before either packing or the scalar fallback.
/// Padding outside a semantic row is not accessed, and read/read alias is legal.
fn validate_last_token_f16_access(
    hidden: u64,
    outputs: u64,
    weight: LastTokenByteSpan,
    rows: &[LastTokenF16Row],
) -> Result<(), String> {
    let input_bytes = hidden
        .checked_mul(2)
        .ok_or("CUDA last-token input extent overflows")?;
    let output_bytes = outputs
        .checked_mul(2)
        .ok_or("CUDA last-token output extent overflows")?;
    let weight_bytes = input_bytes
        .checked_mul(outputs)
        .ok_or("CUDA last-token weight extent overflows")?;
    let weight_read = last_token_access_range(weight, weight_bytes)?;
    let inputs = rows
        .iter()
        .map(|row| {
            last_token_access_range(
                LastTokenByteSpan {
                    pointer: row.input_pointer,
                    bytes: row.input_bytes,
                },
                input_bytes,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let writes = rows
        .iter()
        .map(|row| {
            last_token_access_range(
                LastTokenByteSpan {
                    pointer: row.output_pointer,
                    bytes: row.output_bytes,
                },
                output_bytes,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if rows.is_empty()
        || writes.iter().enumerate().any(|(index, write)| {
            last_token_ranges_overlap(write, &weight_read)
                || inputs
                    .iter()
                    .any(|read| last_token_ranges_overlap(write, read))
                || writes[..index]
                    .iter()
                    .any(|prior| last_token_ranges_overlap(write, prior))
        })
    {
        return Err(
            "CUDA last-token matrix has overlapping participant writes or read/write alias".into(),
        );
    }
    Ok(())
}

fn last_token_output_rows_are_contiguous(outputs: u64, rows: &[LastTokenF16Row]) -> bool {
    let Some(bytes) = outputs.checked_mul(2).filter(|bytes| *bytes > 0) else {
        return false;
    };
    rows.len() > 1
        && rows.iter().all(|row| row.output_bytes == bytes)
        && rows
            .windows(2)
            .all(|pair| pair[0].output_pointer.checked_add(bytes) == Some(pair[1].output_pointer))
}

/// Scratch is optional to the old routes. Alias here rejects only the gather
/// optimization; the ordinary read/write contract was already checked above.
fn last_token_gather_scratch_is_disjoint(
    hidden: u64,
    outputs: u64,
    weight: LastTokenByteSpan,
    scratch: LastTokenByteSpan,
    rows: &[LastTokenF16Row],
) -> bool {
    let Some(input_bytes) = hidden.checked_mul(2) else {
        return false;
    };
    let Some(output_bytes) = outputs.checked_mul(2) else {
        return false;
    };
    let Some(weight_bytes) = input_bytes.checked_mul(outputs) else {
        return false;
    };
    let Some(required) = input_bytes.checked_mul(rows.len() as u64) else {
        return false;
    };
    if scratch.pointer % VALUE_ALIGNMENT_BYTES != 0 {
        return false;
    }
    let Ok(write) = last_token_access_range(scratch, required) else {
        return false;
    };
    let Ok(weight_read) = last_token_access_range(weight, weight_bytes) else {
        return false;
    };
    !last_token_ranges_overlap(&write, &weight_read) && rows.iter().all(|row| {
        let input = last_token_access_range(LastTokenByteSpan { pointer: row.input_pointer, bytes: row.input_bytes }, input_bytes);
        let output = last_token_access_range(LastTokenByteSpan { pointer: row.output_pointer, bytes: row.output_bytes }, output_bytes);
        matches!((input, output), (Ok(input), Ok(output)) if !last_token_ranges_overlap(&write, &input) && !last_token_ranges_overlap(&write, &output))
    })
}

fn copy_last_token_rows_f16(
    stream: &cudarc::driver::CudaStream,
    destination: u64,
    row_bytes: usize,
    sources: impl ExactSizeIterator<Item = u64>,
) -> Result<(), CudaDeviceRuntimeError> {
    for (row, source) in sources.enumerate() {
        let offset = row
            .checked_mul(row_bytes)
            .and_then(|offset| u64::try_from(offset).ok())
            .and_then(|offset| destination.checked_add(offset))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("CUDA last-token copy offset overflows")
            })?;
        // SAFETY: the encoder proves each complete source row and the disjoint
        // packed destination extent; command regions retain every allocation.
        // Copies and the following GEMM are enqueued on this same stream.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(offset, source, row_bytes, stream.cu_stream())
        }
        .map_err(|error| CudaDeviceRuntimeError::driver("CUDA last-token row gather", error))?;
    }
    Ok(())
}

/// Proves that individually retained F16 row regions cover two dense matrices.
/// Product logits use participant windows, whose aligned stride need not equal
/// their semantic row width. Neither shared backing nor unit token ranges alone
/// therefore authorize a multi-row GEMM.
fn packed_last_token_rows(
    input_packed: bool,
    hidden_size: u64,
    out_features: u64,
    token_ranges: impl ExactSizeIterator<Item = std::ops::Range<u64>>,
    regions: impl ExactSizeIterator<Item = LastTokenF16Row>,
) -> Option<i32> {
    let count = token_ranges.len();
    if !input_packed || count < 2 || regions.len() != count {
        return None;
    }
    let rows = i32::try_from(count).ok()?;
    let input_bytes = hidden_size.checked_mul(2).filter(|bytes| *bytes > 0)?;
    let output_bytes = out_features.checked_mul(2).filter(|bytes| *bytes > 0)?;
    let mut input_start = None;
    let mut output_start = None;
    let mut input_end = None;
    let mut output_end = None;
    for (index, (range, region)) in token_ranges.zip(regions).enumerate() {
        let row = u64::try_from(index).ok()?;
        if range != (row..row.checked_add(1)?)
            || region.input_bytes != input_bytes
            || region.output_bytes != output_bytes
            || region.input_pointer == 0
            || region.output_pointer == 0
            || input_end.is_some_and(|end| region.input_pointer != end)
            || output_end.is_some_and(|end| region.output_pointer != end)
        {
            return None;
        }
        input_start.get_or_insert(region.input_pointer);
        output_start.get_or_insert(region.output_pointer);
        input_end = Some(region.input_pointer.checked_add(input_bytes)?);
        output_end = Some(region.output_pointer.checked_add(output_bytes)?);
    }
    // cuBLAS must not overwrite an input row that another matrix column reads.
    if input_start? < output_end? && output_start? < input_end? {
        return None;
    }
    Some(rows)
}

fn encode_last_token_dense_linear(
    provider_fingerprint: &str,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.operation().id.as_str() != LAST_TOKEN_DENSE_LINEAR_OPERATION_ID
        || invocation.participants().is_empty()
    {
        return Err("CUDA last-token dense-linear received another or empty operation".to_owned());
    }
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA last-token dense-linear participant ranges are incomplete".to_owned());
    }
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let out_features = unsigned_attribute(first.attributes(), "out_features")?;
    let input_packed =
        transformer::token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let mut regions = vec![transformer::shared_full_region(
        &invocation,
        ResolvedValueRole::Input,
        1,
        ElementType::F16,
    )?];
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let participant_weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "out_features")? != out_features
        {
            return Err("CUDA last-token dense-linear participant attributes disagree".to_owned());
        }
        validate_last_token_dense_linear_signature(
            input,
            participant_weight,
            output,
            hidden_size,
            out_features,
            ElementType::F16,
        )?;
        let source_range = token_range.source_token_range();
        let packed_range = token_range.immediate_token_range();
        let selected_range = if input_packed {
            packed_range
        } else {
            source_range
        };
        if selected_range.is_empty() {
            return Err("CUDA last-token dense-linear cannot select from an empty span".to_owned());
        }
        let last_token = selected_range.end - 1;
        let input_region = regions.len();
        let source = contiguous_token_region(participant, input, ElementType::F16, last_token, 1)?;
        regions.push(source);
        let output_region = regions.len();
        let destination = contiguous_region(participant, output, ElementType::F16)?;
        regions.push(destination);
        launches.push(LastTokenDenseLinearLaunch {
            input_region,
            output_region,
            rows: 1,
        });
    }

    let physical_rows = launches
        .iter()
        .map(|launch| {
            let input = &regions[launch.input_region];
            let output = &regions[launch.output_region];
            LastTokenF16Row {
                input_pointer: input.device_ptr(),
                input_bytes: input.length_bytes(),
                output_pointer: output.device_ptr(),
                output_bytes: output.length_bytes(),
            }
        })
        .collect::<Vec<_>>();
    let weight_span = LastTokenByteSpan {
        pointer: regions[0].device_ptr(),
        bytes: regions[0].length_bytes(),
    };
    validate_last_token_f16_access(hidden_size, out_features, weight_span, &physical_rows)?;
    let packed_rows = packed_last_token_rows(
        input_packed,
        hidden_size,
        out_features,
        token_ranges
            .iter()
            .map(|range| range.immediate_token_range()),
        physical_rows.iter().copied(),
    );
    let row_bytes = hidden_size
        .checked_mul(2)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("CUDA last-token gather row bytes overflow usize")?;
    let mut gather_sources = Vec::new();
    let mut gather_scratch = None;
    if let Some(rows) = packed_rows {
        // Keep every original region lease, including all output windows. Only
        // coalesce dispatches after proving they form a contiguous matrix.
        launches.truncate(1);
        launches[0].rows = rows;
    } else if last_token_output_rows_are_contiguous(out_features, &physical_rows) {
        let required = row_bytes
            .checked_mul(physical_rows.len())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or("CUDA last-token gather workspace extent overflows")?;
        let scratch = transformer::shared_scratch_region(&invocation, required)?;
        if last_token_gather_scratch_is_disjoint(
            hidden_size,
            out_features,
            weight_span,
            LastTokenByteSpan {
                pointer: scratch.device_ptr(),
                bytes: scratch.length_bytes(),
            },
            &physical_rows,
        ) {
            gather_sources = launches.iter().map(|launch| launch.input_region).collect();
            let index = regions.len();
            regions.push(scratch);
            gather_scratch = Some(index);
            let rows = i32::try_from(launches.len())
                .map_err(|_| "CUDA last-token gather rows exceed i32")?;
            launches.truncate(1);
            launches[0].input_region = index;
            launches[0].rows = rows;
        }
    }
    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "last-token dense-linear participant count exceeds u32".to_owned())?;
    let token_count = u64::from(participant_count);
    let compute_dispatch_count = launches.len() as u64;
    let transfer_command_count = gather_sources.len() as u64;
    let hidden_size = i32::try_from(hidden_size)
        .map_err(|_| "last-token dense-linear hidden size exceeds i32".to_owned())?;
    let out_features = i32::try_from(out_features)
        .map_err(|_| "last-token dense-linear output width exceeds i32".to_owned())?;
    let replay_key = last_token_linear_replay_key(
        provider_fingerprint,
        packed_rows.is_some(),
        hidden_size,
        out_features,
        &launches,
        gather_scratch,
        &gather_sources,
        row_bytes,
    );
    CudaDeviceCommand::replayable_operation_with_blas(
        "vnext_last_token_dense_linear",
        regions,
        replay_key,
        move |stream, blas, regions| {
            let weight = regions[0].device_ptr();
            if let Some(index) = gather_scratch {
                copy_last_token_rows_f16(
                    stream,
                    regions[index].device_ptr(),
                    row_bytes,
                    gather_sources
                        .iter()
                        .map(|source| regions[*source].device_ptr()),
                )?;
            }
            for launch in &launches {
                transformer::launch_gemm_f16(
                    blas,
                    regions[launch.input_region].device_ptr(),
                    weight,
                    regions[launch.output_region].device_ptr(),
                    launch.rows,
                    out_features,
                    hidden_size,
                    "vNext last-token dense-linear GEMM",
                )?;
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            if packed_rows.is_some() || gather_scratch.is_some() {
                DeviceBatchingForm::Packed
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participant_count,
            token_count,
            compute_dispatch_count,
            transfer_command_count,
        )
    })
    .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_arguments)]
fn last_token_linear_replay_key(
    provider_fingerprint: &str,
    packed: bool,
    hidden_size: i32,
    out_features: i32,
    launches: &[LastTokenDenseLinearLaunch],
    gather_scratch: Option<usize>,
    gather_sources: &[usize],
    row_bytes: usize,
) -> super::vnext_replay::CudaCommandReplayKey {
    let mut replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_last_token_dense_linear")
            .boolean(packed)
            .boolean(gather_scratch.is_some())
            .i32(hidden_size)
            .i32(out_features)
            .u64(launches.len() as u64);
    if let Some(index) = gather_scratch {
        replay_key = replay_key
            .u64(index as u64)
            .u64(row_bytes as u64)
            .u64(gather_sources.len() as u64);
        for (row, source) in gather_sources.iter().enumerate() {
            replay_key = replay_key.u64(*source as u64).u64((row * row_bytes) as u64);
        }
    }
    for launch in launches {
        replay_key = replay_key
            .u64(launch.input_region as u64)
            .u64(launch.output_region as u64)
            .i32(launch.rows);
    }
    replay_key.finish()
}

fn validate_last_token_dense_linear_signature(
    input: &ResolvedValueBinding,
    weight: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    hidden_size: u64,
    out_features: u64,
    activation: ElementType,
) -> Result<(), String> {
    let contiguous = |binding: &ResolvedValueBinding| {
        matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
    };
    let input_dimensions = input.tensor().dimensions();
    if input.tensor().element_type() != activation
        || weight.tensor().element_type() != ElementType::F16
        || output.tensor().element_type() != activation
        || input_dimensions.len() != 2
        || input_dimensions[0] == 0
        || input_dimensions[1] != hidden_size
        || weight.tensor().dimensions() != [out_features, hidden_size]
        || output.tensor().dimensions() != [1, out_features]
        || !contiguous(input)
        || !contiguous(weight)
        || !contiguous(output)
    {
        return Err(
            "CUDA last-token dense-linear invocation differs from its resolved signature"
                .to_owned(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingLaunch {
    first_region: usize,
    token_count: u64,
    vocabulary_size: u32,
    hidden_size: i32,
    grid_x: u32,
}

fn encode_token_embedding(
    function: &CudaFunction,
    provider_fingerprint: &str,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.operation().id.as_str() != TOKEN_EMBEDDING_OPERATION_ID
        || invocation.participants().is_empty()
    {
        return Err("CUDA token embedding received another or empty operation".to_owned());
    }

    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA token embedding participant ranges are incomplete".to_owned());
    }
    let input_packed =
        transformer::token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let mut regions = Vec::with_capacity(invocation.participants().len() * 3);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let hidden_size = unsigned_attribute(participant.attributes(), "hidden_size")?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        validate_signature(
            token_ids,
            table,
            output,
            vocabulary_size,
            hidden_size,
            ElementType::F16,
        )?;
        let source_range = token_range.source_token_range();
        let packed_range = token_range.immediate_token_range();
        let token_count = token_range.immediate_tokens();
        let grid_x = hidden_size
            .div_ceil(THREADS_PER_BLOCK as u64)
            .try_into()
            .map_err(|_| "embedding launch grid exceeds u32".to_owned())?;

        let first_region = regions.len();
        regions.push(contiguous_region(participant, table, ElementType::F16)?);
        regions.push(contiguous_token_region(
            participant,
            token_ids,
            ElementType::U32,
            if input_packed {
                packed_range.start
            } else {
                source_range.start
            },
            token_count,
        )?);
        regions.push(contiguous_token_region(
            participant,
            output,
            ElementType::F16,
            packed_range.start,
            token_count,
        )?);
        launches.push(EmbeddingLaunch {
            first_region,
            token_count,
            vocabulary_size: vocabulary_size
                .try_into()
                .map_err(|_| "embedding vocabulary size exceeds u32".to_owned())?,
            hidden_size: hidden_size
                .try_into()
                .map_err(|_| "embedding hidden size exceeds i32".to_owned())?,
            grid_x,
        });
    }

    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "embedding participant count exceeds u32".to_owned())?;
    let token_count = invocation.work_shape().immediate_tokens();
    let compute_dispatch_count = launches
        .iter()
        .map(|launch| launch.token_count.div_ceil(MAXIMUM_TOKENS_PER_LAUNCH))
        .sum();
    let mut replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_token_embedding")
            .u64(launches.len() as u64);
    for launch in &launches {
        replay_key = replay_key
            .u64(launch.first_region as u64)
            .u64(launch.token_count)
            .u32(launch.vocabulary_size)
            .i32(launch.hidden_size)
            .u32(launch.grid_x);
    }
    let function = function.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_token_embedding",
        regions,
        replay_key.finish(),
        move |stream, regions| {
            for launch in &launches {
                let table = regions[launch.first_region].device_ptr();
                let token_ids_base = regions[launch.first_region + 1].device_ptr();
                let output_base = regions[launch.first_region + 2].device_ptr();
                let mut token_offset = 0_u64;
                while token_offset < launch.token_count {
                    let chunk_tokens =
                        (launch.token_count - token_offset).min(MAXIMUM_TOKENS_PER_LAUNCH);
                    let token_ids = checked_pointer_offset(
                        token_ids_base,
                        token_offset,
                        ElementType::U32.size_bytes(),
                        "token id",
                    )?;
                    let output_element_offset = token_offset
                        .checked_mul(launch.hidden_size as u64)
                        .ok_or_else(|| {
                            CudaDeviceRuntimeError::contract(
                                "vNext embedding output element offset overflows",
                            )
                        })?;
                    let output = checked_pointer_offset(
                        output_base,
                        output_element_offset,
                        ElementType::F16.size_bytes(),
                        "embedding output",
                    )?;
                    let batch = chunk_tokens as i32;
                    let mut builder = stream.launch_builder(&function);
                    builder.arg(&table);
                    builder.arg(&token_ids);
                    builder.arg(&output);
                    builder.arg(&batch);
                    builder.arg(&launch.hidden_size);
                    builder.arg(&launch.vocabulary_size);
                    unsafe {
                        builder.launch(LaunchConfig {
                            grid_dim: (launch.grid_x, chunk_tokens as u32, 1),
                            block_dim: (THREADS_PER_BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map_err(|error| {
                        CudaDeviceRuntimeError::driver("vNext token embedding launch", error)
                    })?;
                    token_offset += chunk_tokens;
                }
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
            participant_count,
            token_count,
            compute_dispatch_count,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

fn checked_pointer_offset(
    base: cudarc::driver::sys::CUdeviceptr,
    elements: u64,
    element_bytes: u64,
    context: &'static str,
) -> Result<cudarc::driver::sys::CUdeviceptr, CudaDeviceRuntimeError> {
    elements
        .checked_mul(element_bytes)
        .and_then(|bytes| base.checked_add(bytes))
        .ok_or_else(|| CudaDeviceRuntimeError::contract(format!("{context} pointer overflows")))
}

fn validate_signature(
    token_ids: &ResolvedValueBinding,
    table: &ResolvedValueBinding,
    output: &ResolvedValueBinding,
    vocabulary_size: u64,
    hidden_size: u64,
    activation: ElementType,
) -> Result<u64, String> {
    let token_dimensions = token_ids.tensor().dimensions();
    let table_dimensions = table.tensor().dimensions();
    let output_dimensions = output.tensor().dimensions();
    let contiguous = |binding: &ResolvedValueBinding| {
        matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
    };
    if token_ids.tensor().element_type() != ElementType::U32
        || table.tensor().element_type() != ElementType::F16
        || output.tensor().element_type() != activation
        || token_dimensions.len() != 1
        || table_dimensions != [vocabulary_size, hidden_size]
        || output_dimensions != [token_dimensions[0], hidden_size]
        || !contiguous(token_ids)
        || !contiguous(table)
        || !contiguous(output)
    {
        return Err(
            "CUDA token embedding invocation differs from its resolved signature".to_owned(),
        );
    }
    Ok(token_dimensions[0])
}

fn binding(
    bindings: &[ResolvedValueBinding],
    role: ResolvedValueRole,
    ordinal: u32,
) -> Result<&ResolvedValueBinding, String> {
    bindings
        .iter()
        .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
        .ok_or_else(|| format!("CUDA operation lacks {role:?} binding {ordinal}"))
}

fn unsigned_attribute(
    attributes: &std::collections::BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<u64, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(format!("CUDA operation lacks unsigned attribute {name:?}")),
    }
}

fn reusable_token_topology(
    request: &ReusableExecutionTopologyRequest<'_>,
    domain: &'static [u8],
) -> Result<ReusableExecutionTopology, VNextError> {
    for (role, ordinal) in [
        (ResolvedValueRole::Input, 0),
        (ResolvedValueRole::Input, 1),
        (ResolvedValueRole::Output, 0),
    ] {
        if request
            .binding_reusable_address_scope(role, ordinal)?
            .is_none()
        {
            return Ok(ReusableExecutionTopology::EagerBoundary);
        }
    }

    let bind_source_ranges =
        !request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)?;
    let ranges = request.work_shape().participant_token_ranges();
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update((ranges.len() as u64).to_le_bytes());
    digest.update(request.work_shape().immediate_tokens().to_le_bytes());
    for range in ranges {
        let source = range.source_token_range();
        let packed = range.immediate_token_range();
        digest.update(range.immediate_tokens().to_le_bytes());
        if bind_source_ranges {
            digest.update(source.start.to_le_bytes());
            digest.update(source.end.to_le_bytes());
        }
        digest.update(packed.start.to_le_bytes());
        digest.update(packed.end.to_le_bytes());
    }
    Ok(ReusableExecutionTopology::Dynamic(
        DeviceReusableExecutionTopologyFingerprint::from_sha256(digest.finalize().into()),
    ))
}

fn contiguous_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA operation requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err("CUDA operation storage element type differs from its contract".to_owned());
    }
    contiguous_region_range(
        participant,
        binding,
        element_type,
        component.offset_bytes(),
        component.length_bytes(),
    )
}

fn contiguous_token_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    token_start: u64,
    token_count: u64,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA operation requires one storage component per value".to_owned());
    };
    let projection = participant
        .work()
        .token_projection(binding.role(), binding.ordinal())
        .ok_or_else(|| "CUDA operation binding has no token work projection".to_owned())?;
    let dimensions = binding.tensor().dimensions();
    if projection.axis() != 0
        || projection.rank() as usize != dimensions.len()
        || dimensions.first() != Some(&projection.canonical_extent())
        || component.offset_bytes() != 0
        || component.length_bytes() % projection.canonical_extent() != 0
    {
        return Err(
            "CUDA contiguous token projection is not a canonical leading-axis tensor".to_owned(),
        );
    }
    let bytes_per_token = component.length_bytes() / projection.canonical_extent();
    let logical_offset = token_start
        .checked_mul(bytes_per_token)
        .ok_or_else(|| "CUDA token region offset overflows".to_owned())?;
    let logical_length = token_count
        .checked_mul(bytes_per_token)
        .ok_or_else(|| "CUDA token region length overflows".to_owned())?;
    contiguous_region_range(
        participant,
        binding,
        element_type,
        logical_offset,
        logical_length,
    )
}

fn contiguous_region_range(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    logical_offset_bytes: u64,
    logical_length_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA operation requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err("CUDA operation storage element type differs from its contract".to_owned());
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "CUDA operation value has no resource view".to_owned())?;
    // The invocation construction has already proved the resource view and
    // component describe the same allocation. The provider still translates
    // the logical slice instead of assuming an arena base pointer.
    let translated = view
        .translate(logical_offset_bytes, logical_length_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "CUDA operation translated to no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("CUDA operation requires contiguous physical storage".to_owned());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())?;
    if region.element_type() != element_type || region.length_bytes() != logical_length_bytes {
        return Err(
            "CUDA operation physical region differs from its resolved component".to_owned(),
        );
    }
    Ok(region)
}

fn same_physical_region(left: &CudaBufferRegion, right: &CudaBufferRegion) -> bool {
    left.device_ptr() == right.device_ptr()
        && left.length_bytes() == right.length_bytes()
        && left.element_type() == right.element_type()
}

fn contiguous_bindings() -> Vec<ProviderStorageBindingRequirement> {
    [
        (ResolvedValueRole::Input, 0),
        (ResolvedValueRole::Input, 1),
        (ResolvedValueRole::Output, 0),
    ]
    .into_iter()
    .map(|(role, ordinal)| {
        ProviderStorageBindingRequirement::new(
            role,
            ordinal,
            DynamicStorageRequirement::contiguous(),
        )
    })
    .collect()
}

fn provider_failure(
    identity: ferrum_runtime::ExecutionIdentityEnvelope,
    stage: &'static str,
    message: String,
) -> OperationFailure {
    let message = message.chars().take(2048).collect::<String>();
    OperationFailure::new(identity, ProfilePhase::Forward, stage, message, false)
        .expect("core-issued CUDA operation identity must form a valid provider failure")
}

fn implementation_fingerprint(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    format!("{:x}", digest.finalize())
}

fn contract_error(error: VNextError) -> CudaDeviceRuntimeError {
    CudaDeviceRuntimeError::contract(error.to_string())
}
