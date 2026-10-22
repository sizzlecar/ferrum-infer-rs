use std::collections::BTreeSet;
use std::sync::Arc;

use ferrum_runtime::{
    adjacent_rope_bf16_contract, batched_row_concat_bf16_contract, dense_linear_bf16_contract,
    dense_linear_contract, gated_residual_bf16_contract, gelu_tanh_bf16_contract,
    half_split_rope_bf16_contract, head_rms_norm_bf16_contract, layer_norm_bf16_contract,
    modulation_scale_bf16_contract, plain_residual_add_bf16_contract, row_gather_bf16_contract,
    silu_bf16_contract, silu_mul_bf16_contract, structured_attention_bf16_contract,
    structured_gqa_attention_bf16_contract, tanh_bf16_contract,
    zero_centered_rms_norm_bf16_contract, CapabilityCatalog, CapabilityId, ContractVersion,
    DeviceClass, DeviceDescriptor, DeviceId, DeviceRuntime, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageView, EngineProviderDescriptor, OperationContract,
    OperationProvider, OperationRuntimeRegistry, ProviderId, VNextError, WeightMaterializerId,
    WeightMaterializerRegistry, ADJACENT_ROPE_BF16_CAPABILITY_ID,
    BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID, DENSE_LINEAR_BF16_CAPABILITY_ID,
    DENSE_LINEAR_F16_CAPABILITY_ID, GATED_RESIDUAL_BF16_CAPABILITY_ID,
    GELU_TANH_BF16_CAPABILITY_ID, HALF_SPLIT_ROPE_BF16_CAPABILITY_ID,
    HEAD_RMS_NORM_BF16_CAPABILITY_ID, IDENTITY_WEIGHT_MATERIALIZER_ID,
    LAYER_NORM_BF16_CAPABILITY_ID, MODULATION_SCALE_BF16_CAPABILITY_ID,
    PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID, ROW_GATHER_BF16_CAPABILITY_ID, SILU_BF16_CAPABILITY_ID,
    SILU_MUL_BF16_CAPABILITY_ID, STRUCTURED_ATTENTION_BF16_CAPABILITY_ID,
    STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID, TANH_BF16_CAPABILITY_ID,
    ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID,
};

use super::adjacent_rope::ReferenceAdjacentRopeProvider;
use super::dense_linear::{implementation_fingerprint, ReferenceDenseLinearProvider};
use super::gelu_tanh::ReferenceGeluTanhProvider;
use super::half_split_rope::ReferenceHalfSplitRopeProvider;
use super::head_rms_norm::ReferenceHeadRmsNormProvider;
use super::layer_norm::ReferenceLayerNormProvider;
use super::plain_residual_add::ReferencePlainResidualAddProvider;
use super::pointwise::{ReferencePointwiseKind, ReferencePointwiseProvider};
use super::row_transport::{ReferenceRowTransportKind, ReferenceRowTransportProvider};
use super::runtime::{
    ReferenceDeviceRuntime, ReferenceDeviceRuntimeConfig, ReferenceDeviceRuntimeError,
};
use super::spatial_bf16::{ReferenceSpatialProvider, SpatialKind};
use super::spatial_decoder::{DecoderKind, ReferenceSpatialDecoderProvider};
use super::structured_attention::ReferenceStructuredAttentionProvider;
use super::structured_gqa_attention::ReferenceStructuredGqaAttentionProvider;
use super::transcendental::{
    numerical_implementation_fingerprint, ReferenceTranscendentalKind,
    ReferenceTranscendentalProvider,
};
use super::zero_centered_rms_norm::ReferenceZeroCenteredRmsNormProvider;
use ferrum_runtime::{
    channel_l2_norm_bf16_contract, conv2d_bf16_contract, CHANNEL_L2_NORM_BF16_CAPABILITY_ID,
    CONV2D_BF16_CAPABILITY_ID,
};
use ferrum_runtime::{
    channel_l2_norm_image_bf16_contract, spatial_attention_bf16_contract,
    spatial_upsample_gather_bf16_contract, CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID,
    SPATIAL_ATTENTION_BF16_CAPABILITY_ID, SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID,
};
use ferrum_runtime::{
    plain_residual_add_nchw_bf16_contract, silu_nchw_bf16_contract,
    PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID, SILU_NCHW_BF16_CAPABILITY_ID,
};

const REFERENCE_ENGINE_PROVIDER_ID: &str = "provider.engine.reference.vnext";
pub const REFERENCE_DENSE_SAFETENSORS_FORMAT_ID: &str = "weight-format.safetensors.dense";
const REFERENCE_MEMORY_BYTES: u64 = 64 * 1024 * 1024;

pub fn reference_vnext_capabilities() -> Result<BTreeSet<CapabilityId>, VNextError> {
    Ok(BTreeSet::from([
        CapabilityId::new(CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID)?,
        CapabilityId::new(SPATIAL_ATTENTION_BF16_CAPABILITY_ID)?,
        CapabilityId::new(SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID)?,
        CapabilityId::new(CONV2D_BF16_CAPABILITY_ID)?,
        CapabilityId::new(CHANNEL_L2_NORM_BF16_CAPABILITY_ID)?,
        CapabilityId::new(SILU_NCHW_BF16_CAPABILITY_ID)?,
        CapabilityId::new(PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID)?,
        CapabilityId::new(PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID)?,
        CapabilityId::new(HALF_SPLIT_ROPE_BF16_CAPABILITY_ID)?,
        CapabilityId::new(STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID)?,
        CapabilityId::new(DENSE_LINEAR_F16_CAPABILITY_ID)?,
        CapabilityId::new(DENSE_LINEAR_BF16_CAPABILITY_ID)?,
        CapabilityId::new(LAYER_NORM_BF16_CAPABILITY_ID)?,
        CapabilityId::new(HEAD_RMS_NORM_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ROW_GATHER_BF16_CAPABILITY_ID)?,
        CapabilityId::new(BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID)?,
        CapabilityId::new(GELU_TANH_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ADJACENT_ROPE_BF16_CAPABILITY_ID)?,
        CapabilityId::new(STRUCTURED_ATTENTION_BF16_CAPABILITY_ID)?,
        CapabilityId::new(MODULATION_SCALE_BF16_CAPABILITY_ID)?,
        CapabilityId::new(GATED_RESIDUAL_BF16_CAPABILITY_ID)?,
        CapabilityId::new(SILU_BF16_CAPABILITY_ID)?,
        CapabilityId::new(SILU_MUL_BF16_CAPABILITY_ID)?,
        CapabilityId::new(TANH_BF16_CAPABILITY_ID)?,
    ]))
}

pub(super) fn reference_vnext_runtime_config(
    device_id: DeviceId,
) -> Result<ReferenceDeviceRuntimeConfig, VNextError> {
    reference_vnext_runtime_config_with_capacity(device_id, REFERENCE_MEMORY_BYTES)
}

fn reference_vnext_runtime_config_with_capacity(
    device_id: DeviceId,
    capacity_bytes: u64,
) -> Result<ReferenceDeviceRuntimeConfig, VNextError> {
    if capacity_bytes == 0 || u128::from(capacity_bytes) > isize::MAX as u128 {
        return Err(VNextError::InvalidExecutionPlan {
            reason: "reference memory capacity must be positive and fit isize".into(),
        });
    }
    let numerical = numerical_implementation_fingerprint();
    let descriptor = DeviceDescriptor {
        id: device_id,
        class: DeviceClass::Reference,
        ordinal: 0,
        total_memory_bytes: capacity_bytes,
        runtime_implementation_fingerprint: implementation_fingerprint(&[
            &capacity_bytes.to_le_bytes(),
            include_str!("runtime.rs").as_bytes(),
            include_str!("spatial_decoder.rs").as_bytes(),
            include_str!("runtime/spatial_decoder.rs").as_bytes(),
            include_str!("spatial_bf16.rs").as_bytes(),
            include_str!("runtime/spatial_bf16.rs").as_bytes(),
            include_str!("plain_residual_add.rs").as_bytes(),
            include_str!("runtime/plain_residual_add.rs").as_bytes(),
            include_str!("half_split_rope.rs").as_bytes(),
            include_str!("runtime/half_split_rope.rs").as_bytes(),
            include_str!("structured_gqa_attention.rs").as_bytes(),
            include_str!("runtime/structured_gqa_attention.rs").as_bytes(),
            include_str!("dense_linear.rs").as_bytes(),
            include_str!("layer_norm.rs").as_bytes(),
            include_str!("runtime/layer_norm.rs").as_bytes(),
            include_str!("zero_centered_rms_norm.rs").as_bytes(),
            include_str!("runtime/zero_centered_rms_norm.rs").as_bytes(),
            include_str!("gelu_tanh.rs").as_bytes(),
            include_str!("runtime/gelu_tanh.rs").as_bytes(),
            include_str!("row_transport.rs").as_bytes(),
            include_str!("runtime/row_transport.rs").as_bytes(),
            include_str!("head_rms_norm.rs").as_bytes(),
            include_str!("runtime/head_rms_norm.rs").as_bytes(),
            include_str!("adjacent_rope.rs").as_bytes(),
            include_str!("runtime/adjacent_rope.rs").as_bytes(),
            include_str!("structured_attention.rs").as_bytes(),
            include_str!("runtime/structured_attention.rs").as_bytes(),
            include_str!("pointwise.rs").as_bytes(),
            include_str!("runtime/pointwise.rs").as_bytes(),
            numerical.as_bytes(),
            include_str!("transcendental/provider.rs").as_bytes(),
            include_str!("composition.rs").as_bytes(),
        ]),
        capabilities: reference_vnext_capabilities()?,
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )?]),
    };
    descriptor.validate()?;
    Ok(ReferenceDeviceRuntimeConfig { descriptor })
}

pub fn reference_vnext_operation_registry(
    runtime: &ReferenceDeviceRuntime,
) -> Result<OperationRuntimeRegistry<ReferenceDeviceRuntime>, ReferenceDeviceRuntimeError> {
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(channel_l2_norm_image_bf16_contract().map_err(contract_error)?),
        Box::new(spatial_attention_bf16_contract().map_err(contract_error)?),
        Box::new(spatial_upsample_gather_bf16_contract().map_err(contract_error)?),
        Box::new(conv2d_bf16_contract().map_err(contract_error)?),
        Box::new(channel_l2_norm_bf16_contract().map_err(contract_error)?),
        Box::new(silu_nchw_bf16_contract().map_err(contract_error)?),
        Box::new(plain_residual_add_nchw_bf16_contract().map_err(contract_error)?),
        Box::new(plain_residual_add_bf16_contract().map_err(contract_error)?),
        Box::new(half_split_rope_bf16_contract().map_err(contract_error)?),
        Box::new(structured_gqa_attention_bf16_contract().map_err(contract_error)?),
        Box::new(dense_linear_contract().map_err(contract_error)?),
        Box::new(dense_linear_bf16_contract().map_err(contract_error)?),
        Box::new(layer_norm_bf16_contract().map_err(contract_error)?),
        Box::new(head_rms_norm_bf16_contract().map_err(contract_error)?),
        Box::new(row_gather_bf16_contract().map_err(contract_error)?),
        Box::new(batched_row_concat_bf16_contract().map_err(contract_error)?),
        Box::new(zero_centered_rms_norm_bf16_contract().map_err(contract_error)?),
        Box::new(gelu_tanh_bf16_contract().map_err(contract_error)?),
        Box::new(adjacent_rope_bf16_contract().map_err(contract_error)?),
        Box::new(structured_attention_bf16_contract().map_err(contract_error)?),
        Box::new(modulation_scale_bf16_contract().map_err(contract_error)?),
        Box::new(gated_residual_bf16_contract().map_err(contract_error)?),
        Box::new(silu_bf16_contract().map_err(contract_error)?),
        Box::new(silu_mul_bf16_contract().map_err(contract_error)?),
        Box::new(tanh_bf16_contract().map_err(contract_error)?),
    ];
    let providers = vec![
        reference_vnext_channel_l2_norm_image_bf16_provider(runtime)?,
        reference_vnext_spatial_attention_bf16_provider(runtime)?,
        reference_vnext_spatial_upsample_gather_bf16_provider(runtime)?,
        reference_vnext_conv2d_bf16_provider(runtime)?,
        reference_vnext_channel_l2_norm_bf16_provider(runtime)?,
        reference_vnext_silu_nchw_bf16_provider(runtime)?,
        reference_vnext_plain_residual_add_nchw_bf16_provider(runtime)?,
        reference_vnext_plain_residual_add_bf16_provider(runtime)?,
        reference_vnext_half_split_rope_bf16_provider(runtime)?,
        reference_vnext_structured_gqa_attention_bf16_provider(runtime)?,
        reference_vnext_dense_linear_provider(runtime)?,
        reference_vnext_dense_linear_bf16_provider(runtime)?,
        reference_vnext_layer_norm_bf16_provider(runtime)?,
        reference_vnext_head_rms_norm_bf16_provider(runtime)?,
        reference_vnext_row_gather_bf16_provider(runtime)?,
        reference_vnext_batched_row_concat_bf16_provider(runtime)?,
        reference_vnext_zero_centered_rms_norm_bf16_provider(runtime)?,
        reference_vnext_gelu_tanh_bf16_provider(runtime)?,
        reference_vnext_adjacent_rope_bf16_provider(runtime)?,
        reference_vnext_structured_attention_bf16_provider(runtime)?,
        reference_vnext_modulation_scale_bf16_provider(runtime)?,
        reference_vnext_gated_residual_bf16_provider(runtime)?,
        reference_vnext_silu_bf16_provider(runtime)?,
        reference_vnext_silu_mul_bf16_provider(runtime)?,
        reference_vnext_tanh_bf16_provider(runtime)?,
    ];
    OperationRuntimeRegistry::new(contracts, providers).map_err(contract_error)
}

/// Builds the actual reference dense implementation for an explicitly
/// assembled registry, including runtime wrappers used for fault injection.
/// Core still creates the provider's invocation and submission authority.
pub fn reference_vnext_channel_l2_norm_image_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceSpatialDecoderProvider::new(
        runtime,
        DecoderKind::Norm,
    )?))
}
pub fn reference_vnext_spatial_attention_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceSpatialDecoderProvider::new(
        runtime,
        DecoderKind::Attention,
    )?))
}
pub fn reference_vnext_spatial_upsample_gather_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceSpatialDecoderProvider::new(
        runtime,
        DecoderKind::Upsample,
    )?))
}
pub fn reference_vnext_dense_linear_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceDenseLinearProvider::new(runtime)?))
}

/// Builds the independent BF16-storage, FP32-arithmetic dense provider.
pub fn reference_vnext_dense_linear_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceDenseLinearProvider::new_bf16(runtime)?))
}

/// Builds the fixed-tensor BF16 population LayerNorm implementation without
/// affine parameters. Token invocations are rejected explicitly.
pub fn reference_vnext_layer_norm_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceLayerNormProvider::new(runtime)?))
}

/// Builds head-local BF16 RMSNorm with FP32 pairwise reduction and two BF16 boundaries.
pub fn reference_vnext_conv2d_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceSpatialProvider::new(
        runtime,
        SpatialKind::Conv2d,
    )?))
}
pub fn reference_vnext_silu_nchw_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceSpatialProvider::new(
        runtime,
        SpatialKind::SiluNchw,
    )?))
}
pub fn reference_vnext_plain_residual_add_nchw_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceSpatialProvider::new(
        runtime,
        SpatialKind::ResidualAddNchw,
    )?))
}
pub fn reference_vnext_channel_l2_norm_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceSpatialProvider::new(
        runtime,
        SpatialKind::ChannelL2Norm,
    )?))
}
pub fn reference_vnext_head_rms_norm_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceHeadRmsNormProvider::new(runtime)?))
}

/// Builds whole-row RMSNorm with FP32 zero-centered affine and one final BF16 cast.
pub fn reference_vnext_zero_centered_rms_norm_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceZeroCenteredRmsNormProvider::new(
        runtime,
    )?))
}

/// Builds unary GELU(tanh) with explicit FP32 arithmetic and its own profile.
pub fn reference_vnext_gelu_tanh_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceGeluTanhProvider::new(runtime)?))
}

/// Builds exact BF16 row/window gather with explicitly bounded metadata.
pub fn reference_vnext_row_gather_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceRowTransportProvider::new(
        runtime,
        ReferenceRowTransportKind::Gather,
    )?))
}
/// Builds exact BF16 concatenation of left/right rows separately per sample.
pub fn reference_vnext_batched_row_concat_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceRowTransportProvider::new(
        runtime,
        ReferenceRowTransportKind::Concat,
    )?))
}

/// Builds the independent fixed-tensor BF16 plain residual add provider.
pub fn reference_vnext_plain_residual_add_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferencePlainResidualAddProvider::new(runtime)?))
}

/// Builds the independent fixed-tensor BF16 half split rope provider.
pub fn reference_vnext_half_split_rope_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceHalfSplitRopeProvider::new(runtime)?))
}

/// Builds the independent fixed-tensor BF16 structured gqa attention provider.
pub fn reference_vnext_structured_gqa_attention_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceStructuredGqaAttentionProvider::new(
        runtime,
    )?))
}

/// Builds nonpaged structured attention with explicit mask and FP32 online softmax.
pub fn reference_vnext_structured_attention_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceStructuredAttentionProvider::new(
        runtime,
    )?))
}

/// Builds adjacent-pair BF16 RoPE with explicit FP32 multiplication and FMA.
pub fn reference_vnext_adjacent_rope_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceAdjacentRopeProvider::new(runtime)?))
}

/// Builds BF16 scale-only modulation with explicit BF16 rounding of 1 + scale.
pub fn reference_vnext_modulation_scale_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferencePointwiseProvider::new(
        runtime,
        ReferencePointwiseKind::ModulationScale,
    )?))
}

/// Builds BF16 gated residual with an already-activated gate and an explicit
/// BF16 rounding boundary between multiplication and residual addition.
pub fn reference_vnext_gated_residual_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferencePointwiseProvider::new(
        runtime,
        ReferencePointwiseKind::GatedResidual,
    )?))
}

/// Builds unary SiLU under the fixed Reference FP32/BF16 numerical profile.
pub fn reference_vnext_silu_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceTranscendentalProvider::new(
        runtime,
        ReferenceTranscendentalKind::Silu,
    )?))
}

/// Builds SiLU-mul with a BF16 activation boundary before the FP32 multiply.
pub fn reference_vnext_silu_mul_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceTranscendentalProvider::new(
        runtime,
        ReferenceTranscendentalKind::SiluMul,
    )?))
}

/// Builds tanh under the fixed Reference libm/BF16 numerical profile.
pub fn reference_vnext_tanh_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceTranscendentalProvider::new(
        runtime,
        ReferenceTranscendentalKind::Tanh,
    )?))
}

/// Single composition authority for reference planning, allocation, weight
/// initialization, provider binding, and execution.
pub struct ReferenceVNextComposition {
    runtime: Arc<ReferenceDeviceRuntime>,
    registry: OperationRuntimeRegistry<ReferenceDeviceRuntime>,
    weight_materializers: WeightMaterializerRegistry,
    weight_materializer_id: WeightMaterializerId,
    catalog: CapabilityCatalog,
}

impl ReferenceVNextComposition {
    pub fn create(device_id: DeviceId) -> Result<Self, ReferenceDeviceRuntimeError> {
        Self::create_with_capacity(device_id, REFERENCE_MEMORY_BYTES)
    }

    /// Creates a Reference device with an explicit managed-memory ceiling.
    /// This does not allocate that capacity. Subsequent plans use the existing
    /// shared device ledger; the default constructor remains 64 MiB.
    pub fn create_with_capacity(
        device_id: DeviceId,
        capacity_bytes: u64,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let config = reference_vnext_runtime_config_with_capacity(device_id, capacity_bytes)
            .map_err(contract_error)?;
        let runtime = Arc::new(ReferenceDeviceRuntime::new(config)?);
        let registry = reference_vnext_operation_registry(&runtime)?;
        let weight_materializers =
            WeightMaterializerRegistry::identity_only().map_err(contract_error)?;
        let weight_materializer_id =
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID).map_err(contract_error)?;
        let engine = EngineProviderDescriptor::new(
            ProviderId::new(REFERENCE_ENGINE_PROVIDER_ID).map_err(contract_error)?,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[
                include_str!("composition.rs").as_bytes(),
                REFERENCE_ENGINE_PROVIDER_ID.as_bytes(),
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
            weight_materializer_id,
            catalog,
        })
    }

    pub fn runtime(&self) -> &Arc<ReferenceDeviceRuntime> {
        &self.runtime
    }

    pub fn registry(&self) -> &OperationRuntimeRegistry<ReferenceDeviceRuntime> {
        &self.registry
    }

    pub fn catalog(&self) -> &CapabilityCatalog {
        &self.catalog
    }

    pub fn weight_materializers(&self) -> &WeightMaterializerRegistry {
        &self.weight_materializers
    }

    pub fn weight_materializer_id(&self) -> &WeightMaterializerId {
        &self.weight_materializer_id
    }
}

fn contract_error(error: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(error.to_string())
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    use ferrum_runtime::{DeviceCapacityOwner, DeviceOpaqueMemoryKind};

    #[test]
    fn explicit_capacity_is_checked_and_changes_descriptor_identity_without_allocating() {
        let id = || DeviceId::new("device.reference.explicit-capacity").unwrap();
        assert!(ReferenceVNextComposition::create_with_capacity(id(), 0).is_err());
        assert!(
            ReferenceVNextComposition::create_with_capacity(id(), isize::MAX as u64 + 1).is_err()
        );
        let default = ReferenceVNextComposition::create(id()).unwrap();
        assert_eq!(
            default.runtime().descriptor().total_memory_bytes,
            64 * 1024 * 1024
        );
        let large =
            ReferenceVNextComposition::create_with_capacity(id(), 1024 * 1024 * 1024).unwrap();
        assert_eq!(
            large.runtime().descriptor().total_memory_bytes,
            1024 * 1024 * 1024
        );
        assert_ne!(
            default
                .runtime()
                .descriptor()
                .runtime_implementation_fingerprint,
            large
                .runtime()
                .descriptor()
                .runtime_implementation_fingerprint
        );
        assert_eq!(large.runtime().snapshot().allocations, 0);
        let maximum = ReferenceVNextComposition::create_with_capacity(
            DeviceId::new("device.reference.maximum-capacity").unwrap(),
            isize::MAX as u64,
        )
        .unwrap();
        assert_eq!(maximum.runtime().snapshot().allocations, 0);
    }

    #[test]
    fn configured_capacity_uses_shared_account_and_reclaims_only_the_last_owner() {
        let composition = ReferenceVNextComposition::create_with_capacity(
            DeviceId::new("device.reference.shared-explicit-capacity").unwrap(),
            1024,
        )
        .unwrap();
        let descriptor = composition.runtime().descriptor();
        let first = DeviceCapacityOwner::open(descriptor, 1024).unwrap();
        let second = DeviceCapacityOwner::open(descriptor, 1024).unwrap();
        let claim = first
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1000)
            .unwrap();
        let retained = claim.clone();
        drop(claim);
        assert!(second.reserve(DeviceOpaqueMemoryKind::Stream, 25).is_err());
        let remainder = second.reserve(DeviceOpaqueMemoryKind::Stream, 24).unwrap();
        drop(retained);
        let reclaimed = second
            .reserve(DeviceOpaqueMemoryKind::Stream, 1000)
            .unwrap();
        assert_eq!(reclaimed.bytes() + remainder.bytes(), 1024);
        assert_eq!(composition.runtime().snapshot().allocations, 0);
    }
}
