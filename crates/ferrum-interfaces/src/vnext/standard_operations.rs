use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use super::{
    AliasPolicy, AttributeConstraint, AttributeId, AttributeSchema, AttributeSpec,
    AttributeValueKind, CanonicalRational, CapabilityId, ContractVersion, DimensionConstraint,
    ElementType, LayoutConstraint, OperationContract, OperationDescriptor, OperationId, OracleSpec,
    ProfilePhase, ProviderRequirement, ResourcePresenceRequirement, ResourceRequirements,
    TensorAccess, TensorContract, VNextError,
};

pub const TOKEN_EMBEDDING_OPERATION_ID: &str = "operation.token_embedding";
pub const TOKEN_EMBEDDING_F16_CAPABILITY_ID: &str = "capability.operation.token_embedding.f16";
pub const TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID: &str = "operation.token_embedding.f32-master";
pub const TOKEN_EMBEDDING_F32_MASTER_CAPABILITY_ID: &str =
    "capability.operation.token_embedding.f32-master";
pub const LAST_TOKEN_DENSE_LINEAR_OPERATION_ID: &str = "operation.last_token_dense_linear";
pub const LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID: &str =
    "capability.operation.last_token_dense_linear.f16";
pub const LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID: &str = "operation.last_token_dense_linear.f32";
pub const LAST_TOKEN_DENSE_LINEAR_F32_CAPABILITY_ID: &str =
    "capability.operation.last_token_dense_linear.f32";
pub const LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID: &str = "operation.last_token_masked_argmax";
pub const LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID: &str =
    "capability.operation.last_token_masked_argmax.f16";
pub const LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID: &str =
    "operation.last_token_masked_argmax.f32";
pub const LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID: &str =
    "capability.operation.last_token_masked_argmax.f32";
pub const RMS_NORM_OPERATION_ID: &str = "operation.rms_norm";
pub const RMS_NORM_F16_CAPABILITY_ID: &str = "capability.operation.rms_norm.f16";
pub const RMS_NORM_F32_TO_F16_OPERATION_ID: &str = "operation.rms_norm.f32-to-f16";
pub const RMS_NORM_F32_TO_F16_CAPABILITY_ID: &str = "capability.operation.rms_norm.f32-to-f16";
pub const RMS_NORM_F32_OPERATION_ID: &str = "operation.rms_norm.f32";
pub const RMS_NORM_F32_CAPABILITY_ID: &str = "capability.operation.rms_norm.f32";
pub const DENSE_LINEAR_OPERATION_ID: &str = "operation.dense_linear";
pub const DENSE_LINEAR_F16_CAPABILITY_ID: &str = "capability.operation.dense_linear.f16";
pub const DENSE_SWIGLU_OPERATION_ID: &str = "operation.dense_swiglu";
pub const DENSE_SWIGLU_F16_CAPABILITY_ID: &str = "capability.operation.dense_swiglu.f16";
pub const DENSE_GEGLU_TANH_OPERATION_ID: &str = "operation.dense_geglu_tanh";
pub const DENSE_GEGLU_TANH_F16_CAPABILITY_ID: &str = "capability.operation.dense_geglu_tanh.f16";
pub const CONSTANT_SCALE_OPERATION_ID: &str = "operation.constant_scale";
pub const CONSTANT_SCALE_F16_CAPABILITY_ID: &str = "capability.operation.constant_scale.f16";
pub const LOGIT_SOFTCAP_OPERATION_ID: &str = "operation.logit_softcap";
pub const LOGIT_SOFTCAP_F16_CAPABILITY_ID: &str = "capability.operation.logit_softcap.f16";
pub const ROUTED_SWIGLU_MOE_OPERATION_ID: &str = "operation.routed_swiglu_moe";
pub const ROUTED_SWIGLU_MOE_F16_CAPABILITY_ID: &str = "capability.operation.routed_swiglu_moe.f16";
pub const ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID: &str = "operation.routed_shared_swiglu_moe";
pub const ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID: &str =
    "capability.operation.routed_shared_swiglu_moe.f16";
pub const RESIDUAL_ADD_OPERATION_ID: &str = "operation.residual_add";
pub const RESIDUAL_ADD_F16_CAPABILITY_ID: &str = "capability.operation.residual_add.f16";
pub const RESIDUAL_ADD_F32_F16_OPERATION_ID: &str = "operation.residual_add.f32-f16";
pub const RESIDUAL_ADD_F32_F16_CAPABILITY_ID: &str = "capability.operation.residual_add.f32-f16";
pub const GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID: &str =
    "operation.gated_delta_recurrent_attention";
pub const GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID: &str =
    "capability.operation.gated_delta_recurrent_attention.f16";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID: &str =
    "operation.gated_delta_recurrent_attention.f32-master";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID: &str =
    "capability.operation.gated_delta_recurrent_attention.f32-master";
pub const GATED_DELTA_EXECUTION_FORM_SELECTOR_VERSION: &str =
    "gated-delta-execution-form-selector-v1";
pub const CAUSAL_PAGED_ATTENTION_OPERATION_ID: &str = "operation.causal_paged_attention";
pub const CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID: &str =
    "capability.operation.causal_paged_attention.f16";
pub const HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_OPERATION_ID: &str =
    "operation.hybrid_vnorm_causal_paged_attention";
pub const HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID: &str =
    "capability.operation.hybrid_vnorm_causal_paged_attention.f16";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID: &str =
    "operation.causal_paged_attention.f32-master";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_CAPABILITY_ID: &str =
    "capability.operation.causal_paged_attention.f32-master";
pub const GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID: &str =
    "operation.gpt_oss.causal_paged_attention";
pub const GPT_OSS_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID: &str =
    "capability.operation.gpt_oss.causal_paged_attention.f16";
pub const GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID: &str =
    "operation.gpt_oss.routed_clamped_swiglu_moe";
pub const GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID: &str =
    "capability.operation.gpt_oss.routed_clamped_swiglu_moe.mxfp4_bf16";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedDeltaDecayParameterization {
    LogRate,
    NegativeRate,
}

impl GatedDeltaDecayParameterization {
    pub const ALL: [Self; 2] = [Self::LogRate, Self::NegativeRate];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LogRate => "log_rate",
            Self::NegativeRate => "negative_rate",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedDeltaValueHeadMapping {
    GroupedByKeyHead,
    InterleavedByKeyHead,
}

impl GatedDeltaValueHeadMapping {
    pub const ALL: [Self; 2] = [Self::GroupedByKeyHead, Self::InterleavedByKeyHead];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GroupedByKeyHead => "grouped_by_key_head",
            Self::InterleavedByKeyHead => "interleaved_by_key_head",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == value)
    }
}

/// Physical gated-delta implementation available to one provider for an
/// already-compatible operation shape. This capability is deliberately not a
/// model attribute: the same immutable model plan can select a different form
/// as the request work shape changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaExecutionCapabilities {
    chunked_scan: Option<GatedDeltaChunkedScanCapability>,
}

impl GatedDeltaExecutionCapabilities {
    pub const fn recurrent_only() -> Self {
        Self { chunked_scan: None }
    }

    pub fn with_chunked_scan(chunk_size: u32) -> Result<Self, VNextError> {
        Ok(Self {
            chunked_scan: Some(GatedDeltaChunkedScanCapability::new(chunk_size)?),
        })
    }

    pub const fn chunked_scan(self) -> Option<GatedDeltaChunkedScanCapability> {
        self.chunked_scan
    }

    /// Selects a physical form for one participant. Providers must first
    /// remove capabilities that do not support the resolved dtype or shape.
    pub fn select(
        self,
        token_count: u64,
        preference: GatedDeltaExecutionPreference,
    ) -> Result<GatedDeltaExecutionForm, VNextError> {
        if token_count == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "gated-delta execution requires at least one token".to_owned(),
            });
        }
        match (preference, self.chunked_scan, token_count) {
            (GatedDeltaExecutionPreference::ChunkedScan, Some(capability), 2..) => {
                Ok(GatedDeltaExecutionForm::ChunkedScan(
                    GatedDeltaChunkPlan::new(token_count, capability.chunk_size),
                ))
            }
            _ => Ok(GatedDeltaExecutionForm::RecurrentScan),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaChunkedScanCapability {
    chunk_size: NonZeroU32,
}

impl GatedDeltaChunkedScanCapability {
    fn new(chunk_size: u32) -> Result<Self, VNextError> {
        let chunk_size =
            NonZeroU32::new(chunk_size).ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: "gated-delta chunk size must be positive".to_owned(),
            })?;
        Ok(Self { chunk_size })
    }

    pub const fn chunk_size(self) -> u32 {
        self.chunk_size.get()
    }
}

/// Cost-model preference kept separate from physical support. A provider may
/// derive it from calibrated crossover data and live batch topology without
/// changing the immutable model plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedDeltaExecutionPreference {
    RecurrentScan,
    ChunkedScan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaChunkPlan {
    token_count: u64,
    chunk_size: NonZeroU32,
    chunk_count: u64,
    final_chunk_tokens: u32,
}

impl GatedDeltaChunkPlan {
    fn new(token_count: u64, chunk_size: NonZeroU32) -> Self {
        debug_assert!(token_count > 0);
        let chunk_size_u64 = u64::from(chunk_size.get());
        let chunk_count = ((token_count - 1) / chunk_size_u64) + 1;
        let remainder = (token_count % chunk_size_u64) as u32;
        Self {
            token_count,
            chunk_size,
            chunk_count,
            final_chunk_tokens: if remainder == 0 {
                chunk_size.get()
            } else {
                remainder
            },
        }
    }

    pub const fn token_count(self) -> u64 {
        self.token_count
    }

    pub const fn chunk_size(self) -> u32 {
        self.chunk_size.get()
    }

    pub const fn chunk_count(self) -> u64 {
        self.chunk_count
    }

    pub const fn final_chunk_tokens(self) -> u32 {
        self.final_chunk_tokens
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedDeltaExecutionForm {
    RecurrentScan,
    ChunkedScan(GatedDeltaChunkPlan),
}

impl GatedDeltaExecutionForm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RecurrentScan => "recurrent_scan",
            Self::ChunkedScan(_) => "chunked_scan",
        }
    }
}

/// One checked-in standard operation contract. Construction stays private so
/// production registries cannot mutate a descriptor after a provider binds its
/// fingerprint.
pub struct StandardOperationContract {
    descriptor: OperationDescriptor,
}

impl OperationContract for StandardOperationContract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.descriptor
    }

    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        if inputs != self.descriptor.inputs || outputs != self.descriptor.outputs {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "operation `{}` signature differs from its standard contract",
                    self.descriptor.id
                ),
            });
        }
        Ok(())
    }
}

pub fn token_embedding_contract() -> Result<StandardOperationContract, VNextError> {
    token_embedding_contract_with_output(
        TOKEN_EMBEDDING_OPERATION_ID,
        TOKEN_EMBEDDING_F16_CAPABILITY_ID,
        ElementType::F16,
    )
}

pub fn token_embedding_f32_master_contract() -> Result<StandardOperationContract, VNextError> {
    token_embedding_contract_with_output(
        TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
        TOKEN_EMBEDDING_F32_MASTER_CAPABILITY_ID,
        ElementType::F32,
    )
}

fn token_embedding_contract_with_output(
    operation_id: &str,
    capability_id: &str,
    output_type: ElementType,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![DimensionConstraint::Symbol("tokens".to_owned())],
                [ElementType::U32],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    DimensionConstraint::Symbol("vocab_size".to_owned()),
                    DimensionConstraint::Symbol("hidden_size".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![
                DimensionConstraint::Symbol("tokens".to_owned()),
                DimensionConstraint::Symbol("hidden_size".to_owned()),
            ],
            [output_type],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("vocab_size")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Forbidden,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: OracleSpec::Exact,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::from([CapabilityId::new(capability_id)?]),
        },
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Projects only the final row of a non-empty token-major tensor. Keeping this
/// semantic fusion explicit prevents materializing prompt-length vocabulary
/// logits while leaving providers free to use a pointer offset, row gather,
/// or a fused kernel.
pub fn last_token_dense_linear_contract() -> Result<StandardOperationContract, VNextError> {
    last_token_dense_linear_contract_with_activation(
        LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
        ContractVersion::new(1, 1),
        LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
        ElementType::F16,
    )
}

pub fn last_token_dense_linear_f32_contract() -> Result<StandardOperationContract, VNextError> {
    last_token_dense_linear_contract_with_activation(
        LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
        ContractVersion::new(1, 0),
        LAST_TOKEN_DENSE_LINEAR_F32_CAPABILITY_ID,
        ElementType::F32,
    )
}

fn last_token_dense_linear_contract_with_activation(
    operation_id: &str,
    version: ContractVersion,
    capability_id: &str,
    activation_type: ElementType,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version,
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [activation_type],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    DimensionConstraint::Symbol("out_features".to_owned()),
                    DimensionConstraint::Symbol("hidden_size".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![
                DimensionConstraint::Exact(1),
                DimensionConstraint::Symbol("out_features".to_owned()),
            ],
            [activation_type],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("out_features")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Optional,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(capability_id, version)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Selects one token from a final-position F16 logits row after applying an
/// exact per-vocabulary validity mask and an optional sparse repetition
/// penalty. Selection policy is carried by typed inputs so it remains visible
/// to planning and cannot be hidden in backend flags. Semantic logits remain
/// immutable; providers use invocation-scoped scratch for any penalized view.
///
/// The repetition token ids are unique and occupy
/// `offsets[0]..offsets[1]` within the fixed-capacity input. A penalty of `1.0`
/// or an empty range leaves logits unchanged.
pub fn last_token_masked_argmax_contract() -> Result<StandardOperationContract, VNextError> {
    last_token_masked_argmax_contract_with_logits(
        LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
        ContractVersion::new(3, 0),
        LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID,
        ElementType::F16,
    )
}

pub fn last_token_masked_argmax_f32_contract() -> Result<StandardOperationContract, VNextError> {
    last_token_masked_argmax_contract_with_logits(
        LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
        ContractVersion::new(1, 0),
        LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID,
        ElementType::F32,
    )
}

fn last_token_masked_argmax_contract_with_logits(
    operation_id: &str,
    version: ContractVersion,
    capability_id: &str,
    logits_type: ElementType,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version,
        inputs: vec![
            contiguous_tensor(
                vec![
                    DimensionConstraint::Exact(1),
                    DimensionConstraint::Symbol("vocab_size".to_owned()),
                ],
                [logits_type],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![DimensionConstraint::Symbol("vocab_size".to_owned())],
                [ElementType::U8],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![DimensionConstraint::Symbol(
                    "repetition_capacity".to_owned(),
                )],
                [ElementType::U32],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![DimensionConstraint::Exact(2)],
                [ElementType::U32],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![DimensionConstraint::Exact(1)],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![DimensionConstraint::Exact(1)],
            [ElementType::U32],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([unsigned_attribute("vocab_size")?]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: OracleSpec::Exact,
        provider: provider_requirement(capability_id, version)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn rms_norm_contract() -> Result<StandardOperationContract, VNextError> {
    rms_norm_contract_with_types(
        RMS_NORM_OPERATION_ID,
        RMS_NORM_F16_CAPABILITY_ID,
        ElementType::F16,
        ElementType::F16,
    )
}

pub fn rms_norm_f32_to_f16_contract() -> Result<StandardOperationContract, VNextError> {
    rms_norm_contract_with_types(
        RMS_NORM_F32_TO_F16_OPERATION_ID,
        RMS_NORM_F32_TO_F16_CAPABILITY_ID,
        ElementType::F32,
        ElementType::F16,
    )
}

pub fn rms_norm_f32_contract() -> Result<StandardOperationContract, VNextError> {
    rms_norm_contract_with_types(
        RMS_NORM_F32_OPERATION_ID,
        RMS_NORM_F32_CAPABILITY_ID,
        ElementType::F32,
        ElementType::F32,
    )
}

fn rms_norm_contract_with_types(
    operation_id: &str,
    capability_id: &str,
    input_type: ElementType,
    output_type: ElementType,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(token_hidden_dimensions(), [input_type], TensorAccess::Read)?,
            contiguous_tensor(
                vec![DimensionConstraint::Symbol("hidden_size".to_owned())],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [output_type],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            positive_epsilon_attribute("epsilon")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: if output_type == ElementType::F32 {
            f32_reference_tolerance()?
        } else {
            f16_reference_tolerance()?
        },
        provider: provider_requirement(capability_id, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn dense_linear_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(DENSE_LINEAR_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![
                    DimensionConstraint::Symbol("rows".to_owned()),
                    DimensionConstraint::Symbol("in_features".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    DimensionConstraint::Symbol("out_features".to_owned()),
                    DimensionConstraint::Symbol("in_features".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![
                DimensionConstraint::Symbol("rows".to_owned()),
                DimensionConstraint::Symbol("out_features".to_owned()),
            ],
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("in_features")?,
            unsigned_attribute("out_features")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(DENSE_LINEAR_F16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn dense_swiglu_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(DENSE_SWIGLU_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                packed_gate_up_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                hidden_intermediate_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("intermediate_size")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(DENSE_SWIGLU_F16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Dense GeGLU using the tanh approximation of GELU.
///
/// Gate and up projections remain independent logical weights at this
/// boundary. Physical packing or quantization is a provider concern and must
/// never be inferred from the operation signature.
pub fn dense_geglu_tanh_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(DENSE_GEGLU_TANH_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                intermediate_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                intermediate_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                hidden_intermediate_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("intermediate_size")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(
            DENSE_GEGLU_TANH_F16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Multiplies a token-major F16 hidden tensor by one positive compile-time
/// rational. The output must exactly alias the input so the operation cannot
/// silently materialize an extra residual-sized buffer.
pub fn constant_scale_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(CONSTANT_SCALE_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Read,
        )?],
        outputs: vec![contiguous_tensor_with_alias(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
            AliasPolicy::MustAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            positive_rational_attribute("scale")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(
            CONSTANT_SCALE_F16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Applies `cap * tanh(logit / cap)` to one final-position vocabulary row.
/// The output is deliberately in-place so samplers consume the semantically
/// capped logits without retaining a second vocabulary-sized allocation.
pub fn logit_softcap_contract() -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![
        DimensionConstraint::Exact(1),
        DimensionConstraint::Symbol("vocab_size".to_owned()),
    ];
    let descriptor = OperationDescriptor {
        id: OperationId::new(LOGIT_SOFTCAP_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            dimensions.clone(),
            [ElementType::F16],
            TensorAccess::Read,
        )?],
        outputs: vec![contiguous_tensor_with_alias(
            dimensions,
            [ElementType::F16],
            TensorAccess::Write,
            AliasPolicy::MustAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("vocab_size")?,
            positive_rational_attribute("cap")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(
            LOGIT_SOFTCAP_F16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// A routed SwiGLU expert set plus one sigmoid-gated shared SwiGLU expert.
///
/// The operation boundary intentionally owns routing, routed expert execution,
/// shared expert execution, and the final sum. Providers can choose a
/// monolithic kernel, overlap the shared path with routed experts, or use a
/// decomposed fallback without changing the immutable model program. Weight
/// ordinals and logical stack shapes are part of the stable contract; physical
/// quantization and expert placement remain weight/provider concerns.
pub fn routed_shared_swiglu_moe_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("expert_count"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                routed_expert_gate_up_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                routed_expert_down_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![exact(1), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                shared_expert_gate_up_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                shared_expert_down_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("expert_count")?,
            unsigned_attribute("experts_per_token")?,
            unsigned_attribute("routed_intermediate_size")?,
            unsigned_attribute("shared_intermediate_size")?,
            unconstrained_bool_attribute("normalize_topk")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(
            ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// A top-K routed SwiGLU expert set without a shared expert branch.
///
/// Routing, expert execution, weighted reduction, and their scratch lifetime
/// form one stable operation boundary. This is intentionally separate from
/// [`routed_shared_swiglu_moe_contract`]: a provider must never synthesize
/// shared-expert weights or execute extra shared work for routed-only model
/// families.
pub fn routed_swiglu_moe_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(ROUTED_SWIGLU_MOE_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("expert_count"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                routed_expert_gate_up_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                routed_expert_down_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("expert_count")?,
            unsigned_attribute("experts_per_token")?,
            unsigned_attribute("routed_intermediate_size")?,
            unconstrained_bool_attribute("normalize_topk")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(
            ROUTED_SWIGLU_MOE_F16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// GPT-OSS top-K routed experts with the model's clamped, interleaved SwiGLU.
///
/// Expert matrices are logical BF16 tensors at this boundary. Their checkpoint
/// packing, sidecar scales, and any execution repack remain exclusively in the
/// weight materializer/provider contracts. `gate_up_features` is explicit
/// because the generic tensor contract has no derived-dimension expression;
/// callers and providers must bind it to exactly twice `intermediate_size`.
/// With `gate_up_interleaved=true`, even rows are gate rows and odd rows are up
/// rows in the logical `[E, 2*I, H]` tensor; no implicit reshape is permitted.
pub fn gpt_oss_routed_clamped_swiglu_moe_contract() -> Result<StandardOperationContract, VNextError>
{
    let descriptor = OperationDescriptor {
        id: OperationId::new(GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("expert_count"), symbol("hidden_size")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("expert_count")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    symbol("expert_count"),
                    symbol("gate_up_features"),
                    symbol("hidden_size"),
                ],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("expert_count"), symbol("gate_up_features")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    symbol("expert_count"),
                    symbol("hidden_size"),
                    symbol("intermediate_size"),
                ],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("expert_count"), symbol("hidden_size")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("expert_count")?,
            unsigned_attribute("experts_per_token")?,
            unsigned_attribute("intermediate_size")?,
            unsigned_attribute("gate_up_features")?,
            true_bool_attribute("normalize_topk")?,
            exact_rational_attribute("swiglu_limit", 7, 1)?,
            true_bool_attribute("gate_up_interleaved")?,
            true_bool_attribute("down_bias_before_route_reduction")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(
            GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn residual_add_contract() -> Result<StandardOperationContract, VNextError> {
    residual_add_contract_with_types(
        RESIDUAL_ADD_OPERATION_ID,
        RESIDUAL_ADD_F16_CAPABILITY_ID,
        ElementType::F16,
        ElementType::F16,
        ElementType::F16,
    )
}

pub fn residual_add_f32_f16_contract() -> Result<StandardOperationContract, VNextError> {
    residual_add_contract_with_types(
        RESIDUAL_ADD_F32_F16_OPERATION_ID,
        RESIDUAL_ADD_F32_F16_CAPABILITY_ID,
        ElementType::F32,
        ElementType::F16,
        ElementType::F32,
    )
}

fn residual_add_contract_with_types(
    operation_id: &str,
    capability_id: &str,
    left_type: ElementType,
    right_type: ElementType,
    output_type: ElementType,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(token_hidden_dimensions(), [left_type], TensorAccess::Read)?,
            contiguous_tensor(token_hidden_dimensions(), [right_type], TensorAccess::Read)?,
        ],
        outputs: vec![contiguous_tensor_with_alias(
            token_hidden_dimensions(),
            [output_type],
            TensorAccess::Write,
            AliasPolicy::MayAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([unsigned_attribute("hidden_size")?]))?,
        resources: no_auxiliary_resources(),
        oracle: if output_type == ElementType::F32 {
            OracleSpec::Exact
        } else {
            f16_reference_tolerance()?
        },
        provider: provider_requirement(capability_id, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Gated DeltaNet mixer including input normalization, projections, recurrent
/// convolution/Delta state update, gated normalization, output projection, and
/// the attention residual. Weight ordinals are part of the stable contract.
pub fn gated_delta_recurrent_attention_contract() -> Result<StandardOperationContract, VNextError> {
    gated_delta_recurrent_attention_contract_with_hidden(
        GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
        ContractVersion::new(6, 0),
        GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
        ElementType::F16,
    )
}

pub fn gated_delta_recurrent_attention_f32_master_contract(
) -> Result<StandardOperationContract, VNextError> {
    gated_delta_recurrent_attention_contract_with_hidden(
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
        ContractVersion::new(1, 0),
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID,
        ElementType::F32,
    )
}

fn gated_delta_recurrent_attention_contract_with_hidden(
    operation_id: &str,
    version: ContractVersion,
    capability_id: &str,
    hidden_type: ElementType,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version,
        inputs: vec![
            contiguous_tensor(token_hidden_dimensions(), [hidden_type], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("qkvzba_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("qkv_features"), symbol("conv_kernel")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("value_heads")],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("value_heads")],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("value_head_dim")],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("hidden_size"), symbol("value_features")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("qkv_features"), symbol("conv_state_width")],
                [ElementType::F16],
                TensorAccess::ReadWrite,
            )?,
            contiguous_tensor(
                vec![
                    symbol("value_heads"),
                    symbol("value_head_dim"),
                    symbol("key_head_dim"),
                ],
                [ElementType::F32],
                TensorAccess::ReadWrite,
            )?,
        ],
        outputs: vec![contiguous_tensor_with_alias(
            token_hidden_dimensions(),
            [hidden_type],
            TensorAccess::Write,
            AliasPolicy::MayAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("key_heads")?,
            unsigned_attribute("value_heads")?,
            unsigned_attribute("key_head_dim")?,
            unsigned_attribute("value_head_dim")?,
            unsigned_attribute("qkv_features")?,
            unsigned_attribute("value_features")?,
            unsigned_attribute("qkvz_features")?,
            unsigned_attribute("ba_features")?,
            unsigned_attribute("qkvzba_features")?,
            unsigned_attribute("conv_kernel")?,
            unsigned_attribute("conv_state_width")?,
            positive_epsilon_attribute("epsilon")?,
            nonnegative_unsigned_attribute("layer_index")?,
            text_choices_attribute(
                "decay_parameterization",
                GatedDeltaDecayParameterization::ALL.map(|value| value.as_str()),
            )?,
            text_choices_attribute(
                "value_head_mapping",
                GatedDeltaValueHeadMapping::ALL.map(|value| value.as_str()),
            )?,
        ]))?,
        resources: attention_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(capability_id, version)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Dense causal attention including input normalization, Q/K normalization,
/// RoPE, KV update, attention, optional output gate, output projection, and
/// the attention residual. KV physical paging remains a provider concern.
pub fn causal_paged_attention_contract() -> Result<StandardOperationContract, VNextError> {
    causal_paged_attention_contract_with_hidden(
        CAUSAL_PAGED_ATTENTION_OPERATION_ID,
        ContractVersion::new(2, 0),
        CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
        ElementType::F16,
    )
}

pub fn causal_paged_attention_f32_master_contract() -> Result<StandardOperationContract, VNextError>
{
    causal_paged_attention_contract_with_hidden(
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
        ContractVersion::new(1, 0),
        CAUSAL_PAGED_ATTENTION_F32_MASTER_CAPABILITY_ID,
        ElementType::F32,
    )
}

fn causal_paged_attention_contract_with_hidden(
    operation_id: &str,
    version: ContractVersion,
    capability_id: &str,
    hidden_type: ElementType,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version,
        inputs: vec![
            contiguous_tensor(token_hidden_dimensions(), [hidden_type], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("query_projection_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("kv_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("kv_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("hidden_size"), symbol("query_features")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("head_dim")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("head_dim")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![exact(2), symbol("key_value_heads"), symbol("head_dim")],
                [ElementType::F16],
                TensorAccess::ReadWrite,
            )?,
        ],
        outputs: vec![contiguous_tensor_with_alias(
            token_hidden_dimensions(),
            [hidden_type],
            TensorAccess::Write,
            AliasPolicy::MayAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("query_heads")?,
            unsigned_attribute("key_value_heads")?,
            unsigned_attribute("head_dim")?,
            unsigned_attribute("query_features")?,
            unsigned_attribute("query_projection_features")?,
            unsigned_attribute("kv_features")?,
            unsigned_attribute("rope_dim")?,
            unsigned_attribute("maximum_context_tokens")?,
            positive_rational_attribute("rope_theta")?,
            unconstrained_bool_attribute("rope_interleaved")?,
            unconstrained_bool_attribute("output_gate")?,
            true_bool_attribute("causal")?,
            positive_epsilon_attribute("epsilon")?,
            nonnegative_unsigned_attribute("layer_index")?,
        ]))?,
        resources: causal_attention_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(capability_id, version)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Hybrid causal attention with value normalization and optional K-as-V.
///
/// In addition to the shared causal attention pipeline, this contract makes
/// the hybrid-layer semantics explicit: the active rotary width and its
/// frequency denominator are independent, attention uses a typed scale,
/// local layers carry a sliding window, values use weightless RMSNorm, full
/// layers may bind K as V, and post-attention RMSNorm is applied before the
/// residual is added.
pub fn hybrid_vnorm_causal_paged_attention_contract(
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("query_projection_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("kv_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("kv_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("hidden_size"), symbol("query_features")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("head_dim")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("head_dim")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![exact(2), symbol("key_value_heads"), symbol("head_dim")],
                [ElementType::F16],
                TensorAccess::ReadWrite,
            )?,
            contiguous_tensor(
                vec![symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor_with_alias(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
            AliasPolicy::MayAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("query_heads")?,
            unsigned_attribute("key_value_heads")?,
            unsigned_attribute("head_dim")?,
            unsigned_attribute("query_features")?,
            unsigned_attribute("query_projection_features")?,
            unsigned_attribute("kv_features")?,
            unsigned_attribute("rope_dim")?,
            unsigned_attribute("rope_frequency_denominator")?,
            unsigned_attribute("maximum_context_tokens")?,
            positive_rational_attribute("rope_theta")?,
            unconstrained_bool_attribute("rope_interleaved")?,
            positive_rational_attribute("attention_scale")?,
            nonnegative_unsigned_attribute("sliding_window_tokens")?,
            true_bool_attribute("value_rms_norm")?,
            unconstrained_bool_attribute("attention_k_eq_v")?,
            true_bool_attribute("causal")?,
            positive_epsilon_attribute("epsilon")?,
            nonnegative_unsigned_attribute("layer_index")?,
        ]))?,
        resources: causal_attention_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(
            HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// GPT-OSS causal attention including input normalization, biased Q/K/V/O
/// projections, per-query-head attention sinks, YaRN RoPE, KV update, output
/// projection, and the attention residual. A zero `sliding_window_tokens`
/// selects full causal attention; a positive value selects the typed local
/// window. KV paging and kernel fusion remain provider concerns.
pub fn gpt_oss_causal_paged_attention_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("query_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("kv_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("kv_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("hidden_size"), symbol("query_features")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("query_features")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("kv_features")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("kv_features")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("query_heads")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![exact(2), symbol("kv_heads"), symbol("head_dim")],
                [ElementType::F16],
                TensorAccess::ReadWrite,
            )?,
        ],
        outputs: vec![contiguous_tensor_with_alias(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
            AliasPolicy::MayAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("query_heads")?,
            unsigned_attribute("kv_heads")?,
            unsigned_attribute("head_dim")?,
            unsigned_attribute("query_features")?,
            unsigned_attribute("kv_features")?,
            unsigned_attribute("rope_dim")?,
            unsigned_attribute("maximum_context_tokens")?,
            positive_rational_attribute("rope_theta")?,
            positive_rational_attribute("yarn_factor")?,
            unsigned_attribute("yarn_original_context_tokens")?,
            positive_rational_attribute("yarn_beta_fast")?,
            positive_rational_attribute("yarn_beta_slow")?,
            false_bool_attribute("yarn_truncate")?,
            nonnegative_unsigned_attribute("sliding_window_tokens")?,
            true_bool_attribute("causal")?,
            positive_epsilon_attribute("epsilon")?,
            nonnegative_unsigned_attribute("layer_index")?,
        ]))?,
        resources: causal_attention_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(
            GPT_OSS_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

fn contiguous_tensor(
    dimensions: Vec<DimensionConstraint>,
    element_types: impl IntoIterator<Item = ElementType>,
    access: TensorAccess,
) -> Result<TensorContract, VNextError> {
    contiguous_tensor_with_alias(dimensions, element_types, access, AliasPolicy::NoAlias)
}

fn contiguous_tensor_with_alias(
    dimensions: Vec<DimensionConstraint>,
    element_types: impl IntoIterator<Item = ElementType>,
    access: TensorAccess,
    alias: AliasPolicy,
) -> Result<TensorContract, VNextError> {
    TensorContract::new(
        dimensions,
        element_types.into_iter().collect(),
        vec![LayoutConstraint::Contiguous],
        access,
        alias,
    )
}

fn token_hidden_dimensions() -> Vec<DimensionConstraint> {
    vec![
        DimensionConstraint::Symbol("tokens".to_owned()),
        DimensionConstraint::Symbol("hidden_size".to_owned()),
    ]
}

fn packed_gate_up_dimensions() -> Vec<DimensionConstraint> {
    vec![
        DimensionConstraint::Exact(2),
        DimensionConstraint::Symbol("intermediate_size".to_owned()),
        DimensionConstraint::Symbol("hidden_size".to_owned()),
    ]
}

fn intermediate_hidden_dimensions() -> Vec<DimensionConstraint> {
    vec![
        DimensionConstraint::Symbol("intermediate_size".to_owned()),
        DimensionConstraint::Symbol("hidden_size".to_owned()),
    ]
}

fn hidden_intermediate_dimensions() -> Vec<DimensionConstraint> {
    vec![
        DimensionConstraint::Symbol("hidden_size".to_owned()),
        DimensionConstraint::Symbol("intermediate_size".to_owned()),
    ]
}

fn routed_expert_gate_up_dimensions() -> Vec<DimensionConstraint> {
    vec![
        symbol("expert_count"),
        exact(2),
        symbol("routed_intermediate_size"),
        symbol("hidden_size"),
    ]
}

fn routed_expert_down_dimensions() -> Vec<DimensionConstraint> {
    vec![
        symbol("expert_count"),
        symbol("hidden_size"),
        symbol("routed_intermediate_size"),
    ]
}

fn shared_expert_gate_up_dimensions() -> Vec<DimensionConstraint> {
    vec![
        exact(2),
        symbol("shared_intermediate_size"),
        symbol("hidden_size"),
    ]
}

fn shared_expert_down_dimensions() -> Vec<DimensionConstraint> {
    vec![symbol("hidden_size"), symbol("shared_intermediate_size")]
}

fn no_auxiliary_resources() -> ResourceRequirements {
    ResourceRequirements {
        minimum_value_alignment_bytes: 16,
        scratch: ResourcePresenceRequirement::Forbidden,
        binding: ResourcePresenceRequirement::Forbidden,
        persistent: ResourcePresenceRequirement::Forbidden,
    }
}

fn attention_resources() -> ResourceRequirements {
    ResourceRequirements {
        minimum_value_alignment_bytes: 16,
        scratch: ResourcePresenceRequirement::Required,
        binding: ResourcePresenceRequirement::Optional,
        persistent: ResourcePresenceRequirement::Forbidden,
    }
}

fn causal_attention_resources() -> ResourceRequirements {
    ResourceRequirements {
        minimum_value_alignment_bytes: 16,
        scratch: ResourcePresenceRequirement::Required,
        binding: ResourcePresenceRequirement::Required,
        persistent: ResourcePresenceRequirement::Forbidden,
    }
}

fn symbol(name: &str) -> DimensionConstraint {
    DimensionConstraint::Symbol(name.to_owned())
}

const fn exact(value: u64) -> DimensionConstraint {
    DimensionConstraint::Exact(value)
}

fn provider_requirement(
    capability: &str,
    minimum_version: ContractVersion,
) -> Result<ProviderRequirement, VNextError> {
    Ok(ProviderRequirement {
        minimum_version,
        required_capabilities: BTreeSet::from([CapabilityId::new(capability)?]),
    })
}

fn f16_reference_tolerance() -> Result<OracleSpec, VNextError> {
    Ok(OracleSpec::RelativeTolerance {
        tolerance: CanonicalRational::new(1, 1_000)?,
    })
}

fn f32_reference_tolerance() -> Result<OracleSpec, VNextError> {
    Ok(OracleSpec::RelativeTolerance {
        tolerance: CanonicalRational::new(1, 100_000)?,
    })
}

fn unsigned_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Unsigned,
            required: true,
            constraint: AttributeConstraint::UnsignedRange {
                minimum: 1,
                maximum: u32::MAX as u64,
            },
        },
    ))
}

fn nonnegative_unsigned_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Unsigned,
            required: true,
            constraint: AttributeConstraint::UnsignedRange {
                minimum: 0,
                maximum: u32::MAX as u64,
            },
        },
    ))
}

fn unconstrained_bool_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Bool,
            required: true,
            constraint: AttributeConstraint::None,
        },
    ))
}

fn true_bool_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Bool,
            required: true,
            constraint: AttributeConstraint::BoolEquals(true),
        },
    ))
}

fn false_bool_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Bool,
            required: true,
            constraint: AttributeConstraint::BoolEquals(false),
        },
    ))
}

fn exact_rational_attribute(
    name: &str,
    numerator: i64,
    denominator: u64,
) -> Result<(AttributeId, AttributeSpec), VNextError> {
    let value = CanonicalRational::new(numerator, denominator)?;
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Rational,
            required: true,
            constraint: AttributeConstraint::RationalRange {
                minimum: value,
                maximum: value,
            },
        },
    ))
}

fn positive_rational_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Rational,
            required: true,
            constraint: AttributeConstraint::RationalRange {
                minimum: CanonicalRational::new(1, u64::MAX)?,
                maximum: CanonicalRational::new(i64::MAX, 1)?,
            },
        },
    ))
}

fn positive_epsilon_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Rational,
            required: true,
            constraint: AttributeConstraint::RationalRange {
                minimum: CanonicalRational::new(1, 1_000_000_000_000)?,
                maximum: CanonicalRational::new(1, 1)?,
            },
        },
    ))
}

fn text_choices_attribute(
    name: &str,
    values: impl IntoIterator<Item = &'static str>,
) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Text,
            required: true,
            constraint: AttributeConstraint::TextChoices {
                values: values.into_iter().map(str::to_owned).collect(),
            },
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_delta_recurrent_only_capability_never_claims_chunked_scan() {
        let capabilities = GatedDeltaExecutionCapabilities::recurrent_only();
        assert_eq!(
            capabilities
                .select(1, GatedDeltaExecutionPreference::ChunkedScan)
                .unwrap(),
            GatedDeltaExecutionForm::RecurrentScan
        );
        assert_eq!(
            capabilities
                .select(64, GatedDeltaExecutionPreference::ChunkedScan)
                .unwrap(),
            GatedDeltaExecutionForm::RecurrentScan
        );
    }

    #[test]
    fn gated_delta_chunk_plan_preserves_exact_tail_boundaries() {
        let capabilities = GatedDeltaExecutionCapabilities::with_chunked_scan(64).unwrap();
        assert_eq!(
            capabilities
                .select(1, GatedDeltaExecutionPreference::ChunkedScan)
                .unwrap(),
            GatedDeltaExecutionForm::RecurrentScan
        );
        assert_eq!(
            capabilities
                .select(64, GatedDeltaExecutionPreference::RecurrentScan)
                .unwrap(),
            GatedDeltaExecutionForm::RecurrentScan
        );
        for (tokens, chunks, final_tokens) in [(2, 1, 2), (64, 1, 64), (65, 2, 1)] {
            let GatedDeltaExecutionForm::ChunkedScan(plan) = capabilities
                .select(tokens, GatedDeltaExecutionPreference::ChunkedScan)
                .unwrap()
            else {
                panic!("{tokens} tokens must select chunked scan");
            };
            assert_eq!(plan.token_count(), tokens);
            assert_eq!(plan.chunk_size(), 64);
            assert_eq!(plan.chunk_count(), chunks);
            assert_eq!(plan.final_chunk_tokens(), final_tokens);
            assert_eq!(
                GatedDeltaExecutionForm::ChunkedScan(plan).as_str(),
                "chunked_scan"
            );
        }
    }

    #[test]
    fn gated_delta_execution_capabilities_reject_invalid_domains() {
        assert!(GatedDeltaExecutionCapabilities::with_chunked_scan(0).is_err());
        assert!(GatedDeltaExecutionCapabilities::recurrent_only()
            .select(0, GatedDeltaExecutionPreference::RecurrentScan)
            .is_err());
    }

    #[test]
    fn token_embedding_contract_is_backend_and_model_neutral() {
        let contract = token_embedding_contract().unwrap();
        let descriptor = contract.descriptor();
        assert_eq!(descriptor.id.as_str(), TOKEN_EMBEDDING_OPERATION_ID);
        assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
    }

    #[test]
    fn fp32_master_contracts_form_one_exact_mixed_precision_chain() {
        let contracts = [
            token_embedding_f32_master_contract().unwrap(),
            gated_delta_recurrent_attention_f32_master_contract().unwrap(),
            causal_paged_attention_f32_master_contract().unwrap(),
            rms_norm_f32_to_f16_contract().unwrap(),
            residual_add_f32_f16_contract().unwrap(),
            rms_norm_f32_contract().unwrap(),
            last_token_dense_linear_f32_contract().unwrap(),
            last_token_masked_argmax_f32_contract().unwrap(),
        ];
        let ids = contracts
            .iter()
            .map(|contract| contract.descriptor().id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), contracts.len());
        for contract in &contracts {
            let descriptor = contract.descriptor();
            assert_eq!(descriptor.version, ContractVersion::new(1, 0));
            assert_eq!(descriptor.provider.required_capabilities.len(), 1);
            contract
                .validate_signature(&descriptor.inputs, &descriptor.outputs)
                .unwrap();
        }

        let embedding = contracts[0].descriptor();
        assert_eq!(
            embedding.outputs[0].element_types(),
            &BTreeSet::from([ElementType::F32])
        );
        for attention in [&contracts[1], &contracts[2]] {
            let descriptor = attention.descriptor();
            assert_eq!(
                descriptor.inputs[0].element_types(),
                &BTreeSet::from([ElementType::F32])
            );
            assert_eq!(
                descriptor.outputs[0].element_types(),
                &BTreeSet::from([ElementType::F32])
            );
            assert_eq!(
                descriptor.outputs[0].alias(),
                &AliasPolicy::MayAlias { tensor_index: 0 }
            );
        }

        let branch_norm = contracts[3].descriptor();
        assert_eq!(
            branch_norm.inputs[0].element_types(),
            &BTreeSet::from([ElementType::F32])
        );
        assert_eq!(
            branch_norm.outputs[0].element_types(),
            &BTreeSet::from([ElementType::F16])
        );
        let residual = contracts[4].descriptor();
        assert_eq!(
            residual.inputs[0].element_types(),
            &BTreeSet::from([ElementType::F32])
        );
        assert_eq!(
            residual.inputs[1].element_types(),
            &BTreeSet::from([ElementType::F16])
        );
        assert_eq!(
            residual.outputs[0].element_types(),
            &BTreeSet::from([ElementType::F32])
        );

        let final_norm = contracts[5].descriptor();
        let head = contracts[6].descriptor();
        let argmax = contracts[7].descriptor();
        for tensor in [
            &final_norm.inputs[0],
            &final_norm.outputs[0],
            &head.inputs[0],
            &head.outputs[0],
            &argmax.inputs[0],
        ] {
            assert_eq!(tensor.element_types(), &BTreeSet::from([ElementType::F32]));
        }

        assert!(!ids.contains(TOKEN_EMBEDDING_OPERATION_ID));
        assert!(!ids.contains(RMS_NORM_OPERATION_ID));
        assert!(!ids.contains(RESIDUAL_ADD_OPERATION_ID));
    }

    #[test]
    fn last_token_dense_linear_contract_is_backend_and_model_neutral() {
        let contract = last_token_dense_linear_contract().unwrap();
        let descriptor = contract.descriptor();
        assert_eq!(descriptor.id.as_str(), LAST_TOKEN_DENSE_LINEAR_OPERATION_ID);
        assert_eq!(descriptor.version, ContractVersion::new(1, 1));
        assert_eq!(
            descriptor.resources.scratch,
            ResourcePresenceRequirement::Optional
        );
        assert_eq!(
            descriptor.outputs[0].dimensions(),
            &[
                DimensionConstraint::Exact(1),
                DimensionConstraint::Symbol("out_features".to_owned()),
            ]
        );
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
    }

    #[test]
    fn last_token_masked_argmax_contract_keeps_policy_in_typed_inputs() {
        let contract = last_token_masked_argmax_contract().unwrap();
        let descriptor = contract.descriptor();
        assert_eq!(
            descriptor.id.as_str(),
            LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID
        );
        assert_eq!(descriptor.version, ContractVersion::new(3, 0));
        assert_eq!(descriptor.inputs.len(), 5);
        assert_eq!(descriptor.inputs[0].access(), TensorAccess::Read);
        assert_eq!(
            descriptor.resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            descriptor.resources.binding,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            descriptor.resources.persistent,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            descriptor.inputs[1].element_types(),
            &BTreeSet::from([ElementType::U8])
        );
        assert_eq!(
            descriptor.inputs[2].element_types(),
            &BTreeSet::from([ElementType::U32])
        );
        assert_eq!(
            descriptor.inputs[3].dimensions(),
            &[DimensionConstraint::Exact(2)]
        );
        assert_eq!(
            descriptor.inputs[4].element_types(),
            &BTreeSet::from([ElementType::F32])
        );
        assert_eq!(
            descriptor.outputs[0].element_types(),
            &BTreeSet::from([ElementType::U32])
        );
        assert_eq!(
            descriptor.outputs[0].dimensions(),
            &[DimensionConstraint::Exact(1)]
        );
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
    }

    #[test]
    fn transformer_primitives_have_explicit_math_and_resource_boundaries() {
        let contracts = [
            rms_norm_contract().unwrap(),
            dense_linear_contract().unwrap(),
            dense_swiglu_contract().unwrap(),
            residual_add_contract().unwrap(),
        ];
        for contract in &contracts {
            let descriptor = contract.descriptor();
            assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
            contract
                .validate_signature(&descriptor.inputs, &descriptor.outputs)
                .unwrap();
        }
        assert_eq!(
            contracts[2].descriptor().resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            contracts[3].descriptor().outputs[0].alias(),
            &AliasPolicy::MayAlias { tensor_index: 0 }
        );
    }

    #[test]
    fn hybrid_vnorm_simple_ops_have_typed_shapes_attributes_and_aliasing() {
        let geglu = dense_geglu_tanh_contract().unwrap();
        let geglu_descriptor = geglu.descriptor();
        assert_eq!(geglu_descriptor.id.as_str(), DENSE_GEGLU_TANH_OPERATION_ID);
        assert_eq!(geglu_descriptor.version, ContractVersion::new(1, 0));
        assert_eq!(geglu_descriptor.inputs.len(), 4);
        assert_eq!(
            geglu_descriptor.inputs[0].dimensions(),
            &[symbol("tokens"), symbol("hidden_size")]
        );
        for projection in [&geglu_descriptor.inputs[1], &geglu_descriptor.inputs[2]] {
            assert_eq!(
                projection.dimensions(),
                &[symbol("intermediate_size"), symbol("hidden_size")]
            );
        }
        assert_eq!(
            geglu_descriptor.inputs[3].dimensions(),
            &[symbol("hidden_size"), symbol("intermediate_size")]
        );
        assert_eq!(
            geglu_descriptor.outputs[0].dimensions(),
            &[symbol("tokens"), symbol("hidden_size")]
        );
        assert_eq!(
            geglu_descriptor.outputs[0].element_types(),
            &BTreeSet::from([ElementType::F16])
        );
        assert_eq!(
            geglu_descriptor.resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            geglu_descriptor.provider.required_capabilities,
            BTreeSet::from([CapabilityId::new(DENSE_GEGLU_TANH_F16_CAPABILITY_ID).unwrap()])
        );

        let scale = constant_scale_contract().unwrap();
        let scale_descriptor = scale.descriptor();
        assert_eq!(scale_descriptor.id.as_str(), CONSTANT_SCALE_OPERATION_ID);
        assert_eq!(scale_descriptor.inputs.len(), 1);
        assert_eq!(
            scale_descriptor.outputs[0].alias(),
            &AliasPolicy::MustAlias { tensor_index: 0 }
        );
        assert!(scale_descriptor
            .attributes
            .entries()
            .contains_key(&AttributeId::new("scale").unwrap()));
        assert_eq!(scale_descriptor.resources, no_auxiliary_resources());

        let softcap = logit_softcap_contract().unwrap();
        let softcap_descriptor = softcap.descriptor();
        assert_eq!(softcap_descriptor.id.as_str(), LOGIT_SOFTCAP_OPERATION_ID);
        assert_eq!(
            softcap_descriptor.inputs[0].dimensions(),
            &[exact(1), symbol("vocab_size")]
        );
        assert_eq!(
            softcap_descriptor.outputs[0].alias(),
            &AliasPolicy::MustAlias { tensor_index: 0 }
        );
        assert!(softcap_descriptor
            .attributes
            .entries()
            .contains_key(&AttributeId::new("cap").unwrap()));
        assert_eq!(softcap_descriptor.resources, no_auxiliary_resources());

        for (descriptor, attribute) in [(scale_descriptor, "scale"), (softcap_descriptor, "cap")] {
            let AttributeConstraint::RationalRange { minimum, maximum } = &descriptor
                .attributes
                .entries()
                .get(&AttributeId::new(attribute).unwrap())
                .unwrap()
                .constraint
            else {
                panic!("{attribute} must have a typed rational range");
            };
            assert!(minimum.numerator() > 0);
            assert!(maximum >= minimum);
        }

        for contract in [&geglu, &scale, &softcap] {
            let descriptor = contract.descriptor();
            assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
            contract
                .validate_signature(&descriptor.inputs, &descriptor.outputs)
                .unwrap();
            assert!(contract
                .validate_signature(&[], &descriptor.outputs)
                .is_err());
        }
    }

    #[test]
    fn routed_shared_moe_contract_keeps_fusion_and_weight_abi_generic() {
        let contract = routed_shared_swiglu_moe_contract().unwrap();
        let descriptor = contract.descriptor();

        assert_eq!(
            descriptor.id.as_str(),
            ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID
        );
        assert_eq!(descriptor.version, ContractVersion::new(1, 0));
        assert_eq!(descriptor.inputs.len(), 7);
        assert_eq!(
            descriptor.inputs[2].dimensions(),
            &[
                symbol("expert_count"),
                exact(2),
                symbol("routed_intermediate_size"),
                symbol("hidden_size"),
            ]
        );
        assert_eq!(
            descriptor.inputs[3].dimensions(),
            &[
                symbol("expert_count"),
                symbol("hidden_size"),
                symbol("routed_intermediate_size"),
            ]
        );
        assert_eq!(
            descriptor.inputs[4].dimensions(),
            &[exact(1), symbol("hidden_size")]
        );
        assert_eq!(
            descriptor.resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            descriptor.resources.persistent,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            descriptor.provider.required_capabilities,
            BTreeSet::from([
                CapabilityId::new(ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID).unwrap()
            ])
        );
        for attribute in [
            "hidden_size",
            "expert_count",
            "experts_per_token",
            "routed_intermediate_size",
            "shared_intermediate_size",
            "normalize_topk",
        ] {
            assert!(
                descriptor
                    .attributes
                    .entries()
                    .contains_key(&AttributeId::new(attribute).unwrap()),
                "missing typed MoE attribute {attribute}"
            );
        }
        assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
    }

    #[test]
    fn routed_only_moe_contract_has_no_shared_expert_abi() {
        let contract = routed_swiglu_moe_contract().unwrap();
        let descriptor = contract.descriptor();

        assert_eq!(descriptor.id.as_str(), ROUTED_SWIGLU_MOE_OPERATION_ID);
        assert_eq!(descriptor.version, ContractVersion::new(1, 0));
        assert_eq!(descriptor.inputs.len(), 4);
        assert_eq!(
            descriptor.inputs[2].dimensions(),
            &[
                symbol("expert_count"),
                exact(2),
                symbol("routed_intermediate_size"),
                symbol("hidden_size"),
            ]
        );
        assert_eq!(
            descriptor.inputs[3].dimensions(),
            &[
                symbol("expert_count"),
                symbol("hidden_size"),
                symbol("routed_intermediate_size"),
            ]
        );
        assert_eq!(
            descriptor.resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            descriptor.provider.required_capabilities,
            BTreeSet::from([CapabilityId::new(ROUTED_SWIGLU_MOE_F16_CAPABILITY_ID).unwrap()])
        );
        for attribute in [
            "hidden_size",
            "expert_count",
            "experts_per_token",
            "routed_intermediate_size",
            "normalize_topk",
        ] {
            assert!(
                descriptor
                    .attributes
                    .entries()
                    .contains_key(&AttributeId::new(attribute).unwrap()),
                "missing typed routed-only MoE attribute {attribute}"
            );
        }
        assert!(!descriptor
            .attributes
            .entries()
            .contains_key(&AttributeId::new("shared_intermediate_size").unwrap()));
        assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
    }

    #[test]
    fn gpt_oss_routed_clamped_swiglu_contract_has_exact_logical_bf16_abi() {
        let contract = gpt_oss_routed_clamped_swiglu_moe_contract().unwrap();
        let descriptor = contract.descriptor();

        assert_eq!(
            descriptor.id.as_str(),
            GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID
        );
        assert_eq!(descriptor.version, ContractVersion::new(1, 0));
        assert_eq!(descriptor.inputs.len(), 7);
        assert_eq!(
            descriptor.inputs[0].dimensions(),
            &[symbol("tokens"), symbol("hidden_size")]
        );
        assert_eq!(
            descriptor.inputs[1].dimensions(),
            &[symbol("expert_count"), symbol("hidden_size")]
        );
        assert_eq!(descriptor.inputs[2].dimensions(), &[symbol("expert_count")]);
        assert_eq!(
            descriptor.inputs[3].dimensions(),
            &[
                symbol("expert_count"),
                symbol("gate_up_features"),
                symbol("hidden_size"),
            ]
        );
        assert_eq!(
            descriptor.inputs[4].dimensions(),
            &[symbol("expert_count"), symbol("gate_up_features")]
        );
        assert_eq!(
            descriptor.inputs[5].dimensions(),
            &[
                symbol("expert_count"),
                symbol("hidden_size"),
                symbol("intermediate_size"),
            ]
        );
        assert_eq!(
            descriptor.inputs[6].dimensions(),
            &[symbol("expert_count"), symbol("hidden_size")]
        );
        assert_eq!(
            descriptor.inputs[0].element_types(),
            &BTreeSet::from([ElementType::F16])
        );
        for input in &descriptor.inputs[1..] {
            assert_eq!(input.element_types(), &BTreeSet::from([ElementType::Bf16]));
        }
        assert!(descriptor
            .inputs
            .iter()
            .all(|input| input.access() == TensorAccess::Read));
        assert_eq!(
            descriptor.outputs[0].dimensions(),
            &[symbol("tokens"), symbol("hidden_size")]
        );
        assert_eq!(
            descriptor.outputs[0].element_types(),
            &BTreeSet::from([ElementType::F16])
        );
        assert_eq!(descriptor.outputs[0].alias(), &AliasPolicy::NoAlias);

        assert_eq!(
            descriptor
                .attributes
                .entries()
                .keys()
                .map(AttributeId::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "down_bias_before_route_reduction",
                "expert_count",
                "experts_per_token",
                "gate_up_features",
                "gate_up_interleaved",
                "hidden_size",
                "intermediate_size",
                "normalize_topk",
                "swiglu_limit",
            ])
        );
        for attribute in [
            "normalize_topk",
            "gate_up_interleaved",
            "down_bias_before_route_reduction",
        ] {
            assert_eq!(
                descriptor
                    .attributes
                    .entries()
                    .get(&AttributeId::new(attribute).unwrap())
                    .unwrap()
                    .constraint,
                AttributeConstraint::BoolEquals(true)
            );
        }
        let seven = CanonicalRational::new(7, 1).unwrap();
        assert_eq!(
            descriptor
                .attributes
                .entries()
                .get(&AttributeId::new("swiglu_limit").unwrap())
                .unwrap()
                .constraint,
            AttributeConstraint::RationalRange {
                minimum: seven,
                maximum: seven,
            }
        );
        assert_eq!(
            descriptor.resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            descriptor.resources.binding,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            descriptor.resources.persistent,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            descriptor.provider.required_capabilities,
            BTreeSet::from([CapabilityId::new(
                GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID
            )
            .unwrap()])
        );
        assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
    }

    #[test]
    fn gpt_oss_causal_attention_contract_has_exact_bias_sink_and_yarn_abi() {
        let contract = gpt_oss_causal_paged_attention_contract().unwrap();
        let descriptor = contract.descriptor();

        assert_eq!(
            descriptor.id.as_str(),
            GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID
        );
        assert_eq!(descriptor.version, ContractVersion::new(1, 0));
        assert_eq!(descriptor.inputs.len(), 12);
        let expected_dimensions = [
            vec![symbol("tokens"), symbol("hidden_size")],
            vec![symbol("hidden_size")],
            vec![symbol("query_features"), symbol("hidden_size")],
            vec![symbol("kv_features"), symbol("hidden_size")],
            vec![symbol("kv_features"), symbol("hidden_size")],
            vec![symbol("hidden_size"), symbol("query_features")],
            vec![symbol("query_features")],
            vec![symbol("kv_features")],
            vec![symbol("kv_features")],
            vec![symbol("hidden_size")],
            vec![symbol("query_heads")],
            vec![exact(2), symbol("kv_heads"), symbol("head_dim")],
        ];
        for (input, expected) in descriptor.inputs.iter().zip(expected_dimensions) {
            assert_eq!(input.dimensions(), expected);
            assert_eq!(input.element_types(), &BTreeSet::from([ElementType::F16]));
        }
        assert!(descriptor.inputs[..11]
            .iter()
            .all(|input| input.access() == TensorAccess::Read));
        assert_eq!(descriptor.inputs[11].access(), TensorAccess::ReadWrite);
        assert_eq!(
            descriptor.outputs[0].dimensions(),
            &[symbol("tokens"), symbol("hidden_size")]
        );
        assert_eq!(
            descriptor.outputs[0].element_types(),
            &BTreeSet::from([ElementType::F16])
        );
        assert_eq!(
            descriptor.outputs[0].alias(),
            &AliasPolicy::MayAlias { tensor_index: 0 }
        );

        assert_eq!(
            descriptor
                .attributes
                .entries()
                .keys()
                .map(AttributeId::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "causal",
                "epsilon",
                "head_dim",
                "hidden_size",
                "kv_features",
                "kv_heads",
                "layer_index",
                "maximum_context_tokens",
                "query_features",
                "query_heads",
                "rope_dim",
                "rope_theta",
                "sliding_window_tokens",
                "yarn_beta_fast",
                "yarn_beta_slow",
                "yarn_factor",
                "yarn_original_context_tokens",
                "yarn_truncate",
            ])
        );
        assert_eq!(
            descriptor
                .attributes
                .entries()
                .get(&AttributeId::new("yarn_truncate").unwrap())
                .unwrap()
                .constraint,
            AttributeConstraint::BoolEquals(false)
        );
        assert_eq!(
            descriptor
                .attributes
                .entries()
                .get(&AttributeId::new("causal").unwrap())
                .unwrap()
                .constraint,
            AttributeConstraint::BoolEquals(true)
        );
        assert_eq!(
            descriptor
                .attributes
                .entries()
                .get(&AttributeId::new("sliding_window_tokens").unwrap())
                .unwrap()
                .constraint,
            AttributeConstraint::UnsignedRange {
                minimum: 0,
                maximum: u32::MAX as u64,
            }
        );
        assert_eq!(
            descriptor.resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            descriptor.resources.binding,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            descriptor.resources.persistent,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            descriptor.provider.required_capabilities,
            BTreeSet::from([
                CapabilityId::new(GPT_OSS_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID).unwrap()
            ])
        );
        assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
    }

    #[test]
    fn attention_contracts_fix_weight_order_state_mutability_and_scratch() {
        let linear = gated_delta_recurrent_attention_contract().unwrap();
        let full = causal_paged_attention_contract().unwrap();
        for contract in [&linear, &full] {
            let descriptor = contract.descriptor();
            assert_eq!(
                descriptor.resources.scratch,
                ResourcePresenceRequirement::Required
            );
            assert_eq!(
                descriptor.outputs[0].alias(),
                &AliasPolicy::MayAlias { tensor_index: 0 }
            );
            assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
            contract
                .validate_signature(&descriptor.inputs, &descriptor.outputs)
                .unwrap();
        }
        assert_eq!(linear.descriptor().inputs.len(), 10);
        assert_eq!(linear.descriptor().version, ContractVersion::new(6, 0));
        assert_eq!(
            linear.descriptor().resources.binding,
            ResourcePresenceRequirement::Optional
        );
        assert_eq!(
            full.descriptor().resources.binding,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            linear.descriptor().provider.minimum_version,
            ContractVersion::new(6, 0)
        );
        for (name, values) in [
            (
                "decay_parameterization",
                GatedDeltaDecayParameterization::ALL
                    .map(|value| value.as_str().to_owned())
                    .into_iter()
                    .collect(),
            ),
            (
                "value_head_mapping",
                GatedDeltaValueHeadMapping::ALL
                    .map(|value| value.as_str().to_owned())
                    .into_iter()
                    .collect(),
            ),
        ] {
            assert_eq!(
                linear
                    .descriptor()
                    .attributes
                    .entries()
                    .get(&AttributeId::new(name).unwrap())
                    .unwrap()
                    .constraint,
                AttributeConstraint::TextChoices { values }
            );
        }
        for ordinal in [4, 5, 6, 9] {
            assert_eq!(
                linear.descriptor().inputs[ordinal].element_types(),
                &BTreeSet::from([ElementType::F32])
            );
        }
        assert_eq!(
            linear.descriptor().inputs[8].access(),
            TensorAccess::ReadWrite
        );
        assert_eq!(
            linear.descriptor().inputs[9].access(),
            TensorAccess::ReadWrite
        );
        assert_eq!(full.descriptor().inputs.len(), 9);
        assert_eq!(full.descriptor().version, ContractVersion::new(2, 0));
        assert_eq!(
            full.descriptor().provider.minimum_version,
            ContractVersion::new(2, 0)
        );
        assert_eq!(
            full.descriptor().resources.binding,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            full.descriptor().inputs[8].access(),
            TensorAccess::ReadWrite
        );
    }
}
