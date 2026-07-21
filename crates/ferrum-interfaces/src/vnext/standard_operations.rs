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
pub const LAST_TOKEN_DENSE_LINEAR_OPERATION_ID: &str = "operation.last_token_dense_linear";
pub const LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID: &str =
    "capability.operation.last_token_dense_linear.f16";
pub const RMS_NORM_OPERATION_ID: &str = "operation.rms_norm";
pub const RMS_NORM_F16_CAPABILITY_ID: &str = "capability.operation.rms_norm.f16";
pub const DENSE_LINEAR_OPERATION_ID: &str = "operation.dense_linear";
pub const DENSE_LINEAR_F16_CAPABILITY_ID: &str = "capability.operation.dense_linear.f16";
pub const DENSE_SWIGLU_OPERATION_ID: &str = "operation.dense_swiglu";
pub const DENSE_SWIGLU_F16_CAPABILITY_ID: &str = "capability.operation.dense_swiglu.f16";
pub const ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID: &str = "operation.routed_shared_swiglu_moe";
pub const ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID: &str =
    "capability.operation.routed_shared_swiglu_moe.f16";
pub const RESIDUAL_ADD_OPERATION_ID: &str = "operation.residual_add";
pub const RESIDUAL_ADD_F16_CAPABILITY_ID: &str = "capability.operation.residual_add.f16";
pub const GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID: &str =
    "operation.gated_delta_recurrent_attention";
pub const GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID: &str =
    "capability.operation.gated_delta_recurrent_attention.f16";
pub const GATED_DELTA_EXECUTION_FORM_SELECTOR_VERSION: &str =
    "gated-delta-execution-form-selector-v1";
pub const CAUSAL_PAGED_ATTENTION_OPERATION_ID: &str = "operation.causal_paged_attention";
pub const CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID: &str =
    "capability.operation.causal_paged_attention.f16";

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
    let descriptor = OperationDescriptor {
        id: OperationId::new(TOKEN_EMBEDDING_OPERATION_ID)?,
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
            [ElementType::F16],
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
            required_capabilities: BTreeSet::from([CapabilityId::new(
                TOKEN_EMBEDDING_F16_CAPABILITY_ID,
            )?]),
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
    let descriptor = OperationDescriptor {
        id: OperationId::new(LAST_TOKEN_DENSE_LINEAR_OPERATION_ID)?,
        version: ContractVersion::new(1, 1),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
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
            [ElementType::F16],
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
        provider: provider_requirement(
            LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
            ContractVersion::new(1, 1),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn rms_norm_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(RMS_NORM_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![DimensionConstraint::Symbol("hidden_size".to_owned())],
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
            positive_epsilon_attribute("epsilon")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(RMS_NORM_F16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
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

pub fn residual_add_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(RESIDUAL_ADD_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                token_hidden_dimensions(),
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
        attributes: AttributeSchema::new(BTreeMap::from([unsigned_attribute("hidden_size")?]))?,
        resources: no_auxiliary_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(RESIDUAL_ADD_F16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Gated DeltaNet mixer including input normalization, projections, recurrent
/// convolution/Delta state update, gated normalization, output projection, and
/// the attention residual. Weight ordinals are part of the stable contract.
pub fn gated_delta_recurrent_attention_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID)?,
        version: ContractVersion::new(4, 0),
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
                vec![symbol("qkv_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("value_features"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("value_heads"), symbol("hidden_size")],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("value_heads"), symbol("hidden_size")],
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
            [ElementType::F16],
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
        provider: provider_requirement(
            GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
            ContractVersion::new(4, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Dense causal attention including input normalization, Q/K normalization,
/// RoPE, KV update, attention, optional output gate, output projection, and
/// the attention residual. KV physical paging remains a provider concern.
pub fn causal_paged_attention_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(CAUSAL_PAGED_ATTENTION_OPERATION_ID)?,
        version: ContractVersion::new(2, 0),
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
        provider: provider_requirement(
            CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
            ContractVersion::new(2, 0),
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
        binding: ResourcePresenceRequirement::Forbidden,
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
        assert_eq!(linear.descriptor().inputs.len(), 13);
        assert_eq!(linear.descriptor().version, ContractVersion::new(4, 0));
        assert_eq!(
            linear.descriptor().provider.minimum_version,
            ContractVersion::new(4, 0)
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
        assert_eq!(
            linear.descriptor().resources.binding,
            ResourcePresenceRequirement::Forbidden
        );
        for ordinal in [7, 8, 9, 12] {
            assert_eq!(
                linear.descriptor().inputs[ordinal].element_types(),
                &BTreeSet::from([ElementType::F32])
            );
        }
        assert_eq!(
            linear.descriptor().inputs[11].access(),
            TensorAccess::ReadWrite
        );
        assert_eq!(
            linear.descriptor().inputs[12].access(),
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
