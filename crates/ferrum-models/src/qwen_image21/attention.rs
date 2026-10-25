//! Block-0 attention branch after LN/modulation: real Q/K/V and output
//! projections, learned head RMS normalization, explicit RoPE and structured
//! attention. No residual, gate, MLP, prefix cache or full-block claim.

mod layout;
pub use layout::{
    QwenImage21AttentionConfig, QwenImage21AttentionKeyRun, QwenImage21AttentionLayout,
    QwenImage21AttentionSegment,
};

use super::{
    ExpectedWeights, QwenImage21Block0QkNormFamily as Qk, QwenImage21Config, QwenImage21QkFrontend,
    StructureLimits,
};
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct QwenImage21Block0AttentionFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}

impl QwenImage21Block0AttentionFamily {
    pub const FAMILY_ID: &'static str = "family.qwen_image21.block0_attention";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "qwen_image21.block0_attention.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen_image21.block0_attention.cuda.bf16_f32.v1";
    pub const METAL_PROFILE_ID: &'static str = "qwen_image21.block0_attention.metal.bf16_f32.v1";
    /// Explicit vendor-fused attention choice; does not change the legacy CUDA
    /// profile or claim its sequential FP32 softmax rounding. Initial providers
    /// support head_dim 128; numerical qualification is separate evidence.
    pub const CUDA_FUSED_PROFILE_ID: &'static str =
        "qwen_image21.block0_attention.cuda.fused_attention.experimental.v1";
    pub const INPUT_ID: &'static str = Qk::INPUT_ID;
    pub const FREQUENCIES_ID: &'static str = Qk::FREQUENCIES_ID;
    pub const QUERY_ID: &'static str = Qk::QUERY_ID;
    pub const KEY_ID: &'static str = Qk::KEY_ID;
    pub const QUERY_NORMALIZED_ID: &'static str = Qk::QUERY_NORMALIZED_ID;
    pub const KEY_NORMALIZED_ID: &'static str = Qk::KEY_NORMALIZED_ID;
    pub const QUERY_ROTATED_ID: &'static str = Qk::QUERY_ROTATED_ID;
    pub const KEY_ROTATED_ID: &'static str = Qk::KEY_ROTATED_ID;
    pub const VALUE_ID: &'static str = "value.qwen_image21.block0_attention.value";
    pub const ATTENTION_ID: &'static str = "value.qwen_image21.block0_attention.attention";
    pub const OUTPUT_ID: &'static str = "value.qwen_image21.block0_attention.projected";
    pub const OUTPUT_IDS: [&'static str; 9] = [
        Self::QUERY_ID,
        Self::KEY_ID,
        Self::QUERY_NORMALIZED_ID,
        Self::KEY_NORMALIZED_ID,
        Self::QUERY_ROTATED_ID,
        Self::KEY_ROTATED_ID,
        Self::VALUE_ID,
        Self::ATTENTION_ID,
        Self::OUTPUT_ID,
    ];
    pub const WEIGHT_NAMES: [&'static str; 6] = [
        Qk::WEIGHT_NAMES[0],
        Qk::WEIGHT_NAMES[1],
        Qk::WEIGHT_NAMES[2],
        Qk::WEIGHT_NAMES[3],
        "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.attn.to_out.0.weight",
    ];

    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen_image21.block0_attention")?,
        })
    }

    /// The model chooses 1/sqrt(D), rounded once to F32. Its exact binary value
    /// is then carried as a canonical rational; providers do not infer scale.
    pub fn scale_for_head_dim(head_dim: u64) -> Result<CanonicalRational, VNextError> {
        if !(2..=STRUCTURED_ATTENTION_BF16_MAX_HEAD_DIM).contains(&head_dim)
            || !head_dim.is_power_of_two()
        {
            return Err(invalid(
                "scale",
                "attention head dimension must be a power of two in 2..=256",
            ));
        }
        let effective = (1.0 / (head_dim as f64).sqrt()) as f32;
        let bits = effective.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 150;
        let significand = i64::from((bits & 0x7f_ffff) | (1 << 23));
        // This bounded head dimension gives a finite normal scale in [1/16,1].
        let denominator = 1_u64
            .checked_shl(
                u32::try_from(-exponent)
                    .map_err(|_| invalid("scale", "scale exponent is not representable"))?,
            )
            .ok_or_else(|| invalid("scale", "scale denominator overflows"))?;
        CanonicalRational::new(significand, denominator)
    }
}

impl ModelFamilyProvider for QwenImage21Block0AttentionFamily {
    type Config = QwenImage21AttentionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(config).map_err(|e| invalid("config", e.to_string()))? != *raw {
            return Err(invalid(
                "config",
                "attention requires canonical checked transformer/layout config",
            ));
        }
        config.validate()
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| invalid("config", e.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        let ModelWeights::Stored(mut schema) = Qk::new()?.weights(config.transformer())? else {
            return Err(invalid("weights", "Q/K frontend must have Stored weights"));
        };
        let full = ExpectedWeights::from_config(config.transformer(), StructureLimits::default())
            .and_then(|w| w.weight_schema())
            .map_err(|e| invalid("weights", e.to_string()))?;
        for name in &Self::WEIGHT_NAMES[4..] {
            let id = WeightId::new(format!("weight.qwen_image21.{name}"))?;
            schema.tensors.push(
                full.tensor(&id)
                    .ok_or_else(|| invalid("weights", "attention projection weight missing"))?
                    .clone(),
            );
            schema
                .components
                .extend(full.physical_component_refs(&id)?.into_iter().cloned());
        }
        schema.layout_id = WeightLayoutId::new("weight-layout.qwen_image21.block0_attention.bf16")?;
        schema.validate(&self.id)?;
        Ok(ModelWeights::Stored(schema))
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let frontend = Qk::with_frontend(QwenImage21QkFrontend::Rotary)?;
        let source = frontend.numerical_profiles(config.transformer())?;
        let mut profiles = Vec::new();
        for (name, source_name) in [
            (
                Self::REFERENCE_PROFILE_ID,
                QwenImage21QkFrontend::Rotary.reference_profile_id(),
            ),
            (
                Self::CUDA_PROFILE_ID,
                QwenImage21QkFrontend::Rotary.cuda_profile_id(),
            ),
            (
                Self::CUDA_FUSED_PROFILE_ID,
                QwenImage21QkFrontend::Rotary.cuda_profile_id(),
            ),
            (
                Self::METAL_PROFILE_ID,
                QwenImage21QkFrontend::Rotary.metal_profile_id(),
            ),
        ] {
            let fused = name == Self::CUDA_FUSED_PROFILE_ID;
            if fused && config.transformer().head_dim() != 128 {
                continue;
            }
            let source_id =
                NumericalProfileId::new(source_name).map_err(|e| invalid("profile", e))?;
            let mut profile = source.resolve(&source_id)?.clone();
            profile.id = NumericalProfileId::new(name).map_err(|e| invalid("profile", e))?;
            profile.family_id = self.id.clone();
            profile.primary_activation = ProgramValueId::new(Self::OUTPUT_ID)?;
            for output in &Self::OUTPUT_IDS[6..] {
                profile
                    .boundaries
                    .insert(ProgramValueId::new(*output)?, ElementType::Bf16);
            }
            profile.operations.push(NumericalOperationContract {
                operation_id: OperationId::new(if fused {
                    STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID
                } else {
                    STRUCTURED_ATTENTION_BF16_OPERATION_ID
                })?,
                version: ContractVersion::new(1, 0),
                // Vendor SDPA does not expose each internal probability/
                // reduction rounding boundary. Its distinct operation version
                // owns that contract; do not mislabel it as all-FP32 online.
                multiplication_type: (!fused).then_some(ElementType::F32),
                accumulation_type: (!fused).then_some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            });
            profiles.push(profile);
        }
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE_ID)
                .map_err(|e| invalid("profile", e))?],
        )
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        config.validate()?;
        if self.numerical_profiles(config)?.resolve(&profile.id)? != profile {
            return Err(invalid(
                "profile",
                "attention profile differs from its declared arithmetic",
            ));
        }
        let frontend = Qk::with_frontend(QwenImage21QkFrontend::Rotary)?;
        let base_profiles = frontend.numerical_profiles(config.transformer())?;
        let fused = profile.id.as_str() == Self::CUDA_FUSED_PROFILE_ID;
        let base_name = if profile.id.as_str() == Self::CUDA_PROFILE_ID || fused {
            QwenImage21QkFrontend::Rotary.cuda_profile_id()
        } else if profile.id.as_str() == Self::METAL_PROFILE_ID {
            QwenImage21QkFrontend::Rotary.metal_profile_id()
        } else {
            QwenImage21QkFrontend::Rotary.reference_profile_id()
        };
        let base = frontend.semantic_program(
            config.transformer(),
            base_profiles
                .resolve(&NumericalProfileId::new(base_name).map_err(|e| invalid("profile", e))?)?,
        )?;
        let mut weights = base.weights().to_vec();
        for (index, name) in Self::WEIGHT_NAMES[4..].iter().enumerate() {
            weights.push(WeightReference {
                weight_id: WeightId::new(format!("weight.qwen_image21.{name}"))?,
                value_id: ProgramValueId::new(format!(
                    "value.qwen_image21.block0_attention.weight{index}"
                ))?,
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.transformer().hidden_size(); 2],
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            });
        }
        let value_weight = weights[weights.len() - 2].value_id.clone();
        let output_weight = weights[weights.len() - 1].value_id.clone();
        let mut blocks = base.blocks().to_vec();
        let nodes = &mut blocks
            .last_mut()
            .ok_or_else(|| invalid("program", "Q/K frontend has no block"))?
            .nodes;
        let dense = |suffix: &str,
                     input: &str,
                     weight: ProgramValueId,
                     output: &str|
         -> Result<ProgramNode, VNextError> {
            Ok(ProgramNode {
                id: NodeId::new(format!("node.qwen_image21.block0_attention.{suffix}"))?,
                operation_id: OperationId::new(DENSE_LINEAR_BF16_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![ProgramValueId::new(input)?, weight],
                outputs: vec![ProgramValueId::new(output)?],
                attributes: BTreeMap::from([
                    (
                        AttributeId::new("in_features")?,
                        SemanticValue::Unsigned(config.transformer().hidden_size()),
                    ),
                    (
                        AttributeId::new("out_features")?,
                        SemanticValue::Unsigned(config.transformer().hidden_size()),
                    ),
                ]),
            })
        };
        nodes.push(dense(
            "value_projection",
            Self::INPUT_ID,
            value_weight,
            Self::VALUE_ID,
        )?);
        nodes.push(ProgramNode {
            id: NodeId::new("node.qwen_image21.block0_attention.attention")?,
            operation_id: OperationId::new(if fused {
                STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID
            } else {
                STRUCTURED_ATTENTION_BF16_OPERATION_ID
            })?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: [Self::QUERY_ROTATED_ID, Self::KEY_ROTATED_ID, Self::VALUE_ID]
                .into_iter()
                .map(ProgramValueId::new)
                .collect::<Result<_, _>>()?,
            outputs: vec![ProgramValueId::new(Self::ATTENTION_ID)?],
            attributes: config.attributes()?,
        });
        nodes.push(dense(
            "output_projection",
            Self::ATTENTION_ID,
            output_weight,
            Self::OUTPUT_ID,
        )?);
        ModelProgram::new(
            self.id.clone(),
            base.inputs().to_vec(),
            blocks,
            vec![],
            weights,
            Self::OUTPUT_IDS
                .into_iter()
                .map(ProgramValueId::new)
                .collect::<Result<_, _>>()?,
        )?
        .with_input_tensor_specs(BTreeMap::from([
            (
                ProgramValueId::new(Self::INPUT_ID)?,
                ProgramTensorSpec {
                    dimensions: vec![config.layout().rows(), config.transformer().hidden_size()],
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            ),
            (
                ProgramValueId::new(Self::FREQUENCIES_ID)?,
                ProgramTensorSpec {
                    dimensions: vec![
                        u64::from(config.layout().sequence_length()),
                        config.transformer().head_dim() / 2,
                        2,
                    ],
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            ),
        ]))
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: QwenImage21Block0AttentionFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
