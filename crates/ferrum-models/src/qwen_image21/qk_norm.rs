//! Block 0 Q/K projections and learned per-head RMS normalization only.
//! Input is already normalized/modulated. RoPE, V, attention and residuals
//! are deliberately outside this executable component.

use super::{ExpectedWeights, QwenImage21Config, StructureLimits};
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct QwenImage21Block0QkNormFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}

impl QwenImage21Block0QkNormFamily {
    pub const FAMILY_ID: &'static str = "family.qwen_image21.block0_qk_norm";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "qwen_image21.block0_qk_norm.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen_image21.block0_qk_norm.cuda.bf16_f32.v1";
    pub const INPUT_ID: &'static str = "value.qwen_image21.block0_qk_norm.x";
    pub const QUERY_ID: &'static str = "value.qwen_image21.block0_qk_norm.query";
    pub const KEY_ID: &'static str = "value.qwen_image21.block0_qk_norm.key";
    pub const QUERY_NORMALIZED_ID: &'static str =
        "value.qwen_image21.block0_qk_norm.query_normalized";
    pub const KEY_NORMALIZED_ID: &'static str = "value.qwen_image21.block0_qk_norm.key_normalized";
    pub const WEIGHT_NAMES: [&'static str; 4] = [
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.0.attn.norm_k.weight",
    ];

    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen_image21.block0_qk_norm")?,
        })
    }
}

impl ModelFamilyProvider for QwenImage21Block0QkNormFamily {
    type Config = QwenImage21Config;
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
        if serde_json::to_value(config).map_err(|error| invalid("config", error.to_string()))?
            != *raw
        {
            return Err(invalid(
                "config",
                "Q/K norm requires canonical checked QwenImage21 config",
            ));
        }
        Ok(())
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
        let config = serde_json::from_value(raw.clone())
            .map_err(|error| invalid("config", error.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        let full = ExpectedWeights::from_config(config, StructureLimits::default())
            .and_then(|weights| weights.weight_schema())
            .map_err(|error| invalid("weights", error.to_string()))?;
        let mut components = Vec::new();
        let mut tensors = Vec::new();
        for name in Self::WEIGHT_NAMES {
            let id = WeightId::new(format!("weight.qwen_image21.{name}"))?;
            tensors.push(
                full.tensor(&id)
                    .ok_or_else(|| invalid("weights", "required Q/K norm weight missing"))?
                    .clone(),
            );
            components.extend(full.physical_component_refs(&id)?.into_iter().cloned());
        }
        let schema = WeightSchema {
            format_id: full.format_id,
            layout_id: WeightLayoutId::new("weight-layout.qwen_image21.block0_qk_norm.bf16")?,
            version: full.version,
            components,
            tensors,
        };
        schema.validate(self.family_id())?;
        Ok(ModelWeights::Stored(schema))
    }
    fn numerical_profiles(&self, _: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let reference = NumericalProfileId::new(Self::REFERENCE_PROFILE_ID)
            .map_err(|reason| invalid("profile", reason))?;
        let mut profiles = Vec::new();
        for name in [Self::REFERENCE_PROFILE_ID, Self::CUDA_PROFILE_ID] {
            profiles.push(NumericalExecutionProfile {
                id: NumericalProfileId::new(name).map_err(|reason| invalid("profile", reason))?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::QUERY_NORMALIZED_ID)?,
                boundaries: [
                    Self::QUERY_ID,
                    Self::KEY_ID,
                    Self::QUERY_NORMALIZED_ID,
                    Self::KEY_NORMALIZED_ID,
                ]
                .into_iter()
                .map(|name| Ok((ProgramValueId::new(name)?, ElementType::Bf16)))
                .collect::<Result<_, VNextError>>()?,
                states: vec![],
                kv_storage: vec![],
                operations: [
                    DENSE_LINEAR_BF16_OPERATION_ID,
                    HEAD_RMS_NORM_BF16_OPERATION_ID,
                ]
                .into_iter()
                .map(|name| {
                    Ok(NumericalOperationContract {
                        operation_id: OperationId::new(name)?,
                        version: ContractVersion::new(1, 0),
                        multiplication_type: Some(ElementType::F32),
                        accumulation_type: Some(ElementType::F32),
                        staged_arithmetic: None,
                        composite_arithmetic: None,
                    })
                })
                .collect::<Result<_, VNextError>>()?,
            });
        }
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
            vec![reference],
        )
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(config)?.resolve(&profile.id)? != profile {
            return Err(invalid(
                "profile",
                "Q/K norm profile differs from declared arithmetic",
            ));
        }
        let input = ProgramValueId::new(Self::INPUT_ID)?;
        let output_names = [
            Self::QUERY_ID,
            Self::KEY_ID,
            Self::QUERY_NORMALIZED_ID,
            Self::KEY_NORMALIZED_ID,
        ];
        let outputs = output_names
            .into_iter()
            .map(ProgramValueId::new)
            .collect::<Result<Vec<_>, _>>()?;
        let weights = Self::WEIGHT_NAMES
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                Ok(WeightReference {
                    weight_id: WeightId::new(format!("weight.qwen_image21.{name}"))?,
                    value_id: ProgramValueId::new(format!(
                        "value.qwen_image21.block0_qk_norm.weight{index}"
                    ))?,
                    tensor: ProgramTensorSpec {
                        dimensions: if index < 2 {
                            vec![config.hidden_size(), config.hidden_size()]
                        } else {
                            vec![config.head_dim()]
                        },
                        element_type: ElementType::Bf16,
                        layout: ResolvedTensorLayout::Contiguous,
                    },
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let mut nodes = Vec::new();
        for branch in 0..2 {
            nodes.push(ProgramNode {
                id: NodeId::new(format!(
                    "node.qwen_image21.block0_qk_norm.projection{branch}"
                ))?,
                operation_id: OperationId::new(DENSE_LINEAR_BF16_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![input.clone(), weights[branch].value_id.clone()],
                outputs: vec![outputs[branch].clone()],
                attributes: BTreeMap::from([
                    (
                        AttributeId::new("in_features")?,
                        SemanticValue::Unsigned(config.hidden_size()),
                    ),
                    (
                        AttributeId::new("out_features")?,
                        SemanticValue::Unsigned(config.hidden_size()),
                    ),
                ]),
            });
        }
        for branch in 0..2 {
            nodes.push(ProgramNode {
                id: NodeId::new(format!("node.qwen_image21.block0_qk_norm.norm{branch}"))?,
                operation_id: OperationId::new(HEAD_RMS_NORM_BF16_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![
                    outputs[branch].clone(),
                    weights[branch + 2].value_id.clone(),
                ],
                outputs: vec![outputs[branch + 2].clone()],
                attributes: BTreeMap::from([
                    (
                        AttributeId::new("head_dim")?,
                        SemanticValue::Unsigned(config.head_dim()),
                    ),
                    (
                        AttributeId::new("epsilon")?,
                        SemanticValue::Rational(config.epsilon()),
                    ),
                ]),
            });
        }
        ModelProgram::new(
            self.id.clone(),
            vec![input],
            vec![ProgramBlock {
                id: "block.qwen_image21.block0_qk_norm".into(),
                nodes,
            }],
            vec![],
            weights,
            outputs,
        )
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: QwenImage21Block0QkNormFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
