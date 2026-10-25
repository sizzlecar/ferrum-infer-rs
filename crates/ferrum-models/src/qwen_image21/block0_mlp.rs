//! Block 0's MLP only. `z` is already normalized/modulated at the MLP entrance.
//! Attention, residual gates, conditioning and the rest of the block are absent.

use super::{ExpectedWeights, QwenImage21Config, StructureLimits};
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct QwenImage21Block0MlpFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}

impl QwenImage21Block0MlpFamily {
    pub const FAMILY_ID: &'static str = "family.qwen_image21.block0_mlp";
    pub const REFERENCE_PROFILE_ID: &'static str = "qwen_image21.block0_mlp.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen_image21.block0_mlp.cuda.bf16_f32.v1";
    pub const METAL_PROFILE_ID: &'static str = "qwen_image21.block0_mlp.metal.bf16_f32.v1";
    pub const INPUT_ID: &'static str = "value.qwen_image21.block0_mlp.z";
    pub const GATE_ID: &'static str = "value.qwen_image21.block0_mlp.gate";
    pub const UP_ID: &'static str = "value.qwen_image21.block0_mlp.up";
    pub const ACTIVATION_ID: &'static str = "value.qwen_image21.block0_mlp.activation";
    pub const OUTPUT_ID: &'static str = "value.qwen_image21.block0_mlp.output";
    pub const WEIGHT_NAMES: [&'static str; 3] = [
        "transformer_blocks.0.img_mlp.gate_layer.weight",
        "transformer_blocks.0.img_mlp.proj.weight",
        "transformer_blocks.0.img_mlp.out.weight",
    ];

    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen_image21.block0_mlp")?,
        })
    }
}

impl ModelFamilyProvider for QwenImage21Block0MlpFamily {
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
        let canonical =
            serde_json::to_value(config).map_err(|error| invalid("config", error.to_string()))?;
        if canonical != *raw {
            return Err(invalid(
                "config",
                "MLP requires canonical checked QwenImage21 config",
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
                    .ok_or_else(|| invalid("weights", "required MLP weight missing"))?
                    .clone(),
            );
            components.extend(full.physical_component_refs(&id)?.into_iter().cloned());
        }
        let schema = WeightSchema {
            format_id: full.format_id,
            layout_id: WeightLayoutId::new("weight-layout.qwen_image21.block0_mlp.bf16")?,
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
        for name in [
            Self::REFERENCE_PROFILE_ID,
            Self::CUDA_PROFILE_ID,
            Self::METAL_PROFILE_ID,
        ] {
            let operations = [DENSE_LINEAR_BF16_OPERATION_ID, SILU_MUL_BF16_OPERATION_ID]
                .into_iter()
                .map(|operation| {
                    Ok(NumericalOperationContract {
                        operation_id: OperationId::new(operation)?,
                        version: ContractVersion::new(1, 0),
                        multiplication_type: Some(ElementType::F32),
                        accumulation_type: Some(ElementType::F32),
                        staged_arithmetic: None,
                        composite_arithmetic: None,
                    })
                })
                .collect::<Result<Vec<_>, VNextError>>()?;
            profiles.push(NumericalExecutionProfile {
                id: NumericalProfileId::new(name).map_err(|reason| invalid("profile", reason))?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries: [
                    Self::GATE_ID,
                    Self::UP_ID,
                    Self::ACTIVATION_ID,
                    Self::OUTPUT_ID,
                ]
                .into_iter()
                .map(|name| Ok((ProgramValueId::new(name)?, ElementType::Bf16)))
                .collect::<Result<_, VNextError>>()?,
                states: vec![],
                kv_storage: vec![],
                operations,
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
                "MLP profile differs from the declared arithmetic",
            ));
        }
        let z = ProgramValueId::new(Self::INPUT_ID)?;
        let gate = ProgramValueId::new(Self::GATE_ID)?;
        let up = ProgramValueId::new(Self::UP_ID)?;
        let activation = ProgramValueId::new(Self::ACTIVATION_ID)?;
        let output = ProgramValueId::new(Self::OUTPUT_ID)?;
        let mut weights = Vec::new();
        for (index, name) in Self::WEIGHT_NAMES.into_iter().enumerate() {
            let dimensions = if index == 2 {
                vec![config.hidden_size(), config.mlp_hidden_size()]
            } else {
                vec![config.mlp_hidden_size(), config.hidden_size()]
            };
            weights.push(WeightReference {
                weight_id: WeightId::new(format!("weight.qwen_image21.{name}"))?,
                value_id: ProgramValueId::new(format!(
                    "value.qwen_image21.block0_mlp.weight{index}"
                ))?,
                tensor: ProgramTensorSpec {
                    dimensions,
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            });
        }
        let dense = |name: &str,
                     input: ProgramValueId,
                     weight: ProgramValueId,
                     output: ProgramValueId,
                     inputs,
                     outputs|
         -> Result<ProgramNode, VNextError> {
            Ok(ProgramNode {
                id: NodeId::new(format!("node.qwen_image21.block0_mlp.{name}"))?,
                operation_id: OperationId::new(DENSE_LINEAR_BF16_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![input, weight],
                outputs: vec![output],
                attributes: BTreeMap::from([
                    (
                        AttributeId::new("in_features")?,
                        SemanticValue::Unsigned(inputs),
                    ),
                    (
                        AttributeId::new("out_features")?,
                        SemanticValue::Unsigned(outputs),
                    ),
                ]),
            })
        };
        let nodes = vec![
            dense(
                "gate",
                z.clone(),
                weights[0].value_id.clone(),
                gate.clone(),
                config.hidden_size(),
                config.mlp_hidden_size(),
            )?,
            dense(
                "up",
                z.clone(),
                weights[1].value_id.clone(),
                up.clone(),
                config.hidden_size(),
                config.mlp_hidden_size(),
            )?,
            ProgramNode {
                id: NodeId::new("node.qwen_image21.block0_mlp.silu_mul")?,
                operation_id: OperationId::new(SILU_MUL_BF16_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![gate.clone(), up.clone()],
                outputs: vec![activation.clone()],
                attributes: BTreeMap::new(),
            },
            dense(
                "out",
                activation.clone(),
                weights[2].value_id.clone(),
                output.clone(),
                config.mlp_hidden_size(),
                config.hidden_size(),
            )?,
        ];
        // This diagnostic partial family exposes real BF16 node boundaries.
        ModelProgram::new(
            self.id.clone(),
            vec![z],
            vec![ProgramBlock {
                id: "block.qwen_image21.block0_mlp".into(),
                nodes,
            }],
            vec![],
            weights,
            vec![gate, up, activation, output],
        )
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: QwenImage21Block0MlpFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
