//! One explicitly scoped component of Qwen-Image 2.1: BF16 `img_in`.
//!
//! The family contains no text semantics or denoising state. Input row count
//! belongs to `ProgramPlanCompileOptions`, so the same prepared family can
//! compile separate exact tensor plans for different latent row counts.

use std::collections::{BTreeMap, BTreeSet};

use ferrum_runtime::{
    AttributeId, ContractVersion, ElementType, ExternalModelMetadataId, FamilyNumericalProfiles,
    ModelFamilyId, ModelFamilyProvider, ModelProgram, ModelSemanticMetadata, NodeId,
    NumericalExecutionProfile, NumericalOperationContract, NumericalProfileId, OperationId,
    ProgramBlock, ProgramNode, ProgramNodeWorkSpec, ProgramTensorSpec, ProgramValueId,
    ResolvedTensorLayout, SemanticValue, VNextError, WeightId, WeightLayoutId, WeightReference,
    WeightSchema, DENSE_LINEAR_BF16_OPERATION_ID,
};
use serde_json::Value;

use super::{ExpectedWeights, QwenImage21Config, StructureLimits};

pub const IMAGE_PROJECTION_FAMILY_ID: &str = "family.qwen_image21.img_in";
pub const IMAGE_PROJECTION_PROFILE_ID: &str = "qwen_image21.img_in.bf16_f32";
pub const IMAGE_PROJECTION_INPUT_ID: &str = "value.qwen_image21.img_in.input";
pub const IMAGE_PROJECTION_OUTPUT_ID: &str = "value.qwen_image21.img_in.output";
pub const IMAGE_PROJECTION_NODE_ID: &str = "node.qwen_image21.img_in";
pub const IMAGE_PROJECTION_WEIGHT_ID: &str = "weight.qwen_image21.img_in.weight";

const METADATA_ID: &str = "metadata.qwen_image21.img_in";
const WEIGHT_VALUE_ID: &str = "value.qwen_image21.img_in.weight";
const WEIGHT_LAYOUT_ID: &str = "weight-layout.qwen_image21.img_in.bf16";

/// The image-input projection alone, with its own family and metadata identity.
///
/// Adapt HF JSON with `QwenImage21Config::from_hf_json`, then serialize that
/// checked configuration for `TypedFamilyRegistration::define`. The canonical
/// configuration contains model dimensions; input row count is supplied only
/// when compiling the prepared program. This family is not a full denoiser.
#[derive(Debug, Clone)]
pub struct QwenImage21ImageProjectionFamily {
    family_id: ModelFamilyId,
    metadata_id: ExternalModelMetadataId,
}

impl QwenImage21ImageProjectionFamily {
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            family_id: ModelFamilyId::new(IMAGE_PROJECTION_FAMILY_ID)?,
            metadata_id: ExternalModelMetadataId::new(METADATA_ID)?,
        })
    }
}

impl ModelFamilyProvider for QwenImage21ImageProjectionFamily {
    type Config = QwenImage21Config;

    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata_id.clone()])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        let canonical =
            serde_json::to_value(config).map_err(|error| VNextError::Serialization {
                context: "serialize Qwen image-projection configuration",
                message: error.to_string(),
            })?;
        if canonical != *raw {
            return Err(invalid_config(
                "config",
                "image projection requires the exact canonical QwenImage21 configuration",
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
        Ok(self.metadata_id.clone())
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config: QwenImage21Config = serde_json::from_value(raw.clone())
            .map_err(|error| invalid_config("config", error.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        let full = ExpectedWeights::from_config(config, StructureLimits::default())
            .and_then(|expected| expected.weight_schema())
            .map_err(|error| invalid_config("weight_schema", error.to_string()))?;
        let weight_id = WeightId::new(IMAGE_PROJECTION_WEIGHT_ID)?;
        let tensor = full
            .tensor(&weight_id)
            .ok_or_else(|| invalid_config("weight_schema", "img_in is absent from the schema"))?
            .clone();
        let components = full
            .physical_component_refs(&weight_id)?
            .into_iter()
            .cloned()
            .collect();
        // Required components from the complete transformer cannot be carried
        // by a partial program. Keep the selected tensor's physical identity
        // and required status, while explicitly scoping this family's schema.
        let schema = WeightSchema {
            format_id: full.format_id,
            layout_id: WeightLayoutId::new(WEIGHT_LAYOUT_ID)?,
            version: full.version,
            components,
            tensors: vec![tensor],
        };
        schema.validate(self.family_id())?;
        Ok(schema)
    }

    fn numerical_profiles(&self, _: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile_id = NumericalProfileId::new(IMAGE_PROJECTION_PROFILE_ID)
            .map_err(|reason| invalid_config("numerical_profile", reason))?;
        let output_id = ProgramValueId::new(IMAGE_PROJECTION_OUTPUT_ID)?;
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: output_id.clone(),
                boundaries: BTreeMap::from([(output_id, ElementType::Bf16)]),
                states: Vec::new(),
                kv_storage: Vec::new(),
                operations: vec![NumericalOperationContract {
                    operation_id: OperationId::new(DENSE_LINEAR_BF16_OPERATION_ID)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile_id],
        )
    }

    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(config)?.resolve(&profile.id)? != profile {
            return Err(invalid_config(
                "numerical_profile",
                "image projection requires its declared BF16/FP32 profile",
            ));
        }
        let input = ProgramValueId::new(IMAGE_PROJECTION_INPUT_ID)?;
        let weight = ProgramValueId::new(WEIGHT_VALUE_ID)?;
        let output = ProgramValueId::new(IMAGE_PROJECTION_OUTPUT_ID)?;
        ModelProgram::new(
            self.family_id.clone(),
            vec![input.clone()],
            vec![ProgramBlock {
                id: "block.qwen_image21.img_in".to_owned(),
                nodes: vec![ProgramNode {
                    id: NodeId::new(IMAGE_PROJECTION_NODE_ID)?,
                    operation_id: OperationId::new(DENSE_LINEAR_BF16_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![input, weight.clone()],
                    outputs: vec![output.clone()],
                    attributes: BTreeMap::from([
                        (
                            AttributeId::new("in_features")?,
                            SemanticValue::Unsigned(config.in_channels()),
                        ),
                        (
                            AttributeId::new("out_features")?,
                            SemanticValue::Unsigned(config.hidden_size()),
                        ),
                    ]),
                }],
            }],
            Vec::new(),
            vec![WeightReference {
                weight_id: WeightId::new(IMAGE_PROJECTION_WEIGHT_ID)?,
                value_id: weight,
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.hidden_size(), config.in_channels()],
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }],
            vec![output],
        )
    }

    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn invalid_config(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: IMAGE_PROJECTION_FAMILY_ID.to_owned(),
        field: field.to_owned(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
