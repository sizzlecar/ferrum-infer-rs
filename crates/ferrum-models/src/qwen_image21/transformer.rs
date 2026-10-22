//! Text-to-image transformer step over a homogeneous padded joint sequence.
//!
//! Timestep sinusoidal features and joint rotary frequencies are explicit
//! inputs. The adapter must compute them from the actual timestep/grid; this
//! family does not claim a completed conditioning or image-generation pipeline.
//! All layers run in one tensor program, with only target velocity retained.

mod conditioning;
mod geometry;
mod program;
mod remap;
pub use conditioning::{
    QwenImage21ModelTimestep, QwenImage21RotaryFrequencies, QwenImage21Timestep256,
};
pub use geometry::{QwenImage21T2iLayout, QwenImage21TransformerConfig};

use super::{
    ExpectedWeights, QwenImage21AttentionConfig, QwenImage21AttentionKeyRun,
    QwenImage21AttentionLayout, QwenImage21AttentionSegment, QwenImage21Block0Broadcast,
    QwenImage21Block0Config, QwenImage21Block0Family as Block, QwenImage21Config, StructureLimits,
};
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct QwenImage21TransformerFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}
impl QwenImage21TransformerFamily {
    pub const FAMILY_ID: &'static str = super::FAMILY_ID;
    pub const REFERENCE_PROFILE_ID: &'static str =
        "qwen_image21.transformer.t2i.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen_image21.transformer.t2i.cuda.bf16_f32.v1";
    pub const LATENTS_ID: &'static str = "value.qwen_image21.transformer.latents";
    pub const TEXT_ID: &'static str = "value.qwen_image21.transformer.text";
    pub const TIME_FEATURES_ID: &'static str = "value.qwen_image21.transformer.time_features";
    pub const FREQUENCIES_ID: &'static str = "value.qwen_image21.transformer.frequencies";
    pub const VELOCITY_ID: &'static str = "value.qwen_image21.transformer.velocity";
    pub const INPUT_IDS: [&'static str; 4] = [
        Self::LATENTS_ID,
        Self::TEXT_ID,
        Self::TIME_FEATURES_ID,
        Self::FREQUENCIES_ID,
    ];

    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen_image21.transformer.t2i")?,
        })
    }
}

impl ModelFamilyProvider for QwenImage21TransformerFamily {
    type Config = QwenImage21TransformerConfig;
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
                "transformer requires canonical checked T2I geometry",
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
        config.validate()?;
        let schema = ExpectedWeights::from_config(config.transformer(), StructureLimits::default())
            .and_then(|weights| weights.weight_schema())
            .map_err(|e| invalid("weights", e.to_string()))?;
        schema.validate(&self.id)?;
        Ok(ModelWeights::Stored(schema))
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        config.validate()?;
        let block = Block::new()?;
        let block_config = config.block_config()?;
        let source = block.numerical_profiles(&block_config)?;
        let mut profiles = Vec::new();
        for (name, source_name) in profile_sources() {
            let mut profile = source.resolve(&profile_id(source_name)?)?.clone();
            let graph = program::build(self, config, source_name)?;
            profile.id = profile_id(name)?;
            profile.family_id = self.id.clone();
            profile.primary_activation = ProgramValueId::new(Self::VELOCITY_ID)?;
            // All arithmetic/copy outputs remain BF16 ABI boundaries even when
            // the execution plan reclaims them before completion.
            profile.boundaries = graph
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|node| node.outputs.iter().cloned())
                .map(|id| (id, ElementType::Bf16))
                .collect();
            let mut operations: BTreeMap<_, _> = profile
                .operations
                .into_iter()
                .map(|op| (op.operation_id.clone(), op))
                .collect();
            for (name, arithmetic) in [
                (ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID, true),
                (GELU_TANH_BF16_OPERATION_ID, true),
                (SILU_BF16_OPERATION_ID, true),
                (ROW_GATHER_BF16_OPERATION_ID, false),
                (BATCHED_ROW_CONCAT_BF16_OPERATION_ID, false),
            ] {
                let op = NumericalOperationContract {
                    operation_id: OperationId::new(name)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: arithmetic.then_some(ElementType::F32),
                    accumulation_type: arithmetic.then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                };
                if operations.insert(op.operation_id.clone(), op).is_some() {
                    return Err(invalid(
                        "profile",
                        "unexpected duplicate entrance operation",
                    ));
                }
            }
            profile.operations = operations.into_values().collect();
            profiles.push(profile);
        }
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
            vec![profile_id(Self::REFERENCE_PROFILE_ID)?],
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
                "transformer profile differs from declared arithmetic",
            ));
        }
        let (_, source) = profile_sources()
            .into_iter()
            .find(|(name, _)| *name == profile.id.as_str())
            .ok_or_else(|| invalid("profile", "unknown transformer profile"))?;
        program::build(self, config, source)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn profile_sources() -> [(&'static str, &'static str); 2] {
    [
        (
            QwenImage21TransformerFamily::REFERENCE_PROFILE_ID,
            Block::REFERENCE_PROFILE_ID,
        ),
        (
            QwenImage21TransformerFamily::CUDA_PROFILE_ID,
            Block::CUDA_PROFILE_ID,
        ),
    ]
}
fn profile_id(name: &str) -> Result<NumericalProfileId, VNextError> {
    NumericalProfileId::new(name).map_err(|e| invalid("profile", e))
}
fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: QwenImage21TransformerFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod execution_tests;
