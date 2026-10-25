//! Qwen3-VL image vision graph: patch projection, encoder, main and DeepStack mergers.
//!
//! Inputs are processor-ordered patch rows, interpolated learned positions and
//! explicit FP32 axial rotary tables. Image preprocessing and position-table
//! lookup are governed caller adapters; no HWC reinterpretation occurs here.
mod config;
mod position;
mod program;
mod weights;
pub use config::{Qwen3VLVisionArchitecture, Qwen3VLVisionConfig, Qwen3VLVisionGrid};
pub use position::QWEN3_VL_VISION_POSITION_PROFILE;
pub use weights::Qwen3VLVisionExpectedWeight;

use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct Qwen3VLVisionFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}
impl Qwen3VLVisionFamily {
    pub const FAMILY_ID: &'static str = "family.qwen3_vl.vision";
    pub const REFERENCE_PROFILE_ID: &'static str = "qwen3vl.vision.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen3vl.vision.cuda.bf16_f32.v1";
    pub const METAL_PROFILE_ID: &'static str = "qwen3vl.vision.metal.bf16_f32.v1";
    pub const PATCHES_ID: &'static str = "value.qwen3_vl.vision.patches";
    pub const POSITIONS_ID: &'static str = "value.qwen3_vl.vision.positions";
    pub const COSINE_ID: &'static str = "value.qwen3_vl.vision.cosine";
    pub const SINE_ID: &'static str = "value.qwen3_vl.vision.sine";
    pub const OUTPUT_ID: &'static str = "value.qwen3_vl.vision.merged";
    pub const DEEPSTACK_IDS: [&'static str; 3] = [
        "value.qwen3_vl.vision.deepstack.0",
        "value.qwen3_vl.vision.deepstack.1",
        "value.qwen3_vl.vision.deepstack.2",
    ];
    pub const POSITION_EXTERNAL_NAME: &'static str = "model.visual.pos_embed.weight";
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen3_vl.vision")?,
        })
    }
}
impl ModelFamilyProvider for Qwen3VLVisionFamily {
    type Config = Qwen3VLVisionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(&self, raw: &Value, c: &Self::Config) -> Result<(), VNextError> {
        c.validate()?;
        if serde_json::to_value(c).map_err(|e| invalid("config", e.to_string()))? != *raw {
            return Err(invalid(
                "config",
                "requires the canonical checked vision architecture and ordered grids",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let c =
            serde_json::from_value(raw.clone()).map_err(|e| invalid("config", e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(weights::schema(c, &self.id)?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let (graph, _) = program::build(c, &self.id)?;
        let boundaries = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .cloned()
            .map(|v| (v, ElementType::Bf16))
            .collect::<BTreeMap<_, _>>();
        let ids = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| n.operation_id.clone())
            .collect::<BTreeSet<_>>();
        let operations = ids
            .into_iter()
            .map(|operation_id| {
                let copy = matches!(
                    operation_id.as_str(),
                    ROW_GATHER_BF16_OPERATION_ID | program::RESHAPE
                );
                let add = operation_id.as_str() == PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID;
                NumericalOperationContract {
                    operation_id,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (!copy && !add).then_some(ElementType::F32),
                    accumulation_type: (!copy).then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }
            })
            .collect::<Vec<_>>();
        let mut profiles = Vec::new();
        for name in [
            Self::REFERENCE_PROFILE_ID,
            Self::CUDA_PROFILE_ID,
            Self::METAL_PROFILE_ID,
        ] {
            profiles.push(NumericalExecutionProfile {
                id: NumericalProfileId::new(name).map_err(|e| invalid("profile", e))?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries: boundaries.clone(),
                states: vec![],
                kv_storage: vec![],
                operations: operations.clone(),
            });
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
        c: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&profile.id)? != profile {
            return Err(invalid(
                "profile",
                "vision arithmetic/boundaries differ from the declared profile",
            ));
        }
        Ok(program::build(c, &self.id)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: Qwen3VLVisionFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}
#[cfg(test)]
mod tests;
