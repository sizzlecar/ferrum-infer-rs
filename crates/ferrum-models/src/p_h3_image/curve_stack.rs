//! Composition of the real Curve W4 block into one Stored tensor component.
//! No layer arithmetic is reimplemented here, and no intermediate crosses the
//! Host boundary. This private component is not a complete image pipeline.
use super::{
    PH3ImageBlockProjectionStorage, PH3ImageCurveBlockConfig, PH3ImageCurveBlockFamily,
    PH3ImageCurveBlockLimits, PH3ImageCurveBlockRequest, PH3ImageCurveTimeEmbedding,
    PH3ImageProjectionStorage, PH3ImageTransformerConfig,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod checkpoint;
mod compose;
mod config;
mod remap;
pub(super) use checkpoint::PH3ImageCurveStackCheckpoint;
pub(super) use config::{PH3ImageCurveStackConfig, PH3ImageStackProjectionStorage};

/// Model metadata capacity is a conservative canonical-serialization bound,
/// distinct from the existing Host/device allocation authority. The template
/// and full graph are admitted before expanding layer-scoped metadata.
#[derive(Clone, Copy, Debug)]
pub(super) struct PH3ImageCurveStackLimits {
    pub max_metadata_bytes: usize,
    pub max_plan_nodes: usize,
    pub max_stored_bytes: u64,
}
impl PH3ImageCurveStackLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.max_metadata_bytes == 0 || self.max_plan_nodes == 0 || self.max_stored_bytes == 0 {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
    fn block(self) -> PH3ImageCurveBlockLimits {
        PH3ImageCurveBlockLimits {
            max_metadata_bytes: self.max_metadata_bytes,
            max_plan_nodes: self.max_plan_nodes,
            max_stored_bytes: self.max_stored_bytes,
        }
    }
}
#[derive(Clone, Debug)]
pub(super) struct PH3ImageCurveStackFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageCurveStackLimits,
}
impl PH3ImageCurveStackFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.curve_stack";
    pub const CUDA_PROFILE_ID: &'static str =
        "p_h3_image.curve_stack.cuda.precise_w4_swiglu_f32.native_attention.v1";
    pub const CUDA_MIXED_PROFILE_ID: &'static str =
        "p_h3_image.curve_stack.cuda.precise_mixed_convrot_swiglu_f32.native_attention.v1";
    pub fn cuda_profile_id(config: &PH3ImageCurveStackConfig) -> &'static str {
        if config.projection_storage.all_w4() {
            Self::CUDA_PROFILE_ID
        } else {
            Self::CUDA_MIXED_PROFILE_ID
        }
    }
    pub const INPUT_ID: &'static str = "value.p_h3_image.curve_stack.input";
    pub const OUTPUT_ID: &'static str = "value.p_h3_image.curve_stack.output";
    pub const BASIS_ID: &'static str = PH3ImageCurveBlockFamily::BASIS_ID;
    pub const COSINE_ID: &'static str = PH3ImageCurveBlockFamily::COSINE_ID;
    pub const SINE_ID: &'static str = PH3ImageCurveBlockFamily::SINE_ID;
    pub fn new(limits: PH3ImageCurveStackLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.p_h3_image.curve_stack.comfy_w4")?,
            limits,
        })
    }
    pub fn compile_options(
        &self,
        c: &PH3ImageCurveStackConfig,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(compose::build(c, self.limits)?.options)
    }
}
impl ModelFamilyProvider for PH3ImageCurveStackFamily {
    type Config = PH3ImageCurveStackConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *r {
            return Err(invalid("configuration must be canonical"));
        }
        compose::admit(c, self.limits)?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(r, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(r.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(r, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(compose::build(c, self.limits)?.schema))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![compose::build(c, self.limits)?.profile],
            vec![NumericalProfileId::new(Self::cuda_profile_id(c)).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let b = compose::build(c, self.limits)?;
        if &b.profile != p {
            return Err(invalid("unknown stack numerical profile"));
        }
        Ok(b.program)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("P-H3 Curve stack: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
