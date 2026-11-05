//! Native one-image DiT forward composition, separate from sampling/serving.
//! All projections, refinement, packing, main layers and final heads are real
//! tensor-program nodes. Host input preparation has its own explicit profile.
use super::curve_stack::{
    PH3ImageCurveStackCheckpoint, PH3ImageCurveStackConfig, PH3ImageCurveStackFamily,
    PH3ImageCurveStackLimits, PH3ImageStackProjectionStorage,
};
use super::final_layer::{
    PH3ImageFinalLayerCheckpoint, PH3ImageFinalLayerConfig, PH3ImageFinalLayerFamily,
    PH3ImageFinalLayerLimits, PH3ImageFinalLayerRequest,
};
use super::token_refiner::{
    PH3ImageTokenRefinerCheckpoint, PH3ImageTokenRefinerConfig, PH3ImageTokenRefinerFamily,
    PH3ImageTokenRefinerLimits,
};
use super::{
    PH3ImageBlockProjectionStorage, PH3ImageCurveBlockRequest, PH3ImageCurveTimeEmbedding,
    PH3ImageModality, PH3ImageReferenceImage, PH3ImageRequestLayout, PH3ImageTransformerConfig,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod cache;
/// Attention arithmetic for the complete DiT, including text refinement.
/// Vendor SDPA retains BF16 external tensors but owns its internal rounding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageDenoiserAttentionPolicy {
    #[default]
    Sequential,
    CudnnSdpaExperimentalV1,
}
mod checkpoint;
mod config;
mod fragment;
mod inputs;
mod program;
mod projection;
mod refine_only;
mod refinement_host_requirements;
#[cfg(any(test, feature = "test-support"))]
pub(in crate::p_h3_image) mod test_fixture;
pub use cache::PH3ImageDiTConditioningKey;
pub use cache::PH3ImageRefinedTextCache;
pub(super) use checkpoint::PH3ImageDiTCheckpoint;
pub(super) use config::{PH3ImageDiTConfig, PH3ImageDiTPhase, PH3ImageDiTRequest};
pub(super) use inputs::{
    PH3ImageDiTInputAdapter, PH3ImageDiTReference, PH3ImageDiTSample, PH3ImageDiTTextInput,
    PreparedPH3ImageDiTInputs,
};
pub(super) use refine_only::PH3ImageTextRefinement;
pub use refine_only::{PH3ImageRefinementDestination, PH3ImageTextRefinementError};
pub(super) use refinement_host_requirements::query as query_refinement_host_requirements;
pub use refinement_host_requirements::{
    PH3ImageRefinementCacheHostRequirement, PH3ImageRefinementHostRequirements,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct PH3ImageDiTLimits {
    pub max_metadata_bytes: usize,
    pub max_plan_nodes: usize,
    pub max_stored_bytes: u64,
}
impl PH3ImageDiTLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.max_metadata_bytes == 0 || self.max_plan_nodes == 0 || self.max_stored_bytes == 0 {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
    fn stack(self) -> PH3ImageCurveStackLimits {
        PH3ImageCurveStackLimits {
            max_metadata_bytes: self.max_metadata_bytes,
            max_plan_nodes: self.max_plan_nodes,
            max_stored_bytes: self.max_stored_bytes,
        }
    }
    fn refiner(self) -> PH3ImageTokenRefinerLimits {
        PH3ImageTokenRefinerLimits {
            max_metadata_bytes: self.max_metadata_bytes,
            max_plan_nodes: self.max_plan_nodes,
            max_stored_bytes: self.max_stored_bytes,
        }
    }
    fn final_layer(self) -> PH3ImageFinalLayerLimits {
        PH3ImageFinalLayerLimits {
            max_metadata_bytes: self.max_metadata_bytes,
            max_plan_nodes: self.max_plan_nodes,
            max_stored_bytes: self.max_stored_bytes,
        }
    }
}
#[derive(Clone, Debug)]
pub(super) struct PH3ImageDiTFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageDiTLimits,
}
impl PH3ImageDiTFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.dit_forward";
    pub const FIRST_PROFILE_ID: &'static str =
        "p_h3_image.dit_forward.cuda.with_text_refiner.comfy_mixed.positive_bf16_velocity.v1";
    pub const STEADY_PROFILE_ID: &'static str =
        "p_h3_image.dit_forward.cuda.refined_text.comfy_mixed.positive_bf16_velocity.v1";
    pub const REFINE_PROFILE_ID: &'static str =
        "p_h3_image.dit_forward.cuda.refine_only.shared_complete_weights.v1";
    pub const RAW_TEXT_ID: &'static str = PH3ImageTokenRefinerFamily::INPUT_ID;
    pub const REFINED_TEXT_ID: &'static str = PH3ImageTokenRefinerFamily::OUTPUT_ID;
    pub const IMAGE_PATCHES_ID: &'static str = "value.p_h3_image.dit_forward.image_patches_f32";
    pub const AUDIO_PATCHES_ID: &'static str = "value.p_h3_image.dit_forward.audio_rows_f32";
    pub const BASIS_ID: &'static str = PH3ImageCurveStackFamily::BASIS_ID;
    pub const COSINE_ID: &'static str = PH3ImageCurveStackFamily::COSINE_ID;
    pub const SINE_ID: &'static str = PH3ImageCurveStackFamily::SINE_ID;
    pub const IMAGE_VELOCITY_ID: &'static str = PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID;
    pub const AUDIO_VELOCITY_ID: &'static str = PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID;
    pub fn new(limits: PH3ImageDiTLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.p_h3_image.dit_forward.comfy_mixed")?,
            limits,
        })
    }
    pub fn cuda_profile_id(config: &PH3ImageDiTConfig) -> &'static str {
        if config.attention_policy == PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1 {
            return match config.phase {
                PH3ImageDiTPhase::WithTextRefiner => {
                    "p_h3_image.dit_forward.cuda.with_text_refiner.comfy_mixed.cudnn_sdpa.v1"
                }
                PH3ImageDiTPhase::RefinedText => {
                    "p_h3_image.dit_forward.cuda.refined_text.comfy_mixed.cudnn_sdpa.v1"
                }
                PH3ImageDiTPhase::RefineOnly => {
                    "p_h3_image.dit_forward.cuda.refine_only.comfy_mixed.cudnn_sdpa.v1"
                }
            };
        }
        match config.phase {
            PH3ImageDiTPhase::WithTextRefiner => Self::FIRST_PROFILE_ID,
            PH3ImageDiTPhase::RefinedText => Self::STEADY_PROFILE_ID,
            PH3ImageDiTPhase::RefineOnly => Self::REFINE_PROFILE_ID,
        }
    }
    pub fn compile_options(
        &self,
        c: &PH3ImageDiTConfig,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(program::compose(c, self.limits)?.options)
    }
}
impl ModelFamilyProvider for PH3ImageDiTFamily {
    type Config = PH3ImageDiTConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(r.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(r, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *r {
            return Err(invalid("configuration must be canonical"));
        }
        program::compose(c, self.limits)?;
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
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(
            program::compose(c, self.limits)?.schema,
        ))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![program::compose(c, self.limits)?.profile],
            vec![NumericalProfileId::new(Self::cuda_profile_id(c)).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let b = program::compose(c, self.limits)?;
        if &b.profile != p {
            return Err(invalid("numerical profile differs"));
        }
        Ok(b.program)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("P-H3 DiT forward: {}", reason.into()),
    }
}
#[cfg(test)]
pub(crate) mod tests;
