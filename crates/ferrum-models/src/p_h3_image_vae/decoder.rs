//! One Stored, device-connected single-frame VAE decoder. Latent normalization
//! and image placement use the separately managed image-layout boundary.
//! The numerical route is explicitly F32/A8 with full F32 attention, not the
//! default upstream F16 or opportunistic INT8-attention route.
use super::decoder_block::{
    PH3ImageVaeDecoderBlockCheckpoint, PH3ImageVaeDecoderBlockConfig,
    PH3ImageVaeDecoderBlockExecutionPreset, PH3ImageVaeDecoderBlockFamily,
    PH3ImageVaeDecoderBlockLimits,
};
use super::PH3ImageVaeDecoderConfig;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod checkpoint;
pub(super) mod checkpoint_source;
mod compose;
mod config;
mod outer;
mod remap;
pub(crate) use checkpoint::PH3ImageVaeDecoderCheckpoint;
pub(crate) use config::PH3ImageVaeDecoderExecutionConfig;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PH3ImageVaeDecoderLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
impl PH3ImageVaeDecoderLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.max_metadata_bytes == 0 || self.max_stored_bytes == 0 || self.max_plan_nodes == 0 {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
    fn block(self) -> PH3ImageVaeDecoderBlockLimits {
        PH3ImageVaeDecoderBlockLimits {
            max_metadata_bytes: self.max_metadata_bytes,
            max_stored_bytes: self.max_stored_bytes,
            max_plan_nodes: self.max_plan_nodes,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct PH3ImageVaeDecoderFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageVaeDecoderLimits,
}
impl PH3ImageVaeDecoderFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.vae.decoder";
    pub const CUDA_PROFILE_ID: &'static str =
        "p_h3_image.vae.decoder.cuda.f32_a8.full_attention.two_pass_layernorm.v1";
    pub const INPUT_ID: &'static str = "value.p_h3_image.vae.decoder.latent_rows";
    pub const COSINE_ID: &'static str = "value.p_h3_image.vae.decoder.cosine";
    pub const SINE_ID: &'static str = "value.p_h3_image.vae.decoder.sine";
    pub const OUTPUT_ID: &'static str = "value.p_h3_image.vae.decoder.patch_output";
    pub fn new(limits: PH3ImageVaeDecoderLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.p_h3_image.vae.decoder.comfy_int8")?,
            limits,
        })
    }
    pub fn compile_options(
        &self,
        c: &PH3ImageVaeDecoderExecutionConfig,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(compose::build(c, self.limits)?.options)
    }
}
impl ModelFamilyProvider for PH3ImageVaeDecoderFamily {
    type Config = PH3ImageVaeDecoderExecutionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("configuration must be canonical"));
        }
        compose::admit(c, self.limits)?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(compose::build(c, self.limits)?.schema))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![compose::build(c, self.limits)?.profile],
            vec![NumericalProfileId::new(Self::CUDA_PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let built = compose::build(c, self.limits)?;
        if &built.profile != p {
            return Err(invalid("decoder profile boundaries/arithmetic differ"));
        }
        Ok(built.program)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("P-H3 VAE decoder: {}", reason.into()),
    }
}
#[cfg(any(test, feature = "test-support"))]
pub(in crate::p_h3_image_vae) mod test_fixture;
#[cfg(test)]
pub(super) mod tests;
