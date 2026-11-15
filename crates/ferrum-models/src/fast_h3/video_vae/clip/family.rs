use super::*;
pub struct FastH3VideoClipFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3VideoClipLimits,
}
impl FastH3VideoClipFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.video_clip.original";
    pub const PROFILE_ID: &'static str =
        "fast_h3.video_clip.full_depth.original_bf16_autocast.f32_residual.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.video_clip.latent";
    pub const COSINE_ID: &'static str = "value.fast_h3.video_clip.cosine";
    pub const SINE_ID: &'static str = "value.fast_h3.video_clip.sine";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.video_clip.patches";
    pub fn new(limits: FastH3VideoClipLimits) -> Result<Self, VNextError> {
        FastH3VideoDecoderBlocksFamily::new(limits.blocks)?;
        if limits.maximum_plan_nodes == 0 || limits.maximum_activation_bytes == 0 {
            return Err(invalid("clip limits must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.fast_h3.video_clip.original")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3VideoClipFamily {
    type Config = FastH3VideoClipConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(invalid)? != *raw {
            return Err(invalid("noncanonical clip configuration"));
        }
        program::compose(c, self.limits).map(|_| ())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        if serde_json::to_vec(raw).map_err(invalid)?.len()
            > self.limits.blocks.maximum_metadata_bytes
        {
            return Err(invalid("clip metadata exceeds capacity"));
        }
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(
            program::compose(c, self.limits)?.schema,
        ))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let p = program::compose(c, self.limits)?.profile;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![p.clone()],
            vec![p.id],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let b = program::compose(c, self.limits)?;
        if p != &b.profile {
            return Err(invalid("unknown clip numerical profile"));
        }
        Ok(b.program)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
