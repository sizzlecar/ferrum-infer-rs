use super::*;
pub struct FastH3FinalOutputFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3FinalOutputLimits,
}
impl FastH3FinalOutputFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.final_output";
    pub const PROFILE_ID: &'static str =
        "fast_h3.final_output.staged_bf16.select.promote.pdd_f32.v1";
    pub const INPUT_ID: &'static str = FastH3OutputNormFamily::INPUT_ID;
    pub const TABLE_ID: &'static str = FastH3OutputNormFamily::TABLE_ID;
    pub const VIDEO_OUTPUT_ID: &'static str = "value.fast_h3.pdd_projection.velocity.video";
    pub const AUDIO_OUTPUT_ID: &'static str = "value.fast_h3.pdd_projection.velocity.audio";
    pub fn new(limits: FastH3FinalOutputLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.final_output")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3FinalOutputFamily {
    type Config = FastH3FinalOutputConfig;
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
            return Err(invalid("noncanonical final output configuration"));
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
        let composed = program::compose(c, self.limits)?;
        if &composed.profile != p {
            return Err(invalid("unknown final output numerical profile"));
        }
        Ok(composed.program)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
