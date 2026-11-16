use super::*;
pub struct FastH3DenoisingFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3DenoisingLimits,
}
impl FastH3DenoisingFamily {
    pub(super) const METADATA_ID: &'static str = "metadata.fast_h3.denoising";
    pub const FAMILY_ID: &'static str = "family.fast_h3.denoising";
    pub const PROFILE_ID: &'static str =
        "fast_h3.denoising.media_f32.packed_bf16.roles.prefix_sequential.v1";
    pub const VENDOR_FUSED_PROFILE_ID: &'static str =
        "fast_h3.denoising.media_f32.packed_bf16.roles.prefix_vendor.v1";
    pub const VIDEO_INPUT_ID: &'static str = FastH3InputProjectionFamily::VIDEO_INPUT_ID;
    pub const AUDIO_INPUT_ID: &'static str = FastH3InputProjectionFamily::AUDIO_INPUT_ID;
    pub const TEXT_INPUT_ID: &'static str = FastH3InputPackingFamily::REFINED_TEXT_ID;
    pub const COSINE_ID: &'static str = FastH3DitBackboneFamily::COSINE_ID;
    pub const SINE_ID: &'static str = FastH3DitBackboneFamily::SINE_ID;
    pub const OUTPUT_ID: &'static str = FastH3DitBackboneFamily::OUTPUT_ID;
    pub fn new(limits: FastH3DenoisingLimits) -> Result<Self, VNextError> {
        FastH3InputProjectionFamily::new(limits.input)?;
        FastH3DitBackboneFamily::new(limits.backbone)?;
        if limits.maximum_plan_nodes == 0
            || limits.maximum_stored_bytes == 0
            || limits.maximum_metadata_bytes == 0
        {
            return Err(invalid("positive composition capacities required"));
        }
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id(Self::METADATA_ID)?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3DenoisingFamily {
    type Config = FastH3DenoisingConfig;
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
            return Err(invalid("noncanonical configuration"));
        }
        compose(c, self.limits).map(|_| ())
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
        Ok(ModelWeights::Stored(compose(c, self.limits)?.schema))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let p = compose(c, self.limits)?.profile;
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
        let built = compose(c, self.limits)?;
        if &built.profile != p {
            return Err(invalid("unknown profile"));
        }
        Ok(built.program)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
