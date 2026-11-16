use super::*;
pub struct FastH3DitBackboneFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3DitBackboneLimits,
}
impl FastH3DitBackboneFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.dit_backbone";
    pub const PROFILE_ID: &'static str = "fast_h3.dit_backbone.roles.vsa_prefix_sequential.v1";
    pub const VENDOR_FUSED_PROFILE_ID: &'static str =
        "fast_h3.dit_backbone.roles.vsa_prefix_vendor.v1";
    pub const HIDDEN_ID: &'static str = "value.fast_h3.dit_backbone.hidden";
    pub const COSINE_ID: &'static str = FastH3DitBlockFamily::COSINE_ID;
    pub const SINE_ID: &'static str = FastH3DitBlockFamily::SINE_ID;
    pub const OUTPUT_ID: &'static str = FastH3DitBlockFamily::OUTPUT_ID;
    pub fn new(limits: FastH3DitBackboneLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.dit_backbone")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3DitBackboneFamily {
    type Config = FastH3DitBackboneConfig;
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
            return Err(invalid("noncanonical stack configuration"));
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
            return Err(invalid("unknown stack profile"));
        }
        Ok(built.program)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
