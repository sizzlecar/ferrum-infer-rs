use super::*;
pub struct FastH3AttentionCoreFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3AttentionCoreLimits,
}
impl FastH3AttentionCoreFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.attention_core";
    pub const PROFILE_ID: &'static str =
        "fast_h3.attention_core.shared_scores.center.fp8_prefix_sequential.sage_fp4.fma.to_out.v1";
    pub const VENDOR_FUSED_PROFILE_ID: &'static str =
        "fast_h3.attention_core.shared_scores.center.fp8_prefix_vendor.sage_fp4.fma.to_out.v1";
    pub const QUERY_ID: &'static str = FastH3AttentionInputFamily::QUERY_ID;
    pub const KEY_ID: &'static str = FastH3AttentionInputFamily::KEY_ID;
    pub const VALUE_ID: &'static str = FastH3AttentionInputFamily::VALUE_ID;
    pub const GATE_ID: &'static str = FastH3AttentionInputFamily::GATE_ID;
    pub const POOLED_QUERY_ID: &'static str = FastH3AttentionInputFamily::POOLED_QUERY_ID;
    pub const POOLED_KEY_ID: &'static str = FastH3AttentionInputFamily::POOLED_KEY_ID;
    pub const POOLED_VALUE_ID: &'static str = FastH3AttentionInputFamily::POOLED_VALUE_ID;
    pub const OUTPUT_ID: &'static str = FastH3AttentionOutputFamily::OUTPUT_ID;
    pub fn new(limits: FastH3AttentionCoreLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.attention_core")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3AttentionCoreFamily {
    type Config = FastH3AttentionCoreConfig;
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
        let built = program::compose(c, self.limits)?;
        if &built.profile != p {
            return Err(invalid("unknown numerical profile"));
        }
        Ok(built.program)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
