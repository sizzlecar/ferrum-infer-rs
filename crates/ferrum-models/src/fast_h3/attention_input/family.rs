use super::*;
pub struct FastH3AttentionInputFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3AttentionInputLimits,
}
impl FastH3AttentionInputFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.attention_input";
    pub const PROFILE_ID: &'static str =
        "fast_h3.attention_input.tile_qkv.packed_gate.qk_single_round.pool_ascending_f32.v1";
    pub const HIDDEN_ID: &'static str = "value.fast_h3.attention_input.hidden";
    pub const COSINE_ID: &'static str = "value.fast_h3.attention_input.cosine";
    pub const SINE_ID: &'static str = "value.fast_h3.attention_input.sine";
    pub const QUERY_ID: &'static str = "value.fast_h3.attention_input.tiled_query";
    pub const KEY_ID: &'static str = "value.fast_h3.attention_input.tiled_key";
    pub const VALUE_ID: &'static str = "value.fast_h3.attention_input.tiled_value";
    pub const GATE_ID: &'static str = "value.fast_h3.attention_input.packed_gate";
    pub const POOLED_QUERY_ID: &'static str = "value.fast_h3.vsa.pool.pooled.query";
    pub const POOLED_KEY_ID: &'static str = "value.fast_h3.vsa.pool.pooled.key";
    pub const POOLED_VALUE_ID: &'static str = "value.fast_h3.vsa.pool.pooled.value";
    pub fn new(limits: FastH3AttentionInputLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.attention_input")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3AttentionInputFamily {
    type Config = FastH3AttentionInputConfig;
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
