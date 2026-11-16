//! One immutable composition supplies both the trusted family and compiler
//! geometry. Core still validates the complete definition and selected program.
use super::*;
use serde::{de::DeserializeOwned, Serialize};

struct ComposedFamily<C> {
    metadata: ExternalModelMetadataId,
    config: serde_json::Value,
    schema: WeightSchema,
    program: ModelProgram,
    profile: NumericalExecutionProfile,
    typed: std::marker::PhantomData<C>,
}

impl StoredComposition {
    pub fn prepare_family<C>(
        self,
        config: &C,
        metadata: ExternalModelMetadataId,
    ) -> Result<(PreparedModelFamily, ProgramPlanCompileOptions), VNextError>
    where
        C: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    {
        let raw = serde_json::to_value(config).map_err(invalid)?;
        let profile = self.profile.id.clone();
        let provider = ComposedFamily::<C> {
            metadata,
            config: raw.clone(),
            schema: self.schema,
            program: self.program,
            profile: self.profile,
            typed: std::marker::PhantomData,
        };
        let family = TypedFamilyRegistration::new(provider).prepare_with_profile(&raw, &profile)?;
        Ok((family, self.options))
    }
}

impl<C> ModelFamilyProvider for ComposedFamily<C>
where
    C: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    type Config = C;
    fn family_id(&self) -> &ModelFamilyId {
        &self.profile.family_id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &C,
    ) -> Result<(), VNextError> {
        if *raw != self.config || serde_json::to_value(config).map_err(invalid)? != *raw {
            return Err(invalid(
                "configuration differs from its immutable composition",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &C,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<C, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &C) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(self.schema.clone()))
    }
    fn numerical_profiles(&self, _: &C) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![self.profile.clone()],
            vec![self.profile.id.clone()],
        )
    }
    fn semantic_program(
        &self,
        _: &C,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if profile != &self.profile {
            return Err(invalid("profile differs from its immutable composition"));
        }
        Ok(self.program.clone())
    }
    fn semantic_metadata(&self, _: &C) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
