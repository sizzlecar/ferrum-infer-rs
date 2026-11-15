use super::*;
pub(super) trait Component:
    Clone + Serialize + serde::de::DeserializeOwned + Send + Sync + 'static
{
    const FAMILY_ID: &'static str;
    const PROFILE_ID: &'static str;
    const OUTPUT_ID: &'static str;
    fn validate(&self, limits: FastH3AudioComponentLimits) -> Result<(), VNextError>;
    fn schema(&self) -> Result<WeightSchema, VNextError>;
    fn build(
        &self,
        limits: FastH3AudioComponentLimits,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError>;
}
pub(super) struct Family<C: Component = EncoderConfig> {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3AudioComponentLimits,
    config: std::marker::PhantomData<C>,
}
impl<C: Component> Family<C> {
    pub(super) const FAMILY_ID: &'static str = C::FAMILY_ID;
    pub(super) const PROFILE_ID: &'static str = C::PROFILE_ID;
    pub(super) fn new(limits: FastH3AudioComponentLimits) -> Result<Self, VNextError> {
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id(format!("metadata.{}", C::FAMILY_ID))?,
            limits,
            config: std::marker::PhantomData,
        })
    }
}
impl<C: Component> ModelFamilyProvider for Family<C> {
    type Config = C;
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
            return Err(invalid("noncanonical audio encoder config"));
        }
        c.validate(self.limits)
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
        if serde_json::to_vec(raw).map_err(invalid)?.len() > self.limits.maximum_metadata_bytes {
            return Err(invalid("audio encoder metadata limit"));
        }
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        c.validate(self.limits)?;
        Ok(ModelWeights::Stored(c.schema()?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        c.validate(self.limits)?;
        let (program, specs) = c.build(self.limits)?;
        let mut operations = BTreeMap::new();
        for n in program.blocks().iter().flat_map(|b| &b.nodes) {
            operations.insert(
                n.operation_id.clone(),
                NumericalOperationContract {
                    operation_id: n.operation_id.clone(),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (n.operation_id.as_str()
                        != GROUPED_COLUMN_SLICE_F32_OPERATION_ID)
                        .then_some(ElementType::F32),
                    accumulation_type: (n.operation_id.as_str()
                        != GROUPED_COLUMN_SLICE_F32_OPERATION_ID)
                        .then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                },
            );
        }
        let profile = NumericalExecutionProfile {
            id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(C::OUTPUT_ID)?,
            boundaries: program
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|n| &n.outputs)
                .map(|v| (v.clone(), specs[v].element_type))
                .collect(),
            states: vec![],
            kv_storage: vec![],
            operations: operations.into_values().collect(),
        };
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![profile],
            vec![NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown audio encoder numerical profile"));
        }
        Ok(c.build(self.limits)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
