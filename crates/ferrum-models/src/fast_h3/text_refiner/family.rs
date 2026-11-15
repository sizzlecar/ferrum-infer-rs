use super::*;
pub struct FastH3TextRefinerFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3TextRefinerLimits,
}
impl FastH3TextRefinerFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.text_refiner";
    pub const PROFILE_ID: &'static str = "fast_h3.text_refiner.dense_bf16_staged.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.text_refiner.projected_text";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.text_refiner.refined_text";
    pub fn new(limits: FastH3TextRefinerLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.text_refiner")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3TextRefinerFamily {
    type Config = FastH3TextRefinerConfig;
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
        let (program, specs) = program::build(c)?;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Self::OUTPUT_ID)?,
                boundaries: program
                    .blocks()
                    .iter()
                    .flat_map(|b| &b.nodes)
                    .flat_map(|n| &n.outputs)
                    .map(|v| (v.clone(), specs[v].element_type))
                    .collect(),
                states: vec![],
                kv_storage: vec![],
                operations: program
                    .blocks()
                    .iter()
                    .flat_map(|b| &b.nodes)
                    .map(|n| n.operation_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .map(|operation_id| {
                        let (multiplication_type, accumulation_type) = match operation_id.as_str() {
                            DENSE_LINEAR_BF16_OPERATION_ID => {
                                (Some(ElementType::Bf16), Some(ElementType::F32))
                            }
                            SEGMENTED_ATTENTION_BF16_OPERATION_ID => {
                                (Some(ElementType::F32), Some(ElementType::F32))
                            }
                            _ => (None, None),
                        };
                        NumericalOperationContract {
                            operation_id,
                            version: ContractVersion::new(1, 0),
                            multiplication_type,
                            accumulation_type,
                            staged_arithmetic: None,
                            composite_arithmetic: None,
                        }
                    })
                    .collect(),
            }],
            vec![NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown numerical profile"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
