use super::*;
pub struct FastH3OutputTimeFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3OutputConditioningLimits,
}
impl FastH3OutputTimeFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.output_time";
    pub const PROFILE_ID: &'static str = "fast_h3.output.time_projection.silu_f32.bf16_affine.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.output_time.embedding";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.output_time.table";

    pub fn new(limits: FastH3OutputConditioningLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.output_time")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3OutputTimeFamily {
    type Config = FastH3OutputTimeConfig;
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
            return Err(invalid("noncanonical output config"));
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
        profiles(&self.id, Self::PROFILE_ID, Self::OUTPUT_ID, c.program()?)
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown final output profile"));
        }
        Ok(c.program()?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
pub struct FastH3OutputNormFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3OutputConditioningLimits,
}
impl FastH3OutputNormFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.output_norm";
    pub const PROFILE_ID: &'static str = "fast_h3.output.rms_bf16.staged_modulation.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.output_norm.hidden";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.output_norm.normalized";
    pub const TABLE_ID: &'static str = "value.fast_h3.output_norm.table";
    pub fn new(limits: FastH3OutputConditioningLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.output_norm")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3OutputNormFamily {
    type Config = FastH3OutputNormConfig;
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
            return Err(invalid("noncanonical output config"));
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
        profiles(&self.id, Self::PROFILE_ID, Self::OUTPUT_ID, c.program()?)
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown final output profile"));
        }
        Ok(c.program()?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn profiles(
    family: &ModelFamilyId,
    profile: &str,
    output: &str,
    (program, specs): (ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>),
) -> Result<FamilyNumericalProfiles, VNextError> {
    let operations = program
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .map(|n| n.operation_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|operation_id| {
            let (mul, acc) = match operation_id.as_str() {
                AFFINE_LINEAR_BF16_OPERATION_ID => {
                    (Some(ElementType::Bf16), Some(ElementType::F32))
                }
                SILU_F32_OPERATION_ID | ROW_SELECTED_MODULATION_BF16_OPERATION_ID => {
                    (Some(ElementType::F32), None)
                }
                HEAD_RMS_NORM_BF16_OPERATION_ID => (Some(ElementType::F32), Some(ElementType::F32)),
                _ => (None, None),
            };
            NumericalOperationContract {
                operation_id,
                version: ContractVersion::new(1, 0),
                multiplication_type: mul,
                accumulation_type: acc,
                staged_arithmetic: None,
                composite_arithmetic: None,
            }
        })
        .collect();
    FamilyNumericalProfiles::new(
        family,
        ContractVersion::new(1, 0),
        vec![NumericalExecutionProfile {
            id: NumericalProfileId::new(profile).map_err(invalid)?,
            family_id: family.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(output)?,
            boundaries: program
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|n| &n.outputs)
                .map(|v| (v.clone(), specs[v].element_type))
                .collect(),
            states: vec![],
            kv_storage: vec![],
            operations,
        }],
        vec![NumericalProfileId::new(profile).map_err(invalid)?],
    )
}
