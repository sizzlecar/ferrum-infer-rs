use super::*;
pub struct FastH3VsaRowTransportFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3VsaRowTransportLimits,
}
impl FastH3VsaRowTransportFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.vsa_rows";
    pub const PROFILE_ID: &'static str = "fast_h3.vsa_rows.original_map.exact_bits.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.vsa_rows.source";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.vsa_rows.output";
    pub fn new(limits: FastH3VsaRowTransportLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.vsa_rows")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3VsaRowTransportFamily {
    type Config = FastH3VsaRowTransportConfig;
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
            return Err(invalid("noncanonical row transport config"));
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
        let profile = NumericalExecutionProfile {
            id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(Self::OUTPUT_ID)?,
            boundaries: BTreeMap::from([(id(Self::OUTPUT_ID)?, ElementType::Bf16)]),
            states: vec![],
            kv_storage: vec![],
            operations: vec![NumericalOperationContract {
                operation_id: id(ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID)?,
                version: ContractVersion::new(1, 0),
                multiplication_type: None,
                accumulation_type: None,
                staged_arithmetic: None,
                composite_arithmetic: None,
            }],
        };
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![profile.clone()],
            vec![profile.id],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown row transport profile"));
        }
        let input: ProgramValueId = id(Self::INPUT_ID)?;
        let output: ProgramValueId = id(Self::OUTPUT_ID)?;
        let map: ProgramValueId = id("value.fast_h3.vsa_rows.map")?;
        ModelProgram::new(
            self.id.clone(),
            vec![input.clone()],
            vec![ProgramBlock {
                id: "block.fast_h3.vsa_rows".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fast_h3.vsa_rows.gather")?,
                    operation_id: id(ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![input.clone(), map.clone()],
                    outputs: vec![output.clone()],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.fast_h3.vsa_rows.map")?,
                value_id: map,
                tensor: tensor(vec![c.output_rows()], ElementType::I32),
            }],
            vec![output],
        )?
        .with_input_tensor_specs(BTreeMap::from([(
            input,
            tensor(
                vec![c.batches, c.source_rows(), c.columns],
                ElementType::Bf16,
            ),
        )]))
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
