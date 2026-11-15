use super::*;
pub struct FastH3VideoDecoderBlocksFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3VideoDecoderBlocksLimits,
}
impl FastH3VideoDecoderBlocksFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.video_decoder.blocks.original";
    pub const PROFILE_ID: &'static str =
        "fast_h3.video_decoder.f32_residual.staged_bf16.unit_nvfp4.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.video_decoder.input";
    pub const COSINE_ID: &'static str = "value.fast_h3.video_decoder.cosine";
    pub const SINE_ID: &'static str = "value.fast_h3.video_decoder.sine";
    pub fn new(limits: FastH3VideoDecoderBlocksLimits) -> Result<Self, VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_blocks == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
        {
            return Err(invalid("block limits must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new(
                "metadata.fast_h3.video_decoder.blocks.original",
            )?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3VideoDecoderBlocksFamily {
    type Config = FastH3VideoDecoderBlocksConfig;
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
            return Err(invalid("noncanonical block configuration"));
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
            return Err(invalid("block configuration exceeds metadata capacity"));
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
        let (program, specs) = program::build(c)?;
        let mut operations = BTreeMap::new();
        for node in program.blocks().iter().flat_map(|b| &b.nodes) {
            let no_arithmetic = matches!(
                node.operation_id.as_str(),
                CAST_F32_TO_BF16_OPERATION_ID
                    | CAST_BF16_TO_F32_OPERATION_ID
                    | ROW_GATHER_BF16_OPERATION_ID
            );
            operations.insert(
                node.operation_id.clone(),
                NumericalOperationContract {
                    operation_id: node.operation_id.clone(),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (!no_arithmetic).then_some(ElementType::F32),
                    accumulation_type: (!no_arithmetic).then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                },
            );
        }
        let profile = NumericalExecutionProfile {
            id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: c.output_id()?,
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
            return Err(invalid("unknown block numerical profile"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
