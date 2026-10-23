use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AttentionConfig {
    pub rows: u64,
    pub columns: u64,
    pub attributes: BTreeMap<AttributeId, SemanticValue>,
}
impl AttentionConfig {
    pub(super) fn spec(&self) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![self.rows, self.columns],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
}

pub(super) struct AttentionFamily;
impl ModelFamilyProvider for AttentionFamily {
    type Config = AttentionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.tensor.structured_attention.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.tensor.structured_attention.fixture")])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &AttentionConfig,
    ) -> Result<(), VNextError> {
        let invalid = |reason| VNextError::InvalidModelConfig {
            family_id: self.family_id().to_string(),
            field: "config".into(),
            reason,
        };
        if config.rows == 0 || config.columns == 0 || serde_json::to_value(config).unwrap() != *raw
        {
            return Err(invalid(
                "fixture requires canonical nonempty tensor shape and explicit attributes".into(),
            ));
        }
        let spec = ResolvedTensorSpec::new(
            vec![config.rows, config.columns],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        StructuredAttentionBf16Geometry::new(&spec, &spec, &spec, &spec, &config.attributes)?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &AttentionConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.cuda.tensor.structured_attention.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<AttentionConfig, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &AttentionConfig) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(
        &self,
        _: &AttentionConfig,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile =
            NumericalProfileId::new(CUDA_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_ID).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT_ID),
                boundaries: BTreeMap::from([(id(OUTPUT_ID), ElementType::Bf16)]),
                kv_storage: vec![],
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(STRUCTURED_ATTENTION_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        config: &AttentionConfig,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            INPUTS.into_iter().map(id).collect(),
            vec![ProgramBlock {
                id: "block.tensor.structured_attention".into(),
                nodes: vec![ProgramNode {
                    id: id("node.tensor.structured_attention"),
                    operation_id: id(STRUCTURED_ATTENTION_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: INPUTS.into_iter().map(id).collect(),
                    outputs: vec![id(OUTPUT_ID)],
                    attributes: config.attributes.clone(),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT_ID)],
        )
    }
    fn semantic_metadata(&self, _: &AttentionConfig) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
