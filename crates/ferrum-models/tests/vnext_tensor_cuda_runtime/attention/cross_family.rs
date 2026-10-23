use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CrossConfig {
    pub query_rows: u64,
    pub key_value_rows: u64,
    pub columns: u64,
    pub attributes: BTreeMap<AttributeId, SemanticValue>,
}
impl CrossConfig {
    pub fn spec(&self, query: bool) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![
                if query {
                    self.query_rows
                } else {
                    self.key_value_rows
                },
                self.columns,
            ],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
    pub fn geometry(&self) -> Result<CrossAttentionBf16Geometry, VNextError> {
        let spec = |query| {
            let s = self.spec(query);
            ResolvedTensorSpec::new(s.dimensions, s.element_type, s.layout)
        };
        CrossAttentionBf16Geometry::new(
            &spec(true)?,
            &spec(false)?,
            &spec(false)?,
            &spec(true)?,
            &self.attributes,
        )
    }
}

pub(super) struct CrossFamily;
impl ModelFamilyProvider for CrossFamily {
    type Config = CrossConfig;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.tensor.cross_attention.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.tensor.cross_attention.fixture")])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &CrossConfig,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: "cross fixture requires canonical typed configuration".into(),
            });
        }
        config.geometry()?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &CrossConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.cuda.tensor.cross_attention.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<CrossConfig, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &CrossConfig) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &CrossConfig) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).unwrap();
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
                    operation_id: id(CROSS_ATTENTION_FUSED_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        config: &CrossConfig,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            INPUTS.into_iter().map(id).collect(),
            vec![ProgramBlock {
                id: "block.tensor.cross_attention".into(),
                nodes: vec![ProgramNode {
                    id: id("node.tensor.cross_attention"),
                    operation_id: id(CROSS_ATTENTION_FUSED_BF16_OPERATION_ID),
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
    fn semantic_metadata(&self, _: &CrossConfig) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
