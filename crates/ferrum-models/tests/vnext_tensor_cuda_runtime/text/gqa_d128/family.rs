use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GqaConfig {
    pub rows: u64,
    pub query_columns: u64,
    pub kv_columns: u64,
    pub attributes: BTreeMap<AttributeId, SemanticValue>,
}

impl GqaConfig {
    pub(super) fn wide() -> Self {
        Self::new(
            10,
            4096,
            1024,
            5,
            [
                vec![5, 4],
                vec![1, 1, 1, 1, 1, 2, 1, 1, 1],
                vec![1, 0, 1, 0, 1, 0, 1, 0, 1],
            ],
        )
    }
    pub(super) fn tail() -> Self {
        Self::new(3, 128, 128, 3, [vec![1], vec![3], vec![1]])
    }
    fn new(
        rows: u64,
        query_columns: u64,
        kv_columns: u64,
        sequence: u64,
        runs: [Vec<i64>; 3],
    ) -> Self {
        let mut attributes = BTreeMap::from([
            (id("sequence_length"), SemanticValue::Unsigned(sequence)),
            (id("head_dim"), SemanticValue::Unsigned(128)),
            (
                id("scale"),
                SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
            ),
            (
                id("segment_lengths"),
                SemanticValue::Integers(vec![sequence as i64]),
            ),
            (
                id("segment_bidirectional"),
                SemanticValue::Integers(vec![0]),
            ),
        ]);
        for (name, values) in ["key_run_counts", "key_run_lengths", "key_run_valid"]
            .into_iter()
            .zip(runs)
        {
            attributes.insert(id(name), SemanticValue::Integers(values));
        }
        Self {
            rows,
            query_columns,
            kv_columns,
            attributes,
        }
    }
    pub(super) fn spec(&self, query: bool) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![
                self.rows,
                if query {
                    self.query_columns
                } else {
                    self.kv_columns
                },
            ],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
    pub(super) fn inputs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        PORTS
            .into_iter()
            .enumerate()
            .map(|(i, name)| (id(name), self.spec(i == 0)))
            .collect()
    }
}

pub(super) struct GqaFamily;
impl ModelFamilyProvider for GqaFamily {
    type Config = GqaConfig;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.gqa_d128.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.gqa_d128.fixture")])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: "GQA fixture requires canonical typed geometry".into(),
            });
        }
        let q = input_spec(config.spec(true).dimensions, ElementType::Bf16);
        let kv = input_spec(config.spec(false).dimensions, ElementType::Bf16);
        StructuredGqaAttentionBf16Geometry::new(&q, &kv, &kv, &q, &config.attributes)?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.cuda.gqa_d128.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(TEXT_PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(ATTENDED),
                boundaries: BTreeMap::from([(id(ATTENDED), ElementType::Bf16)]),
                kv_storage: vec![],
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID),
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
        config: &Self::Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            PORTS.into_iter().map(id).collect(),
            vec![ProgramBlock {
                id: "block.cuda.gqa_d128".into(),
                nodes: vec![ProgramNode {
                    id: id("node.cuda.gqa_d128"),
                    operation_id: id(STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: PORTS.into_iter().map(id).collect(),
                    outputs: vec![id(ATTENDED)],
                    attributes: config.attributes.clone(),
                }],
            }],
            vec![],
            vec![],
            vec![id(ATTENDED)],
        )?
        .with_input_tensor_specs(config.inputs())
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
