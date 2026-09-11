use super::*;

pub(super) struct Family {
    pub family_id: ModelFamilyId,
    pub states: Vec<StateSpec>,
    pub declare_inputs: bool,
    pub conditioning: bool,
}

impl ModelFamilyProvider for Family {
    type Config = TestConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }
    fn validate_config_identity(&self, raw: &Value, config: &TestConfig) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &TestConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        TestFamily.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<TestConfig, VNextError> {
        TestFamily.parse_config(raw)
    }
    fn weight_schema(&self, config: &TestConfig) -> Result<WeightSchema, VNextError> {
        TestFamily.weight_schema(config)
    }
    fn numerical_profiles(&self, _: &TestConfig) -> Result<FamilyNumericalProfiles, VNextError> {
        fixture_f32_profiles(
            self.family_id(),
            &["value.output"],
            &["operation.main"],
            self.states.clone(),
        )
    }
    fn semantic_metadata(&self, config: &TestConfig) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
    fn semantic_program(
        &self,
        config: &TestConfig,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let original = TestFamily.semantic_program(config, profile)?;
        let mut inputs = vec![id("value.input")];
        let mut bindings = vec![id("value.input"), id("value.weight")];
        bindings.extend(self.states.iter().map(|state| state.value_id.clone()));
        if self.conditioning {
            inputs.push(id("value.conditioning"));
            bindings.push(id("value.conditioning"));
        }
        let program = ModelProgram::new(
            self.family_id.clone(),
            inputs,
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.main"),
                    operation_id: id("operation.main"),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::tokens(id("value.input"), 0),
                    inputs: bindings,
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::new(),
                }],
            }],
            self.states.clone(),
            original.weights().to_vec(),
            vec![id("value.output")],
        )?;
        if self.declare_inputs {
            program.with_checkpoint_inputs(ProgramCheckpointInputs::new(
                id("value.input"),
                if self.conditioning {
                    BTreeSet::from([id("value.conditioning")])
                } else {
                    BTreeSet::new()
                },
            )?)
        } else {
            Ok(program)
        }
    }
}
