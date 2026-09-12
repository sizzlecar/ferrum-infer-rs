use super::*;

pub(super) struct Family {
    pub family_id: ModelFamilyId,
    pub states: Vec<StateSpec>,
    pub declare_inputs: bool,
    pub conditioning: bool,
    pub output_only_input: Option<ProgramValueId>,
    pub output_only_feeds_state: bool,
    pub output_only_unknown_operation: bool,
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
        let mut activations = vec!["value.output"];
        let mut operations = vec!["operation.main"];
        if self.output_only_input.is_some() {
            activations.push("value.final");
            operations.push(self.output_operation_id());
        }
        fixture_f32_profiles(
            self.family_id(),
            &activations,
            &operations,
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
        if let Some(input) = &self.output_only_input {
            inputs.push(input.clone());
            if self.output_only_feeds_state {
                bindings.push(input.clone());
            }
        }
        let mut nodes = vec![ProgramNode {
            id: id("node.main"),
            operation_id: id("operation.main"),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(id("value.input"), 0),
            inputs: bindings,
            outputs: vec![id("value.output")],
            attributes: BTreeMap::new(),
        }];
        let output = if let Some(input) = &self.output_only_input {
            nodes.push(ProgramNode {
                id: id("node.output"),
                operation_id: id(self.output_operation_id()),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::tokens(id("value.output"), 0),
                inputs: vec![id("value.output"), input.clone()],
                outputs: vec![id("value.final")],
                attributes: BTreeMap::new(),
            });
            id("value.final")
        } else {
            id("value.output")
        };
        let program = ModelProgram::new(
            self.family_id.clone(),
            inputs,
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes,
            }],
            self.states.clone(),
            original.weights().to_vec(),
            vec![output],
        )?;
        if self.declare_inputs {
            program.with_checkpoint_inputs(
                ProgramCheckpointInputs::new(
                    id("value.input"),
                    if self.conditioning {
                        BTreeSet::from([id("value.conditioning")])
                    } else {
                        BTreeSet::new()
                    },
                )?
                .with_output_only_inputs(self.output_only_input.iter().cloned().collect())?,
            )
        } else {
            Ok(program)
        }
    }
}

impl Family {
    fn output_operation_id(&self) -> &'static str {
        if self.output_only_unknown_operation {
            "operation.unknown"
        } else {
            "operation.output"
        }
    }
}
