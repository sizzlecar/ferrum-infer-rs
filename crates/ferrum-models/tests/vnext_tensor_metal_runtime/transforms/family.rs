use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Fixture {
    operation: OperationId,
    pub inputs: Vec<ProgramTensorSpec>,
    pub attributes: BTreeMap<AttributeId, SemanticValue>,
    pub output: Option<ProgramTensorSpec>,
}
impl Fixture {
    pub fn new(
        kind: MetalTensorOperation,
        inputs: Vec<ProgramTensorSpec>,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Self {
        Self {
            operation: kind.contract().unwrap().descriptor().id.clone(),
            inputs,
            attributes,
            output: None,
        }
    }
    pub fn kind(&self) -> MetalTensorOperation {
        MetalTensorOperation::ALL
            .into_iter()
            .find(|kind| kind.contract().unwrap().descriptor().id == self.operation)
            .unwrap()
    }
    pub fn specs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        self.inputs
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, spec)| (input_id(i), spec))
            .collect()
    }
}
pub(super) struct TransformFamily;
impl ModelFamilyProvider for TransformFamily {
    type Config = Fixture;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.metal.transform.test"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.metal.transform.test")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Fixture) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw
            || !OPERATIONS
                .into_iter()
                .any(|kind| kind.contract().unwrap().descriptor().id == config.operation)
        {
            return Err(invalid("unknown transform fixture operation"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Fixture,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.metal.transform.test"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Fixture, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Fixture) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, config: &Fixture) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = id::<NumericalProfileId>(config.kind().profile());
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]),
                kv_storage: vec![],
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: config.operation.clone(),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (!matches!(
                        config.kind(),
                        MetalTensorOperation::RowGather | MetalTensorOperation::BatchedRowConcat
                    ))
                    .then_some(ElementType::F32),
                    accumulation_type: (!matches!(
                        config.kind(),
                        MetalTensorOperation::RowGather | MetalTensorOperation::BatchedRowConcat
                    ))
                    .then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        config: &Fixture,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let inputs: Vec<_> = (0..config.inputs.len()).map(input_id).collect();
        ModelProgram::new(
            self.family_id().clone(),
            inputs.clone(),
            vec![ProgramBlock {
                id: "block.metal.transform".into(),
                nodes: vec![ProgramNode {
                    id: id("node.metal.transform"),
                    operation_id: config.operation.clone(),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: vec![id(OUTPUT)],
                    attributes: config.attributes.clone(),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT)],
        )?
        .with_input_tensor_specs(config.specs())
    }
    fn semantic_metadata(&self, _: &Fixture) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
