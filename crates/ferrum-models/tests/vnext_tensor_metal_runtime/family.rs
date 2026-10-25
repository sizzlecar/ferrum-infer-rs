use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    operation: OperationId,
    pub rows: u64,
    pub width: u64,
    pub row_broadcast: bool,
    pub outputs: Option<u64>,
}
impl Config {
    pub fn new(kind: MetalTensorOperation, rows: u64, width: u64, row_broadcast: bool) -> Self {
        Self {
            operation: kind.contract().unwrap().descriptor().id.clone(),
            rows,
            width,
            row_broadcast,
            outputs: None,
        }
    }
    pub fn kind(&self) -> MetalTensorOperation {
        MetalTensorOperation::ALL
            .into_iter()
            .find(|kind| kind.contract().unwrap().descriptor().id == self.operation)
            .unwrap()
    }
    pub fn input_count(&self) -> usize {
        use MetalTensorOperation::*;
        match self.kind() {
            Silu | Tanh | GeluTanh => 1,
            SiluMul | ModulationScale | PlainResidualAdd => 2,
            GatedResidual => 3,
            DenseLinear => 2,
            BlockLinear => unreachable!("stored weights use the block fixture"),
        }
    }
    pub fn dimensions(&self, port: usize) -> Vec<u64> {
        if self.kind() == MetalTensorOperation::DenseLinear && port == 1 {
            return vec![self.outputs.unwrap(), self.width];
        }
        vec![
            if port == 1 && self.row_broadcast {
                1
            } else {
                self.rows
            },
            self.width,
        ]
    }
    pub fn input_specs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        (0..self.input_count())
            .map(|port| {
                (
                    input_id(port),
                    ProgramTensorSpec {
                        dimensions: self.dimensions(port),
                        element_type: ElementType::Bf16,
                        layout: ResolvedTensorLayout::Contiguous,
                    },
                )
            })
            .collect()
    }
}

pub(super) struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.metal.pointwise.test"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.metal.pointwise.test")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows == 0
            || config.width == 0
            || config.rows.checked_mul(config.width).is_none()
            || serde_json::to_value(config).unwrap() != *raw
            || !MetalTensorOperation::POINTWISE
                .into_iter()
                .chain(std::iter::once(MetalTensorOperation::DenseLinear))
                .any(|kind| kind.contract().unwrap().descriptor().id == config.operation)
        {
            return Err(invalid(
                "pointwise fixture requires a known operation and positive shape",
            ));
        }
        if (config.kind() == MetalTensorOperation::DenseLinear) != config.outputs.is_some()
            || config.outputs == Some(0)
        {
            return Err(invalid(
                "only linear fixtures declare a positive output width",
            ));
        }
        if config.row_broadcast
            && !matches!(
                config.kind(),
                MetalTensorOperation::ModulationScale | MetalTensorOperation::GatedResidual
            )
        {
            return Err(invalid(
                "broadcast is only valid for modulation and gated residual",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.metal.pointwise.test"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, config: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = id::<NumericalProfileId>(config.kind().profile());
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: config.operation.clone(),
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
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let inputs: Vec<_> = (0..config.input_count()).map(input_id).collect();
        let attributes = if config.kind() == MetalTensorOperation::DenseLinear {
            BTreeMap::from([
                (id("in_features"), SemanticValue::Unsigned(config.width)),
                (
                    id("out_features"),
                    SemanticValue::Unsigned(config.outputs.unwrap()),
                ),
            ])
        } else if matches!(
            config.kind(),
            MetalTensorOperation::ModulationScale | MetalTensorOperation::GatedResidual
        ) {
            BTreeMap::from([(
                id("broadcast"),
                SemanticValue::Text(
                    if config.row_broadcast {
                        "row"
                    } else {
                        "elementwise"
                    }
                    .into(),
                ),
            )])
        } else {
            BTreeMap::new()
        };
        ModelProgram::new(
            self.family_id().clone(),
            inputs.clone(),
            vec![ProgramBlock {
                id: "block.metal.pointwise".into(),
                nodes: vec![ProgramNode {
                    id: id("node.metal.pointwise"),
                    operation_id: config.operation.clone(),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: vec![id(OUTPUT)],
                    attributes,
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT)],
        )?
        .with_input_tensor_specs(config.input_specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TensorPolicy {
    pub capacity_bytes: u64,
    pub storage_profiles: Vec<DynamicStorageProfile>,
}
impl RuntimePolicy for TensorPolicy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }
    fn memory_capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }
    fn memory_reserve_bytes(&self) -> u64 {
        0
    }
    fn maximum_active_sequences(&self) -> u32 {
        1
    }
    fn work_limits(&self) -> RuntimeWorkLimits {
        RuntimeWorkLimits::FixedTensor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn execution_determinism_requirement(&self) -> ExecutionDeterminismRequirement {
        ExecutionDeterminismRequirement::BitwiseSameRuntime
    }
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.storage_profiles
    }
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy> {
        None
    }
    fn validate(&self) -> Result<(), VNextError> {
        if self.capacity_bytes == 0 || self.storage_profiles.is_empty() {
            return Err(invalid("empty tensor policy"));
        }
        Ok(())
    }
}
