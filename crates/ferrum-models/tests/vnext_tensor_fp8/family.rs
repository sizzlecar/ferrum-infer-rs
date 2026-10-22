use super::*;
pub(super) struct Family;
impl ModelFamilyProvider for Family {
    type Config = Shape;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.tensor.fp8.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.tensor.fp8.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Shape) -> Result<(), VNextError> {
        if config.m == 0
            || config.m > 129
            || config.n == 0
            || config.n > 128
            || config.k == 0
            || config.k > 128
            || config.k % 16 != 0
            || config.n % 16 != 0
            || serde_json::to_value(config).unwrap() != *raw
        {
            return Err(invalid(
                "FP8 fixture requires bounded canonical matrix dimensions",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Shape,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.tensor.fp8.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Shape, VNextError> {
        let config = serde_json::from_value(raw.clone())
            .map_err(|e| invalid(format!("FP8 fixture: {e}")))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, c: &Shape) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fp8.fixture"),
            version: ContractVersion::new(1, 0),
            components: vec![
                WeightComponentSpec {
                    id: id("component.fp8.values"),
                    role: WeightComponentRole::PackedValues,
                    external_names: vec!["projection.weight".into()],
                    dimensions: vec![c.n, c.k],
                    encoding: WeightEncoding::Quantized(
                        QuantizationSpec::fp8_e4m3fn_whole_tensor_f32(),
                    ),
                    required: true,
                },
                WeightComponentSpec {
                    id: id("component.fp8.scale"),
                    role: WeightComponentRole::Scales,
                    external_names: vec!["projection.scale".into()],
                    dimensions: vec![1],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                    required: true,
                },
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.fp8.projection"),
                dimensions: vec![c.n, c.k],
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::QuantizedTensor {
                    packed_values: PhysicalWeightComponentBinding::exact_contiguous(id(
                        "component.fp8.values",
                    )),
                    packed_dimensions: vec![c.n, c.k],
                    scales: PhysicalWeightComponentBinding::exact_contiguous(id(
                        "component.fp8.scale",
                    )),
                },
                required: true,
            }],
        }))
    }
    fn numerical_profiles(&self, _: &Shape) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = id::<NumericalProfileId>(PROFILE);
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(Y),
                boundaries: BTreeMap::from([(id(Y), ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(OP),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
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
        c: &Shape,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(X)],
            vec![ProgramBlock {
                id: "block.fp8.fixture".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fp8.fixture"),
                    operation_id: id(OP),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(X), id(W)],
                    outputs: vec![id(Y)],
                    attributes: BTreeMap::from([
                        (id("in_features"), SemanticValue::Unsigned(c.k)),
                        (id("out_features"), SemanticValue::Unsigned(c.n)),
                    ]),
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.fp8.projection"),
                value_id: id(W),
                tensor: tensor(vec![c.n, c.k]),
            }],
            vec![id(Y)],
        )?
        .with_input_tensor_specs(BTreeMap::from([(id(X), tensor(vec![c.m, c.k]))]))
    }
    fn semantic_metadata(&self, _: &Shape) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
pub(super) fn prepare(shape: &Shape) -> PreparedModelFamily {
    TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&serde_json::to_value(shape).unwrap(), &id(PROFILE))
        .unwrap()
}
