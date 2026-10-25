use super::*;
pub(super) struct Family {
    profile: &'static str,
}
impl ModelFamilyProvider for Family {
    type Config = Shape;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.tensor.block.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.tensor.block.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, c: &Shape) -> Result<(), VNextError> {
        c.blocks.validate()?;
        if c.m == 0
            || c.n == 0
            || c.k == 0
            || c.k % u64::from(c.blocks.logical_values_per_block) != 0
            || serde_json::to_value(c).unwrap() != *raw
        {
            return Err(invalid(
                "block fixture requires canonical, nonempty complete-block matrices",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &Shape,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.tensor.block.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Shape, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|error| invalid(error.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Shape) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.gguf.native-block"),
            layout_id: id("weight-layout.block.fixture"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("component.block.values"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["projection.weight".into()],
                dimensions: vec![c.n, c.k / u64::from(c.blocks.logical_values_per_block)],
                encoding: WeightEncoding::BlockQuantized(c.blocks.clone()),
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.block.projection"),
                dimensions: vec![c.n, c.k],
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding::exact_contiguous(id(
                        "component.block.values",
                    )),
                    block_axis: 1,
                    block_padding: PhysicalWeightPadding::Exact,
                },
                required: true,
            }],
        }))
    }
    fn numerical_profiles(&self, _: &Shape) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = id::<NumericalProfileId>(self.profile);
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
                    operation_id: id(DENSE_LINEAR_BLOCK_BF16_OPERATION_ID),
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
        c: &Shape,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(X)],
            vec![ProgramBlock {
                id: "block.fixture".into(),
                nodes: vec![ProgramNode {
                    id: id("node.block.fixture"),
                    operation_id: id(DENSE_LINEAR_BLOCK_BF16_OPERATION_ID),
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
                weight_id: id("weight.block.projection"),
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
    prepare_with_profile(shape, PROFILE)
}
pub(super) fn prepare_with_profile(shape: &Shape, profile: &'static str) -> PreparedModelFamily {
    TypedFamilyRegistration::new(Family { profile })
        .prepare_with_profile(&serde_json::to_value(shape).unwrap(), &id(profile))
        .unwrap()
}
