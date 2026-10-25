//! The same stored OI matrix, declared as decoded BF16 rather than GGUF blocks.
use super::*;

pub(super) struct DenseFamily(pub(super) family::Family);
impl ModelFamilyProvider for DenseFamily {
    type Config = Shape;
    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(&self, raw: &Value, c: &Shape) -> Result<(), VNextError> {
        self.0.validate_config_identity(raw, c)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &Shape,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(raw, c)
    }
    fn parse_config(&self, raw: &Value) -> Result<Shape, VNextError> {
        self.0.parse_config(raw)
    }
    fn weights(&self, c: &Shape) -> Result<ModelWeights, VNextError> {
        let ModelWeights::Stored(mut schema) = self.0.weights(c)? else {
            unreachable!()
        };
        schema.format_id = id("weight-format.safetensors.dense");
        schema.layout_id = id("weight-layout.block-benchmark.decoded-oi");
        schema.components[0].role = WeightComponentRole::Values;
        schema.components[0].dimensions = vec![c.n, c.k];
        schema.components[0].encoding = WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        };
        schema.tensors[0].physical_layout = PhysicalWeightLayout::Dense {
            component_id: schema.components[0].id.clone(),
        };
        Ok(ModelWeights::Stored(schema))
    }
    fn numerical_profiles(&self, c: &Shape) -> Result<FamilyNumericalProfiles, VNextError> {
        let mut profile = self.0.numerical_profiles(c)?.profiles()[0].clone();
        profile.operations[0].operation_id = id(DENSE_LINEAR_BF16_OPERATION_ID);
        profile.operations[0].multiplication_type = Some(ElementType::F32);
        profile.operations[0].accumulation_type = Some(ElementType::F32);
        let selected = profile.id.clone();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![profile],
            vec![selected],
        )
    }
    fn semantic_program(
        &self,
        c: &Shape,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = self.0.semantic_program(c, p)?;
        let mut blocks = base.blocks().to_vec();
        blocks[0].nodes[0].operation_id = id(DENSE_LINEAR_BF16_OPERATION_ID);
        ModelProgram::new(
            self.family_id().clone(),
            base.inputs().to_vec(),
            blocks,
            base.states().to_vec(),
            base.weights().to_vec(),
            base.outputs().to_vec(),
        )?
        .with_input_tensor_specs(base.input_tensor_specs().clone())
    }
    fn semantic_metadata(&self, c: &Shape) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(c)
    }
}

pub(super) struct DenseSource(pub(super) ManagedHostBytes);
impl WeightComponentSource for DenseSource {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        WeightComponentPayload::from_managed_host(
            component,
            vec!["projection.weight".into()],
            vec!["candle-q4k-decoded-bf16".into()],
            component.dimensions.clone(),
            ElementType::Bf16,
            self.0.clone(),
        )
    }
}
