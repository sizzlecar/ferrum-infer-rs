//! The normal recurrent F32 trunk followed by a real native Q6_K output head.
use super::*;

pub(super) const OUTPUTS: u64 = 1024;

#[derive(Clone, Copy, Debug)]
pub(super) enum Head {
    Strict,
    HalfOperands,
}

impl Head {
    pub fn operation(self) -> &'static str {
        match self {
            Self::Strict => LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
            Self::HalfOperands => LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Strict => "vnext_last_token_dense_linear_f32",
            Self::HalfOperands => "vnext_last_token_dense_linear_f32_f16_operands",
        }
    }

    pub fn operand(self, value: f32) -> f32 {
        match self {
            Self::Strict => value,
            Self::HalfOperands => f16::from_f32(value).to_f32(),
        }
    }
}

pub(super) struct HeadFamily {
    pub base: Family,
    pub head: Head,
}

impl ModelFamilyProvider for HeadFamily {
    type Config = AttentionKind;

    fn family_id(&self) -> &ModelFamilyId {
        self.base.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.base.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &AttentionKind,
    ) -> Result<(), VNextError> {
        self.base.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &AttentionKind,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.base.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<AttentionKind, VNextError> {
        self.base.parse_config(raw)
    }
    fn semantic_metadata(
        &self,
        config: &AttentionKind,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        self.base.semantic_metadata(config)
    }
    fn weight_schema(&self, config: &AttentionKind) -> Result<WeightSchema, VNextError> {
        let mut schema = self.base.weight_schema(config)?;
        schema.components.push(WeightComponentSpec {
            id: id("component.head"),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["head".into()],
            dimensions: vec![OUTPUTS, HIDDEN / 256],
            encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id("quantization.gguf.q6-k"),
                logical_values_per_block: 256,
                bytes_per_block: 210,
            }),
            required: true,
        });
        schema.tensors.push(WeightTensorSpec {
            id: id("weight.head"),
            dimensions: vec![OUTPUTS, HIDDEN],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(id("component.head")),
                block_axis: 1,
                block_padding: PhysicalWeightPadding::Exact,
            },
            required: true,
        });
        Ok(schema)
    }
    fn numerical_profiles(
        &self,
        config: &AttentionKind,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let original = self.base.numerical_profiles(config)?;
        let mut profile = original.profiles()[0].clone();
        profile
            .boundaries
            .insert(id("value.head"), ElementType::F32);
        profile.operations.push(NumericalOperationContract {
            operation_id: id(self.head.operation()),
            version: ContractVersion::new(1, 0),
            multiplication_type: Some(match self.head {
                Head::Strict => ElementType::F32,
                Head::HalfOperands => ElementType::F16,
            }),
            accumulation_type: Some(ElementType::F32),
        });
        FamilyNumericalProfiles::new(
            self.family_id(),
            original.version(),
            vec![profile],
            original.auto_preference().to_vec(),
        )
    }
    fn semantic_program(
        &self,
        config: &AttentionKind,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let original = self.base.semantic_program(config, profile)?;
        let mut blocks = original.blocks().to_vec();
        blocks[0].nodes.push(ProgramNode {
            id: id("node.head"),
            operation_id: id(self.head.operation()),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(id("value.output"), 0),
            inputs: vec![id("value.output"), id("value.weight.head")],
            outputs: vec![id("value.head")],
            attributes: BTreeMap::from([
                (id("hidden_size"), SemanticValue::Unsigned(HIDDEN)),
                (id("out_features"), SemanticValue::Unsigned(OUTPUTS)),
            ]),
        });
        let mut weights = original.weights().to_vec();
        weights.push(WeightReference {
            weight_id: id("weight.head"),
            value_id: id("value.weight.head"),
            tensor: ProgramTensorSpec {
                dimensions: vec![OUTPUTS, HIDDEN],
                element_type: ElementType::F16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        });
        ModelProgram::new(
            self.family_id().clone(),
            original.inputs().to_vec(),
            blocks,
            original.states().to_vec(),
            weights,
            vec![id("value.output"), id("value.head")],
        )?
        .with_checkpoint_inputs(ProgramCheckpointInputs::new(
            id("value.tokens"),
            BTreeSet::new(),
        )?)
    }
}
