//! Existing real F16 trunk followed by explicit Q8 Q4_K/Q6_K FFN providers.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) enum FfnPolicy {
    Strict,
    RnFragment,
    Q8,
    Q8InputSum,
    StreamMmq,
    Residual2M2To8,
}

pub(super) struct Q8FfnFamily {
    pub base: Family,
    pub policy: FfnPolicy,
}

impl Q8FfnFamily {
    fn operation(&self) -> &'static str {
        match self.policy {
            FfnPolicy::Strict => DENSE_SWIGLU_OPERATION_ID,
            FfnPolicy::RnFragment => DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID,
            FfnPolicy::Q8 => DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID,
            FfnPolicy::Q8InputSum => DENSE_SWIGLU_Q8_F32SCALE_INPUT_SUM_OPERATION_ID,
            FfnPolicy::StreamMmq => DENSE_SWIGLU_Q8_GATE_UP_STREAM_MMQ_OPERATION_ID,
            FfnPolicy::Residual2M2To8 => DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID,
        }
    }
}

impl ModelFamilyProvider for Q8FfnFamily {
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
        for (name, dimensions, format, bytes) in [
            (
                "ffn.gateup",
                vec![2, HIDDEN, HIDDEN],
                "quantization.gguf.q4-k",
                144,
            ),
            (
                "ffn.down",
                vec![HIDDEN, HIDDEN],
                "quantization.gguf.q6-k",
                210,
            ),
        ] {
            let component = id::<WeightId>(format!("component.{name}"));
            let mut storage = dimensions.clone();
            *storage.last_mut().unwrap() /= 256;
            schema.components.push(WeightComponentSpec {
                id: component.clone(),
                role: WeightComponentRole::PackedValues,
                external_names: vec![name.into()],
                dimensions: storage,
                encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: id(format),
                    logical_values_per_block: 256,
                    bytes_per_block: bytes,
                }),
                required: true,
            });
            schema.tensors.push(WeightTensorSpec {
                id: id(format!("weight.{name}")),
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding::exact_contiguous(component),
                    block_axis: (dimensions.len() - 1) as u32,
                    block_padding: PhysicalWeightPadding::Exact,
                },
                dimensions,
                required: true,
            });
        }
        Ok(schema)
    }

    fn numerical_profiles(
        &self,
        config: &AttentionKind,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let original = self.base.numerical_profiles(config)?;
        let mut profile = original.profiles()[0].clone();
        profile.boundaries.insert(id("value.ffn"), ElementType::F16);
        profile.operations.push(NumericalOperationContract {
            operation_id: id(self.operation()),
            version: ContractVersion::new(1, 0),
            multiplication_type: Some(
                if matches!(self.policy, FfnPolicy::Strict | FfnPolicy::RnFragment) {
                    ElementType::F16
                } else {
                    ElementType::I8
                },
            ),
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
            id: id("node.ffn"),
            operation_id: id(self.operation()),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(id("value.output"), 0),
            inputs: vec![
                id("value.output"),
                id("value.weight.ffn.gateup"),
                id("value.weight.ffn.down"),
            ],
            outputs: vec![id("value.ffn")],
            attributes: BTreeMap::from([
                (id("hidden_size"), SemanticValue::Unsigned(HIDDEN)),
                (id("intermediate_size"), SemanticValue::Unsigned(HIDDEN)),
            ]),
        });
        let mut weights = original.weights().to_vec();
        for (name, dimensions) in [
            ("ffn.gateup", vec![2, HIDDEN, HIDDEN]),
            ("ffn.down", vec![HIDDEN, HIDDEN]),
        ] {
            weights.push(WeightReference {
                weight_id: id(format!("weight.{name}")),
                value_id: id(format!("value.weight.{name}")),
                tensor: ProgramTensorSpec {
                    dimensions,
                    element_type: ElementType::F16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            });
        }
        ModelProgram::new(
            self.family_id().clone(),
            original.inputs().to_vec(),
            blocks,
            original.states().to_vec(),
            weights,
            vec![id("value.output"), id("value.ffn")],
        )?
        .with_checkpoint_inputs(ProgramCheckpointInputs::new(
            id("value.tokens"),
            BTreeSet::new(),
        )?)
    }
}
