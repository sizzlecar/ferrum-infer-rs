//! A tiny real compiled program using the existing validated weight source.
use super::*;

pub(super) const LINEAR_OUTPUT: u64 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LinearWeight {
    Dense,
    Q4K,
}

pub(super) struct PrimitiveFamily {
    pub base: Family,
    pub vocabulary: u64,
    pub linear: LinearWeight,
    /// Include a real stateful consumer whose declared binding workspace
    /// authorizes whole-program direct replay. The plain primitive family has
    /// no such workspace and remains adaptive-capture-only.
    pub graph_attention: bool,
}

pub(super) fn tensor(dtype: ElementType, dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: dtype,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

impl PrimitiveFamily {
    pub fn additional_inputs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([
            (
                id("value.logits"),
                tensor(ElementType::F16, vec![1, self.vocabulary]),
            ),
            (
                id("value.mask"),
                tensor(ElementType::U8, vec![self.vocabulary]),
            ),
            (
                id("value.repetition_ids"),
                tensor(ElementType::U32, vec![1]),
            ),
            (
                id("value.repetition_offsets"),
                tensor(ElementType::U32, vec![2]),
            ),
            (
                id("value.repetition_penalty"),
                tensor(ElementType::F32, vec![1]),
            ),
        ])
    }
}

impl ModelFamilyProvider for PrimitiveFamily {
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
    fn weight_schema(&self, config: &AttentionKind) -> Result<WeightSchema, VNextError> {
        let mut schema = self.base.weight_schema(config)?;
        schema.tensors.retain(|tensor| {
            self.graph_attention
                || matches!(tensor.id.as_str(), "weight.embedding" | "weight.input_norm")
                || (config.hadamard() && tensor.id.as_str() == "weight.o")
        });
        // The stateless primitive program keeps the required forward
        // Hadamard metadata as an optional source-only output projection.
        // Its graph variant really executes that projection in the GDN.
        for tensor in &mut schema.tensors {
            if !self.graph_attention && tensor.id.as_str() == "weight.o" {
                tensor.required = false;
            }
        }
        let mut used = BTreeSet::new();
        for tensor in &schema.tensors {
            used.extend(
                schema
                    .physical_component_refs(&tensor.id)?
                    .into_iter()
                    .map(|component| component.id.clone()),
            );
        }
        schema
            .components
            .retain(|component| used.contains(&component.id));
        for component in &mut schema.components {
            if !self.graph_attention && component.id.as_str() == "component.o" {
                component.required = false;
            }
        }
        let component_id: WeightId = id("component.route_linear");
        let quantized = self.linear == LinearWeight::Q4K;
        schema.components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: if quantized {
                WeightComponentRole::PackedValues
            } else {
                WeightComponentRole::Values
            },
            external_names: vec!["route_linear".into()],
            dimensions: vec![LINEAR_OUTPUT, if quantized { HIDDEN / 256 } else { HIDDEN }],
            encoding: if quantized {
                WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: id("quantization.gguf.q4-k"),
                    logical_values_per_block: 256,
                    bytes_per_block: 144,
                })
            } else {
                WeightEncoding::Dense {
                    element_type: ElementType::F16,
                }
            },
            required: true,
        });
        schema.tensors.push(WeightTensorSpec {
            id: id("weight.route_linear"),
            dimensions: vec![LINEAR_OUTPUT, HIDDEN],
            logical_element_type: ElementType::F16,
            physical_layout: if quantized {
                PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
                    block_axis: 1,
                    block_padding: PhysicalWeightPadding::Exact,
                }
            } else {
                PhysicalWeightLayout::Dense { component_id }
            },
            required: true,
        });
        Ok(schema)
    }
    fn semantic_metadata(
        &self,
        config: &AttentionKind,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        self.base.semantic_metadata(config)
    }
    fn numerical_profiles(
        &self,
        config: &AttentionKind,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let original = self.base.numerical_profiles(config)?;
        let mut profile = original.profiles()[0].clone();
        profile.kv_storage.clear();
        if !self.graph_attention {
            profile.states.clear();
        }
        profile.boundaries = BTreeMap::from([
            (id("value.embedding"), ElementType::F16),
            (id("value.output"), ElementType::F16),
            (id("value.norm"), ElementType::F16),
            (id("value.projected"), ElementType::F16),
            (id("value.token"), ElementType::U32),
        ]);
        profile.operations = [
            (TOKEN_EMBEDDING_OPERATION_ID, 1),
            (RMS_NORM_OPERATION_ID, 1),
            (RESIDUAL_ADD_OPERATION_ID, 1),
            (LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID, 3),
        ]
        .into_iter()
        .map(|(operation, version)| NumericalOperationContract {
            operation_id: id(operation),
            version: ContractVersion::new(version, 0),
            multiplication_type: None,
            accumulation_type: None,
        })
        .collect();
        profile.operations.push(NumericalOperationContract {
            operation_id: id(DENSE_LINEAR_OPERATION_ID),
            version: ContractVersion::new(1, 1),
            multiplication_type: None,
            accumulation_type: None,
        });
        if self.graph_attention {
            profile
                .boundaries
                .insert(id("value.graph_attention"), ElementType::F16);
            profile.operations.push(NumericalOperationContract {
                operation_id: id(GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID),
                version: ContractVersion::new(6, 0),
                multiplication_type: Some(ElementType::F32),
                accumulation_type: Some(ElementType::F32),
            });
        }
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
        let mut weights: Vec<_> = original
            .weights()
            .iter()
            .filter(|weight| {
                self.graph_attention
                    || matches!(
                        weight.weight_id.as_str(),
                        "weight.embedding" | "weight.input_norm"
                    )
            })
            .cloned()
            .collect();
        weights.push(WeightReference {
            weight_id: id("weight.route_linear"),
            value_id: id("value.weight.route_linear"),
            tensor: tensor(ElementType::F16, vec![LINEAR_OUTPUT, HIDDEN]),
        });
        let epsilon = SemanticValue::Rational(CanonicalRational::new(1, 1_000_000)?);
        let mut inputs = vec![id("value.tokens")];
        inputs.extend(self.additional_inputs().into_keys());
        let primitive = ModelProgram::new(
            self.family_id().clone(),
            inputs,
            vec![ProgramBlock {
                id: "block.fixture.primitive-routes".into(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.embedding"),
                        operation_id: id(TOKEN_EMBEDDING_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::tokens(id("value.tokens"), 0),
                        inputs: vec![id("value.tokens"), id("value.weight.embedding")],
                        outputs: vec![id("value.embedding")],
                        attributes: BTreeMap::from([
                            (id("vocab_size"), SemanticValue::Unsigned(32)),
                            (id("hidden_size"), SemanticValue::Unsigned(HIDDEN)),
                        ]),
                    },
                    ProgramNode {
                        id: id("node.norm"),
                        operation_id: id(RMS_NORM_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::tokens(id("value.embedding"), 0),
                        inputs: vec![id("value.embedding"), id("value.weight.input_norm")],
                        outputs: vec![id("value.norm")],
                        attributes: BTreeMap::from([
                            (id("hidden_size"), SemanticValue::Unsigned(HIDDEN)),
                            (id("epsilon"), epsilon),
                        ]),
                    },
                    ProgramNode {
                        id: id("node.residual"),
                        operation_id: id(RESIDUAL_ADD_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::tokens(id("value.norm"), 0),
                        inputs: vec![id("value.norm"), id("value.embedding")],
                        outputs: vec![id("value.output")],
                        attributes: BTreeMap::from([(
                            id("hidden_size"),
                            SemanticValue::Unsigned(HIDDEN),
                        )]),
                    },
                    ProgramNode {
                        id: id("node.linear"),
                        operation_id: id(DENSE_LINEAR_OPERATION_ID),
                        required_version: ContractVersion::new(1, 1),
                        work: ProgramNodeWorkSpec::tokens(id("value.output"), 0),
                        inputs: vec![id("value.output"), id("value.weight.route_linear")],
                        outputs: vec![id("value.projected")],
                        attributes: BTreeMap::from([
                            (id("in_features"), SemanticValue::Unsigned(HIDDEN)),
                            (id("out_features"), SemanticValue::Unsigned(LINEAR_OUTPUT)),
                        ]),
                    },
                    ProgramNode {
                        id: id("node.argmax"),
                        operation_id: id(LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID),
                        required_version: ContractVersion::new(3, 0),
                        work: ProgramNodeWorkSpec::Fixed,
                        inputs: vec![
                            id("value.logits"),
                            id("value.mask"),
                            id("value.repetition_ids"),
                            id("value.repetition_offsets"),
                            id("value.repetition_penalty"),
                        ],
                        outputs: vec![id("value.token")],
                        attributes: BTreeMap::from([(
                            id("vocab_size"),
                            SemanticValue::Unsigned(self.vocabulary),
                        )]),
                    },
                ],
            }],
            vec![],
            weights,
            vec![id("value.output"), id("value.token"), id("value.projected")],
        )?;
        let program = if self.graph_attention {
            let mut blocks = primitive.blocks().to_vec();
            let mut attention = original.blocks()[0]
                .nodes
                .iter()
                .find(|node| node.id.as_str() == "node.attention")
                .expect("the original real GDN family declares its stateful consumer")
                .clone();
            attention.operation_id = id(GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID);
            attention.required_version = ContractVersion::new(6, 0);
            // Keep all five primitive nodes and their output oracle unchanged.
            // This actual GDN consumes the retained residual and owns the real
            // per-sequence state pointers written into the program binding arena.
            attention.inputs[0] = id("value.output");
            attention.work = ProgramNodeWorkSpec::tokens(id("value.output"), 0);
            attention.outputs = vec![id("value.graph_attention")];
            blocks[0].nodes.push(attention);
            let mut outputs = primitive.outputs().to_vec();
            outputs.push(id("value.graph_attention"));
            ModelProgram::new(
                self.family_id().clone(),
                primitive.inputs().to_vec(),
                blocks,
                original.states().to_vec(),
                primitive.weights().to_vec(),
                outputs,
            )?
        } else {
            primitive
        };
        program.with_checkpoint_inputs(
            ProgramCheckpointInputs::new(id("value.tokens"), BTreeSet::new())?
                .with_output_only_inputs(self.additional_inputs().into_keys().collect())?,
        )
    }
}

#[test]
fn primitive_graph_family_prepares_real_recurrent_state_and_preserves_io() {
    for kind in [
        AttentionKind::GatedDelta,
        AttentionKind::GatedDeltaHadamardF16,
    ] {
        for graph_attention in [false, true] {
            let definition = PrimitiveFamily {
                base: Family::new(kind),
                vocabulary: 8192,
                linear: LinearWeight::Q4K,
                graph_attention,
            };
            let profile = definition.base.profile_id();
            let prepared = TypedFamilyRegistration::new(definition)
                .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
                .unwrap();
            let nodes = &prepared.program().blocks()[0].nodes;
            assert_eq!(nodes.len(), if graph_attention { 6 } else { 5 });
            assert_eq!(nodes[0].operation_id.as_str(), TOKEN_EMBEDDING_OPERATION_ID);
            assert_eq!(
                nodes[4].operation_id.as_str(),
                LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID
            );
            if graph_attention {
                let attention = &nodes[5];
                assert_eq!(
                    attention.operation_id.as_str(),
                    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID
                );
                assert_eq!(attention.required_version, ContractVersion::new(6, 0));
                assert_eq!(attention.inputs[0].as_str(), "value.output");
                assert_eq!(attention.outputs[0].as_str(), "value.graph_attention");
                for state in prepared.program().states() {
                    assert!(attention.inputs.contains(&state.value_id));
                }
                assert_eq!(prepared.program().states().len(), 2);
                assert!(
                    prepared
                        .weight_schema()
                        .tensors
                        .iter()
                        .find(|w| w.id.as_str() == "weight.o")
                        .unwrap()
                        .required
                );
            } else {
                assert!(prepared.program().states().is_empty());
            }
        }
    }
}
