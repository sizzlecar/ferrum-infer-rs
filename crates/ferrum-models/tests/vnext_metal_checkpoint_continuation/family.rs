use super::*;

pub const HIDDEN: u64 = 256;
pub const MAX_TOKENS: u64 = 80;
const VOCAB: u64 = 32;
const FORMAT: &str = "weight-format.gguf.native-block";
const PROFILE: &str = "fixture.attention.f32-master";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum AttentionKind {
    Causal,
    GatedDelta,
}

pub struct Family {
    id: ModelFamilyId,
    pub kind: AttentionKind,
}

impl Family {
    pub fn new(kind: AttentionKind) -> Self {
        Self {
            id: id(match kind {
                AttentionKind::Causal => "family.fixture.causal-checkpoint",
                AttentionKind::GatedDelta => "family.fixture.gdn-checkpoint",
            }),
            kind,
        }
    }

    pub fn operation(&self) -> &'static str {
        match self.kind {
            AttentionKind::Causal => CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
            AttentionKind::GatedDelta => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
        }
    }

    pub fn states(&self) -> Vec<StateSpec> {
        let specs = match self.kind {
            AttentionKind::Causal => vec![("kv", vec![2, 2, 128], ElementType::F16, true)],
            AttentionKind::GatedDelta => vec![
                ("conv", vec![512, 3], ElementType::F16, false),
                ("delta", vec![4, 64, 64], ElementType::F32, false),
            ],
        };
        specs
            .into_iter()
            .map(|(name, dims, dtype, prefix)| {
                let tensor = tensor(dtype, dims);
                StateSpec {
                    id: id(format!("state.{name}")),
                    value_id: id(format!("value.state.{name}")),
                    capacity_demand: if prefix {
                        StateCapacityDemand::TokenScaled {
                            bytes_per_token: tensor.byte_len().unwrap(),
                            maximum_tokens: MAX_TOKENS,
                        }
                    } else {
                        StateCapacityDemand::FixedPerScope
                    },
                    tensor,
                    lifetime: StateLifetime::Sequence,
                    initialization: if prefix {
                        StateInitialization::None
                    } else {
                        StateInitialization::Zero
                    },
                    checkpoint: StateCheckpointCapability::CompletedBoundary(
                        StateCheckpointContract::new(
                            if prefix {
                                StateCheckpointContents::PrefixPositions
                            } else {
                                StateCheckpointContents::BoundaryValue
                            },
                            CheckpointInputDependency::ExactTokenPrefix,
                        ),
                    ),
                }
            })
            .collect()
    }

    fn weights(&self) -> Vec<Weight> {
        let mut weights = vec![
            Weight::dense("embedding", vec![VOCAB, HIDDEN], ElementType::F16),
            Weight::dense("input_norm", vec![HIDDEN], ElementType::F16),
        ];
        weights.extend(match self.kind {
            AttentionKind::Causal => vec![
                Weight::quantized("q", vec![1024, HIDDEN]),
                Weight::quantized("k", vec![256, HIDDEN]),
                Weight::quantized("v", vec![256, HIDDEN]),
                Weight::quantized("o", vec![HIDDEN, 512]),
                Weight::dense("q_norm", vec![128], ElementType::F16),
                Weight::dense("k_norm", vec![128], ElementType::F16),
            ],
            AttentionKind::GatedDelta => vec![
                Weight::quantized("qkvzba", vec![776, HIDDEN]),
                Weight::dense("conv", vec![512, 4], ElementType::F16),
                Weight::dense("negative_rate", vec![4], ElementType::F32),
                Weight::dense("dt_bias", vec![4], ElementType::F32),
                Weight::dense("output_norm", vec![64], ElementType::F32),
                Weight::quantized("o", vec![HIDDEN, 256]),
            ],
        });
        weights
    }

    fn attributes(&self) -> BTreeMap<AttributeId, SemanticValue> {
        let values = match self.kind {
            AttentionKind::Causal => vec![
                ("query_heads", 4),
                ("key_value_heads", 2),
                ("head_dim", 128),
                ("hidden_size", HIDDEN),
                ("query_features", 512),
                ("query_projection_features", 1024),
                ("kv_features", 256),
                ("rope_dim", 32),
                ("maximum_context_tokens", MAX_TOKENS),
            ],
            AttentionKind::GatedDelta => vec![
                ("key_heads", 2),
                ("value_heads", 4),
                ("key_head_dim", 64),
                ("value_head_dim", 64),
                ("hidden_size", HIDDEN),
                ("qkv_features", 512),
                ("value_features", 256),
                ("qkvz_features", 768),
                ("ba_features", 8),
                ("qkvzba_features", 776),
                ("conv_kernel", 4),
                ("conv_state_width", 3),
            ],
        };
        let mut attributes = values
            .into_iter()
            .map(|(key, value)| (id(key), SemanticValue::Unsigned(value)))
            .collect::<BTreeMap<_, _>>();
        attributes.insert(id("layer_index"), SemanticValue::Unsigned(0));
        attributes.insert(
            id("epsilon"),
            SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
        );
        match self.kind {
            AttentionKind::Causal => {
                attributes.insert(
                    id("rope_theta"),
                    SemanticValue::Rational(CanonicalRational::new(10_000_000, 1).unwrap()),
                );
                for key in ["rope_interleaved", "output_gate", "causal"] {
                    attributes.insert(id(key), SemanticValue::Bool(true));
                }
            }
            AttentionKind::GatedDelta => {
                attributes.insert(
                    id("decay_parameterization"),
                    SemanticValue::Text("negative_rate".to_owned()),
                );
                attributes.insert(
                    id("value_head_mapping"),
                    SemanticValue::Text("interleaved_by_key_head".to_owned()),
                );
            }
        }
        attributes
    }
}

impl ModelFamilyProvider for Family {
    type Config = AttentionKind;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.attention")])
    }
    fn validate_config_identity(
        &self,
        _raw: &Value,
        config: &AttentionKind,
    ) -> Result<(), VNextError> {
        if config != &self.kind {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.id.to_string(),
                field: "kind".into(),
                reason: "fixture kind differs from registered family".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &AttentionKind,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.fixture.attention"))
    }
    fn parse_config(&self, raw: &Value) -> Result<AttentionKind, VNextError> {
        let value = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.id.to_string(),
                field: "kind".into(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &value)?;
        Ok(value)
    }
    fn weight_schema(&self, _config: &AttentionKind) -> Result<WeightSchema, VNextError> {
        let weights = self.weights();
        Ok(WeightSchema {
            format_id: id(FORMAT),
            layout_id: id("weight-layout.fixture.attention.native"),
            version: ContractVersion::new(1, 0),
            components: weights.iter().map(Weight::component).collect(),
            tensors: weights.iter().map(Weight::logical).collect(),
        })
    }
    fn numerical_profiles(
        &self,
        _config: &AttentionKind,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id("value.embedding"),
                boundaries: BTreeMap::from([
                    (id("value.embedding"), ElementType::F32),
                    (id("value.output"), ElementType::F32),
                ]),
                states: self.states(),
                operations: vec![
                    NumericalOperationContract {
                        operation_id: id(TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID),
                        version: ContractVersion::new(1, 0),
                        multiplication_type: None,
                        accumulation_type: None,
                    },
                    NumericalOperationContract {
                        operation_id: id(self.operation()),
                        version: ContractVersion::new(1, 0),
                        multiplication_type: Some(ElementType::F32),
                        accumulation_type: Some(ElementType::F32),
                    },
                ],
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        _config: &AttentionKind,
        _profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let weights = self.weights();
        let mut attention_inputs = vec![id("value.embedding")];
        attention_inputs.extend(
            weights
                .iter()
                .skip(1)
                .map(|weight| id(format!("value.weight.{}", weight.name))),
        );
        attention_inputs.extend(self.states().iter().map(|state| state.value_id.clone()));
        ModelProgram::new(
            self.id.clone(),
            vec![id("value.tokens")],
            vec![ProgramBlock {
                id: "block.fixture.attention".into(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.embedding"),
                        operation_id: id(TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::tokens(id("value.tokens"), 0),
                        inputs: vec![id("value.tokens"), id("value.weight.embedding")],
                        outputs: vec![id("value.embedding")],
                        attributes: BTreeMap::from([
                            (id("vocab_size"), SemanticValue::Unsigned(VOCAB)),
                            (id("hidden_size"), SemanticValue::Unsigned(HIDDEN)),
                        ]),
                    },
                    ProgramNode {
                        id: id("node.attention"),
                        operation_id: id(self.operation()),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::tokens(id("value.embedding"), 0),
                        inputs: attention_inputs,
                        outputs: vec![id("value.output")],
                        attributes: self.attributes(),
                    },
                ],
            }],
            self.states(),
            weights
                .iter()
                .map(|weight| WeightReference {
                    weight_id: id(format!("weight.{}", weight.name)),
                    value_id: id(format!("value.weight.{}", weight.name)),
                    tensor: tensor(weight.dtype, weight.dimensions.clone()),
                })
                .collect(),
            vec![id("value.output")],
        )?
        .with_checkpoint_inputs(ProgramCheckpointInputs::new(
            id("value.tokens"),
            BTreeSet::new(),
        )?)
    }
    fn semantic_metadata(
        &self,
        _config: &AttentionKind,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".into(),
                source_file: "fixture-template.json".into(),
                sha256: format!("{:x}", Sha256::digest(b"{{ messages }}")),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: Some(1),
                eos_token_ids: BTreeSet::from([2]),
                pad_token_id: Some(0),
                collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
            },
        })
    }
}

fn tensor(dtype: ElementType, dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: dtype,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

struct Weight {
    name: &'static str,
    dimensions: Vec<u64>,
    dtype: ElementType,
    quantized: bool,
}
impl Weight {
    fn dense(name: &'static str, dimensions: Vec<u64>, dtype: ElementType) -> Self {
        Self {
            name,
            dimensions,
            dtype,
            quantized: false,
        }
    }
    fn quantized(name: &'static str, dimensions: Vec<u64>) -> Self {
        Self {
            name,
            dimensions,
            dtype: ElementType::F16,
            quantized: true,
        }
    }
    fn component(&self) -> WeightComponentSpec {
        let mut dimensions = self.dimensions.clone();
        let encoding = if self.quantized {
            let width = dimensions.last_mut().unwrap();
            assert!(width.is_multiple_of(256));
            *width /= 256;
            WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id("quantization.gguf.q4-k"),
                logical_values_per_block: 256,
                bytes_per_block: 144,
            })
        } else {
            WeightEncoding::Dense {
                element_type: self.dtype,
            }
        };
        WeightComponentSpec {
            id: id(format!("component.{}", self.name)),
            role: if self.quantized {
                WeightComponentRole::PackedValues
            } else {
                WeightComponentRole::Values
            },
            external_names: vec![self.name.into()],
            dimensions,
            encoding,
            required: true,
        }
    }
    fn logical(&self) -> WeightTensorSpec {
        let component_id = id(format!("component.{}", self.name));
        WeightTensorSpec {
            id: id(format!("weight.{}", self.name)),
            dimensions: self.dimensions.clone(),
            logical_element_type: self.dtype,
            physical_layout: if self.quantized {
                PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
                    block_axis: 1,
                    block_padding: PhysicalWeightPadding::Exact,
                }
            } else {
                PhysicalWeightLayout::Dense { component_id }
            },
            required: true,
        }
    }
}

pub struct Weights(pub BTreeMap<WeightId, Vec<u8>>);
impl Weights {
    pub fn new(schema: &WeightSchema) -> Self {
        let values = schema
            .components
            .iter()
            .enumerate()
            .map(|(ordinal, component)| {
                let elements = component.dimensions.iter().product::<u64>() as usize;
                let bytes = match &component.encoding {
                    WeightEncoding::Dense { element_type } => (0..elements)
                        .flat_map(|index| {
                            let phase = (index + ordinal * 13) as f32 * 0.031;
                            let value = match component.external_names[0].as_str() {
                                "negative_rate" => -0.2 - phase.sin().abs() * 0.1,
                                "dt_bias" => -0.1 + phase.sin() * 0.03,
                                name if name.contains("norm") => 1.0 + phase.sin() * 0.03,
                                _ => phase.sin() * 0.1,
                            };
                            match element_type {
                                ElementType::F16 => f16::from_f32(value).to_le_bytes().to_vec(),
                                ElementType::F32 => value.to_le_bytes().to_vec(),
                                _ => panic!("unsupported fixture weight dtype"),
                            }
                        })
                        .collect(),
                    WeightEncoding::BlockQuantized(_) => (0..elements)
                        .flat_map(|block| {
                            let mut bytes = vec![0; 144];
                            // Normal F16 scales avoid depending on subnormal
                            // handling while keeping varied projections bounded.
                            bytes[..2].copy_from_slice(&f16::from_f32(1.0 / 8192.0).to_le_bytes());
                            bytes[2..4]
                                .copy_from_slice(&f16::from_f32(1.0 / 16384.0).to_le_bytes());
                            for (index, byte) in bytes[4..].iter_mut().enumerate() {
                                *byte = ((index * 17 + block * 7 + ordinal * 19) % 251) as u8;
                            }
                            bytes
                        })
                        .collect(),
                    _ => panic!("unsupported fixture weight encoding"),
                };
                (component.id.clone(), bytes)
            })
            .collect();
        Self(values)
    }
}
impl WeightComponentSource for Weights {
    fn component<'s>(
        &'s self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        WeightComponentPayload::new(
            component,
            component.external_names[0].clone(),
            "generated-native-blocks.bin",
            component.dimensions.clone(),
            component.physical_element_type(),
            self.0[&component.id].as_slice(),
        )
    }
}
