use super::*;

/// Wrap the existing tiny stateful model without changing its public fixture.
/// Attention inputs and state ABI remain real; only versioned operation policy
/// and the physical projection parts differ.
pub struct Family {
    base: legacy_family::Family,
    selected: Q8ActAttentionProfile,
}
impl Family {
    pub fn new(kind: AttentionKind) -> Self {
        let selected = match kind {
            AttentionKind::GatedDelta => Q8ActAttentionProfile::GatedDelta,
            AttentionKind::Causal => Q8ActAttentionProfile::Causal,
            _ => panic!("attention-Q8 gate only declares GDN and FP16-KV causal"),
        };
        Self {
            base: legacy_family::Family::new(kind),
            selected,
        }
    }
    pub fn states(&self) -> Vec<StateSpec> {
        self.base.states()
    }
    pub fn profile_id(&self) -> &'static str {
        "fixture.attention.require-q4k-q5k-iq4xs-g32"
    }
}

impl ModelFamilyProvider for Family {
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
        let names: &[&str] = match self.selected {
            Q8ActAttentionProfile::GatedDelta => &["qkvzba", "o"],
            Q8ActAttentionProfile::Causal => &["q", "k", "v", "o"],
        };
        for &name in names {
            schema
                .components
                .retain(|c| c.id.as_str() != format!("component.{name}"));
            let tensor = schema
                .tensors
                .iter_mut()
                .find(|t| t.id.as_str() == format!("weight.{name}"))
                .unwrap();
            let (rows, inputs) = (tensor.dimensions[0], tensor.dimensions[1]);
            let chunk = if name == "qkvzba" { 256 } else { rows / 4 };
            let mut parts = Vec::new();
            let mut offset = 0;
            for (part, format, bytes) in [
                (0, "q4-k", 144),
                (1, "q5-k", 176),
                (2, "iq4-xs", 136),
                (3, "dense", 0),
            ] {
                let count = if part == 3 { rows - offset } else { chunk };
                // Reverse lexical order ensures component sorting cannot stand
                // in for the real output offsets of a physical composite.
                let component_id: WeightId = id(format!("component.{name}.{}", 3 - part));
                let quantized = bytes != 0;
                schema.components.push(WeightComponentSpec {
                    id: component_id.clone(),
                    role: if quantized {
                        WeightComponentRole::PackedValues
                    } else {
                        WeightComponentRole::Values
                    },
                    external_names: vec![format!("{name}.{format}")],
                    dimensions: vec![count, if quantized { inputs / 256 } else { inputs }],
                    encoding: if quantized {
                        WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                            format_id: id(format!("quantization.gguf.{format}")),
                            logical_values_per_block: 256,
                            bytes_per_block: bytes,
                        })
                    } else {
                        WeightEncoding::Dense {
                            element_type: ElementType::F16,
                        }
                    },
                    required: true,
                });
                let layout = if quantized {
                    PhysicalWeightLayout::BlockQuantized {
                        blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
                        block_axis: 1,
                        block_padding: PhysicalWeightPadding::Exact,
                    }
                } else {
                    PhysicalWeightLayout::Dense { component_id }
                };
                parts.push(CompositeWeightPart {
                    layout: Box::new(layout),
                    logical_offsets: vec![offset, 0],
                    extents: vec![count, inputs],
                });
                offset += count;
            }
            assert_eq!(offset, rows);
            tensor.physical_layout = PhysicalWeightLayout::Composite { parts };
        }
        Ok(schema)
    }

    fn numerical_profiles(
        &self,
        config: &AttentionKind,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let mut profile = self.base.numerical_profiles(config)?.profiles()[0].clone();
        profile.id = id(self.profile_id());
        let attention = profile
            .operations
            .iter_mut()
            .find(|o| o.operation_id.as_str() == self.base.operation())
            .unwrap();
        attention.operation_id = id(self.selected.operation_id());
        attention.multiplication_type = None;
        attention.accumulation_type = None;
        attention.composite_arithmetic = Some(self.selected.arithmetic());
        // The test explicitly selects Require. No automatic preference exists.
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![profile],
            vec![],
        )
    }

    fn semantic_program(
        &self,
        config: &AttentionKind,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let old = self.base.semantic_program(config, profile)?;
        let mut blocks = old.blocks().to_vec();
        blocks[0].nodes[1].operation_id = id(self.selected.operation_id());
        let program = ModelProgram::new(
            self.family_id().clone(),
            old.inputs().to_vec(),
            blocks,
            old.states().to_vec(),
            old.weights().to_vec(),
            old.outputs().to_vec(),
        )?;
        if let Some(inputs) = old.checkpoint_inputs() {
            program.with_checkpoint_inputs(inputs.clone())
        } else {
            Ok(program)
        }
    }
}

pub struct Weights {
    values: BTreeMap<WeightId, Vec<u8>>,
}
impl Weights {
    pub fn new(schema: &WeightSchema) -> Self {
        let values = schema
            .components
            .iter()
            .enumerate()
            .map(|(ordinal, component)| {
                let elements = component.dimensions.iter().product::<u64>() as usize;
                let data = match &component.encoding {
                    WeightEncoding::Dense { element_type } => (0..elements)
                        .flat_map(|i| {
                            let phase = (i + ordinal * 13) as f32 * 0.031;
                            let value = match component.external_names[0].as_str() {
                                "negative_rate" => -0.2 - phase.sin().abs() * 0.1,
                                "dt_bias" => -0.1 + phase.sin() * 0.03,
                                name if name.contains("norm") => 1.0 + phase.sin() * 0.03,
                                name if name.ends_with(".dense") => phase.sin() / 512.0,
                                _ => phase.sin() * 0.1,
                            };
                            match element_type {
                                ElementType::F16 => f16::from_f32(value).to_le_bytes().to_vec(),
                                ElementType::F32 => value.to_le_bytes().to_vec(),
                                _ => panic!("undeclared fixture type"),
                            }
                        })
                        .collect(),
                    WeightEncoding::BlockQuantized(spec) => (0..elements)
                        .flat_map(|block| {
                            let mut bytes = vec![0; spec.bytes_per_block as usize];
                            bytes[..2].copy_from_slice(&f16::from_f32(1.0 / 8192.0).to_le_bytes());
                            let start = match spec.format_id.as_str() {
                                "quantization.gguf.q4-k" | "quantization.gguf.q5-k" => {
                                    bytes[2..4].copy_from_slice(
                                        &f16::from_f32(1.0 / 16384.0).to_le_bytes(),
                                    );
                                    4
                                }
                                "quantization.gguf.iq4-xs" => 2,
                                _ => panic!("undeclared fixture block format"),
                            };
                            for (i, byte) in bytes[start..].iter_mut().enumerate() {
                                *byte = ((i * 17 + block * 7 + ordinal * 19) % 251) as u8;
                            }
                            bytes
                        })
                        .collect(),
                    _ => panic!("undeclared fixture encoding"),
                };
                (component.id.clone(), data)
            })
            .collect();
        Self { values }
    }
    pub fn set_nonfinite_embedding(&mut self, token: u32) {
        let data = self
            .values
            .get_mut(&id::<WeightId>("component.embedding"))
            .unwrap();
        let start = token as usize * HIDDEN as usize * 2;
        data[start..start + 2].copy_from_slice(&f16::NAN.to_le_bytes());
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
            "generated-mixed-attention.bin",
            component.dimensions.clone(),
            component.physical_element_type(),
            &self.values[&component.id],
        )
    }
}
