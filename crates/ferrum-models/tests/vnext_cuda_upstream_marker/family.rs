use super::*;
#[path = "family/extra.rs"]
pub mod extra;
#[path = "family/extra_all_rows.rs"]
pub mod extra_all_rows;
#[path = "family/extra_prefill.rs"]
pub mod extra_prefill;
pub use extra::Weights;
#[path = "family/q6_f16.rs"]
pub mod q6_f16;

const INTERMEDIATE: u64 = 768;
const PROFILE: &str = "fixture.attention-ffn.require-upstream-marker-v2";

/// Keep the tested attention/state source; add a real typed FFN block rather
/// than pretending the F32 residual activation is an F16 projection input.
pub struct Family {
    base: attention_family::Family,
    selected: UpstreamMarkerV2Profile,
    maximum_tokens: u64,
    g32_baseline: bool,
    small_q8_projection: bool,
    q8_projection_pair_adjacent: Option<bool>,
    causal_sixteen_token_pages: bool,
}

impl Family {
    pub fn new(kind: AttentionKind) -> Self {
        Self {
            base: attention_family::Family::new(kind),
            maximum_tokens: MAX_TOKENS,
            g32_baseline: false,
            small_q8_projection: false,
            q8_projection_pair_adjacent: None,
            causal_sixteen_token_pages: false,
            selected: match kind {
                AttentionKind::GatedDelta => UpstreamMarkerV2Profile::GatedDelta,
                AttentionKind::Causal => UpstreamMarkerV2Profile::Causal,
                _ => panic!("MarkerV2 gate declares GDN and FP16-KV causal"),
            },
        }
    }
    pub fn with_small_q8_projection(mut self) -> Self {
        self.small_q8_projection = true;
        self.q8_projection_pair_adjacent = None;
        self
    }
    /// 8 KV heads x 128 x K/V x F16 = 4096 bytes/token: a real 64-KiB
    /// physical page holds exactly one 16-token vLLM block.
    pub fn with_causal_sixteen_token_pages(mut self) -> Self {
        assert_eq!(self.selected, UpstreamMarkerV2Profile::Causal);
        self.causal_sixteen_token_pages = true;
        self
    }
    pub fn with_small_q8_projection_pair(mut self, adjacent: bool) -> Self {
        self.small_q8_projection = false;
        self.q8_projection_pair_adjacent = Some(adjacent);
        self
    }
    pub fn small_q8_projection_layout(&self) -> Vec<(u64, u64)> {
        match self.q8_projection_pair_adjacent {
            Some(true) => vec![(192, 13), (205, 19)],
            Some(false) => vec![(192, 13), (221, 19)],
            None if self.small_q8_projection => vec![(192, 32)],
            None => Vec::new(),
        }
    }
    pub fn prefill(kind: AttentionKind, maximum_tokens: u64) -> Self {
        assert!(maximum_tokens >= 2048);
        let mut value = Self::new(kind);
        value.selected = match kind {
            AttentionKind::GatedDelta => UpstreamMarkerV2Profile::GatedDeltaPrefill,
            AttentionKind::Causal => UpstreamMarkerV2Profile::CausalPrefill,
            _ => unreachable!(),
        };
        value.maximum_tokens = maximum_tokens;
        value
    }
    pub fn hybrid(kind: AttentionKind, maximum_tokens: u64) -> Self {
        let mut value = Self::new(kind);
        value.selected = match kind {
            AttentionKind::GatedDelta => UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill,
            AttentionKind::Causal => UpstreamMarkerV2Profile::CausalG32MmqPrefill,
            _ => unreachable!(),
        };
        value.maximum_tokens = maximum_tokens;
        value
    }
    pub fn g32_baseline(kind: AttentionKind) -> Self {
        let mut value = Self::new(kind);
        value.g32_baseline = true;
        value
    }
    pub fn is_g32_baseline(&self) -> bool {
        self.g32_baseline
    }
    pub fn attention_operation(&self) -> &'static str {
        if self.g32_baseline {
            self.g32_attention().operation_id()
        } else {
            self.selected.operation_id()
        }
    }
    pub fn attention_arithmetic(&self) -> CompositeNumericalArithmetic {
        if self.g32_baseline {
            self.g32_attention().arithmetic()
        } else {
            self.selected.arithmetic()
        }
    }
    fn g32_attention(&self) -> Q8ActAttentionProfile {
        if self.selected.strict_operation_id()
            == GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID
        {
            Q8ActAttentionProfile::GatedDelta
        } else {
            Q8ActAttentionProfile::Causal
        }
    }
    pub fn swiglu_operation(&self) -> &'static str {
        if self.g32_baseline {
            Q8ActSwiGluProfile::Q4KQ5KIq4Xs.operation_id()
        } else {
            self.swiglu_profile().operation_id()
        }
    }
    pub fn swiglu_arithmetic(&self) -> CompositeNumericalArithmetic {
        if self.g32_baseline {
            Q8ActSwiGluProfile::Q4KQ5KIq4Xs.arithmetic()
        } else {
            self.swiglu_profile().arithmetic()
        }
    }
    pub fn maximum_tokens(&self) -> u64 {
        self.maximum_tokens
    }
    pub fn with_maximum_tokens(mut self, maximum_tokens: u64) -> Self {
        assert!(maximum_tokens > 0);
        self.maximum_tokens = maximum_tokens;
        self
    }
    pub fn attention_profile(&self) -> UpstreamMarkerV2Profile {
        self.selected
    }
    pub fn swiglu_profile(&self) -> UpstreamMarkerV2Profile {
        if self.selected.q6_f16() {
            UpstreamMarkerV2Profile::SwiGluQ6F16
        } else if self.selected.extra_all_rows() {
            UpstreamMarkerV2Profile::SwiGluExtraAllRows
        } else if self.selected.extra_prefill() {
            UpstreamMarkerV2Profile::SwiGluExtraLargePrefill
        } else if self.selected.extra() {
            UpstreamMarkerV2Profile::SwiGluExtraPrefill
        } else if self.selected.hybrid() {
            UpstreamMarkerV2Profile::SwiGluG32MmqPrefill
        } else if self.selected.prefill() {
            UpstreamMarkerV2Profile::SwiGluPrefill
        } else {
            UpstreamMarkerV2Profile::SwiGlu
        }
    }
    pub fn hidden_size(&self) -> u64 {
        if self.selected.q6_f16() {
            q6_f16::WIDE_HIDDEN
        } else {
            HIDDEN
        }
    }
    pub fn states(&self) -> Vec<StateSpec> {
        let mut states = self.base.states();
        if self.causal_sixteen_token_pages {
            let kv = states
                .iter_mut()
                .find(|s| s.id.as_str() == "state.kv")
                .unwrap();
            kv.tensor.dimensions = vec![2, 8, 128];
            if let StateCapacityDemand::TokenScaled {
                bytes_per_token, ..
            } = &mut kv.capacity_demand
            {
                *bytes_per_token = kv.tensor.byte_len().unwrap();
            }
        }
        if self.selected.q6_f16() {
            q6_f16::resize_states(&mut states);
        }
        for state in &mut states {
            if let StateCapacityDemand::TokenScaled { maximum_tokens, .. } =
                &mut state.capacity_demand
            {
                *maximum_tokens = self.maximum_tokens;
            }
        }
        states
    }
    pub fn profile_id(&self) -> &'static str {
        if self.selected.q6_f16() {
            q6_f16::PROFILE
        } else if self.g32_baseline {
            "fixture.attention-ffn.require-atn-g32"
        } else if self.selected.extra_all_rows() {
            extra_all_rows::PROFILE
        } else if self.selected.extra_prefill() {
            extra_prefill::PROFILE
        } else if self.selected.extra() {
            extra::PROFILE
        } else if self.selected.hybrid() {
            "fixture.attention-ffn.require-g32-mmq-prefill"
        } else if self.selected.prefill() {
            "fixture.attention-ffn.require-upstream-marker-v2-prefill"
        } else {
            PROFILE
        }
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
        if self.causal_sixteen_token_pages {
            for (name, axis, factor) in [("q", 0, 2), ("k", 0, 4), ("v", 0, 4), ("o", 1, 2)] {
                let tensor = schema
                    .tensors
                    .iter_mut()
                    .find(|t| t.id.as_str() == format!("weight.{name}"))
                    .unwrap();
                tensor.dimensions[axis] *= factor;
                let PhysicalWeightLayout::Composite { parts } = &mut tensor.physical_layout else {
                    unreachable!()
                };
                for part in parts {
                    part.logical_offsets[axis] *= factor;
                    part.extents[axis] *= factor;
                    let component_id = match part.layout.as_ref() {
                        PhysicalWeightLayout::Dense { component_id } => component_id,
                        PhysicalWeightLayout::BlockQuantized { blocks, .. } => &blocks.component_id,
                        _ => unreachable!(),
                    };
                    schema
                        .components
                        .iter_mut()
                        .find(|c| c.id == *component_id)
                        .unwrap()
                        .dimensions[axis] *= factor;
                }
            }
        }
        let hidden = self.hidden_size();
        let intermediate = self.intermediate_size();
        if self.selected.q6_f16() {
            q6_f16::resize_schema(&mut schema, *config);
        }
        let norm: WeightId = id("component.ffn_norm");
        schema.components.push(WeightComponentSpec {
            id: norm.clone(),
            role: WeightComponentRole::Values,
            external_names: vec!["ffn_norm".into()],
            dimensions: vec![hidden],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        });
        schema.tensors.push(WeightTensorSpec {
            id: id("weight.ffn_norm"),
            dimensions: vec![hidden],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Dense { component_id: norm },
            required: true,
        });
        mixed_weight(&mut schema, "ffn_gate_up", 2, intermediate, hidden);
        mixed_weight(&mut schema, "ffn_down", 1, hidden, intermediate);
        // One extra physical dense leaf in each operation makes the validation
        // payload an odd number of eight-byte banks. Its admitted Plan buffer
        // must include alignment padding; no dimensions or budget change.
        if self.selected.q6_f16() {
            q6_f16::replace_projection_leaves(&mut schema);
        } else if self.selected.extra() {
            extra::extend_schema(&mut schema);
            // 7 leaves per bank gives 21 FFN leaves; only attention needs a
            // split to keep both admitted flag extents genuinely padded.
            split_dense_leaf(&mut schema, "o");
        } else {
            split_dense_leaf(&mut schema, "o");
            split_dense_leaf(&mut schema, "ffn_down");
        }
        if self.small_q8_projection {
            // Replace one of the split o leaves; the other remains dense.
            // Logical offsets and output stride therefore exercise a real
            // mixed projection, rather than a standalone synthetic dot.
            let tensor = schema
                .tensors
                .iter_mut()
                .find(|t| t.id.as_str() == "weight.o")
                .unwrap();
            let PhysicalWeightLayout::Composite { parts } = &mut tensor.physical_layout else {
                unreachable!()
            };
            let part = parts
                .iter_mut()
                .find(|p| matches!(p.layout.as_ref(), PhysicalWeightLayout::Dense { .. }))
                .unwrap();
            let PhysicalWeightLayout::Dense { component_id } = part.layout.as_ref() else {
                unreachable!()
            };
            let component_id = component_id.clone();
            let component = schema
                .components
                .iter_mut()
                .find(|c| c.id == component_id)
                .unwrap();
            assert_eq!(part.extents, vec![32, HIDDEN]);
            component.role = WeightComponentRole::PackedValues;
            component.dimensions = vec![32, HIDDEN / 32];
            component.encoding = WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id("quantization.gguf.q8-0"),
                logical_values_per_block: 32,
                bytes_per_block: 34,
            });
            part.layout = Box::new(PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
                block_axis: 1,
                block_padding: PhysicalWeightPadding::Exact,
            });
        }
        if let Some(adjacent) = self.q8_projection_pair_adjacent {
            let tensor = schema
                .tensors
                .iter_mut()
                .find(|t| t.id.as_str() == "weight.o")
                .unwrap();
            let PhysicalWeightLayout::Composite { parts } = &mut tensor.physical_layout else {
                unreachable!()
            };
            let removed: BTreeSet<_> = parts
                .iter()
                .filter_map(|part| match part.layout.as_ref() {
                    PhysicalWeightLayout::Dense { component_id } => Some(component_id.clone()),
                    _ => None,
                })
                .collect();
            parts
                .retain(|part| !matches!(part.layout.as_ref(), PhysicalWeightLayout::Dense { .. }));
            schema
                .components
                .retain(|component| !removed.contains(&component.id));
            // The same 64 output columns contain two odd, unequal Q8 leaves.
            // The fallback inserts a dense output interval between them while
            // their separately prepared F32 destinations remain contiguous.
            let widths: &[(u64, bool)] = if adjacent {
                &[(13, true), (19, true), (32, false)]
            } else {
                &[(13, true), (16, false), (19, true), (16, false)]
            };
            let mut offset = 192;
            for (index, &(outputs, quantized)) in widths.iter().enumerate() {
                let component_id: WeightId = id(format!("component.o.q8-pair.{index}"));
                schema.components.push(WeightComponentSpec {
                    id: component_id.clone(),
                    role: if quantized {
                        WeightComponentRole::PackedValues
                    } else {
                        WeightComponentRole::Values
                    },
                    external_names: vec![format!("o.q8-pair.{index}")],
                    dimensions: vec![outputs, if quantized { HIDDEN / 32 } else { HIDDEN }],
                    encoding: if quantized {
                        WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                            format_id: id("quantization.gguf.q8-0"),
                            logical_values_per_block: 32,
                            bytes_per_block: 34,
                        })
                    } else {
                        WeightEncoding::Dense {
                            element_type: ElementType::F16,
                        }
                    },
                    required: true,
                });
                parts.push(CompositeWeightPart {
                    layout: Box::new(if quantized {
                        PhysicalWeightLayout::BlockQuantized {
                            blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
                            block_axis: 1,
                            block_padding: PhysicalWeightPadding::Exact,
                        }
                    } else {
                        PhysicalWeightLayout::Dense { component_id }
                    }),
                    logical_offsets: vec![offset, 0],
                    extents: vec![outputs, HIDDEN],
                });
                offset += outputs;
            }
            assert_eq!(offset, HIDDEN);
        }
        Ok(schema)
    }

    fn numerical_profiles(
        &self,
        config: &AttentionKind,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let mut profile = self.base.numerical_profiles(config)?.profiles()[0].clone();
        profile.id = id(self.profile_id());
        profile.states = self.states();
        let attention = profile
            .operations
            .iter_mut()
            .find(|op| op.composite_arithmetic.is_some())
            .unwrap();
        attention.operation_id = id(self.attention_operation());
        attention.composite_arithmetic = Some(self.attention_arithmetic());
        for (operation, multiplication_type, accumulation_type, composite_arithmetic) in [
            (
                RMS_NORM_F32_TO_F16_OPERATION_ID,
                None,
                Some(ElementType::F32),
                None,
            ),
            (
                self.swiglu_operation(),
                None,
                None,
                Some(self.swiglu_arithmetic()),
            ),
            (RESIDUAL_ADD_F32_F16_OPERATION_ID, None, None, None),
        ] {
            profile.operations.push(NumericalOperationContract {
                operation_id: id(operation),
                version: ContractVersion::new(1, 0),
                multiplication_type,
                accumulation_type,
                staged_arithmetic: None,
                composite_arithmetic,
            });
        }
        profile.boundaries.extend([
            (id("value.attention"), ElementType::F32),
            (id("value.normalized"), ElementType::F16),
            (id("value.ffn"), ElementType::F16),
        ]);
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
        // Validate the complete declaration, not a caller-supplied profile with
        // the same ID but different arithmetic or activation boundaries.
        let declared = self.numerical_profiles(config)?;
        if declared.resolve(&profile.id)? != profile {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "numerical_profile".into(),
                reason: "fixture requires its exact MarkerV2 declaration".into(),
            });
        }
        let old = self.base.semantic_program(config, profile)?;
        let mut blocks = old.blocks().to_vec();
        let hidden_size = self.hidden_size();
        let intermediate = self.intermediate_size();
        if self.selected.q6_f16() {
            q6_f16::resize_nodes(&mut blocks, *config);
        }
        let attention = &mut blocks[0].nodes[1];
        if self.causal_sixteen_token_pages {
            for (name, value) in [
                ("query_heads", 8),
                ("key_value_heads", 8),
                ("query_features", 1024),
                ("query_projection_features", 2048),
                ("kv_features", 1024),
            ] {
                attention
                    .attributes
                    .insert(id(name), SemanticValue::Unsigned(value));
            }
        }
        attention.operation_id = id(self.attention_operation());
        if attention
            .attributes
            .contains_key(&id("maximum_context_tokens"))
        {
            attention.attributes.insert(
                id("maximum_context_tokens"),
                SemanticValue::Unsigned(self.maximum_tokens),
            );
        }
        attention.outputs = vec![id("value.attention")];
        let mut node = |name: &str,
                        operation: &str,
                        inputs: &[&str],
                        output: &str,
                        attrs: BTreeMap<AttributeId, SemanticValue>| {
            blocks[0].nodes.push(ProgramNode {
                id: id(name),
                operation_id: id(operation),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::tokens(id(inputs[0]), 0),
                inputs: inputs.iter().map(|s| id(*s)).collect(),
                outputs: vec![id(output)],
                attributes: attrs,
            });
        };
        let hidden = || BTreeMap::from([(id("hidden_size"), SemanticValue::Unsigned(hidden_size))]);
        let mut norm = hidden();
        norm.insert(
            id("epsilon"),
            SemanticValue::Rational(CanonicalRational::new(1, 1_000_000)?),
        );
        node(
            "node.ffn_norm",
            RMS_NORM_F32_TO_F16_OPERATION_ID,
            &["value.attention", "value.weight.ffn_norm"],
            "value.normalized",
            norm,
        );
        let mut ffn = hidden();
        ffn.insert(
            id("intermediate_size"),
            SemanticValue::Unsigned(intermediate),
        );
        node(
            "node.swiglu",
            self.swiglu_operation(),
            &[
                "value.normalized",
                "value.weight.ffn_gate_up",
                "value.weight.ffn_down",
            ],
            "value.ffn",
            ffn,
        );
        node(
            "node.ffn_residual",
            RESIDUAL_ADD_F32_F16_OPERATION_ID,
            &["value.attention", "value.ffn"],
            "value.output",
            hidden(),
        );
        let mut weights = old.weights().to_vec();
        if self.causal_sixteen_token_pages {
            for (name, axis, factor) in [("q", 0, 2), ("k", 0, 4), ("v", 0, 4), ("o", 1, 2)] {
                weights
                    .iter_mut()
                    .find(|w| w.weight_id.as_str() == format!("weight.{name}"))
                    .unwrap()
                    .tensor
                    .dimensions[axis] *= factor;
            }
        }
        if self.selected.q6_f16() {
            q6_f16::resize_weights(&mut weights, *config);
        }
        for (name, dimensions) in [
            ("ffn_norm", vec![hidden_size]),
            ("ffn_gate_up", vec![2, intermediate, hidden_size]),
            ("ffn_down", vec![hidden_size, intermediate]),
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
        let program = ModelProgram::new(
            self.family_id().clone(),
            old.inputs().to_vec(),
            blocks,
            self.states(),
            weights,
            old.outputs().to_vec(),
        )?;
        if let Some(inputs) = old.checkpoint_inputs() {
            program.with_checkpoint_inputs(inputs.clone())
        } else {
            Ok(program)
        }
    }
}

fn split_dense_leaf(schema: &mut WeightSchema, name: &str) {
    let tensor = schema
        .tensors
        .iter_mut()
        .find(|t| t.id.as_str() == format!("weight.{name}"))
        .unwrap();
    let PhysicalWeightLayout::Composite { parts } = &mut tensor.physical_layout else {
        panic!("fixture projection must be composite")
    };
    let index = parts
        .iter()
        .position(|p| matches!(p.layout.as_ref(), PhysicalWeightLayout::Dense { .. }))
        .unwrap();
    let PhysicalWeightLayout::Dense { component_id } = parts[index].layout.as_ref() else {
        unreachable!()
    };
    let component = schema
        .components
        .iter_mut()
        .find(|c| c.id == *component_id)
        .unwrap();
    assert_eq!(component.dimensions.len(), 2);
    let head_rows = parts[index].extents[0] / 2;
    assert!(head_rows > 0);
    let tail_rows = parts[index].extents[0] - head_rows;
    let mut tail_component = component.clone();
    tail_component.id = id(format!("{}.tail", component.id.as_str()));
    tail_component.external_names = vec![format!("{name}.tail.dense")];
    tail_component.dimensions[0] = tail_rows;
    component.dimensions[0] = head_rows;
    let mut tail = parts[index].clone();
    tail.layout = Box::new(PhysicalWeightLayout::Dense {
        component_id: tail_component.id.clone(),
    });
    tail.logical_offsets[0] += head_rows;
    tail.extents[0] = tail_rows;
    parts[index].extents[0] = head_rows;
    parts.insert(index + 1, tail);
    schema.components.push(tail_component);
}

fn mixed_weight(schema: &mut WeightSchema, name: &str, banks: u64, outputs: u64, inputs: u64) {
    let mut parts = Vec::new();
    for bank in 0..banks {
        for (part, format, bytes) in [
            (0, "q4-k", 144),
            (1, "q5-k", 176),
            (2, "iq4-xs", 136),
            (3, "dense", 0),
        ] {
            let count = outputs / 4;
            let cid: WeightId = id(format!(
                "component.{name}.{}.{}",
                banks - 1 - bank,
                3 - part
            ));
            let quantized = bytes != 0;
            let mut dims = vec![count, if quantized { inputs / 256 } else { inputs }];
            let mut offsets = vec![part * count, 0];
            let mut extents = vec![count, inputs];
            if banks == 2 {
                dims.insert(0, 1);
                offsets.insert(0, bank);
                extents.insert(0, 1);
            }
            schema.components.push(WeightComponentSpec {
                id: cid.clone(),
                role: if quantized {
                    WeightComponentRole::PackedValues
                } else {
                    WeightComponentRole::Values
                },
                external_names: vec![format!("{name}.{bank}.{format}")],
                dimensions: dims,
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
            parts.push(CompositeWeightPart {
                layout: Box::new(if quantized {
                    PhysicalWeightLayout::BlockQuantized {
                        blocks: PhysicalWeightComponentBinding::exact_contiguous(cid),
                        block_axis: if banks == 2 { 2 } else { 1 },
                        block_padding: PhysicalWeightPadding::Exact,
                    }
                } else {
                    PhysicalWeightLayout::Dense { component_id: cid }
                }),
                logical_offsets: offsets,
                extents,
            });
        }
    }
    schema.tensors.push(WeightTensorSpec {
        id: id(format!("weight.{name}")),
        dimensions: if banks == 2 {
            vec![2, outputs, inputs]
        } else {
            vec![outputs, inputs]
        },
        logical_element_type: ElementType::F16,
        physical_layout: PhysicalWeightLayout::Composite { parts },
        required: true,
    });
}

#[cfg(test)]
#[path = "family/tests.rs"]
mod tests;
