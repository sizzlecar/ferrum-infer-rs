use super::*;
#[path = "family/extra.rs"]
pub mod extra;
#[path = "family/extra_all_rows.rs"]
pub mod extra_all_rows;
#[path = "family/extra_prefill.rs"]
pub mod extra_prefill;
pub use extra::Weights;

const INTERMEDIATE: u64 = 768;
const PROFILE: &str = "fixture.attention-ffn.require-upstream-marker-v2";

/// Keep the tested attention/state source; add a real typed FFN block rather
/// than pretending the F32 residual activation is an F16 projection input.
pub struct Family {
    base: attention_family::Family,
    selected: UpstreamMarkerV2Profile,
    maximum_tokens: u64,
    g32_baseline: bool,
}

impl Family {
    pub fn new(kind: AttentionKind) -> Self {
        Self {
            base: attention_family::Family::new(kind),
            maximum_tokens: MAX_TOKENS,
            g32_baseline: false,
            selected: match kind {
                AttentionKind::GatedDelta => UpstreamMarkerV2Profile::GatedDelta,
                AttentionKind::Causal => UpstreamMarkerV2Profile::Causal,
                _ => panic!("MarkerV2 gate declares GDN and FP16-KV causal"),
            },
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
        if self.selected.extra_all_rows() {
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
    pub fn states(&self) -> Vec<StateSpec> {
        let mut states = self.base.states();
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
        if self.g32_baseline {
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
        let norm: WeightId = id("component.ffn_norm");
        schema.components.push(WeightComponentSpec {
            id: norm.clone(),
            role: WeightComponentRole::Values,
            external_names: vec!["ffn_norm".into()],
            dimensions: vec![HIDDEN],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        });
        schema.tensors.push(WeightTensorSpec {
            id: id("weight.ffn_norm"),
            dimensions: vec![HIDDEN],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Dense { component_id: norm },
            required: true,
        });
        mixed_weight(&mut schema, "ffn_gate_up", 2, INTERMEDIATE, HIDDEN);
        mixed_weight(&mut schema, "ffn_down", 1, HIDDEN, INTERMEDIATE);
        // One extra physical dense leaf in each operation makes the validation
        // payload an odd number of eight-byte banks. Its admitted Plan buffer
        // must include alignment padding; no dimensions or budget change.
        if self.selected.extra() {
            extra::extend_schema(&mut schema);
            // 7 leaves per bank gives 21 FFN leaves; only attention needs a
            // split to keep both admitted flag extents genuinely padded.
            split_dense_leaf(&mut schema, "o");
        } else {
            split_dense_leaf(&mut schema, "o");
            split_dense_leaf(&mut schema, "ffn_down");
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
        let attention = &mut blocks[0].nodes[1];
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
        let hidden = || BTreeMap::from([(id("hidden_size"), SemanticValue::Unsigned(HIDDEN))]);
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
            SemanticValue::Unsigned(INTERMEDIATE),
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
        for (name, dimensions) in [
            ("ffn_norm", vec![HIDDEN]),
            ("ffn_gate_up", vec![2, INTERMEDIATE, HIDDEN]),
            ("ffn_down", vec![HIDDEN, INTERMEDIATE]),
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
