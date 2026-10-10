//! Opt-in wide provider fixture, not a Qwen family or Q6-head qualification.
//! Its attention contracts are the exact new production operation contracts.
use super::*;

pub const PROFILE: &str = "fixture.attention-ffn.require-m8-mmq-geometry-v1";
pub const INHERITED_PROFILE: &str = "fixture.attention-ffn.wide-extra-all-rows-control";
pub const WIDE_HIDDEN: u64 = 5120;

#[derive(Clone, Copy, Debug)]
pub enum WideArithmetic {
    Geometry,
    InheritedExtraAllRows,
}

/// Synthetic FFN input range, independent of attention arithmetic/weights.
/// The legacy wide first-wave failure remains reproducible with an explicit
/// LegacyUnscaled selection; it is not a positive finite regression gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnInputGain {
    LegacyUnscaled,
    OneSixteenth,
}

impl FfnInputGain {
    pub fn multiplier(self) -> f32 {
        match self {
            Self::LegacyUnscaled => 1.0,
            Self::OneSixteenth => 1.0 / 16.0,
        }
    }
}

/// Delegate every component byte unchanged except the explicitly selected
/// wide FFN normalization gain. In particular, no packed scale is modified.
pub struct FixtureWeights {
    base: Weights,
    ffn_input_gain: FfnInputGain,
}

impl FixtureWeights {
    pub fn new(schema: &WeightSchema, ffn_input_gain: FfnInputGain) -> Self {
        Self {
            base: Weights::new(schema),
            ffn_input_gain,
        }
    }
}

impl WeightComponentSource for FixtureWeights {
    fn component<'s>(
        &'s self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        let original = self.base.component(component)?;
        if self.ffn_input_gain == FfnInputGain::LegacyUnscaled
            || component.id.as_str() != "component.ffn_norm"
        {
            return Ok(original);
        }
        assert_eq!(original.element_type(), ElementType::F16);
        let bytes: Vec<u8> = original
            .bytes()
            .chunks_exact(2)
            .flat_map(|b| {
                let gain = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let scaled = f16::from_f32(gain * self.ffn_input_gain.multiplier());
                assert!(scaled.is_normal() && scaled.to_f32() > 0.0);
                scaled.to_le_bytes()
            })
            .collect();
        WeightComponentPayload::new(
            component,
            original.external_name(),
            original.source_file(),
            original.dimensions().to_vec(),
            original.element_type(),
            bytes,
        )
    }
}

impl Family {
    pub fn m8_geometry(kind: AttentionKind) -> Self {
        Self::wide_arithmetic(kind, WideArithmetic::Geometry)
    }

    pub fn wide_arithmetic(kind: AttentionKind, arithmetic: WideArithmetic) -> Self {
        let mut value = Self::new(kind);
        value.geometry = true;
        value.ffn_input_gain = FfnInputGain::OneSixteenth;
        value.maximum_tokens = 128;
        value.selected = match (kind, arithmetic) {
            (AttentionKind::GatedDelta, WideArithmetic::Geometry) => {
                UpstreamMarkerV2Profile::GatedDeltaM8Geometry
            }
            (AttentionKind::Causal, WideArithmetic::Geometry) => {
                UpstreamMarkerV2Profile::CausalM8Geometry
            }
            (AttentionKind::GatedDelta, WideArithmetic::InheritedExtraAllRows) => {
                UpstreamMarkerV2Profile::GatedDeltaExtraAllRows
            }
            (AttentionKind::Causal, WideArithmetic::InheritedExtraAllRows) => {
                UpstreamMarkerV2Profile::CausalExtraAllRows
            }
            _ => unreachable!(),
        };
        value
    }

    pub fn with_ffn_input_gain(mut self, gain: FfnInputGain) -> Self {
        assert!(
            self.geometry,
            "range adjustment is only for the wide fixture"
        );
        self.ffn_input_gain = gain;
        self
    }

    pub fn ffn_input_gain(&self) -> FfnInputGain {
        self.ffn_input_gain
    }

    pub fn with_intermediate_observation(mut self) -> Self {
        self.observe_intermediates = true;
        self
    }

    pub fn observes_intermediates(&self) -> bool {
        self.observe_intermediates
    }

    pub fn is_geometry(&self) -> bool {
        self.geometry
    }

    pub fn hidden_size(&self) -> u64 {
        if self.geometry {
            WIDE_HIDDEN
        } else {
            HIDDEN
        }
    }
}

fn dimensions(name: &str, kind: AttentionKind) -> Vec<u64> {
    let h = WIDE_HIDDEN;
    match name {
        "embedding" => vec![32, h],
        "input_norm" => vec![h],
        "q" => vec![2 * h + 2048, h],
        "k" | "v" => vec![256, h],
        "q_norm" | "k_norm" | "output_norm" => vec![128],
        "qkvzba" => vec![2 * h + 512 + h / 64, h],
        "conv" => vec![h + 512, 4],
        "negative_rate" | "dt_bias" => vec![h / 128],
        "o" => vec![
            h,
            if kind == AttentionKind::Causal {
                h + 1024
            } else {
                h
            },
        ],
        _ => panic!("undeclared wide fixture weight {name}"),
    }
}

pub(super) fn resize_states(states: &mut [StateSpec]) {
    for state in states {
        state.tensor.dimensions = match state.id.as_str() {
            "state.conv" => vec![WIDE_HIDDEN + 512, 3],
            "state.delta" => vec![WIDE_HIDDEN / 128, 128, 128],
            // Preserve the original FP16 KV ABI, including physical page size.
            "state.kv" => vec![2, 2, 128],
            _ => panic!("undeclared wide fixture state"),
        };
    }
}

pub(super) fn resize_weights(weights: &mut [WeightReference], kind: AttentionKind) {
    for weight in weights {
        weight.tensor.dimensions = dimensions(
            weight.weight_id.as_str().strip_prefix("weight.").unwrap(),
            kind,
        );
    }
}

pub(super) fn resize_nodes(blocks: &mut [ProgramBlock], kind: AttentionKind) {
    blocks[0].nodes[0]
        .attributes
        .insert(id("hidden_size"), SemanticValue::Unsigned(WIDE_HIDDEN));
    let values = match kind {
        AttentionKind::GatedDelta => vec![
            ("key_heads", 2),
            ("value_heads", WIDE_HIDDEN / 128),
            ("key_head_dim", 128),
            ("value_head_dim", 128),
            ("hidden_size", WIDE_HIDDEN),
            ("qkv_features", WIDE_HIDDEN + 512),
            ("value_features", WIDE_HIDDEN),
            ("qkvz_features", 2 * WIDE_HIDDEN + 512),
            ("ba_features", WIDE_HIDDEN / 64),
            ("qkvzba_features", 2 * WIDE_HIDDEN + 512 + WIDE_HIDDEN / 64),
        ],
        AttentionKind::Causal => vec![
            ("query_heads", WIDE_HIDDEN / 128 + 8),
            ("key_value_heads", 2),
            ("head_dim", 128),
            ("hidden_size", WIDE_HIDDEN),
            ("query_features", WIDE_HIDDEN + 1024),
            ("query_projection_features", 2 * WIDE_HIDDEN + 2048),
            ("kv_features", 256),
        ],
        _ => unreachable!(),
    };
    for (key, value) in values {
        blocks[0].nodes[1]
            .attributes
            .insert(id(key), SemanticValue::Unsigned(value));
    }
}

pub(super) fn resize_schema(schema: &mut WeightSchema, kind: AttentionKind) {
    let names: &[&str] = if kind == AttentionKind::GatedDelta {
        &["qkvzba", "o"]
    } else {
        &["q", "k", "v", "o"]
    };
    for tensor in &mut schema.tensors {
        let name = tensor.id.as_str().strip_prefix("weight.").unwrap();
        tensor.dimensions = dimensions(name, kind);
        if !names.contains(&name) {
            let PhysicalWeightLayout::Dense { component_id } = &tensor.physical_layout else {
                panic!("non-projection weights are dense")
            };
            schema
                .components
                .iter_mut()
                .find(|c| c.id == *component_id)
                .unwrap()
                .dimensions = tensor.dimensions.clone();
        }
    }
    for &name in names {
        let shape = dimensions(name, kind);
        let (n, k) = (shape[0], shape[1]);
        schema
            .components
            .retain(|c| !c.id.as_str().starts_with(&format!("component.{name}.")));
        let tensor = schema
            .tensors
            .iter_mut()
            .find(|t| t.id.as_str() == format!("weight.{name}"))
            .unwrap();
        let counts = match name {
            // Large Q4 and Q5 leaves plus a small Q4 leaf in the same projection.
            "qkvzba" => vec![
                ("q4-k", 5120, 144),
                ("q5-k", 5120, 176),
                ("q4-k", 256, 144),
                ("iq4-xs", 256, 136),
                ("dense", 80, 0),
            ],
            "q" => vec![
                ("q4-k", 5120, 144),
                ("q5-k", 5120, 176),
                ("q4-k", 512, 144),
                ("iq4-xs", 1024, 136),
                ("dense", 512, 0),
            ],
            _ => vec![
                ("q4-k", n / 4, 144),
                ("q5-k", n / 4, 176),
                ("iq4-xs", n / 4, 136),
                ("dense", n / 4, 0),
            ],
        };
        let mut offset = 0;
        let mut parts = Vec::new();
        for (index, (format, rows, bytes)) in counts.into_iter().enumerate() {
            let component_id: WeightId = id(format!("component.{name}.wide.{index}"));
            let quantized = bytes != 0;
            schema.components.push(WeightComponentSpec {
                id: component_id.clone(),
                role: if quantized {
                    WeightComponentRole::PackedValues
                } else {
                    WeightComponentRole::Values
                },
                external_names: vec![format!("{name}.{index}.{format}")],
                dimensions: vec![rows, if quantized { k / 256 } else { k }],
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
                        blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
                        block_axis: 1,
                        block_padding: PhysicalWeightPadding::Exact,
                    }
                } else {
                    PhysicalWeightLayout::Dense { component_id }
                }),
                logical_offsets: vec![offset, 0],
                extents: vec![rows, k],
            });
            offset += rows;
        }
        assert_eq!(offset, n);
        tensor.physical_layout = PhysicalWeightLayout::Composite { parts };
    }
}

#[test]
fn wide_geometry_fixture_prepares_exact_contract_and_physical_mixed_leaf_extents() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let family = Family::m8_geometry(kind);
        let operation = family.attention_operation();
        let arithmetic = family.attention_arithmetic();
        let prepared = TypedFamilyRegistration::new(family)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(PROFILE))
            .unwrap();
        let attention = &prepared.program().blocks()[0].nodes[1];
        assert_eq!(attention.operation_id.as_str(), operation);
        assert_eq!(
            attention.attributes[&id("hidden_size")],
            SemanticValue::Unsigned(WIDE_HIDDEN)
        );
        assert_eq!(
            prepared
                .numerical_profile()
                .operations
                .iter()
                .find(|o| o.operation_id.as_str() == operation)
                .unwrap()
                .composite_arithmetic,
            Some(arithmetic)
        );
        let name = if kind == AttentionKind::GatedDelta {
            "weight.qkvzba"
        } else {
            "weight.q"
        };
        let tensor = prepared
            .weight_schema()
            .tensors
            .iter()
            .find(|t| t.id.as_str() == name)
            .unwrap();
        let PhysicalWeightLayout::Composite { parts } = &tensor.physical_layout else {
            panic!("mixed physical leaves")
        };
        assert!(parts.iter().any(|p| p.extents == [5120, 5120]));
        assert!(parts
            .iter()
            .any(|p| p.extents[0] < 5120 && p.extents[1] == 5120));
        assert_ne!(PROFILE, ferrum_models::vnext::qwen35::F32_MASTER_Q6_HEAD_ATTENTION_M8_MMQ_GEOMETRY_V1_NUMERICAL_PROFILE_ID);
    }
}
