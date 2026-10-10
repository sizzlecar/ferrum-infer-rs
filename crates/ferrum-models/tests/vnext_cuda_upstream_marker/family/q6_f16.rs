//! Opt-in real Q6 blocks, bounded wide state and joined FFN output banks.
//! This synthetic provider family does not claim Qwen profile/head eligibility.
use super::*;

pub const PROFILE: &str = "fixture.attention-ffn.require-q6-f16-mmq-v1";
pub const INTERMEDIATE: u64 = 6656;

impl Family {
    pub fn q6_f16(kind: AttentionKind) -> Self {
        let mut value = Self::m8_geometry(kind);
        value.selected = match kind {
            AttentionKind::GatedDelta => UpstreamMarkerV2Profile::GatedDeltaQ6F16,
            AttentionKind::Causal => UpstreamMarkerV2Profile::CausalQ6F16,
            _ => unreachable!(),
        };
        value
    }

    pub fn intermediate_size(&self) -> u64 {
        if self.selected.q6_f16() {
            INTERMEDIATE
        } else {
            super::INTERMEDIATE
        }
    }
}

pub(super) fn replace_projection_leaves(schema: &mut WeightSchema) {
    for tensor in &mut schema.tensors {
        let PhysicalWeightLayout::Composite { parts } = &tensor.physical_layout else {
            continue;
        };
        let old_ids: BTreeSet<_> = parts
            .iter()
            .map(|part| match part.layout.as_ref() {
                PhysicalWeightLayout::Dense { component_id } => component_id.clone(),
                PhysicalWeightLayout::BlockQuantized { blocks, .. } => blocks.component_id.clone(),
                _ => panic!("fixture only replaces explicit dense/block leaves"),
            })
            .collect();
        schema.components.retain(|c| !old_ids.contains(&c.id));
        let rank = tensor.dimensions.len();
        let banked = rank == 3;
        let banks = if banked { tensor.dimensions[0] } else { 1 };
        let n = tensor.dimensions[rank - 2];
        let k = tensor.dimensions[rank - 1];
        assert_eq!(k % 256, 0);
        let rows: &[u64] = match tensor.id.as_str() {
            // Each large input projection contains an always-eligible Q6 leaf,
            // M8-only middle leaves, and a physically small strict leaf.
            "weight.qkvzba" => &[5120, 4096, 1024, 512, 80],
            "weight.q" => &[5120, 4096, 2048, 768, 256],
            "weight.k" | "weight.v" => &[128, 128],
            "weight.o" | "weight.ffn_down" => &[4096, 768, 256],
            "weight.ffn_gate_up" => &[5120, 1024, 256, 256],
            other => panic!("undeclared Q6 fixture projection {other}"),
        };
        assert_eq!(rows.iter().sum::<u64>(), n);
        let mut next = Vec::new();
        for bank in 0..banks {
            let mut offset = 0;
            for (part, &count) in rows.iter().enumerate() {
                let quantized = part + 1 != rows.len();
                let cid: WeightId = id(format!("component.q6.{}.bank{bank}.{part}", tensor.id));
                let mut dimensions = vec![count, if quantized { k / 256 } else { k }];
                let mut logical_offsets = vec![offset, 0];
                let mut extents = vec![count, k];
                if banked {
                    dimensions.insert(0, 1);
                    logical_offsets.insert(0, bank);
                    extents.insert(0, 1);
                }
                schema.components.push(WeightComponentSpec {
                    id: cid.clone(),
                    role: if quantized {
                        WeightComponentRole::PackedValues
                    } else {
                        WeightComponentRole::Values
                    },
                    external_names: vec![format!(
                        "{}.bank{bank}.{part}.{}",
                        tensor.id,
                        if quantized { "q6-k" } else { "dense" }
                    )],
                    dimensions,
                    encoding: if quantized {
                        WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                            format_id: id("quantization.gguf.q6-k"),
                            logical_values_per_block: 256,
                            bytes_per_block: 210,
                        })
                    } else {
                        WeightEncoding::Dense {
                            element_type: ElementType::F16,
                        }
                    },
                    required: true,
                });
                next.push(CompositeWeightPart {
                    layout: Box::new(if quantized {
                        PhysicalWeightLayout::BlockQuantized {
                            blocks: PhysicalWeightComponentBinding::exact_contiguous(cid),
                            block_axis: if banked { 2 } else { 1 },
                            block_padding: PhysicalWeightPadding::Exact,
                        }
                    } else {
                        PhysicalWeightLayout::Dense { component_id: cid }
                    }),
                    logical_offsets,
                    extents,
                });
                offset += count;
            }
        }
        tensor.physical_layout = PhysicalWeightLayout::Composite { parts: next };
    }
}

#[test]
fn q6_f16_fixture_prepares_real_blocks_and_joined_ffn_banks() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let family = Family::q6_f16(kind);
        let attention = family.attention_profile();
        let ffn = family.swiglu_profile();
        let prepared = TypedFamilyRegistration::new(family)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(PROFILE))
            .unwrap();
        for selected in [attention, ffn] {
            assert_eq!(
                prepared
                    .numerical_profile()
                    .operations
                    .iter()
                    .find(|op| op.operation_id.as_str() == selected.operation_id())
                    .unwrap()
                    .composite_arithmetic,
                Some(selected.arithmetic())
            );
        }
        let schema = prepared.weight_schema();
        let joined = schema
            .tensors
            .iter()
            .find(|t| t.id.as_str() == "weight.ffn_gate_up")
            .unwrap();
        assert_eq!(joined.dimensions, [2, INTERMEDIATE, geometry::WIDE_HIDDEN]);
        let PhysicalWeightLayout::Composite { parts } = &joined.physical_layout else {
            panic!("joined physical leaves")
        };
        for bank in 0..2 {
            let eligible = parts
                .iter()
                .find(|p| p.logical_offsets == [bank, 0, 0])
                .unwrap();
            assert_eq!(eligible.extents, [1, 5120, 5120]);
            let PhysicalWeightLayout::BlockQuantized { blocks, .. } = eligible.layout.as_ref()
            else {
                panic!("actual Q6 block leaf")
            };
            let component = schema
                .components
                .iter()
                .find(|c| c.id == blocks.component_id)
                .unwrap();
            assert!(
                matches!(&component.encoding, WeightEncoding::BlockQuantized(s) if s.format_id.as_str() == "quantization.gguf.q6-k" && s.bytes_per_block == 210)
            );
        }
        let down = schema
            .tensors
            .iter()
            .find(|t| t.id.as_str() == "weight.ffn_down")
            .unwrap();
        assert_eq!(down.dimensions, [geometry::WIDE_HIDDEN, INTERMEDIATE]);
        assert_eq!(
            prepared.program().blocks()[0]
                .nodes
                .iter()
                .find(|n| n.id.as_str() == "node.swiglu")
                .unwrap()
                .attributes[&id("intermediate_size")],
            SemanticValue::Unsigned(INTERMEDIATE)
        );
    }
}
