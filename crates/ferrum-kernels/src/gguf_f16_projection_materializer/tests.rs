use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use half::f16;
use std::sync::atomic::{AtomicUsize, Ordering};

fn component(name: &str) -> WeightComponentSpec {
    WeightComponentSpec {
        id: WeightId::new(format!("component.{name}")).unwrap(),
        role: WeightComponentRole::PackedValues,
        external_names: vec![format!("{name}.weight")],
        dimensions: vec![32, 1],
        encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: QuantizationFormatId::new(GgufBlockFormat::Q8_0.format_id()).unwrap(),
            logical_values_per_block: 32,
            bytes_per_block: 34,
        }),
        required: true,
    }
}
fn block(id: &WeightId, rank: usize) -> PhysicalWeightLayout {
    PhysicalWeightLayout::BlockQuantized {
        blocks: PhysicalWeightComponentBinding {
            component_id: id.clone(),
            storage: if rank == 2 {
                PhysicalStorageLayout::exact_contiguous()
            } else {
                PhysicalStorageLayout::Strided {
                    strides_in_elements: vec![32, 1, 1],
                    padding: PhysicalWeightPadding::Exact,
                }
            },
        },
        block_axis: (rank - 1) as u32,
        block_padding: PhysicalWeightPadding::Exact,
    }
}
fn fixture(tied_original: bool, unauthorized: bool) -> (WeightSchema, ModelProgram) {
    let mut components = vec![component("gate"), component("up"), component("down")];
    let gateup = WeightId::new("weight.gateup").unwrap();
    let down = WeightId::new("weight.down").unwrap();
    let mut tensors = vec![
        WeightTensorSpec {
            id: gateup.clone(),
            dimensions: vec![2, 32, 32],
            logical_element_type: ElementType::F16,
            required: true,
            physical_layout: PhysicalWeightLayout::Composite {
                parts: components[..2]
                    .iter()
                    .enumerate()
                    .map(|(i, c)| CompositeWeightPart {
                        layout: Box::new(block(&c.id, 3)),
                        logical_offsets: vec![i as u64, 0, 0],
                        extents: vec![1, 32, 32],
                    })
                    .collect(),
            },
        },
        WeightTensorSpec {
            id: down.clone(),
            dimensions: vec![32, 32],
            logical_element_type: ElementType::F16,
            required: true,
            physical_layout: block(&components[2].id, 2),
        },
    ];
    let input = ProgramValueId::new("value.input").unwrap();
    let gate_value = ProgramValueId::new("value.gateup").unwrap();
    let down_value = ProgramValueId::new("value.down").unwrap();
    let output = ProgramValueId::new("value.output").unwrap();
    let mut nodes = vec![ProgramNode {
        id: NodeId::new("node.ffn").unwrap(),
        operation_id: OperationId::new(DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID).unwrap(),
        required_version: VERSION,
        work: ProgramNodeWorkSpec::Fixed,
        inputs: vec![input.clone(), gate_value.clone(), down_value.clone()],
        outputs: vec![output.clone()],
        attributes: BTreeMap::new(),
    }];
    let mut weights = vec![
        WeightReference {
            weight_id: gateup,
            value_id: gate_value,
            tensor: ProgramTensorSpec {
                dimensions: vec![2, 32, 32],
                element_type: ElementType::F16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        },
        WeightReference {
            weight_id: down.clone(),
            value_id: down_value.clone(),
            tensor: ProgramTensorSpec {
                dimensions: vec![32, 32],
                element_type: ElementType::F16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        },
    ];
    let mut final_output = output.clone();
    if tied_original || unauthorized {
        let head_value = if tied_original {
            let id = WeightId::new("weight.head").unwrap();
            let value = ProgramValueId::new("value.head").unwrap();
            // Tied weights share one logical weight value across consumers.
            // Component identities and external names stay unique in a schema.
            let head = component("head");
            tensors.push(WeightTensorSpec {
                id: id.clone(),
                dimensions: vec![32, 32],
                logical_element_type: ElementType::F16,
                required: true,
                physical_layout: block(&head.id, 2),
            });
            components.push(head);
            weights.push(WeightReference {
                weight_id: id,
                value_id: value.clone(),
                tensor: ProgramTensorSpec {
                    dimensions: vec![32, 32],
                    element_type: ElementType::F16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            });
            let embedded = ProgramValueId::new("value.embedded").unwrap();
            nodes[0].inputs[0] = embedded.clone();
            nodes.insert(
                0,
                ProgramNode {
                    id: NodeId::new("node.embedding").unwrap(),
                    operation_id: OperationId::new(TOKEN_EMBEDDING_OPERATION_ID).unwrap(),
                    required_version: VERSION,
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![input.clone(), value.clone()],
                    outputs: vec![embedded],
                    attributes: BTreeMap::new(),
                },
            );
            value
        } else {
            down_value
        };
        final_output = ProgramValueId::new("value.head.output").unwrap();
        nodes.push(ProgramNode {
            id: NodeId::new("node.head").unwrap(),
            operation_id: OperationId::new(LAST_TOKEN_DENSE_LINEAR_OPERATION_ID).unwrap(),
            required_version: VERSION,
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![output, head_value],
            outputs: vec![final_output.clone()],
            attributes: BTreeMap::new(),
        });
    }
    let schema = WeightSchema {
        format_id: WeightFormatId::new(SOURCE_FORMAT).unwrap(),
        layout_id: WeightLayoutId::new("weight-layout.fixture.gguf-projection").unwrap(),
        version: VERSION,
        components,
        tensors,
    };
    let program = ModelProgram::new(
        ModelFamilyId::new("family.fixture.gguf-rn-f16").unwrap(),
        vec![input],
        vec![ProgramBlock {
            id: "block.main".into(),
            nodes,
        }],
        vec![],
        weights,
        vec![final_output],
    )
    .unwrap();
    schema.validate(program.family_id()).unwrap();
    (schema, program)
}

#[test]
fn gguf_rn_f16_conversion_matches_independent_coefficients_and_rounding_boundaries() {
    for v in quality::vectors() {
        let mut decoded = vec![0.; v.reference.len()];
        v.format.decode(&v.source, &mut decoded).unwrap();
        assert_eq!(
            decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            v.reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let converted = conversion::convert(v.format, &v.source).unwrap();
        assert!(converted
            .chunks_exact(2)
            .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff != 0));
        let mut err = 0f64;
        let mut norm = 0f64;
        for (bytes, reference) in converted.chunks_exact(2).zip(&v.reference) {
            let actual = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
            assert!(actual.is_finite());
            err += (f64::from(actual) - f64::from(*reference)).powi(2);
            norm += f64::from(*reference).powi(2);
        }
        assert!(err.sqrt() / norm.sqrt() < 1. / 1024.);
        assert!(conversion::convert(v.format, &v.source[..v.source.len() - 1]).is_err());
    }
    for (input, bits) in [
        (1. + 2f32.powi(-11), 0x3c00),
        (1. + 3. * 2f32.powi(-11), 0x3c02),
        (2f32.powi(-25), 0),
        (3. * 2f32.powi(-25), 2),
        (-0., 0x8000),
    ] {
        assert_eq!(conversion::round_finite(input).unwrap(), bits);
    }
    for value in [f32::NAN, f32::INFINITY, -f32::INFINITY, 65520., -65520.] {
        assert!(conversion::round_finite(value).is_err());
    }
}
#[test]
fn gguf_rn_f16_schema_keeps_composite_order_and_shared_unconverted_head_embedding() {
    let (schema, program) = fixture(true, false);
    let prepared = plan::prepare_program(&schema, &program).unwrap();
    let source_head = &schema.components[3];
    assert!(prepared.schema.components.iter().any(|c| c == source_head));
    assert!(!prepared
        .schema
        .components
        .iter()
        .any(|c| c.id == schema.components[2].id));
    assert_eq!(prepared.sources.len(), 3);
    assert_eq!(prepared.inventory.converted_f16_bytes, 3 * 32 * 32 * 2);
    assert_eq!(prepared.inventory.retained_consumed_bytes, 32 * 34);
    let head_inventory = prepared
        .inventory
        .weights
        .iter()
        .find(|weight| weight.logical_weight.as_str() == "weight.head")
        .unwrap();
    assert_eq!(head_inventory.consumers.len(), 2);
    assert!(head_inventory
        .consumers
        .iter()
        .all(|consumer| consumer.role.is_none()));
    // This inventory counts declared components, not unique bytes in the file.
    assert_eq!(prepared.inventory.source_consumed_bytes, 4 * 32 * 34);
    assert_eq!(
        prepared
            .schema
            .tensor(&WeightId::new("weight.head").unwrap()),
        schema.tensor(&WeightId::new("weight.head").unwrap())
    );
    let PhysicalWeightLayout::Dense { component_id } = &prepared
        .schema
        .tensor(&WeightId::new("weight.gateup").unwrap())
        .unwrap()
        .physical_layout
    else {
        panic!("dense packed gate/up is required by cuBLAS")
    };
    assert_eq!(
        prepared.sources[component_id],
        vec![
            schema.components[0].id.clone(),
            schema.components[1].id.clone()
        ]
    );
    assert_eq!(
        prepared
            .schema
            .components
            .iter()
            .find(|c| &c.id == component_id)
            .unwrap()
            .dimensions,
        vec![2, 32, 32]
    );
    let (schema, program) = fixture(false, false);
    let prepared = plan::prepare_program(&schema, &program).unwrap();
    assert_eq!(prepared.schema.components.len(), 2);
    assert!(prepared.schema.components.iter().all(|c| c.encoding
        == WeightEncoding::Dense {
            element_type: ElementType::F16
        }));
    assert!(prepared
        .schema
        .components
        .iter()
        .all(|c| !schema.components.iter().any(|s| s.id == c.id)));
}
#[test]
fn gguf_rn_f16_rejects_mixed_consumer_and_unauthorized_source_layout() {
    let (schema, program) = fixture(false, true);
    assert!(plan::prepare_program(&schema, &program).is_err());
    let (mut schema, program) = fixture(false, false);
    schema.format_id = WeightFormatId::new("weight-format.safetensors").unwrap();
    assert!(plan::prepare_program(&schema, &program).is_err());
    let (schema, program) = fixture(false, false);
    let mut wire = serde_json::to_value(&program).unwrap();
    wire["blocks"][0]["nodes"][0]["operation_id"] = serde_json::json!(DENSE_SWIGLU_OPERATION_ID);
    let old: ModelProgram = serde_json::from_value(wire).unwrap();
    assert!(plan::prepare_program(&schema, &old).is_err());
    let mut schema = schema;
    let PhysicalWeightLayout::BlockQuantized { blocks, .. } =
        &mut schema.tensors[1].physical_layout
    else {
        unreachable!()
    };
    blocks.storage = PhysicalStorageLayout::Strided {
        strides_in_elements: vec![2, 1],
        padding: PhysicalWeightPadding::Exact,
    };
    assert!(plan::prepare_program(&schema, &program).is_err());
}
struct CountingSource {
    calls: AtomicUsize,
    bytes: Vec<u8>,
}
impl WeightComponentSource for CountingSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            vec!["fixture.gguf".into()],
            component.dimensions.clone(),
            component.physical_element_type(),
            self.bytes.as_slice(),
        )
    }
}
#[test]
fn gguf_rn_f16_materialization_reads_once_and_rejects_wrong_target_before_io() {
    let source_component = component("gate");
    let derived = conversion::derived_component(&[&source_component], &[32, 32]).unwrap();
    let block = quality::vectors()
        .into_iter()
        .find(|v| v.format == GgufBlockFormat::Q8_0)
        .unwrap()
        .source;
    let source = CountingSource {
        calls: AtomicUsize::new(0),
        bytes: block.repeat(32),
    };
    let converted =
        conversion::materialize_group(&source, &[&source_component], &[&derived]).unwrap();
    assert_eq!(source.calls.load(Ordering::Relaxed), 1);
    assert_eq!(converted[0].bytes().len(), 32 * 32 * 2);
    assert_eq!(converted[0].source_files(), ["fixture.gguf"]);
    let mut wrong = derived.clone();
    wrong.dimensions[0] += 1;
    assert!(conversion::materialize_group(&source, &[&source_component], &[&wrong]).is_err());
    assert_eq!(source.calls.load(Ordering::Relaxed), 1);
    assert!(
        conversion::materialize_group(&source, &[&source_component], &[&derived, &derived])
            .is_err()
    );
    assert_eq!(source.calls.load(Ordering::Relaxed), 1);
    let shared = conversion::materialize_group(
        &source,
        &[&source_component],
        &[&source_component, &derived],
    )
    .unwrap();
    assert_eq!(source.calls.load(Ordering::Relaxed), 2);
    assert_eq!(shared[0].bytes(), source.bytes);
    assert_eq!(shared[1].bytes(), converted[0].bytes());
    let mut huge = source_component;
    huge.dimensions = vec![u64::MAX, u64::MAX];
    assert!(conversion::derived_component(&[&huge], &[32, 32]).is_err());
}

#[test]
fn gguf_rn_f16_materialization_stacks_whole_source_groups_in_declared_order() {
    let gate = component("gate");
    let up = component("up");
    let target = conversion::derived_component(&[&gate, &up], &[2, 32, 32]).unwrap();
    struct Sources {
        gate: WeightId,
        first: Vec<u8>,
        second: Vec<u8>,
    }
    impl WeightComponentSource for Sources {
        fn component<'source>(
            &'source self,
            c: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'source>, VNextError> {
            WeightComponentPayload::from_ordered_sources(
                c,
                c.external_names.clone(),
                vec!["fixture.gguf".into()],
                c.dimensions.clone(),
                c.physical_element_type(),
                if c.id == self.gate {
                    self.first.as_slice()
                } else {
                    self.second.as_slice()
                },
            )
        }
    }
    let mut first = vec![1u8; 34];
    first[..2].copy_from_slice(&f16::from_f32(1.).to_bits().to_le_bytes());
    let mut second = first.clone();
    second[2..].fill(2);
    let source = Sources {
        gate: gate.id.clone(),
        first: first.repeat(32),
        second: second.repeat(32),
    };
    let payload = conversion::materialize_group(&source, &[&gate, &up], &[&target])
        .unwrap()
        .remove(0);
    let values = payload
        .bytes()
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect::<Vec<_>>();
    assert!(values[..1024].iter().all(|v| *v == 1.));
    assert!(values[1024..].iter().all(|v| *v == 2.));
    assert!(conversion::materialize_group(&source, &[&up, &gate], &[&target]).is_err());
}
#[test]
fn gguf_rn_f16_locked_artifact_is_strict_numeric_evidence_not_exact_permission() {
    let (schema, program) = fixture(false, false);
    let prepared = plan::prepare_program(&schema, &program).unwrap();
    let materializer = GgufF16ProjectionMaterializer::new().unwrap();
    assert_eq!(
        materializer.descriptor.fidelity(),
        WeightMaterializationFidelity::Approximate
    );
    let artifact = quality::artifact(materializer.descriptor(), &schema, &prepared.schema).unwrap();
    let selection = WeightMaterializerSelection::numeric_quality_artifact(
        materializer.descriptor.id().clone(),
        artifact.clone(),
    )
    .unwrap();
    assert!(selection.has_numeric_quality_artifact());
    let mut bad: serde_json::Value = serde_json::from_slice(&artifact).unwrap();
    bad["quality_vector_payload"]["fixture_id"] = serde_json::json!("unlocked-vector");
    assert!(WeightMaterializerSelection::numeric_quality_artifact(
        materializer.descriptor.id().clone(),
        serde_json::to_vec(&bad).unwrap()
    )
    .is_err());
    // Complete live schema/descriptor containment is additionally enforced by
    // WeightMaterializerRegistry::select; this parser test grants no live plan.
}

#[test]
fn gguf_rn_f16_rank_three_packed_gate_up_preserves_every_matrix_row() {
    let (mut schema, program) = fixture(false, false);
    let mut packed = component("packed_gateup");
    packed.dimensions = vec![2, 32, 1];
    schema.components.drain(..2);
    schema.components.push(packed.clone());
    schema.tensors[0].physical_layout = PhysicalWeightLayout::BlockQuantized {
        blocks: PhysicalWeightComponentBinding::exact_contiguous(packed.id.clone()),
        block_axis: 2,
        block_padding: PhysicalWeightPadding::Exact,
    };
    schema.validate(program.family_id()).unwrap();
    let prepared = plan::prepare_program(&schema, &program).unwrap();
    assert_eq!(prepared.inventory.converted_f16_bytes, 3 * 32 * 32 * 2);
    let target = conversion::derived_component(&[&packed], &[2, 32, 32]).unwrap();
    assert!(prepared.schema.components.contains(&target));
    let mut source_bytes = Vec::new();
    for row in 1..=64_u8 {
        source_bytes.extend_from_slice(&f16::from_f32(1.).to_le_bytes());
        source_bytes.extend_from_slice(&[row; 32]);
    }
    let source = CountingSource {
        calls: AtomicUsize::new(0),
        bytes: source_bytes,
    };
    let payloads = conversion::materialize_group(&source, &[&packed], &[&target]).unwrap();
    assert_eq!(source.calls.load(Ordering::Relaxed), 1);
    assert_eq!(payloads[0].bytes().len(), 2 * 32 * 32 * 2);
    for (index, value) in payloads[0].bytes().chunks_exact(2).enumerate() {
        assert_eq!(
            f16::from_le_bytes([value[0], value[1]]).to_f32(),
            (index / 32 + 1) as f32,
            "gate/up row order at element {index}"
        );
    }
    let mut invalid = packed;
    invalid.dimensions = vec![2, u64::MAX, 1];
    assert!(conversion::derived_component(&[&invalid], &[2, 32, 32]).is_err());
    invalid.dimensions = vec![2, 32, u64::MAX];
    assert!(conversion::derived_component(&[&invalid], &[2, 32, 32]).is_err());
}
