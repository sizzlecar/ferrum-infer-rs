use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use std::sync::atomic::{AtomicUsize, Ordering};

fn component(name: &str) -> WeightComponentSpec {
    WeightComponentSpec {
        id: WeightId::new(format!("component.{name}")).unwrap(),
        role: WeightComponentRole::PackedValues,
        external_names: vec![format!("{name}.weight")],
        dimensions: vec![256, 1],
        encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: QuantizationFormatId::new(GgufBlockFormat::Q4K.format_id()).unwrap(),
            logical_values_per_block: 256,
            bytes_per_block: 144,
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
                    strides_in_elements: vec![256, 1, 1],
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
            dimensions: vec![2, 256, 256],
            logical_element_type: ElementType::F16,
            required: true,
            physical_layout: PhysicalWeightLayout::Composite {
                parts: components[..2]
                    .iter()
                    .enumerate()
                    .map(|(i, c)| CompositeWeightPart {
                        layout: Box::new(block(&c.id, 3)),
                        logical_offsets: vec![i as u64, 0, 0],
                        extents: vec![1, 256, 256],
                    })
                    .collect(),
            },
        },
        WeightTensorSpec {
            id: down.clone(),
            dimensions: vec![256, 256],
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
        operation_id: OperationId::new(DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID)
            .unwrap(),
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
                dimensions: vec![2, 256, 256],
                element_type: ElementType::F16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        },
        WeightReference {
            weight_id: down.clone(),
            value_id: down_value.clone(),
            tensor: ProgramTensorSpec {
                dimensions: vec![256, 256],
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
                dimensions: vec![256, 256],
                logical_element_type: ElementType::F16,
                required: true,
                physical_layout: block(&head.id, 2),
            });
            components.push(head);
            weights.push(WeightReference {
                weight_id: id,
                value_id: value.clone(),
                tensor: ProgramTensorSpec {
                    dimensions: vec![256, 256],
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
        format_id: WeightFormatId::new("weight-format.gguf.native-block").unwrap(),
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
fn dual_schema_binds_same_sources_and_accounts_dense_packet_and_retained_weights() {
    let (schema, program) = fixture(true, false);
    let prepared = plan::prepare_program(&schema, &program).unwrap();
    let mut packet_bytes = 0;
    for original in &schema.tensors[..2] {
        let tensor = prepared.schema.tensor(&original.id).unwrap();
        let PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
            dense_values,
            fragment_values,
            source_format,
        } = &tensor.physical_layout
        else {
            panic!("dual layout required");
        };
        assert_eq!(
            prepared.sources[&dense_values.component_id],
            prepared.sources[&fragment_values.component_id]
        );
        let checked =
            RnF16FragmentPlanV1::from_dimensions(*source_format, &original.dimensions).unwrap();
        packet_bytes += checked.packed_bytes();
        let packet = prepared
            .schema
            .components
            .iter()
            .find(|c| c.id == fragment_values.component_id)
            .unwrap();
        assert_eq!(packet.physical_bytes().unwrap(), checked.packed_bytes());
        assert_eq!(packet.encoding, checked.packed_encoding());
        assert_eq!(packet.dimensions, checked.packed_dimensions());
    }
    assert_eq!(prepared.inventory.packed_fragment_bytes, packet_bytes);
    assert_eq!(prepared.inventory.converted_f16_bytes, 3 * 256 * 256 * 2);
    assert_eq!(prepared.inventory.retained_consumed_bytes, 256 * 144);
    assert_eq!(
        prepared.inventory.unique_consumed_execution_bytes,
        3 * 256 * 256 * 2 + packet_bytes + 256 * 144
    );
    assert_eq!(prepared.schema.components.len(), 5);
    assert!(!prepared.inventory.includes_placement_alignment);
    // Unchanged strict embedding/head remains the original source component.
    assert!(prepared.schema.components.contains(&schema.components[3]));
}

#[test]
fn fragment_consumer_and_format_authority_do_not_leak_to_original_profiles() {
    let (schema, program) = fixture(false, true);
    assert!(plan::prepare_program(&schema, &program).is_err());
    let (mut schema, program) = fixture(false, false);
    let mut old_wire = serde_json::to_value(&program).unwrap();
    old_wire["blocks"][0]["nodes"][0]["operation_id"] =
        serde_json::json!(DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID);
    let old: ModelProgram = serde_json::from_value(old_wire).unwrap();
    assert!(plan::prepare_program(&schema, &old).is_err());
    assert!(
        crate::gguf_f16_projection_materializer::plan::prepare_program_with_roles(
            &schema,
            &old,
            gguf_f16_projection_role_v1
        )
        .is_ok()
    );
    schema.components[1].encoding = WeightEncoding::BlockQuantized(
        RnF16FragmentPlanV1::new(RnF16FragmentSourceFormatV1::Q5K, 256, 256)
            .unwrap()
            .source_block_spec(),
    );
    assert!(plan::prepare_program(&schema, &program).is_err());
    let (mut schema, program) = fixture(false, false);
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

struct Sources {
    calls: AtomicUsize,
    bytes: BTreeMap<WeightId, Vec<u8>>,
}
impl WeightComponentSource for Sources {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            vec![format!("{}.gguf", component.id)],
            component.dimensions.clone(),
            component.physical_element_type(),
            self.bytes[&component.id].as_slice(),
        )
    }
}

#[test]
fn dual_materialization_reads_each_source_once_and_checks_targets_before_io() {
    for (source_format, format) in [
        (RnF16FragmentSourceFormatV1::Q4K, GgufBlockFormat::Q4K),
        (RnF16FragmentSourceFormatV1::Q5K, GgufBlockFormat::Q5K),
        (RnF16FragmentSourceFormatV1::Q6K, GgufBlockFormat::Q6K),
    ] {
        let mut gate = component("gate");
        gate.encoding = WeightEncoding::BlockQuantized(
            RnF16FragmentPlanV1::new(source_format, 256, 256)
                .unwrap()
                .source_block_spec(),
        );
        let mut up = component("up");
        up.encoding = gate.encoding.clone();
        let source_group = [&gate, &up];
        let dense = crate::gguf_f16_projection_materializer::conversion::derived_component(
            &source_group,
            &[2, 256, 256],
        )
        .unwrap();
        let (checked, packet) = conversion::fragment_component(&source_group).unwrap();
        let block = crate::gguf_f16_projection_materializer::quality::vectors()
            .into_iter()
            .find(|v| v.format == format)
            .unwrap()
            .source;
        let mut up_block = block.clone();
        up_block[16] ^= 15;
        let source = Sources {
            calls: AtomicUsize::new(0),
            bytes: BTreeMap::from([
                (gate.id.clone(), block.repeat(256)),
                (up.id.clone(), up_block.repeat(256)),
            ]),
        };
        let output =
            conversion::materialize_group(&source, &source_group, &[&packet, &dense]).unwrap();
        assert_eq!(source.calls.load(Ordering::Relaxed), 2);
        let ordered = [
            source.bytes[&gate.id].as_slice(),
            source.bytes[&up.id].as_slice(),
        ];
        assert_eq!(
            output[0].bytes(),
            crate::gguf_rn_fragment::pack_rn_f16_fragments(&checked, &ordered).unwrap()
        );
        let expected = crate::gguf_f16_projection_materializer::convert_rn_f16_diagnostic(
            format,
            &ordered.concat(),
        )
        .unwrap();
        assert_eq!(output[1].bytes(), expected);
        assert_eq!(output[0].source_files(), output[1].source_files());
        assert!(conversion::materialize_group(&source, &[&up, &gate], &[&packet, &dense]).is_err());
        assert!(
            conversion::materialize_group(&source, &source_group, &[&packet, &packet]).is_err()
        );
        let mut altered = packet.clone();
        altered.dimensions[0] += 1;
        assert!(
            conversion::materialize_group(&source, &source_group, &[&altered, &dense]).is_err()
        );
        assert_eq!(source.calls.load(Ordering::Relaxed), 2);
        // A retained alias, dense RN, and packet can also share one physical read.
        let single_dense = crate::gguf_f16_projection_materializer::conversion::derived_component(
            &[&gate],
            &[256, 256],
        )
        .unwrap();
        let (_, single_packet) = conversion::fragment_component(&[&gate]).unwrap();
        let shared = conversion::materialize_group(
            &source,
            &[&gate],
            &[&gate, &single_dense, &single_packet],
        )
        .unwrap();
        assert_eq!(source.calls.load(Ordering::Relaxed), 3);
        assert_eq!(shared[0].bytes(), source.bytes[&gate.id]);
    }
}

#[test]
fn fragment_quality_artifact_keeps_original_numeric_approval_boundary() {
    let (schema, program) = fixture(false, false);
    let prepared = plan::prepare_program(&schema, &program).unwrap();
    let materializer = GgufRnFragmentMaterializer::new().unwrap();
    assert_eq!(
        materializer.descriptor.fidelity(),
        WeightMaterializationFidelity::Approximate
    );
    let artifact = quality::artifact(materializer.descriptor(), &schema, &prepared.schema).unwrap();
    assert!(WeightMaterializerSelection::numeric_quality_artifact(
        materializer.descriptor.id().clone(),
        artifact.clone()
    )
    .unwrap()
    .has_numeric_quality_artifact());
    let mut bad: serde_json::Value = serde_json::from_slice(&artifact).unwrap();
    bad["quality_vector_payload"]["fixture_id"] = serde_json::json!("unlocked");
    assert!(WeightMaterializerSelection::numeric_quality_artifact(
        materializer.descriptor.id().clone(),
        serde_json::to_vec(&bad).unwrap()
    )
    .is_err());
}

#[test]
fn rank_three_source_keeps_both_leading_axes_and_one_shared_read() {
    let (mut schema, program) = fixture(false, false);
    let mut packed = component("packed_gate_up");
    packed.dimensions = vec![2, 256, 1];
    schema.components.drain(..2);
    schema.components.push(packed.clone());
    schema.tensors[0].physical_layout = PhysicalWeightLayout::BlockQuantized {
        blocks: PhysicalWeightComponentBinding::exact_contiguous(packed.id.clone()),
        block_axis: 2,
        block_padding: PhysicalWeightPadding::Exact,
    };
    let prepared = plan::prepare_program(&schema, &program).unwrap();
    let components = prepared
        .schema
        .physical_component_refs(&schema.tensors[0].id)
        .unwrap();
    assert_eq!(components.len(), 2);
    for target in &components {
        assert_eq!(prepared.sources[&target.id], vec![packed.id.clone()]);
    }
    let block = crate::gguf_f16_projection_materializer::quality::vectors()
        .into_iter()
        .find(|v| v.format == GgufBlockFormat::Q4K)
        .unwrap()
        .source;
    let source = Sources {
        calls: AtomicUsize::new(0),
        bytes: BTreeMap::from([(packed.id.clone(), block.repeat(512))]),
    };
    let _ = conversion::materialize_group(&source, &[&packed], &components).unwrap();
    assert_eq!(source.calls.load(Ordering::Relaxed), 1);
    assert!(components.iter().any(|c| c.dimensions == [2, 256, 256]));
}
