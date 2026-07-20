mod vnext_core_contract;

use vnext_core_contract::*;

fn exact_component(component_id: &str) -> PhysicalWeightComponentBinding {
    PhysicalWeightComponentBinding::exact_contiguous(id(component_id))
}

#[test]
fn physical_weight_layout_tree_accepts_dense_fixture() {
    let family = TestRegistry::new().prepare();
    let schema = family.weight_schema();
    schema.validate(family.family_id()).unwrap();
    let components = schema
        .physical_component_refs(&id("weight.matrix"))
        .unwrap();
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].dimensions, [4]);
    assert_eq!(schema.physical_bytes(&id("weight.matrix")).unwrap(), 16);
    assert_eq!(
        schema
            .physical_resource_requirements(&id("weight.matrix"))
            .unwrap()[0]
            .physical_dimensions,
        [4]
    );
}

fn grouped_quantized_axis_index_schema() -> WeightSchema {
    let quantization = QuantizationSpec {
        format_id: id("quantization.grouped"),
        bits_per_weight: 4,
        group_size: 4,
        packing: QuantizationPacking::Linear,
        scale_type: ElementType::F16,
        zero_point_type: Some(ElementType::U8),
    };
    WeightSchema {
        format_id: id("weight-format.quantized"),
        layout_id: id("weight-layout.quantized-composite"),
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: id("component.packed"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["packed.bin".to_owned()],
                dimensions: vec![4, 8],
                encoding: WeightEncoding::Quantized(quantization),
                required: true,
            },
            WeightComponentSpec {
                id: id("component.scales"),
                role: WeightComponentRole::Scales,
                external_names: vec!["scales.bin".to_owned()],
                dimensions: vec![8, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            },
            WeightComponentSpec {
                id: id("component.zeros"),
                role: WeightComponentRole::ZeroPoints,
                external_names: vec!["zeros.bin".to_owned()],
                dimensions: vec![8, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::U8,
                },
                required: true,
            },
            WeightComponentSpec {
                id: id("component.axis-indices"),
                role: WeightComponentRole::Indices,
                external_names: vec!["axis-indices.bin".to_owned()],
                dimensions: vec![8],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::I32,
                },
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: id("weight.quantized"),
            dimensions: vec![8, 8],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Quantized {
                packed_values: exact_component("component.packed"),
                packed_dimensions: vec![4, 8],
                scales: exact_component("component.scales"),
                zero_points: Some(exact_component("component.zeros")),
                axis_indices: Some(AxisWeightComponent {
                    component: exact_component("component.axis-indices"),
                    axis: 1,
                }),
                permutation: None,
                codebook: None,
                group_axis: 1,
                group_padding: PhysicalWeightPadding::Exact,
            },
            required: true,
        }],
    }
}

#[test]
fn physical_weight_layout_tree_accepts_grouped_quantized_axis_index_fixture() {
    let schema = grouped_quantized_axis_index_schema();
    schema.validate(&id("family.quantized")).unwrap();
    assert_eq!(
        schema
            .physical_component_refs(&id("weight.quantized"))
            .unwrap()
            .len(),
        4
    );
    assert_eq!(schema.physical_bytes(&id("weight.quantized")).unwrap(), 112);
    let resources = schema
        .physical_resource_requirements(&id("weight.quantized"))
        .unwrap()
        .into_iter()
        .map(|component| {
            (
                component.component_id,
                (component.physical_dimensions, component.resource_bytes),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(resources[&id("component.packed")], (vec![4, 8], 32));
    assert_eq!(resources[&id("component.scales")], (vec![8, 2], 32));
    assert_eq!(resources[&id("component.zeros")], (vec![8, 2], 16));
    assert_eq!(resources[&id("component.axis-indices")], (vec![8], 32));

    let mut wrong = schema.clone();
    wrong.components[1].role = WeightComponentRole::Indices;
    assert!(wrong.validate(&id("family.quantized")).is_err());

    let mut indexed = schema.clone();
    indexed.components.push(WeightComponentSpec {
        id: id("component.lookup-indices"),
        role: WeightComponentRole::Indices,
        external_names: vec!["lookup-indices.bin".to_owned()],
        dimensions: vec![8],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::U32,
        },
        required: true,
    });
    let quantized_values = indexed.tensors[0].physical_layout.clone();
    indexed.tensors[0].physical_layout = PhysicalWeightLayout::Indexed {
        indices: AxisWeightComponent {
            component: exact_component("component.lookup-indices"),
            axis: 0,
        },
        values: Box::new(quantized_values),
        source_axis_extent: 8,
    };
    indexed.validate(&id("family.indexed-quantized")).unwrap();
    assert_eq!(
        indexed
            .physical_component_refs(&id("weight.quantized"))
            .unwrap()
            .len(),
        5
    );
}

fn block_quantization(format_id: &str, bytes_per_block: u32) -> BlockQuantizationSpec {
    BlockQuantizationSpec {
        format_id: id(format_id),
        logical_values_per_block: 256,
        bytes_per_block,
    }
}

fn q4_k_block_schema(logical_dimensions: Vec<u64>) -> WeightSchema {
    let block_dimensions = vec![logical_dimensions[0], logical_dimensions[1].div_ceil(256)];
    WeightSchema {
        format_id: id("weight-format.gguf"),
        layout_id: id("weight-layout.gguf-q4-k"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("component.q4-k.blocks"),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["blk.0.attn_q.weight".to_owned()],
            dimensions: block_dimensions,
            encoding: WeightEncoding::BlockQuantized(block_quantization(
                "quantization.gguf.q4-k",
                144,
            )),
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.attn-q"),
            dimensions: logical_dimensions,
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::BlockQuantized {
                blocks: exact_component("component.q4-k.blocks"),
                block_axis: 1,
                block_padding: PhysicalWeightPadding::Exact,
            },
            required: true,
        }],
    }
}

#[test]
fn block_quantized_layout_accounts_for_gguf_q4_k_blocks_exactly() {
    let schema = q4_k_block_schema(vec![2, 512]);
    schema.validate(&id("family.gguf-q4-k")).unwrap();

    assert_eq!(schema.physical_bytes(&id("weight.attn-q")).unwrap(), 576);
    assert_eq!(
        schema.quantization_formats(),
        BTreeSet::from([id("quantization.gguf.q4-k")])
    );
    let components = schema
        .physical_component_refs(&id("weight.attn-q"))
        .unwrap();
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].dimensions, [2, 2]);
    assert_eq!(components[0].physical_bytes().unwrap(), 576);
    assert_eq!(components[0].physical_element_type(), ElementType::U8);
    let resources = schema
        .physical_resource_requirements(&id("weight.attn-q"))
        .unwrap();
    assert_eq!(resources[0].physical_dimensions, [2, 2]);
    assert_eq!(resources[0].resource_bytes, 576);

    let encoded = serde_json::to_vec(&schema).unwrap();
    let restored: WeightSchema = serde_json::from_slice(&encoded).unwrap();
    restored.validate(&id("family.gguf-q4-k-restored")).unwrap();
    assert_eq!(restored, schema);
}

#[test]
fn composite_layout_supports_mixed_gguf_block_formats_without_model_cases() {
    let schema = WeightSchema {
        format_id: id("weight-format.gguf"),
        layout_id: id("weight-layout.gguf-mixed-fused-projection"),
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: id("component.fused.q4-k"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["blk.0.ffn_gate.weight".to_owned()],
                dimensions: vec![4, 1],
                encoding: WeightEncoding::BlockQuantized(block_quantization(
                    "quantization.gguf.q4-k",
                    144,
                )),
                required: true,
            },
            WeightComponentSpec {
                id: id("component.fused.q6-k"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["blk.0.ffn_up.weight".to_owned()],
                dimensions: vec![2, 1],
                encoding: WeightEncoding::BlockQuantized(block_quantization(
                    "quantization.gguf.q6-k",
                    210,
                )),
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: id("weight.fused-gate-up"),
            dimensions: vec![6, 256],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Composite {
                parts: vec![
                    CompositeWeightPart {
                        layout: Box::new(PhysicalWeightLayout::BlockQuantized {
                            blocks: exact_component("component.fused.q4-k"),
                            block_axis: 1,
                            block_padding: PhysicalWeightPadding::Exact,
                        }),
                        logical_offsets: vec![0, 0],
                        extents: vec![4, 256],
                    },
                    CompositeWeightPart {
                        layout: Box::new(PhysicalWeightLayout::BlockQuantized {
                            blocks: exact_component("component.fused.q6-k"),
                            block_axis: 1,
                            block_padding: PhysicalWeightPadding::Exact,
                        }),
                        logical_offsets: vec![4, 0],
                        extents: vec![2, 256],
                    },
                ],
            },
            required: true,
        }],
    };

    schema.validate(&id("family.gguf-mixed")).unwrap();
    assert_eq!(
        schema.physical_bytes(&id("weight.fused-gate-up")).unwrap(),
        996
    );
    assert_eq!(
        schema.quantization_formats(),
        BTreeSet::from([id("quantization.gguf.q4-k"), id("quantization.gguf.q6-k")])
    );

    let mut conflicting_abi = schema;
    let WeightEncoding::BlockQuantized(spec) = &mut conflicting_abi.components[1].encoding else {
        unreachable!();
    };
    spec.format_id = id("quantization.gguf.q4-k");
    assert!(conflicting_abi
        .validate(&id("family.gguf-conflicting-quantization-abi"))
        .unwrap_err()
        .to_string()
        .contains("conflicting physical ABIs"));
}

fn q5_k_tiled_qkv_schema() -> WeightSchema {
    WeightSchema {
        format_id: id("weight-format.gguf"),
        layout_id: id("weight-layout.gguf-qwen35-tiled-v-heads"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("component.linear-attn-qkv.q5-k"),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["blk.0.attn_qkv.weight".to_owned()],
            dimensions: vec![8192, 10],
            encoding: WeightEncoding::BlockQuantized(block_quantization(
                "quantization.gguf.q5-k",
                176,
            )),
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.linear-attn-qkv"),
            dimensions: vec![8192, 2560],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::AxisReshapePermutation {
                values: Box::new(PhysicalWeightLayout::BlockQuantized {
                    blocks: exact_component("component.linear-attn-qkv.q5-k"),
                    block_axis: 1,
                    block_padding: PhysicalWeightPadding::Exact,
                }),
                axis: 0,
                logical_offset: 4096,
                extent: 4096,
                reshape: vec![16, 2, 128],
                stored_axis_order: vec![1, 0, 2],
            },
            required: true,
        }],
    }
}

#[test]
fn axis_reshape_permutation_describes_gguf_tiled_head_subranges() {
    let schema = q5_k_tiled_qkv_schema();
    schema.validate(&id("family.gguf-tiled-heads")).unwrap();
    assert_eq!(
        schema
            .physical_bytes(&id("weight.linear-attn-qkv"))
            .unwrap(),
        14_417_920
    );
    assert_eq!(
        schema.quantization_formats(),
        BTreeSet::from([id("quantization.gguf.q5-k")])
    );
    let encoded = serde_json::to_vec(&schema).unwrap();
    let restored: WeightSchema = serde_json::from_slice(&encoded).unwrap();
    restored
        .validate(&id("family.gguf-tiled-heads-restored"))
        .unwrap();
    assert_eq!(restored, schema);

    let resolved =
        ResolvedWeightBinding::from_schema(&schema, &id("weight.linear-attn-qkv")).unwrap();
    assert!(matches!(
        resolved.physical_layout(),
        PhysicalWeightLayout::AxisReshapePermutation { .. }
    ));
    assert_eq!(resolved.components().len(), 1);
    assert_eq!(
        resolved.components()[0].physical_bytes().unwrap(),
        14_417_920
    );
    let encoded = serde_json::to_vec(&resolved).unwrap();
    let restored: ResolvedWeightBinding = serde_json::from_slice(&encoded).unwrap();
    restored
        .validate_logical(&[8192, 2560], ElementType::F16)
        .unwrap();
    assert_eq!(restored, resolved);
}

#[test]
fn axis_reshape_permutation_rejects_ambiguous_or_out_of_range_mappings() {
    let mut duplicate_axis = q5_k_tiled_qkv_schema();
    let PhysicalWeightLayout::AxisReshapePermutation {
        stored_axis_order, ..
    } = &mut duplicate_axis.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *stored_axis_order = vec![1, 1, 2];
    assert!(duplicate_axis
        .validate(&id("family.gguf-duplicate-stored-axis"))
        .is_err());

    let mut identity = q5_k_tiled_qkv_schema();
    let PhysicalWeightLayout::AxisReshapePermutation {
        stored_axis_order, ..
    } = &mut identity.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *stored_axis_order = vec![0, 1, 2];
    assert!(identity
        .validate(&id("family.gguf-identity-permutation"))
        .is_err());

    let mut singleton_identity = q5_k_tiled_qkv_schema();
    let PhysicalWeightLayout::AxisReshapePermutation {
        reshape,
        stored_axis_order,
        ..
    } = &mut singleton_identity.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *reshape = vec![1, 4096];
    *stored_axis_order = vec![1, 0];
    assert!(singleton_identity
        .validate(&id("family.gguf-singleton-identity-permutation"))
        .is_err());

    let mut wrong_product = q5_k_tiled_qkv_schema();
    let PhysicalWeightLayout::AxisReshapePermutation { reshape, .. } =
        &mut wrong_product.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *reshape = vec![16, 2, 64];
    assert!(wrong_product
        .validate(&id("family.gguf-wrong-reshape-product"))
        .is_err());

    let mut out_of_range = q5_k_tiled_qkv_schema();
    let PhysicalWeightLayout::AxisReshapePermutation { logical_offset, .. } =
        &mut out_of_range.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *logical_offset = 4097;
    assert!(out_of_range
        .validate(&id("family.gguf-transform-out-of-range"))
        .is_err());
}

#[test]
fn block_quantized_layout_rejects_implicit_geometry_and_invalid_encodings() {
    let mut wrong_grid = q4_k_block_schema(vec![2, 512]);
    wrong_grid.components[0].dimensions = vec![2, 3];
    assert!(wrong_grid.validate(&id("family.gguf-wrong-grid")).is_err());

    let mut zero_block_width = q4_k_block_schema(vec![2, 512]);
    let WeightEncoding::BlockQuantized(spec) = &mut zero_block_width.components[0].encoding else {
        unreachable!();
    };
    spec.logical_values_per_block = 0;
    assert!(zero_block_width
        .validate(&id("family.gguf-zero-block-width"))
        .is_err());

    let mut integer_logical_weight = q4_k_block_schema(vec![2, 512]);
    integer_logical_weight.tensors[0].logical_element_type = ElementType::U8;
    assert!(integer_logical_weight
        .validate(&id("family.gguf-integer-logical"))
        .is_err());

    let mut separate_scale_encoding = q4_k_block_schema(vec![2, 512]);
    separate_scale_encoding.components[0].encoding = WeightEncoding::Quantized(QuantizationSpec {
        format_id: id("quantization.separate-scale"),
        bits_per_weight: 4,
        group_size: 256,
        packing: QuantizationPacking::Linear,
        scale_type: ElementType::F16,
        zero_point_type: None,
    });
    assert!(separate_scale_encoding
        .validate(&id("family.gguf-wrong-encoding"))
        .is_err());

    let mut padded = q4_k_block_schema(vec![2, 300]);
    padded.tensors[0].physical_layout = PhysicalWeightLayout::BlockQuantized {
        blocks: exact_component("component.q4-k.blocks"),
        block_axis: 1,
        block_padding: PhysicalWeightPadding::ZeroFill {
            padded_dimensions: vec![2, 512],
        },
    };
    padded.validate(&id("family.gguf-minimal-padding")).unwrap();
    assert_eq!(padded.physical_bytes(&id("weight.attn-q")).unwrap(), 576);

    let mut non_minimal_padding = padded;
    let PhysicalWeightLayout::BlockQuantized { block_padding, .. } =
        &mut non_minimal_padding.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *block_padding = PhysicalWeightPadding::ZeroFill {
        padded_dimensions: vec![2, 768],
    };
    assert!(non_minimal_padding
        .validate(&id("family.gguf-non-minimal-padding"))
        .is_err());
}

fn recursive_quantized_expert_schema() -> WeightSchema {
    let quantization = QuantizationSpec {
        format_id: id("quantization.expert-grouped"),
        bits_per_weight: 4,
        group_size: 4,
        packing: QuantizationPacking::Tiled,
        scale_type: ElementType::F16,
        zero_point_type: Some(ElementType::U8),
    };
    let mut components = Vec::new();
    let mut experts = Vec::new();
    for expert in 0..2 {
        let packed = format!("component.expert.{expert}.packed");
        let scales = format!("component.expert.{expert}.scales");
        let zero_points = format!("component.expert.{expert}.zeros");
        components.extend([
            WeightComponentSpec {
                id: id(&packed),
                role: WeightComponentRole::PackedValues,
                external_names: vec![format!("expert.{expert}.packed.bin")],
                dimensions: vec![4, 8],
                encoding: WeightEncoding::Quantized(quantization.clone()),
                required: true,
            },
            WeightComponentSpec {
                id: id(&scales),
                role: WeightComponentRole::Scales,
                external_names: vec![format!("expert.{expert}.scales.bin")],
                dimensions: vec![8, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            },
            WeightComponentSpec {
                id: id(&zero_points),
                role: WeightComponentRole::ZeroPoints,
                external_names: vec![format!("expert.{expert}.zeros.bin")],
                dimensions: vec![8, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::U8,
                },
                required: true,
            },
        ]);
        experts.push(PhysicalWeightLayout::Quantized {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(id(packed)),
            packed_dimensions: vec![4, 8],
            scales: PhysicalWeightComponentBinding::exact_contiguous(id(scales)),
            zero_points: Some(PhysicalWeightComponentBinding::exact_contiguous(id(
                zero_points,
            ))),
            axis_indices: None,
            permutation: None,
            codebook: None,
            group_axis: 1,
            group_padding: PhysicalWeightPadding::Exact,
        });
    }
    WeightSchema {
        format_id: id("weight-format.expert-quantized"),
        layout_id: id("weight-layout.recursive-expert-stack"),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![WeightTensorSpec {
            id: id("weight.expert-stack"),
            dimensions: vec![2, 8, 8],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::ExpertStack {
                experts,
                expert_axis: 0,
            },
            required: true,
        }],
    }
}

#[test]
fn physical_weight_layout_tree_accepts_recursive_quantized_expert_stack_fixture() {
    let schema = recursive_quantized_expert_schema();
    schema.validate(&id("family.expert-stack")).unwrap();
    assert_eq!(
        schema
            .physical_component_refs(&id("weight.expert-stack"))
            .unwrap()
            .len(),
        6
    );
    assert_eq!(
        schema.physical_bytes(&id("weight.expert-stack")).unwrap(),
        160
    );
    assert_eq!(
        schema.quantization_formats(),
        BTreeSet::from([id("quantization.expert-grouped")])
    );

    let encoded = serde_json::to_vec(&schema).unwrap();
    let restored: WeightSchema = serde_json::from_slice(&encoded).unwrap();
    restored.validate(&id("family.expert-stack")).unwrap();
    assert_eq!(restored, schema);

    let mut reordered_experts = schema.clone();
    let PhysicalWeightLayout::ExpertStack { experts, .. } =
        &mut reordered_experts.tensors[0].physical_layout
    else {
        unreachable!();
    };
    experts.reverse();
    reordered_experts
        .validate(&id("family.expert-stack-reordered"))
        .unwrap();
    assert_ne!(serde_json::to_vec(&reordered_experts).unwrap(), encoded);
}

#[test]
fn weight_schema_order_is_normalized_before_fingerprinting() {
    let canonical = TypedFamilyRegistration::new(OrderedSchemaFamily {
        reverse: false,
        reverse_sources: false,
    })
    .prepare(&json!({"width": 4}))
    .unwrap();
    let reversed = TypedFamilyRegistration::new(OrderedSchemaFamily {
        reverse: true,
        reverse_sources: false,
    })
    .prepare(&json!({"width": 4}))
    .unwrap();
    assert_eq!(canonical.weight_schema(), reversed.weight_schema());
    assert_eq!(
        canonical.fingerprint().unwrap(),
        reversed.fingerprint().unwrap()
    );
    assert_eq!(
        canonical.weight_schema().components[0].external_names,
        ["weight.a", "weight.z"]
    );
    let PhysicalWeightLayout::Composite { parts } = &canonical
        .weight_schema()
        .tensor(&id("weight.optional"))
        .unwrap()
        .physical_layout
    else {
        panic!("optional fixture must use a composite tree");
    };
    assert_eq!(parts[0].logical_offsets, [0]);
    assert_eq!(parts[1].logical_offsets, [2]);

    let source_reversed = TypedFamilyRegistration::new(OrderedSchemaFamily {
        reverse: false,
        reverse_sources: true,
    })
    .prepare(&json!({"width": 4}))
    .unwrap();
    assert_ne!(canonical.weight_schema(), source_reversed.weight_schema());
    assert_ne!(
        canonical.fingerprint().unwrap(),
        source_reversed.fingerprint().unwrap()
    );
    assert_eq!(
        source_reversed.weight_schema().components[0].external_names,
        ["weight.z", "weight.a"]
    );
}

fn blocked_schema(
    logical_shape: Vec<u64>,
    raw_storage_shape: Vec<u64>,
    tile_shape: Vec<u64>,
    axis_order: Vec<u32>,
    tile_strides_in_elements: Vec<u64>,
    padding: PhysicalWeightPadding,
) -> WeightSchema {
    WeightSchema {
        format_id: id("weight-format.blocked"),
        layout_id: id("weight-layout.blocked"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("weight.component.blocked"),
            role: WeightComponentRole::Values,
            external_names: vec!["blocked.bin".to_owned()],
            dimensions: raw_storage_shape,
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.blocked"),
            dimensions: logical_shape,
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Stored {
                component: PhysicalWeightComponentBinding {
                    component_id: id("weight.component.blocked"),
                    storage: PhysicalStorageLayout::Tiled {
                        tile_shape,
                        axis_order,
                        tile_strides_in_elements,
                        padding,
                    },
                },
            },
            required: true,
        }],
    }
}

#[test]
fn blocked_weight_layout_requires_explicit_exact_or_zero_fill_padding() {
    let exact = blocked_schema(
        vec![4, 6],
        vec![6, 4],
        vec![2, 3],
        vec![1, 0],
        vec![12, 6],
        PhysicalWeightPadding::Exact,
    );
    exact.validate(&id("family.blocked")).unwrap();

    let zero_filled = blocked_schema(
        vec![5, 6],
        vec![8, 8],
        vec![4, 4],
        vec![1, 0],
        vec![32, 16],
        PhysicalWeightPadding::ZeroFill {
            padded_dimensions: vec![8, 8],
        },
    );
    zero_filled.validate(&id("family.blocked")).unwrap();

    let implicit_padding = blocked_schema(
        vec![5, 6],
        vec![6, 5],
        vec![4, 4],
        vec![1, 0],
        vec![32, 16],
        PhysicalWeightPadding::Exact,
    );
    assert!(implicit_padding.validate(&id("family.blocked")).is_err());

    let unnecessary_zero_fill = blocked_schema(
        vec![4, 8],
        vec![8, 4],
        vec![4, 4],
        vec![1, 0],
        vec![16, 16],
        PhysicalWeightPadding::ZeroFill {
            padded_dimensions: vec![4, 8],
        },
    );
    assert!(unnecessary_zero_fill
        .validate(&id("family.blocked"))
        .is_err());

    let wrong_padded_shape = blocked_schema(
        vec![5, 6],
        vec![64],
        vec![4, 4],
        vec![1, 0],
        vec![32, 16],
        PhysicalWeightPadding::ZeroFill {
            padded_dimensions: vec![8, 12],
        },
    );
    assert!(wrong_padded_shape.validate(&id("family.blocked")).is_err());
}

#[test]
fn physical_weight_layout_tree_rejects_invalid_shape_reuse_padding_overflow_and_limits() {
    let mut wrong_axis_shape = grouped_quantized_axis_index_schema();
    wrong_axis_shape
        .components
        .iter_mut()
        .find(|component| component.id.as_str() == "component.axis-indices")
        .unwrap()
        .dimensions = vec![8, 2];
    assert!(wrong_axis_shape
        .validate(&id("family.wrong-axis-shape"))
        .is_err());

    let mut same_bytes_wrong_packed_shape = grouped_quantized_axis_index_schema();
    let PhysicalWeightLayout::Quantized {
        packed_dimensions, ..
    } = &mut same_bytes_wrong_packed_shape.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *packed_dimensions = vec![2, 16];
    assert!(same_bytes_wrong_packed_shape
        .validate(&id("family.wrong-packed-shape"))
        .is_err());

    let mut reused_component = recursive_quantized_expert_schema();
    let PhysicalWeightLayout::ExpertStack { experts, .. } =
        &mut reused_component.tensors[0].physical_layout
    else {
        unreachable!();
    };
    experts[1] = experts[0].clone();
    let error = reused_component
        .validate(&id("family.reused-component"))
        .unwrap_err();
    assert!(error.to_string().contains("referenced more than once"));

    let strided = WeightSchema {
        format_id: id("weight-format.strided"),
        layout_id: id("weight-layout.strided"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("component.strided"),
            role: WeightComponentRole::Values,
            external_names: vec!["strided.bin".to_owned()],
            dimensions: vec![8],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.strided"),
            dimensions: vec![2, 3],
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Stored {
                component: PhysicalWeightComponentBinding {
                    component_id: id("component.strided"),
                    storage: PhysicalStorageLayout::Strided {
                        strides_in_elements: vec![4, 1],
                        padding: PhysicalWeightPadding::ZeroFill {
                            padded_dimensions: vec![2, 4],
                        },
                    },
                },
            },
            required: true,
        }],
    };
    strided.validate(&id("family.strided")).unwrap();
    let mut overlapping_stride = strided.clone();
    let PhysicalWeightLayout::Stored { component } =
        &mut overlapping_stride.tensors[0].physical_layout
    else {
        unreachable!();
    };
    let PhysicalStorageLayout::Strided {
        strides_in_elements,
        ..
    } = &mut component.storage
    else {
        unreachable!();
    };
    *strides_in_elements = vec![1, 1];
    assert!(overlapping_stride
        .validate(&id("family.overlapping-stride"))
        .is_err());

    let mut overflowing = TestFamily.weight_schema(&TestConfig { width: 4 }).unwrap();
    overflowing.components[0].dimensions = vec![u64::MAX, 2];
    overflowing.tensors[0].dimensions = vec![u64::MAX, 2];
    assert!(overflowing.validate(&id("family.overflowing")).is_err());

    let mut too_deep = TestFamily.weight_schema(&TestConfig { width: 4 }).unwrap();
    let mut nested = too_deep.tensors[0].physical_layout.clone();
    for _ in 0..MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH {
        nested = PhysicalWeightLayout::Composite {
            parts: vec![CompositeWeightPart {
                layout: Box::new(nested),
                logical_offsets: vec![0],
                extents: vec![4],
            }],
        };
    }
    too_deep.tensors[0].physical_layout = nested;
    assert!(too_deep.validate(&id("family.too-deep")).is_err());
    assert!(too_deep
        .physical_component_refs(&id("weight.matrix"))
        .is_err());
    assert!(
        TypedFamilyRegistration::new(FixedSchemaFamily { schema: too_deep })
            .prepare(&json!({"width": 4}))
            .is_err()
    );

    let expert_count = MAX_PHYSICAL_WEIGHT_LAYOUT_NODES / 2 + 1;
    let mut components = Vec::with_capacity(expert_count);
    let mut experts = Vec::with_capacity(expert_count);
    for index in 0..expert_count {
        let component_id = format!("component.node-limit.{index:04}");
        components.push(WeightComponentSpec {
            id: id(component_id.clone()),
            role: WeightComponentRole::Values,
            external_names: vec![format!("node-limit.{index:04}.bin")],
            dimensions: vec![1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        experts.push(PhysicalWeightLayout::Dense {
            component_id: id(component_id),
        });
    }
    let too_many_nodes = WeightSchema {
        format_id: id("weight-format.node-limit"),
        layout_id: id("weight-layout.node-limit"),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![WeightTensorSpec {
            id: id("weight.node-limit"),
            dimensions: vec![expert_count as u64, 1],
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::ExpertStack {
                experts,
                expert_axis: 0,
            },
            required: true,
        }],
    };
    assert!(too_many_nodes
        .validate(&id("family.too-many-nodes"))
        .is_err());
    assert!(too_many_nodes
        .physical_component_refs(&id("weight.node-limit"))
        .is_err());
}

#[test]
fn blocked_tensor_storage_requires_explicit_exact_or_zero_fill_padding() {
    let exact = ResolvedTensorSpec::new(
        vec![4],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4],
            axis_order: vec![0],
            padding: BlockedTensorPadding::Exact,
        },
    )
    .unwrap();
    assert_eq!(exact.minimum_storage_bytes().unwrap(), 16);

    assert!(ResolvedTensorSpec::new(
        vec![3],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4],
            axis_order: vec![0],
            padding: BlockedTensorPadding::Exact,
        },
    )
    .is_err());

    let zero_filled = ResolvedTensorSpec::new(
        vec![3],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4],
            axis_order: vec![0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![4],
            },
        },
    )
    .unwrap();
    assert_eq!(zero_filled.minimum_storage_bytes().unwrap(), 16);

    for physical_dimensions in [vec![3], vec![8]] {
        assert!(ResolvedTensorSpec::new(
            vec![3],
            ElementType::F32,
            ResolvedTensorLayout::Blocked {
                block: vec![4],
                axis_order: vec![0],
                padding: BlockedTensorPadding::ZeroFill {
                    physical_dimensions,
                },
            },
        )
        .is_err());
    }

    let transposed = ResolvedTensorSpec::new(
        vec![3, 5],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4, 4],
            axis_order: vec![1, 0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![8, 4],
            },
        },
    )
    .unwrap();
    assert_eq!(transposed.minimum_storage_bytes().unwrap(), 128);
    assert!(ResolvedTensorSpec::new(
        vec![3, 5],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4, 4],
            axis_order: vec![0, 0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![4, 8],
            },
        },
    )
    .is_err());
}

#[test]
fn model_program_rejects_duplicate_declared_outputs() {
    let family = TestRegistry::new().prepare();
    let mut value = serde_json::to_value(family.program()).unwrap();
    value["outputs"] = json!(["value.output", "value.output"]);
    assert!(serde_json::from_value::<ModelProgram>(value).is_err());
}
