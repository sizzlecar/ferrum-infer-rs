use super::*;
use ferrum_interfaces::vnext::{
    BlockQuantizationSpec, CompositeWeightPart, ContractVersion, PhysicalWeightComponentBinding,
    QuantizationFormatId, WeightComponentRole, WeightComponentSpec, WeightFormatId, WeightLayoutId,
    WeightSchema, WeightTensorSpec,
};

fn id(value: &str) -> WeightId {
    WeightId::new(value).unwrap()
}

fn schema() -> WeightSchema {
    schema_with_block_format(GgufBlockFormat::Iq4Xs, 512)
}

fn schema_with_block_format(format: GgufBlockFormat, columns: u64) -> WeightSchema {
    WeightSchema {
        format_id: WeightFormatId::new("weight-format.gguf.native-block").unwrap(),
        layout_id: WeightLayoutId::new("weight-layout.test.native-matrix").unwrap(),
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: id("component.z_first"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["first".into()],
                dimensions: vec![1, 3, columns / format.block_values() as u64],
                encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: QuantizationFormatId::new(format.format_id()).unwrap(),
                    logical_values_per_block: format.block_values() as u32,
                    bytes_per_block: format.block_bytes() as u32,
                }),
                required: true,
            },
            WeightComponentSpec {
                id: id("component.a_second"),
                role: WeightComponentRole::Values,
                external_names: vec!["second".into()],
                dimensions: vec![1, 3, columns],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: id("weight.matrix"),
            dimensions: vec![2, 3, columns],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Composite {
                parts: vec![
                    CompositeWeightPart {
                        layout: Box::new(PhysicalWeightLayout::Stored {
                            component: PhysicalWeightComponentBinding::exact_contiguous(id(
                                "component.a_second",
                            )),
                        }),
                        logical_offsets: vec![1, 0, 0],
                        extents: vec![1, 3, columns],
                    },
                    CompositeWeightPart {
                        layout: Box::new(PhysicalWeightLayout::BlockQuantized {
                            blocks: PhysicalWeightComponentBinding::exact_contiguous(id(
                                "component.z_first",
                            )),
                            block_axis: 2,
                            block_padding: PhysicalWeightPadding::Exact,
                        }),
                        logical_offsets: vec![0, 0, 0],
                        extents: vec![1, 3, columns],
                    },
                ],
            },
            required: true,
        }],
    }
}

fn parts(schema: &WeightSchema) -> Result<Vec<MatrixPart>, String> {
    let weight = ResolvedWeightBinding::from_schema(schema, &schema.tensors[0].id)
        .map_err(|error| error.to_string())?;
    matrix_parts(&weight, &schema.tensors[0].dimensions)
}

#[test]
fn mixed_composite_maps_component_identity_to_complete_logical_rows() {
    let parts = parts(&schema()).unwrap();
    assert_eq!(
        parts,
        vec![
            MatrixPart {
                transform: None,
                signs_region: None,
                component_id: id("component.z_first"),
                format: MatrixFormat::Block(GgufBlockFormat::Iq4Xs),
                rows: 3,
                columns: 512,
                output_offset: 0
            },
            MatrixPart {
                transform: None,
                signs_region: None,
                component_id: id("component.a_second"),
                format: MatrixFormat::DenseF16,
                rows: 3,
                columns: 512,
                output_offset: 3
            },
        ]
    );
}

#[test]
fn hadamard_parts_keep_shared_sign_identity_and_independent_projection_transforms() {
    use ferrum_interfaces::vnext::{
        GroupedFeatureTranspose, HadamardApplication, HadamardSigns, HadamardTransformSpec,
    };
    use std::num::NonZeroU32;
    let mut schema = schema_with_block_format(GgufBlockFormat::Pq2_0, 512);
    schema.components.insert(
        0,
        WeightComponentSpec {
            id: id("component.shared-signs"),
            role: WeightComponentRole::TransformSigns,
            external_names: vec!["metadata.signs.512".into()],
            dimensions: vec![512],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        },
    );
    let transform = HadamardTransformSpec {
        block_size: NonZeroU32::new(256).unwrap(),
        signs: HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(id(
            "component.shared-signs",
        ))),
        application: HadamardApplication::BeforeMatmul {
            input_permutation: None,
        },
    };
    let PhysicalWeightLayout::Composite { parts: children } =
        &mut schema.tensors[0].physical_layout
    else {
        panic!()
    };
    for child in children.iter_mut() {
        child.layout = Box::new(PhysicalWeightLayout::Hadamard {
            values: child.layout.clone(),
            transform: transform.clone(),
        });
    }
    let resolved = parts(&schema).unwrap();
    assert_eq!(region_count(&resolved), 3);
    assert_eq!(resolved[0].signs_region, Some(2));
    assert_eq!(resolved[1].signs_region, Some(2));
    assert_eq!(resolved[0].component_id, id("component.z_first"));
    assert_eq!(dispatches(&resolved), 4);
    let replay = |parts: &[MatrixPart]| {
        key(
            crate::backend::cuda::vnext_replay::CudaCommandReplayKeyBuilder::new(
                "test.hadamard",
                "matrix",
            ),
            parts,
        )
        .finish()
    };
    let original_key = replay(&resolved);
    let mut permuted = resolved.clone();
    permuted[0].transform.as_mut().unwrap().application = HadamardApplication::BeforeMatmul {
        input_permutation: Some(GroupedFeatureTranspose {
            inner_extent: 64,
            first_outer_extent: 2,
            second_outer_extent: 4,
        }),
    };
    assert_ne!(original_key, replay(&permuted));
    let mut plain = resolved.clone();
    plain[1].transform = None;
    plain[1].signs_region = None;
    assert_eq!(dispatches(&plain), 3);
    assert_ne!(original_key, replay(&plain));
    let mut relocated = resolved.clone();
    relocated[0].signs_region = Some(3);
    assert_ne!(original_key, replay(&relocated));
}

#[test]
fn pq2_matrix_rows_resolve_their_128_value_blocks_without_aliasing_q8() {
    for columns in [128, 384] {
        let mut source = schema_with_block_format(GgufBlockFormat::Pq2_0, columns);
        let resolved = parts(&source).unwrap();
        assert_eq!(resolved[0].format.parameters(), [142, 128, 34]);
        assert_eq!(resolved[0].columns, columns as u32);
        assert_eq!(resolved[0].rows, 3);
        assert_eq!(resolved[1].format, MatrixFormat::DenseF16);
        assert_eq!(resolved[1].output_offset, 3);

        // Q8_0 also occupies 34 bytes, but encodes only 32 values. Neither
        // its identity nor its physical row extent can substitute for PQ2_0.
        let WeightEncoding::BlockQuantized(spec) = &mut source.components[0].encoding else {
            unreachable!()
        };
        spec.format_id = QuantizationFormatId::new(GgufBlockFormat::Q8_0.format_id()).unwrap();
        assert!(parts(&source).is_err());
        let WeightEncoding::BlockQuantized(spec) = &mut source.components[0].encoding else {
            unreachable!()
        };
        spec.logical_values_per_block = GgufBlockFormat::Q8_0.block_values() as u32;
        assert!(parts(&source).is_err());
    }
}

#[test]
fn native_matrix_accepts_explicit_contiguous_reshape_in_elements_and_blocks() {
    let mut source = schema();
    let expected = parts(&source).unwrap();
    // The raw file omits the singleton axis retained by the operation schema.
    source.components[0].dimensions = vec![3, 2];
    source.components[1].dimensions = vec![3, 512];
    let PhysicalWeightLayout::Composite { parts: children } =
        &mut source.tensors[0].physical_layout
    else {
        unreachable!()
    };
    let PhysicalWeightLayout::Stored { component } = children[0].layout.as_mut() else {
        unreachable!()
    };
    component.storage = PhysicalStorageLayout::Strided {
        strides_in_elements: vec![1536, 512, 1],
        padding: PhysicalWeightPadding::Exact,
    };
    let PhysicalWeightLayout::BlockQuantized { blocks, .. } = children[1].layout.as_mut() else {
        unreachable!()
    };
    blocks.storage = PhysicalStorageLayout::Strided {
        strides_in_elements: vec![6, 2, 1],
        padding: PhysicalWeightPadding::Exact,
    };
    assert_eq!(parts(&source).unwrap(), expected);
}

#[test]
fn native_matrix_rejects_noncontiguous_storage_and_wrong_quantized_axis() {
    let mut source = schema();
    let PhysicalWeightLayout::Composite { parts: children } =
        &mut source.tensors[0].physical_layout
    else {
        unreachable!()
    };
    let PhysicalWeightLayout::Stored { component } = children[0].layout.as_mut() else {
        unreachable!()
    };
    component.storage = PhysicalStorageLayout::Strided {
        strides_in_elements: vec![1536, 1, 3],
        padding: PhysicalWeightPadding::Exact,
    };
    assert!(parts(&source).is_err());
    let mut source = schema();
    let PhysicalWeightLayout::Composite { parts: children } =
        &mut source.tensors[0].physical_layout
    else {
        unreachable!()
    };
    let PhysicalWeightLayout::BlockQuantized { block_axis, .. } = children[1].layout.as_mut()
    else {
        unreachable!()
    };
    *block_axis = 1;
    assert!(parts(&source).is_err());
}

#[test]
fn native_matrix_storage_checks_stride_units_rank_and_extent() {
    let storage = |strides| PhysicalStorageLayout::Strided {
        strides_in_elements: strides,
        padding: PhysicalWeightPadding::Exact,
    };
    // Singleton strides cannot change any addressed element.
    assert!(exact_storage(&storage(vec![123, 512, 1]), &[1, 3, 512]).is_ok());
    for (strides, shape) in [
        (vec![513, 1], vec![3, 512]),
        (vec![512, 2], vec![3, 512]),
        (vec![512, 1], vec![1, 3, 512]),
        (vec![512, 1], vec![3, 2]), // Blocks are not logical scalar elements.
        (vec![1, 1], vec![0, 1]),
        (vec![2, 1], vec![u64::MAX, 2]),
    ] {
        assert!(exact_storage(&storage(strides), &shape).is_err());
    }
}

#[test]
fn native_matrix_rejects_overlapping_missing_and_partial_input_rows() {
    for (offset, extent) in [
        (vec![0, 0, 0], vec![1, 3, 512]),
        (vec![1, 1, 0], vec![1, 2, 512]),
        (vec![1, 0, 256], vec![1, 3, 256]),
    ] {
        let mut source = schema();
        let PhysicalWeightLayout::Composite { parts: children } =
            &mut source.tensors[0].physical_layout
        else {
            unreachable!()
        };
        children[0].logical_offsets = offset;
        children[0].extents = extent;
        assert!(parts(&source).is_err());
    }
}

#[test]
fn native_matrix_rejects_wrong_physical_shape_and_unknown_block_abi() {
    let mut source = schema();
    source.components[0].dimensions = vec![1, 2, 2];
    assert!(parts(&source).is_err());
    let mut source = schema();
    let WeightEncoding::BlockQuantized(spec) = &mut source.components[0].encoding else {
        unreachable!()
    };
    spec.format_id = QuantizationFormatId::new("quantization.test.unknown").unwrap();
    assert!(parts(&source).is_err());
    let mut source = schema();
    let WeightEncoding::BlockQuantized(spec) = &mut source.components[0].encoding else {
        unreachable!()
    };
    spec.bytes_per_block += 2;
    assert!(parts(&source).is_err());
}
