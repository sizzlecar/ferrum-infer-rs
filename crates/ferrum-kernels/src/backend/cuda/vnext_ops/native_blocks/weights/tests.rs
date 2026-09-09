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
    let format = GgufBlockFormat::Iq4Xs;
    WeightSchema {
        format_id: WeightFormatId::new("weight-format.gguf.native-block").unwrap(),
        layout_id: WeightLayoutId::new("weight-layout.test.native-matrix").unwrap(),
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: id("component.z_first"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["first".into()],
                dimensions: vec![1, 3, 2],
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
                dimensions: vec![1, 3, 512],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: id("weight.matrix"),
            dimensions: vec![2, 3, 512],
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
                        extents: vec![1, 3, 512],
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
                        extents: vec![1, 3, 512],
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
                component_id: id("component.z_first"),
                format: MatrixFormat::Block(GgufBlockFormat::Iq4Xs),
                rows: 3,
                columns: 512,
                output_offset: 0
            },
            MatrixPart {
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
fn native_matrix_rejects_strided_storage_and_wrong_quantized_axis() {
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
        strides_in_elements: vec![1536, 512, 1],
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
