use super::*;
use ferrum_interfaces::vnext::{
    BlockQuantizationSpec, ContractVersion, HadamardTransformSpec, PhysicalWeightComponentBinding,
    ResolvedWeightBinding, WeightComponentRole, WeightComponentSpec, WeightSchema,
    WeightTensorSpec,
};
use std::num::NonZeroU32;

fn id<T>(value: &str) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(value.to_owned()).unwrap()
}

fn schema(transformed: bool, quantized: bool) -> WeightSchema {
    let table: ferrum_interfaces::vnext::WeightId = id("component.z-table");
    let signs: ferrum_interfaces::vnext::WeightId = id("component.a-signs");
    let mut components = vec![WeightComponentSpec {
        id: table.clone(),
        role: if quantized {
            WeightComponentRole::PackedValues
        } else {
            WeightComponentRole::Values
        },
        external_names: vec!["embedding".into()],
        dimensions: if quantized { vec![2, 1] } else { vec![2, 256] },
        encoding: if quantized {
            WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id(Q4_K_FORMAT_ID),
                logical_values_per_block: 256,
                bytes_per_block: 144,
            })
        } else {
            WeightEncoding::Dense {
                element_type: ElementType::F16,
            }
        },
        required: true,
    }];
    let mut layout = if quantized {
        PhysicalWeightLayout::BlockQuantized {
            blocks: PhysicalWeightComponentBinding::exact_contiguous(table),
            block_axis: 1,
            block_padding: PhysicalWeightPadding::Exact,
        }
    } else {
        PhysicalWeightLayout::Dense {
            component_id: table,
        }
    };
    if transformed {
        components.push(WeightComponentSpec {
            id: signs.clone(),
            role: WeightComponentRole::TransformSigns,
            external_names: vec!["signs".into()],
            dimensions: vec![256],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        layout = PhysicalWeightLayout::Hadamard {
            values: Box::new(layout),
            transform: HadamardTransformSpec {
                block_size: NonZeroU32::new(256).unwrap(),
                signs: HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
                    signs,
                )),
                application: HadamardApplication::AfterEmbeddingLookup,
            },
        };
    }
    WeightSchema {
        format_id: id("weight-format.gguf.native-block"),
        layout_id: id("weight-layout.primitive-cost"),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![WeightTensorSpec {
            id: id("weight.embedding"),
            dimensions: vec![2, 256],
            logical_element_type: ElementType::F16,
            physical_layout: layout,
            required: true,
        }],
    }
}

fn resolved(schema: &WeightSchema) -> ResolvedWeightBinding {
    ResolvedWeightBinding::from_schema(schema, &id("weight.embedding")).unwrap()
}

#[test]
fn static_embedding_abi_uses_actual_component_order_and_transform_direction() {
    for quantized in [false, true] {
        let plain = schema(false, quantized);
        let (format, at, transform) = embedding_weight_layout(&resolved(&plain), 2, 256).unwrap();
        assert_eq!(
            format,
            if quantized {
                EmbeddingPhysicalFormat::Q4K
            } else {
                EmbeddingPhysicalFormat::DenseF16
            }
        );
        assert_eq!(at, 0);
        assert!(transform.is_none());
        let mut inverse = schema(true, quantized);
        let (_, at, transform) = embedding_weight_layout(&resolved(&inverse), 2, 256).unwrap();
        assert_eq!(at, 1, "signs sort before the table");
        let transform = transform.unwrap();
        assert_eq!(transform.signs_region, Some(0));
        assert!(transform.inverse);
        let PhysicalWeightLayout::Hadamard { transform, .. } =
            &mut inverse.tensors[0].physical_layout
        else {
            unreachable!()
        };
        transform.application = HadamardApplication::BeforeMatmul {
            input_permutation: None,
        };
        assert!(embedding_weight_layout(&resolved(&inverse), 2, 256).is_err());
    }
}

#[test]
fn static_embedding_abi_rejects_valid_but_unsupported_native_formats_and_shapes() {
    let mut unsupported = schema(false, true);
    let WeightEncoding::BlockQuantized(spec) = &mut unsupported.components[0].encoding else {
        unreachable!()
    };
    spec.format_id = id("quantization.fixture.unsupported");
    assert!(embedding_weight_layout(&resolved(&unsupported), 2, 256).is_err());
    let valid = resolved(&schema(false, true));
    assert!(embedding_weight_layout(&valid, 2, 255).is_err());
    assert!(embedding_weight_layout(&valid, 3, 256).is_err());
    let mut wrong_axis = schema(false, true);
    // A valid row-axis block ABI is outside this embedding implementation.
    wrong_axis.tensors[0].dimensions = vec![256, 2];
    wrong_axis.components[0].dimensions = vec![1, 2];
    let PhysicalWeightLayout::BlockQuantized { block_axis, .. } =
        &mut wrong_axis.tensors[0].physical_layout
    else {
        unreachable!()
    };
    *block_axis = 0;
    assert!(embedding_weight_layout(&resolved(&wrong_axis), 256, 2).is_err());
}

#[test]
fn shared_launch_parameters_reject_real_kernel_integer_boundaries() {
    let maximum = u64::from(u32::MAX);
    assert_eq!(
        embedding_params(maximum, 256, 32).unwrap().token_count,
        u32::MAX
    );
    assert!(embedding_params(maximum + 1, 256, 32).is_err());
    assert!(embedding_params(1, maximum + 1, 32).is_err());
    assert!(embedding_params(1, 256, maximum + 1).is_err());
    assert!(rms_norm_params(maximum + 1, 256, 1e-6).is_err());
    assert!(rms_norm_params(1, maximum + 1, 1e-6).is_err());
    assert!(residual_add_params(maximum / 256 + 1, 256).is_err());
    assert!(residual_add_params(u64::MAX, 2).is_err());
    assert!(masked_argmax_params(maximum + 1, 1).is_err());
    for zero in [
        embedding_params(0, 1, 1).is_err(),
        rms_norm_params(0, 1, 1e-6).is_err(),
        residual_add_params(1, 0).is_err(),
        masked_argmax_params(0, 1).is_err(),
    ] {
        assert!(zero);
    }
    assert_eq!(embedding_scratch_bytes(16, 3, 5).unwrap(), 112);
    assert!(embedding_scratch_bytes(u64::MAX, 1, 256).is_err());
    assert!(embedding_scratch_bytes(0, u64::MAX, 256).is_err());
}

#[test]
fn explicit_row_major_stored_and_blocks_match_contiguous_embedding_abi() {
    for quantized in [false, true] {
        let contiguous = schema(true, quantized);
        let expected = embedding_weight_layout(&resolved(&contiguous), 2, 256).unwrap();
        let mut strided = contiguous.clone();
        let PhysicalWeightLayout::Hadamard { values, transform } =
            &mut strided.tensors[0].physical_layout
        else {
            unreachable!()
        };
        let storage = |strides| PhysicalStorageLayout::Strided {
            strides_in_elements: strides,
            padding: PhysicalWeightPadding::Exact,
        };
        match values.as_mut() {
            PhysicalWeightLayout::Dense { component_id } => {
                **values = PhysicalWeightLayout::Stored {
                    component: PhysicalWeightComponentBinding {
                        component_id: component_id.clone(),
                        storage: storage(vec![256, 1]),
                    },
                };
            }
            PhysicalWeightLayout::BlockQuantized { blocks, .. } => {
                blocks.storage = storage(vec![1, 1]); // Physical blocks [2, 1].
            }
            _ => unreachable!(),
        }
        let HadamardSigns::Explicit(signs) = &mut transform.signs else {
            unreachable!()
        };
        // Signs are a narrower immutable auxiliary ABI: schema requires
        // exact-contiguous storage even when explicit row-major values/blocks
        // are legal. Do not infer sign-layout support from equal byte span.
        assert_eq!(signs.storage, PhysicalStorageLayout::exact_contiguous());
        let actual = embedding_weight_layout(&resolved(&strided), 2, 256).unwrap();
        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);
        assert_eq!(
            actual.2.unwrap().signs_region,
            expected.2.unwrap().signs_region
        );
        assert!(actual.2.unwrap().inverse);
    }
    // This is valid dense storage, but the future flat shader ABI has no
    // transposition contract. Do not declare it equivalent from byte size.
    let mut transposed = schema(false, false);
    transposed.tensors[0].physical_layout = PhysicalWeightLayout::Stored {
        component: PhysicalWeightComponentBinding {
            component_id: transposed.components[0].id.clone(),
            storage: PhysicalStorageLayout::Strided {
                strides_in_elements: vec![1, 2],
                padding: PhysicalWeightPadding::Exact,
            },
        },
    };
    assert!(embedding_weight_layout(&resolved(&transposed), 2, 256).is_err());
}

#[test]
fn hadamard_signs_reject_explicit_row_major_strided_storage() {
    for quantized in [false, true] {
        let mut invalid = schema(true, quantized);
        let PhysicalWeightLayout::Hadamard { transform, .. } =
            &mut invalid.tensors[0].physical_layout
        else {
            unreachable!()
        };
        let HadamardSigns::Explicit(signs) = &mut transform.signs else {
            unreachable!()
        };
        signs.storage = PhysicalStorageLayout::Strided {
            strides_in_elements: vec![1],
            padding: PhysicalWeightPadding::Exact,
        };
        // Validation must reject the declaration before either actual encoding
        // or future route interpretation; no synthetic resolved ABI is forged.
        assert!(ResolvedWeightBinding::from_schema(&invalid, &id("weight.embedding")).is_err());
    }
}
