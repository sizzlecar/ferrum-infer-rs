use super::*;
use ferrum_interfaces::vnext::*;

fn projection(
    ordinal: u32,
    dtype: ElementType,
    block: bool,
    strided: bool,
) -> ResolvedValueBinding {
    let component = WeightId::new(format!("component.projection.{ordinal}")).unwrap();
    let weight = WeightId::new(format!("weight.projection.{ordinal}")).unwrap();
    let encoding = if block {
        WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: QuantizationFormatId::new("quantization.gguf.q8-0").unwrap(),
            logical_values_per_block: 32,
            bytes_per_block: 34,
        })
    } else {
        WeightEncoding::Dense {
            element_type: dtype,
        }
    };
    let component_spec = WeightComponentSpec {
        id: component.clone(),
        role: if block {
            WeightComponentRole::PackedValues
        } else {
            WeightComponentRole::Values
        },
        external_names: vec![format!("projection.{ordinal}")],
        dimensions: if block { vec![32, 1] } else { vec![32, 32] },
        encoding,
        required: true,
    };
    let physical_bytes = component_spec.physical_bytes().unwrap();
    let schema = WeightSchema {
        format_id: WeightFormatId::new(
            crate::gguf_f16_projection_materializer::GGUF_F16_PROJECTION_FORMAT_ID,
        )
        .unwrap(),
        layout_id: WeightLayoutId::new("layout.rn-f16.fixture").unwrap(),
        version: ContractVersion::new(1, 0),
        components: vec![component_spec],
        tensors: vec![WeightTensorSpec {
            id: weight.clone(),
            dimensions: vec![32, 32],
            logical_element_type: dtype,
            physical_layout: if block {
                PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding {
                        component_id: component.clone(),
                        storage: PhysicalStorageLayout::exact_contiguous(),
                    },
                    block_axis: 1,
                    block_padding: PhysicalWeightPadding::Exact,
                }
            } else {
                PhysicalWeightLayout::Dense {
                    component_id: component.clone(),
                }
            },
            required: true,
        }],
    };
    schema
        .validate(&ModelFamilyId::new("family.rn-f16.fixture").unwrap())
        .unwrap();
    ResolvedValueBinding::new(
        ProgramValueId::new(format!("value.projection.{ordinal}")).unwrap(),
        ResolvedValueRole::Input,
        ordinal,
        ResolvedTensorSpec::new(
            vec![32, 32],
            dtype,
            if strided {
                ResolvedTensorLayout::Strided {
                    byte_strides: vec![2, 64],
                }
            } else {
                ResolvedTensorLayout::Contiguous
            },
        )
        .unwrap(),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(ResolvedWeightBinding::from_schema(&schema, &weight).unwrap()),
        ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
            Some(component),
            ResourceId::new(format!("resource.projection.{ordinal}")).unwrap(),
            128,
            physical_bytes,
            if block { ElementType::U8 } else { dtype },
        )
        .unwrap()])
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn gguf_rn_f16_cuda_consumers_require_every_full_dense_f16_projection() {
    for name in [
        DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
    ] {
        let operation = OperationId::new(name).unwrap();
        let ordinals = (0..=7)
            .filter(|&i| gguf_f16_projection_role_v1(&operation, i).is_some())
            .collect::<Vec<_>>();
        let valid = ordinals
            .iter()
            .map(|&i| projection(i, ElementType::F16, false, false))
            .collect::<Vec<_>>();
        validate_values(&operation, &valid).unwrap();
        for (index, &ordinal) in ordinals.iter().enumerate() {
            let mut missing = valid.clone();
            missing.remove(index);
            assert!(validate_values(&operation, &missing).is_err());
            for replacement in [
                projection(ordinal, ElementType::F32, false, false),
                projection(ordinal, ElementType::F16, true, false),
                projection(ordinal, ElementType::F16, false, true),
            ] {
                let mut invalid = valid.clone();
                invalid[index] = replacement;
                assert!(
                    validate_values(&operation, &invalid).is_err(),
                    "{name} input {ordinal}"
                );
            }
        }
    }
}

#[test]
fn gguf_rn_f16_cuda_mixed_container_preserves_retained_native_consumers() {
    let native = WeightFormatId::new("weight-format.gguf.native-block").unwrap();
    let mixed =
        WeightFormatId::new(crate::gguf_f16_projection_materializer::GGUF_F16_PROJECTION_FORMAT_ID)
            .unwrap();
    let head = OperationId::new(LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID).unwrap();
    let formats = provider_formats(&head, BTreeSet::from([native.clone()])).unwrap();
    assert!(formats.contains(&native) && formats.contains(&mixed));
    let rounded = OperationId::new(DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID).unwrap();
    assert_eq!(
        provider_formats(&rounded, formats).unwrap(),
        BTreeSet::from([mixed])
    );
    // Existing operation physical policy remains with its original provider.
    let strict = OperationId::new(DENSE_SWIGLU_OPERATION_ID).unwrap();
    assert!(!is_operation(&strict));
    validate_values(&strict, &[projection(1, ElementType::F16, true, false)]).unwrap();
}
