use super::*;

fn scalar_contract() -> TensorContract {
    TensorContract::new(
        vec![DimensionConstraint::Exact(1)],
        BTreeSet::from([ElementType::F32]),
        vec![LayoutConstraint::Contiguous],
        TensorAccess::Read,
        AliasPolicy::NoAlias,
    )
    .unwrap()
}
fn either() -> TensorContract {
    scalar_contract()
        .with_numerical_or_physical_weight_component(
            WeightComponentRole::Metadata,
            WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
        )
        .unwrap()
}
fn scalar_schema(
    physical: bool,
    role: WeightComponentRole,
    dtype: ElementType,
    elements: u64,
) -> WeightSchema {
    let mut s = schema();
    s.components[0].role = role;
    s.components[0].dimensions = if physical && elements == 1 {
        vec![]
    } else {
        vec![elements]
    };
    s.components[0].encoding = WeightEncoding::Dense {
        element_type: dtype,
    };
    s.tensors[0].dimensions = vec![elements];
    s.tensors[0].logical_element_type = dtype;
    if !physical {
        s.tensors[0].physical_layout = PhysicalWeightLayout::Dense {
            component_id: s.components[0].id.clone(),
        };
    }
    s.validate(&id("family.scalar")).unwrap();
    s
}
fn scalar_binding(s: &WeightSchema) -> ResolvedValueBinding {
    let c = &s.components[0];
    let t = &s.tensors[0];
    ResolvedValueBinding::new(
        id("value.scalar"),
        ResolvedValueRole::Input,
        0,
        ResolvedTensorSpec::new(
            t.dimensions.clone(),
            t.logical_element_type,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(ResolvedWeightBinding::from_schema(s, &t.id).unwrap()),
        ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
            Some(c.id.clone()),
            id("resource.scalar"),
            0,
            c.physical_bytes().unwrap(),
            c.physical_element_type(),
        )
        .unwrap()])
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn union_admits_numerical_constant_and_exact_physical_scale_but_not_other_interpretations() {
    let numerical = scalar_binding(&scalar_schema(
        false,
        WeightComponentRole::Values,
        ElementType::F32,
        1,
    ));
    let physical = scalar_binding(&scalar_schema(
        true,
        WeightComponentRole::Metadata,
        ElementType::F32,
        1,
    ));
    for value in [numerical.clone(), physical.clone()] {
        validate_input(either(), value).unwrap();
    }
    assert!(validate_input(scalar_contract(), physical.clone()).is_err());
    let exclusive = scalar_contract()
        .with_physical_weight_component(
            WeightComponentRole::Metadata,
            WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
        )
        .unwrap();
    assert!(validate_input(exclusive, numerical.clone()).is_err());
    for (physical, role, dtype, elements) in [
        (true, WeightComponentRole::Scales, ElementType::F32, 1),
        (true, WeightComponentRole::Metadata, ElementType::Bf16, 1),
        (true, WeightComponentRole::Metadata, ElementType::F32, 2),
        (false, WeightComponentRole::Values, ElementType::Bf16, 1),
        (false, WeightComponentRole::Values, ElementType::F32, 2),
    ] {
        assert!(validate_input(
            either(),
            scalar_binding(&scalar_schema(physical, role, dtype, elements))
        )
        .is_err());
    }
    // Existing packed ports remain exclusive: a same-shaped numeric byte
    // tensor cannot satisfy a PackedValues/Quantized physical contract.
    let b = nvfp4_matmul_bf16_contract().unwrap();
    assert!(matches!(
        b.descriptor().inputs[1].stored_weight_interpretation(),
        StoredWeightInterpretation::PhysicalComponent { .. }
    ));
    assert!(matches!(
        b.descriptor().inputs[3].stored_weight_interpretation(),
        StoredWeightInterpretation::PhysicalComponent { .. }
    ));
    let quant = nvfp4_quantize_bf16_contract().unwrap();
    assert_eq!(
        quant.descriptor().inputs[1].stored_weight_interpretation(),
        either().stored_weight_interpretation()
    );
}

#[test]
fn union_wire_reconstruction_preserves_both_branches_and_rejects_mutation_or_encoding_conversion() {
    let original = either();
    let wire = serde_json::to_value(&original).unwrap();
    let roundtrip: TensorContract = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(original, roundtrip);
    for physical in [false, true] {
        let s = scalar_schema(
            physical,
            if physical {
                WeightComponentRole::Metadata
            } else {
                WeightComponentRole::Values
            },
            ElementType::F32,
            1,
        );
        let original = scalar_binding(&s);
        let value: ResolvedValueBinding =
            serde_json::from_value(serde_json::to_value(original).unwrap()).unwrap();
        validate_input(roundtrip.clone(), value).unwrap();
    }
    for (field, value) in [
        ("access", serde_json::json!("write")),
        ("alias", serde_json::json!({"may_alias":{"tensor_index":0}})),
        ("element_types", serde_json::json!(["bf16"])),
        (
            "layouts",
            serde_json::json!([{ "strided":{"strides":[{"exact_bytes":4}]}}]),
        ),
    ] {
        let mut changed = wire.clone();
        changed[field] = value;
        assert!(serde_json::from_value::<TensorContract>(changed).is_err());
    }
    let mut changed = wire.clone();
    changed["stored_weight_interpretation"]["numerical_or_physical_component"]["unchecked"] =
        true.into();
    assert!(serde_json::from_value::<TensorContract>(changed).is_err());
    let mut changed_encoding = QuantizationSpec::nvfp4_e2m1_e4m3_128x4();
    changed_encoding.packing = QuantizationPacking::Linear;
    let wrong_encoding = contract()
        .with_numerical_or_physical_weight_component(
            WeightComponentRole::PackedValues,
            WeightEncoding::Quantized(changed_encoding),
        )
        .unwrap();
    assert!(validate_input(wrong_encoding, binding(&schema())).is_err());
    // Equal physical dtype alone is insufficient: DenseAffine requires an
    // actual conversion and cannot enter the physical branch via wire data.
    assert!(scalar_contract()
        .with_numerical_or_physical_weight_component(
            WeightComponentRole::Metadata,
            WeightEncoding::DenseAffine {
                element_type: ElementType::F32,
                scale: CanonicalRational::new(1, 1).unwrap(),
                bias: CanonicalRational::new(0, 1).unwrap()
            }
        )
        .is_err());
}
