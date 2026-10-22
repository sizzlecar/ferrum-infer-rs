use super::*;
fn t(d: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        d.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("padding").unwrap(),
            SemanticValue::Integers(vec![1, 0, 2, 1]),
        ),
        (
            AttributeId::new("stride").unwrap(),
            SemanticValue::Integers(vec![2, 2]),
        ),
        (
            AttributeId::new("dilation").unwrap(),
            SemanticValue::Integers(vec![2, 2]),
        ),
        (
            AttributeId::new("groups").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("padding_mode").unwrap(),
            SemanticValue::Text("explicit_zero".into()),
        ),
    ])
}
#[test]
fn grouped_asymmetric_dilated_geometry_checks_real_extents_and_padding_mode() {
    let (x, w, b, y) = (
        t(&[2, 4, 4, 5]),
        t(&[6, 2, 2, 2]),
        t(&[6]),
        t(&[2, 6, 2, 3]),
    );
    let a = attrs();
    let g = Conv2dBf16Geometry::new(&x, &w, &b, &y, &a).unwrap();
    assert_eq!(g.element_count(), 72);
    assert_eq!(g.logical_arithmetic_operations(), 1152);
    assert_eq!(g.padded_input_bytes(), 640);
    assert_eq!(g.padding_mode(), Conv2dPaddingMode::ExplicitZero);
    let mut implicit = a.clone();
    implicit.insert(
        AttributeId::new("padding_mode").unwrap(),
        SemanticValue::Text("implicit_zero".into()),
    );
    assert_eq!(
        Conv2dBf16Geometry::new(&x, &w, &b, &y, &implicit)
            .unwrap()
            .padding_mode(),
        Conv2dPaddingMode::ImplicitZero
    );
    assert!(Conv2dBf16Geometry::new(&x, &w, &b, &t(&[2, 6, 3, 2]), &a).is_err());
    assert!(Conv2dBf16Geometry::new(&x, &t(&[6, 4, 2, 2]), &b, &y, &a).is_err());
    assert!(Conv2dBf16Geometry::new(&x, &w, &t(&[1, 6]), &y, &a).is_err());
}
#[test]
fn invalid_padding_stride_attributes_and_unrepresentable_work_fail_before_execution() {
    for (name, v) in [
        ("padding", SemanticValue::Integers(vec![-1, 0, 0, 0])),
        ("stride", SemanticValue::Integers(vec![0, 1])),
        ("dilation", SemanticValue::Integers(vec![1])),
        ("padding_mode", SemanticValue::Text("automatic".into())),
        ("groups", SemanticValue::Unsigned(0)),
    ] {
        let mut a = attrs();
        a.insert(AttributeId::new(name).unwrap(), v);
        assert!(Conv2dBf16Geometry::new(
            &t(&[2, 4, 4, 5]),
            &t(&[6, 2, 2, 2]),
            &t(&[6]),
            &t(&[2, 6, 2, 3]),
            &a
        )
        .is_err());
    }
    let mut a = attrs();
    for (n, v) in [
        ("padding", SemanticValue::Integers(vec![0; 4])),
        ("stride", SemanticValue::Integers(vec![1; 2])),
        ("dilation", SemanticValue::Integers(vec![1; 2])),
        ("groups", SemanticValue::Unsigned(1)),
    ] {
        a.insert(AttributeId::new(n).unwrap(), v);
    }
    let n = u64::MAX / 4;
    assert!(Conv2dBf16Geometry::new(
        &t(&[1, 1, 1, n]),
        &t(&[1, 1, 1, 4]),
        &t(&[1]),
        &t(&[1, 1, 1, n - 3]),
        &a
    )
    .is_err());
    let c = conv2d_bf16_contract().unwrap();
    assert!(c
        .descriptor
        .outputs
        .iter()
        .all(|t| t.alias() == &AliasPolicy::NoAlias));
    a.insert(
        AttributeId::new("allow_fused_bias").unwrap(),
        SemanticValue::Bool(true),
    );
    assert!(c.descriptor.validate_attributes(&a).is_err());
}
