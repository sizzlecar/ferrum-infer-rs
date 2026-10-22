use super::*;
use crate::SemanticValue;
fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn epsilon() -> CanonicalRational {
    CanonicalRational::new(1, 1_000_000).unwrap()
}
#[test]
fn whole_row_contract_has_original_bf16_weight_and_only_rational_epsilon() {
    let c = zero_centered_rms_norm_bf16_contract().unwrap();
    let d = c.descriptor();
    assert_eq!(d.id.as_str(), ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID);
    assert_eq!(d.version, ContractVersion::new(1, 0));
    assert_eq!(d.inputs.len(), 2);
    assert_eq!(d.inputs[1].dimensions(), [symbol("columns")]);
    for p in d.inputs.iter().chain(&d.outputs) {
        assert_eq!(p.alias(), &AliasPolicy::NoAlias);
        assert_eq!(p.element_types(), &BTreeSet::from([ElementType::Bf16]));
        assert_eq!(p.layouts(), [LayoutConstraint::Contiguous]);
    }
    assert_eq!(d.oracle, OracleSpec::OperationDefined);
    let a = BTreeMap::from([(
        AttributeId::new("epsilon").unwrap(),
        SemanticValue::Rational(epsilon()),
    )]);
    d.validate_attributes(&a).unwrap();
    assert!(d.validate_attributes(&BTreeMap::new()).is_err());
    for (name, value) in [
        ("epsilon", SemanticValue::Unsigned(1)),
        ("head_dim", SemanticValue::Unsigned(128)),
        ("hidden_size", SemanticValue::Unsigned(4096)),
    ] {
        let mut bad = a.clone();
        bad.insert(AttributeId::new(name).unwrap(), value);
        assert!(d.validate_attributes(&bad).is_err());
    }
}
#[test]
fn geometry_uses_entire_actual_row_and_rejects_dtype_layout_or_shape_mismatch() {
    for width in [1, 3, 128, 4096, 1 << 24] {
        let x = tensor(&[2, width], ElementType::Bf16);
        let w = tensor(&[width], ElementType::Bf16);
        let g = ZeroCenteredRmsNormBf16Geometry::new(&x, &w, &x, epsilon()).unwrap();
        assert_eq!((g.rows(), g.columns()), (2, width));
        assert_eq!(g.effective_epsilon(), 1e-6_f32);
    }
    let x = tensor(&[2, 8], ElementType::Bf16);
    for w in [
        tensor(&[4], ElementType::Bf16),
        tensor(&[2, 4], ElementType::Bf16),
        tensor(&[8], ElementType::F32),
        ResolvedTensorSpec::new(
            vec![8],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![2],
            },
        )
        .unwrap(),
    ] {
        assert!(ZeroCenteredRmsNormBf16Geometry::new(&x, &w, &x, epsilon()).is_err());
    }
    let w = tensor(&[8], ElementType::Bf16);
    assert!(ZeroCenteredRmsNormBf16Geometry::new(
        &x,
        &w,
        &tensor(&[4, 4], ElementType::Bf16),
        epsilon()
    )
    .is_err());
    let f = tensor(&[2, 8], ElementType::F16);
    assert!(ZeroCenteredRmsNormBf16Geometry::new(&f, &w, &f, epsilon()).is_err());
    let huge = tensor(&[1, (1 << 24) + 1], ElementType::Bf16);
    assert!(ZeroCenteredRmsNormBf16Geometry::new(
        &huge,
        &tensor(&[(1 << 24) + 1], ElementType::Bf16),
        &huge,
        epsilon()
    )
    .is_err());
}
#[test]
fn epsilon_exact_bounds_and_checked_byte_extents_are_enforced() {
    let x = tensor(&[1, 1], ElementType::Bf16);
    let w = tensor(&[1], ElementType::Bf16);
    for e in [
        CanonicalRational::new(1, 1_000_000_000_000).unwrap(),
        CanonicalRational::new(1, 1).unwrap(),
    ] {
        ZeroCenteredRmsNormBf16Geometry::new(&x, &w, &x, e).unwrap();
    }
    for (n, d) in [(0, 1), (-1, 1), (1, 1_000_000_000_001), (2, 1)] {
        assert!(ZeroCenteredRmsNormBf16Geometry::new(
            &x,
            &w,
            &x,
            CanonicalRational::new(n, d).unwrap()
        )
        .is_err());
    }
    assert!(ResolvedTensorSpec::new(
        vec![u64::MAX, 2],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous
    )
    .is_err());
}
