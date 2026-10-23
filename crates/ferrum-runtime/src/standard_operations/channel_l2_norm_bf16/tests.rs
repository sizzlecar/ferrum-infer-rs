use super::*;
fn t(d: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(d.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("epsilon").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(3, 2).unwrap()),
        ),
    ])
}
#[test]
fn real_rank_four_gamma_and_nchw_geometry_are_preserved() {
    let x = t(&[2, 1152, 3, 5], ElementType::Bf16);
    let gamma = t(&[1152, 1, 1, 1], ElementType::Bf16);
    let g = ChannelL2NormBf16Geometry::new(&x, &gamma, &x, &attrs()).unwrap();
    assert_eq!(g.element_count(), 34560);
    assert_eq!(g.logical_arithmetic_operations(), 207360);
    assert_eq!(g.effective_scale(), 1.5);
    assert!(
        ChannelL2NormBf16Geometry::new(&x, &t(&[1152, 1, 1], ElementType::Bf16), &x, &attrs())
            .is_err()
    );
    assert!(ChannelL2NormBf16Geometry::new(
        &x,
        &t(&[1152, 1, 1, 1], ElementType::F32),
        &x,
        &attrs()
    )
    .is_err());
    assert!(ChannelL2NormBf16Geometry::new(
        &x,
        &gamma,
        &t(&[2, 1152, 5, 3], ElementType::Bf16),
        &attrs()
    )
    .is_err());
}
#[test]
fn explicit_scalars_and_checked_logical_work_are_not_inferred_from_numel() {
    let x = t(&[1, 3, 1, 1], ElementType::Bf16);
    let gamma = t(&[3, 1, 1, 1], ElementType::Bf16);
    for (name, v) in [
        ("scale", CanonicalRational::new(0, 1).unwrap()),
        ("scale", CanonicalRational::new(-1, 1).unwrap()),
        (
            "epsilon",
            CanonicalRational::new(1, 1_000_000_000_001).unwrap(),
        ),
    ] {
        let mut a = attrs();
        a.insert(AttributeId::new(name).unwrap(), SemanticValue::Rational(v));
        assert!(ChannelL2NormBf16Geometry::new(&x, &gamma, &x, &a).is_err());
    }
    let huge = t(&[1, 1, 1, u64::MAX / 4], ElementType::Bf16);
    assert!(ChannelL2NormBf16Geometry::new(
        &huge,
        &t(&[1, 1, 1, 1], ElementType::Bf16),
        &huge,
        &attrs()
    )
    .is_err());
    let mut extra = attrs();
    extra.insert(AttributeId::new("rms").unwrap(), SemanticValue::Bool(true));
    assert!(channel_l2_norm_bf16_contract()
        .unwrap()
        .descriptor
        .validate_attributes(&extra)
        .is_err());
}
