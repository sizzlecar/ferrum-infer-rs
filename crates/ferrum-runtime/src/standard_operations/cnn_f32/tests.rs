use super::*;
fn t(shape: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("padding").unwrap(),
            SemanticValue::Integers(vec![0, 1, 0, 1]),
        ),
        (
            AttributeId::new("stride").unwrap(),
            SemanticValue::Integers(vec![2, 2]),
        ),
        (
            AttributeId::new("dilation").unwrap(),
            SemanticValue::Integers(vec![1, 1]),
        ),
        (
            AttributeId::new("groups").unwrap(),
            SemanticValue::Unsigned(1),
        ),
        (
            AttributeId::new("padding_mode").unwrap(),
            SemanticValue::Text("reflect".into()),
        ),
    ])
}
#[test]
fn f32_reflect_convolution_preserves_exact_original_shapes_and_no_edge_repeat() {
    let x = t(&[2, 3, 5, 7]);
    let w = t(&[4, 3, 3, 3]);
    let b = t(&[4]);
    let y = t(&[2, 4, 2, 3]);
    let g = Conv2dF32Geometry::new(&x, &w, &b, &y, &attrs()).unwrap();
    assert_eq!(g.padding(), [0, 1, 0, 1]);
    assert_eq!(g.source_spatial(5, 7), Some((3, 5)));
    assert_eq!(g.source_spatial(-1, -1), Some((1, 1)));
    assert!(Conv2dF32Geometry::new(&x, &w, &b, &t(&[2, 4, 3, 2]), &attrs()).is_err());
    let half = ResolvedTensorSpec::new(
        vec![2, 3, 5, 7],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(Conv2dF32Geometry::new(&half, &w, &b, &y, &attrs()).is_err());
    let mut bad = attrs();
    bad.insert(
        AttributeId::new("padding").unwrap(),
        SemanticValue::Integers(vec![0, 5, 0, 1]),
    );
    assert!(Conv2dF32Geometry::new(&x, &w, &b, &y, &bad).is_err());
}
#[test]
fn group_statistics_are_nchw_channel_spatial_groups_and_residual_does_not_fake_matrix_views() {
    let x = t(&[2, 8, 17, 19]);
    let p = t(&[8]);
    let eps = CanonicalRational::new(1, 1_000_000).unwrap();
    let g = GroupNormSiluNchwF32Geometry::new(&x, &p, &p, &x, 4, eps).unwrap();
    assert_eq!(g.group_elements(), 2 * 17 * 19);
    assert!(GroupNormSiluNchwF32Geometry::new(&x, &p, &p, &x, 3, eps).is_err());
    assert!(GroupNormSiluNchwF32Geometry::new(&x, &t(&[4]), &p, &x, 4, eps).is_err());
    assert!(F32NchwGeometry::new(&x, &t(&[2, 8, 19, 17])).is_err());
    assert!(F32NchwGeometry::new(&t(&[16, 17 * 19]), &t(&[16, 17 * 19])).is_err());
    for kind in [F32CnnOp::GroupNormSilu, F32CnnOp::ResidualAdd] {
        assert_eq!(
            kind.contract().unwrap().descriptor().resources.scratch,
            ResourcePresenceRequirement::Forbidden
        );
    }
}
