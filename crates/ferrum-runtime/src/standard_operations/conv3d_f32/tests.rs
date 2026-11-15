use super::*;

fn tensor(d: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        d.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = BTreeMap::new();
    for (k, v) in [
        ("padding", vec![2, 0, 0, 1, 0, 1]),
        ("stride", vec![2, 2, 2]),
        ("dilation", vec![1, 1, 1]),
        ("padding_modes", vec![0, 1, 1]),
    ] {
        a.insert(AttributeId::new(k).unwrap(), SemanticValue::Integers(v));
    }
    a.insert(
        AttributeId::new("groups").unwrap(),
        SemanticValue::Unsigned(2),
    );
    a
}
#[test]
fn causal_depth_and_asymmetric_spatial_reflection_keep_independent_axes() {
    let g = Conv3dF32Geometry::new(
        &tensor(&[2, 4, 5, 6, 8]),
        &tensor(&[6, 2, 3, 3, 3]),
        &tensor(&[6]),
        &tensor(&[2, 6, 3, 3, 4]),
        &attrs(),
    )
    .unwrap();
    assert_eq!(g.source_spatial([-1, 6, 8]), None);
    assert_eq!(g.source_spatial([0, 6, 8]), Some([0, 4, 6]));
    assert_eq!(g.source_spatial([4, 5, 7]), Some([4, 5, 7]));
    assert_eq!(g.source_spatial([5, 5, 7]), None);
    assert_eq!(g.padded_spatial(), [7, 7, 9]);
    assert_eq!(g.padded_input_bytes(), 2 * 4 * 7 * 7 * 9 * 4);
    // Every admitted output temporal window ends at 2*t. Future frames cannot
    // enter it, including when the first window consists mostly of zero pad.
    for t in 0..3 {
        let sources: Vec<_> = (0..3)
            .filter_map(|k| g.source_spatial([t * 2 + k - 2, 0, 0]))
            .collect();
        assert_eq!(sources.last().unwrap()[0], (t * 2) as u64);
        assert!(sources.iter().all(|p| p[0] <= (t * 2) as u64));
    }
}
#[test]
fn invalid_groups_reflection_and_overflow_fail_before_execution() {
    let make = |a: &BTreeMap<_, _>, w: &[u64], y: &[u64]| {
        Conv3dF32Geometry::new(
            &tensor(&[2, 4, 5, 6, 8]),
            &tensor(w),
            &tensor(&[6]),
            &tensor(y),
            a,
        )
    };
    assert!(make(&attrs(), &[6, 3, 3, 3, 3], &[2, 6, 3, 3, 4]).is_err());
    assert!(make(&attrs(), &[6, 2, 3, 3, 3], &[2, 6, 5, 3, 4]).is_err());
    for (key, value) in [
        ("padding_modes", vec![0, 2, 1]),
        ("padding", vec![2, 0, 0, 6, 0, 1]),
        ("stride", vec![0, 2, 2]),
        ("padding", vec![i64::MAX, i64::MAX, 0, 1, 0, 1]),
        ("dilation", vec![i64::MAX, 1, 1]),
    ] {
        let mut a = attrs();
        a.insert(
            AttributeId::new(key).unwrap(),
            SemanticValue::Integers(value),
        );
        assert!(
            make(&a, &[6, 2, 3, 3, 3], &[2, 6, 3, 3, 4]).is_err(),
            "{key}"
        );
    }
}
