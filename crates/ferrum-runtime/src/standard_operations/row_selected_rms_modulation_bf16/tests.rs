use super::*;
use crate::ResolvedTensorLayout;
fn spec(shape: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = RowSelectedRmsModulationProfile::Block8192Vec8Warp16Fma
        .attributes()
        .unwrap();
    for (k, v) in [
        (
            "epsilon",
            SemanticValue::Rational(CanonicalRational::new(1, 100000).unwrap()),
        ),
        ("run_lengths", SemanticValue::Integers(vec![1, 2])),
        ("parameter_rows", SemanticValue::Integers(vec![2, 0])),
        ("shift_column", SemanticValue::Unsigned(0)),
        ("scale_column", SemanticValue::Unsigned(5376)),
    ] {
        a.insert(AttributeId::new(k).unwrap(), v);
    }
    a
}
#[test]
fn fused_profile_keeps_original_compact_table_and_checked_routing() {
    let x = spec(&[3, 5376]);
    let table = spec(&[4, 6 * 5376]);
    let gamma = spec(&[5376]);
    let mut a = attrs();
    let g = RowSelectedRmsModulationGeometry::new(
        RowSelectedRmsModulationKind::Normalize,
        &[&x, &table, &gamma],
        &[&x],
        &a,
    )
    .unwrap();
    assert_eq!(g.routing().parameter_row(0), Some(2));
    assert_eq!(g.routing().parameter_row(2), Some(0));
    assert_eq!(g.routing().parameter_row(3), None);
    a.insert(
        AttributeId::new("gate_column").unwrap(),
        SemanticValue::Unsigned(2 * 5376),
    );
    let g = RowSelectedRmsModulationGeometry::new(
        RowSelectedRmsModulationKind::Residual,
        &[&x, &table, &gamma, &x],
        &[&x, &x],
        &a,
    )
    .unwrap();
    assert_eq!(g.gate_column(), 2 * 5376);
    assert_eq!(g.logical_arithmetic_operations(), 3 * (9 * 5376 + 3));
}
#[test]
fn routing_shape_and_numerical_profile_fail_closed() {
    let x = spec(&[3, 5376]);
    let table = spec(&[4, 6 * 5376]);
    let gamma = spec(&[5376]);
    for (key, value) in [
        (
            "profile",
            SemanticValue::Text("old-staged-modulation".into()),
        ),
        ("parameter_rows", SemanticValue::Integers(vec![4, 0])),
        ("run_lengths", SemanticValue::Integers(vec![1, 1])),
        ("scale_column", SemanticValue::Unsigned(6 * 5376)),
    ] {
        let mut a = attrs();
        a.insert(AttributeId::new(key).unwrap(), value);
        assert!(RowSelectedRmsModulationGeometry::new(
            RowSelectedRmsModulationKind::Normalize,
            &[&x, &table, &gamma],
            &[&x],
            &a
        )
        .is_err());
    }
    let x_large = spec(&[3, 8193]);
    let gamma_large = spec(&[8193]);
    assert!(RowSelectedRmsModulationGeometry::new(
        RowSelectedRmsModulationKind::Normalize,
        &[&x_large, &table, &gamma_large],
        &[&x_large],
        &attrs()
    )
    .is_err());
}
