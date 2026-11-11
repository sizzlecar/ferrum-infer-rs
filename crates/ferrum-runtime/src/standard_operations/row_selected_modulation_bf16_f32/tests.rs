use super::*;
fn t(rows: u64, columns: u64, dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![rows, columns], dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn attrs(lengths: Vec<i64>, rows: Vec<i64>) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("run_lengths").unwrap(),
            SemanticValue::Integers(lengths),
        ),
        (
            AttributeId::new("parameter_rows").unwrap(),
            SemanticValue::Integers(rows),
        ),
    ])
}
#[test]
fn strict_mixed_precision_ports_and_exact_bounded_rle_rows_are_validated() {
    let x = t(5, 3, ElementType::Bf16);
    let p = t(2, 6, ElementType::F32);
    let y = t(5, 3, ElementType::F32);
    let a = attrs(vec![2, 1, 2], vec![1, 0, 1]);
    let g = RowSelectedModulationBf16F32Geometry::new(&x, &p, &y, &a).unwrap();
    assert_eq!(
        (0..6).map(|r| g.parameter_row(r)).collect::<Vec<_>>(),
        [Some(1), Some(1), Some(0), Some(1), Some(1), None]
    );
    assert_eq!(g.logical_arithmetic_operations(), 45);
    for bad in [
        attrs(vec![2, 2], vec![0, 1]),
        attrs(vec![0, 5], vec![0, 1]),
        attrs(vec![-1, 6], vec![0, 1]),
        attrs(vec![5], vec![-1]),
        attrs(vec![5], vec![2]),
        attrs(vec![2, 3], vec![0]),
    ] {
        assert!(RowSelectedModulationBf16F32Geometry::new(&x, &p, &y, &bad).is_err());
    }
    assert!(
        RowSelectedModulationBf16F32Geometry::new(&x, &t(1, 12, ElementType::F32), &y, &a).is_err()
    );
    assert!(
        RowSelectedModulationBf16F32Geometry::new(&t(5, 3, ElementType::F32), &p, &y, &a).is_err()
    );
    assert!(
        RowSelectedModulationBf16F32Geometry::new(&x, &t(2, 6, ElementType::Bf16), &y, &a).is_err()
    );
    assert!(RowSelectedModulationBf16F32Geometry::new(&x, &p, &x, &a).is_err());
    assert!(RowSelectedModulationBf16F32Geometry::new(
        &t(32, 3, ElementType::Bf16),
        &p,
        &t(32, 3, ElementType::F32),
        &attrs(vec![1; 32], vec![0; 32])
    )
    .is_ok());
    assert!(RowSelectedModulationBf16F32Geometry::new(
        &t(33, 3, ElementType::Bf16),
        &p,
        &t(33, 3, ElementType::F32),
        &attrs(vec![1; 33], vec![0; 33])
    )
    .is_err());
    let d = row_selected_modulation_bf16_f32_contract().unwrap();
    assert_eq!(
        d.descriptor().resources.scratch,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(
        d.descriptor().inputs[0].element_types(),
        &BTreeSet::from([ElementType::Bf16])
    );
    assert_eq!(
        d.descriptor().inputs[1].element_types(),
        &BTreeSet::from([ElementType::F32])
    );
    assert_eq!(
        d.descriptor().outputs[0].element_types(),
        &BTreeSet::from([ElementType::F32])
    );
}
