use super::*;
fn tensor(rows: u64, columns: u64, e: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![rows, columns], e, ResolvedTensorLayout::Contiguous).unwrap()
}
fn attrs(starts: &[i64], lengths: &[i64]) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("destination_starts").unwrap(),
            SemanticValue::Integers(starts.to_vec()),
        ),
        (
            AttributeId::new("run_lengths").unwrap(),
            SemanticValue::Integers(lengths.to_vec()),
        ),
    ])
}
#[test]
fn replacement_and_addition_have_distinct_closed_f32_contracts_and_one_shared_row_map() {
    let a = row_scatter_add_f32_contract().unwrap();
    let r = row_scatter_replace_f32_contract().unwrap();
    assert_ne!(
        a.descriptor().fingerprint().unwrap(),
        r.descriptor().fingerprint().unwrap()
    );
    for kind in [RowScatterF32Op::Add, RowScatterF32Op::Replace] {
        let c = kind.contract().unwrap();
        for p in c.descriptor().inputs.iter().chain(&c.descriptor().outputs) {
            assert_eq!(p.element_types(), &BTreeSet::from([ElementType::F32]));
            assert_eq!(p.alias(), &AliasPolicy::NoAlias);
        }
    }
    let x = tensor(8, 7, ElementType::F32);
    let u = tensor(3, 7, ElementType::F32);
    let g = RowScatterF32Geometry::new(&x, &u, &x, &attrs(&[1, 6], &[2, 1])).unwrap();
    assert_eq!(g.runs()[1].source_start, 2);
    for (row, expected) in [
        (0, None),
        (1, Some(0)),
        (2, Some(1)),
        (3, None),
        (6, Some(2)),
        (7, None),
        (8, None),
    ] {
        assert_eq!(g.update_row(row), expected);
    }
    assert_eq!(g.logical_operations(RowScatterF32Op::Add), 21);
    assert_eq!(g.logical_operations(RowScatterF32Op::Replace), 0);
}
#[test]
fn invalid_runs_dtype_shape_and_native_packet_capacity_fail_before_any_storage() {
    let x = tensor(8, 7, ElementType::F32);
    let u = tensor(3, 7, ElementType::F32);
    for a in [
        attrs(&[1, 2], &[2, 1]),
        attrs(&[6, 1], &[1, 2]),
        attrs(&[-1, 6], &[2, 1]),
        attrs(&[1, 8], &[2, 1]),
        attrs(&[1, 6], &[0, 3]),
        attrs(&[1], &[2]),
        attrs(
            &vec![0; ROW_SCATTER_F32_MAX_RUNS + 1],
            &vec![1; ROW_SCATTER_F32_MAX_RUNS + 1],
        ),
    ] {
        assert!(RowScatterF32Geometry::new(&x, &u, &x, &a).is_err());
    }
    assert!(RowScatterF32Geometry::new(
        &x,
        &tensor(3, 7, ElementType::Bf16),
        &x,
        &attrs(&[1, 6], &[2, 1])
    )
    .is_err());
    assert!(RowScatterF32Geometry::new(
        &x,
        &u,
        &tensor(7, 8, ElementType::F32),
        &attrs(&[1, 6], &[2, 1])
    )
    .is_err());
}
