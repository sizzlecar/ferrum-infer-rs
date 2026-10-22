use super::*;
fn tensor(rows: u64, columns: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs(
    batch: u64,
    start: u64,
    columns: u64,
    runs: &[[i64; 4]],
) -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = BTreeMap::from([
        (
            AttributeId::new("batch").unwrap(),
            SemanticValue::Unsigned(batch),
        ),
        (
            AttributeId::new("column_start").unwrap(),
            SemanticValue::Unsigned(start),
        ),
        (
            AttributeId::new("column_count").unwrap(),
            SemanticValue::Unsigned(columns),
        ),
    ]);
    for (index, name) in [
        "run_lengths",
        "source_starts",
        "source_batch_strides",
        "source_row_strides",
    ]
    .into_iter()
    .enumerate()
    {
        a.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(runs.iter().map(|r| r[index]).collect()),
        );
    }
    a
}
#[test]
fn contracts_are_distinct_exact_bf16_bit_transport_without_auxiliary_resources() {
    for (contract, input_count) in [
        (row_gather_bf16_contract().unwrap(), 1),
        (batched_row_concat_bf16_contract().unwrap(), 2),
    ] {
        let d = contract.descriptor();
        assert_eq!(d.inputs.len(), input_count);
        assert_eq!(d.outputs.len(), 1);
        assert_eq!(d.oracle, OracleSpec::OperationDefined);
        assert_eq!(d.version, ContractVersion::new(1, 0));
        for tensor in d.inputs.iter().chain(&d.outputs) {
            assert_eq!(tensor.alias(), &AliasPolicy::NoAlias);
            assert_eq!(tensor.element_types(), &BTreeSet::from([ElementType::Bf16]));
            assert_eq!(tensor.layouts(), [LayoutConstraint::Contiguous]);
        }
        assert_eq!(d.resources, no_auxiliary_resources());
        assert!(d.validate_attributes(&BTreeMap::new()).is_err());
    }
    let mut a = attrs(2, 1, 2, &[[2, 2, 0, 0], [1, 0, 1, 0]]);
    row_gather_bf16_contract()
        .unwrap()
        .descriptor()
        .validate_attributes(&a)
        .unwrap();
    a.insert(
        AttributeId::new("implicit_batch").unwrap(),
        SemanticValue::Unsigned(2),
    );
    assert!(row_gather_bf16_contract()
        .unwrap()
        .descriptor()
        .validate_attributes(&a)
        .is_err());
}
#[test]
fn checked_gather_supports_shared_rows_per_batch_rows_and_consecutive_slices() {
    let g = RowGatherBf16Geometry::new(
        &tensor(3, 4),
        &tensor(6, 2),
        &attrs(2, 1, 2, &[[2, 2, 0, 0], [1, 0, 1, 0]]),
    )
    .unwrap();
    assert_eq!(
        (
            g.batch(),
            g.source_rows(),
            g.source_width(),
            g.rows_per_batch(),
            g.column_start(),
            g.column_count(),
            g.element_count()
        ),
        (2, 3, 4, 3, 1, 2, 12)
    );
    assert_eq!(g.run_lengths(), [2, 1]);
    assert_eq!(g.source_starts(), [2, 0]);
    assert_eq!(g.source_batch_strides(), [0, 1]);
    assert_eq!(g.source_row_strides(), [0, 0]);
    let s = RowGatherBf16Geometry::new(
        &tensor(8, 3),
        &tensor(4, 3),
        &attrs(2, 0, 3, &[[2, 1, 4, 1]]),
    )
    .unwrap();
    assert_eq!(s.rows_per_batch(), 2);
    assert!(RowGatherBf16Geometry::new(
        &tensor(6, 3),
        &tensor(4, 3),
        &attrs(2, 0, 3, &[[2, 1, 4, 1]])
    )
    .is_err());
}
#[test]
fn invalid_runs_shapes_column_windows_and_address_overflow_fail_closed() {
    let source = tensor(3, 4);
    let output = tensor(6, 2);
    let base = attrs(2, 1, 2, &[[2, 2, 0, 0], [1, 0, 1, 0]]);
    for (name, value) in [
        ("batch", SemanticValue::Unsigned(0)),
        ("column_count", SemanticValue::Unsigned(0)),
        ("column_start", SemanticValue::Unsigned(u64::MAX)),
        ("run_lengths", SemanticValue::Integers(vec![0, 3])),
        ("source_starts", SemanticValue::Integers(vec![-1, 0])),
        ("source_batch_strides", SemanticValue::Integers(vec![0])),
        ("source_row_strides", SemanticValue::Integers(vec![0, 2])),
        ("run_lengths", SemanticValue::Integers(vec![1; 33])),
        ("run_lengths", SemanticValue::Integers(vec![])),
    ] {
        let mut a = base.clone();
        a.insert(AttributeId::new(name).unwrap(), value);
        assert!(
            RowGatherBf16Geometry::new(&source, &output, &a).is_err(),
            "{name}"
        );
    }
    assert!(RowGatherBf16Geometry::new(&source, &tensor(3, 4), &base).is_err());
    let f16 = ResolvedTensorSpec::new(
        vec![3, 4],
        ElementType::F16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(RowGatherBf16Geometry::new(&f16, &output, &base).is_err());
    let strided = ResolvedTensorSpec::new(
        vec![3, 4],
        ElementType::Bf16,
        ResolvedTensorLayout::Strided {
            byte_strides: vec![8, 2],
        },
    )
    .unwrap();
    assert!(RowGatherBf16Geometry::new(&strided, &output, &base).is_err());
    assert!(RowGatherBf16Geometry::new(
        &tensor(u64::MAX / 2, 1),
        &tensor(3, 1),
        &attrs(3, 0, 1, &[[1, 3, i64::MAX, 0]])
    )
    .is_err());
    assert!(RowGatherBf16Geometry::new(
        &tensor(1, 1),
        &tensor(1, 1),
        &attrs(1, 0, 1, &[[i64::MAX, 0, 0, 0]; 3])
    )
    .is_err());
}
#[test]
fn concat_geometry_proves_per_sample_lengths_and_exact_output_shape() {
    let g =
        BatchedRowConcatBf16Geometry::new(&tensor(4, 3), &tensor(2, 3), &tensor(6, 3), 2).unwrap();
    assert_eq!(
        (
            g.batch(),
            g.left_rows_per_batch(),
            g.right_rows_per_batch(),
            g.columns(),
            g.element_count()
        ),
        (2, 2, 1, 3, 18)
    );
    for batch in [0, 3, 5] {
        assert!(BatchedRowConcatBf16Geometry::new(
            &tensor(4, 3),
            &tensor(2, 3),
            &tensor(6, 3),
            batch
        )
        .is_err());
    }
    assert!(
        BatchedRowConcatBf16Geometry::new(&tensor(4, 3), &tensor(2, 2), &tensor(6, 3), 2).is_err()
    );
    assert!(
        BatchedRowConcatBf16Geometry::new(&tensor(4, 3), &tensor(2, 3), &tensor(3, 6), 2).is_err()
    );
    let a = BTreeMap::from([(
        AttributeId::new("batch").unwrap(),
        SemanticValue::Unsigned(2),
    )]);
    batched_row_concat_bf16_contract()
        .unwrap()
        .descriptor()
        .validate_attributes(&a)
        .unwrap();
}
