use super::*;
fn t(shape: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
#[test]
fn explicit_batch_axes_and_transpose_are_checked_without_broadcast() {
    let a = t(&[2, 3, 5, 7]);
    let y = t(&[2, 3, 5, 11]);
    for transpose in [false, true] {
        let rhs = t(if transpose {
            &[2, 3, 11, 7]
        } else {
            &[2, 3, 7, 11]
        });
        let attrs = BatchedF32Geometry::matmul_attributes(transpose).unwrap();
        let g = BatchedF32Geometry::new(BatchedF32Op::Matmul, &[&a, &rhs], &y, &attrs).unwrap();
        assert_eq!(
            (g.batch(), g.group(), g.m(), g.n(), g.k()),
            (2, 3, 5, 11, 7)
        );
        assert_eq!(g.arithmetic_operations().unwrap(), 2 * 3 * 5 * 11 * 7 * 2);
        for bad in [t(&[1, 3, 5, 11]), t(&[3, 2, 5, 11]), t(&[2, 3, 11, 5])] {
            assert!(
                BatchedF32Geometry::new(BatchedF32Op::Matmul, &[&a, &rhs], &bad, &attrs).is_err()
            );
        }
    }
    assert!(BatchedF32Geometry::new(
        BatchedF32Op::Matmul,
        &[&a, &t(&[1, 3, 7, 11])],
        &y,
        &BatchedF32Geometry::matmul_attributes(false).unwrap()
    )
    .is_err());
}
#[test]
fn scale_bits_nonfinite_shapes_and_capacity_are_validated() {
    let x = t(&[2, 3, 5, 7]);
    let attrs = BatchedF32Geometry::scale_attributes(-0.0).unwrap();
    let g = BatchedF32Geometry::new(BatchedF32Op::Scale, &[&x], &x, &attrs).unwrap();
    assert_eq!(g.scale().to_bits(), 0x80000000);
    assert!(BatchedF32Geometry::scale_attributes(f32::NAN).is_err());
    let mut bad = attrs.clone();
    bad.insert(
        AttributeId::new("scale_bits").unwrap(),
        SemanticValue::Unsigned(0x7f800000),
    );
    assert!(BatchedF32Geometry::new(BatchedF32Op::Scale, &[&x], &x, &bad).is_err());
    // Valid u64 storage extent, but not addressable by a signed host pointer.
    let huge = t(&[1, 1, 1, 1 << 61]);
    assert!(
        BatchedF32Geometry::new(BatchedF32Op::Softmax, &[&huge], &huge, &BTreeMap::new()).is_err()
    );
    for op in BatchedF32Op::ALL {
        assert_eq!(
            op.contract().unwrap().descriptor().outputs[0].alias(),
            &AliasPolicy::NoAlias
        );
    }
}
