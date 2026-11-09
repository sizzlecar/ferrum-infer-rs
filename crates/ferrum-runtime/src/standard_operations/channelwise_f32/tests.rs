use super::*;
fn spec(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    [("group_width", 6), ("column_start", 2), ("column_count", 2)]
        .into_iter()
        .map(|(k, v)| (AttributeId::new(k).unwrap(), SemanticValue::Unsigned(v)))
        .collect()
}
#[test]
fn grouped_slice_checks_complete_groups_and_original_column_window() {
    let x = spec(&[2, 12], ElementType::F32);
    let y = spec(&[2, 4], ElementType::F32);
    let a = attrs();
    let g =
        F32ChannelwiseGeometry::new(F32ChannelwiseOp::GroupedColumnSlice, &[&x], &y, &a).unwrap();
    assert_eq!(
        (
            g.source_columns(),
            g.columns(),
            g.group_width(),
            g.column_start(),
            g.column_count()
        ),
        (12, 4, 6, 2, 2)
    );
    assert_eq!(g.input_elements(0), Some(24));
    assert_eq!(g.logical_arithmetic_operations(), 0);
    for (name, bad) in [
        ("group_width", 5),
        ("group_width", 0),
        ("column_start", 5),
        ("column_count", 0),
        ("column_count", u64::MAX),
    ] {
        let mut a = a.clone();
        a.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Unsigned(bad),
        );
        assert!(
            F32ChannelwiseGeometry::new(F32ChannelwiseOp::GroupedColumnSlice, &[&x], &y, &a)
                .is_err()
        );
    }
    assert!(F32ChannelwiseGeometry::new(
        F32ChannelwiseOp::GroupedColumnSlice,
        &[&x],
        &spec(&[1, 8], ElementType::F32),
        &a
    )
    .is_err());
}
#[test]
fn channel_vectors_and_arithmetic_are_exactly_typed_without_numel_broadcast_shortcuts() {
    let x = spec(&[2, 4], ElementType::F32);
    let vector = spec(&[4], ElementType::F32);
    let empty = BTreeMap::new();
    for kind in F32ChannelwiseOp::ALL {
        let contract = kind.contract().unwrap();
        for port in contract
            .descriptor()
            .inputs
            .iter()
            .chain(&contract.descriptor().outputs)
        {
            assert_eq!(port.element_types(), &BTreeSet::from([ElementType::F32]));
            assert_eq!(port.alias(), &AliasPolicy::NoAlias);
        }
        assert_eq!(
            contract.descriptor().resources.scratch,
            ResourcePresenceRequirement::Forbidden
        );
    }
    let g = F32ChannelwiseGeometry::new(
        F32ChannelwiseOp::ScaledResidual,
        &[&x, &x, &vector],
        &x,
        &empty,
    )
    .unwrap();
    assert_eq!(g.logical_arithmetic_operations(), 16);
    for wrong in [
        spec(&[1, 4], ElementType::F32),
        spec(&[2, 2], ElementType::F32),
        spec(&[4], ElementType::Bf16),
    ] {
        assert!(
            F32ChannelwiseGeometry::new(F32ChannelwiseOp::BiasAdd, &[&x, &wrong], &x, &empty)
                .is_err()
        );
        assert!(F32ChannelwiseGeometry::new(
            F32ChannelwiseOp::ScaledResidual,
            &[&x, &x, &wrong],
            &x,
            &empty
        )
        .is_err());
    }
    assert!(F32ChannelwiseGeometry::new(
        F32ChannelwiseOp::ScaledResidual,
        &[&x, &vector, &x],
        &x,
        &empty
    )
    .is_err());
    let half = spec(&[2, 4], ElementType::Bf16);
    assert!(
        F32ChannelwiseGeometry::new(F32ChannelwiseOp::NanToNum, &[&half], &half, &empty).is_err()
    );
}
