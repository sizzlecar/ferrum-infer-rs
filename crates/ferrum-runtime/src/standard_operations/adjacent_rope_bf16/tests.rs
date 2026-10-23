use super::*;
use crate::SemanticValue;
fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn attrs(d: u64) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Unsigned(d),
    )])
}

#[test]
fn contract_has_explicit_f32_frequencies_and_no_alias_or_auxiliary_resources() {
    let contract = adjacent_rope_bf16_contract().unwrap();
    let d = contract.descriptor();
    assert_eq!(d.id.as_str(), ADJACENT_ROPE_BF16_OPERATION_ID);
    assert_eq!(d.version, ContractVersion::new(1, 0));
    assert_eq!(d.inputs.len(), 2);
    assert_eq!(d.outputs.len(), 1);
    assert_eq!(
        d.inputs[0].dimensions(),
        [symbol("rows"), symbol("columns")]
    );
    assert_eq!(
        d.inputs[1].dimensions(),
        [symbol("sequence"), symbol("pairs"), exact(2)]
    );
    assert_eq!(
        d.inputs[0].element_types(),
        &BTreeSet::from([ElementType::Bf16])
    );
    assert_eq!(
        d.inputs[1].element_types(),
        &BTreeSet::from([ElementType::F32])
    );
    assert_eq!(
        d.outputs[0].element_types(),
        &BTreeSet::from([ElementType::Bf16])
    );
    for port in d.inputs.iter().chain(&d.outputs) {
        assert_eq!(port.alias(), &AliasPolicy::NoAlias);
        assert_eq!(port.layouts(), [LayoutConstraint::Contiguous]);
    }
    assert_eq!(d.oracle, OracleSpec::OperationDefined);
    assert_eq!(d.resources.scratch, ResourcePresenceRequirement::Forbidden);
    assert_eq!(d.resources.binding, ResourcePresenceRequirement::Forbidden);
    assert_eq!(
        d.resources.persistent,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(
        d.provider.required_capabilities,
        BTreeSet::from([CapabilityId::new(ADJACENT_ROPE_BF16_CAPABILITY_ID).unwrap()])
    );
}

#[test]
fn actual_frequency_shape_proves_sequence_batch_heads_and_large_even_dimensions() {
    for d in [2, 6, 128, (1 << 24) + 2] {
        let x = tensor(&[4, 2 * d], ElementType::Bf16);
        let f = tensor(&[2, d / 2, 2], ElementType::F32);
        let g = AdjacentRopeBf16Geometry::new(&x, &f, &x, d).unwrap();
        assert_eq!(
            (
                g.rows(),
                g.columns(),
                g.sequence(),
                g.batch(),
                g.heads(),
                g.head_dim()
            ),
            (4, 2 * d, 2, 2, 2, d)
        );
    }
}

#[test]
fn geometry_rejects_wrong_partition_rank_dtype_layout_and_same_numel_shapes() {
    let x = tensor(&[4, 8], ElementType::Bf16);
    let f = tensor(&[2, 2, 2], ElementType::F32);
    for d in [0, 1, 3, 6, 8] {
        assert!(AdjacentRopeBf16Geometry::new(&x, &f, &x, d).is_err());
    }
    for bad in [
        tensor(&[3, 2, 2], ElementType::F32),
        tensor(&[2, 1, 4], ElementType::F32),
        tensor(&[2, 4], ElementType::F32),
        tensor(&[2, 2, 2], ElementType::Bf16),
        ResolvedTensorSpec::new(
            vec![2, 2, 2],
            ElementType::F32,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![32, 8, 4],
            },
        )
        .unwrap(),
    ] {
        assert!(AdjacentRopeBf16Geometry::new(&x, &bad, &x, 4).is_err());
    }
    let bad_output = tensor(&[2, 16], ElementType::Bf16);
    assert!(AdjacentRopeBf16Geometry::new(&x, &f, &bad_output, 4).is_err());
    for bad in [
        tensor(&[4, 8], ElementType::F16),
        tensor(&[32], ElementType::Bf16),
        ResolvedTensorSpec::new(
            vec![4, 8],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![32, 2],
            },
        )
        .unwrap(),
    ] {
        assert!(AdjacentRopeBf16Geometry::new(&bad, &f, &bad, 4).is_err());
    }
}

#[test]
fn attributes_are_closed_and_tensor_byte_overflow_is_rejected() {
    let contract = adjacent_rope_bf16_contract().unwrap();
    let descriptor = contract.descriptor();
    descriptor.validate_attributes(&attrs(128)).unwrap();
    assert!(descriptor.validate_attributes(&BTreeMap::new()).is_err());
    for (name, value) in [
        ("head_dim", SemanticValue::Integer(128)),
        ("theta", SemanticValue::Unsigned(10000)),
        ("sequence", SemanticValue::Unsigned(2)),
    ] {
        let mut bad = attrs(128);
        bad.insert(AttributeId::new(name).unwrap(), value);
        assert!(descriptor.validate_attributes(&bad).is_err());
    }
    assert!(ResolvedTensorSpec::new(
        vec![u64::MAX / 4 + 1, 1, 2],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous
    )
    .is_err());
}
