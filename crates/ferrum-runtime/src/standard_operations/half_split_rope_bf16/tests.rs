use super::*;
use crate::SemanticValue;
fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
#[test]
fn contract_requires_three_bf16_ports_explicit_head_dim_and_no_alias() {
    let c = half_split_rope_bf16_contract().unwrap();
    let d = c.descriptor();
    assert_eq!(d.id.as_str(), HALF_SPLIT_ROPE_BF16_OPERATION_ID);
    assert_eq!(d.version, ContractVersion::new(1, 0));
    assert_eq!(d.inputs.len(), 3);
    assert_eq!(
        d.inputs[1].dimensions(),
        [symbol("rows"), symbol("head_dim")]
    );
    assert_eq!(d.inputs[1].dimensions(), d.inputs[2].dimensions());
    for p in d.inputs.iter().chain(&d.outputs) {
        assert_eq!(p.element_types(), &BTreeSet::from([ElementType::Bf16]));
        assert_eq!(p.alias(), &AliasPolicy::NoAlias);
        assert_eq!(p.layouts(), [LayoutConstraint::Contiguous]);
    }
    assert_eq!(d.oracle, OracleSpec::OperationDefined);
    assert_eq!(d.resources.scratch, ResourcePresenceRequirement::Forbidden);
    assert_eq!(d.resources.binding, ResourcePresenceRequirement::Forbidden);
    assert_eq!(
        d.resources.persistent,
        ResourcePresenceRequirement::Forbidden
    );
    assert!(d.validate_attributes(&BTreeMap::new()).is_err());
    let mut attrs = BTreeMap::from([(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Unsigned(128),
    )]);
    d.validate_attributes(&attrs).unwrap();
    attrs.insert(
        AttributeId::new("theta").unwrap(),
        SemanticValue::Unsigned(10000),
    );
    assert!(d.validate_attributes(&attrs).is_err());
}
#[test]
fn geometry_validates_each_frequency_row_without_requiring_power_two_heads() {
    for head in [2, 6, 128, (1 << 24) + 2] {
        let x = t(&[3, 2 * head], ElementType::Bf16);
        let f = t(&[3, head], ElementType::Bf16);
        let g = HalfSplitRopeBf16Geometry::new(&x, &f, &f, &x, head).unwrap();
        assert_eq!(
            (g.rows(), g.columns(), g.heads(), g.head_dim()),
            (3, 2 * head, 2, head)
        );
    }
    let x = t(&[4, 8], ElementType::Bf16);
    let f = t(&[4, 4], ElementType::Bf16);
    for head in [0, 1, 3, 6, 8] {
        assert!(HalfSplitRopeBf16Geometry::new(&x, &f, &f, &x, head).is_err());
    }
    for bad in [
        t(&[2, 8], ElementType::Bf16),
        t(&[4, 4], ElementType::F32),
        t(&[4, 2, 2], ElementType::Bf16),
        ResolvedTensorSpec::new(
            vec![4, 4],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![16, 2],
            },
        )
        .unwrap(),
    ] {
        assert!(HalfSplitRopeBf16Geometry::new(&x, &bad, &f, &x, 4).is_err());
        assert!(HalfSplitRopeBf16Geometry::new(&x, &f, &bad, &x, 4).is_err());
    }
    assert!(
        HalfSplitRopeBf16Geometry::new(&x, &f, &f, &t(&[2, 16], ElementType::Bf16), 4).is_err()
    );
    assert!(ResolvedTensorSpec::new(
        vec![u64::MAX / 4 + 1, 2],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous
    )
    .is_err());
}
