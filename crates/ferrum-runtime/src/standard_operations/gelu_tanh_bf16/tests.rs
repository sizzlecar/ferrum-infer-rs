use super::*;
use crate::SemanticValue;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
#[test]
fn gelu_contract_is_distinct_unary_bf16_noalias_with_closed_empty_attributes() {
    let c = gelu_tanh_bf16_contract().unwrap();
    let d = c.descriptor();
    assert_eq!(d.id.as_str(), GELU_TANH_BF16_OPERATION_ID);
    assert_eq!(d.version, ContractVersion::new(1, 0));
    assert_eq!(d.inputs.len(), 1);
    assert_eq!(d.outputs.len(), 1);
    assert_eq!(d.oracle, OracleSpec::OperationDefined);
    for p in d.inputs.iter().chain(&d.outputs) {
        assert_eq!(p.alias(), &AliasPolicy::NoAlias);
        assert_eq!(p.element_types(), &BTreeSet::from([ElementType::Bf16]));
        assert_eq!(p.layouts(), [LayoutConstraint::Contiguous]);
    }
    d.validate_attributes(&BTreeMap::new()).unwrap();
    assert!(d
        .validate_attributes(&BTreeMap::from([(
            AttributeId::new("approximate").unwrap(),
            SemanticValue::Text("tanh".into())
        )]))
        .is_err());
}
#[test]
fn same_shape_geometry_preserves_rank_dtype_layout_and_allows_large_metadata_width() {
    let tensor = |shape, dtype, layout| ResolvedTensorSpec::new(shape, dtype, layout).unwrap();
    let x = tensor(
        vec![2, 3],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    );
    Bf16SameShapeGeometry::new(&x, &x).unwrap();
    for wrong in [
        tensor(
            vec![3, 2],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        ),
        tensor(
            vec![2, 3],
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        ),
        tensor(
            vec![2, 3],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![6, 2],
            },
        ),
    ] {
        assert!(Bf16SameShapeGeometry::new(&x, &wrong).is_err());
    }
    let wide = tensor(
        vec![1, (1 << 24) + 1],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    );
    assert_eq!(
        Bf16SameShapeGeometry::new(&wide, &wide).unwrap().width(),
        (1 << 24) + 1
    );
}
