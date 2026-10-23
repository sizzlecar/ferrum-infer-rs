use super::*;
use crate::{Bf16SameShapeGeometry, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

#[test]
fn plain_add_is_same_shape_without_attributes_gate_or_auxiliary_resources() {
    let c = plain_residual_add_bf16_contract().unwrap();
    let d = c.descriptor();
    assert_eq!(d.id.as_str(), PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID);
    assert_eq!(d.version, ContractVersion::new(1, 0));
    assert_eq!(d.inputs.len(), 2);
    assert_eq!(d.inputs[0].dimensions(), d.inputs[1].dimensions());
    assert_eq!(d.inputs[0].dimensions(), d.outputs[0].dimensions());
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
    d.validate_attributes(&BTreeMap::new()).unwrap();
    assert!(d
        .validate_attributes(&BTreeMap::from([(
            AttributeId::new("broadcast").unwrap(),
            SemanticValue::Text("row".into())
        )]))
        .is_err());
}

#[test]
fn shared_geometry_preserves_real_shape_and_has_no_normalization_width_limit() {
    let t = |shape: Vec<u64>, dtype| {
        ResolvedTensorSpec::new(shape, dtype, ResolvedTensorLayout::Contiguous).unwrap()
    };
    let x = t(vec![2, (1 << 24) + 1], ElementType::Bf16);
    assert_eq!(
        Bf16SameShapeGeometry::new(&x, &x).unwrap().width(),
        (1 << 24) + 1
    );
    assert!(
        Bf16SameShapeGeometry::new(&x, &t(vec![1, 2 * ((1 << 24) + 1)], ElementType::Bf16))
            .is_err()
    );
    let f = t(vec![2, 4], ElementType::F16);
    assert!(Bf16SameShapeGeometry::new(&f, &f).is_err());
}
