use super::*;
use crate::SemanticValue;

fn spec(dimensions: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dimensions.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
#[test]
fn variant_is_contract_identity_with_only_f32_unary_no_alias_ports() {
    let tanh = gelu_tanh_f32_contract().unwrap();
    let erf = gelu_erf_f32_contract().unwrap();
    assert_ne!(tanh.descriptor().id, erf.descriptor().id);
    assert_ne!(
        tanh.descriptor().fingerprint().unwrap(),
        erf.descriptor().fingerprint().unwrap()
    );
    for variant in [GeluF32Variant::Tanh, GeluF32Variant::Erf] {
        let c = variant.contract().unwrap();
        let d = c.descriptor();
        assert_eq!(d.id.as_str(), variant.operation_id());
        assert_eq!(d.inputs.len(), 1);
        assert_eq!(d.outputs.len(), 1);
        for p in d.inputs.iter().chain(&d.outputs) {
            assert_eq!(p.element_types(), &BTreeSet::from([ElementType::F32]));
            assert_eq!(p.alias(), &AliasPolicy::NoAlias);
            assert_eq!(p.layouts(), [LayoutConstraint::Contiguous]);
        }
        assert!(d
            .validate_attributes(&BTreeMap::from([(
                AttributeId::new("approximate").unwrap(),
                SemanticValue::Text("tanh".into()),
            )]))
            .is_err());
    }
}
#[test]
fn original_matrix_extent_dtype_layout_and_checked_work_are_preserved() {
    let x = spec(&[3, 7], ElementType::F32);
    let g = GeluF32Geometry::new(&x, &x).unwrap();
    assert_eq!(g.elements(), 21);
    assert_eq!(g.logical_operations(GeluF32Variant::Tanh).unwrap(), 189);
    assert_eq!(g.logical_operations(GeluF32Variant::Erf).unwrap(), 105);
    for bad in [
        spec(&[7, 3], ElementType::F32),
        spec(&[3, 7], ElementType::Bf16),
        spec(&[21], ElementType::F32),
        ResolvedTensorSpec::new(
            vec![3, 7],
            ElementType::F32,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![28, 4],
            },
        )
        .unwrap(),
    ] {
        assert!(GeluF32Geometry::new(&x, &bad).is_err());
        assert!(GeluF32Geometry::new(&bad, &x).is_err());
    }
    let too_large = spec(&[1, isize::MAX as u64 / 4 + 1], ElementType::F32);
    assert!(GeluF32Geometry::new(&too_large, &too_large).is_err());
    let large = spec(&[1, isize::MAX as u64 / 4], ElementType::F32);
    let g = GeluF32Geometry::new(&large, &large).unwrap();
    assert!(g.logical_operations(GeluF32Variant::Tanh).is_err());
    assert!(g.logical_operations(GeluF32Variant::Erf).is_ok());
}
