use super::*;
fn tensor(rows: u64, columns: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs(batch: u64, d: u64) -> BTreeMap<AttributeId, SemanticValue> {
    [
        ("batch", SemanticValue::Unsigned(batch)),
        ("head_dim", SemanticValue::Unsigned(d)),
        (
            "scale",
            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
        ),
        ("key_run_counts", SemanticValue::Integers(vec![3, 1])),
        ("key_run_lengths", SemanticValue::Integers(vec![1, 1, 3, 5])),
        ("key_run_valid", SemanticValue::Integers(vec![1, 0, 1, 0])),
    ]
    .into_iter()
    .map(|(k, v)| (AttributeId::new(k).unwrap(), v))
    .collect()
}
fn check(
    q: &ResolvedTensorSpec,
    k: &ResolvedTensorSpec,
    a: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<CrossAttentionBf16Geometry, VNextError> {
    CrossAttentionBf16Geometry::new(q, k, k, q, a)
}
#[test]
fn rectangular_batch_geometry_keeps_key_padding_and_empty_samples_independent() {
    let g = check(&tensor(6, 8), &tensor(10, 8), &attrs(2, 4)).unwrap();
    assert_eq!(
        (g.batch(), g.query_length(), g.key_value_length(), g.heads()),
        (2, 3, 5, 2)
    );
    assert_eq!(
        (g.query_bytes(), g.key_value_bytes(), g.element_count()),
        (96, 160, 48)
    );
    assert_eq!(g.valid_keys(0), Some(4));
    assert_eq!(g.valid_keys(1), Some(0));
    for k in 0..5 {
        assert_eq!(g.key_is_valid(0, k), Some(k != 1));
        assert_eq!(g.key_is_valid(1, k), Some(false));
    }
    assert_eq!(g.key_is_valid(2, 0), None);
    assert_eq!(g.key_is_valid(0, 5), None);
    assert_eq!(
        g.logical_arithmetic_operations(),
        3 * 2 * ((5 * 4 + 6) * 4 - (2 * 4 + 5))
    );
}
#[test]
fn different_query_length_does_not_change_key_run_coverage() {
    assert!(check(&tensor(2, 144), &tensor(10, 144), &attrs(2, 72)).is_ok());
    assert!(check(&tensor(14, 256), &tensor(10, 256), &attrs(2, 128)).is_ok());
    assert!(check(&tensor(3, 8), &tensor(10, 8), &attrs(2, 4)).is_err());
    assert!(check(&tensor(6, 8), &tensor(9, 8), &attrs(2, 4)).is_err());
    assert!(check(&tensor(6, 8), &tensor(10, 12), &attrs(2, 4)).is_err());
    assert!(check(&tensor(6, 70), &tensor(10, 70), &attrs(2, 70)).is_err());
}
#[test]
fn malformed_or_noncanonical_masks_and_unknown_attributes_are_rejected() {
    let base = attrs(2, 4);
    for (name, value) in [
        ("batch", SemanticValue::Unsigned(0)),
        ("batch", SemanticValue::Unsigned(17)),
        ("key_run_counts", SemanticValue::Integers(vec![0, 4])),
        ("key_run_counts", SemanticValue::Integers(vec![3])),
        ("key_run_lengths", SemanticValue::Integers(vec![1, 0, 4, 5])),
        ("key_run_lengths", SemanticValue::Integers(vec![1, 1, 4, 5])),
        ("key_run_valid", SemanticValue::Integers(vec![1, 1, 0, 0])),
        ("key_run_valid", SemanticValue::Integers(vec![1, 2, 1, 0])),
        (
            "scale",
            SemanticValue::Rational(CanonicalRational::new(0, 1).unwrap()),
        ),
        ("causal", SemanticValue::Unsigned(1)),
    ] {
        let mut a = base.clone();
        a.insert(AttributeId::new(name).unwrap(), value);
        assert!(check(&tensor(6, 8), &tensor(10, 8), &a).is_err(), "{name}");
    }
}
#[test]
fn dtype_layout_output_and_byte_extent_cannot_be_inferred_from_equal_numel() {
    let q = tensor(6, 8);
    let k = tensor(10, 8);
    let a = attrs(2, 4);
    assert!(CrossAttentionBf16Geometry::new(&q, &k, &tensor(5, 16), &q, &a).is_err());
    assert!(CrossAttentionBf16Geometry::new(&q, &k, &k, &tensor(3, 16), &a).is_err());
    let f32 = ResolvedTensorSpec::new(
        vec![6, 8],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(check(&f32, &k, &a).is_err());
    assert!(check(&tensor(6, 1 << 59), &tensor(10, 1 << 59), &a).is_err());
}
#[test]
fn vendor_contract_is_separate_and_declares_only_optional_invocation_scratch() {
    let regular = cross_attention_bf16_contract().unwrap();
    let fused = cross_attention_fused_bf16_contract().unwrap();
    let a = regular.descriptor();
    let b = fused.descriptor();
    assert_ne!(a.id, b.id);
    assert_ne!(a.fingerprint().unwrap(), b.fingerprint().unwrap());
    assert_eq!(a.inputs, b.inputs);
    assert_eq!(a.outputs, b.outputs);
    assert_eq!(a.attributes, b.attributes);
    assert_eq!(a.resources.scratch, ResourcePresenceRequirement::Forbidden);
    assert_eq!(b.resources.scratch, ResourcePresenceRequirement::Optional);
    assert_eq!(a.resources.binding, b.resources.binding);
    assert_eq!(a.resources.persistent, b.resources.persistent);
    assert_eq!(a.outputs[0].alias(), &AliasPolicy::NoAlias);
}
