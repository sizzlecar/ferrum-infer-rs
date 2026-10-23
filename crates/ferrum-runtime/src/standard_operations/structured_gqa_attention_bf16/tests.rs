use super::*;
fn t(rows: u64, columns: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(3),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
        ),
    ]);
    for (name, values) in [
        ("segment_lengths", vec![3]),
        ("segment_bidirectional", vec![0]),
        ("key_run_counts", vec![1, 1]),
        ("key_run_lengths", vec![3, 3]),
        ("key_run_valid", vec![1, 1]),
    ] {
        a.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values),
        );
    }
    a
}
#[test]
fn independent_contract_keeps_compact_kv_and_reuses_the_closed_structured_mask_schema() {
    let c = structured_gqa_attention_bf16_contract().unwrap();
    let d = c.descriptor();
    assert_eq!(d.id.as_str(), STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID);
    assert_eq!(d.version, ContractVersion::new(1, 0));
    assert_ne!(d.inputs[0].dimensions(), d.inputs[1].dimensions());
    assert_eq!(d.inputs[1].dimensions(), d.inputs[2].dimensions());
    assert_eq!(d.inputs[0].dimensions(), d.outputs[0].dimensions());
    assert_eq!(
        d.attributes,
        structured_attention_bf16_contract()
            .unwrap()
            .descriptor()
            .attributes
    );
    for p in d.inputs.iter().chain(&d.outputs) {
        assert_eq!(p.element_types(), &BTreeSet::from([ElementType::Bf16]));
        assert_eq!(p.alias(), &AliasPolicy::NoAlias);
    }
    assert_eq!(d.resources.scratch, ResourcePresenceRequirement::Forbidden);
    assert_eq!(d.resources.binding, ResourcePresenceRequirement::Forbidden);
    assert_eq!(
        d.resources.persistent,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(d.oracle, OracleSpec::OperationDefined);
    let q = t(6, 8);
    let kv = t(6, 4);
    // Existing equal-QKV contract stays strict.
    assert!(StructuredAttentionBf16Geometry::new(&q, &kv, &kv, &q, &attrs()).is_err());
}
#[test]
fn group_mapping_and_nominal_work_follow_query_heads_not_compact_kv_heads() {
    let q = t(6, 8);
    let kv = t(6, 4);
    let g = StructuredGqaAttentionBf16Geometry::new(&q, &kv, &kv, &q, &attrs()).unwrap();
    assert_eq!(
        (
            g.rows(),
            g.query_columns(),
            g.key_value_columns(),
            g.batch(),
            g.query_heads(),
            g.key_value_heads(),
            g.group_size()
        ),
        (6, 8, 4, 2, 4, 2, 2)
    );
    assert_eq!(g.visible_pairs(), 12);
    assert_eq!(g.nonempty_rows(), 6);
    assert_eq!(g.element_count(), 48);
    assert_eq!(g.logical_arithmetic_operations(), 4 * (16 * 12 - 9 * 6));
    let mqa = t(6, 2);
    assert_eq!(
        StructuredGqaAttentionBf16Geometry::new(&q, &mqa, &mqa, &q, &attrs())
            .unwrap()
            .group_size(),
        4
    );
    assert_eq!(
        StructuredGqaAttentionBf16Geometry::new(&q, &q, &q, &q, &attrs())
            .unwrap()
            .group_size(),
        1
    );
    let mut actual = attrs();
    actual.insert(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Unsigned(128),
    );
    let actual_q = t(6, 4096);
    let actual_kv = t(6, 1024);
    let g = StructuredGqaAttentionBf16Geometry::new(
        &actual_q, &actual_kv, &actual_kv, &actual_q, &actual,
    )
    .unwrap();
    assert_eq!(
        (g.query_heads(), g.key_value_heads(), g.group_size()),
        (32, 8, 4)
    );
}
#[test]
fn actual_head_ratio_specs_mask_and_scale_are_validated_without_equal_numel_shortcuts() {
    let q = t(6, 8);
    let kv = t(6, 4);
    let a = attrs();
    for bad in [t(6, 6), t(6, 10), t(3, 8), t(6, 1)] {
        assert!(StructuredGqaAttentionBf16Geometry::new(&q, &bad, &bad, &q, &a).is_err());
    }
    assert!(StructuredGqaAttentionBf16Geometry::new(&q, &kv, &t(3, 8), &q, &a).is_err());
    assert!(StructuredGqaAttentionBf16Geometry::new(&q, &kv, &kv, &t(3, 16), &a).is_err());
    let half = ResolvedTensorSpec::new(
        vec![6, 4],
        ElementType::F16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(StructuredGqaAttentionBf16Geometry::new(&q, &half, &half, &q, &a).is_err());
    for (name, v) in [
        ("segment_lengths", SemanticValue::Integers(vec![2])),
        ("key_run_valid", SemanticValue::Integers(vec![1, 2])),
        ("head_dim", SemanticValue::Unsigned(3)),
        ("scale", SemanticValue::Unsigned(1)),
        ("causal", SemanticValue::Unsigned(1)),
    ] {
        let mut bad = a.clone();
        bad.insert(AttributeId::new(name).unwrap(), v);
        assert!(StructuredGqaAttentionBf16Geometry::new(&q, &kv, &kv, &q, &bad).is_err());
    }
}
