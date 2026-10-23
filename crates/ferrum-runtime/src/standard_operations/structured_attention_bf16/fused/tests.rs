use super::*;

fn tensor(rows: u64, columns: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

fn attributes() -> BTreeMap<AttributeId, SemanticValue> {
    let mut result = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(5),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(128),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
        ),
    ]);
    for (name, values) in [
        ("segment_lengths", vec![2, 3]),
        ("segment_bidirectional", vec![0, 1]),
        ("key_run_counts", vec![3, 1]),
        ("key_run_lengths", vec![1, 1, 3, 5]),
        ("key_run_valid", vec![1, 0, 1, 0]),
    ] {
        result.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values),
        );
    }
    result
}

#[test]
fn fused_identity_allows_only_scratch_and_preserves_the_original_descriptor() {
    let original = structured_attention_bf16_contract().unwrap();
    let fused = structured_attention_fused_bf16_contract().unwrap();
    let old = original.descriptor();
    let new = fused.descriptor();
    assert_ne!(old.id, new.id);
    assert_eq!(
        new.id.as_str(),
        STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID
    );
    assert_eq!(old.version, ContractVersion::new(1, 0));
    assert_eq!(new.version, ContractVersion::new(1, 0));
    assert_eq!(
        old.resources.scratch,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(new.resources.scratch, ResourcePresenceRequirement::Optional);
    assert_eq!(
        new.resources.binding,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(
        new.resources.persistent,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(new.oracle, OracleSpec::OperationDefined);
    for port in new.inputs.iter().chain(&new.outputs) {
        assert_eq!(port.element_types(), &BTreeSet::from([ElementType::Bf16]));
        assert_eq!(port.alias(), &AliasPolicy::NoAlias);
        assert_eq!(port.layouts(), [LayoutConstraint::Contiguous]);
    }
    // No accidental change to the original contract's other fields/rounding ABI.
    let mut restored = new.clone();
    restored.id = old.id.clone();
    restored.provider = old.provider.clone();
    restored.resources.scratch = old.resources.scratch;
    assert_eq!(&restored, old);
    assert_ne!(
        serde_json::to_vec(old).unwrap(),
        serde_json::to_vec(new).unwrap()
    );
    assert_ne!(old.fingerprint().unwrap(), new.fingerprint().unwrap());
}

#[test]
fn fused_geometry_preserves_causal_prefix_internal_padding_and_empty_samples() {
    let attrs = attributes();
    structured_attention_fused_bf16_contract()
        .unwrap()
        .descriptor()
        .validate_attributes(&attrs)
        .unwrap();
    let t = tensor(10, 256);
    let g = StructuredAttentionBf16Geometry::new(&t, &t, &t, &t, &attrs).unwrap();
    assert_eq!((g.batch(), g.heads(), g.head_dim()), (2, 2, 128));
    assert_eq!(g.effective_scale(), 0.125);
    let mut visible_pairs = 0;
    let mut nonempty = 0;
    for b in 0..2 {
        for q in 0..5 {
            let mut row = 0;
            for k in 0..5 {
                let expected = b == 0 && k != 1 && (k <= q || q >= 2);
                let actual = k < g.mask().visible_end(q).unwrap()
                    && g.mask().key_is_valid(b, k) == Some(true);
                assert_eq!(actual, expected);
                row += u64::from(actual);
            }
            visible_pairs += row;
            nonempty += u64::from(row != 0);
        }
    }
    assert_eq!(visible_pairs, 14);
    assert_eq!(nonempty, 5);
    assert_eq!(g.visible_pairs(), visible_pairs);
    assert_eq!(g.nonempty_rows(), nonempty);
}

#[test]
fn fused_ports_and_attributes_fail_closed_before_provider_workspace_estimation() {
    let contract = structured_attention_fused_bf16_contract().unwrap();
    let attrs = attributes();
    let t = tensor(10, 256);
    for (name, value) in [
        ("sequence_length", SemanticValue::Unsigned(3)),
        ("head_dim", SemanticValue::Unsigned(1152)),
        ("key_run_counts", SemanticValue::Integers(vec![3])),
        ("key_run_lengths", SemanticValue::Integers(vec![1, 0, 4, 5])),
        (
            "scale",
            SemanticValue::Rational(CanonicalRational::new(0, 1).unwrap()),
        ),
    ] {
        let mut bad = attrs.clone();
        bad.insert(AttributeId::new(name).unwrap(), value);
        assert!(StructuredAttentionBf16Geometry::new(&t, &t, &t, &t, &bad).is_err());
    }
    let mut unknown = attrs.clone();
    unknown.insert(
        AttributeId::new("workspace_bytes").unwrap(),
        SemanticValue::Unsigned(0),
    );
    assert!(contract.descriptor().validate_attributes(&unknown).is_err());
    let wrong_dtype = ResolvedTensorSpec::new(
        vec![10, 256],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(StructuredAttentionBf16Geometry::new(&t, &wrong_dtype, &t, &t, &attrs).is_err());
    assert!(StructuredAttentionBf16Geometry::new(&t, &t, &t, &tensor(5, 256), &attrs).is_err());
    let mut huge = attrs;
    let s = i64::from(u32::MAX);
    for (name, values) in [
        ("segment_lengths", vec![s]),
        ("segment_bidirectional", vec![1]),
        ("key_run_counts", vec![1]),
        ("key_run_lengths", vec![s]),
        ("key_run_valid", vec![1]),
    ] {
        huge.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values),
        );
    }
    huge.insert(
        AttributeId::new("sequence_length").unwrap(),
        SemanticValue::Unsigned(u64::from(u32::MAX)),
    );
    let huge_tensor = tensor(u64::from(u32::MAX), 256);
    assert!(StructuredAttentionBf16Geometry::new(
        &huge_tensor,
        &huge_tensor,
        &huge_tensor,
        &huge_tensor,
        &huge
    )
    .is_err());
}
