use super::*;
use crate::{AliasPolicy, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

#[test]
fn separate_fused_identity_preserves_strict_contract_and_f32_boundaries() {
    let old = structured_gqa_attention_f32_contract().unwrap();
    let new = structured_gqa_attention_fused_f32_contract().unwrap();
    let old = old.descriptor();
    let new = new.descriptor();
    assert_ne!(old.id, new.id);
    assert_ne!(old.fingerprint().unwrap(), new.fingerprint().unwrap());
    assert_eq!(new.resources.scratch, ResourcePresenceRequirement::Optional);
    assert_eq!(
        old.resources.scratch,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(
        new.resources.binding,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(
        new.resources.persistent,
        ResourcePresenceRequirement::Forbidden
    );
    for p in new.inputs.iter().chain(&new.outputs) {
        assert_eq!(p.element_types(), &BTreeSet::from([ElementType::F32]));
        assert_eq!(p.alias(), &AliasPolicy::NoAlias);
    }
    let mut restored = new.clone();
    restored.id = old.id.clone();
    restored.provider = old.provider.clone();
    restored.resources.scratch = old.resources.scratch;
    assert_eq!(&restored, old);
}

#[test]
fn fused_geometry_keeps_compact_heads_partial_masks_empty_samples_and_scale() {
    let mut attrs = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(5),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(72),
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
        attrs.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values),
        );
    }
    let tensor = |columns, dtype| {
        ResolvedTensorSpec::new(vec![10, columns], dtype, ResolvedTensorLayout::Contiguous).unwrap()
    };
    let q = tensor(144, ElementType::F32);
    let kv = tensor(72, ElementType::F32);
    let contract = structured_gqa_attention_fused_f32_contract().unwrap();
    contract.descriptor().validate_attributes(&attrs).unwrap();
    let g = StructuredGqaAttentionF32Geometry::new(&q, &kv, &kv, &q, &attrs).unwrap();
    assert_eq!(
        (
            g.batch(),
            g.query_heads(),
            g.key_value_heads(),
            g.head_dim()
        ),
        (2, 2, 1, 72)
    );
    assert_eq!(g.effective_scale(), 0.125);
    assert_eq!(g.visible_pairs(), 14);
    assert_eq!(g.nonempty_rows(), 5);
    for b in 0..2 {
        for row in 0..5 {
            for key in 0..5 {
                assert_eq!(
                    key < g.mask().visible_end(row).unwrap()
                        && g.mask().key_is_valid(b, key) == Some(true),
                    b == 0 && key != 1 && (key <= row || row >= 2)
                );
            }
        }
    }
    assert!(StructuredGqaAttentionF32Geometry::new(
        &q,
        &tensor(72, ElementType::Bf16),
        &kv,
        &q,
        &attrs
    )
    .is_err());
    attrs.insert(
        AttributeId::new("workspace_bytes").unwrap(),
        SemanticValue::Unsigned(0),
    );
    assert!(contract.descriptor().validate_attributes(&attrs).is_err());
}
