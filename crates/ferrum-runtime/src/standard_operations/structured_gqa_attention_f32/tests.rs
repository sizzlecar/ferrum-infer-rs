use super::*;
use crate::{AliasPolicy, ResolvedTensorLayout};
fn tensor(columns: u64, dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![6, columns], dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn attributes() -> BTreeMap<AttributeId, SemanticValue> {
    let mut attributes = BTreeMap::from([
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
            SemanticValue::Rational(CanonicalRational::new(1, 2).unwrap()),
        ),
    ]);
    for (name, values) in [
        ("segment_lengths", vec![1, 2]),
        ("segment_bidirectional", vec![0, 1]),
        ("key_run_counts", vec![2, 2]),
        ("key_run_lengths", vec![1, 2, 2, 1]),
        ("key_run_valid", vec![0, 1, 1, 0]),
    ] {
        attributes.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values),
        );
    }
    attributes
}
#[test]
fn independent_f32_ports_reject_half_and_keep_compact_gqa_mask_semantics() {
    let contract = structured_gqa_attention_f32_contract().unwrap();
    for port in contract
        .descriptor()
        .inputs
        .iter()
        .chain(&contract.descriptor().outputs)
    {
        assert_eq!(port.element_types(), &BTreeSet::from([ElementType::F32]));
        assert_eq!(port.alias(), &AliasPolicy::NoAlias);
    }
    let q = tensor(8, ElementType::F32);
    let kv = tensor(4, ElementType::F32);
    let attrs = attributes();
    let g = StructuredGqaAttentionF32Geometry::new(&q, &kv, &kv, &q, &attrs).unwrap();
    assert_eq!(
        (g.query_heads(), g.key_value_heads(), g.group_size()),
        (4, 2, 2)
    );
    assert_eq!(g.mask().visible_end(1), Some(3));
    assert_eq!(g.mask().key_is_valid(0, 0), Some(false));
    assert_eq!(g.mask().key_is_valid(1, 0), Some(true));
    assert!(StructuredGqaAttentionBf16Geometry::new(&q, &kv, &kv, &q, &attrs).is_err());
    for dtype in [ElementType::F16, ElementType::Bf16] {
        let half = tensor(4, dtype);
        assert!(StructuredGqaAttentionF32Geometry::new(&q, &half, &half, &q, &attrs).is_err());
    }
    for bad in [tensor(6, ElementType::F32), tensor(10, ElementType::F32)] {
        assert!(StructuredGqaAttentionF32Geometry::new(&q, &bad, &bad, &q, &attrs).is_err());
    }
    let mut bad = attrs;
    bad.insert(
        AttributeId::new("causal").unwrap(),
        SemanticValue::Unsigned(1),
    );
    assert!(StructuredGqaAttentionF32Geometry::new(&q, &kv, &kv, &q, &bad).is_err());
}

#[test]
fn f32_nonpower_heads_are_bounded_without_expanding_the_bf16_domain() {
    for width in [2, 3, 64, 72, 80, 96, 128, 255, 256] {
        let q = tensor(4 * width, ElementType::F32);
        let kv = tensor(2 * width, ElementType::F32);
        let mut a = attributes();
        a.insert(
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(width),
        );
        let g = StructuredGqaAttentionF32Geometry::new(&q, &kv, &kv, &q, &a).unwrap();
        assert_eq!(
            (g.head_dim(), g.query_heads(), g.key_value_heads()),
            (width, 4, 2)
        );
        let half_q = tensor(4 * width, ElementType::Bf16);
        let half_kv = tensor(2 * width, ElementType::Bf16);
        assert_eq!(
            StructuredGqaAttentionBf16Geometry::new(&half_q, &half_kv, &half_kv, &half_q, &a)
                .is_ok(),
            width.is_power_of_two(),
        );
        let wrong = tensor(2 * width + 1, ElementType::F32);
        assert!(StructuredGqaAttentionF32Geometry::new(&q, &wrong, &wrong, &q, &a).is_err());
    }
    for width in [0, 1, 257] {
        let q = tensor(4 * width.max(1), ElementType::F32);
        let kv = tensor(2 * width.max(1), ElementType::F32);
        let mut a = attributes();
        a.insert(
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(width),
        );
        assert!(StructuredGqaAttentionF32Geometry::new(&q, &kv, &kv, &q, &a).is_err());
    }
}
