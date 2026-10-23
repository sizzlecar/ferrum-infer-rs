use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
fn t(d: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        d.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn scale() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("scale").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
    )])
}
fn gather() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("scale_h").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("scale_w").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("channel_stride").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("channel_offsets").unwrap(),
            SemanticValue::Integers(vec![0, 0, 1, 1]),
        ),
    ])
}
#[test]
fn image_norm_preserves_actual_rank_three_gamma_and_staged_contract() {
    let x = t(&[2, 3, 2, 5]);
    let mut attrs = scale();
    attrs.insert(
        AttributeId::new("epsilon").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
    );
    let g = ChannelL2NormImageBf16Geometry::new(&x, &t(&[3, 1, 1]), &x, &attrs).unwrap();
    assert_eq!(g.element_count(), 60);
    assert_eq!(g.logical_arithmetic_operations(), 360);
    assert!(ChannelL2NormImageBf16Geometry::new(&x, &t(&[3, 1, 1, 1]), &x, &attrs).is_err());
    assert!(ChannelL2NormImageBf16Geometry::new(&x, &t(&[3, 1, 2]), &x, &attrs).is_err());
    assert!(
        channel_l2_norm_image_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
            != channel_l2_norm_bf16_contract()
                .unwrap()
                .descriptor()
                .fingerprint()
                .unwrap()
    );
}
#[test]
fn spatial_attention_validates_packed_layout_non_power_width_and_work() {
    let g = SpatialAttentionBf16Geometry::new(&t(&[2, 3456, 3, 5]), &t(&[2, 1152, 3, 5]), &scale())
        .unwrap();
    assert_eq!((g.batch(), g.channels(), g.spatial()), (2, 1152, 15));
    assert_eq!(g.element_count(), 34560);
    let tiny =
        SpatialAttentionBf16Geometry::new(&t(&[1, 9, 1, 2]), &t(&[1, 3, 1, 2]), &scale()).unwrap();
    // P=4, C=3, S=2: 2 * (2*7 + 1*15 + 4) = 66.
    assert_eq!(tiny.logical_arithmetic_operations(), 66);
    for (x, y) in [
        (t(&[1, 9, 2, 1]), t(&[1, 3, 1, 2])),
        (t(&[1, 8, 1, 2]), t(&[1, 3, 1, 2])),
        (t(&[1, 6147, 1, 1]), t(&[1, 2049, 1, 1])),
    ] {
        assert!(SpatialAttentionBf16Geometry::new(&x, &y, &scale()).is_err());
    }
    let mut attrs = scale();
    attrs.insert(
        AttributeId::new("causal").unwrap(),
        SemanticValue::Unsigned(1),
    );
    assert!(
        SpatialAttentionBf16Geometry::new(&t(&[1, 3, 1, 1]), &t(&[1, 1, 1, 1]), &attrs).is_err()
    );
    // Tensor-byte or logical-work overflow must be rejected without allocating.
    if let (Ok(x), Ok(y)) = (
        ResolvedTensorSpec::new(
            vec![1, 3, 1, 1 << 61],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        ),
        ResolvedTensorSpec::new(
            vec![1, 1, 1, 1 << 61],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        ),
    ) {
        assert!(SpatialAttentionBf16Geometry::new(&x, &y, &scale()).is_err());
    }
}

#[test]
fn spatial_attention_permits_estimated_scratch_and_rejects_other_auxiliary_storage() {
    let contract = spatial_attention_bf16_contract().unwrap();
    let resources = &contract.descriptor().resources;
    // The compiler tests these declarations against each real provider's
    // estimate. Both online (none) and matrix (some) scratch must be eligible;
    // neither provider may acquire binding or persistent storage this way.
    for scratch in [false, true] {
        for binding in [false, true] {
            for persistent in [false, true] {
                let allowed = resources.scratch.accepts(scratch)
                    && resources.binding.accepts(binding)
                    && resources.persistent.accepts(persistent);
                assert_eq!(allowed, !binding && !persistent);
            }
        }
    }
    assert_eq!(resources.minimum_value_alignment_bytes, 16);
    contract.descriptor().validate().unwrap();

    // Plans/providers pinned to the former no-scratch contract must receive a
    // different contract identity, rather than silently gaining permission.
    let mut previous = contract.descriptor().clone();
    previous.resources.scratch = ResourcePresenceRequirement::Forbidden;
    previous.validate().unwrap();
    assert_ne!(
        previous.fingerprint().unwrap(),
        contract.descriptor().fingerprint().unwrap()
    );
}

#[test]
fn upsample_checks_output_shape_tile_metadata_and_source_channel_bound() {
    let x = t(&[2, 4, 3, 5]);
    let y = t(&[2, 2, 6, 10]);
    let attrs = gather();
    let g = SpatialUpsampleGatherBf16Geometry::new(&x, &y, &attrs).unwrap();
    assert_eq!(g.element_count(), 240);
    assert_eq!(g.channel_offsets(), [0, 0, 1, 1]);
    assert_eq!(g.logical_arithmetic_operations(), 0);
    for (key, value) in [
        ("scale_h", SemanticValue::Unsigned(0)),
        ("scale_h", SemanticValue::Unsigned(u64::MAX)),
        ("channel_stride", SemanticValue::Unsigned(u64::MAX)),
        (
            "channel_offsets",
            SemanticValue::Integers(vec![-1, 0, 1, 1]),
        ),
        ("channel_offsets", SemanticValue::Integers(vec![0, 1])),
        ("channel_offsets", SemanticValue::Integers(vec![0, 0, 2, 2])),
    ] {
        let mut a = attrs.clone();
        a.insert(AttributeId::new(key).unwrap(), value);
        assert!(SpatialUpsampleGatherBf16Geometry::new(&x, &y, &a).is_err());
    }
    assert!(SpatialUpsampleGatherBf16Geometry::new(&x, &t(&[2, 2, 10, 6]), &attrs).is_err());
    for c in [
        channel_l2_norm_image_bf16_contract().unwrap(),
        spatial_attention_bf16_contract().unwrap(),
        spatial_upsample_gather_bf16_contract().unwrap(),
    ] {
        assert!(c
            .descriptor
            .outputs
            .iter()
            .all(|x| x.alias() == &AliasPolicy::NoAlias));
    }
}
