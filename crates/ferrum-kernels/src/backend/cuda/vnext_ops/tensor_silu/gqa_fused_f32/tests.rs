use super::*;
use ferrum_runtime::{CanonicalRational, ResolvedTensorLayout};
fn geometry_for(
    batch: usize,
    sequence: u64,
    width: u64,
    qheads: u64,
    kvheads: u64,
    segments: &[i64],
    bidirectional: &[i64],
    valid: i64,
) -> StructuredGqaAttentionF32Geometry {
    let spec = |heads| {
        ResolvedTensorSpec::new(
            vec![batch as u64 * sequence, heads * width],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    };
    let mut attrs = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(sequence),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(width),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
        ),
    ]);
    for (name, values) in [
        ("segment_lengths", segments.to_vec()),
        ("segment_bidirectional", bidirectional.to_vec()),
        ("key_run_counts", vec![1; batch]),
        ("key_run_lengths", vec![sequence as i64; batch]),
        ("key_run_valid", vec![valid; batch]),
    ] {
        attrs.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values),
        );
    }
    StructuredGqaAttentionF32Geometry::new(
        &spec(qheads),
        &spec(kvheads),
        &spec(kvheads),
        &spec(qheads),
        &attrs,
    )
    .unwrap()
}
#[test]
fn fused_route_retains_exact_bshd_geometry() {
    for batch in [1, 3, 17] {
        for sequence in [1, 7, 65] {
            for width in [64, 72] {
                let g = geometry_for(batch, sequence, width, 4, 4, &[sequence as i64], &[1], 1);
                let shape = Shape::new(g).unwrap();
                let request = shape.native_request.unwrap();
                assert_eq!(u64::from(request.batches), batch as u64);
                assert_eq!(u64::from(request.sequence), sequence);
                assert_eq!(u64::from(request.heads), 4);
                assert_eq!(u64::from(request.dimension), width);
                assert_eq!(request.elements().unwrap() * 4, shape.query_bytes);
                assert_eq!(shape.query_bytes, shape.key_value_bytes);
                assert_eq!(request.scale.to_bits(), (1.0f32 / 8.0).to_bits());
            }
        }
    }
}
#[test]
fn causal_segmented_padding_gqa_and_other_widths_select_strict_before_enqueue() {
    for g in [
        geometry_for(2, 7, 64, 4, 4, &[7], &[0], 1),
        geometry_for(2, 7, 64, 4, 4, &[3, 4], &[1, 1], 1),
        geometry_for(2, 7, 72, 4, 4, &[7], &[1], 0),
        geometry_for(2, 7, 72, 4, 2, &[7], &[1], 1),
        geometry_for(2, 7, 96, 4, 4, &[7], &[1], 1),
    ] {
        let shape = Shape::new(g).unwrap();
        assert!(shape.native_request.is_none());
        assert_eq!(shape.masks.iter().map(|tile| tile.batches).sum::<u64>(), 2);
        assert_eq!(
            shape.launch_config_for_groups(shape.groups).block_dim.0,
            shape.width
        );
    }
}
#[test]
fn opaque_native_policy_requires_explicit_finite_identity() {
    let valid = CudaCutlassFusedF32Config {
        opaque_memory_bytes: 64 << 20,
        opaque_memory_profile: "cutlass_fused_f32_v1".into(),
    };
    valid.validate().unwrap();
    for profile in ["".to_owned(), "\n".to_owned(), "x".repeat(257)] {
        assert!(CudaCutlassFusedF32Config {
            opaque_memory_profile: profile,
            ..valid.clone()
        }
        .validate()
        .is_err());
    }
    assert!(CudaCutlassFusedF32Config {
        opaque_memory_bytes: 0,
        ..valid
    }
    .validate()
    .is_err());
}
