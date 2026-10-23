use super::*;
use ferrum_runtime::{CanonicalRational, ResolvedTensorLayout, ResolvedTensorSpec};

fn shape(batch: u64, channels: u64, height: u64, width: u64) -> Result<Shape, String> {
    let tensor = |dimensions| {
        ResolvedTensorSpec::new(
            dimensions,
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .map_err(|e| e.to_string())
    };
    Shape::new(
        SpatialAttentionBf16Geometry::new(
            &tensor(vec![batch, 3 * channels, height, width])?,
            &tensor(vec![batch, channels, height, width])?,
            &BTreeMap::from([(
                AttributeId::new("scale").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(1, 32).unwrap()),
            )]),
        )
        .map_err(|e| e.to_string())?,
    )
}

#[test]
fn wide_single_head_uses_linear_bounded_workspace_and_sequential_batches() {
    let one = shape(1, 1152, 128, 128).unwrap();
    let two = shape(2, 1152, 128, 128).unwrap();
    assert_eq!((one.channels, one.spatial, one.tile), (1152, 16384, 256));
    assert_eq!(one.elements_per_batch, 1152 * 16384);
    assert_eq!(one.score_offset, 75_497_472);
    assert_eq!(one.result_offset, 92_274_688);
    assert_eq!(one.scratch_bytes, 93_454_336);
    assert_eq!(two.scratch_bytes, one.scratch_bytes);
    assert_eq!(two.input_bytes, 2 * one.input_bytes);
    assert_eq!(two.output_bytes, 2 * one.output_bytes);
    assert_eq!(two.operations, 2 * one.operations);
    assert_eq!(two.launches(), 2 * one.launches());
    // This is workspace for 256 queries over ALL keys, not a square S² buffer.
    assert!(one.result_offset - one.score_offset < 4 * 16384_u64.pow(2));
}

#[test]
fn odd_channels_and_query_tail_keep_all_ranges_aligned_and_disjoint() {
    for (channels, spatial) in [(1, 1), (3, 17), (3, 257), (1152, 513), (2048, 259)] {
        let s = shape(2, channels, 1, spatial).unwrap();
        assert_eq!(u64::from(s.tile), spatial.min(layout::QUERY_TILE));
        let ranges = [
            (0, 4 * channels * spatial),
            (s.score_offset, 4 * spatial * u64::from(s.tile)),
            (s.result_offset, 4 * channels * u64::from(s.tile)),
        ];
        for (i, &(offset, bytes)) in ranges.iter().enumerate() {
            assert_eq!(offset % SCRATCH_ALIGNMENT, 0);
            assert!(offset + bytes <= s.scratch_bytes);
            if let Some(&(next, _)) = ranges.get(i + 1) {
                assert!(offset + bytes <= next);
                assert!(next - offset - bytes < SCRATCH_ALIGNMENT);
            }
        }
        assert_eq!(s.scratch_bytes % SCRATCH_ALIGNMENT, 0);
        assert_eq!(s.launches(), 2 * (1 + 4 * spatial.div_ceil(256)));
    }
}

#[test]
fn unsupported_blas_extents_and_invalid_single_head_geometry_fail_closed() {
    assert!(shape(1, 3, 1, i32::MAX as u64 + 1).is_err());
    assert!(shape(1, 2049, 1, 2).is_err());
    assert!(shape(0, 3, 1, 2).is_err());
    assert!(shape(1, 3, 0, 2).is_err());
    assert!(shape(u64::MAX / 8, 2048, 16384, 16384).is_err());
}
