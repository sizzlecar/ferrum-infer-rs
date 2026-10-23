use super::*;
use ferrum_runtime::{CanonicalRational, ResolvedTensorLayout, ResolvedTensorSpec};
fn geometry(
    b: u64,
    q: u64,
    k: u64,
    d: u64,
    counts: &[i64],
    lengths: &[i64],
    flags: &[i64],
) -> CrossAttentionBf16Geometry {
    let t = |r| {
        ResolvedTensorSpec::new(
            vec![r, 2 * d],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    };
    let a = [
        ("batch", SemanticValue::Unsigned(b)),
        ("head_dim", SemanticValue::Unsigned(d)),
        (
            "scale",
            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
        ),
        ("key_run_counts", SemanticValue::Integers(counts.to_vec())),
        ("key_run_lengths", SemanticValue::Integers(lengths.to_vec())),
        ("key_run_valid", SemanticValue::Integers(flags.to_vec())),
    ]
    .into_iter()
    .map(|(k, v)| (AttributeId::new(k).unwrap(), v))
    .collect();
    CrossAttentionBf16Geometry::new(&t(b * q), &t(b * k), &t(b * k), &t(b * q), &a).unwrap()
}
#[test]
fn rectangular_nonempty_batches_share_one_plan_with_independent_q_and_kv_strides() {
    for d in [72, 128] {
        let s = Schedule::new(geometry(
            2,
            3,
            5,
            d,
            &[3, 2],
            &[1, 1, 3, 2, 3],
            &[1, 0, 1, 0, 1],
        ))
        .unwrap();
        assert_eq!(s.stages.len(), 1);
        assert_eq!(s.stages[0].method, Method::Sdpa { packed_rows: 4 });
        let native = s.signature(s.stages[0]).unwrap();
        assert_eq!(
            (
                native.batch,
                native.query_length,
                native.key_value_length,
                native.head_dim
            ),
            (2, 3, 4, d as i64)
        );
        assert_eq!(native.query_batch_stride, 3 * 2 * d as i64);
        assert_eq!(native.key_value_batch_stride, 4 * 2 * d as i64);
        assert_eq!(native.output_batch_stride, native.query_batch_stride);
        native.validate().unwrap();
        let layout = ScratchLayout::new(2, 4, 2 * d, 512).unwrap();
        assert!(layout.value_offset >= 2 * 4 * 2 * d * 2);
        assert!(layout.query_lengths_offset >= layout.value_offset + 2 * 4 * 2 * d * 2);
        assert_eq!(layout.workspace_offset + 512, layout.bytes);
    }
}
#[test]
fn mixed_empty_keys_split_batches_and_all_empty_never_prepare_native_sdpa() {
    let s = Schedule::new(geometry(2, 7, 3, 128, &[1, 1], &[3, 3], &[0, 1])).unwrap();
    assert_eq!(s.stages.len(), 2);
    assert_eq!(s.stages[0].method, Method::Empty);
    assert_eq!(
        (
            s.stages[1].first_batch,
            s.stages[1].batches,
            s.stages[1].query_count
        ),
        (1, 1, 7)
    );
    assert_eq!(s.signature(s.stages[1]).unwrap().query_length, 7);
    assert!(s.signature(s.stages[0]).is_err());
    let empty = Schedule::new(geometry(2, 7, 3, 128, &[1, 1], &[3, 3], &[0, 0])).unwrap();
    assert_eq!(empty.stages.len(), 1);
    assert_eq!(empty.stages[0].method, Method::Empty);
    assert_eq!(empty.geometry.logical_arithmetic_operations(), 0);
    assert!(Schedule::new(geometry(1, 2, 3, 8, &[1], &[3], &[1])).is_err());
}
#[test]
fn prepared_identity_distinguishes_empty_vendor_and_batch_range() {
    let s = Schedule::new(geometry(2, 7, 3, 128, &[1, 1], &[3, 3], &[0, 1])).unwrap();
    let native = "1".repeat(64);
    let a =
        prepared_stages_fingerprint([(s.stages[0], None), (s.stages[1], Some(native.as_str()))])
            .unwrap();
    let b =
        prepared_stages_fingerprint([(s.stages[1], Some(native.as_str())), (s.stages[0], None)])
            .unwrap();
    assert_ne!(a, b);
    assert!(prepared_stages_fingerprint([(s.stages[0], Some(native.as_str()))]).is_err());
    assert!(prepared_stages_fingerprint([(s.stages[1], None)]).is_err());
}
