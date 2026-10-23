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
        assert_eq!(
            s.stages[0].method,
            Method::Sdpa {
                packed_rows: 4,
                direct_kv: false,
            }
        );
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
    assert_eq!(
        s.stages[1].method,
        Method::Sdpa {
            packed_rows: 3,
            direct_kv: false,
        }
    );
    assert!(s.signature(s.stages[0]).is_err());
    let empty = Schedule::new(geometry(2, 7, 3, 128, &[1, 1], &[3, 3], &[0, 0])).unwrap();
    assert_eq!(empty.stages.len(), 1);
    assert_eq!(empty.stages[0].method, Method::Empty);
    assert_eq!(empty.geometry.logical_arithmetic_operations(), 0);
    assert!(Schedule::new(geometry(1, 2, 3, 8, &[1], &[3], &[1])).is_err());
}

#[test]
fn all_valid_direct_kv_preserves_native_signature_and_conservative_scratch() {
    for (b, q, k, d) in [(1, 3, 5, 128), (2, 7, 3, 72), (16, 1, 9, 128)] {
        let s = Schedule::new(geometry(
            b,
            q,
            k,
            d,
            &vec![1; b as usize],
            &vec![k as i64; b as usize],
            &vec![1; b as usize],
        ))
        .unwrap();
        assert_eq!(s.stages.len(), 1);
        let direct = s.stages[0];
        assert_eq!(direct.first_batch, 0);
        assert_eq!(direct.batches, b as u32);
        assert_eq!(direct.first_query, 0);
        assert_eq!(
            direct.method,
            Method::Sdpa {
                packed_rows: k as u32,
                direct_kv: true,
            }
        );
        let packed = Stage {
            method: Method::Sdpa {
                packed_rows: k as u32,
                direct_kv: false,
            },
            ..direct
        };
        // The vendor key and its engine selection must not change. Only the
        // binding source and the payload-copy work differ.
        let signature = s.signature(direct).unwrap();
        assert_eq!(signature, s.signature(packed).unwrap());
        assert_eq!(signature.key_value_batch_stride, (k * 2 * d) as i64);
        let scratch = ScratchLayout::new(b as u32, k as u32, 2 * d, 512).unwrap();
        let payload_bytes = b * k * 2 * d * 2;
        assert!(scratch.value_offset >= payload_bytes);
        assert!(scratch.query_lengths_offset >= scratch.value_offset + payload_bytes);
        assert_eq!(scratch.workspace_offset + 512, scratch.bytes);
        // The provider implementation receipt still distinguishes the two
        // execution paths even though their native SDPA fingerprints agree.
        let native = "2".repeat(64);
        assert_ne!(
            prepared_stages_fingerprint([(direct, Some(native.as_str()))]).unwrap(),
            prepared_stages_fingerprint([(packed, Some(native.as_str()))]).unwrap()
        );
    }
}

#[test]
fn any_key_hole_keeps_all_batches_on_the_packed_path() {
    // Equal valid lengths must not be mistaken for fully visible storage.
    let equal_holes = Schedule::new(geometry(
        2,
        3,
        5,
        128,
        &[3, 3],
        &[1, 1, 3, 2, 1, 2],
        &[1, 0, 1, 1, 0, 1],
    ))
    .unwrap();
    assert_eq!(
        equal_holes.stages[0].method,
        Method::Sdpa {
            packed_rows: 4,
            direct_kv: false,
        }
    );
    // Even when max(valid length) equals the physical KV length, another
    // sample's hole still needs packing and its shorter seq_len_kv.
    let one_hole = Schedule::new(geometry(
        2,
        3,
        5,
        128,
        &[1, 3],
        &[5, 1, 1, 3],
        &[1, 1, 0, 1],
    ))
    .unwrap();
    assert_eq!(
        one_hole.stages[0].method,
        Method::Sdpa {
            packed_rows: 5,
            direct_kv: false,
        }
    );
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
