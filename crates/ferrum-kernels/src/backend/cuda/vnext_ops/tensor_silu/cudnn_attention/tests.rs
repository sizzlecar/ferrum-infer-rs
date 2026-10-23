//! Hardware-independent packing/scheduling boundaries. These do not qualify
//! vendor numerics or prove native execution; the managed CUDA fixtures do that.

use super::*;
use ferrum_runtime::{CanonicalRational, ResolvedTensorLayout, ResolvedTensorSpec};

fn attrs(
    sequence: u64,
    dim: u64,
    segments: &[i64],
    bidir: &[i64],
    counts: &[i64],
    runs: &[i64],
    valid: &[i64],
) -> BTreeMap<AttributeId, SemanticValue> {
    let mut attributes = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(sequence),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(dim),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 2).unwrap()),
        ),
    ]);
    for (name, list) in [
        ("segment_lengths", segments),
        ("segment_bidirectional", bidir),
        ("key_run_counts", counts),
        ("key_run_lengths", runs),
        ("key_run_valid", valid),
    ] {
        attributes.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(list.to_vec()),
        );
    }
    attributes
}

fn schedule(
    batch: u64,
    sequence: u64,
    heads: u64,
    dim: u64,
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Schedule, String> {
    let t = ResolvedTensorSpec::new(
        vec![batch * sequence, heads * dim],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    Schedule::new(
        StructuredAttentionBf16Geometry::new(&t, &t, &t, &t, attributes)
            .map_err(|e| e.to_string())?,
    )
}

#[test]
fn adjacent_bidir_segments_keep_distinct_prefixes_and_one_batched_plan_each() {
    let a = attrs(
        10,
        128,
        &[2, 3, 2, 3],
        &[0, 1, 1, 1],
        &[3, 5],
        &[3, 1, 6, 3, 1, 4, 1, 1],
        &[1, 0, 1, 1, 0, 1, 0, 1],
    );
    let s = schedule(2, 10, 2, 128, &a).unwrap();
    assert_eq!(
        s.stages,
        vec![
            Stage {
                first_query: 0,
                query_count: 2,
                first_batch: 0,
                batches: 2,
                method: Method::Causal
            },
            Stage {
                first_query: 2,
                query_count: 3,
                first_batch: 0,
                batches: 2,
                method: Method::Sdpa { packed_rows: 4 }
            },
            Stage {
                first_query: 5,
                query_count: 2,
                first_batch: 0,
                batches: 2,
                method: Method::Sdpa { packed_rows: 6 }
            },
            Stage {
                first_query: 7,
                query_count: 3,
                first_batch: 0,
                batches: 2,
                method: Method::Sdpa { packed_rows: 9 }
            },
        ]
    );
    assert_eq!(layout::valid_prefix(s.geometry.mask(), 0, 10).unwrap(), 9);
    assert_eq!(layout::valid_prefix(s.geometry.mask(), 1, 10).unwrap(), 8);
    let signature = s.signature(s.stages[3]).unwrap();
    assert_eq!(
        (
            signature.batch,
            signature.heads,
            signature.query_length,
            signature.key_value_length
        ),
        (2, 2, 3, 9)
    );
    assert_eq!(
        (
            signature.query_batch_stride,
            signature.output_batch_stride,
            signature.key_value_batch_stride
        ),
        (2560, 2560, 2304)
    );
    // Query padding is deliberately retained: query 3 and batch-1 query 8
    // each occur exactly once although neither is a valid key.
    for batch in 0..2 {
        for query in 0..10 {
            assert_eq!(
                s.stages
                    .iter()
                    .filter(|stage| batch >= stage.first_batch
                        && batch < stage.first_batch + stage.batches
                        && query >= stage.first_query
                        && query < stage.first_query + stage.query_count)
                    .count(),
                1
            );
        }
    }
}

#[test]
fn empty_batch_is_zero_only_and_later_nonempty_segments_rejoin_batched_path() {
    let a = attrs(6, 128, &[2, 4], &[1, 1], &[2, 1], &[2, 4, 6], &[0, 1, 1]);
    let s = schedule(2, 6, 1, 128, &a).unwrap();
    assert_eq!(
        s.stages,
        vec![
            Stage {
                first_query: 0,
                query_count: 2,
                first_batch: 0,
                batches: 1,
                method: Method::Empty
            },
            Stage {
                first_query: 0,
                query_count: 2,
                first_batch: 1,
                batches: 1,
                method: Method::Sdpa { packed_rows: 2 }
            },
            Stage {
                first_query: 2,
                query_count: 4,
                first_batch: 0,
                batches: 2,
                method: Method::Sdpa { packed_rows: 6 }
            },
        ]
    );
    assert!(s.signature(s.stages[0]).is_err());
    assert_eq!(layout::valid_prefix(s.geometry.mask(), 0, 6).unwrap(), 4);
    let a = attrs(6, 128, &[6], &[1], &[1, 1], &[6, 6], &[0, 0]);
    let empty = schedule(2, 6, 1, 128, &a).unwrap();
    assert_eq!(empty.stages.len(), 1);
    assert_eq!(empty.stages[0].method, Method::Empty);
}

#[test]
fn valid_prefix_counts_match_independent_small_mask_enumeration() {
    // Every run pattern, all prefix endpoints, and two different batch masks.
    for pattern in 0u32..16 {
        let flags = (0..4)
            .map(|i| i64::from((pattern >> i) & 1))
            .chain((0..4).map(|i| i64::from(((pattern ^ 15) >> i) & 1)))
            .collect::<Vec<_>>();
        let mut counts = Vec::new();
        let mut canonical_lengths = Vec::new();
        let mut canonical_flags = Vec::new();
        for batch in 0..2 {
            let start = canonical_flags.len();
            for (&length, &flag) in [1i64, 2, 3, 2].iter().zip(&flags[batch * 4..batch * 4 + 4]) {
                if canonical_flags.len() > start && canonical_flags.last() == Some(&flag) {
                    *canonical_lengths.last_mut().unwrap() += length;
                } else {
                    canonical_lengths.push(length);
                    canonical_flags.push(flag);
                }
            }
            counts.push((canonical_flags.len() - start) as i64);
        }
        let a = attrs(
            8,
            128,
            &[8],
            &[1],
            &counts,
            &canonical_lengths,
            &canonical_flags,
        );
        let s = schedule(2, 8, 1, 128, &a).unwrap();
        for batch in 0..2 {
            for end in 1..=8 {
                let original = [
                    flags[batch * 4],
                    flags[batch * 4 + 1],
                    flags[batch * 4 + 1],
                    flags[batch * 4 + 2],
                    flags[batch * 4 + 2],
                    flags[batch * 4 + 2],
                    flags[batch * 4 + 3],
                    flags[batch * 4 + 3],
                ];
                let count = original[..end as usize]
                    .iter()
                    .filter(|&&flag| flag != 0)
                    .count() as u32;
                assert_eq!(
                    layout::valid_prefix(s.geometry.mask(), batch, end).unwrap(),
                    count
                );
            }
        }
    }
}

#[test]
fn scratch_has_disjoint_aligned_payload_lengths_workspace_and_reuses_peak() {
    let layout = ScratchLayout::new(2, 9, 256, 513).unwrap();
    assert_eq!(layout.value_offset, 9216);
    assert_eq!(layout.query_lengths_offset, 18432);
    assert_eq!(layout.key_lengths_offset, 18688);
    assert_eq!(layout.workspace_offset, 18944);
    assert_eq!(layout.bytes, 19457);
    assert!(layout.value_offset >= 2 * 9 * 256 * 2);
    assert!(layout.query_lengths_offset >= layout.value_offset + 2 * 9 * 256 * 2);
    for offset in [
        layout.value_offset,
        layout.query_lengths_offset,
        layout.key_lengths_offset,
        layout.workspace_offset,
    ] {
        assert_eq!(offset % SCRATCH_ALIGNMENT, 0);
    }
    let earlier = ScratchLayout::new(2, 4, 256, 1024).unwrap();
    assert!(earlier.bytes < layout.bytes);
    assert_eq!(
        [earlier.bytes, layout.bytes].into_iter().max(),
        Some(layout.bytes)
    );
    assert!(ScratchLayout::new(0, 1, 128, 0).is_err());
    assert!(ScratchLayout::new(17, 1, 128, 0).is_err());
    assert!(ScratchLayout::new(1, 0, 128, 0).is_err());
    assert!(ScratchLayout::new(16, u32::MAX, u64::MAX, 0).is_err());
    assert!(ScratchLayout::new(1, 1, 128, u64::MAX).is_err());
}

#[test]
fn profile_rejects_unsupported_head_and_signed_sequence_lengths_before_planning() {
    let a = attrs(2, 64, &[2], &[1], &[1], &[2], &[1]);
    assert!(schedule(1, 2, 1, 64, &a).unwrap_err().contains("head_dim"));
    let n = i32::MAX as u64 + 1;
    let a = attrs(n, 128, &[n as i64], &[1], &[1], &[n as i64], &[0]);
    assert!(schedule(1, n, 1, 128, &a).unwrap_err().contains("i32"));
    // Full native parameter lists, including the mask, remain below 4 KiB.
    assert!(std::mem::size_of::<attention::NativeMask>() + 6 * 8 + 7 * 4 < 4096);
}

#[test]
fn prepared_identity_distinguishes_actual_engine_range_order_and_fallback() {
    let sdpa = Stage {
        first_query: 2,
        query_count: 4,
        first_batch: 0,
        batches: 2,
        method: Method::Sdpa { packed_rows: 5 },
    };
    let causal = Stage {
        first_query: 0,
        query_count: 2,
        first_batch: 0,
        batches: 2,
        method: Method::Causal,
    };
    let first = "a".repeat(64);
    let second = "b".repeat(64);
    let fingerprint =
        prepared_stages_fingerprint([(causal, None), (sdpa, Some(first.as_str()))]).unwrap();
    assert_eq!(
        fingerprint,
        prepared_stages_fingerprint([(causal, None), (sdpa, Some(first.as_str()))]).unwrap()
    );
    assert_ne!(
        fingerprint,
        prepared_stages_fingerprint([(causal, None), (sdpa, Some(second.as_str()))]).unwrap()
    );
    assert_ne!(
        fingerprint,
        prepared_stages_fingerprint([(sdpa, Some(first.as_str())), (causal, None)]).unwrap()
    );
    assert_ne!(
        fingerprint,
        prepared_stages_fingerprint([
            (
                Stage {
                    method: Method::Empty,
                    ..causal
                },
                None
            ),
            (sdpa, Some(first.as_str()))
        ])
        .unwrap()
    );
    assert_ne!(
        fingerprint,
        prepared_stages_fingerprint([
            (causal, None),
            (
                Stage {
                    first_query: 3,
                    ..sdpa
                },
                Some(first.as_str())
            )
        ])
        .unwrap()
    );
    assert!(prepared_stages_fingerprint([(sdpa, None)]).is_err());
    assert!(prepared_stages_fingerprint([(causal, Some(first.as_str()))]).is_err());
    assert!(prepared_stages_fingerprint([(sdpa, Some("not-a-fingerprint"))]).is_err());
}
