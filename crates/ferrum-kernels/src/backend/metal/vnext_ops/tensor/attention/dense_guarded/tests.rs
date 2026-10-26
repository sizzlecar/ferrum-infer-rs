//! Host scheduling and contract checks; these do not qualify GPU arithmetic.
use super::*;
use ferrum_runtime::{
    CanonicalRational, OperationContract, ResolvedTensorSpec, ResourcePresenceRequirement,
    StructuredAttentionBf16Geometry,
};

#[path = "probe.rs"]
mod probe;

fn geometry(query: u64, key: u64, start: u64) -> GuardGeometry {
    GuardGeometry {
        batch_heads: 6,
        query,
        key,
        query_heads: 3,
        key_value_heads: 3,
        head_dim: 128,
        scale: 0.125,
        all_keys_valid: true,
        dense_start: Some(start),
    }
}

fn schedule(query: u64, key: u64, start: u64) -> Schedule {
    Schedule::new(geometry(query, key, start)).unwrap().unwrap()
}

fn checked_mask(
    segments: &[(i64, bool)],
    runs: &[Vec<(i64, bool)>],
) -> Result<StructuredAttentionBf16Geometry, VNextError> {
    let sequence = segments.iter().map(|s| s.0 as u64).sum::<u64>();
    let tensor = ResolvedTensorSpec::new(
        vec![sequence * runs.len() as u64, 128],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )?;
    let mut attributes = BTreeMap::from([
        (
            AttributeId::new("sequence_length")?,
            SemanticValue::Unsigned(sequence),
        ),
        (AttributeId::new("head_dim")?, SemanticValue::Unsigned(128)),
        (
            AttributeId::new("scale")?,
            SemanticValue::Rational(CanonicalRational::new(1, 8)?),
        ),
    ]);
    for (name, values) in [
        ("segment_lengths", segments.iter().map(|s| s.0).collect()),
        (
            "segment_bidirectional",
            segments.iter().map(|s| i64::from(s.1)).collect(),
        ),
        (
            "key_run_counts",
            runs.iter().map(|r| r.len() as i64).collect(),
        ),
        (
            "key_run_lengths",
            runs.iter().flatten().map(|r| r.0).collect(),
        ),
        (
            "key_run_valid",
            runs.iter().flatten().map(|r| i64::from(r.1)).collect(),
        ),
    ] {
        attributes.insert(AttributeId::new(name)?, SemanticValue::Integers(values));
    }
    StructuredAttentionBf16Geometry::new(&tensor, &tensor, &tensor, &tensor, &attributes)
}

#[test]
fn fallback_retains_original_query_pitch_and_full_grid_after_suffix_selection() {
    let plan = schedule(97, 17, 17);
    assert_eq!(plan.strict.query_sequence, 97);
    assert_eq!(plan.strict.key_sequence, 17);
    assert_eq!(plan.strict.query_tiles, 4);
    assert_eq!(plan.strict.jobs, 24);
    assert_eq!(plan.gate.original_query_tiles, plan.strict.query_tiles);
    assert_eq!(plan.gate.first_fast_tile, 1);
    assert_eq!(plan.dense.query, 65);
    assert_eq!(plan.dense.query_tiles, 3);
    assert_eq!(plan.flag_bytes(), 24 * 4);

    // A capped physical grid must still visit every original logical job.
    let mut g = geometry(4097, 17, 1);
    g.query_heads = 257;
    g.key_value_heads = 257;
    g.batch_heads = 514;
    let plan = Schedule::new(g).unwrap().unwrap();
    let mut visits = vec![0_u8; (2 * 257 * 129) as usize];
    assert!(plan.strict.grid_groups < plan.strict.jobs);
    for group in 0..plan.strict.grid_groups {
        for job in (group..plan.strict.jobs).step_by(plan.strict.grid_groups as usize) {
            visits[job as usize] += 1;
        }
    }
    assert!(visits.iter().all(|&count| count == 1));
}

#[test]
fn prefix_rejected_heads_and_rejected_dense_jobs_cover_every_original_row_once() {
    let plan = schedule(97, 17, 17);
    let head_is_admitted = [true, true, false, true, true, true];
    let mut flags = vec![0_u32; plan.strict.jobs as usize];
    for (head, admitted) in head_is_admitted.into_iter().enumerate() {
        if admitted {
            for tile in plan.gate.first_fast_tile..plan.gate.original_query_tiles {
                flags[head * plan.gate.original_query_tiles as usize + tile as usize] = 1;
            }
        }
    }

    // A failed same-PSO check clears only this CTA's original-grid flag.
    let rejected_head = 4;
    let rejected_local_tile = 1;
    let rejected_job = rejected_head * plan.gate.original_query_tiles
        + plan.gate.first_fast_tile
        + rejected_local_tile;
    flags[rejected_job as usize] = 0;
    assert_eq!(flags[(rejected_job - 1) as usize], 1);
    assert_eq!(flags[(rejected_job + 1) as usize], 1);

    let mut visits = vec![(0_u8, 0_u8); (6 * 97) as usize];
    for batch in 0..plan.dense.batch as u64 {
        for head in 0..plan.dense.heads as u64 {
            let batch_head = batch * plan.dense.heads as u64 + head;
            for local_tile in 0..plan.dense.query_tiles as u64 {
                let tile = plan.gate.first_fast_tile + local_tile;
                let flag = batch_head * plan.gate.original_query_tiles + tile;
                if flags[flag as usize] != 0 {
                    for query in tile * 32..((tile + 1) * 32).min(97) {
                        visits[(batch_head * 97 + query) as usize].0 += 1;
                    }
                }
            }
        }
    }
    for job in 0..plan.strict.jobs {
        if flags[job as usize] == 0 {
            let batch_head = job / plan.strict.query_tiles;
            let first = (job % plan.strict.query_tiles) * 32;
            for query in first..(first + 32).min(plan.strict.query_sequence) {
                visits[(batch_head * 97 + query) as usize].1 += 1;
            }
        }
    }
    for head in 0..6 {
        for query in 0..97 {
            let fast = query >= 32 && head != 2 && !(head == 4 && (64..96).contains(&query));
            assert_eq!(visits[head * 97 + query], (u8::from(fast), u8::from(!fast)));
        }
    }
}

#[test]
fn suffix_boundaries_keep_partial_last_tiles_and_exclude_mixed_prefix_tiles() {
    for suffix in [31, 32, 33] {
        let plan = schedule(32 + suffix, 17, 17);
        assert_eq!(plan.gate.first_fast_tile, 1);
        assert_eq!(plan.dense.query as u64, suffix);
        assert_eq!(plan.dense.query_tiles as u64, suffix.div_ceil(32));
        assert_eq!(plan.dense.aligned_query_tiles as u64, suffix / 32);
        assert_eq!(plan.dense.query_remainder as u64, suffix % 32);
    }
    for (query, start) in [(31, 0), (32, 1), (63, 33), (64, 33), (65, 65)] {
        assert!(Schedule::new(geometry(query, 17, start)).unwrap().is_none());
    }
    let tail = schedule(33, 17, 1);
    assert_eq!(tail.gate.first_fast_tile, 1);
    assert_eq!(tail.dense.query, 1);
    assert_eq!(tail.dense.query_remainder, 1);
}

#[test]
fn suffix_offsets_preserve_original_batch_strides_for_independent_query_key_lengths() {
    let plan = schedule(97, 17, 17);
    assert_eq!(std::mem::size_of::<DenseParams>(), 152);
    assert_eq!(std::mem::align_of::<DenseParams>(), 8);
    assert_eq!(plan.dense.query_strides, [97 * 3 * 128, 128, 3 * 128]);
    assert_eq!(plan.dense.output_strides, plan.dense.query_strides);
    assert_eq!(plan.dense.key_strides, [17 * 3 * 128, 128, 3 * 128]);
    assert_eq!(plan.dense.value_strides, plan.dense.key_strides);
    assert_eq!(plan.query_offset_bytes, 32 * 3 * 128 * 2);
    for batch in 0..2_i64 {
        for head in 0..3_i64 {
            for row in [0_i64, 64] {
                for dimension in [0_i64, 127] {
                    let actual = plan.query_offset_bytes as i64 / 2
                        + batch * plan.dense.query_strides[0]
                        + head * plan.dense.query_strides[1]
                        + row * plan.dense.query_strides[2]
                        + dimension;
                    let expected = ((batch * 97 + 32 + row) * 3 + head) * 128 + dimension;
                    assert_eq!(actual, expected);
                    assert!(actual < 2 * 97 * 3 * 128);
                }
            }
            let last_key = batch * plan.dense.key_strides[0]
                + head * plan.dense.key_strides[1]
                + 16 * plan.dense.key_strides[2]
                + 127;
            assert_eq!(last_key, ((batch * 17 + 16) * 3 + head) * 128 + 127);
            assert!(last_key < 2 * 17 * 3 * 128);
        }
    }
}

#[test]
fn key_tails_and_scale_boundaries_are_explicit_admission_limits() {
    for key in [1_u64, 15, 16, 17, 65535, 65536] {
        let plan = schedule(65, key, 0);
        assert_eq!(plan.dense.key_tiles as u64, key.div_ceil(16));
        assert_eq!(plan.dense.aligned_key_tiles as u64, key / 16);
        assert_eq!(plan.dense.key_remainder as u64, key % 16);
    }
    for scale in [f32::MIN_POSITIVE, 0.125, 1_048_576.0] {
        let mut g = geometry(65, 17, 0);
        g.scale = scale;
        assert!(Schedule::new(g).unwrap().is_some());
    }
    for scale in [
        0.0,
        -0.0,
        -0.125,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1_048_576.0_f32.to_bits() + 1),
    ] {
        let mut g = geometry(65, 17, 0);
        g.scale = scale;
        assert!(Schedule::new(g).unwrap().is_none());
    }
    for key in [0, 65537] {
        assert!(Schedule::new(geometry(65, key, 0)).unwrap().is_none());
    }
}

#[test]
fn unsupported_shapes_and_missing_dense_segment_keep_strict_attention() {
    let variants: [fn(&mut GuardGeometry); 8] = [
        |g| g.head_dim = 64,
        |g| g.key_value_heads = 1,
        |g| {
            g.query_heads = 0;
            g.key_value_heads = 0;
        },
        |g| g.batch_heads = 0,
        |g| g.batch_heads = 7,
        |g| g.batch_heads = 17 * g.query_heads,
        |g| g.query = i32::MAX as u64 + 1,
        |g| g.dense_start = None,
    ];
    for mutate in variants {
        let mut g = geometry(65, 17, 0);
        mutate(&mut g);
        assert!(Schedule::new(g).unwrap().is_none());
    }
}

#[test]
fn final_bidirectional_segment_selects_mask_variant_without_changing_query_pitch() {
    for (segments, expected) in [
        (vec![(65, true)], Some(0)),
        (vec![(17, false), (48, true)], Some(17)),
        (vec![(17, true), (48, true)], Some(17)),
        (vec![(17, true), (48, false)], None),
        (vec![(65, false)], None),
    ] {
        let checked = checked_mask(&segments, &[vec![(65, true)], vec![(65, true)]]).unwrap();
        let mask = NativeMask::new(checked.mask(), 65);
        assert_eq!(dense_start(&mask, 65), expected);
    }
    for invalid_batch in 0..2 {
        for bad_runs in [vec![(8, true), (1, false), (56, true)], vec![(65, false)]] {
            let mut runs = vec![vec![(65, true)], vec![(65, true)]];
            runs[invalid_batch] = bad_runs;
            let checked = checked_mask(&[(17, false), (48, true)], &runs).unwrap();
            let mut g = geometry(65, 65, 17);
            g.dense_start = dense_start(&NativeMask::new(checked.mask(), 65), 65);
            g.all_keys_valid = register128::KeyVisibility::from_mask(checked.mask(), 65)
                == register128::KeyVisibility::AllValid;
            let plan = Schedule::new(g).unwrap().unwrap();
            assert!(plan.masked_keys);
            // Key validity changes the PSO, not the physical B2 tensor pitch,
            // original fallback grid, dense suffix or admitted flag extent.
            let all_valid = schedule(65, 65, 17);
            assert!(!all_valid.masked_keys);
            assert_eq!(plan.gate.first_fast_tile, all_valid.gate.first_fast_tile);
            assert_eq!(plan.strict.jobs, all_valid.strict.jobs);
            assert_eq!(plan.flag_bytes(), all_valid.flag_bytes());
            assert_eq!(plan.query_offset_bytes, all_valid.query_offset_bytes);
            assert_eq!(plan.dense.query_strides, all_valid.dense.query_strides);
            assert_eq!(plan.dense.key_strides, all_valid.dense.key_strides);
        }
    }
    assert!(checked_mask(&[(65, true)], &[vec![(64, true)]]).is_err());
    assert!(checked_mask(&[(65, true)], &[vec![(8, true), (57, true)]]).is_err());
    let checked = checked_mask(&[(17, false), (48, true)], &[vec![(65, true)]]).unwrap();
    let mut mask = NativeMask::new(checked.mask(), 65);
    assert_eq!(dense_start(&mask, 64), None);
    mask.isolated_segments = 1;
    assert_eq!(dense_start(&mask, 65), None);
    mask.isolated_segments = 0;
    mask.segment_count = 0;
    assert_eq!(dense_start(&mask, 65), None);
    mask.segment_count = 33;
    assert_eq!(dense_start(&mask, 65), None);
}

#[test]
fn masked_schedule_keeps_tail_tiles_and_all_empty_batches_for_gpu_qualification() {
    for sequence in [32_u64, 33, 65] {
        for runs in [
            vec![vec![(1, false), (sequence as i64 - 1, true)]; 2],
            vec![
                vec![(sequence as i64, true)],
                vec![(sequence as i64, false)],
            ],
            vec![vec![(sequence as i64, false)]; 2],
        ] {
            let checked = checked_mask(&[(sequence as i64, true)], &runs).unwrap();
            let mask = NativeMask::new(checked.mask(), sequence as u32);
            let mut g = geometry(sequence, sequence, 0);
            g.all_keys_valid =
                register128::KeyVisibility::from_mask(checked.mask(), sequence as u32)
                    == register128::KeyVisibility::AllValid;
            g.dense_start = dense_start(&mask, sequence);
            let plan = Schedule::new(g).unwrap().unwrap();
            assert!(plan.masked_keys);
            assert_eq!(plan.strict.query_tiles, sequence.div_ceil(32));
            assert_eq!(plan.dense.query_tiles as u64, sequence.div_ceil(32));
            assert_eq!(plan.dense.query_remainder as u64, sequence % 32);
            // An empty batch is a real zero-output request, not a reason to
            // discard another batch's work. The GPU probe checks flags/output.
            assert_eq!(plan.strict.jobs, 6 * sequence.div_ceil(32));
        }
    }
}

#[test]
fn flag_alignment_and_tensor_extent_arithmetic_fail_closed_at_address_limits() {
    for (jobs, bytes) in [(1, 16), (4, 16), (5, 32)] {
        assert_eq!(checked_flag_bytes(jobs).unwrap(), bytes);
    }
    let maximum_bytes = isize::MAX as u64 & !15;
    let maximum_jobs = maximum_bytes / 4;
    assert_eq!(checked_flag_bytes(maximum_jobs).unwrap(), maximum_bytes);
    for jobs in [0, maximum_jobs + 1, u64::MAX / 4, u64::MAX] {
        assert!(checked_flag_bytes(jobs).is_err());
    }
    for (query, heads, batch) in [
        (i32::MAX as u64, 1_u64 << 25, 1),
        (1_u64 << 30, 1_u64 << 24, 4),
        (i32::MAX as u64, i32::MAX as u64, 1),
    ] {
        let mut g = geometry(query, 17, 0);
        g.query_heads = heads;
        g.key_value_heads = heads;
        g.batch_heads = batch * heads;
        assert!(Schedule::new(g).is_err());
    }
}

#[test]
fn fused_operations_require_explicit_selection_and_keep_original_contracts() {
    for (original, fused) in [
        (
            MetalTensorOperation::StructuredAttention,
            MetalTensorOperation::StructuredAttentionFused,
        ),
        (
            MetalTensorOperation::CrossAttention,
            MetalTensorOperation::CrossAttentionFused,
        ),
    ] {
        assert!(MetalTensorOperation::ALL.contains(&original));
        assert!(!MetalTensorOperation::ALL.contains(&fused));
        assert!(MetalTensorOperation::FUSED_DENSE.contains(&fused));
        assert_ne!(original.profile(), fused.profile());
        assert_ne!(original.operation(), fused.operation());
        assert_ne!(original.capability(), fused.capability());
        let original_contract = original.contract().unwrap();
        let fused_contract = fused.contract().unwrap();
        let old = original_contract.descriptor();
        let new = fused_contract.descriptor();
        assert_eq!(old.id.as_str(), original.operation());
        assert_eq!(new.id.as_str(), fused.operation());
        assert_ne!(old.fingerprint().unwrap(), new.fingerprint().unwrap());
        assert_eq!(
            old.resources.scratch,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(new.resources.scratch, ResourcePresenceRequirement::Optional);
        let mut restored = new.clone();
        restored.id = old.id.clone();
        restored.provider = old.provider.clone();
        restored.resources.scratch = old.resources.scratch;
        assert_eq!(&restored, old);
    }
    assert!(MetalTensorOperation::ALL
        .iter()
        .all(|operation| !MetalTensorOperation::FUSED_DENSE.contains(operation)));
    assert!(MetalTensorOperation::ALL.contains(&MetalTensorOperation::StructuredGqaAttention));
    assert!(MetalTensorOperation::ALL.contains(&MetalTensorOperation::SegmentedAttention));
}
