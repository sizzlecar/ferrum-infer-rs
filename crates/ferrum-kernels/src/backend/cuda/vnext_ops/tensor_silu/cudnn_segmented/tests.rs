use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, CanonicalRational, ProgramValueId, ResolvedTensorLayout,
    ResolvedTensorSpec, ResolvedValueStorage, ResourceId, TensorAccess,
};

fn ports(rows: u64, columns: u64) -> Vec<ResolvedValueBinding> {
    (0..4)
        .map(|port| {
            let tensor = ResolvedTensorSpec::new(
                vec![rows, columns],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap();
            let name = format!("segmented.{port}");
            let storage = ResolvedValueStorage::single(
                ResourceId::new(format!("resource.{name}")).unwrap(),
                0,
                tensor.minimum_storage_bytes().unwrap(),
                ElementType::Bf16,
            )
            .unwrap();
            ResolvedValueBinding::new(
                ProgramValueId::new(format!("value.{name}")).unwrap(),
                if port < 3 {
                    ResolvedValueRole::Input
                } else {
                    ResolvedValueRole::Output
                },
                if port < 3 { port } else { 0 },
                tensor,
                if port < 3 {
                    TensorAccess::Read
                } else {
                    TensorAccess::Write
                },
                AliasPolicy::NoAlias,
                BufferUsage::Activations,
                None,
                storage,
            )
            .unwrap()
        })
        .collect()
}
fn attrs(dim: u64, lengths: &[i64]) -> BTreeMap<AttributeId, SemanticValue> {
    [
        ("head_dim", SemanticValue::Unsigned(dim)),
        (
            "scale",
            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
        ),
        ("segment_lengths", SemanticValue::Integers(lengths.to_vec())),
    ]
    .into_iter()
    .map(|(k, v)| (AttributeId::new(k).unwrap(), v))
    .collect()
}

#[test]
fn d72_segments_are_independent_checked_subranges_with_no_prefix_keys() {
    let s = Schedule::new(&ports(5, 144), &attrs(72, &[2, 3])).unwrap();
    assert_eq!(s.geometry.output_bytes, 1440);
    assert_eq!(s.segments.len(), 2);
    assert_eq!((s.segments[0].offset_bytes, s.segments[0].bytes), (0, 576));
    assert_eq!(
        (s.segments[1].offset_bytes, s.segments[1].bytes),
        (576, 864)
    );
    for p in &s.segments {
        assert_eq!(p.signature.head_dim, 72);
        assert_eq!(p.signature.batch, 1);
        assert_eq!(p.signature.query_length, i64::from(p.rows));
        assert_eq!(p.signature.key_value_length, i64::from(p.rows));
        assert_eq!(p.signature.key_value_batch_stride * 2, p.bytes as i64);
        p.signature.validate().unwrap();
        assert_eq!(p.offset_bytes % 16, 0);
        assert!(p.offset_bytes + p.bytes <= s.geometry.output_bytes);
        attention::NativeMask::isolated_sequence(p.rows).unwrap();
    }
    assert_eq!(s.operations, 2044 + 4818);
}

#[test]
fn direct_segment_scratch_contains_only_disjoint_lengths_and_exact_workspace() {
    for batches in [1, 16] {
        for workspace in [0, 256, 256 << 20] {
            let s = ScratchLayout::without_payload(batches, workspace).unwrap();
            assert_eq!(s.value_offset, 0);
            assert_eq!(s.query_lengths_offset, 0);
            assert!(s.query_lengths_offset + u64::from(batches) * 4 <= s.key_lengths_offset);
            assert!(s.key_lengths_offset + u64::from(batches) * 4 <= s.workspace_offset);
            assert_eq!(s.workspace_offset % SCRATCH_ALIGNMENT, 0);
            assert_eq!(s.workspace_bytes, workspace);
            assert_eq!(s.workspace_offset + workspace, s.bytes);
            let packed = ScratchLayout::new(batches, 9359, 56 * 128, workspace).unwrap();
            assert_eq!(packed.bytes - s.bytes, packed.query_lengths_offset);
        }
    }
    assert!(ScratchLayout::without_payload(0, 0).is_err());
    assert!(ScratchLayout::without_payload(17, 0).is_err());
    assert!(ScratchLayout::without_payload(1, u64::MAX).is_err());
}

#[test]
fn equal_length_segments_keep_independent_native_batch_strides() {
    let s = Schedule::new(&ports(4, 144), &attrs(72, &[2, 2])).unwrap();
    assert_eq!(s.segments.len(), 1);
    let group = &s.segments[0];
    assert_eq!(group.signature.batch, 2);
    assert_eq!(group.signature.query_length, 2);
    assert_eq!(group.signature.query_batch_stride, 2 * 144);
    assert_eq!(group.signature.key_value_batch_stride, 2 * 144);
    assert_eq!(group.signature.output_batch_stride, 2 * 144);
    assert_eq!(group.bytes, 4 * 144 * 2);
    assert!(Schedule::new(&ports(4, 64), &attrs(32, &[2, 2])).is_err());
    assert!(Schedule::new(&ports(4, 144), &attrs(72, &[3])).is_err());
    assert!(Schedule::new(&ports(4, 144), &attrs(72, &[0, 4])).is_err());
    assert!(attention::NativeMask::isolated_sequence(0).is_err());
    assert!(logical_work(u64::MAX, 16, 72).is_err());
}

#[test]
fn d64_decoder_segments_keep_batches_disjoint_and_share_the_native_plan() {
    let s = Schedule::new(&ports(232, 2048), &attrs(64, &[116, 116])).unwrap();
    assert_eq!(s.segments.len(), 1);
    assert_eq!(s.segments[0].signature.heads, 32);
    assert_eq!(s.segments[0].signature.head_dim, 64);
    assert_eq!(s.segments[0].signature.batch, 2);
    assert_eq!(s.segments[0].signature.scale_bits, 0.125_f32.to_bits());
    assert_eq!(s.segments[0].offset_bytes, 0);
    assert_eq!(s.segments[0].signature.query_batch_stride, 116 * 2048);
    assert_eq!(s.segments[0].bytes, 232 * 2048 * 2);
    assert_eq!(s.segments[0].bytes, s.geometry.output_bytes);
    for segment in &s.segments {
        segment.signature.validate().unwrap();
    }
}

#[test]
fn ragged_runs_and_native_batch_limit_preserve_every_original_segment() {
    let maximum = CudnnSdpaSignature::MAX_BATCH as usize;
    let mut lengths = vec![2i64; maximum + 3];
    lengths.extend([3, 3, 2, 2]);
    for dim in [64, 72, 128] {
        let columns = dim * 2;
        let rows = lengths.iter().sum::<i64>() as u64;
        let s = Schedule::new(&ports(rows, columns), &attrs(dim, &lengths)).unwrap();
        assert_eq!(s.segments.len(), 4);
        let mut reconstructed = Vec::new();
        for segment in &s.segments {
            segment.signature.validate().unwrap();
            assert_eq!(segment.signature.query_length, i64::from(segment.rows));
            assert_eq!(segment.signature.key_value_length, i64::from(segment.rows));
            let stride = segment.signature.query_batch_stride as u64 * 2;
            assert_eq!(segment.bytes, stride * segment.signature.batch as u64);
            for batch in 0..segment.signature.batch as u64 {
                reconstructed.push((segment.offset_bytes + batch * stride, segment.rows as u64));
            }
            let scratch =
                ScratchLayout::without_payload(segment.signature.batch as u32, 0).unwrap();
            assert!(
                scratch.query_lengths_offset + segment.signature.batch as u64 * 4
                    <= scratch.key_lengths_offset
            );
        }
        assert_eq!(
            reconstructed,
            s.geometry
                .starts
                .iter()
                .zip(&s.geometry.lengths)
                .map(|(&start, &length)| (start * columns * 2, length))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            s.operations,
            lengths
                .iter()
                .map(|&n| logical_work(n as u64, 2, dim).unwrap())
                .sum::<u64>()
        );
    }
}
