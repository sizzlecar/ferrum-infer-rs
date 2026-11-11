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
fn equal_length_segments_share_native_shape_but_keep_distinct_offsets() {
    let s = Schedule::new(&ports(4, 144), &attrs(72, &[2, 2])).unwrap();
    assert_eq!(s.segments[0].signature, s.segments[1].signature);
    assert_ne!(s.segments[0].offset_bytes, s.segments[1].offset_bytes);
    assert!(Schedule::new(&ports(4, 128), &attrs(64, &[2, 2])).is_err());
    assert!(Schedule::new(&ports(4, 144), &attrs(72, &[3])).is_err());
    assert!(Schedule::new(&ports(4, 144), &attrs(72, &[0, 4])).is_err());
    assert!(attention::NativeMask::isolated_sequence(0).is_err());
    assert!(logical_work(u64::MAX, 16, 72).is_err());
}
