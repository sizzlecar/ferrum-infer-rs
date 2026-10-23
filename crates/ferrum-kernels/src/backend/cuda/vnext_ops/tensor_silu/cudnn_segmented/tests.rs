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
fn scratch_reuses_the_largest_segment_with_exact_workspace_and_two_length_arrays() {
    let s = Schedule::new(&ports(5, 144), &attrs(72, &[2, 3])).unwrap();
    let layouts: Vec<_> = s
        .segments
        .iter()
        .map(|p| ScratchLayout::new(1, p.rows, 144, 256).unwrap())
        .collect();
    let peak = layouts.iter().map(|s| s.bytes).max().unwrap();
    assert!(peak < layouts.iter().map(|s| s.bytes).sum::<u64>());
    for (s, segment) in layouts.iter().zip(&s.segments) {
        assert!(segment.bytes <= s.value_offset);
        assert!(s.value_offset + segment.bytes <= s.query_lengths_offset);
        assert!(s.query_lengths_offset + 4 <= s.key_lengths_offset);
        assert!(s.key_lengths_offset + 4 <= s.workspace_offset);
        assert_eq!(s.workspace_bytes, 256);
        assert_eq!(s.workspace_offset + 256, s.bytes);
        assert!(s.bytes <= peak);
    }
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
