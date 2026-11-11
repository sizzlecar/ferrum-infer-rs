use super::*;

fn tensor(dimensions: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dimensions.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}

#[test]
fn distinct_contract_preserves_original_full_head_abi() {
    let old = half_split_rope_bf16_contract().unwrap();
    let new = partial_half_split_rope_bf16_contract().unwrap();
    assert_eq!(
        old.descriptor().id.as_str(),
        HALF_SPLIT_ROPE_BF16_OPERATION_ID
    );
    assert_ne!(
        old.descriptor().fingerprint().unwrap(),
        new.descriptor().fingerprint().unwrap()
    );
    assert_eq!(
        new.descriptor().id.as_str(),
        PARTIAL_HALF_SPLIT_ROPE_BF16_OPERATION_ID
    );
    let x = tensor(&[3, 256], ElementType::Bf16);
    let partial = tensor(&[3, 96], ElementType::Bf16);
    assert!(HalfSplitRopeBf16Geometry::new(&x, &partial, &partial, &x, 128).is_err());
    let checked =
        PartialHalfSplitRopeBf16Geometry::new(&x, &partial, &partial, &x, 128, 96).unwrap();
    assert_eq!(
        (checked.rows(), checked.heads(), checked.rotary_dim()),
        (3, 2, 96)
    );
}

#[test]
fn geometry_handles_multiple_heads_full_rotation_and_odd_tail_width() {
    for (head, rotary) in [(8, 6), (128, 96), (8, 8), (7, 6)] {
        let x = tensor(&[4, head * 3], ElementType::Bf16);
        let f = tensor(&[4, rotary], ElementType::Bf16);
        let g = PartialHalfSplitRopeBf16Geometry::new(&x, &f, &f, &x, head, rotary).unwrap();
        assert_eq!(g.heads(), 3);
    }
}

#[test]
fn invalid_geometry_dtypes_and_frequency_extents_fail_closed() {
    let x = tensor(&[2, 16], ElementType::Bf16);
    let f = tensor(&[2, 6], ElementType::Bf16);
    for (head, rotary) in [(0, 6), (8, 0), (8, 5), (8, 10), (7, 6)] {
        assert!(PartialHalfSplitRopeBf16Geometry::new(&x, &f, &f, &x, head, rotary).is_err());
    }
    for bad in [
        tensor(&[2, 8], ElementType::Bf16),
        tensor(&[2, 6], ElementType::F32),
        tensor(&[1, 12], ElementType::Bf16),
    ] {
        assert!(PartialHalfSplitRopeBf16Geometry::new(&x, &bad, &f, &x, 8, 6).is_err());
    }
    let wrong = tensor(&[4, 8], ElementType::Bf16);
    assert!(PartialHalfSplitRopeBf16Geometry::new(&x, &f, &f, &wrong, 8, 6).is_err());
}
