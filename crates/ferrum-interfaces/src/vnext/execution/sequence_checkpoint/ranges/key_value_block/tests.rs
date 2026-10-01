use super::*;

fn tensor(heads: u64, dimension: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![2, heads, dimension],
        ElementType::F16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

fn apply(source: &[u8], destination: &mut [u8], region: StridedCopyRegion) {
    for row in 0..region.height() {
        let a = (region.source_offset_bytes() + row * region.source_pitch_bytes()) as usize;
        let b =
            (region.destination_offset_bytes() + row * region.destination_pitch_bytes()) as usize;
        let n = region.width_bytes() as usize;
        destination[b..b + n].copy_from_slice(&source[a..a + n]);
    }
}

/// Independent indexing follows the provider's [block,K/V,head,pack,token]
/// physical ABI and writes only real prefix positions, not an enclosing range.
fn initialized_bytes(heads: u64, dimension: u64, tokens: u64) -> Vec<usize> {
    let mut positions = Vec::new();
    for token in 0..tokens {
        let block = token / 16;
        let within = token % 16;
        for head in 0..heads {
            for dim in 0..dimension {
                let k = block * 2 * heads * dimension * 16
                    + head * dimension * 16
                    + (dim / 8) * 16 * 8
                    + within * 8
                    + dim % 8;
                let v = block * 2 * heads * dimension * 16
                    + heads * dimension * 16
                    + head * dimension * 16
                    + dim * 16
                    + within;
                positions.extend([
                    k as usize * 2,
                    k as usize * 2 + 1,
                    v as usize * 2,
                    v as usize * 2 + 1,
                ]);
            }
        }
    }
    positions
}

#[test]
fn arbitrary_prefix_round_trip_copies_exact_slots_and_preserves_unwritten_tail() {
    let tensor = tensor(2, 16);
    let stride = tensor.minimum_storage_bytes().unwrap();
    for boundary in [1, 2, 7, 15, 16, 17, 31, 32, 33, 47] {
        let (ranges, rectangles) =
            source_copy_plan(&tensor, 16, 8, 0, boundary, stride * 47, 65536).unwrap();
        assert!(ranges.len() <= 1);
        assert_eq!(rectangles.len(), if boundary % 16 == 0 { 0 } else { 2 });
        let mut original = vec![0xD3; 65536];
        let mut expected = vec![0xA7; 65536];
        for (ordinal, index) in initialized_bytes(2, 16, boundary).into_iter().enumerate() {
            let value = (ordinal % 127) as u8;
            original[index] = value;
            expected[index] = value;
        }
        let mut compact = vec![0xEE; (stride * boundary) as usize];
        for range in &ranges {
            compact[..(range.end - range.start) as usize]
                .copy_from_slice(&original[range.start as usize..range.end as usize]);
        }
        for region in &rectangles {
            apply(&original, &mut compact, *region);
        }
        assert!(
            !compact.contains(&0xD3),
            "read uninitialized tail at {boundary}"
        );
        let mut restored = vec![0xA7; 65536];
        for range in &ranges {
            restored[range.start as usize..range.end as usize]
                .copy_from_slice(&compact[..(range.end - range.start) as usize]);
        }
        for region in rectangles {
            apply(&compact, &mut restored, region.reversed());
        }
        assert_eq!(restored, expected, "wrong physical prefix at {boundary}");
    }
}

#[test]
fn real_attention_geometry_has_two_tail_commands_and_exact_semantic_bytes() {
    let tensor = tensor(4, 256);
    let stride = tensor.minimum_storage_bytes().unwrap();
    let (full, tail) = source_copy_plan(&tensor, 16, 8, 0, 31, stride * 128, 65536 * 8).unwrap();
    assert_eq!(full.len(), 1);
    assert_eq!(tail.len(), 2);
    assert_eq!(tail[0].height(), 128);
    assert_eq!(tail[1].height(), 1024);
    assert_eq!(
        full[0].end - full[0].start + tail.iter().map(|r| r.length_bytes().unwrap()).sum::<u64>(),
        31 * stride
    );
    assert!(tail[1].source_end_bytes().unwrap() > 31 * stride);
}

#[test]
fn invalid_or_uncommitted_geometry_is_rejected() {
    let tensor = tensor(1, 8);
    let stride = tensor.minimum_storage_bytes().unwrap();
    assert!(source_copy_plan(&tensor, 16, 8, 0, 0, stride * 16, 65536).is_err());
    assert!(source_copy_plan(&tensor, 16, 8, 0, 17, stride * 16, 65536).is_err());
    assert!(source_copy_plan(&tensor, 16, 8, 0, 1, stride * 16, stride).is_err());
    assert!(source_copy_plan(&tensor, 16, 8, u64::MAX, 1, stride * 16, u64::MAX).is_err());
}
