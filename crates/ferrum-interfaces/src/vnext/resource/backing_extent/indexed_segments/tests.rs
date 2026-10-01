use super::*;

fn segments(count: usize) -> Vec<BackingSegment> {
    let pool: DynamicBackingPoolId = serde_json::from_value(serde_json::json!(format!(
        "dynamic-pool/sha256/{}",
        "a".repeat(64)
    )))
    .unwrap();
    (0..count)
        .map(|i| {
            BackingSegment::from_chunk(&pool, i as u32 + 1, 7, 96 + i as u64 * 128, 64).unwrap()
        })
        .collect()
}

#[test]
fn indexed_windows_match_linear_validation_and_reject_changed_evidence() {
    let source = segments(4);
    let indexed = IndexedBackingSegments::new(source.clone());
    assert!(indexed.ends.is_some());
    for (offset, length) in [(0, 256), (32, 80), (63, 66), (64, 64), (255, 1)] {
        let expected = backing_segment_range(&source, offset, length).unwrap();
        for evidence in [expected.clone(), vec![], expected[..1].to_vec()] {
            assert_eq!(
                indexed
                    .range_matches_with_poll(offset, length, &evidence, || true)
                    .unwrap(),
                backing_segment_range_matches_with_poll(&source, offset, length, &evidence, || {
                    true
                })
                .unwrap()
            );
        }
        let mut altered = expected;
        altered[0] = source[(offset as usize / 64 + 1) % source.len()].clone();
        assert_eq!(
            indexed
                .range_matches_with_poll(offset, length, &altered, || true)
                .unwrap(),
            Some(false)
        );
    }
    for (offset, length) in [(0, 0), (256, 1), (255, 2), (u64::MAX, 1)] {
        assert!(indexed
            .range_matches_with_poll(offset, length, &[], || true)
            .is_err());
    }
}

#[test]
fn indexed_budget_charges_overlaps_without_rescanning_unrelated_leading_segments() {
    let source = segments(512);
    let expected = backing_segment_range(&source, 510 * 64 + 32, 64).unwrap();
    let indexed = IndexedBackingSegments::new(source);
    let mut calls = 0;
    assert_eq!(
        indexed
            .range_matches_with_poll(510 * 64 + 32, 64, &expected, || {
                calls += 1;
                true
            })
            .unwrap(),
        Some(true)
    );
    assert!(calls <= expected.len() + 2);
    assert_eq!(
        indexed
            .range_matches_with_poll(510 * 64 + 32, 64, &expected, || false)
            .unwrap(),
        None
    );
    let mut remaining = 1_usize;
    assert_eq!(
        indexed
            .range_matches_with_poll(510 * 64 + 32, 64, &expected, || {
                let available = remaining > 0;
                remaining = remaining.saturating_sub(1_usize);
                available
            })
            .unwrap(),
        None
    );
}

#[test]
fn absent_index_keeps_original_validation_including_malformed_ranges() {
    let source = segments(3);
    let expected = backing_segment_range(&source, 63, 66).unwrap();
    let mut indexed = IndexedBackingSegments::new(source.clone());
    indexed.ends = None;
    assert_eq!(
        indexed
            .range_matches_with_poll(63, 66, &expected, || true)
            .unwrap(),
        Some(true)
    );
    let mut malformed = source;
    malformed[1].offset_bytes = u64::MAX - 16;
    let indexed = IndexedBackingSegments::new(malformed);
    assert!(indexed.ends.is_none());
    assert!(indexed
        .range_matches_with_poll(32, 80, &[], || true)
        .is_err());
    for count in [0, 1] {
        let indexed = IndexedBackingSegments::new(segments(count));
        assert!(indexed.ends.is_none());
        assert!(indexed
            .range_matches_with_poll(64, 1, &[], || true)
            .is_err());
    }
}
