use super::{
    backing_segment_range, backing_segment_range_matches, backing_segment_range_matches_with_poll,
    BackingSegment, DynamicBackingPoolId,
};

#[test]
fn borrowed_projection_budget_polls_segments_before_the_requested_window() {
    let source = source();
    let expected = backing_segment_range(&source, 128, 64).unwrap();
    let mut remaining = 1_usize;
    let interrupted = backing_segment_range_matches_with_poll(&source, 128, 64, &expected, || {
        let available = remaining > 0;
        remaining = remaining.saturating_sub(1);
        available
    })
    .unwrap();
    assert_eq!(
        interrupted, None,
        "skipped leading segments consume the budget too"
    );
    assert_eq!(
        backing_segment_range_matches_with_poll(&source, 128, 64, &expected, || true).unwrap(),
        Some(true)
    );
    let mut wrong = expected;
    wrong[0] = source[0].clone();
    assert_eq!(
        backing_segment_range_matches_with_poll(&source, 128, 64, &wrong, || true).unwrap(),
        Some(false)
    );
}

fn pool(hash_digit: char) -> DynamicBackingPoolId {
    serde_json::from_value(serde_json::json!(format!(
        "dynamic-pool/sha256/{}",
        hash_digit.to_string().repeat(64)
    )))
    .unwrap()
}

fn source() -> Vec<BackingSegment> {
    let pool = pool('a');
    vec![
        BackingSegment::from_chunk(&pool, 1, 7, 96, 64).unwrap(),
        BackingSegment::from_chunk(&pool, 2, 8, 160, 64).unwrap(),
        BackingSegment::from_chunk(&pool, 3, 9, 224, 64).unwrap(),
    ]
}

#[test]
fn borrowed_projection_matches_exact_and_cross_chunk_windows() {
    let source = source();
    for (offset, length) in [(0, 192), (32, 80), (63, 66), (64, 64), (191, 1)] {
        let owned = backing_segment_range(&source, offset, length).unwrap();
        assert!(backing_segment_range_matches(&source, offset, length, &owned).unwrap());
    }
    let expected = [
        BackingSegment::from_chunk(&pool('a'), 1, 7, 128, 32).unwrap(),
        BackingSegment::from_chunk(&pool('a'), 2, 8, 160, 48).unwrap(),
    ];
    assert_eq!(backing_segment_range(&source, 32, 80).unwrap(), expected);
    assert!(backing_segment_range_matches(&source, 32, 80, &expected).unwrap());
}

#[test]
fn borrowed_projection_rejects_altered_evidence_and_segment_count() {
    let source = source();
    let expected = backing_segment_range(&source, 32, 80).unwrap();
    for replacement in [
        BackingSegment::from_chunk(&pool('b'), 1, 7, 128, 32).unwrap(),
        BackingSegment::from_chunk(&pool('a'), 4, 7, 128, 32).unwrap(),
        BackingSegment::from_chunk(&pool('a'), 1, 8, 128, 32).unwrap(),
        BackingSegment::from_chunk(&pool('a'), 1, 7, 127, 32).unwrap(),
        BackingSegment::from_chunk(&pool('a'), 1, 7, 128, 31).unwrap(),
    ] {
        let mut evidence = expected.clone();
        evidence[0] = replacement;
        assert!(!backing_segment_range_matches(&source, 32, 80, &evidence).unwrap());
    }
    let mut reversed = expected.clone();
    reversed.reverse();
    let mut extra = expected.clone();
    extra.push(expected[0].clone());
    for evidence in [&[][..], &expected[..1], &reversed, &extra] {
        assert!(!backing_segment_range_matches(&source, 32, 80, evidence).unwrap());
    }
}

#[test]
fn mismatched_evidence_does_not_skip_range_or_overflow_validation() {
    let source = source();
    for (offset, length) in [(0, 0), (192, 1), (160, 64), (u64::MAX, 1)] {
        assert!(backing_segment_range(&source, offset, length).is_err());
        assert!(backing_segment_range_matches(&source, offset, length, &[]).is_err());
    }
    assert!(backing_segment_range_matches(&[], 0, 1, &[]).is_err());

    // A corrupted later segment must fail even when the first comparison
    // already differs. Keep this private malformed fixture at the validator.
    let mut malformed = source;
    malformed[1].offset_bytes = u64::MAX - 16;
    assert!(backing_segment_range_matches(&malformed, 32, 80, &[]).is_err());
    assert!(backing_segment_range(&malformed, 32, 80).is_err());
}
