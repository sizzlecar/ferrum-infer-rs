use super::*;

fn n(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}
fn limits(maximum: u32, alignment: u32, budget: usize) -> ReferenceChunkLimits {
    ReferenceChunkLimits {
        maximum_tokens: n(maximum),
        alignment: n(alignment),
        allow_final_short_chunk: true,
        maximum_candidates: NonZeroUsize::new(budget).unwrap(),
    }
}

#[test]
fn chunk_ladder_keeps_intermediate_latency_and_throughput_choices() {
    let chunks = piecewise_chunk_candidates(618, n(32), limits(128, 1, 64)).unwrap();
    for required in [1, 16, 32, 64, 128] {
        assert!(chunks.contains(&n(required)));
    }
    // Reference granule and maximum survive a tighter optional budget.
    let bounded = piecewise_chunk_candidates(618, n(32), limits(128, 1, 4)).unwrap();
    assert!(bounded.contains(&n(32)) && bounded.contains(&n(128)));
    assert!(bounded
        .iter()
        .any(|count| count.get() > 32 && count.get() < 128));
}

#[test]
fn chunk_ladder_respects_real_alignment_tail_and_overflow_boundaries() {
    for (remaining, granule, maximum, alignment) in [
        (618, 32, 128, 1),
        (503, 32, 127, 16),
        (619, 288, 600, 96),
        (13, 32, 128, 16),
        (u32::MAX, u32::MAX, u32::MAX, 1),
        (u32::MAX, 32, u32::MAX, 1 << 31),
    ] {
        for budget in 1..=64 {
            let chunks = piecewise_chunk_candidates(
                remaining,
                n(granule),
                limits(maximum, alignment, budget),
            )
            .unwrap();
            assert!(chunks.len() <= budget);
            assert!(chunks.windows(2).all(|pair| pair[0] < pair[1]));
            for count in chunks {
                assert!(count.get() <= maximum && count.get() <= remaining);
                assert!(count.get() % alignment == 0 || count.get() == remaining);
            }
        }
    }
    let mut no_tail = limits(128, 16, 64);
    no_tail.allow_final_short_chunk = false;
    assert!(matches!(
        piecewise_chunk_candidates(13, n(32), no_tail),
        Err(ReferenceUnknown::NoLegalChunk)
    ));
    assert!(matches!(
        piecewise_chunk_candidates(0, n(32), no_tail),
        Err(ReferenceUnknown::NoLegalChunk)
    ));
}
