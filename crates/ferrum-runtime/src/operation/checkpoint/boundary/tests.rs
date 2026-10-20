use super::*;
use crate::CheckpointTokenSpanConstraint;
use std::num::NonZeroU64;

fn rules(
    prefix_min: u64,
    prefix_align: u64,
    suffix_min: u64,
    suffix_align: u64,
) -> CheckpointBoundaryConstraint {
    let span = |minimum, alignment| {
        CheckpointTokenSpanConstraint::new(
            NonZeroU64::new(minimum).unwrap(),
            NonZeroU64::new(alignment).unwrap(),
        )
        .unwrap()
    };
    CheckpointBoundaryConstraint::new(
        span(prefix_min, prefix_align),
        span(suffix_min, suffix_align),
    )
    .unwrap()
}

#[test]
fn boundary_crt_matches_independent_bruteforce_for_spans_and_offsets() {
    for a in 1..=6 {
        for b in 1..=6 {
            for prefix_min in [1, 3] {
                for suffix_min in [1, 4] {
                    let constraints = rules(prefix_min, a, suffix_min, b);
                    for processed in 0..=5 {
                        for prompt in processed + 1..=32 {
                            let expected = (processed + 1..prompt).rev().find(|&candidate| {
                                constraints.permits_from(processed, candidate, prompt)
                            });
                            assert_eq!(constraints.prompt_tail_boundary(processed, prompt), expected,
                                "a={a} b={b} prefix_min={prefix_min} suffix_min={suffix_min} start={processed} prompt={prompt}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn boundary_crt_handles_modulus_one_negative_delta_and_incompatible_gcd() {
    assert_eq!(rules(1, 7, 1, 1).prompt_tail_boundary(5, 30), Some(26));
    assert_eq!(rules(1, 7, 1, 11).prompt_tail_boundary(5, 100), Some(89));
    assert_eq!(rules(1, 4, 1, 6).prompt_tail_boundary(1, 12), None);
    assert_eq!(modular_inverse(7, 1), Some(0));
    assert_eq!(modular_inverse(7, 11), Some(8));
    assert_eq!(modular_inverse(4, 6), None);
}

#[test]
fn boundary_crt_handles_large_coprime_period_and_u64_edges() {
    let a = (1_u64 << 63) - 1;
    let b = 1_u64 << 63;
    assert_eq!(rules(1, a, 1, b).prompt_tail_boundary(0, u64::MAX), Some(a));
    let (a, b) = (1_000_000_007, 1_000_000_009);
    assert_eq!(rules(1, a, 1, b).prompt_tail_boundary(0, a + b), Some(a));
    let any = CheckpointBoundaryConstraint::any_positive();
    assert_eq!(any.prompt_tail_boundary(0, u64::MAX), Some(u64::MAX - 1));
    assert_eq!(any.prompt_tail_boundary(u64::MAX - 1, u64::MAX), None);
    assert_eq!(any.prompt_tail_boundary(u64::MAX, u64::MAX), None);
    assert_eq!(
        rules(2, 1, 1, 1).prompt_tail_boundary(u64::MAX, u64::MAX),
        None
    );
    assert_eq!(any.prompt_tail_boundary(8, 7), None);
}

#[test]
fn boundary_crt_preserves_shared_prefix_cap_and_all_follower_suffixes() {
    let constraints = rules(1, 2, 2, 3);
    for processed in 0..=4 {
        for source in 8..=20 {
            for common in 0..=source {
                for followers in [&[14_u64, 17][..], &[15_u64, 17][..], &[][..]] {
                    let expected = (processed + 1..source).rev().find(|&candidate| {
                        candidate <= common
                            && constraints.permits_from(processed, candidate, source)
                            && followers.iter().all(|&prompt| {
                                constraints.permits_from(processed, candidate, prompt)
                            })
                    });
                    assert_eq!(
                        constraints.latest_reusable_boundary(processed, source, common, followers),
                        expected
                    );
                }
            }
        }
    }
}
