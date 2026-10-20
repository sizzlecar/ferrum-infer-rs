use super::*;
use ferrum_runtime::CheckpointTokenSpanConstraint;
use std::num::NonZeroU64;

fn tail(suffix: u64) -> PrefillShapePolicy {
    PrefillShapePolicy::PromptTail(
        CheckpointBoundaryConstraint::new(
            CheckpointTokenSpanConstraint::any_positive(),
            CheckpointTokenSpanConstraint::new(NonZeroU64::new(suffix).unwrap(), NonZeroU64::MIN)
                .unwrap(),
        )
        .unwrap(),
    )
}

#[test]
fn unknown_preserves_range_without_qualifying_identity_or_replay() {
    let chunk = PrefillChunk::new(2, 3, 5).unwrap();
    for original in [false, true] {
        let plan = PrefillShapePolicy::Unknown
            .project(chunk, original)
            .unwrap();
        assert_eq!(plan.chunk(), chunk);
        assert!(!plan.is_exact());
        let known = PrefillShapePolicy::Identity
            .project(chunk, original)
            .unwrap();
        assert_eq!(known.chunk(), chunk);
        assert!(known.is_exact());
    }
}

#[test]
fn tail_projection_covers_before_at_after_boundary_and_is_idempotent() {
    for (requested, expected) in [(2, 2), (3, 3), (4, 3), (5, 3)] {
        let policy = tail(2);
        let candidate = PrefillChunk::new(0, requested, 5).unwrap();
        let plan = policy.project(candidate, true).unwrap();
        assert!(plan.is_exact());
        assert_eq!(plan.chunk(), PrefillChunk::new(0, expected, 5).unwrap());
        assert_eq!(policy.project(plan.chunk(), true).unwrap(), plan);
        assert_eq!(policy.project(candidate, false).unwrap().chunk(), candidate);
    }
    let suffix = PrefillChunk::new(3, 2, 5).unwrap();
    assert_eq!(tail(2).project(suffix, true).unwrap().chunk(), suffix);
    assert!(suffix.is_final());
}

#[test]
fn tail_projection_uses_restored_start_and_legal_span_alignment() {
    let aligned = CheckpointTokenSpanConstraint::new(
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(2).unwrap(),
    )
    .unwrap();
    let policy = PrefillShapePolicy::PromptTail(
        CheckpointBoundaryConstraint::new(aligned, aligned).unwrap(),
    );
    let restored = PrefillChunk::new(1, 6, 7).unwrap();
    let actual = policy.project(restored, true).unwrap();
    assert_eq!(actual.chunk(), PrefillChunk::new(1, 4, 7).unwrap());
    let incompatible = PrefillChunk::new(2, 5, 7).unwrap();
    assert_eq!(
        policy.project(incompatible, true).unwrap().chunk(),
        incompatible
    );
    let maximal = PrefillChunk::new(usize::MAX - 2, 2, usize::MAX).unwrap();
    let expected = PrefillChunk::new(usize::MAX - 2, 1, usize::MAX).unwrap();
    assert_eq!(tail(1).project(maximal, true).unwrap().chunk(), expected);
}
