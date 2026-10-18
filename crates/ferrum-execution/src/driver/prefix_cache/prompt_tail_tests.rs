use super::*;
use checkpoint_fixture::{Fixture, Spec};
use std::num::NonZeroU64;

#[test]
fn prompt_tail_boundary_respects_provider_span_and_suffix_constraints() {
    for (minimum, prefix_alignment, suffix_minimum, suffix_alignment) in
        [(1, 1, 1, 1), (3, 2, 5, 3), (4, 4, 2, 4)]
    {
        let fixture = Fixture::build(Spec {
            dependency: CheckpointInputDependency::ExactTokenPrefix,
            boundaries: CheckpointBoundaryConstraint::new(
                CheckpointTokenSpanConstraint::new(
                    NonZeroU64::new(minimum).unwrap(),
                    NonZeroU64::new(prefix_alignment).unwrap(),
                )
                .unwrap(),
                CheckpointTokenSpanConstraint::new(
                    NonZeroU64::new(suffix_minimum).unwrap(),
                    NonZeroU64::new(suffix_alignment).unwrap(),
                )
                .unwrap(),
            )
            .unwrap(),
            checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
            ..Spec::default()
        })
        .unwrap();
        let layout = usable_layout(&fixture.plan).unwrap();
        for processed in [0, 2, 7] {
            for prompt in processed + 1..=32 {
                let expected = (processed + 1..prompt)
                    .rev()
                    .find(|&boundary| layout.permits_capture_from(processed, boundary, prompt));
                assert_eq!(layout.prompt_tail_boundary(processed, prompt), expected);
            }
        }
        assert!(layout
            .prompt_tail_boundary(u64::MAX - 1, u64::MAX)
            .is_none());
        assert!(layout.prompt_tail_boundary(u64::MAX, u64::MAX).is_none());
    }
}

#[test]
fn prompt_tail_planner_only_splits_the_chunk_that_reaches_the_boundary() {
    let fixture = Fixture::build(Spec {
        dependency: CheckpointInputDependency::ExactTokenPrefix,
        boundaries: CheckpointBoundaryConstraint::new(
            CheckpointTokenSpanConstraint::any_positive(),
            CheckpointTokenSpanConstraint::new(
                NonZeroU64::new(4).unwrap(),
                NonZeroU64::new(4).unwrap(),
            )
            .unwrap(),
        )
        .unwrap(),
        checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
        ..Spec::default()
    })
    .unwrap();
    let layout = usable_layout(&fixture.plan).unwrap();
    assert!(
        rendezvous::select_prompt_tail_boundary(layout, PrefillChunk::new(0, 4, 13).unwrap())
            .is_none()
    );
    // The required suffix is four tokens, so the useful cut precedes the final
    // natural chunk rather than always subtracting one token from the prompt.
    let plan =
        rendezvous::select_prompt_tail_boundary(layout, PrefillChunk::new(8, 4, 13).unwrap())
            .unwrap();
    assert_eq!(plan.boundary, 9);
    assert_eq!(plan.span, CheckpointTokenSpanConstraint::any_positive());
    let retired_tail = PrefillChunk::new(8, 1, 13).unwrap();
    assert_eq!(
        rendezvous::select_prompt_tail_boundary(layout, retired_tail)
            .unwrap()
            .boundary,
        9
    );
    assert!(
        !capture_candidate(retired_tail),
        "the old natural-boundary heuristic misses this legal tail"
    );
    assert!(prompt_tail_capture_candidate(layout, retired_tail));
    assert!(!prompt_tail_capture_candidate(
        layout,
        PrefillChunk::new(9, 4, 13).unwrap()
    ));
    assert!(
        rendezvous::select_prompt_tail_boundary(layout, PrefillChunk::new(9, 4, 13).unwrap())
            .is_none()
    );
}

#[test]
fn prompt_tail_planner_requires_usable_prefix_capability() {
    for spec in [
        Spec::default(),
        Spec {
            dependency: CheckpointInputDependency::EntireTokenInput,
            checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
            ..Spec::default()
        },
        Spec {
            conditioning: true,
            checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
            ..Spec::default()
        },
        Spec {
            numerics: CheckpointPartitionNumerics::SamePartitionOnly,
            checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
            ..Spec::default()
        },
    ] {
        let fixture = Fixture::build(spec).unwrap();
        let plan = usable_layout(&fixture.plan).and_then(|layout| {
            rendezvous::select_prompt_tail_boundary(layout, PrefillChunk::new(0, 8, 8).unwrap())
        });
        assert!(plan.is_none());
    }
}
