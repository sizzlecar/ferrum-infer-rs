use super::*;

fn branching_rows() -> [[f64; 5]; 6] {
    [
        [1., 1., 16., 0., 0.],
        [1., 8., 512., 0., 0.],
        [1., 1., 64., 0., 0.],
        [1., 1., 16., 1., 0.],
        [1., 1., 16., 0., 1.],
        [1., 8., 512., 0., 0.],
    ]
}

fn equivalent_success<const D: usize>(
    inputs: &[[f64; D]],
    anchors: &[usize],
) -> StructuredInputPivotsV1 {
    let rows: Vec<_> = inputs.iter().map(|row| row.as_slice()).collect();
    let settings = StructuredSettingsV2::default();
    let scratch = input_geometry_pivot_scratch_bytes_v1(rows.len(), D, settings.max_rank).unwrap();
    let mut old_work = work(u64::MAX);
    let original =
        input_geometry_pivots_original_v1(&rows, anchors, &settings, &mut old_work, scratch)
            .unwrap();
    let mut new_work = work(u64::MAX);
    let actual =
        input_geometry_pivots_v1(&rows, anchors, &settings, &mut new_work, scratch).unwrap();
    assert_eq!(actual.pivot_indices, original.pivot_indices);
    assert_eq!(actual.rank, original.rank);
    assert_eq!(actual.anchor_rank, original.anchor_rank);
    assert_eq!(original.work_visits, old_work.visits());
    assert_eq!(actual.work_visits, new_work.visits());
    assert!(new_work.visits() <= old_work.visits());
    assert!(!old_work.exhausted());
    assert!(!new_work.exhausted());
    actual
}

#[test]
fn first_pass_prefix_keeps_original_pivots_across_anchor_transition_and_ties() {
    let inputs = branching_rows();
    for anchors in [&[][..], &[0, 2][..], &[0, 1, 2, 3, 4, 5][..]] {
        let actual = equivalent_success(&inputs, anchors);
        assert_eq!(actual.rank, 5);
        assert!(!actual.pivot_indices.contains(&5));
        if anchors.is_empty() {
            assert_eq!(actual.anchor_rank, 0);
        } else if anchors.len() == inputs.len() {
            assert_eq!(actual.anchor_rank, actual.rank);
        } else {
            assert_eq!(actual.anchor_rank, 2);
            assert!(actual.rank - actual.anchor_rank >= 2);
        }
    }
}

#[test]
fn first_pass_prefix_keeps_zero_rank_anchor_transition() {
    let inputs = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let actual = equivalent_success(&inputs, &[0]);
    assert_eq!(actual.anchor_rank, 0);
    assert_eq!(actual.rank, 3);
    assert_eq!(actual.pivot_indices, [1, 2, 3]);
}

#[test]
fn first_pass_prefix_preserves_dependent_and_ill_conditioned_decisions() {
    // These are the same tolerance regions exercised by readiness_v3: the
    // constant large columns make the unit perturbation dependent, ambiguous,
    // or independently identifiable after the original normalization.
    for (magnitude, expected_rank) in [
        (10_000_000_000_000., Some(1)),
        (20_000_000., None),
        (2_000_000., Some(2)),
    ] {
        let inputs = [
            [0., magnitude, -0., magnitude, 0.],
            [0., magnitude, -0., magnitude + 1., 0.],
            [0., magnitude, -0., magnitude, 0.],
        ];
        for anchors in [&[][..], &[0][..], &[0, 1, 2][..]] {
            if let Some(rank) = expected_rank {
                assert_eq!(equivalent_success(&inputs, anchors).rank, rank);
            } else {
                let rows: Vec<_> = inputs.iter().map(|row| row.as_slice()).collect();
                let settings = StructuredSettingsV2::default();
                let scratch =
                    input_geometry_pivot_scratch_bytes_v1(rows.len(), 5, settings.max_rank)
                        .unwrap();
                let mut old_work = work(u64::MAX);
                let original = input_geometry_pivots_original_v1(
                    &rows,
                    anchors,
                    &settings,
                    &mut old_work,
                    scratch,
                );
                assert_eq!(original, Err(StructuredUnknown::IllConditioned));
                let mut new_work = work(u64::MAX);
                let actual =
                    input_geometry_pivots_v1(&rows, anchors, &settings, &mut new_work, scratch);
                assert_eq!(actual, original);
                assert!(!old_work.exhausted());
                assert!(!new_work.exhausted());
                assert!(new_work.visits() <= old_work.visits());
            }
        }
    }
}

#[test]
fn first_pass_prefix_keeps_exact_shared_work_and_original_scratch_boundaries() {
    let inputs = branching_rows();
    let anchors = &[0, 2];
    let full = equivalent_success(&inputs, anchors);
    let rows: Vec<_> = inputs.iter().map(|row| row.as_slice()).collect();
    let settings = StructuredSettingsV2::default();
    let scratch = input_geometry_pivot_scratch_bytes_v1(rows.len(), 5, settings.max_rank).unwrap();
    let mut original_work = work(u64::MAX);
    input_geometry_pivots_original_v1(&rows, anchors, &settings, &mut original_work, scratch)
        .unwrap();
    let required = full.work_visits;
    assert!(required > 1);
    assert!(required < original_work.visits());

    let mut exact = work(required);
    assert_eq!(
        input_geometry_pivots_v1(&rows, anchors, &settings, &mut exact, scratch).unwrap(),
        full
    );
    assert_eq!(exact.visits(), required);
    assert!(!exact.exhausted());

    let mut short = work(required - 1);
    assert_eq!(
        input_geometry_pivots_v1(&rows, anchors, &settings, &mut short, scratch),
        Err(StructuredUnknown::Capacity)
    );
    assert!(short.exhausted());
    let spent = short.visits();
    assert!(spent > 0 && spent <= short.maximum_visits());
    assert_eq!(
        input_geometry_pivots_v1(&rows, anchors, &settings, &mut short, scratch),
        Err(StructuredUnknown::Capacity)
    );
    assert_eq!(short.visits(), spent);

    // First-pass scratch is call-local. Repeated calls retain the original
    // shared, nonrefundable ledger instead of gaining a cross-call cache hit.
    let mut shared = work(required.checked_mul(2).unwrap());
    for _ in 0..2 {
        assert_eq!(
            input_geometry_pivots_v1(&rows, anchors, &settings, &mut shared, scratch).unwrap(),
            full
        );
    }
    assert_eq!(shared.visits(), required * 2);
    assert_eq!(
        input_geometry_pivots_v1(&rows, anchors, &settings, &mut shared, scratch),
        Err(StructuredUnknown::Capacity)
    );
    assert!(shared.exhausted());
    assert_eq!(shared.visits(), required * 2);

    let mut memory = work(required);
    assert_eq!(
        input_geometry_pivots_v1(&rows, anchors, &settings, &mut memory, scratch - 1),
        Err(StructuredUnknown::Capacity)
    );
    assert_eq!(memory.visits(), 0);
    assert!(!memory.exhausted());
    assert_eq!(
        input_geometry_pivots_v1(&rows, anchors, &settings, &mut memory, scratch).unwrap(),
        full
    );
}
