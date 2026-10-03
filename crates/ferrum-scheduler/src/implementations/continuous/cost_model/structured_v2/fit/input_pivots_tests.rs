use super::*;

fn work(limit: u64) -> StructuredInputGeometryWorkV1 {
    StructuredInputGeometryWorkV1::new(NonZeroU64::new(limit).unwrap())
}

#[test]
fn cold_input_pivots_preserve_cross_direction_without_requiring_full_column_rank() {
    // Constant overhead, B, B*KV, a dependent counter, and an unused axis.
    let inputs = [
        [1., 1., 16., 2., 0.],
        [1., 8., 512., 16., 0.],
        [1., 1., 64., 2., 0.],
    ];
    let rows: Vec<_> = inputs.iter().map(|row| row.as_slice()).collect();
    let settings = StructuredSettingsV2::default();
    let scratch = input_geometry_pivot_scratch_bytes_v1(rows.len(), 5, settings.max_rank).unwrap();
    let mut budget = work(1_000_000);
    let diagonal =
        input_geometry_pivots_v1(&rows[..2], &[], &settings, &mut budget, scratch).unwrap();
    assert_eq!(diagonal.rank, 2);
    let before = budget.visits();
    let complete = input_geometry_pivots_v1(&rows, &[], &settings, &mut budget, scratch).unwrap();
    assert_eq!(complete.rank, 3);
    assert_eq!(complete.work_visits, budget.visits() - before);
    let anchored =
        input_geometry_pivots_v1(&rows, &[0, 1], &settings, &mut budget, scratch).unwrap();
    assert_eq!(anchored.anchor_rank, diagonal.rank);
    assert_eq!(anchored.rank, complete.rank);
    assert_eq!(&anchored.pivot_indices[anchored.anchor_rank..], &[2]);
    assert!(complete.rank < rows[0].len());
    assert_eq!(complete.pivot_indices.len(), complete.rank);
    assert!(complete.pivot_indices.iter().all(|&i| i < inputs.len()));
    assert!(complete.pivot_indices.contains(&2));
    // Cold pivots do not supply the actual independent samples that Fit needs.
    let samples: Vec<_> = rows
        .iter()
        .map(|basis| FitRow {
            basis,
            wall_ns: 1000,
        })
        .collect();
    assert!(matches!(
        input_geometry(&samples, &settings),
        Err(StructuredUnknown::InsufficientRedundancy)
    ));
}

#[test]
fn cold_input_pivots_keep_original_tie_order_and_nonrefundable_work_and_scratch_limits() {
    let inputs = [[1., 1., 16.], [1., 8., 512.], [1., 1., 64.], [1., 8., 512.]];
    let rows: Vec<_> = inputs.iter().map(|row| row.as_slice()).collect();
    let settings = StructuredSettingsV2::default();
    let scratch = input_geometry_pivot_scratch_bytes_v1(rows.len(), 3, settings.max_rank).unwrap();
    let mut full = work(1_000_000);
    let pivots = input_geometry_pivots_v1(&rows, &[], &settings, &mut full, scratch).unwrap();
    assert!(!pivots.pivot_indices.contains(&3));
    let mut short = work(full.visits() - 1);
    assert_eq!(
        input_geometry_pivots_v1(&rows, &[], &settings, &mut short, scratch),
        Err(StructuredUnknown::Capacity)
    );
    assert!(short.exhausted());
    let spent = short.visits();
    assert_eq!(
        input_geometry_pivots_v1(&rows, &[], &settings, &mut short, scratch),
        Err(StructuredUnknown::Capacity)
    );
    assert_eq!(short.visits(), spent);
    let mut exact = work(full.visits());
    assert_eq!(
        input_geometry_pivots_v1(&rows, &[], &settings, &mut exact, scratch).unwrap(),
        pivots
    );
    let mut memory = work(1_000_000);
    assert_eq!(
        input_geometry_pivots_v1(&rows, &[], &settings, &mut memory, scratch - 1),
        Err(StructuredUnknown::Capacity)
    );
    assert_eq!(memory.visits(), 0);
    assert_eq!(
        input_geometry_pivots_v1(&rows, &[], &settings, &mut memory, scratch).unwrap(),
        pivots
    );
}

#[path = "input_pivots_tests/capture_replay.rs"]
mod capture_replay;

#[path = "input_pivots_tests/first_pass.rs"]
mod first_pass;
