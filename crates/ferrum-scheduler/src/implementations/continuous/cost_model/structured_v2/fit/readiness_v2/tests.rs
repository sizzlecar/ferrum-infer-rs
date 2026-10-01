use super::*;

fn configuration() -> StructuredSettingsV2 {
    StructuredSettingsV2 {
        max_axes: 512,
        max_rank: 64,
        max_phase_samples: 4096,
        min_fit_redundancy: 8,
        ..Default::default()
    }
}

fn matrix(distinct: bool) -> Vec<Vec<f64>> {
    (0..228)
        .map(|i| {
            let mut row = vec![0.0; 207];
            row[0] = 1.0;
            if distinct && i >= 12 {
                // Distinct input work, still in the same twelve-dimensional
                // space. This case cannot benefit from duplicate elimination.
                row[1] = i as f64;
                for (j, value) in row.iter_mut().enumerate().take(12).skip(2) {
                    *value = ((i * (j + 1) + i * i) % (23 + j)) as f64;
                }
            } else if i % 12 != 0 {
                row[i % 12] = 1.0;
            }
            row
        })
        .collect()
}

fn borrowed(rows: &[Vec<f64>]) -> Vec<FitRow<'_>> {
    rows.iter()
        .map(|basis| FitRow {
            basis,
            wall_ns: 1000,
        })
        .collect()
}

fn work(limit: u64) -> GeometryWork {
    GeometryWork {
        used: 0,
        limit,
        exhausted: false,
    }
}

#[test]
fn readiness_v2_rank_twelve_wide_inputs_fit_unchanged_work_cap() {
    for distinct in [false, true] {
        let matrix = matrix(distinct);
        let rows = borrowed(&matrix);
        let settings = configuration();
        let original = input_geometry(&rows, &settings).unwrap();
        assert_eq!(original.basis.len(), 12);
        let mut old_work = work(16_000_000);
        assert!(input_geometry_with_work(&rows, &settings, Some(&mut old_work)).is_err());
        assert!(old_work.exhausted);
        for cap in [15_000_000, 16_000_000] {
            let mut bounded = work(cap);
            assert_eq!(
                input_geometry_readiness_v2(&rows, &settings, &mut bounded),
                Ok(12)
            );
            assert!(!bounded.exhausted && bounded.used <= cap);
        }
    }
}

#[test]
fn readiness_v2_late_direction_and_changing_scale_are_reverified() {
    let mut matrix = matrix(true);
    let settings = configuration();
    let mut cumulative = work(16_000_000);
    assert_eq!(
        input_geometry_readiness_v2(&borrowed(&matrix[..227]), &settings, &mut cumulative),
        Ok(12)
    );
    // A direction absent from every earlier row must survive cache updates.
    matrix[227][12] = 1.0;
    // A later coordinate maximum also invalidates any earlier normalization.
    matrix[227][1] *= 1000.0;
    let rows = borrowed(&matrix);
    assert_eq!(input_geometry(&rows, &settings).unwrap().basis.len(), 13);
    assert_eq!(
        input_geometry_readiness_v2(&rows, &settings, &mut cumulative),
        Ok(13)
    );
    assert!(!cumulative.exhausted);
}

#[test]
fn readiness_v2_ambiguous_pivots_keep_original_thresholds() {
    for (magnitude, expected) in [
        (10_000_000_000_000.0, Ok(1)),
        (20_000_000.0, Err(StructuredUnknown::IllConditioned)),
        (2_000_000.0, Ok(2)),
    ] {
        let matrix: Vec<_> = (0..16)
            .map(|i| vec![magnitude, magnitude + (i % 2) as f64])
            .collect();
        let rows = borrowed(&matrix);
        let settings = configuration();
        assert_eq!(
            input_geometry(&rows, &settings).map(|g| g.basis.len()),
            expected
        );
        assert_eq!(
            input_geometry_readiness_v2(&rows, &settings, &mut work(16_000_000)),
            expected
        );
    }
}

#[test]
fn readiness_v2_exact_work_boundary_and_walls_have_no_authority() {
    let matrix = matrix(false);
    let mut rows = borrowed(&matrix);
    let settings = configuration();
    let mut measured = work(16_000_000);
    assert_eq!(
        input_geometry_readiness_v2(&rows, &settings, &mut measured),
        Ok(12)
    );
    let required = measured.used;
    for row in &mut rows {
        row.wall_ns = u64::MAX;
    }
    let mut exact = work(required);
    assert_eq!(
        input_geometry_readiness_v2(&rows, &settings, &mut exact),
        Ok(12)
    );
    assert_eq!(exact.used, required);
    let mut short = work(required - 1);
    assert_eq!(
        input_geometry_readiness_v2(&rows, &settings, &mut short),
        Err(StructuredUnknown::Capacity)
    );
    assert!(short.exhausted && short.used <= short.limit);
}
