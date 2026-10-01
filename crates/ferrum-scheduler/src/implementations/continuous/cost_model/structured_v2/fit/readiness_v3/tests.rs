use super::*;

fn settings() -> StructuredSettingsV2 {
    StructuredSettingsV2 {
        max_axes: 512,
        max_rank: 32,
        max_phase_samples: 4096,
        min_fit_redundancy: 8,
        ..Default::default()
    }
}
fn work(limit: u64) -> GeometryWork {
    GeometryWork {
        used: 0,
        limit,
        exhausted: false,
    }
}
fn borrowed(rows: &[Vec<f64>]) -> Vec<FitRow<'_>> {
    rows.iter()
        .map(|basis| FitRow { basis, wall_ns: 1 })
        .collect()
}
fn wide_inputs() -> Vec<Vec<f64>> {
    // A declared 303-axis roster with 199 positive columns but only twelve
    // independent directions. Collinear positive columns must remain present.
    (0..422)
        .map(|i| {
            let mut row = vec![0.; 303];
            for (j, value) in row.iter_mut().take(199).enumerate() {
                *value = if j % 12 == 0 || j % 12 == i % 12 {
                    1.
                } else {
                    0.
                };
            }
            row
        })
        .collect()
}

#[test]
fn readiness_v3_zero_columns_preserve_rank_with_measured_work() {
    let matrix = wide_inputs();
    let rows = borrowed(&matrix);
    let mut original = work(u64::MAX);
    let mut compact = work(u64::MAX);
    let a = input_geometry_readiness_v2(&rows, &settings(), &mut original);
    let b = input_geometry_readiness_v3(&rows, &settings(), &mut compact);
    assert_eq!(a, Ok(12));
    assert_eq!(a, b);
    assert!(compact.used < original.used);
    eprintln!(
        "READINESS_V3_WORK n=422 d=303 positive_columns=199 rank=12 v2_visits={} v3_visits={}",
        original.used, compact.used
    );
    // Original sample/axis ownership is immutable; only private scratch shrinks.
    assert!(matrix.iter().all(|r| r.len() == 303));
    assert_eq!(matrix[1].iter().filter(|&&v| v != 0.).count(), 34);
}

#[test]
fn readiness_v3_each_prefix_rediscovers_late_zero_column_and_scale() {
    let mut matrix = wide_inputs();
    let mut cumulative = work(u64::MAX);
    let before =
        input_geometry_readiness_v3(&borrowed(&matrix[..421]), &settings(), &mut cumulative);
    assert_eq!(before, Ok(12));
    let used = cumulative.used;
    matrix[421][302] = 1.;
    matrix[421][1] = 1_000.;
    let rows = borrowed(&matrix);
    let old = input_geometry_readiness_v2(&rows, &settings(), &mut work(u64::MAX));
    let new = input_geometry_readiness_v3(&rows, &settings(), &mut cumulative);
    assert_eq!(old, Ok(13));
    assert_eq!(new, old);
    assert!(cumulative.used > used);
}

#[test]
fn readiness_v3_exact_zeros_keep_ambiguous_pivot_tolerances() {
    for (magnitude, expected) in [
        (10_000_000_000_000., Ok(1)),
        (20_000_000., Err(StructuredUnknown::IllConditioned)),
        (2_000_000., Ok(2)),
    ] {
        let plain: Vec<_> = (0..16)
            .map(|i| vec![magnitude, magnitude + (i % 2) as f64])
            .collect();
        let padded: Vec<_> = plain
            .iter()
            .map(|r| vec![0., r[0], -0., 0., r[1], 0.])
            .collect();
        for matrix in [&plain, &padded] {
            let rows = borrowed(matrix);
            assert_eq!(
                input_geometry_readiness_v2(&rows, &settings(), &mut work(u64::MAX)),
                expected
            );
            assert_eq!(
                input_geometry_readiness_v3(&rows, &settings(), &mut work(u64::MAX)),
                expected
            );
        }
    }
    let rows: Vec<_> = (0..16)
        .map(|i| vec![1., 0., (i % 2) as f64 * 1e-20])
        .collect();
    assert_eq!(
        input_geometry_readiness_v3(&borrowed(&rows), &settings(), &mut work(u64::MAX)),
        input_geometry_readiness_v2(&borrowed(&rows), &settings(), &mut work(u64::MAX))
    );
}

#[test]
fn readiness_v3_budget_boundary_counts_scans_and_never_uses_walls() {
    let matrix = wide_inputs();
    let mut rows = borrowed(&matrix);
    let mut measured = work(u64::MAX);
    assert_eq!(
        input_geometry_readiness_v3(&rows, &settings(), &mut measured),
        Ok(12)
    );
    let required = measured.used;
    for row in &mut rows {
        row.wall_ns = u64::MAX;
    }
    let mut exact = work(required);
    assert_eq!(
        input_geometry_readiness_v3(&rows, &settings(), &mut exact),
        Ok(12)
    );
    assert_eq!(exact.used, required);
    let mut short = work(required - 1);
    assert_eq!(
        input_geometry_readiness_v3(&rows, &settings(), &mut short),
        Err(StructuredUnknown::Capacity)
    );
    assert!(short.exhausted && short.used <= short.limit);
    let mut reused = work(required);
    reused.used = 1;
    assert!(input_geometry_readiness_v3(&rows, &settings(), &mut reused).is_err());
    assert!(
        reused.exhausted,
        "a new prefix cannot reset the original cumulative allowance"
    );
}

#[test]
fn readiness_v3_validates_even_columns_that_would_otherwise_be_zero() {
    for invalid in [f64::NAN, f64::INFINITY, -1., (1u64 << 54) as f64] {
        let mut matrix = wide_inputs();
        matrix[421][302] = invalid;
        assert_eq!(
            input_geometry_readiness_v3(&borrowed(&matrix), &settings(), &mut work(u64::MAX)),
            Err(StructuredUnknown::InvalidInput)
        );
    }
    let empty_work = vec![vec![0.; 9]; 16];
    assert_eq!(
        input_geometry_readiness_v3(&borrowed(&empty_work), &settings(), &mut work(u64::MAX)),
        Err(StructuredUnknown::Numerical)
    );
}
