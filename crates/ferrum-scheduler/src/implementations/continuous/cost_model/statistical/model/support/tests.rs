use super::*;

fn point(x: u64, y: u64) -> [u64; AXES] {
    let mut result = [10; AXES];
    result[0] = x;
    result[AXES - 1] = y;
    result
}

// Independent definition of the old joint predicate. In particular, the lower
// bounds come from ALL observations, not just points retained in the index.
fn original_contains(points: &[[u64; AXES]], query: &[u64; AXES]) -> bool {
    (0..AXES).all(|axis| {
        let min = points.iter().map(|p| p[axis]).min().unwrap();
        let max = points.iter().map(|p| p[axis]).max().unwrap();
        query[axis] >= min && query[axis] <= max
    }) && points
        .iter()
        .any(|p| (0..AXES).all(|axis| query[axis] <= p[axis]))
}

#[test]
fn skyline_matches_all_small_joint_domains_and_input_orders() {
    let universe: Vec<_> = (1..=3)
        .flat_map(|x| (1..=3).map(move |y| point(x, y)))
        .collect();
    for mask in 1..(1usize << universe.len()) {
        let points: Vec<_> = universe
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (mask & (1 << i) != 0).then_some(*p))
            .collect();
        let forward = Support::new(points.iter().copied()).unwrap();
        let reverse = Support::new(points.iter().rev().copied()).unwrap();
        assert_eq!(forward.points, reverse.points);
        for x in 0..=4 {
            for y in 0..=4 {
                let query = point(x, y);
                let expected = original_contains(&points, &query);
                assert_eq!(
                    forward.contains(&query),
                    expected,
                    "mask={mask} q={query:?}"
                );
                assert_eq!(reverse.contains(&query), expected);
            }
        }
    }
}

#[test]
fn skyline_keeps_original_minimum_and_rejects_unobserved_cartesian_corner() {
    let points = [point(1, 1), point(1, 4), point(4, 1)];
    let support = Support::new(points.into_iter()).unwrap();
    assert_eq!(support.points.len(), 2);
    assert!(support.contains(&point(1, 1)));
    assert!(support.contains(&point(1, 3)));
    assert!(support.contains(&point(3, 1)));
    assert!(!support.contains(&point(3, 3)));
    assert!(!support.contains(&point(0, 1)));
    let chain = Support::new([point(1, 1), point(4, 4)].into_iter()).unwrap();
    assert_eq!(chain.points, vec![point(4, 4)]);
    assert!(chain.contains(&point(2, 2)));
    assert!(!chain.contains(&point(0, 2)));
    assert!(!chain.contains(&point(5, 2)));
}

#[test]
fn skyline_checks_every_axis_and_preserves_u64_edges_without_arithmetic() {
    let low = [0; AXES];
    let high = [u64::MAX; AXES];
    let support = Support::new([low, high, high].into_iter()).unwrap();
    assert_eq!(support.points, vec![high]);
    for axis in 0..AXES {
        let mut query = low;
        query[axis] = u64::MAX;
        assert!(support.contains(&query));
        let mut a = high;
        let mut b = high;
        a[axis] = 0;
        b[(axis + 1) % AXES] = 0;
        let crossed = Support::new([a, b].into_iter()).unwrap();
        assert_eq!(crossed.points.len(), 2);
        assert!(!crossed.contains(&high));
    }
}

#[test]
fn skyline_compression_does_not_relax_original_observation_capacity() {
    assert!(matches!(
        Support::new(std::iter::empty()),
        Err(ModelUnknown::Capacity)
    ));
    assert!(matches!(
        Support::new(std::iter::repeat_n([1; AXES], 4097)),
        Err(ModelUnknown::Capacity)
    ));
    let at_limit = Support::new(std::iter::repeat_n([1; AXES], 4096)).unwrap();
    assert_eq!(at_limit.points, vec![[1; AXES]]);
}

#[test]
fn skyline_query_is_immutable_and_all_retained_points_are_real_observations() {
    let points = [point(3, 1), point(2, 2), point(1, 3), point(1, 1)];
    let support = Support::new(points.into_iter()).unwrap();
    let before = support.clone();
    for query in points {
        assert_eq!(support.contains(&query), original_contains(&points, &query));
    }
    assert_eq!(support.points, before.points);
    assert_eq!(support.minimum, before.minimum);
    assert_eq!(support.maximum, before.maximum);
    assert!(support.points.iter().all(|p| points.contains(p)));
}

#[test]
#[ignore = "diagnostic CPU benchmark; prints timings, never asserts a speed threshold"]
fn skyline_query_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;
    for count in [333usize, 4096] {
        for antichain in [false, true] {
            let points: Vec<_> = (0..count)
                .map(|i| {
                    point(
                        i as u64,
                        if antichain {
                            (count - i) as u64
                        } else {
                            i as u64
                        },
                    )
                })
                .collect();
            let started = Instant::now();
            let compact = Support::new(points.iter().copied()).unwrap();
            let build_ns = started.elapsed().as_nanos();
            // The old query already had frozen bounds; avoid timing the
            // independent test oracle's deliberate recomputation of min/max.
            let original = Support {
                minimum: compact.minimum,
                maximum: compact.maximum,
                points: points.clone(),
            };
            for query in &points {
                assert_eq!(original.contains(query), compact.contains(query));
            }
            let iterations = 16_384usize;
            let mut hit_counts = Vec::new();
            let mut timings = Vec::new();
            for support in [&original, &compact] {
                let started = Instant::now();
                let mut hits = 0usize;
                for iteration in 0..iterations {
                    hits += usize::from(
                        black_box(support).contains(black_box(&points[iteration % count])),
                    );
                }
                timings.push(started.elapsed().as_nanos());
                hit_counts.push(black_box(hits));
            }
            assert_eq!(hit_counts[0], hit_counts[1]);
            eprintln!("support skyline points={count} retained={} antichain={antichain} build_ns={build_ns} queries={iterations} original_ns={} compact_ns={}", compact.points.len(), timings[0], timings[1]);
        }
    }
}
