use super::*;

#[test]
fn required_query_observation_bounds_cloned_algorithm_and_terminal_allocations() {
    let mut query = StructuredQueryV2::exact(
        project(&wave(0, true, 8, "fixture.first", [false, false])).unwrap(),
    );
    let before = query.observation_retained_bytes().unwrap();
    let axis_capacity = query.input.algorithm_axes.capacity();
    query.input.algorithm_axes.reserve_exact(axis_capacity + 31);
    let axes_bytes = (query.input.algorithm_axes.capacity() - axis_capacity)
        * std::mem::size_of::<super::super::algorithm_universe::AlgorithmAxisV1>();
    assert!(axes_bytes > 0);
    assert_eq!(
        query.observation_retained_bytes(),
        Some(before + axes_bytes)
    );

    let mut causes = Vec::with_capacity(17);
    causes.push((0, ferrum_types::FinishReason::Length));
    let causes_bytes = causes.capacity() * std::mem::size_of::<(u32, ferrum_types::FinishReason)>();
    query.input.settled_terminal_causes = Some(causes);
    let retained = query.observation_retained_bytes().unwrap();
    assert_eq!(retained, before + axes_bytes + causes_bytes);

    // The producer reserves this amount before deep cloning into its bounded
    // queue. The cloned payload must fit even when the original keeps spare
    // capacity or optional terminal evidence.
    let queued = query.clone();
    assert!(queued.retained_payload_bytes().unwrap() <= retained);
    assert_eq!(queued.input.algorithm_axes, query.input.algorithm_axes);
    assert_eq!(
        queued.input.settled_terminal_causes,
        query.input.settled_terminal_causes
    );
}

#[test]
fn required_future_coverage_retains_fixed_peer_and_joint_length_count() {
    // The FullLogits anchor marks the eligible row pending. Row 0 remains a
    // fixed FullLogits peer, so this route admits AnySubset, never NonEmpty.
    let wave = wave(1, true, 16, "fixture.first", [true, true]);
    let selected = wave.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let exact = StructuredQueryV2::from_future(
        &wave.exact,
        selected,
        recipe,
        &HostContentForecastV2::Exact,
    )
    .unwrap();
    assert_eq!(
        exact.required_coverage().unwrap().reachable_joint_counts,
        [(2, 1)]
    );
    assert!(matches!(
        HostPendingSetV2::new(
            &wave.exact,
            recipe,
            &[1],
            HostPendingConstraintV2::NonEmptySubset
        ),
        Err(StatisticalEvidenceUnknown::MissingHostDomain)
    ));
    for (constraint, expected) in [(HostPendingConstraintV2::AnySubset, vec![(1, 1), (2, 1)])] {
        let forecast = HostContentForecastV2::Unresolved(
            HostPendingSetV2::new(&wave.exact, recipe, &[1], constraint).unwrap(),
        );
        let query =
            StructuredQueryV2::from_future(&wave.exact, selected, recipe, &forecast).unwrap();
        let demand = query.required_coverage().unwrap();
        assert_eq!(demand.fixed_pending_positions, [0]);
        assert_eq!(demand.eligible_pending_positions, [1]);
        assert_eq!(demand.reachable_joint_counts, expected);
        assert_eq!(demand.length_positions, [1]);
        assert_eq!(
            demand.joint_support_coordinates,
            query.input.joint_support_coordinates()
        );
        assert_eq!(demand.owner, *query.owner());
        assert_eq!(demand.domain_signature, *query.domain_signature());
    }
}

#[test]
fn required_future_coverage_does_not_authorize_unobserved_pair() {
    let wave = wave(1, true, 16, "fixture.first", [true, true]);
    let selected = wave.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let forecast = HostContentForecastV2::Unresolved(
        HostPendingSetV2::new(
            &wave.exact,
            recipe,
            &[0, 1],
            HostPendingConstraintV2::NonEmptySubset,
        )
        .unwrap(),
    );
    let query = StructuredQueryV2::from_future(&wave.exact, selected, recipe, &forecast).unwrap();
    let demand = query.required_coverage().unwrap();
    assert_eq!(demand.reachable_joint_counts, [(1, 1), (2, 1)]);
    let mut scope = StructuredScopeV2 {
        owner: demand.owner,
        numerical_family: None,
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![0, 1],
            authorized_pending_constraints: vec![HostPendingConstraintV2::NonEmptySubset],
            pending_counts: vec![1, 2],
            length_counts: vec![1],
            pending_positions: vec![0, 1],
            length_positions: vec![1],
            joint_counts: vec![(2, 1)],
        },
    };
    assert_eq!(
        scope.authorize(&query),
        Err(StructuredUnknown::QualificationCoverage)
    );
    scope.coverage.joint_counts.insert(0, (1, 1));
    assert!(scope.authorize(&query).is_ok());
    // Authorization still does not supply any phase population or fitted model.
    assert!(!scope.coverage_report(&[]).unwrap().complete());
}
