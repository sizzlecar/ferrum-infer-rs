use super::*;

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
