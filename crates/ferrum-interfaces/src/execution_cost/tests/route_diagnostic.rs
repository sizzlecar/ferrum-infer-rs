use super::*;

fn selected(reason: PreparedCostRouteReasonV1) -> PreparedCostRouteV1 {
    PreparedCostRouteV1 {
        class: PreparedCostRouteClassV1::Unknown,
        reason,
        program_id: None,
        non_reusable_wave: None,
        lane_id: 3,
        lane_epoch: 7,
        catalog_epoch: Some(6),
        graph_state: None,
        batch_step: Some(11),
        batch_invocation: Some(12),
    }
}

#[test]
fn route_diagnostic_retains_selection_after_rejection_without_restoring_receipt() {
    for enabled in [false, true] {
        let mut recorder = recorder(1);
        if enabled {
            recorder.enable_route_diagnostics();
        }
        let clock = TestClock(std::sync::atomic::AtomicU64::new(2));
        {
            let mut context = PlanRuntimeCostObservationContext::new(
                &mut recorder,
                &clock,
                &[],
                Some(1),
                WaveObservationBoundary::IsolatedPreparationToCommit,
            );
            context.prepared_route(
                selected(PreparedCostRouteReasonV1::CatalogEpochMismatch),
                Ok(Vec::new()),
            );
            context.route_submission(None);
        }
        assert!(recorder.prepared_route().is_none());
        assert!(recorder.observations().is_empty());
        assert_eq!(recorder.route_diagnostic().is_some(), enabled);
        if let Some(diagnostic) = recorder.route_diagnostic() {
            let prepared = diagnostic.prepared.unwrap();
            assert_eq!(
                prepared.reason,
                PreparedCostRouteReasonV1::CatalogEpochMismatch
            );
            assert_eq!((prepared.lane_epoch, prepared.catalog_epoch), (7, Some(6)));
            assert_eq!(
                diagnostic.first_rejection,
                Some("submission_attribution_missing")
            );
            assert!(diagnostic.submitted.is_none());
        }
    }
}

#[test]
fn route_diagnostic_keeps_first_clock_failure_across_later_preparations() {
    let mut recorder = recorder(1);
    recorder.enable_route_diagnostics();
    let clock = TestClock(std::sync::atomic::AtomicU64::new(2));
    {
        let mut context = PlanRuntimeCostObservationContext::new(
            &mut recorder,
            &clock,
            &[],
            Some(3),
            WaveObservationBoundary::IsolatedPreparationToCommit,
        );
        context.prepared_route(
            selected(PreparedCostRouteReasonV1::ProgramIdentityUnavailable),
            Ok(Vec::new()),
        );
        context.prepared_route(
            selected(PreparedCostRouteReasonV1::CatalogUnavailable),
            Ok(Vec::new()),
        );
        context.route_submission(None);
    }
    let diagnostic = recorder.route_diagnostic().unwrap();
    assert_eq!(diagnostic.preparation_attempts, 2);
    assert_eq!(
        diagnostic.first_rejection,
        Some("selection_before_preparation_or_missing_clock")
    );
    assert_eq!(
        diagnostic.prepared.unwrap().reason,
        PreparedCostRouteReasonV1::ProgramIdentityUnavailable
    );
    assert!(recorder.prepared_route().is_none());
    assert!(recorder.observations().is_empty());
}
