//! Same original selector and controlled submission, with an explicitly fixed
//! private capture. No live population ticket supplies this route permission.
use super::*;

#[tokio::test]
async fn private_route_diagnostics_preserve_settlement_fifo_and_authority() {
    use ferrum_interfaces::execution_cost::PreparedCostRouteClassV1;

    for (authorized, outside, extra_unknown, missing_host_features) in [
        (true, false, false, false),
        (true, true, false, false),
        (true, true, true, false),
        (true, true, false, true),
        (false, true, false, false),
    ] {
        let mut baseline = None;
        for debug in [false, true] {
            let (f, outcome) =
                tracing::subscriber::with_default(diagnostics::RuntimeDebug(debug), || {
                    let f = Automatic::new(8);
                    let capture = if authorized {
                        CostCalibrationCapture::default()
                            .with_original_route_capture()
                            .unwrap()
                    } else {
                        CostCalibrationCapture::default()
                    };
                    let capture = Arc::new(capture);
                    let before = f.runtime.sink.stats().raw_accepted;
                    let stages = fixture::record_private_route_with_hook(
                        &f.runtime,
                        &f.clock,
                        wave("private.route-diagnostic"),
                        outside,
                        extra_unknown,
                        capture.clone(),
                        |call| {
                            if missing_host_features {
                                call.participants[0].host_features = None;
                            }
                            assert!(call.live_ticket.is_none());
                            let diagnostic = call.recorder.route_diagnostic();
                            assert_eq!(diagnostic.is_some(), authorized && debug);
                            if let Some(diagnostic) = diagnostic {
                                assert_eq!(diagnostic.preparation_attempts, 1);
                                let prepared = diagnostic.prepared.unwrap();
                                if outside {
                                    assert!(prepared.class.is_outside());
                                } else {
                                    assert_eq!(
                                        prepared.class,
                                        PreparedCostRouteClassV1::GraphDisabled
                                    );
                                }
                                let submitted = diagnostic.submitted.unwrap();
                                assert_eq!(submitted.lane_id, prepared.lane_id);
                                assert!(submitted.graph.is_some());
                                assert!(diagnostic.first_rejection.is_none());
                                assert_eq!(diagnostic.other_evidence_unknown, extra_unknown);
                            }
                            let mut gate = None;
                            assert_eq!(
                                call.make_route_evidence_recording_failure(&mut gate)
                                    .is_some(),
                                authorized && !extra_unknown && !missing_host_features
                            );
                            assert_eq!(
                                gate,
                                if debug && authorized && extra_unknown {
                                    Some("private_prepared_route_missing")
                                } else if debug && authorized && missing_host_features {
                                    Some("outside_host_features_missing")
                                } else {
                                    None
                                }
                            );
                        },
                    );
                    let accepted = authorized && !extra_unknown && !missing_host_features;
                    assert_eq!(stages.is_some(), accepted);
                    f.runtime.consume_samples();
                    assert_eq!(capture.host_stages().is_some(), accepted);
                    assert!(!matches!(capture.status(), CostCalibrationStatus::Pending));
                    let queue = capture.host_stage_queue();
                    if accepted {
                        let queue = queue.unwrap();
                        assert_eq!(queue.disposition, HostStageQueueDisposition::Published);
                        assert_eq!(queue.accepted_ordinal, Some(before + 1));
                    }
                    let audit = f.live().audit();
                    assert_eq!(audit.population.issued, 0);
                    assert_eq!(audit.qualified_publications, 0);
                    let outcome = (
                        stages.is_some(),
                        queue.map(|q| (q.disposition, q.accepted_ordinal.map(|n| n - before))),
                        f.runtime.sink.stats().raw_accepted - before,
                    );
                    (f, outcome)
                });
            if let Some(expected) = &baseline {
                assert_eq!(
                    &outcome, expected,
                    "diagnostics cannot alter settlement or FIFO"
                );
            } else {
                baseline = Some(outcome);
            }
            f.runtime.shutdown().await.unwrap();
        }
    }
}

#[tokio::test]
async fn private_route_capture_retains_original_warm_and_outside_receipts_without_live_ticket() {
    for outside in [false, true] {
        let f = Automatic::new(8);
        let capture = Arc::new(
            CostCalibrationCapture::default()
                .with_original_route_capture()
                .unwrap(),
        );
        let stages = fixture::record_private_route(
            &f.runtime,
            &f.clock,
            wave("private.route"),
            outside,
            false,
            capture.clone(),
        )
        .unwrap();
        assert_eq!(
            stages.route_evidence.as_ref().unwrap().is_outside(),
            outside
        );
        f.runtime.consume_samples();
        let resolved = capture.host_stages().expect("original worker settlement");
        assert_eq!(resolved.call_id, stages.call_id);
        assert_eq!(
            resolved.route_evidence.as_ref().unwrap().is_outside(),
            outside
        );
        let queue = capture.host_stage_queue().unwrap();
        assert_eq!(queue.disposition, HostStageQueueDisposition::Published);
        assert!(queue.accepted_ordinal.is_some());
        if outside {
            let receipt = resolved
                .route_evidence
                .as_ref()
                .unwrap()
                .outside_record(
                    &resolved,
                    1,
                    StructuredPhaseV2::Fit,
                    queue.accepted_ordinal.unwrap(),
                )
                .unwrap();
            assert_eq!(receipt.fifo, queue.accepted_ordinal.unwrap());
        }
        assert_eq!(
            f.runtime
                .audit_snapshot()
                .live_calibration
                .unwrap()
                .population
                .issued,
            0
        );
        // Ownership after the original call cannot retroactively enable a new
        // capture policy, even when all other Arc users have completed.
        let completed = Arc::try_unwrap(capture).expect("worker released capture");
        assert!(completed.with_original_route_capture().is_err());
        f.runtime.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn private_route_capture_requires_prior_permission_and_keeps_unknown_failure() {
    for authorized in [false, true] {
        let f = Automatic::new(8);
        let capture = if authorized {
            CostCalibrationCapture::default()
                .with_original_route_capture()
                .unwrap()
        } else {
            CostCalibrationCapture::default()
        };
        let capture = Arc::new(capture);
        // Unrequested capture is insufficient; a requested capture with real
        // additional InvalidLifecycle remains ineligible despite GraphPath.
        let stages = fixture::record_private_route(
            &f.runtime,
            &f.clock,
            wave("private.route.rejected"),
            true,
            authorized,
            capture.clone(),
        );
        assert!(stages.is_none());
        f.runtime.consume_samples();
        assert!(capture.host_stages().is_none());
        assert!(!matches!(capture.status(), CostCalibrationStatus::Pending));
        assert_eq!(
            f.runtime
                .audit_snapshot()
                .live_calibration
                .unwrap()
                .population
                .issued,
            0
        );
        f.runtime.shutdown().await.unwrap();
    }
    let f = Automatic::with_population(8, SloCalibrationRoutePopulationV1::AllAttempts);
    let stages = fixture::record_route_with_hook(
        &f.runtime,
        &f.clock,
        wave("legacy.all-attempts"),
        false,
        false,
        false,
        |_| {},
    )
    .unwrap();
    assert!(stages.route_evidence.is_none());
    f.runtime.consume_samples();
    f.runtime.shutdown().await.unwrap();
}
