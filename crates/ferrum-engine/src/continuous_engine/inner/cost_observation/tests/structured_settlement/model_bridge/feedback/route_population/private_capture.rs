//! Same original selector and controlled submission, with an explicitly fixed
//! private capture. No live population ticket supplies this route permission.
use super::*;

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
