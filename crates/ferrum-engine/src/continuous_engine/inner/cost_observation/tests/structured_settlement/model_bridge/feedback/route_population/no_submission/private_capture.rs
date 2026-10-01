//! Original recorder proof survives the manual/startup capture without a live ticket.
use super::*;

#[tokio::test]
async fn private_no_submission_capture_preserves_original_fifo_without_live_population() {
    let f = v2(8);
    // This empty maintenance consumer owns a runtime-lifetime working-byte
    // permit independently of any original source block or private receipt.
    let maintenance_owner = f
        .runtime
        .prefix_cost_sink()
        .expect("typed maintenance sink");
    assert!(f.runtime.prefix_cost_snapshot().is_none());
    let sink = Arc::clone(&f.runtime.sink);
    let runtime_bytes = sink.stats().memory.worker_and_result_bytes;
    assert!(runtime_bytes > 0);
    let before = f.runtime.sink.stats();
    let capture = defer_original(&f, Fault::None, false);
    assert!(
        matches!(capture.status(), CostCalibrationStatus::Complete(value)
        if matches!(&*value, CostCalibrationResult::Rejected(_)))
    );
    let proof = capture
        .no_submission_proof()
        .expect("original producer proof");
    assert!(proof.call_id() > 0);
    assert_eq!(proof.fifo(), before.raw_accepted + 1);
    assert!(proof.observed_at_ns() > proof.issued_at_ns());
    assert_eq!(proof.participants().len(), 1);
    let wire = proof
        .bind_source_position(7, StructuredPhaseV2::Residual)
        .unwrap();
    assert_eq!(wire.ticket, 7);
    assert_eq!(wire.phase, StructuredPhaseV2::Residual);
    assert_eq!(wire.fifo, proof.fifo());
    assert_eq!(wire.call_id(), proof.call_id());
    assert_eq!(wire.issued_at_ns, proof.issued_at_ns());
    assert!(proof
        .bind_source_position(0, StructuredPhaseV2::Fit)
        .is_err());
    let diagnostic = serde_json::to_value(&wire).unwrap();
    assert_eq!(
        diagnostic["participants"][0]["request_id"],
        serde_json::to_value(&proof.participants()[0].request_id).unwrap()
    );
    assert_eq!(
        diagnostic["finalized_at_ns"],
        serde_json::json!(proof.observed_at_ns())
    );
    // Only the private source adapter can bind this original call to its own
    // predeclared offer. Ordinary live accounting did not silently gain a vote.
    let audit = f.runtime.audit_snapshot().live_calibration.unwrap();
    assert_eq!(audit.population.issued, 0);
    assert_eq!(audit.population.no_submission, 0);
    assert_eq!(
        f.runtime.sink.stats().raw_no_submission,
        before.raw_no_submission
    );
    assert!(capture.host_stages().is_none());
    let retained = f.runtime.sink.stats().memory.worker_and_result_bytes;
    let proof_bytes = retained.checked_sub(runtime_bytes).unwrap();
    assert!(proof_bytes >= proof.retained_bytes().unwrap());
    drop(capture);
    assert_eq!(
        f.runtime.sink.stats().memory.worker_and_result_bytes,
        retained
    );
    drop(proof);
    assert_eq!(
        f.runtime.sink.stats().memory.worker_and_result_bytes,
        runtime_bytes
    );
    f.runtime.shutdown().await.unwrap();
    // Keeping the typed sink alive pins exactly the runtime base reservation.
    // Dropping its final owner releases that reservation as well.
    drop(f);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, runtime_bytes);
    drop(maintenance_owner);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, 0);
}

#[tokio::test]
async fn private_no_submission_capture_never_promotes_generic_deferred_loss_or_host_failure() {
    let f = v2(8);
    // This empty maintenance consumer owns a runtime-lifetime working-byte
    // permit independently of any original source block or private receipt.
    let maintenance_owner = f
        .runtime
        .prefix_cost_sink()
        .expect("typed maintenance sink");
    assert!(f.runtime.prefix_cost_snapshot().is_none());
    let sink = Arc::clone(&f.runtime.sink);
    let runtime_bytes = sink.stats().memory.worker_and_result_bytes;
    assert!(runtime_bytes > 0);
    for fault in [
        Fault::GenericDeferred,
        Fault::Lost,
        Fault::Cancelled,
        Fault::Unknown,
    ] {
        let capture = defer_original(&f, fault, false);
        assert!(capture.no_submission_proof().is_none());
        assert!(!matches!(capture.status(), CostCalibrationStatus::Pending));
        drop(capture);
        assert_eq!(
            f.runtime.sink.stats().memory.worker_and_result_bytes,
            runtime_bytes
        );
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
    f.runtime.shutdown().await.unwrap();
    // Keeping the typed sink alive pins exactly the runtime base reservation.
    // Dropping its final owner releases that reservation as well.
    drop(f);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, runtime_bytes);
    drop(maintenance_owner);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, 0);
}

#[tokio::test]
async fn private_no_submission_duplicate_capture_is_conflicting_and_hides_original_proof() {
    let f = v2(8);
    // This empty maintenance consumer owns a runtime-lifetime working-byte
    // permit independently of any original source block or private receipt.
    let maintenance_owner = f
        .runtime
        .prefix_cost_sink()
        .expect("typed maintenance sink");
    assert!(f.runtime.prefix_cost_snapshot().is_none());
    let sink = Arc::clone(&f.runtime.sink);
    let runtime_bytes = sink.stats().memory.worker_and_result_bytes;
    assert!(runtime_bytes > 0);
    let capture = defer_original(&f, Fault::None, false);
    let proof = capture.no_submission_proof().unwrap();
    // Reuse an actually minted receipt to exercise duplicate completion; no
    // public diagnostic or manually fabricated receipt can take this path.
    capture.complete_no_submission(Arc::clone(&proof));
    assert!(matches!(
        capture.status(),
        CostCalibrationStatus::ConflictingCalls
    ));
    assert!(capture.no_submission_proof().is_none());
    drop(capture);
    drop(proof);
    assert_eq!(
        f.runtime.sink.stats().memory.worker_and_result_bytes,
        runtime_bytes
    );
    f.runtime.shutdown().await.unwrap();
    // Keeping the typed sink alive pins exactly the runtime base reservation.
    // Dropping its final owner releases that reservation as well.
    drop(f);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, runtime_bytes);
    drop(maintenance_owner);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, 0);
}
