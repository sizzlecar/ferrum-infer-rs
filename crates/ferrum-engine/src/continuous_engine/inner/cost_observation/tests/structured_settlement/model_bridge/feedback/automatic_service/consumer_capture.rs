use super::*;

#[tokio::test]
async fn consumer_capture_tracks_real_tickets_and_published_feedback_through_epoch_close() {
    let f = AutomaticFixture::with_capture_policy(
        SloAutomaticCalibrationDiagnosticsV1::MemoryOnly,
        ferrum_types::SloStructuredActualCapturePolicy::ConsumerDrivenV1,
    );
    assert!(f.runtime.reserve_live_ticket(f.clock.now_ns()).is_none());
    let opening = f
        .runtime
        .actual_capture_demand(&f.config, false, false, false);
    assert_eq!(opening.source_generation, 0);
    assert!(!opening.structured_sample);
    assert!(
        f.runtime
            .actual_capture_demand(&f.config, true, false, false)
            .structured_sample
    );
    assert!(
        f.runtime
            .actual_capture_demand(&f.config, false, false, true)
            .structured_sample
    );

    f.start();
    // Original private tickets and real host settlements pass every discovery
    // and three-phase offer through the same production capture decision.
    f.generation(1);
    let published = f
        .runtime
        .actual_capture_demand(&f.config, false, false, false);
    assert!(published.source_generation > 0);
    assert!(
        published.structured_sample,
        "no ticket is needed by installed feedback"
    );
    // A demand captured before publication remains bound to its old source;
    // publication cannot relabel it as a current feedback observation.
    assert_eq!(opening.source_generation, 0);
    assert!(!opening.structured_sample);

    let before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    f.record(501, "fixture.feedback.a", false);
    let compared = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(compared.compared, before.compared + 1);
    assert!(compared.revoked.is_none());

    f.runtime.training.close_structured_epoch();
    let closed = f
        .runtime
        .actual_capture_demand(&f.config, false, false, false);
    assert_eq!(closed.source_generation, published.source_generation);
    assert!(closed.structured_sample);
    f.record(502, "fixture.feedback.a", false);
    let revoked = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert!(
        revoked.revoked.is_some(),
        "closed evidence must still reach the feedback failure gate"
    );
    let after_revoke = f
        .runtime
        .actual_capture_demand(&f.config, false, false, false);
    assert_eq!(after_revoke.source_generation, published.source_generation);
    assert!(after_revoke.structured_sample);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn consumer_capture_preserves_legacy_and_disabled_runtime_gates() {
    let mut f = AutomaticFixture::new();
    assert!(
        f.runtime
            .actual_capture_demand(&f.config, false, false, false)
            .structured_sample
    );
    f.runtime.structured_capture = false;
    assert!(
        !f.runtime
            .actual_capture_demand(&f.config, true, true, true)
            .structured_sample
    );
    f.runtime.shutdown().await.unwrap();
}
