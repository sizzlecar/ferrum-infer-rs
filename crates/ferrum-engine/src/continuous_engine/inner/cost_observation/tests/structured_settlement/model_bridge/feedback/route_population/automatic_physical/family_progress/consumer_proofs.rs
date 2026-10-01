//! A completed source block does not end the independently enabled feedback
//! consumer. Its pristine original no-submit proof must survive without a vote
//! in the next source block. No source membership or cost value is fabricated.

#[tokio::test]
async fn automatic_feedback_original_outside_route_survives_closed_block_without_live_ticket() {
    let mut f = Families::new();
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
    for _ in 0..4 {
        f.block(A);
    }
    let original = f.runtime.snapshot().unwrap();
    let before = f.runtime.audit_snapshot();
    let stages = fixture::record_feedback_route(&f.runtime, &f.clock, wave(A), false)
        .expect("original outside submission and host settlement");
    assert!(stages.route_evidence.as_ref().unwrap().is_outside());
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    f.runtime.consume_samples();
    let after = f.runtime.audit_snapshot();
    let feedback = after.structured_feedback.unwrap();
    assert_eq!(feedback.revoked, None, "{feedback:#?}");
    assert_eq!(
        feedback.outside_route_observations,
        before
            .structured_feedback
            .unwrap()
            .outside_route_observations
            + 1
    );
    // No missing ticket is replaced with synthetic source membership.
    assert!(before.live_calibration.unwrap().population.closed);
    assert_eq!(after.live_calibration.unwrap().population.issued, 0);
    assert!(original.current());
    assert!(f.known(A));
    assert_eq!(
        after.sink.raw_resolution_failed,
        before.sink.raw_resolution_failed
    );
    assert_eq!(
        after.sink.memory.worker_and_result_bytes, runtime_bytes,
        "closed source storage and feedback-only evidence release back to the empty maintenance consumer"
    );
    drop(original);
    drop(stages);
    f.runtime.shutdown().await.unwrap();
    // Keeping the typed sink alive pins exactly the runtime base reservation.
    // Dropping its final owner releases that reservation as well.
    drop(f);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, runtime_bytes);
    drop(maintenance_owner);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, 0);
}

#[tokio::test]
async fn automatic_feedback_outside_route_still_revokes_incomplete_execution() {
    let mut f = Families::new();
    for _ in 0..4 {
        f.block(A);
    }
    assert!(f.known(A));
    let before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert!(fixture::record_feedback_route(&f.runtime, &f.clock, wave(A), true).is_none());
    f.runtime.consume_samples();
    let after = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert!(after.revoked.is_some(), "{after:#?}");
    assert_eq!(
        after.uncomparable_observations,
        before.uncomparable_observations + 1
    );
    assert_eq!(
        after.outside_route_observations,
        before.outside_route_observations
    );
    assert!(!f.known(A));
    f.runtime.shutdown().await.unwrap();
}

use super::*;

#[tokio::test]
async fn automatic_feedback_original_no_submission_survives_closed_block_without_live_ticket() {
    let mut f = Families::new();
    for _ in 0..4 {
        f.block(A);
    }
    assert!(f.known(A));
    assert_eq!(f.live().audit().qualified_publications, 1);
    let original = f.runtime.snapshot().unwrap();
    let before = f.runtime.audit_snapshot();
    let at = f.clock.now_ns().unwrap() + 100;
    f.clock.set(at);
    let ticket = f.runtime.reserve_live_ticket(Some(at));
    assert!(ticket.is_none(), "the just-published block is still closed");
    let w = Families::wave(A);
    let mut call = EngineCostCall::begin(
        &f.runtime.ids,
        f.clock.clone(),
        f.runtime.sink.clone(),
        EngineCostCallSpec {
            identity: identity(),
            participants: w
                .actual
                .rows
                .iter()
                .map(|row| CostObservationParticipant {
                    request_id: row.request_id.clone(),
                    owner_incarnation: row.owner_incarnation,
                    work_generation: row.work_generation,
                    input_index: row.input_index,
                    output_policy_signature: Some([6; 32]),
                    host_features: Some(w.host),
                })
                .collect(),
            prepare_started_at_ns: Some(at),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 8,
                max_retained_rows: 128,
            },
        },
    )
    .unwrap()
    .with_live_ticket(ticket);
    {
        let mut context = call.context().unwrap();
        f.clock.set(at + 3);
        context.finish_capacity_deferred(&super::super::super::no_submission::capacity());
        f.clock.set(at + 5);
    }
    assert!(
        call.recorder.no_submission().is_some(),
        "the original recorder minted the proof"
    );
    f.clock.set(at + 8);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    assert_eq!(
        f.live().audit().population.issued,
        before.live_calibration.as_ref().unwrap().population.issued
    );
    f.clock.set(at + 20);
    f.runtime.consume_samples();
    let after = f.runtime.audit_snapshot();
    assert_eq!(after.sink.raw_accepted, before.sink.raw_accepted + 1);
    assert_eq!(after.sink.raw_resolved, before.sink.raw_resolved + 1);
    assert_eq!(after.sink.raw_lost, before.sink.raw_lost);
    let before_feedback = before.structured_feedback.unwrap();
    let feedback = after.structured_feedback.unwrap();
    assert!(
        feedback.revoked.is_none(),
        "pristine deferred call is not an unknown execution: {feedback:#?}"
    );
    assert_eq!(
        feedback.uncomparable_observations,
        before_feedback.uncomparable_observations
    );
    assert_eq!(
        feedback.no_submission_observations,
        before_feedback.no_submission_observations + 1
    );
    assert!(original.current());
    assert!(f.known(A));
    f.runtime.shutdown().await.unwrap();
}
