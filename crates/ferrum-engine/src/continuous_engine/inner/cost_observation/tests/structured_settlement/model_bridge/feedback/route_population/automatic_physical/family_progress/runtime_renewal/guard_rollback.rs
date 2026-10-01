//! A real native rejection after preparation remains a non-numerical offer.
//! Three independent original phases can still renew the active source7 base.
use super::*;
use fixture::GuardRollbackFault;

#[tokio::test]
async fn guard_rollback_original_fifo_preserves_catalog_and_source7_renewal() {
    let mut f = Families::new();
    complete_epoch(&mut f, 1, 0);
    let first = only_child(&f);
    let snapshot = f.runtime.snapshot().unwrap();
    f.runtime.consume_samples();
    let before = f.runtime.audit_snapshot();
    fixture::record_guard_rollback(
        &f.runtime,
        &f.clock,
        Families::wave(A),
        GuardRollbackFault::None,
    );
    f.recorded += 1;
    let after = f.runtime.audit_snapshot();
    assert_eq!(
        after.sink.raw_no_submission,
        before.sink.raw_no_submission + 1
    );
    assert_eq!(after.sink.raw_resolution_failed, 0);
    let feedback = after.structured_feedback.unwrap();
    assert!(feedback.revoked.is_none());
    assert_eq!(
        feedback.no_submission_observations,
        before
            .structured_feedback
            .unwrap()
            .no_submission_observations
            + 1
    );
    assert!(snapshot.current());
    assert_eq!(f.live().audit().population.no_submission, 1);
    // Complete this original Discovery block with seven real physical waves.
    for _ in 1..OFFERS {
        f.record(A);
    }
    for _ in 0..3 {
        f.block(A);
    }
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 2, "{audit:#?}");
    assert_eq!(audit.failed_generations, 0);
    let second = only_child(&f);
    assert_eq!(
        second.provenance().phases.each_ref().map(|p| p.members),
        [OFFERS; 3]
    );
    assert_ne!(
        second.provenance().source_sha256,
        first.provenance().source_sha256
    );
    assert!(!snapshot.current());
    let renewed = f.runtime.snapshot().unwrap();
    let predicted = renewed
        .audit_structured_query_v2(&f.query(A), f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(predicted.model_version, renewed.model_version());
    assert!(f
        .runtime
        .audit_snapshot()
        .structured_feedback
        .unwrap()
        .revoked
        .is_none());
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn guard_rollback_requires_bound_participants_one_attempt_and_complete_observation() {
    for fault in [
        GuardRollbackFault::UnboundReceipt,
        GuardRollbackFault::ChangedParticipant,
        GuardRollbackFault::RepeatedPreparation,
        GuardRollbackFault::Lost,
        GuardRollbackFault::Unknown,
        GuardRollbackFault::ObservedWave,
    ] {
        let mut f = Families::new();
        complete_epoch(&mut f, 1, 0);
        f.runtime.consume_samples();
        let before = f.runtime.audit_snapshot().sink.raw_no_submission;
        let original_snapshot = f.runtime.snapshot().unwrap();
        fixture::record_guard_rollback(&f.runtime, &f.clock, Families::wave(A), fault);
        let audit = f.runtime.audit_snapshot();
        assert_eq!(audit.sink.raw_no_submission, before);
        assert_eq!(audit.sink.raw_resolution_failed, 1);
        assert!(audit.structured_feedback.unwrap().revoked.is_some());
        assert!(!original_snapshot.current());
        assert!(f.runtime.snapshot().is_none());
        assert_eq!(f.live().audit().qualified_publications, 1);
        f.runtime.shutdown().await.unwrap();
    }
}
