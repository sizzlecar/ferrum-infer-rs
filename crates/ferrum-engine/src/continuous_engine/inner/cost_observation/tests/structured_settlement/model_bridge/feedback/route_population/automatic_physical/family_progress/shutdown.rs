//! The final original block is queued by producers and drained only by the
//! shared runtime's shutdown path. An existing A monitor must remain open long
//! enough to install B; stopping never manufactures an absent eighth offer.
use super::*;

fn before_final_block() -> Families {
    let mut f = Families::new();
    for algorithm in [A, A, B, B, A, A, B] {
        f.block(algorithm);
    }
    assert_eq!(f.live().audit().qualified_publications, 1);
    assert!(f.known(A));
    assert!(!f.known(B));
    let feedback = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(feedback.scopes.len(), 1);
    assert!(!feedback.persistence_failed);
    // Freeze B's Qualification assignment at the real preceding FIFO cut.
    f.runtime.consume_samples();
    assert_eq!(f.live().audit().population.phase, 8);
    assert_eq!(f.live().audit().population.issued, 0);
    f
}

fn queue_without_consuming(f: &mut Families, count: usize) {
    let resolved = f.runtime.audit_snapshot().sink.raw_resolved;
    for _ in 0..count {
        let w = Families::wave(B);
        let rows = w.actual.rows.clone();
        let stages = fixture::record_cohort_route(&f.runtime, &f.clock, w)
            .expect("real original CPU submission and complete host settlement");
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert!(stages
            .structured_evidence
            .as_ref()
            .is_some_and(Result::is_ok));
        assert_eq!(stages.rows.len(), rows.len());
        for (original, settled) in rows.iter().zip(&stages.rows) {
            assert_eq!(settled.request_id, original.request_id);
            assert_eq!(settled.owner_incarnation, original.owner_incarnation);
            assert_eq!(settled.work_generation, original.work_generation);
            assert_eq!(settled.input_index, original.input_index);
        }
        f.recorded += 1;
    }
    let audit = f.live().audit();
    assert_eq!(
        audit.qualified_publications, 1,
        "shutdown has not drained yet"
    );
    assert_eq!(audit.population.declared_offers, OFFERS);
    assert_eq!(audit.population.issued, count);
    assert_eq!(audit.population.retired, 0);
    let sink = f.runtime.audit_snapshot().sink;
    assert_eq!(sink.raw_offered, f.recorded);
    assert_eq!(sink.raw_accepted, f.recorded);
    assert_eq!(sink.raw_resolved, resolved);
    assert_eq!(sink.raw_pending, count as u64);
    assert_eq!(sink.raw_lost, 0);
}

fn assert_originals_drained(f: &Families, count: usize) {
    let sink = f.runtime.audit_snapshot().sink;
    assert_eq!(sink.raw_offered, f.recorded);
    assert_eq!(sink.raw_accepted, f.recorded);
    assert_eq!(sink.raw_resolved, f.recorded);
    assert_eq!(sink.raw_resolution_failed, 0);
    assert_eq!(sink.raw_lost, 0);
    assert_eq!(sink.raw_pending, 0);
    let audit = f.live().audit();
    assert_eq!(audit.population.declared_offers, OFFERS);
    assert_eq!(audit.population.issued, count);
    assert_eq!(audit.population.retired, count);
    assert_eq!(audit.population.eligible_route, count);
    let automatic = audit.automatic.as_ref().unwrap();
    assert!(automatic.stopped);
    assert!(
        automatic.owner_blocks.is_none(),
        "the stopped epoch must be sealed: {audit:#?}"
    );
    // Shutdown closes execution authority even when it installs a last model.
    assert!(f.runtime.snapshot().is_none());
}

#[tokio::test]
async fn automatic_family_progress_shutdown_drains_complete_last_block_and_publishes() {
    let mut f = before_final_block();
    let first_receipt = f.runtime.training.published_catalog_receipt().unwrap();
    queue_without_consuming(&mut f, OFFERS);

    f.runtime.shutdown().await.unwrap();

    assert_originals_drained(&f, OFFERS);
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 2, "{audit:#?}");
    assert_eq!(audit.failed_generations, 0, "{audit:#?}");
    assert!(audit.publication_error.is_none(), "{audit:#?}");
    assert!(!audit.population.failed);
    let receipt = f.runtime.training.published_catalog_receipt().unwrap();
    assert!(receipt.model_version > first_receipt.model_version);
    assert_eq!(receipt.offered_samples, 8 * OFFERS);
    assert_eq!(receipt.recorded_samples, 3 * OFFERS);
    assert_eq!(receipt.storage, SloCostProfileStorage::Memory);
    assert!(receipt.path.is_none());
    let declared = receipt.structured_whole_wave_v2.as_ref().unwrap();
    assert_eq!(declared.child_count, 1, "only newly qualified B is rebound");
    let first = first_receipt.structured_whole_wave_v2.as_ref().unwrap();
    assert_ne!(
        declared.children[0].domain_signature,
        first.children[0].domain_signature
    );
    let feedback = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(
        feedback.scopes.len(),
        2,
        "A remains installed when B is added"
    );
    assert!(feedback
        .scopes
        .iter()
        .any(|scope| scope.signature == first.children[0].domain_signature));
    assert!(feedback
        .scopes
        .iter()
        .any(|scope| scope.signature == declared.children[0].domain_signature));
    assert!(!feedback.persistence_failed, "{feedback:#?}");
    assert!(!feedback.worker_failed, "{feedback:#?}");
}

#[tokio::test]
async fn automatic_family_progress_shutdown_cannot_complete_seven_of_eight_offers() {
    let mut f = before_final_block();
    let first_receipt =
        serde_json::to_value(f.runtime.training.published_catalog_receipt().unwrap()).unwrap();
    queue_without_consuming(&mut f, OFFERS - 1);

    f.runtime.shutdown().await.unwrap();

    assert_originals_drained(&f, OFFERS - 1);
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 1, "{audit:#?}");
    assert_eq!(audit.failed_generations, 1, "{audit:#?}");
    assert!(audit.population.failed);
    assert_eq!(
        serde_json::to_value(f.runtime.training.published_catalog_receipt().unwrap()).unwrap(),
        first_receipt
    );
    let feedback = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(feedback.scopes.len(), 1, "unqualified B must not replace A");
    assert!(!feedback.persistence_failed, "{feedback:#?}");
}
