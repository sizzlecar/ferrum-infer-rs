//! The private worker accepts only already qualified imported children. Reuse
//! actual CPU source7/profile14 qualification to test that installation kernel;
//! this is not a claim that the separate startup source8 producer is complete.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    audit::CatalogExpiryReason, checkpoint::CheckpointRequestError, live_calibration::Publication,
};
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::StructuredUnknownV2, cost_profile::ImportedStructuredModelV2,
};
use ferrum_types::SloCostProfileReceipt;

struct Qualified {
    children: Vec<ImportedStructuredModelV2>,
    receipt: SloCostProfileReceipt,
    query: StructuredQueryV2,
    at: u64,
    expires: u64,
}

#[tokio::test(start_paused = true)]
async fn startup_series_waiter_drop_deadline_and_manual_guard_preserve_installed_prefix() {
    let qualified = Qualified::actual().await;
    let target = qualified.recipient();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(1);
    let mut series = target
        .runtime
        .begin_prepared_owner_series(2, deadline)
        .unwrap();
    let first = target
        .runtime
        .sink
        .request_catalog_activation_in_series(0, qualified.publication(), &mut series)
        .unwrap();
    assert_eq!(series.installed_epoch(), None);
    drop(first);
    target.runtime.consume_samples();
    let previous = target.runtime.snapshot().unwrap();
    assert_eq!(series.installed_epoch(), Some(previous.model_version()));
    let original = target.runtime.training.published_catalog_receipt().unwrap();
    let manual = target
        .runtime
        .sink
        .request_catalog_activation(0, qualified.publication())
        .unwrap();
    target.runtime.consume_samples();
    assert!(manual
        .wait()
        .await
        .unwrap()
        .catalog_activation
        .unwrap()
        .is_err());
    let second = target
        .runtime
        .sink
        .request_catalog_activation_in_series(0, qualified.publication(), &mut series)
        .unwrap();
    // The original caller may have timed out and dropped its waiter. Worker
    // authorization must still reject installation after the global deadline.
    tokio::time::advance(std::time::Duration::from_secs(1)).await;
    target.runtime.consume_samples();
    assert!(second
        .wait()
        .await
        .unwrap()
        .catalog_activation
        .unwrap()
        .is_err());
    assert!(Arc::ptr_eq(&previous, &target.runtime.snapshot().unwrap()));
    assert_eq!(
        target.runtime.training.published_catalog_receipt().unwrap(),
        original
    );
    assert_eq!(target.live().audit().qualified_publications, 1);
    target
        .runtime
        .finish_prepared_owner_series(&series)
        .unwrap();
    assert_eq!(series.installed_epoch(), Some(previous.model_version()));
    assert!(target
        .runtime
        .sink
        .request_catalog_activation_in_series(0, qualified.publication(), &mut series)
        .is_err());
    target.runtime.begin_automatic_calibration().unwrap();
    target.runtime.consume_samples();
    assert_eq!(target.live().audit().population.generation, 1);
    target.runtime.shutdown().await.unwrap();
}

impl Qualified {
    async fn actual() -> Self {
        let mut donor = Families::new();
        // Original Discovery/Fit/Residual/Qualification CPU execution. No
        // public receipt, raw axes or manually built model replaces training.
        for _ in 0..4 {
            donor.block(A);
        }
        assert_eq!(donor.live().audit().qualified_publications, 1);
        assert!(donor.known(A));
        let at = donor.clock.now_ns().unwrap();
        let children = donor.runtime.training.live_catalog_children(at).unwrap();
        assert_eq!(children.len(), 1);
        let receipt = donor.runtime.training.published_catalog_receipt().unwrap();
        let query = donor.query(A);
        let (prediction, model_now) = children[0]
            .predict_query_local_with_clock(children[0].fingerprint(), &query, at)
            .unwrap();
        let expires = at
            .checked_add(prediction.valid_until_ns.checked_sub(model_now).unwrap())
            .unwrap();
        donor.runtime.shutdown().await.unwrap();
        Self {
            children,
            receipt,
            query,
            at,
            expires,
        }
    }

    fn publication(&self) -> Publication {
        Publication {
            children: self.children.clone(),
            receipt: self.receipt.clone(),
        }
    }

    fn recipient(&self) -> Families {
        let target = Families::unstarted();
        target.clock.set(self.at);
        assert!(target.runtime.snapshot().is_none());
        let audit = target.live().audit();
        assert_eq!(audit.population.generation, 0);
        assert_eq!(audit.population.issued, 0);
        assert!(!audit.automatic.unwrap().start_requested);
        assert_eq!(self.children[0].workload_domain(), Some(&target.domain));
        target
    }
}

fn queue_original_receipt(target: &Families) -> u64 {
    // This original recorder/settlement helper fills FIFO only. It has no live
    // ticket and is not reused as a numerical member of the qualified source.
    let w = wave(A);
    let stages = record(
        &target.runtime.ids,
        &target.runtime.sink,
        &target.clock,
        w.actual,
        w.host,
        None,
        0,
    );
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    target.runtime.sink.stats().entries_published
}

#[tokio::test]
async fn startup_catalog_activation_preserves_real_model_age_and_starts_live_at_generation_one() {
    let qualified = Qualified::actual().await;
    let target = qualified.recipient();
    let original = serde_json::to_value(qualified.children[0].provenance()).unwrap();
    let delay = 10;
    target.clock.set(qualified.at + delay);
    let waiter = target
        .runtime
        .sink
        .request_catalog_activation(0, qualified.publication())
        .unwrap();
    assert!(target.runtime.snapshot().is_none());
    target.runtime.consume_samples();
    let cut = waiter.wait().await.unwrap();
    assert_eq!(cut.accepted_ordinal, 0);
    let epoch = cut.catalog_activation.unwrap().unwrap();
    let snapshot = cut.snapshot.unwrap();
    assert_eq!(snapshot.model_version(), epoch);
    assert_eq!(target.runtime.sink.source_generation(), epoch);
    let prediction = snapshot
        .audit_structured_query_v2(&qualified.query, target.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(
        prediction.valid_for_ns,
        qualified.expires - target.clock.now_ns().unwrap()
    );
    let children = target
        .runtime
        .training
        .live_catalog_children(target.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(
        serde_json::to_value(children[0].provenance()).unwrap(),
        original,
        "installation cannot rewrite source age, loaded time, phase clocks or hashes"
    );
    let audit = target.live().audit();
    assert_eq!(audit.qualified_publications, 1);
    let automatic = audit.automatic.unwrap();
    assert_eq!(automatic.startup_publication, Some(Ok(epoch)));
    assert_eq!(automatic.generation, 0);
    assert!(!automatic.start_requested);

    target.runtime.begin_automatic_calibration().unwrap();
    target.runtime.consume_samples();
    assert_eq!(target.live().audit().population.generation, 1);
    assert!(Arc::ptr_eq(&snapshot, &target.runtime.snapshot().unwrap()));
    target.clock.set(qualified.expires + 1);
    assert!(
        snapshot
            .audit_structured_query_v2(&qualified.query, target.clock.now_ns().unwrap())
            .is_err(),
        "startup activation must not renew an expired numerical model"
    );
    target.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn startup_catalog_activation_uses_exact_cut_and_excludes_post_cut_observations() {
    let qualified = Qualified::actual().await;
    let target = qualified.recipient();
    assert_eq!(queue_original_receipt(&target), 1);
    assert!(matches!(
        target
            .runtime
            .sink
            .request_catalog_activation(0, qualified.publication()),
        Err(CheckpointRequestError::CutoffChanged)
    ));
    let ordinary = target.runtime.request_checkpoint().unwrap();
    assert!(matches!(
        target
            .runtime
            .sink
            .request_catalog_activation(1, qualified.publication()),
        Err(CheckpointRequestError::Busy)
    ));
    target.runtime.consume_samples();
    assert_eq!(ordinary.wait().await.unwrap().accepted_ordinal, 1);
    assert!(target.runtime.snapshot().is_none());

    let activation = target
        .runtime
        .sink
        .request_catalog_activation(1, qualified.publication())
        .unwrap();
    assert!(matches!(
        target.runtime.request_checkpoint(),
        Err(CheckpointRequestError::Busy)
    ));
    assert_eq!(queue_original_receipt(&target), 2);
    target.runtime.consume_samples();
    let cut = activation.wait().await.unwrap();
    assert_eq!(cut.accepted_ordinal, 1);
    assert!(cut.catalog_activation.unwrap().is_ok());
    assert!(cut
        .snapshot
        .unwrap()
        .audit_structured_query_v2(&qualified.query, target.clock.now_ns().unwrap())
        .is_ok());
    assert_eq!(
        target.runtime.sink.stats().entries_drained,
        1,
        "the post-cut actual receipt cannot be consumed ahead of catalog activation"
    );
    target.runtime.consume_samples();
    assert_eq!(target.runtime.sink.stats().entries_drained, 2);
    assert!(target
        .runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&qualified.query, target.clock.now_ns().unwrap())
        .is_ok());
    target.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn startup_catalog_activation_survives_waiter_drop_without_reissuing_the_cut() {
    let qualified = Qualified::actual().await;
    let target = qualified.recipient();
    let waiter = target
        .runtime
        .sink
        .request_catalog_activation(0, qualified.publication())
        .unwrap();
    drop(waiter);
    assert!(target.runtime.snapshot().is_none());
    assert!(matches!(
        target.runtime.request_checkpoint(),
        Err(CheckpointRequestError::Busy)
    ));
    target.runtime.consume_samples();
    let snapshot = target.runtime.snapshot().unwrap();
    assert!(snapshot
        .audit_structured_query_v2(&qualified.query, qualified.at)
        .is_ok());
    assert_eq!(target.live().audit().qualified_publications, 1);
    let checkpoint = target.runtime.request_checkpoint().unwrap();
    target.runtime.consume_samples();
    let checkpoint = checkpoint.wait().await.unwrap();
    assert!(checkpoint.catalog_activation.is_none());
    assert!(Arc::ptr_eq(&snapshot, &checkpoint.snapshot.unwrap()));
    assert_eq!(target.live().audit().qualified_publications, 1);
    target.runtime.shutdown().await.unwrap();
}

#[derive(Debug, Clone, Copy)]
enum Refusal {
    BadReceipt,
    Expired,
    LiveStarted,
}

#[tokio::test]
async fn startup_catalog_activation_invalid_or_late_source_preserves_existing_snapshot() {
    let qualified = Qualified::actual().await;
    // Reuse one original imported source across recipients. Its age/clock are
    // never reanchored to make an expired source appear fresh.
    for refusal in [Refusal::BadReceipt, Refusal::Expired, Refusal::LiveStarted] {
        let target = qualified.recipient();
        let seed = qualified.publication();
        let epoch = target
            .runtime
            .training
            .publish_live_catalog(seed.children, seed.receipt, qualified.at)
            .unwrap();
        let previous = target.runtime.snapshot().unwrap();
        assert!(previous
            .audit_structured_query_v2(&qualified.query, qualified.at)
            .is_ok());
        let previous_receipt = target.runtime.training.published_catalog_receipt().unwrap();
        let mut publication = qualified.publication();
        match refusal {
            Refusal::BadReceipt => {
                publication
                    .receipt
                    .structured_whole_wave_v2
                    .as_mut()
                    .unwrap()
                    .children[0]
                    .source_sha256[0] ^= 1;
            }
            Refusal::Expired => target.clock.set(qualified.expires + 1),
            Refusal::LiveStarted => {
                target.runtime.begin_automatic_calibration().unwrap();
                target.runtime.consume_samples();
                assert_eq!(target.live().audit().population.generation, 1);
            }
        }
        let waiter = target
            .runtime
            .sink
            .request_catalog_activation(0, publication)
            .unwrap();
        target.runtime.consume_samples();
        let cut = waiter.wait().await.unwrap();
        assert!(cut.catalog_activation.unwrap().is_err(), "{refusal:?}");
        let cut_snapshot = cut.snapshot.unwrap();
        assert!(Arc::ptr_eq(&previous, &cut_snapshot), "{refusal:?}");
        assert_eq!(
            target.runtime.training.published_catalog_receipt().unwrap(),
            previous_receipt
        );
        assert_eq!(target.live().audit().qualified_publications, 0);
        let now = target.clock.now_ns().unwrap();
        let expiry = target.runtime.audit_snapshot().training.catalog_expiry;
        match refusal {
            Refusal::Expired => {
                // The rejected source and installed source share the original
                // age. The same worker turn freezes the refusal cut, then
                // removes the now-expired catalog; retaining the cut's Arc
                // cannot keep its publication gate or predictions valid.
                assert_eq!(now, qualified.expires + 1);
                assert_eq!(
                    qualified.children[0].is_current_local(now),
                    Err(StructuredUnknownV2::Stale)
                );
                assert!(target.runtime.snapshot().is_none());
                assert_eq!(target.runtime.sink.source_generation(), 0);
                assert!(!cut_snapshot.current());
                assert!(cut_snapshot
                    .audit_structured_query_v2(&qualified.query, now)
                    .is_err());
                let expiry =
                    expiry.expect("original model age must cause a typed expiry transition");
                assert_eq!(expiry.observed_at_ns, now);
                assert_eq!(expiry.previous_runtime_epoch, epoch);
                assert_eq!(expiry.current_runtime_epoch, 0);
                assert_eq!(expiry.previous_children, qualified.children.len());
                assert_eq!(expiry.current_children, 0);
                assert_eq!(expiry.removed_expired_children, qualified.children.len());
                assert_eq!(expiry.reason, CatalogExpiryReason::OriginalSampleAgeExpired);
            }
            Refusal::BadReceipt | Refusal::LiveStarted => {
                assert!(Arc::ptr_eq(&previous, &target.runtime.snapshot().unwrap()));
                assert_eq!(target.runtime.sink.source_generation(), epoch);
                assert!(cut_snapshot.current());
                assert!(cut_snapshot
                    .audit_structured_query_v2(&qualified.query, now)
                    .is_ok());
                assert!(expiry.is_none());
            }
        }
        target.runtime.shutdown().await.unwrap();
    }
}
