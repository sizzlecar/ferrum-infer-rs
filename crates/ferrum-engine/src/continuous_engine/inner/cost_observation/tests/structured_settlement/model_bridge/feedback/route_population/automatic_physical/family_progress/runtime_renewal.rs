//! Renewal uses the default owner-block schedule and the original two-row CPU
//! producer. No imported child, phase sample or replacement receipt is supplied
//! by the test. The epoch checks exercise the runtime's real prediction gate;
//! complete controller/backend submission remains a separate integration gate.
use super::*;
#[path = "runtime_renewal/installation_events.rs"]
mod installation_events;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2;
use ferrum_scheduler::implementations::continuous::cost_profile::ImportedStructuredModelV2;
use installation_events::Installations;

#[path = "runtime_renewal/numerical_family.rs"]
mod numerical_family;

#[path = "runtime_renewal/guard_rollback.rs"]
mod guard_rollback;
#[path = "runtime_renewal/issued_source.rs"]
mod issued_source;

fn only_child(f: &Families) -> ImportedStructuredModelV2 {
    let mut children = f
        .runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(
        children.len(),
        1,
        "same owner must be replaced, not duplicated"
    );
    children.pop().unwrap()
}

fn expires_at(child: &ImportedStructuredModelV2, query: &StructuredQueryV2, at: u64) -> u64 {
    let (prediction, model_now) = child
        .predict_query_local_with_clock(child.fingerprint(), query, at)
        .unwrap();
    at.checked_add(prediction.valid_until_ns.checked_sub(model_now).unwrap())
        .unwrap()
}

fn complete_epoch(f: &mut Families, generation: u64, prior_publications: u64) {
    f.runtime.consume_samples();
    assert_eq!(f.live().audit().population.generation, generation);
    for _ in 0..3 {
        f.block(A);
        assert_eq!(
            f.live().audit().qualified_publications,
            prior_publications,
            "Discovery, Fit and Residual cannot replace independent Qualification"
        );
    }
    f.block(A);
    let audit = f.live().audit();
    assert_eq!(
        audit.qualified_publications,
        prior_publications + 1,
        "{audit:#?}"
    );
    assert!(audit.publication_error.is_none(), "{audit:#?}");
    assert!(f.known(A));
}

#[tokio::test]
async fn automatic_owner_blocks_runtime_renewal_preserves_age_and_invalidates_prior_epoch() {
    assert_eq!(
        SloAutomaticCalibrationSettingsV1::default().population_schedule,
        ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksV1
    );
    let mut f = Families::new();
    let events = Installations::default();
    events.during(|| complete_epoch(&mut f, 1, 0));
    let query = f.query(A);
    let first = only_child(&f);
    let original = serde_json::to_value(first.provenance()).unwrap();
    let first_receipt = f.runtime.training.published_catalog_receipt().unwrap();
    let original_child = serde_json::to_value(
        &first_receipt
            .structured_whole_wave_v2
            .as_ref()
            .unwrap()
            .children[0],
    )
    .unwrap();
    let first_event = &events.records()[0];
    assert_eq!(first_event["changed_children"][0]["kind"], "added");
    assert_eq!(first_event["changed_children"][0]["after"], original_child);
    assert!(first_event["previous_runtime_epoch"].is_null());
    let first_snapshot = f.runtime.snapshot().unwrap();
    let first_epoch = first_snapshot.prospective_identity().unwrap();
    let first_prediction = first_snapshot
        .audit_structured_query_v2(&query, f.clock.now_ns().unwrap())
        .unwrap();
    let first_expiry = expires_at(&first, &query, f.clock.now_ns().unwrap());
    assert!(first_epoch.current());
    assert_eq!(first_prediction.model_version, first_epoch.model_version());

    events.during(|| complete_epoch(&mut f, 2, 1));
    let installed = f.runtime.training.published_catalog_receipt().unwrap();
    let records = events.records();
    assert_eq!(records.len(), 2);
    let change = &records[1]["changed_children"][0];
    assert_eq!(change["kind"], "replaced");
    assert_eq!(change["same_domain"], true);
    assert_eq!(change["independent_source"], true);
    assert_eq!(change["previous_was_current_at_validation"], true);
    assert_eq!(change["before"], original_child);
    assert_eq!(
        change["after"],
        serde_json::to_value(
            &installed
                .structured_whole_wave_v2
                .as_ref()
                .unwrap()
                .children[0],
        )
        .unwrap()
    );
    assert_eq!(
        records[1]["installed_runtime_epoch"],
        installed.model_version
    );
    assert_eq!(
        records[1]["previous_runtime_epoch"],
        first_prediction.model_version
    );
    let second = only_child(&f);
    let second_snapshot = f.runtime.snapshot().unwrap();
    let now = f.clock.now_ns().unwrap();
    let second_prediction = second_snapshot
        .audit_structured_query_v2(&query, now)
        .unwrap();
    assert_eq!(first.owner(), second.owner());
    assert_eq!(first.workload_domain(), second.workload_domain());
    assert_eq!(second.provenance().schema_version, 14);
    assert_ne!(
        first.provenance().capture_identity,
        second.provenance().capture_identity
    );
    assert_ne!(
        first.provenance().source_sha256,
        second.provenance().source_sha256
    );
    assert!(
        second.provenance().phases[0].accepted_fifo_cutoff
            > first.provenance().phases[2].accepted_fifo_cutoff,
        "the replacement Fit must use a fresh original FIFO population"
    );
    for phase in &second.provenance().phases {
        assert_eq!(phase.members, OFFERS, "the original min8 is not reduced");
    }
    assert!(second_prediction.model_version > first_prediction.model_version);
    assert!(second_snapshot.current());
    assert!(
        !first_epoch.current(),
        "retained candidate identity must become stale"
    );
    assert!(!first_snapshot.current());
    assert_eq!(
        first_snapshot
            .audit_structured_query_v2(&query, now)
            .unwrap_err(),
        StructuredUnknownV2::RuntimeValidity,
        "the previous model cannot serve another candidate after publication"
    );
    assert_eq!(serde_json::to_value(first.provenance()).unwrap(), original);
    assert_eq!(expires_at(&first, &query, now), first_expiry);
    let second_expiry = expires_at(&second, &query, now);
    assert!(
        second_expiry > first_expiry,
        "only independently new data gains a new lifetime"
    );
    assert_eq!(f.recorded, (8 * OFFERS) as u64);

    // Distinguish original source age from the runtime epoch invalidation:
    // the retained child has no epoch gate, but must still expire at its old TTL.
    f.clock.set(first_expiry + 1);
    assert!(first.is_current_local(f.clock.now_ns().unwrap()).is_err());
    assert!(second.is_current_local(f.clock.now_ns().unwrap()).is_ok());
    assert!(f.known(A));
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_owner_blocks_runtime_renewal_failed_generation_preserves_good_model() {
    let mut f = Families::new();
    let events = Installations::default();
    events.during(|| complete_epoch(&mut f, 1, 0));
    let good = f.runtime.snapshot().unwrap();
    let good_epoch = good.prospective_identity().unwrap();
    let good_child = only_child(&f);
    let query = f.query(A);
    let original = serde_json::to_value(good_child.provenance()).unwrap();
    let receipt =
        serde_json::to_value(f.runtime.training.published_catalog_receipt().unwrap()).unwrap();
    let expiry = expires_at(&good_child, &query, f.clock.now_ns().unwrap());

    // Generation 2 has genuine, independent Discovery/Fit/Residual and seven
    // genuine Qualification members. Losing the last original ticket is a
    // failed population, never a smaller complete population or a retry slot.
    f.runtime.consume_samples();
    assert_eq!(f.live().audit().population.generation, 2);
    for _ in 0..3 {
        events.during(|| f.block(A));
        assert_eq!(f.live().audit().qualified_publications, 1);
    }
    f.runtime.consume_samples();
    for _ in 0..OFFERS - 1 {
        events.during(|| f.record(A));
    }
    let at = f.clock.now_ns().unwrap() + 100;
    f.clock.set(at);
    drop(
        f.runtime
            .reserve_live_ticket(Some(at))
            .expect("last original qualification offer"),
    );
    events.during(|| f.runtime.consume_samples());
    let failed = f.live().audit();
    assert_eq!(
        events.records().len(),
        1,
        "failed population is not installed"
    );
    assert_eq!(failed.failed_generations, 1, "{failed:#?}");
    assert_eq!(failed.qualified_publications, 1, "{failed:#?}");
    assert_eq!(failed.population.generation, 2);
    assert!(failed.population.failed);
    assert_eq!(failed.population.issued, OFFERS);
    assert_eq!(failed.population.retired, OFFERS);
    assert!(Arc::ptr_eq(&good, &f.runtime.snapshot().unwrap()));
    assert!(good_epoch.current());
    assert!(f.known(A));
    assert_eq!(
        serde_json::to_value(f.runtime.training.published_catalog_receipt().unwrap()).unwrap(),
        receipt
    );
    assert_eq!(
        serde_json::to_value(only_child(&f).provenance()).unwrap(),
        original
    );
    assert_eq!(expires_at(&good_child, &query, at), expiry);

    // A genuinely qualified child plus a mismatched public receipt is also
    // rejected by the real installer, with no success event or gate change.
    let mut invalid = f.runtime.training.published_catalog_receipt().unwrap();
    invalid.structured_whole_wave_v2.as_mut().unwrap().children[0].parameters_sha256[0] ^= 1;
    let children = f.runtime.training.live_catalog_children(at).unwrap();
    assert!(events
        .during(|| f
            .runtime
            .training
            .publish_live_catalog(children, invalid, at))
        .is_err());
    assert_eq!(events.records().len(), 1);
    assert!(good_epoch.current());
    assert!(Arc::ptr_eq(&good, &f.runtime.snapshot().unwrap()));

    // A failed generation does not stop automatic collection. Its successor
    // must collect all three fresh phases before replacing the good epoch.
    events.during(|| complete_epoch(&mut f, 3, 1));
    assert_eq!(events.records().len(), 2);
    assert_eq!(f.live().audit().failed_generations, 1);
    assert!(!good_epoch.current());
    assert!(f.runtime.snapshot().unwrap().current());
    let replacement = only_child(&f);
    assert_eq!(replacement.owner(), good_child.owner());
    assert_ne!(
        replacement.provenance().capture_identity,
        good_child.provenance().capture_identity
    );
    assert_eq!(
        serde_json::to_value(good_child.provenance()).unwrap(),
        original
    );
    assert_eq!(f.recorded, (12 * OFFERS - 1) as u64);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_owner_blocks_runtime_renewal_diagnostic_distinguishes_added_family() {
    let mut f = Families::new();
    let events = Installations::default();
    for algorithm in [A, A, B, B, A, A] {
        events.during(|| f.block(algorithm));
    }
    let first = f.runtime.training.published_catalog_receipt().unwrap();
    let original = &first.structured_whole_wave_v2.as_ref().unwrap().children[0];
    assert_eq!(events.records().len(), 1);
    for _ in 0..2 {
        events.during(|| f.block(B));
    }
    assert!(f.known(A));
    assert!(f.known(B));
    let records = events.records();
    assert_eq!(records.len(), 2);
    assert_eq!(records[1]["catalog_child_count"], 2);
    let changes = records[1]["changed_children"].as_array().unwrap();
    assert_eq!(changes.len(), 1, "retained A is not a replacement");
    assert_eq!(changes[0]["kind"], "added");
    assert_ne!(
        changes[0]["after"]["owner_sha256"],
        serde_json::to_value(original.owner_sha256).unwrap()
    );
    assert_eq!(
        records[1]["retained_domain_signatures"],
        serde_json::to_value([original.domain_signature]).unwrap()
    );
    let newest = f.runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(
        changes[0]["after"],
        serde_json::to_value(&newest.structured_whole_wave_v2.as_ref().unwrap().children[0])
            .unwrap()
    );
    let children = f
        .runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap();
    let retained = children
        .iter()
        .find(|child| child.domain_signature() == &original.domain_signature)
        .unwrap();
    assert_eq!(
        retained.provenance().capture_identity,
        original.capture_identity_sha256
    );
    assert_eq!(retained.provenance().source_sha256, original.source_sha256);
    f.runtime.shutdown().await.unwrap();
}
