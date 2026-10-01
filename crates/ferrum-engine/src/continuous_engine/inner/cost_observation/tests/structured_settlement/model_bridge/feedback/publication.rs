//! Original recorder -> source3 replay/load -> shared runtime publication.
//! No handcrafted qualified model or zero-cost prediction is installed.
use super::*;

#[tokio::test]
async fn expiry_subset_publishes_new_epoch_preserving_fresh_feedback_and_original_receipt() {
    let f = Fixture::new();
    let runtime = f.build();
    // Calibrate the younger owner's existing feedback state before pruning.
    for _ in 0..2 {
        let w = wave("fixture.feedback.b");
        record(
            &runtime.ids,
            &runtime.sink,
            &f.clock,
            w.actual,
            w.host,
            None,
            30,
        );
        runtime.consume_samples();
    }
    let old = runtime.snapshot().unwrap();
    let now = f.clock.now_ns().unwrap();
    let before_a = old.audit_structured_query_v2(&f.queries[0], now).unwrap();
    let before_b = old.audit_structured_query_v2(&f.queries[1], now).unwrap();
    assert!(before_b.valid_for_ns > before_a.valid_for_ns);
    let original_children = old.live_children(now).unwrap();
    let original_receipt = serde_json::to_value(runtime.profile_receipt()).unwrap();
    let feedback = runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(feedback.compared, 2);
    assert!(feedback.corrections > 0);
    f.clock.set(now + before_a.valid_for_ns + 1);
    assert!(old
        .validate_live_freshness(f.clock.now_ns().unwrap())
        .is_err());
    assert!(runtime.training.prune_expired_live_catalog().unwrap());
    let next = runtime.snapshot().unwrap();
    assert!(next.model_version() > old.model_version());
    assert!(!old.current());
    let at = f.clock.now_ns().unwrap();
    next.validate_live_freshness(at).unwrap();
    let retained = next.live_children(at).unwrap();
    assert!(!retained.is_empty());
    assert!(retained.len() < original_children.len());
    for child in retained {
        let original = original_children
            .iter()
            .find(|old| old.domain_signature() == child.domain_signature())
            .unwrap();
        assert_eq!(
            child.parameters_signature(),
            original.parameters_signature()
        );
        assert_eq!(child.provenance().clock, original.provenance().clock);
        assert_eq!(
            child.provenance().oldest_imported_age_ns,
            original.provenance().oldest_imported_age_ns
        );
    }
    let b = next.audit_structured_query_v2(&f.queries[1], at).unwrap();
    assert_eq!(b.planning_ns, before_b.planning_ns);
    assert_eq!(at + b.valid_for_ns, now + before_b.valid_for_ns);
    assert_eq!(
        serde_json::to_value(runtime.profile_receipt()).unwrap(),
        original_receipt
    );
    let after = runtime.audit_snapshot();
    let after_feedback = after.structured_feedback.unwrap();
    assert_eq!(after_feedback.compared, feedback.compared);
    assert_eq!(after_feedback.corrections, feedback.corrections);
    assert!(after_feedback.revoked.is_none());
    let transition = after.training.catalog_expiry.unwrap();
    assert_eq!(transition.previous_runtime_epoch, old.model_version());
    assert_eq!(transition.current_runtime_epoch, next.model_version());
    assert_eq!(transition.removed_expired_children, 1);
    assert!(!runtime.training.prune_expired_live_catalog().unwrap());
    f.clock.set(now + before_b.valid_for_ns + 1);
    assert!(runtime.training.prune_expired_live_catalog().unwrap());
    assert!(
        runtime.snapshot().is_none(),
        "all-expired is unavailable, never an empty valid model"
    );
    assert!(!next.current());
    assert_eq!(
        runtime
            .audit_snapshot()
            .training
            .catalog_expiry
            .unwrap()
            .current_runtime_epoch,
        0
    );
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_feedback_starts_at_first_qualified_base_and_excludes_earlier_samples() {
    let mut f = Fixture::new();
    let SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 { policy, .. } =
        &f.config.structured_feedback
    else {
        unreachable!()
    };
    let policy = policy.clone();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let imported = f.build();
    let at = f.clock.now_ns().unwrap();
    let (children, receipt) = replacement(&imported, at);
    let mut settings = ferrum_types::SloAutomaticCalibrationSettingsV1::default();
    settings.feedback = policy;
    f.config.live_structured_calibration =
        ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings };
    let runtime = EngineCostRuntime::build(identity(), f.clock.clone(), &f.config, false).unwrap();
    assert!(runtime.snapshot().is_none());
    assert!(runtime.audit_snapshot().structured_feedback.is_none());
    let w = wave("fixture.feedback.a");
    record(
        &runtime.ids,
        &runtime.sink,
        &f.clock,
        w.actual,
        w.host,
        None,
        90,
    );
    let now = f.clock.now_ns().unwrap();
    let epoch = runtime
        .training
        .publish_live_catalog(children, receipt, now)
        .unwrap();
    let base = runtime.snapshot().unwrap();
    let before = base.audit_structured_query_v2(&f.queries[0], now).unwrap();
    assert_eq!(before.model_version, epoch);
    runtime.consume_samples();
    let audit = runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(audit.epoch, epoch);
    assert_eq!(audit.compared, 0);
    assert_eq!(audit.uncomparable_observations, 0);
    f.observe(&runtime, 90);
    f.observe(&runtime, 90);
    let after = runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&f.queries[0], f.clock.now_ns().unwrap())
        .unwrap();
    assert!(after.planning_ns > before.planning_ns);
    assert!(after.model_version > epoch);
    assert!(!base.current());
    assert_eq!(
        runtime
            .audit_snapshot()
            .structured_feedback
            .unwrap()
            .compared,
        2
    );
    assert!(!f.directory.join("feedback.json").exists());
    f.unchanged();
    runtime.shutdown().await.unwrap();
    imported.shutdown().await.unwrap();
}

fn replacement(
    runtime: &EngineCostRuntime,
    now: u64,
) -> (
    Vec<file::ImportedStructuredModelV2>,
    ferrum_types::SloCostProfileReceipt,
) {
    (
        runtime.training.live_catalog_children(now).unwrap(),
        runtime.profile_receipt().unwrap().clone(),
    )
}

#[tokio::test]
async fn structured_epoch_without_feedback_revokes_retained_query_and_final_submission_version() {
    let mut f = Fixture::new();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let runtime = f.build();
    let old = runtime.snapshot().unwrap();
    let at = f.clock.now_ns().unwrap();
    let issued = old.audit_structured_query_v2(&f.queries[0], at).unwrap();
    assert_eq!(
        runtime.try_model_version_current(issued.model_version),
        Some(true)
    );
    let (children, receipt) = replacement(&runtime, at);
    let epoch = runtime
        .training
        .publish_live_catalog(children, receipt, at)
        .unwrap();
    assert!(epoch > issued.model_version);
    assert!(!old.current());
    assert!(matches!(
        old.audit_structured_query_v2(&f.queries[0], at),
        Err(StructuredUnknownV2::RuntimeValidity)
    ));
    // Both already-issued-plan publication and final HostGuard call this exact
    // shared runtime predicate, including its concurrent-publication Busy case.
    assert_eq!(
        runtime.try_model_version_current(issued.model_version),
        Some(false)
    );
    assert_eq!(runtime.try_model_version_current(epoch), Some(true));
    let next = runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&f.queries[0], at)
        .unwrap();
    assert_eq!(next.planning_ns, issued.planning_ns);
    assert_eq!(next.valid_for_ns, issued.valid_for_ns);
    assert_eq!(
        runtime
            .training
            .published_catalog_receipt()
            .unwrap()
            .model_version,
        epoch
    );
    f.unchanged();
    runtime.shutdown().await.unwrap();
    assert!(!runtime.snapshot().is_some());
}

#[tokio::test]
async fn structured_epoch_feedback_and_replacement_share_order_and_exclude_queued_old_origin() {
    let f = Fixture::new();
    let runtime = f.build();
    f.observe(&runtime, 30);
    f.observe(&runtime, 30);
    let old = runtime.snapshot().unwrap();
    let at = f.clock.now_ns().unwrap();
    let corrected = old.audit_structured_query_v2(&f.queries[0], at).unwrap();
    let compared = runtime
        .audit_snapshot()
        .structured_feedback
        .unwrap()
        .compared;
    // This real physical call is prepared under the original base, then queued
    // before the new catalog is published. It must not correct the new base.
    let w = wave("fixture.feedback.a");
    record(
        &runtime.ids,
        &runtime.sink,
        &f.clock,
        w.actual,
        w.host,
        None,
        90,
    );
    let now = f.clock.now_ns().unwrap();
    let (children, receipt) = replacement(&runtime, now);
    let replaced = runtime
        .training
        .publish_live_catalog(children, receipt, now)
        .unwrap();
    assert!(replaced > old.model_version());
    runtime.consume_samples();
    let a = runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(a.compared, compared);
    assert_eq!(a.uncomparable_observations, 0);
    let next = runtime.snapshot().unwrap();
    assert_eq!(
        next.audit_structured_query_v2(&f.queries[0], now)
            .unwrap()
            .planning_ns,
        corrected.planning_ns
    );
    f.observe(&runtime, 90);
    f.observe(&runtime, 90);
    assert!(runtime.snapshot().unwrap().model_version() > replaced);
    assert!(!next.current());
    assert!(runtime
        .audit_snapshot()
        .structured_feedback
        .unwrap()
        .revoked
        .is_none());
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn structured_epoch_invalid_receipt_or_durable_rebind_failure_preserves_current_base() {
    let f = Fixture::new();
    let runtime = f.build();
    let old = runtime.snapshot().unwrap();
    let at = f.clock.now_ns().unwrap();
    let (children, mut receipt) = replacement(&runtime, at);
    receipt.structured_whole_wave_v2.as_mut().unwrap().children[0].source_sha256[0] ^= 1;
    assert!(runtime
        .training
        .publish_live_catalog(children, receipt, at)
        .is_err());
    assert!(Arc::ptr_eq(&runtime.snapshot().unwrap(), &old));
    assert!(old.audit_structured_query_v2(&f.queries[0], at).is_ok());
    // Existing unfinished pending receipt is a real Store::persist failure,
    // not a fabricated writer result. Neither gate nor in-memory base changes.
    let pending = f.directory.join("feedback.json.pending");
    fs::write(&pending, b"preserve failed publication evidence").unwrap();
    let (children, receipt) = replacement(&runtime, at);
    assert!(runtime
        .training
        .publish_live_catalog(children, receipt, at)
        .is_err());
    assert!(Arc::ptr_eq(&runtime.snapshot().unwrap(), &old));
    assert_eq!(
        runtime.try_model_version_current(old.model_version()),
        Some(true)
    );
    assert!(old.audit_structured_query_v2(&f.queries[0], at).is_ok());
    fs::remove_file(pending).unwrap();
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn structured_epoch_no_initial_model_bootstraps_only_from_real_imported_children() {
    let mut f = Fixture::new();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let imported = f.build();
    let at = f.clock.now_ns().unwrap();
    let (children, receipt) = replacement(&imported, at);
    let runtime = EngineCostRuntime::build(identity(), f.clock.clone(), &f.config, false).unwrap();
    assert!(runtime.snapshot().is_none());
    let epoch = runtime
        .training
        .publish_live_catalog(children, receipt, at)
        .unwrap();
    assert!(epoch > 0);
    let snapshot = runtime.snapshot().unwrap();
    let p = snapshot
        .audit_structured_query_v2(&f.queries[0], at)
        .unwrap();
    let latest_valid_for = f
        .queries
        .iter()
        .map(|query| {
            snapshot
                .audit_structured_query_v2(query, at)
                .unwrap()
                .valid_for_ns
        })
        .max()
        .unwrap();
    f.clock.set(at + p.valid_for_ns + 1);
    assert!(matches!(
        snapshot.audit_structured_query_v2(&f.queries[0], f.clock.now_ns().unwrap()),
        Err(StructuredUnknownV2::Stale)
    ));
    // Each imported child retains its own original age. Expiration of one
    // owner must remove that owner without expiring younger catalog entries.
    let retained = runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap();
    assert!(!retained
        .iter()
        .any(|child| child.owner() == f.queries[0].owner()));
    for query in &f.queries {
        if snapshot
            .audit_structured_query_v2(query, f.clock.now_ns().unwrap())
            .is_ok()
        {
            assert!(retained.iter().any(|child| child.owner() == query.owner()));
        }
    }
    f.clock.set(at + latest_valid_for + 1);
    assert!(runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap()
        .is_empty());
    f.unchanged();
    runtime.shutdown().await.unwrap();
    imported.shutdown().await.unwrap();
}
