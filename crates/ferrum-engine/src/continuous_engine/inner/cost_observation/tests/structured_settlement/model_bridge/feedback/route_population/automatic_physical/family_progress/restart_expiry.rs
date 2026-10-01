//! Original CPU submission -> mixed-age source7 -> worker prune -> cache save
//! -> independent runtime load. No synthetic qualified model is installed.
use super::*;
use ferrum_interfaces::execution_cost::{CostMonotonicDomainV1, CostObservationClock};
use ferrum_types::{SloAutomaticCalibrationCacheLocationV1, SloAutomaticCalibrationReuseV1};

struct BootClock {
    local: Arc<VirtualClock>,
    domain: CostMonotonicDomainV1,
}
impl CostObservationClock for BootClock {
    fn now_ns(&self) -> Option<u64> {
        self.local.now_ns()
    }
    fn monotonic_domain(&self) -> Option<&CostMonotonicDomainV1> {
        Some(&self.domain)
    }
}
struct CacheDirectory(std::path::PathBuf);
impl CacheDirectory {
    fn clean_manifest(&self) -> serde_json::Value {
        use sha2::{Digest, Sha256};
        let root = self.0.join("cache-v1");
        assert!(
            !root.join("dirty").exists(),
            "clean acknowledgement must remove actual dirty marker"
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let generation: [u8; 16] = serde_json::from_value(manifest["generation"].clone()).unwrap();
        assert_ne!(generation, [0; 16]);
        let name: String = generation
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let payload = root.join(format!("g-{name}"));
        assert!(
            payload.is_dir(),
            "manifest must name an actual committed generation"
        );
        assert_eq!(manifest["sources"].as_array().unwrap().len(), 1);
        assert_eq!(
            manifest["sources"][0]["domains"].as_array().unwrap().len(),
            1
        );
        for receipt in [
            &manifest["sources"][0]["journal"],
            &manifest["sources"][0]["profile"],
            &manifest["feedback"],
        ] {
            let path = payload.join(format!(
                "artifact-{}.bin",
                receipt["index"].as_u64().unwrap()
            ));
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(bytes.len() as u64, receipt["bytes"].as_u64().unwrap());
            let digest: [u8; 32] = serde_json::from_value(receipt["sha256"].clone()).unwrap();
            assert_eq!(<[u8; 32]>::from(Sha256::digest(&bytes)), digest);
        }
        // The persisted profile still describes the complete mixed source;
        // its manifest selects only the independently fresh surviving child.
        let profile = &manifest["sources"][0]["profile"];
        let profile: serde_json::Value = serde_json::from_slice(
            &std::fs::read(payload.join(format!(
                "artifact-{}.bin",
                profile["index"].as_u64().unwrap()
            )))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(profile["children"].as_array().unwrap().len(), 2);
        manifest
    }
}
impl Drop for CacheDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn automatic_mixed_child_expiry_saves_and_restarts_only_original_fresh_models() {
    let directory = CacheDirectory(std::env::temp_dir().join(format!(
        "ferrum-mixed-expiry-cache-{}",
        uuid::Uuid::new_v4()
    )));
    let identity = identity();
    let ExecutorCostIdentityAvailability::Known(executor) = &identity else {
        unreachable!()
    };
    let domain = CostWorkloadDomainV1::new_vnext(
        executor,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
            repetition_slot_capacity: 0,
            fixed_state_bytes_per_row: 64,
        },
    )
    .unwrap();
    let local = Arc::new(VirtualClock(AtomicU64::new(1)));
    let clock = Arc::new(BootClock {
        local: local.clone(),
        domain: CostMonotonicDomainV1::new_macos_continuous([73; 16]).unwrap(),
    });
    let mut settings = SloAutomaticCalibrationSettingsV1 {
        discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
        phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
        ..Default::default()
    };
    let SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 { location, .. } =
        &mut settings.reuse
    else {
        panic!("default automatic reuse policy")
    };
    *location = SloAutomaticCalibrationCacheLocationV1::Directory {
        path: directory.0.clone(),
    };
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 { settings };
    let runtime = EngineCostRuntime::build_with_manual_worker_and_reuse(
        identity.clone(),
        clock.clone(),
        &config,
        domain.clone(),
    )
    .unwrap();
    assert!(runtime.audit_snapshot().automatic_reuse.is_some());
    runtime.begin_automatic_calibration().unwrap();
    runtime.consume_samples();
    let mut f = Families {
        runtime,
        clock: local.clone(),
        domain: domain.clone(),
        recorded: 0,
    };
    for algorithm in [A, A, B, B, A, A, B, B] {
        f.block(algorithm);
    }
    assert!(f.known(A));
    assert!(f.known(B));
    let original = f.runtime.snapshot().unwrap();
    let before = local.now_ns().unwrap();
    let a = original
        .audit_structured_query_v2(&f.query(A), before)
        .unwrap();
    let b = original
        .audit_structured_query_v2(&f.query(B), before)
        .unwrap();
    assert!(b.valid_for_ns > a.valid_for_ns);
    let originals = original.live_children(before).unwrap();
    assert_eq!(originals.len(), 2);
    assert_eq!(
        originals[0].provenance().capture_identity,
        originals[1].provenance().capture_identity,
        "same original source has mixed child ages"
    );
    // Resolve the child through the same catalog route as prediction. The
    // original query and installed model can use different template policies.
    let selected_b = original.prospective_source(&f.query(B)).unwrap().domain;
    let original_b = originals
        .iter()
        .find(|child| *child.domain_signature() == selected_b)
        .unwrap();
    let expected_parameters = original_b.parameters_signature();
    let expected_clock = original_b.provenance().clock.clone();
    let expected_b_query = f.query(B);
    local.set(before + a.valid_for_ns + 1);
    // The same worker publication transaction runs without another sample.
    assert!(f.runtime.training.prune_expired_live_catalog().unwrap());
    let fresh = f.runtime.snapshot().unwrap();
    fresh
        .validate_live_freshness(local.now_ns().unwrap())
        .unwrap();
    assert_eq!(
        fresh.live_children(local.now_ns().unwrap()).unwrap().len(),
        1
    );
    assert_eq!(
        fresh
            .audit_structured_query_v2(&expected_b_query, local.now_ns().unwrap())
            .unwrap()
            .planning_ns,
        b.planning_ns
    );
    let feedback_before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    f.runtime.shutdown().await.unwrap();
    f.runtime.acknowledge_restart_shutdown();
    let audit = f.runtime.audit_snapshot().automatic_reuse.unwrap();
    assert!(audit.clean_shutdown, "{audit:?}");
    assert!(audit.first_failure.is_none(), "{audit:?}");
    let first_clean = directory.clean_manifest();
    drop(f);
    drop(fresh);
    drop(original);
    let warm =
        EngineCostRuntime::build_with_manual_worker_and_reuse(identity, clock, &config, domain)
            .unwrap();
    let restored = warm
        .snapshot()
        .expect("valid remaining model is automatically restored");
    let children = restored.live_children(local.now_ns().unwrap()).unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].parameters_signature(), expected_parameters);
    assert_eq!(children[0].provenance().clock, expected_clock);
    let prediction = restored
        .audit_structured_query_v2(&expected_b_query, local.now_ns().unwrap())
        .unwrap();
    assert_eq!(prediction.planning_ns, b.planning_ns);
    assert_eq!(
        local.now_ns().unwrap() + prediction.valid_for_ns,
        before + b.valid_for_ns
    );
    let feedback_after = warm.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(feedback_after.compared, feedback_before.compared);
    assert_eq!(feedback_after.corrections, feedback_before.corrections);
    assert!(feedback_after.revoked.is_none());
    // A second clean save exercises warm source archival and feedback epoch
    // resume without changing original numerical parameters or clocks.
    warm.shutdown().await.unwrap();
    warm.acknowledge_restart_shutdown();
    let audit = warm.audit_snapshot().automatic_reuse.unwrap();
    assert!(audit.clean_shutdown, "{audit:?}");
    assert!(audit.first_failure.is_none(), "{audit:?}");
    let second_clean = directory.clean_manifest();
    assert_ne!(
        first_clean["generation"], second_clean["generation"],
        "warm shutdown must publish a new clean generation, not leave the old manifest"
    );
}

#[path = "restart_expiry/rolling_restore.rs"]
mod rolling_restore;

/// Exercise the ordinary worker order, rather than invoking the expiry helper
/// before consumption. Both models and the queued wave are genuine CPU receipts.
#[tokio::test]
async fn automatic_feedback_consumption_expires_selected_child_without_revoking_fresh_peer() {
    consume_expiring_original_child(false).await;
}

#[tokio::test]
async fn automatic_feedback_consumption_handles_expiry_during_original_fifo_resolution() {
    consume_expiring_original_child(true).await;
}

async fn consume_expiring_original_child(expire_during_pop: bool) {
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2;
    let mut f = Families::new();
    for algorithm in [A, A, B, B, A, A, B, B] {
        f.block(algorithm);
    }
    let original = f.runtime.snapshot().unwrap();
    let before = f.clock.now_ns().unwrap();
    let query_a = f.query(A);
    let query_b = f.query(B);
    let a = original
        .audit_structured_query_v2(&query_a, before)
        .unwrap();
    let b = original
        .audit_structured_query_v2(&query_b, before)
        .unwrap();
    assert!(b.valid_for_ns > a.valid_for_ns);
    let b_domain = original.prospective_source(&query_b).unwrap().domain;
    let original_b = original
        .live_children(before)
        .unwrap()
        .into_iter()
        .find(|child| *child.domain_signature() == b_domain)
        .unwrap();
    let feedback_before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    let accepted_before = f.runtime.audit_snapshot().sink.raw_accepted;

    // The old publication remains installed when a real ordinary wave arrives
    // just after A's oldest-sample TTL. No manual prune, new sample, or model
    // replacement is inserted ahead of the original FIFO worker.
    let expires_at = before + a.valid_for_ns;
    f.clock.set(if expire_during_pop {
        expires_at - 1_000
    } else {
        expires_at + 1
    });
    let stages = fixture::record_unticketed_cohort_with_hook(
        &f.runtime,
        &f.clock,
        Families::wave(A),
        |_| {},
    )
    .expect("original CPU submission and complete host settlement");
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    assert!(stages
        .structured_evidence
        .as_ref()
        .is_some_and(Result::is_ok));
    let consumed_at = if expire_during_pop {
        let before_pop = f.clock.now_ns().unwrap();
        assert!(original
            .audit_structured_query_v2(&query_a, before_pop)
            .is_ok());
        let clock = f.clock.clone();
        // Existing real FIFO hook: the pre-batch snapshot was obtained while
        // A was fresh. Only dequeue/resolution crosses the original TTL.
        f.runtime
            .sink
            .on_next_pop(move || clock.set(expires_at + 1));
        expires_at + 1
    } else {
        f.clock.now_ns().unwrap()
    };
    assert!(
        consumed_at < before + b.valid_for_ns,
        "the peer must still be genuinely fresh"
    );
    assert!(matches!(
        original.audit_structured_query_v2(&query_a, consumed_at),
        Err(StructuredUnknownV2::Stale)
    ));
    assert!(original
        .audit_structured_query_v2(&query_b, consumed_at)
        .is_ok());
    let queued = f.runtime.audit_snapshot().sink;
    assert_eq!(queued.raw_accepted, accepted_before + 1);
    assert_eq!(queued.raw_pending, 1);
    // This is the real consumer entry. In the defective order it converts
    // Lookup(Stale) into Uncomparable and revokes B before expiry can run.
    f.runtime.consume_samples();
    let audit = f.runtime.audit_snapshot();
    let feedback_after = audit.structured_feedback.unwrap();
    assert!(
        feedback_after.revoked.is_none(),
        "time-driven expiry must not become model-error revocation: {feedback_after:?}"
    );
    assert_eq!(
        feedback_after.uncomparable_observations,
        feedback_before.uncomparable_observations
    );
    assert_eq!(feedback_after.compared, feedback_before.compared);
    assert_eq!(audit.sink.raw_pending, 0);
    assert_eq!(audit.sink.raw_resolved, queued.raw_resolved + 1);
    let next = f
        .runtime
        .snapshot()
        .expect("fresh peer survives the expiry transaction");
    assert!(!original.current());
    assert!(next.model_version() > original.model_version());
    let at = f.clock.now_ns().unwrap();
    let children = next.live_children(at).unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(
        children[0].domain_signature(),
        original_b.domain_signature()
    );
    assert_eq!(
        children[0].parameters_signature(),
        original_b.parameters_signature()
    );
    assert_eq!(
        children[0].provenance().source_sha256,
        original_b.provenance().source_sha256
    );
    assert_eq!(
        children[0].provenance().clock,
        original_b.provenance().clock
    );
    let after_b = next.audit_structured_query_v2(&query_b, at).unwrap();
    assert_eq!(after_b.planning_ns, b.planning_ns);
    assert_eq!(at + after_b.valid_for_ns, before + b.valid_for_ns);
    let expiry = audit
        .training
        .catalog_expiry
        .expect("typed original-age expiry");
    assert_eq!(expiry.previous_runtime_epoch, original.model_version());
    assert_eq!(expiry.current_runtime_epoch, next.model_version());
    assert_eq!(expiry.removed_expired_children, 1);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_feedback_expiry_does_not_hide_original_consumption_lag() {
    use crate::continuous_engine::inner::cost_observation::selected_feedback::Revocation;
    let mut f = Families::new();
    for algorithm in [A, A, B, B, A, A, B, B] {
        f.block(algorithm);
    }
    let original = f.runtime.snapshot().unwrap();
    let before = f.clock.now_ns().unwrap();
    let a = original
        .audit_structured_query_v2(&f.query(A), before)
        .unwrap();
    let b = original
        .audit_structured_query_v2(&f.query(B), before)
        .unwrap();
    assert!(b.valid_for_ns > a.valid_for_ns);
    fixture::record_unticketed_cohort_with_hook(&f.runtime, &f.clock, Families::wave(A), |_| {})
        .unwrap();
    // This actual receipt was queued while the model was fresh. Its original
    // consumption lag still violates the configured policy when A expires.
    f.clock.set(before + a.valid_for_ns + 1);
    assert!(original
        .audit_structured_query_v2(&f.query(B), f.clock.now_ns().unwrap())
        .is_ok());
    f.runtime.consume_samples();
    let audit = f.runtime.audit_snapshot();
    assert!(matches!(
        audit.structured_feedback.unwrap().revoked,
        Some(Revocation::ObservationLag)
    ));
    assert!(f.runtime.snapshot().is_none());
    assert!(!original.current());
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_feedback_expiry_does_not_hide_damaged_original_settlement() {
    use crate::continuous_engine::inner::cost_observation::selected_feedback::Revocation;
    let mut f = Families::new();
    for algorithm in [A, A, B, B, A, A, B, B] {
        f.block(algorithm);
    }
    let original = f.runtime.snapshot().unwrap();
    let before = f.clock.now_ns().unwrap();
    let a = original
        .audit_structured_query_v2(&f.query(A), before)
        .unwrap();
    f.clock.set(before + a.valid_for_ns + 1);
    let recorder = EngineCostRuntime::build_with_profile_and_domain(
        identity(),
        f.clock.clone(),
        &SloCostObservationConfig::structured_whole_wave_v2(),
        false,
        None,
        None,
        Some(f.domain.clone()),
    )
    .unwrap();
    let stages =
        fixture::record_unticketed_cohort_with_hook(&recorder, &f.clock, Families::wave(A), |_| {})
            .unwrap();
    assert!(original
        .audit_structured_query_v2(&f.query(B), f.clock.now_ns().unwrap())
        .is_ok());
    // A genuine submitted/settled CPU wave with its private settlement removed
    // is not allowed to claim the harmless time-driven lifecycle outcome.
    let mut damaged = stages.as_ref().clone();
    damaged.structured_evidence = None;
    f.runtime
        .sink
        .offer_evidence_bound(
            entry(Arc::new(damaged)),
            None,
            f.runtime.sink.source_generation(),
        )
        .unwrap();
    f.runtime.consume_samples();
    let audit = f.runtime.audit_snapshot();
    assert!(matches!(
        audit.structured_feedback.unwrap().revoked,
        Some(Revocation::UncomparableObservation)
    ));
    assert!(f.runtime.snapshot().is_none());
    assert!(!original.current());
    drop(recorder);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_feedback_consumption_all_expired_stays_unknown() {
    let mut f = Families::new();
    for algorithm in [A, A, B, B, A, A, B, B] {
        f.block(algorithm);
    }
    let original = f.runtime.snapshot().unwrap();
    let before = f.clock.now_ns().unwrap();
    let b = original
        .audit_structured_query_v2(&f.query(B), before)
        .unwrap();
    f.clock.set(before + b.valid_for_ns + 1);
    fixture::record_unticketed_cohort_with_hook(&f.runtime, &f.clock, Families::wave(B), |_| {})
        .unwrap();
    f.runtime.consume_samples();
    let audit = f.runtime.audit_snapshot();
    assert!(audit.structured_feedback.unwrap().revoked.is_none());
    assert!(f.runtime.snapshot().is_none());
    assert!(!original.current());
    assert_eq!(f.runtime.sink.source_generation(), 0);
    let expiry = audit.training.catalog_expiry.unwrap();
    assert_eq!(expiry.current_children, 0);
    assert_eq!(expiry.current_runtime_epoch, 0);
    assert_eq!(expiry.removed_expired_children, 2);
    let publications = f.live().audit().qualified_publications;
    f.runtime.consume_samples();
    assert!(
        f.runtime.snapshot().is_none(),
        "empty consumption cannot renew old data"
    );
    // The extra original feedback call above still owns one accepted ordinal.
    f.recorded += 1;
    // Only a new Discovery/Fit/Residual/Qualification sequence, with the
    // original min8 per block, may reopen prediction authority.
    for phase in 0..4 {
        f.block(B);
        if phase < 3 {
            assert!(f.runtime.snapshot().is_none());
            assert_eq!(f.live().audit().qualified_publications, publications);
        }
    }
    let renewed = f
        .runtime
        .snapshot()
        .expect("fresh independent qualification");
    assert!(renewed.model_version() > original.model_version());
    assert!(renewed
        .audit_structured_query_v2(&f.query(B), f.clock.now_ns().unwrap())
        .is_ok());
    assert!(!original.current());
    assert_eq!(f.live().audit().qualified_publications, publications + 1);
    f.runtime.shutdown().await.unwrap();
}
