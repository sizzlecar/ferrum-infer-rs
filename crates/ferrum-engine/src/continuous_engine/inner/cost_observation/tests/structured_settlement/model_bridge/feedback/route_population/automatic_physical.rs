//! Shared run/serve cost runtime, real private route selector/CPU submission,
//! original host settlement and automatic worker. Clocks are controlled protocol
//! evidence; this test makes no throughput or natural-EOS coverage claim.
use super::*;
use ferrum_interfaces::execution_cost::{CostWorkloadDomainV1, CostWorkloadLimitsV1};
use ferrum_types::SloCostProfileStorage;
use std::num::{NonZeroU32, NonZeroU64};

#[tokio::test]
async fn automatic_default_memory_physical_domain_publishes_and_predicts_without_import() {
    check_automatic_physical_publication(SloAutomaticCalibrationSettingsV1::default()).await;
}

#[tokio::test]
async fn automatic_partial_settings_keep_owner_blocks_and_publish_without_import() {
    let settings: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(serde_json::json!({
        "discovery_offered_waves": 8,
        "phase_offered_waves": [8, 8, 8]
    }))
    .unwrap();
    check_automatic_physical_publication(settings).await;
}

async fn check_automatic_physical_publication(settings: SloAutomaticCalibrationSettingsV1) {
    assert_eq!(
        settings.population_schedule,
        ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksV1
    );
    let identity = identity();
    let ExecutorCostIdentityAvailability::Known(executor) = &identity else {
        panic!("fixture identity");
    };
    // This one-row controlled CPU fixture has one scheduled decode token, a
    // 64-byte recurrent payload and no repetition slots in use.
    let domain = CostWorkloadDomainV1::new_vnext(
        executor,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(1).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(1).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
            repetition_slot_capacity: 0,
            fixed_state_bytes_per_row: 64,
        },
    )
    .unwrap();
    let clock = Arc::new(VirtualClock(AtomicU64::new(1)));
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    assert!(matches!(
        settings.diagnostics,
        SloAutomaticCalibrationDiagnosticsV1::MemoryOnly
    ));
    let offered = [
        settings.discovery_offered_waves.get(),
        settings.phase_offered_waves[0].get(),
        settings.phase_offered_waves[1].get(),
        settings.phase_offered_waves[2].get(),
    ];
    config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 { settings };
    let runtime = EngineCostRuntime::build_with_profile_and_domain(
        identity,
        clock.clone(),
        &config,
        false,
        None,
        None,
        Some(domain.clone()),
    )
    .unwrap();
    let live = runtime.training.live.as_ref().unwrap();
    assert_eq!(live.audit().population.generation, 0);
    assert!(runtime.snapshot().is_none());
    // Same post-bootstrap cut used by both product entrypoints. No manual
    // declaration, training API, diagnostic directory or imported model.
    runtime.begin_automatic_calibration().unwrap();
    runtime.consume_samples();
    assert!(offered.iter().all(|quota| *quota == offered[0]));
    for (window, quota) in offered.into_iter().enumerate() {
        // No background worker in this fixture. Advance the completed FIFO
        // barrier before reserving a new original block; add no observations.
        runtime.consume_samples();
        for _ in 0..quota {
            fixture::record_route(
                &runtime,
                &clock,
                wave("fixture.automatic-physical"),
                false,
                false,
            )
            .unwrap();
            runtime.consume_samples();
        }
        let audit = live.audit();
        assert_eq!(audit.failed_generations, 0, "{audit:#?}");
        assert!(audit.publication_error.is_none(), "{audit:#?}");
        assert_eq!(audit.qualified_publications, u64::from(window == 3));
        let blocks = audit
            .automatic
            .as_ref()
            .unwrap()
            .owner_blocks
            .as_ref()
            .unwrap();
        assert_eq!(blocks.block, window as u64 + 1);
        assert_eq!(
            blocks.offered,
            offered[..=window].iter().sum::<usize>() as u64
        );
        assert_eq!(blocks.block_offered, quota);
        assert_eq!(blocks.owners.len(), 1);
        let expected_phase = [
            Some(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Fit),
            Some(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Residual),
            Some(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Qualification),
            None,
        ][window];
        assert_eq!(blocks.owners[0].phase, expected_phase);
        assert_eq!(blocks.owners[0].qualified, window == 3);
    }
    let audit = live.audit();
    assert_eq!(audit.qualified_publications, 1, "{audit:#?}");
    assert!(audit
        .automatic
        .as_ref()
        .unwrap()
        .last_diagnostic_publication
        .is_none());
    let receipt = runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.storage, SloCostProfileStorage::Memory);
    assert!(receipt.path.is_none());
    // Source7 certifies the complete prefix, including discovery. Numerical
    // provenance below separately certifies the three fresh training phases.
    assert_eq!(receipt.offered_samples, offered.iter().sum::<usize>());
    assert_eq!(receipt.recorded_samples, offered[1..].iter().sum::<usize>());
    let now = clock.now_ns().unwrap();
    let children = runtime.training.live_catalog_children(now).unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(
        children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|phase| phase.members),
        [offered[1], offered[2], offered[3]]
    );
    assert_eq!(children[0].owner().cost_template_policy(),
        ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1);
    assert_eq!(
        children[0].workload_domain(),
        Some(&domain),
        "automatic selected the explicit physical-envelope contract"
    );
    // The next request's longer, still legal context was not in Fit. Use
    // the same pre-execution projection constructor as the planner; no actual
    // completion or newly measured wall is supplied to this query.
    let query_wave = wave_with_kv("fixture.automatic-physical", 8);
    let query = StructuredQueryV2::from_future_with_domain(
        &query_wave.prepared.exact,
        &query_wave.prepared.selected,
        &query_wave.prepared.recipe,
        &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
        &domain,
    )
    .unwrap();
    let snapshot = runtime.snapshot().unwrap();
    snapshot.validate_workload_domain(Some(&domain)).unwrap();
    assert!(snapshot.validate_workload_domain(None).is_err());
    let unscoped = StructuredQueryV2::from_future(
        &query_wave.prepared.exact,
        &query_wave.prepared.selected,
        &query_wave.prepared.recipe,
        &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
    )
    .unwrap();
    assert!(snapshot.audit_structured_query_v2(&unscoped, now).is_err());
    assert!(
        !snapshot.undeclared_structured_owner(&unscoped),
        "missing physical identity is invalid evidence, never an outside-owner exclusion"
    );
    let prediction = snapshot.audit_structured_query_v2(&query, now).unwrap();
    assert!(prediction.planning_ns > 0);
    assert!(snapshot.current());
    // Future projection preserves the legacy identity and supplies the checked
    // new interpretation without copying its basis. The selected source and
    // all retrospective feedback must use the qualified model's actual key.
    assert_ne!(query.domain_signature(), children[0].domain_signature());
    assert_eq!(
        snapshot.prospective_source(&query).unwrap().domain,
        *children[0].domain_signature()
    );
    // The completed prefix remains the installed predictor while the worker
    // opens a fresh block. This turn cannot create or replace an offered row.
    let raw_before = runtime.audit_snapshot().sink.raw_offered;
    runtime.consume_samples();
    assert_eq!(runtime.audit_snapshot().sink.raw_offered, raw_before);
    let reopened = live.audit();
    assert_eq!(
        (reopened.population.issued, reopened.population.retired),
        (0, 0)
    );
    assert!(
        !reopened.population.closed && !reopened.population.failed,
        "{reopened:#?}"
    );
    assert_eq!(reopened.qualified_publications, 1);
    assert_eq!(reopened.failed_generations, 0);
    let before = runtime.audit_snapshot().structured_feedback.unwrap();
    fixture::record_route(
        &runtime,
        &clock,
        wave("fixture.automatic-physical"),
        false,
        false,
    )
    .unwrap();
    runtime.consume_samples();
    let after = runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(after.compared, before.compared + 1);
    assert_eq!(
        after.uncomparable_observations,
        before.uncomparable_observations
    );
    assert_eq!(
        after.outside_catalog_observations,
        before.outside_catalog_observations
    );
    assert!(after.revoked.is_none());
    // A real complete receipt outside the validated finite descriptor cannot
    // mint the new identity. It must be invalid/uncomparable, not an exclusion
    // that would keep the current automatic feedback model silently trusted.
    let invalid = fixture::record_route(
        &runtime,
        &clock,
        wave_with_kv("fixture.automatic-physical", 129),
        false,
        false,
    )
    .unwrap();
    assert!(invalid
        .structured_evidence
        .as_ref()
        .is_some_and(Result::is_ok));
    runtime.consume_samples();
    let rejected = runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(
        rejected.outside_catalog_observations,
        after.outside_catalog_observations
    );
    assert_eq!(
        rejected.uncomparable_observations,
        after.uncomparable_observations + 1
    );
    runtime.shutdown().await.unwrap();
}

#[path = "automatic_physical/planner.rs"]
mod planner;

#[path = "automatic_physical/shadow_batch.rs"]
mod shadow_batch;

#[path = "automatic_physical/family_progress.rs"]
mod family_progress;
