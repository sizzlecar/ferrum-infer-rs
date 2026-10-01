use super::*;

#[tokio::test]
async fn automatic_constructor_allows_unknown_initial_coverage_and_preserves_explicit_load_guards()
{
    let fixture = Fixture::new().await;
    let mut config = fixture.config.clone();
    config.scheduler.slo.cost_observation =
        ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration = ferrum_types::SloLiveStructuredCalibration::AutomaticV1 {
        settings: Default::default(),
    };
    config.scheduler.slo.cost_profile = None;
    config.scheduler.slo.prefill_reference = None;
    validate_execution(&config, fixture.executor.as_ref(), false).unwrap();
    validate_loaded(&config, None, None).unwrap();
    let engine = fixture.build(config.clone()).unwrap();
    assert!(engine.inner.prefill_reference_runtime.is_none());
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    assert!(engine.inner.config.slo_cost_profile_receipt.is_none());
    engine.shutdown().await.unwrap();

    let mut explicit = config.clone();
    explicit.scheduler.slo.cost_profile = fixture.config.scheduler.slo.cost_profile.clone();
    assert!(
        validate_loaded(&explicit, None, None).is_err(),
        "automatic cannot rescue an explicitly requested but unusable cost import"
    );
    explicit = config.clone();
    let mut reference = fixture
        .config
        .scheduler
        .slo
        .prefill_reference
        .clone()
        .unwrap();
    reference.artifact_path = fixture.dir.join("missing-explicit-reference.json");
    explicit.scheduler.slo.prefill_reference = Some(reference);
    assert!(
        fixture.build(explicit).is_err(),
        "automatic cannot replace an invalid explicit reference"
    );

    let mut strict = config.clone();
    strict.scheduler.slo.admission.time_policy = SloTimeAdmissionPolicy::RequireSlo;
    assert!(validate_execution(&strict, fixture.executor.as_ref(), false).is_err());
    assert!(validate_execution(&config, fixture.executor.as_ref(), true).is_err());
    config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration = ferrum_types::SloLiveStructuredCalibration::Disabled;
    assert!(validate_execution(&config, fixture.executor.as_ref(), false).is_err());
}

#[tokio::test]
async fn automatic_profile_composition_checks_attached_sink_for_run_and_serve() {
    use ferrum_interfaces::vnext::DeviceTimingMode;
    use ferrum_types::{ObservabilityProfileDetail as Detail, ProfileEntrypoint};
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        for (detail, journal) in [
            (Detail::Off, false),
            (Detail::Basic, false),
            (Detail::Basic, true),
            (Detail::Replay, true),
            (Detail::Kernel, true),
            (Detail::Verify, true),
            (Detail::Full, true),
        ] {
            let fixture = Fixture::new().await;
            let mut config = fixture.config.clone();
            config.scheduler.slo.cost_profile = None;
            config.scheduler.slo.prefill_reference = None;
            config.scheduler.slo.cost_observation =
                ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
            config
                .scheduler
                .slo
                .cost_observation
                .live_structured_calibration =
                ferrum_types::SloLiveStructuredCalibration::AutomaticV1 {
                    settings: Default::default(),
                };
            config.runtime.profile_entrypoint = Some(entrypoint);
            config.runtime.profile_detail = detail;
            config.runtime.profile_jsonl = journal.then(|| fixture.dir.join("profile.jsonl"));
            validate_execution(&config, fixture.executor.as_ref(), false).unwrap();
            let result = fixture.build(config);
            if matches!(detail, Detail::Off | Detail::Basic) {
                let engine = result.expect("completion-only profiling supports guarded startup");
                let mode = fixture
                    .executor
                    .profile_sink
                    .lock()
                    .as_ref()
                    .map(|s| s.device_timing_mode())
                    .unwrap_or(DeviceTimingMode::Off);
                assert_eq!(mode, DeviceTimingMode::for_profile_detail(detail));
                assert_eq!(
                    fixture.executor.slo_execution_capability(),
                    ExecutorSloCapability::GuardedEagerWaves
                );
                assert!(engine.inner.cost_runtime.is_some());
                assert!(!engine
                    .inner
                    .bg_loop_spawned
                    .load(std::sync::atomic::Ordering::Acquire));
                engine.shutdown().await.unwrap();
            } else {
                assert!(
                    matches!(result, Err(FerrumError::Unsupported { .. })),
                    "final attached sink must be checked before bootstrap"
                );
            }
            assert_eq!(
                fixture
                    .executor
                    .entries
                    .load(std::sync::atomic::Ordering::Acquire),
                0
            );
            assert_eq!(
                fixture
                    .executor
                    .physical
                    .load(std::sync::atomic::Ordering::Acquire),
                0
            );
        }
    }
}

#[tokio::test]
async fn automatic_completion_profile_bootstrap_records_original_runtime_identity() {
    use ferrum_types::{ObservabilityProfileDetail, ProfileEntrypoint};
    use std::{
        num::{NonZeroU64, NonZeroUsize},
        sync::atomic::Ordering,
    };
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        let fixture = Fixture::new().await;
        let mut config = fixture.config.clone();
        config.scheduler.slo.cost_profile = None;
        config.scheduler.slo.prefill_reference = None;
        config.runtime.max_model_len = Some(7);
        config.runtime.profile_entrypoint = Some(entrypoint);
        config.runtime.profile_detail = ObservabilityProfileDetail::Basic;
        config.runtime.profile_jsonl = Some(fixture.dir.join("bootstrap-profile.jsonl"));
        let mut settings = ferrum_types::SloAutomaticCalibrationSettingsV1::default();
        settings.reference_probe.fresh_trials_per_anchor = NonZeroUsize::MIN;
        settings.reference_probe.maximum_probe_requests = NonZeroUsize::new(128).unwrap();
        settings.reference_probe.maximum_duration_ms = NonZeroU64::new(30_000).unwrap();
        config.scheduler.slo.cost_observation =
            ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
        config
            .scheduler
            .slo
            .cost_observation
            .live_structured_calibration =
            ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings };
        fixture
            .executor
            .recycle_completed_bindings
            .store(true, Ordering::Release);
        fixture
            .executor
            .emit_cost_observations
            .store(true, Ordering::Release);
        let engine = fixture.build(config).unwrap();
        let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
        let original_identity = fixture.executor.execution_cost_identity();
        assert_eq!(runtime.identity, original_identity);
        let engine =
            tokio::time::timeout(Duration::from_secs(40), engine.finish_automatic_startup())
                .await
                .unwrap()
                .unwrap();
        assert!(engine.inner.prefill_reference_runtime.is_some());
        assert_eq!(runtime.identity, fixture.executor.execution_cost_identity());
        assert!(
            Arc::ptr_eq(&runtime, engine.inner.cost_runtime.as_ref().unwrap()),
            "bootstrap must retain the original already-bound observation runtime"
        );
        assert!(fixture.executor.physical.load(Ordering::Acquire) > 0);
        engine.shutdown().await.unwrap();
        let audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
        assert!(audit["training"]["consumed"].as_u64().unwrap() > 0);
        // Structured V2 has no legacy trainer. Verify actual acceptance at the
        // original preparation/FIFO boundary instead of that unrelated counter.
        for field in ["offered_completed", "published", "drained"] {
            assert!(audit["sink"][field].as_u64().unwrap() > 0, "{field}");
        }
        let sink = runtime.audit_snapshot().sink;
        use crate::continuous_engine::inner::cost_observation::CostCallRejection;
        for rejection in [
            CostCallRejection::IdentityUnknown,
            CostCallRejection::IdentitySchema,
        ] {
            assert_eq!(sink.preparation_rejected[rejection as usize], 0);
            assert_eq!(sink.initialization_rejected[rejection as usize], 0);
        }
        let errors = audit["training"]["outcomes"]["errors"].as_array().unwrap();
        assert!(errors
            .iter()
            .filter(|v| v["reason"] == "fingerprint_mismatch")
            .all(|v| v["count"] == 0));
    }
}
