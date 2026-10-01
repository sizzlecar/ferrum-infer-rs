use super::*;

#[test]
fn automatic_population_schedule_validates_complete_last_block_capacity() {
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    settings.input_readiness = SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {};
    settings.discovery_offered_waves = NonZeroUsize::new(4089).unwrap();
    assert!(settings.validate().is_ok());
    settings.discovery_offered_waves = NonZeroUsize::new(4090).unwrap();
    assert!(settings.validate().is_err());
    settings.population_schedule = SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1;
    assert!(settings.validate().is_ok());
    settings.population_schedule = SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksV1;
    settings.phase_offered_waves = [NonZeroUsize::new(4096).unwrap(); 3];
    settings.discovery_offered_waves = NonZeroUsize::new(256).unwrap();
    assert!(settings.validate().is_ok());
    settings.discovery_offered_waves = NonZeroUsize::new(255).unwrap();
    assert!(settings.validate().is_err());
}
use serde_json::json;

#[test]
fn automatic_encoded_source_budget_is_typed_separate_and_bounded() {
    let defaults = SloAutomaticCalibrationSettingsV1::default();
    for partial in [json!({}), json!({"diagnostics":{"kind":"memory_only"}})] {
        let parsed: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(partial).unwrap();
        assert_eq!(
            parsed.maximum_encoded_source_bytes,
            defaults.maximum_encoded_source_bytes
        );
        parsed.validate().unwrap();
    }
    assert_eq!(
        crate::SloCostProfileImportConfig::default()
            .max_file_bytes
            .get(),
        16 * 1024 * 1024
    );
    assert_eq!(
        defaults.maximum_encoded_source_bytes.get(),
        1024 * 1024 * 1024
    );
    assert!(serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(
        json!({"maximum_encoded_source_bytes":0})
    )
    .is_err());
    let mut settings = defaults;
    // Cumulative hash work is not multiplied by retained generation count.
    settings.maximum_encoded_source_bytes = NonZeroU64::new(MAX_ENCODED_SOURCE_BYTES).unwrap();
    settings.validate().unwrap();
    let encoded = serde_json::to_value(&settings).unwrap();
    assert_eq!(
        serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(encoded).unwrap(),
        settings
    );
    settings.maximum_encoded_source_bytes = NonZeroU64::new(MAX_ENCODED_SOURCE_BYTES + 1).unwrap();
    assert!(settings.validate().is_err());
}

fn automatic() -> SloLiveStructuredCalibration {
    serde_json::from_value(json!({ "kind": "automatic_v1" })).unwrap()
}

#[test]
fn automatic_population_schedule_defaults_survive_partial_settings() {
    for settings in [
        json!({}),
        json!({"diagnostics": {"kind": "memory_only"}}),
        json!({"maximum_owners": 4}),
    ] {
        let policy: SloLiveStructuredCalibration = serde_json::from_value(json!({
            "kind": "automatic_v1", "settings": settings
        }))
        .unwrap();
        let SloLiveStructuredCalibration::AutomaticV1 { settings } = policy else {
            panic!("automatic policy");
        };
        assert_eq!(
            settings.population_schedule,
            SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksV1
        );
        settings.validate().unwrap();
    }
    let explicit: SloAutomaticCalibrationSettingsV1 =
        serde_json::from_value(json!({"population_schedule": "fixed_windows_v1"})).unwrap();
    assert_eq!(
        explicit.population_schedule,
        SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1
    );
    assert_eq!(
        serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(
            serde_json::to_value(&explicit).unwrap()
        )
        .unwrap(),
        explicit
    );
    assert!(serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(
        json!({"population_schedule": "unknown"})
    )
    .is_err());
}

#[test]
fn automatic_reference_work_budget_defaults_roundtrip_and_reject_unbounded_reservations() {
    let settings: SloAutomaticReferenceProbeSettingsV1 = serde_json::from_value(json!({})).unwrap();
    assert_eq!(settings.work_estimate_margin_percent, 25);
    assert_eq!(settings.finalization_reserve_percent, 10);
    assert_eq!(
        serde_json::from_value::<SloAutomaticReferenceProbeSettingsV1>(
            serde_json::to_value(&settings).unwrap()
        )
        .unwrap(),
        settings
    );
    for reserve in [0, 100] {
        let mut invalid = settings.clone();
        invalid.finalization_reserve_percent = reserve;
        assert!(invalid.validate().is_err());
    }
    let mut invalid = settings;
    invalid.work_estimate_margin_percent = 1001;
    assert!(invalid.validate().is_err());
}

#[test]
fn automatic_feedback_is_explicit_bounded_and_requires_no_storage_path() {
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    settings.validate().unwrap();
    let memory = SloSelectedFeedbackStorageV1::MemoryOnly;
    assert!(memory.path().is_none());
    assert_eq!(
        serde_json::to_value(&memory).unwrap(),
        json!({"mode":"memory_only"})
    );
    assert!(serde_json::from_value::<SloSelectedFeedbackStorageV1>(
        json!({"mode":"memory_only","path":"unused"})
    )
    .is_err());
    settings.feedback.minimum_consecutive_underestimates = NonZeroUsize::new(33).unwrap();
    assert!(settings.validate().is_err());
    settings = SloAutomaticCalibrationSettingsV1::default();
    settings.feedback.maximum_state_bytes =
        NonZeroUsize::new(settings.maximum_retained_numeric_bytes.get() + 1).unwrap();
    assert!(settings.validate().is_err());
}

fn config(mode: crate::SloMode) -> crate::SloConfig {
    let mut observation = SloCostObservationConfig::structured_whole_wave_v2();
    observation.live_structured_calibration = automatic();
    crate::SloConfig {
        mode,
        default_service_class: Some("interactive".into()),
        services: vec![crate::ServiceSloConfig {
            id: "interactive".into(),
            server_token_commit: crate::SloLatencyBudgets {
                ttft_ms: NonZeroU64::new(500).unwrap(),
                tpot_ms: NonZeroU64::new(40).unwrap(),
                itl_ms: NonZeroU64::new(80).unwrap(),
            },
            attainment: Default::default(),
            client_visible: None,
        }],
        cost_observation: observation,
        ..Default::default()
    }
}

#[test]
fn automatic_minimal_wire_has_bounded_memory_defaults_and_needs_no_artifact_paths() {
    let policy = automatic();
    let SloLiveStructuredCalibration::AutomaticV1 { settings } = &policy else {
        panic!("explicit automatic policy");
    };
    settings.validate().unwrap();
    assert_eq!(settings.discovery_offered_waves.get(), 256);
    assert_eq!(
        settings.phase_offered_waves.map(NonZeroUsize::get),
        [256; 3]
    );
    assert_eq!(settings.maximum_window_ns.get(), 300_000_000_000);
    assert_eq!(settings.maximum_retained_generations.get(), 4);
    assert_eq!(
        settings.diagnostics,
        SloAutomaticCalibrationDiagnosticsV1::MemoryOnly
    );
    assert_eq!(settings.reference_probe.fresh_trials_per_anchor.get(), 3);
    assert_eq!(settings.reference_probe.maximum_duration_ms.get(), 120_000);
    assert_eq!(
        settings.reference_probe.maximum_source_bytes.get(),
        64 * 1024 * 1024
    );
    assert_eq!(settings.reference_probe.maximum_probe_requests.get(), 128);
    let wire = serde_json::to_value(&policy).unwrap();
    assert_eq!(
        serde_json::from_value::<SloLiveStructuredCalibration>(wire).unwrap(),
        policy
    );
    let config = config(crate::SloMode::Observe);
    config.validate().unwrap();
    assert!(config.cost_profile.is_none());
    assert!(config.prefill_reference.is_none());
    assert!(config
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns
        .is_none());
}

#[test]
fn automatic_keeps_existing_off_admission_predictor_and_explicit_artifact_guards() {
    for mode in [crate::SloMode::Observe, crate::SloMode::Enforce] {
        let mut policy = config(mode);
        policy.validate().unwrap();
        policy.admission.time_policy = crate::SloTimeAdmissionPolicy::RequireSlo;
        assert!(policy.validate().unwrap_err().contains("CompleteRequests"));
        policy.admission.time_policy = crate::SloTimeAdmissionPolicy::CompleteRequests;
        policy.cost_profile = Some(PathBuf::new());
        assert!(policy.validate().unwrap_err().contains("cost_profile path"));
        policy.cost_profile = None;
        policy.prefill_reference = Some(crate::SloPrefillReferenceConfig {
            artifact_path: "explicit-reference.json".into(),
            expected_protocol_sha256: [0; 32],
            limits: Default::default(),
        });
        assert!(policy
            .validate()
            .unwrap_err()
            .contains("explicit protocol digest"));
    }
    assert!(config(crate::SloMode::Off)
        .validate()
        .unwrap_err()
        .contains("Observe/Enforce"));
    let mut policy = config(crate::SloMode::Observe);
    policy.cost_observation.predictor = SloCostPredictor::LegacyFeatureModel;
    assert!(policy.validate().unwrap_err().contains("V2 host-settled"));
    policy.cost_observation.predictor = SloCostPredictor::StructuredWholeWaveV2;
    policy.cost_observation.structured_capture = SloStructuredCostCapture::Disabled;
    assert!(policy.validate().is_err());
}

#[test]
fn automatic_does_not_reinterpret_legacy_model_or_bypass_required_query_guards() {
    let mut policy = config(crate::SloMode::Observe);
    policy.cost_observation.model.drift_margin_ns += 1;
    assert!(policy
        .validate()
        .unwrap_err()
        .contains("legacy model overrides"));
    policy.cost_observation.model = Default::default();
    policy.required_query_observation =
        crate::SloRequiredQueryObservationConfig::StructuredUncalibratedV1 {
            path: "queries.jsonl".into(),
            limits: Default::default(),
        };
    assert!(policy
        .validate()
        .unwrap_err()
        .contains("real prefill reference"));
}

#[test]
fn automatic_rejects_unknown_fields_zero_budgets_and_manual_path_fallbacks() {
    for wire in [
        json!({"kind":"automatic_v1", "declaration":"old.json"}),
        json!({"kind":"automatic_v1", "settings":{"maximum_generations":4}}),
        json!({"kind":"automatic_v1", "settings":{"discovery_offered_waves":0}}),
        json!({"kind":"automatic_v1", "settings":{"phase_offered_waves":[8,0,8]}}),
        json!({"kind":"automatic_v1", "settings":{"phase_offered_waves":[8,8]}}),
        json!({"kind":"automatic_v1", "settings":{"reference_probe":{"maximum_duration_ms":0}}}),
        json!({"kind":"automatic_v1", "settings":{"reference_probe":{"assume_reference_ready":true}}}),
        json!({"kind":"automatic_v1", "settings":{"diagnostics":{"kind":"memory_only", "directory":"unused"}}}),
        json!({"kind":"disabled", "settings":{}}),
    ] {
        assert!(
            serde_json::from_value::<SloLiveStructuredCalibration>(wire.clone()).is_err(),
            "{wire}"
        );
    }
}

#[test]
fn automatic_caps_independent_windows_and_combined_retention_allowance() {
    for (field, value) in [
        ("discovery_offered_waves", json!(65_537)),
        ("phase_offered_waves", json!([7, 8, 8])),
        ("phase_offered_waves", json!([8, 4097, 8])),
        ("maximum_window_ns", json!(MAX_WINDOW_NS + 1)),
        ("maximum_owners", json!(129)),
        ("maximum_discovery_bytes", json!(MAX_STATE_BYTES + 1)),
        ("maximum_retained_numeric_bytes", json!(MAX_STATE_BYTES + 1)),
        (
            "maximum_retained_generations",
            json!(MAX_RETAINED_GENERATIONS + 1),
        ),
    ] {
        let mut wire = serde_json::to_value(SloAutomaticCalibrationSettingsV1::default()).unwrap();
        wire[field] = value;
        let settings: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(wire).unwrap();
        assert!(settings.validate().is_err(), "{field}");
    }
    let mut combined = SloAutomaticCalibrationSettingsV1::default();
    combined.maximum_retained_generations = NonZeroUsize::new(16).unwrap();
    assert!(combined
        .validate()
        .unwrap_err()
        .contains("combined retained"));
    // Each field by itself is within its bound; multiplication plus the
    // independent discovery/reference allowances crosses the aggregate limit.
    assert!(combined.maximum_retained_generations.get() <= MAX_RETAINED_GENERATIONS);
    assert!(combined.maximum_retained_numeric_bytes.get() <= MAX_STATE_BYTES);
}

#[test]
fn automatic_reference_probe_caps_are_limits_and_do_not_claim_protocol_completion() {
    for (field, value) in [
        (
            "fresh_trials_per_anchor",
            json!(crate::PREFILL_REFERENCE_MAX_REPETITIONS + 1),
        ),
        ("maximum_duration_ms", json!(MAX_PROBE_DURATION_MS + 1)),
        (
            "maximum_source_bytes",
            json!(crate::SloCostProfileImportConfig::MAX_FILE_BYTES + 1),
        ),
        (
            "maximum_probe_requests",
            json!(crate::PREFILL_REFERENCE_MAX_SAMPLES + 1),
        ),
    ] {
        let mut wire =
            serde_json::to_value(SloAutomaticReferenceProbeSettingsV1::default()).unwrap();
        wire[field] = value;
        let settings: SloAutomaticReferenceProbeSettingsV1 = serde_json::from_value(wire).unwrap();
        assert!(settings.validate().is_err(), "{field}");
    }
    // A tiny allowed budget is still a budget. The native engine must return
    // Unknown if its complete probe cannot finish; types grant no ready state.
    SloAutomaticReferenceProbeSettingsV1 {
        maximum_probe_requests: NonZeroUsize::MIN,
        maximum_duration_ms: NonZeroU64::MIN,
        ..Default::default()
    }
    .validate()
    .unwrap();
}

#[test]
fn automatic_disk_diagnostics_require_total_bytes_and_retained_generation_limits() {
    let valid = json!({
        "kind":"directory", "directory":"diagnostics",
        "maximum_source_bytes":1024, "maximum_total_bytes":4096,
        "maximum_retained_generations":2
    });
    let diagnostics: SloAutomaticCalibrationDiagnosticsV1 =
        serde_json::from_value(valid.clone()).unwrap();
    diagnostics.validate().unwrap();
    for field in [
        "maximum_source_bytes",
        "maximum_total_bytes",
        "maximum_retained_generations",
    ] {
        let mut wire = valid.clone();
        wire.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<SloAutomaticCalibrationDiagnosticsV1>(wire).is_err());
    }
    for (field, value) in [
        ("directory", json!("")),
        ("maximum_source_bytes", json!(4097)),
        ("maximum_total_bytes", json!(MAX_DIAGNOSTIC_TOTAL_BYTES + 1)),
        ("maximum_retained_generations", json!(65_537)),
    ] {
        let mut wire = valid.clone();
        wire[field] = value;
        let invalid: SloAutomaticCalibrationDiagnosticsV1 = serde_json::from_value(wire).unwrap();
        assert!(invalid.validate().is_err(), "{field}");
    }
    let wire = serde_json::to_value(&diagnostics).unwrap();
    assert_eq!(
        serde_json::from_value::<SloAutomaticCalibrationDiagnosticsV1>(wire).unwrap(),
        diagnostics
    );
}

#[test]
fn automatic_route_population_defaults_match_omitted_empty_and_partial_runtime_settings() {
    let SloLiveStructuredCalibration::AutomaticV1 { settings } = automatic() else {
        panic!("automatic")
    };
    assert_eq!(
        settings.route_population,
        SloCalibrationRoutePopulationV1::WarmOrGraphDisabledWithNoSubmissionV2
    );
    for partial in [
        json!({}),
        json!({"diagnostics": {"kind": "memory_only"}}),
        json!({"reference_probe": {"maximum_duration_ms": 1000}}),
    ] {
        let partial: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(partial).unwrap();
        assert_eq!(partial.route_population, settings.route_population);
        partial.validate().unwrap();
    }
    assert_eq!(
        serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(
            serde_json::to_value(&settings).unwrap()
        )
        .unwrap(),
        settings
    );
}

#[test]
fn automatic_runtime_explicit_route_populations_keep_their_meaning() {
    assert_eq!(
        SloCalibrationRoutePopulationV1::default(),
        SloCalibrationRoutePopulationV1::AllAttempts
    );
    for population in [
        SloCalibrationRoutePopulationV1::AllAttempts,
        SloCalibrationRoutePopulationV1::WarmOrGraphDisabledV1,
        SloCalibrationRoutePopulationV1::WarmOrGraphDisabledWithNoSubmissionV2,
    ] {
        let explicit: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(json!({
            "route_population": population,
            "diagnostics": {"kind": "memory_only"}
        }))
        .unwrap();
        assert_eq!(explicit.route_population, population);
        explicit.validate().unwrap();
    }
}
#[test]
fn rolling_population_is_explicit_and_requires_input_readiness() {
    let original = SloAutomaticCalibrationSettingsV1::default();
    assert_eq!(
        original.population_schedule,
        SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksV1
    );
    let mut rolling = original.clone();
    rolling.population_schedule = SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2;
    rolling.validate().unwrap();
    let encoded = serde_json::to_value(&rolling).unwrap();
    assert_eq!(encoded["population_schedule"], "owner_blocks_rolling_v2");
    assert_eq!(
        serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(encoded).unwrap(),
        rolling
    );
    rolling.input_readiness = SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {};
    assert!(rolling.validate().is_err());
}
