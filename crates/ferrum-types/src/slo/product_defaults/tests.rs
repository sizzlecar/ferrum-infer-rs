use super::*;
use serde_json::json;

fn minimal() -> Value {
    json!({
        "mode": "enforce",
        "default_service_class": "interactive",
        "services": [{
            "id": "interactive",
            "server_token_commit": { "ttft_ms": 500, "tpot_ms": 40, "itl_ms": 80 }
        }]
    })
}

fn parse(value: Value) -> SloConfig {
    serde_json::from_value(value).unwrap()
}

fn automatic(config: &SloConfig) -> &SloAutomaticCalibrationSettingsV1 {
    let SloLiveStructuredCalibration::AutomaticV1 { settings } =
        &config.cost_observation.live_structured_calibration
    else {
        panic!("expected automatic calibration");
    };
    settings
}

#[test]
fn minimal_enforce_selects_the_product_preset_with_original_bounds() {
    let config = parse(minimal());
    config.validate().unwrap();
    assert!(config.cost_profile.is_none());
    assert!(config.prefill_reference.is_none());
    assert_eq!(config.output.transport, SloOutputTransport::Credited);
    assert_eq!(
        config.cost_observation.predictor,
        SloCostPredictor::StructuredWholeWaveV2
    );
    assert_eq!(
        config.cost_observation.structured_capture,
        SloStructuredCostCapture::HostSettledV1
    );
    assert_eq!(
        config.cost_observation.structured_actual_capture,
        SloStructuredActualCapturePolicy::ConsumerDrivenV1
    );
    let expected = SloAutomaticCalibrationSettingsV1 {
        population_schedule: SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2,
        numerical_strategy:
            SloAutomaticCalibrationNumericalStrategyV1::IdentifiedFitGlobalResidualV1,
        ..Default::default()
    };
    // Includes readiness, cache reuse, original age/qualification limits and
    // both startup probe budgets; mode selection must not relax any of them.
    assert_eq!(automatic(&config), &expected);
    let bounded = SloCostObservationConfig {
        predictor: config.cost_observation.predictor,
        structured_capture: config.cost_observation.structured_capture,
        structured_actual_capture: config.cost_observation.structured_actual_capture,
        live_structured_calibration: config.cost_observation.live_structured_calibration.clone(),
        ..Default::default()
    };
    assert_eq!(config.cost_observation, bounded);
}

#[test]
fn partial_resource_limits_inherit_automatic_selection() {
    let mut wire = minimal();
    wire["cost_observation"] = json!({
        "max_queued_samples": 17, "max_samples_per_update": 11,
        "maximum_template_bytes": 1048576
    });
    wire["output"] = json!({ "max_queued_events_per_request": 19 });
    let config = parse(wire);
    config.validate().unwrap();
    automatic(&config);
    assert_eq!(config.cost_observation.max_queued_samples.get(), 17);
    assert_eq!(config.cost_observation.max_samples_per_update.get(), 11);
    assert_eq!(
        config.cost_observation.maximum_template_bytes.get(),
        1048576
    );
    assert_eq!(config.output.max_queued_events_per_request.get(), 19);
    assert_eq!(config.output.transport, SloOutputTransport::Credited);
}

#[test]
fn explicit_legacy_disabled_and_manual_artifacts_are_not_promoted() {
    for cost in [
        json!({ "predictor": "legacy_feature_model" }),
        json!({ "structured_capture": "disabled" }),
        json!({ "live_structured_calibration": { "kind": "disabled" } }),
        json!({ "model": { "min_samples": 9 } }),
        json!({ "profile_export": {
            "path": "cost.json", "observations_path": "observations.jsonl",
            "declared_clock_max_error_ns": 0
        } }),
    ] {
        let mut wire = minimal();
        wire["cost_observation"] = cost;
        let config = parse(wire);
        config.validate().unwrap();
        assert!(config
            .cost_observation
            .live_structured_calibration
            .is_disabled());
        assert_eq!(
            config.cost_observation.predictor,
            SloCostPredictor::LegacyFeatureModel
        );
    }
    for (field, artifact) in [
        ("cost_profile", json!("cost.json")),
        (
            "prefill_reference",
            serde_json::to_value(SloPrefillReferenceConfig {
                artifact_path: "reference.json".into(),
                expected_protocol_sha256: [7; 32],
                limits: Default::default(),
            })
            .unwrap(),
        ),
    ] {
        let mut wire = minimal();
        wire[field] = artifact;
        let config = parse(wire);
        config.validate().unwrap();
        assert!(config
            .cost_observation
            .live_structured_calibration
            .is_disabled());
    }
    let mut wire = minimal();
    wire["output"] = json!({ "transport": "legacy" });
    let config = parse(wire);
    automatic(&config);
    // Execution validation retains authority to reject this explicit choice.
    assert_eq!(config.output.transport, SloOutputTransport::Legacy);
}

#[test]
fn explicit_automatic_keeps_its_declared_settings_and_capture_policy() {
    let mut wire = minimal();
    wire["cost_observation"] = json!({
        "live_structured_calibration": { "kind": "automatic_v1", "settings": {} }
    });
    let config = parse(wire.clone());
    config.validate().unwrap();
    assert_eq!(
        automatic(&config),
        &SloAutomaticCalibrationSettingsV1::default()
    );
    assert_eq!(
        config.cost_observation.structured_actual_capture,
        SloStructuredActualCapturePolicy::ConsumerDrivenV1
    );

    let declared = SloAutomaticCalibrationSettingsV1 {
        maximum_owners: NonZeroUsize::new(7).unwrap(),
        reuse: SloAutomaticCalibrationReuseV1::Disabled {},
        ..Default::default()
    };
    wire["cost_observation"]["live_structured_calibration"]["settings"] =
        serde_json::to_value(&declared).unwrap();
    wire["cost_observation"]["structured_actual_capture"] = json!("legacy_every_wave");
    let config = parse(wire);
    config.validate().unwrap();
    assert_eq!(automatic(&config), &declared);
    assert_eq!(
        config.cost_observation.structured_actual_capture,
        SloStructuredActualCapturePolicy::LegacyEveryWave
    );

    let mut incompatible = minimal();
    incompatible["cost_observation"] = json!({
        "predictor": "legacy_feature_model",
        "live_structured_calibration": { "kind": "automatic_v1" }
    });
    let config = parse(incompatible);
    assert_eq!(
        config.cost_observation.predictor,
        SloCostPredictor::LegacyFeatureModel
    );
    assert!(config.validate().unwrap_err().contains("V2 host-settled"));
}

#[test]
fn effective_roundtrip_preserves_explicit_disabled_and_automatic_choices() {
    let automatic = parse(minimal());
    let mut legacy = automatic.clone();
    legacy.cost_observation = SloCostObservationConfig::default();
    legacy.output = SloOutputConfig::default();
    for original in [automatic, legacy, SloConfig::default()] {
        let wire = serde_json::to_value(&original).unwrap();
        let cost = &wire["cost_observation"];
        for field in [
            "predictor",
            "structured_capture",
            "structured_actual_capture",
            "live_structured_calibration",
        ] {
            assert!(
                cost.get(field).is_some(),
                "effective config omitted {field}"
            );
        }
        let decoded = parse(wire);
        assert_eq!(decoded, original);
        let text = serde_json::to_string(&original).unwrap();
        // Deserializing text also rejects duplicate fields in the effective
        // serialization, unlike passing an already collapsed JSON Value.
        assert_eq!(serde_json::from_str::<SloConfig>(&text).unwrap(), original);
    }
}

#[test]
fn off_observe_and_controlled_experiments_keep_their_existing_defaults() {
    for mode in ["off", "observe"] {
        let mut wire = minimal();
        wire["mode"] = json!(mode);
        let config = parse(wire);
        config.validate().unwrap();
        assert!(config
            .cost_observation
            .live_structured_calibration
            .is_disabled());
        assert_eq!(config.output.transport, SloOutputTransport::Legacy);
    }
    let mut wire = minimal();
    wire["experiment_stage"] =
        serde_json::to_value(SloExperimentStageV1::ControlledAdaptiveBaseline).unwrap();
    let config = parse(wire);
    config.validate().unwrap();
    assert!(config
        .cost_observation
        .live_structured_calibration
        .is_disabled());
    assert_eq!(config.output.transport, SloOutputTransport::Legacy);
}

#[test]
fn strict_admission_retains_its_explicit_manual_contract() {
    let mut wire = minimal();
    wire["admission"] = json!({ "time_policy": "require-slo" });
    let config = parse(wire.clone());
    assert!(config
        .cost_observation
        .live_structured_calibration
        .is_disabled());
    assert!(config
        .validate()
        .unwrap_err()
        .contains("explicit cost_profile"));
    wire["cost_observation"] = json!({ "live_structured_calibration": { "kind": "automatic_v1" } });
    assert!(parse(wire)
        .validate()
        .unwrap_err()
        .contains("CompleteRequests"));
}

#[test]
fn product_defaults_retain_strict_typed_wire_validation() {
    for (field, value) in [
        ("cost_observation", json!({ "max_queued_sampels": 17 })),
        (
            "cost_observation",
            json!({ "live_structured_calibration": {
            "kind": "automatic_v1", "settings": { "maximum_owner": 7 }
        } }),
        ),
        (
            "cost_observation",
            json!({ "live_structured_calibration": {
            "kind": "disabled", "ignored": true
        } }),
        ),
        ("cost_observation", json!({ "predictor": null })),
        ("cost_observation", json!({ "max_queued_samples": 0 })),
        ("output", json!({ "transprot": "credited" })),
        ("unexpected", json!(true)),
    ] {
        let mut wire = minimal();
        wire[field] = value;
        assert!(
            serde_json::from_value::<SloConfig>(wire).is_err(),
            "{field}"
        );
    }
    assert!(serde_json::from_str::<SloConfig>(
        r#"{"mode":"enforce","cost_observation":{"predictor":"legacy_feature_model","predictor":"structured_whole_wave_v2"}}"#
    ).is_err());
    for duplicate in [
        r#"{"cost_observation":{"live_structured_calibration":{"kind":"automatic_v1","settings":{"maximum_owners":7,"maximum_owners":8}}}}"#,
        r#"{"cost_observation":{"model":{"min_samples":8,"min_samples":9}}}"#,
    ] {
        assert!(serde_json::from_str::<SloConfig>(duplicate).is_err());
    }
}
