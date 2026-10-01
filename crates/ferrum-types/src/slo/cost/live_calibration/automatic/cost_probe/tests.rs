use super::*;
use serde_json::json;

#[test]
fn automatic_cost_probe_defaults_preserve_existing_partial_configuration() {
    for wire in [
        json!({}),
        json!({"maximum_owners": 4}),
        json!({"cost_probe":{"maximum_output_tokens":2}}),
    ] {
        let settings: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(wire).unwrap();
        settings.validate().unwrap();
        let roundtrip: SloAutomaticCalibrationSettingsV1 =
            serde_json::from_value(serde_json::to_value(&settings).unwrap()).unwrap();
        assert_eq!(roundtrip, settings);
        assert_eq!(
            settings.cost_probe.maximum_input_projection_requests.get(),
            2048
        );
        assert_eq!(
            settings.phase_offered_waves,
            SloAutomaticCalibrationSettingsV1::default().phase_offered_waves
        );
    }
}

#[test]
fn automatic_cost_probe_rejects_unbounded_or_duplicate_declarations() {
    for field in [
        "maximum_offered_waves",
        "maximum_probe_requests",
        "maximum_input_projection_requests",
        "maximum_output_tokens",
    ] {
        let mut wire = serde_json::to_value(SloAutomaticCostProbeSettingsV1::default()).unwrap();
        wire[field] = json!(65_537);
        let settings: SloAutomaticCostProbeSettingsV1 = serde_json::from_value(wire).unwrap();
        assert!(settings.validate().is_err());
    }
    for presets in [json!([]), json!(["configured", "configured"])] {
        let settings: SloAutomaticCostProbeSettingsV1 =
            serde_json::from_value(json!({"sampling_presets":presets})).unwrap();
        assert!(settings.validate().is_err());
    }
    assert!(serde_json::from_value::<SloAutomaticCostProbeSettingsV1>(
        json!({"maximum_probe_requests":0})
    )
    .is_err());
    assert!(serde_json::from_value::<SloAutomaticCostProbeSettingsV1>(
        json!({"maximum_input_projection_requests":0})
    )
    .is_err());
    assert!(serde_json::from_value::<SloAutomaticCostProbeSettingsV1>(
        json!({"minimum_samples":1})
    )
    .is_err());
}

#[test]
fn automatic_cost_probe_token_discovery_limits_are_independent_resource_bounds() {
    let mut settings = SloAutomaticPrefixTokenDiscoveryBudgetV1::default();
    settings.maximum_prefix_tokens = NonZeroUsize::new(65).unwrap();
    assert!(settings.validate().is_err());
    settings = Default::default();
    settings.maximum_total_token_bytes = NonZeroUsize::new(128 * 1024 * 1024 + 1).unwrap();
    assert!(settings.validate().is_err());
    settings = Default::default();
    settings.maximum_search_states = NonZeroUsize::new(65_537).unwrap();
    assert!(settings.validate().is_err());
}

#[test]
fn automatic_cost_probe_projection_owner_limit_is_independent_of_execution_requests() {
    let settings: SloAutomaticCostProbeSettingsV1 = serde_json::from_value(json!({
        "maximum_input_projection_requests": 1
    }))
    .unwrap();
    settings.validate().unwrap();
    assert_eq!(settings.maximum_input_projection_requests.get(), 1);
    assert_eq!(settings.maximum_probe_requests.get(), 2048);
    let settings: SloAutomaticCostProbeSettingsV1 = serde_json::from_value(json!({
        "maximum_input_projection_requests": 65536
    }))
    .unwrap();
    settings.validate().unwrap();
    assert_eq!(settings.maximum_probe_requests.get(), 2048);
}
