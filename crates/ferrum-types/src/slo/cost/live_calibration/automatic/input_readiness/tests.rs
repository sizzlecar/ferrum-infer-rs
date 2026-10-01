use super::*;
use serde_json::json;

fn geometry(blocks: [usize; 3], visits: u64) -> SloAutomaticCalibrationInputReadinessV1 {
    SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV2 {
        maximum_phase_blocks: blocks.map(|n| NonZeroUsize::new(n).unwrap()),
        maximum_geometry_visits: NonZeroU64::new(visits).unwrap(),
    }
}

fn zero_column_geometry(
    blocks: [usize; 3],
    visits: u64,
) -> SloAutomaticCalibrationInputReadinessV1 {
    SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV3 {
        maximum_phase_blocks: blocks.map(|n| NonZeroUsize::new(n).unwrap()),
        maximum_geometry_visits: NonZeroU64::new(visits).unwrap(),
    }
}

#[test]
fn automatic_input_readiness_defaults_survive_partial_settings_and_roundtrip() {
    let defaults = SloAutomaticCalibrationSettingsV1::default();
    assert_eq!(
        defaults.input_readiness,
        zero_column_geometry([16; 3], 32_000_000)
    );
    for partial in [json!({}), json!({"diagnostics": {"kind":"memory_only"}})] {
        let settings: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(partial).unwrap();
        assert_eq!(settings.input_readiness, defaults.input_readiness);
        settings.validate().unwrap();
        assert_eq!(
            serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(
                serde_json::to_value(&settings).unwrap()
            )
            .unwrap(),
            settings
        );
    }
    let legacy: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(json!({
        "input_readiness": {"kind":"count_only_v1"}
    }))
    .unwrap();
    assert_eq!(
        legacy.input_readiness,
        SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {}
    );
    legacy.validate().unwrap();
}

#[test]
fn automatic_input_readiness_checks_complete_block_products_and_phase_minima() {
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    settings.input_readiness = geometry([16; 3], MAX_GEOMETRY_VISITS);
    settings.validate().unwrap(); // 256 * 16 = the unchanged 4096-member cap.
    settings.input_readiness = geometry([16, 17, 16], MAX_GEOMETRY_VISITS);
    assert!(settings.validate().is_err());
    settings.input_readiness = geometry([1; 3], 1);
    settings.validate().unwrap();
    settings.phase_offered_waves[1] = NonZeroUsize::new(257).unwrap();
    assert!(settings.validate().is_err()); // The second whole block is mandatory.
    settings.input_readiness = geometry([1, 2, 1], 1);
    settings.validate().unwrap();
    settings.discovery_offered_waves = NonZeroUsize::new(65_536).unwrap();
    assert!(settings.validate().is_err());
    settings.discovery_offered_waves = NonZeroUsize::new(4096).unwrap();
    settings.input_readiness = geometry([1; 3], 1);
    settings.validate().unwrap();
    assert!(geometry([2; 3], 1)
        .validate_owner_blocks(usize::MAX, [8; 3])
        .is_err());
}

#[test]
fn automatic_input_readiness_does_not_enlarge_age_or_retention_and_keeps_fixed_windows() {
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    let age = settings.maximum_window_ns;
    let bytes = settings.maximum_retained_numeric_bytes;
    settings.input_readiness = geometry([16; 3], MAX_GEOMETRY_VISITS + 1);
    assert!(settings.validate().is_err());
    settings.input_readiness = geometry([4097; 3], 1);
    assert!(settings.validate().is_err());
    settings.input_readiness = geometry([16; 3], 1);
    settings.population_schedule = SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1;
    settings.discovery_offered_waves = NonZeroUsize::new(65_536).unwrap();
    settings.validate().unwrap(); // Its original fixed-window limits remain separate.
    assert_eq!(settings.maximum_window_ns, age);
    assert_eq!(settings.maximum_retained_numeric_bytes, bytes);
    for invalid in [
        json!({"kind":"count_only_v1","maximum_geometry_visits":1}),
        json!({"kind":"work_axes_and_branches_v1","maximum_phase_blocks":[0,1,1],"maximum_geometry_visits":1}),
        json!({"kind":"work_axes_and_branches_v1","maximum_phase_blocks":[1,1,1],"maximum_geometry_visits":0}),
    ] {
        assert!(
            serde_json::from_value::<SloAutomaticCalibrationInputReadinessV1>(invalid).is_err()
        );
    }
}

#[test]
fn automatic_input_readiness_keeps_explicit_legacy_geometry_and_rejects_unknown_v2_fields() {
    let old = json!({"kind":"work_axes_and_branches_v1","maximum_phase_blocks":[16,16,16],"maximum_geometry_visits":32000000});
    let parsed: SloAutomaticCalibrationInputReadinessV1 =
        serde_json::from_value(old.clone()).unwrap();
    parsed.validate().unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), old);
    assert_ne!(parsed, SloAutomaticCalibrationInputReadinessV1::default());
    // Explicit V2 retains its wire and strict parsing after the default moves
    // to V3; omitted settings choose the new version independently.
    let explicit_v2 = geometry([16; 3], 32_000_000);
    let mut revised = serde_json::to_value(explicit_v2).unwrap();
    assert_eq!(revised["kind"], "work_axes_and_branches_v2");
    assert_eq!(
        serde_json::from_value::<SloAutomaticCalibrationInputReadinessV1>(revised.clone()).unwrap(),
        explicit_v2
    );
    assert_ne!(
        explicit_v2,
        SloAutomaticCalibrationInputReadinessV1::default()
    );
    revised["unexpected"] = json!(true);
    assert!(serde_json::from_value::<SloAutomaticCalibrationInputReadinessV1>(revised).is_err());
}

#[test]
fn automatic_input_readiness_v3_is_default_with_original_resource_limits() {
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    let before = settings.clone();
    settings.input_readiness = SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV3 {
        maximum_phase_blocks: [NonZeroUsize::new(16).unwrap(); 3],
        maximum_geometry_visits: NonZeroU64::new(32_000_000).unwrap(),
    };
    settings.validate().unwrap();
    let wire = serde_json::to_value(&settings).unwrap();
    assert_eq!(wire["input_readiness"]["kind"], "work_axes_and_branches_v3");
    assert_eq!(
        serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(wire).unwrap(),
        settings
    );
    assert_eq!(settings.maximum_window_ns, before.maximum_window_ns);
    assert_eq!(
        settings.maximum_retained_numeric_bytes,
        before.maximum_retained_numeric_bytes
    );
    assert_eq!(
        before.input_readiness,
        zero_column_geometry([16; 3], 32_000_000)
    );
    settings.input_readiness = SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV3 {
        maximum_phase_blocks: [NonZeroUsize::new(17).unwrap(); 3],
        maximum_geometry_visits: NonZeroU64::new(32_000_000).unwrap(),
    };
    assert!(settings.validate().is_err());
    settings.input_readiness = SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV3 {
        maximum_phase_blocks: [NonZeroUsize::new(16).unwrap(); 3],
        maximum_geometry_visits: NonZeroU64::new(MAX_GEOMETRY_VISITS + 1).unwrap(),
    };
    assert!(settings.validate().is_err());
}
