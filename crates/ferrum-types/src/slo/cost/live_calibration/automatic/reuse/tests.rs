use super::*;
use serde_json::json;

#[test]
fn automatic_reuse_missing_and_partial_settings_use_same_boot_defaults() {
    let defaults = SloAutomaticCalibrationSettingsV1::default();
    let mut previous = serde_json::to_value(&defaults).unwrap();
    previous.as_object_mut().unwrap().remove("reuse");
    assert_eq!(
        serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(previous).unwrap(),
        defaults
    );
    for partial in [
        json!({}),
        json!({"reuse":{"kind":"same_boot_clean_shutdown_v1"}}),
        json!({"reuse":{"kind":"same_boot_clean_shutdown_v1","limits":{}}}),
    ] {
        let settings: SloAutomaticCalibrationSettingsV1 = serde_json::from_value(partial).unwrap();
        settings.validate().unwrap();
        assert_eq!(settings.reuse, defaults.reuse);
        assert_eq!(settings.route_population, defaults.route_population);
        assert_eq!(
            settings.diagnostics,
            SloAutomaticCalibrationDiagnosticsV1::MemoryOnly
        );
    }
    let SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 { location, limits } =
        defaults.reuse
    else {
        panic!("automatic same-boot default")
    };
    assert_eq!(
        location,
        SloAutomaticCalibrationCacheLocationV1::PlatformDefault {}
    );
    assert_eq!(limits.maximum_total_bytes.get(), 256 * 1024 * 1024);
    assert_eq!(limits.maximum_operation_duration_ms.get(), 30_000);
}

#[test]
fn automatic_reuse_disabled_and_directory_roundtrip_without_manual_import() {
    for reuse in [
        SloAutomaticCalibrationReuseV1::Disabled {},
        SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 {
            location: SloAutomaticCalibrationCacheLocationV1::Directory {
                path: "cache/calibration".into(),
            },
            limits: SloAutomaticCalibrationReuseLimitsV1::default(),
        },
    ] {
        let settings = SloAutomaticCalibrationSettingsV1 {
            reuse: reuse.clone(),
            ..Default::default()
        };
        settings.validate().unwrap();
        let parsed: SloAutomaticCalibrationSettingsV1 =
            serde_json::from_slice(&serde_json::to_vec(&settings).unwrap()).unwrap();
        assert_eq!(parsed, settings);
        assert_eq!(parsed.reuse, reuse);
        assert_eq!(
            parsed.diagnostics,
            SloAutomaticCalibrationDiagnosticsV1::MemoryOnly
        );
    }
    assert_eq!(crate::SloConfig::default().mode, crate::SloMode::Off);
    assert!(SloLiveStructuredCalibration::default().is_disabled());
}

#[test]
fn automatic_reuse_rejects_unknown_fields_and_zero_resource_limits() {
    for wire in [
        json!({"kind":"disabled","location":{"kind":"platform_default"}}),
        json!({"kind":"same_boot_clean_shutdown_v1","unsafe_ignore_age":true}),
        json!({"kind":"same_boot_clean_shutdown_v1","location":{"kind":"platform_default","path":"unused"}}),
        json!({"kind":"same_boot_clean_shutdown_v1","location":{"kind":"directory","path":"cache","other":1}}),
        json!({"kind":"same_boot_clean_shutdown_v1","limits":{"maximum_total_bytes":0}}),
        json!({"kind":"same_boot_clean_shutdown_v1","limits":{"maximum_operation_duration_ms":0}}),
        json!({"kind":"same_boot_clean_shutdown_v1","limits":{"maximum_sources":9999}}),
    ] {
        assert!(
            serde_json::from_value::<SloAutomaticCalibrationReuseV1>(wire.clone()).is_err(),
            "accepted undeclared policy: {wire}"
        );
    }
}

#[test]
fn automatic_reuse_checks_disk_and_whole_transaction_bounds_through_settings() {
    let bounded = SloAutomaticCalibrationReuseLimitsV1 {
        maximum_total_bytes: NonZeroU64::new(MAX_DIAGNOSTIC_TOTAL_BYTES).unwrap(),
        maximum_operation_duration_ms: NonZeroU64::new(MAX_PROBE_DURATION_MS).unwrap(),
    };
    bounded.validate().unwrap();
    for limits in [
        SloAutomaticCalibrationReuseLimitsV1 {
            maximum_total_bytes: NonZeroU64::new(MAX_DIAGNOSTIC_TOTAL_BYTES + 1).unwrap(),
            ..bounded.clone()
        },
        SloAutomaticCalibrationReuseLimitsV1 {
            maximum_operation_duration_ms: NonZeroU64::new(MAX_PROBE_DURATION_MS + 1).unwrap(),
            ..bounded.clone()
        },
    ] {
        let settings = SloAutomaticCalibrationSettingsV1 {
            reuse: SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 {
                location: SloAutomaticCalibrationCacheLocationV1::default(),
                limits,
            },
            ..Default::default()
        };
        assert!(settings.validate().is_err());
    }
    let invalid = SloAutomaticCalibrationSettingsV1 {
        reuse: SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 {
            location: SloAutomaticCalibrationCacheLocationV1::Directory { path: "".into() },
            limits: bounded,
        },
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn automatic_reuse_does_not_expand_existing_memory_or_generation_caps() {
    for reuse in [
        SloAutomaticCalibrationReuseV1::default(),
        SloAutomaticCalibrationReuseV1::Disabled {},
    ] {
        let mut settings = SloAutomaticCalibrationSettingsV1 {
            reuse,
            ..Default::default()
        };
        settings.maximum_retained_generations =
            NonZeroUsize::new(MAX_RETAINED_GENERATIONS + 1).unwrap();
        assert!(settings.validate().is_err());
        settings.maximum_retained_generations = NonZeroUsize::new(1).unwrap();
        settings.maximum_retained_numeric_bytes = NonZeroUsize::new(MAX_STATE_BYTES + 1).unwrap();
        assert!(settings.validate().is_err());
    }
}
