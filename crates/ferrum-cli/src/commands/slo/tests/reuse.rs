use super::*;
use ferrum_types::{
    SloAutomaticCalibrationCacheLocationV1 as Location, SloAutomaticCalibrationDiagnosticsV1,
    SloAutomaticCalibrationReuseLimitsV1, SloAutomaticCalibrationReuseV1 as Reuse,
    SloAutomaticCalibrationSettingsV1, SloCostObservationConfig, SloLiveStructuredCalibration,
};

fn automatic_policy(reuse: Reuse) -> SloConfig {
    let mut policy = policy(SloMode::Observe);
    policy.cost_observation = SloCostObservationConfig::structured_whole_wave_v2();
    policy.cost_observation.live_structured_calibration =
        SloLiveStructuredCalibration::AutomaticV1 {
            settings: SloAutomaticCalibrationSettingsV1 {
                reuse,
                ..Default::default()
            },
        };
    policy
}

#[tokio::test]
async fn automatic_reuse_run_and_serve_resolve_explicit_cache_without_opening_storage() {
    let directory = tempfile::tempdir().unwrap();
    let base = directory.path().canonicalize().unwrap();
    let cache = base.join("reuse/cache");
    for path in [PathBuf::from("reuse/cache"), cache.clone()] {
        let reuse = Reuse::SameBootCleanShutdownV1 {
            location: Location::Directory { path },
            limits: SloAutomaticCalibrationReuseLimitsV1::default(),
        };
        let policy = automatic_policy(reuse);
        for extension in ["json", "toml"] {
            let contents = match extension {
                "json" => serde_json::to_string(&policy).unwrap(),
                _ => toml::to_string(&policy).unwrap(),
            };
            let config_path = base.join(format!("policy.{extension}"));
            tokio::fs::write(&config_path, &contents).await.unwrap();
            for command in ["run", "serve"] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test",
                    "--slo-config",
                    config_path.to_str().unwrap(),
                ])
                .unwrap();
                let loaded = load(parsed.command.config_path(), None)
                    .await
                    .unwrap()
                    .unwrap();
                let SloLiveStructuredCalibration::AutomaticV1 { settings } =
                    &loaded.config.cost_observation.live_structured_calibration
                else {
                    panic!("automatic policy lost")
                };
                let Reuse::SameBootCleanShutdownV1 { location, limits } = &settings.reuse else {
                    panic!("reuse policy lost")
                };
                assert_eq!(location.resolve_directory().unwrap(), cache);
                assert_eq!(limits, &SloAutomaticCalibrationReuseLimitsV1::default());
                assert_eq!(
                    settings.diagnostics,
                    SloAutomaticCalibrationDiagnosticsV1::MemoryOnly
                );
                assert!(loaded.config.cost_profile.is_none());
                let mut snapshot = RuntimeConfigSnapshot::default();
                loaded.apply_to_snapshot(&mut snapshot);
                let mut engine = EngineConfig::default();
                engine.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(engine.scheduler.slo, loaded.config);
                assert_eq!(
                    crate::runtime_env::runtime_snapshot_value(
                        &snapshot,
                        SLO_CONFIG_DIGEST_RUNTIME_KEY
                    ),
                    Some(format!("sha256:{:x}", Sha256::digest(contents.as_bytes())).as_str())
                );
                assert!(
                    !base.join("reuse").exists(),
                    "configuration opened cache storage"
                );
            }
        }
    }
}

#[tokio::test]
async fn automatic_reuse_run_and_serve_preserve_omitted_and_disabled_policy() {
    let directory = tempfile::tempdir().unwrap();
    for explicit in [false, true] {
        let mut wire = serde_json::to_value(automatic_policy(Reuse::Disabled {})).unwrap();
        if !explicit {
            wire["cost_observation"]["live_structured_calibration"]["settings"]
                .as_object_mut()
                .unwrap()
                .remove("reuse");
        }
        for extension in ["json", "toml"] {
            let contents = if extension == "json" {
                serde_json::to_string(&wire).unwrap()
            } else {
                let parsed: SloConfig = serde_json::from_value(wire.clone()).unwrap();
                let mut toml_wire: toml::Value =
                    toml::from_str(&toml::to_string(&parsed).unwrap()).unwrap();
                if !explicit {
                    toml_wire["cost_observation"]["live_structured_calibration"]["settings"]
                        .as_table_mut()
                        .unwrap()
                        .remove("reuse");
                }
                toml::to_string(&toml_wire).unwrap()
            };
            let config_path = directory.path().join(format!("policy.{extension}"));
            tokio::fs::write(&config_path, &contents).await.unwrap();
            for command in ["run", "serve"] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test",
                    "--slo-config",
                    config_path.to_str().unwrap(),
                ])
                .unwrap();
                let loaded = load(parsed.command.config_path(), None)
                    .await
                    .unwrap()
                    .unwrap();
                let SloLiveStructuredCalibration::AutomaticV1 { settings } =
                    loaded.config.cost_observation.live_structured_calibration
                else {
                    panic!("automatic policy lost")
                };
                assert_eq!(
                    settings.reuse,
                    if explicit {
                        Reuse::Disabled {}
                    } else {
                        Reuse::default()
                    }
                );
                assert_eq!(
                    settings.diagnostics,
                    SloAutomaticCalibrationDiagnosticsV1::MemoryOnly
                );
                assert!(loaded.config.cost_profile.is_none());
            }
        }
    }
}
