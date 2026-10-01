use super::*;
use ferrum_types::{
    SloAutomaticCalibrationNumericalStrategyV1, SloAutomaticCalibrationPopulationScheduleV1,
    SloCostPredictor, SloLiveStructuredCalibration, SloOutputTransport,
    SloStructuredActualCapturePolicy,
};

// These are user-authored declarations, not serialization of an already
// resolved SloConfig: omission is the behavior exercised by this shared loader.
fn minimal_wire(extension: &str, extra: &str) -> String {
    match extension {
        "json" => format!(
            r#"{{
            "mode":"enforce", "default_service_class":"interactive",
            "services":[{{"id":"interactive",
                "server_token_commit":{{"ttft_ms":500,"tpot_ms":40,"itl_ms":80}}}}]
            {extra}
        }}"#
        ),
        "toml" => format!(
            r#"
mode = "enforce"
default_service_class = "interactive"
[[services]]
id = "interactive"
[services.server_token_commit]
ttft_ms = 500
tpot_ms = 40
itl_ms = 80
{extra}
"#
        ),
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn product_default_enforce_json_and_toml_reach_run_and_serve_runtime_snapshots() {
    let directory = tempfile::tempdir().unwrap();
    for extension in ["json", "toml"] {
        let path = directory.path().join(format!("policy.{extension}"));
        tokio::fs::write(&path, minimal_wire(extension, ""))
            .await
            .unwrap();
        for command in ["run", "serve"] {
            let parsed = TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--slo-config",
                path.to_str().unwrap(),
            ])
            .unwrap();
            let loaded = load(parsed.command.config_path(), None)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(loaded.config.output.transport, SloOutputTransport::Credited);
            assert_eq!(
                loaded.config.cost_observation.predictor,
                SloCostPredictor::StructuredWholeWaveV2
            );
            assert_eq!(
                loaded.config.cost_observation.structured_actual_capture,
                SloStructuredActualCapturePolicy::ConsumerDrivenV1
            );
            let SloLiveStructuredCalibration::AutomaticV1 { settings } =
                &loaded.config.cost_observation.live_structured_calibration
            else {
                panic!("ordinary Enforce did not select automatic calibration");
            };
            assert_eq!(
                settings.population_schedule,
                SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2
            );
            assert_eq!(
                settings.numerical_strategy,
                SloAutomaticCalibrationNumericalStrategyV1::IdentifiedFitGlobalResidualV1
            );
            assert!(loaded.config.cost_profile.is_none());
            assert!(loaded.config.prefill_reference.is_none());
            let mut snapshot = RuntimeConfigSnapshot::default();
            loaded.apply_to_snapshot(&mut snapshot);
            let mut engine = EngineConfig::default();
            engine.apply_runtime_config_snapshot(&snapshot).unwrap();
            assert_eq!(engine.scheduler.slo, loaded.config);
            assert!(engine.slo_cost_profile_receipt.is_none());
            // TOML uses typed Option serialization; effective automatic and
            // explicit inactive settings both need a stable file roundtrip.
            let effective = toml::to_string(&loaded.config).unwrap();
            assert_eq!(
                toml::from_str::<SloConfig>(&effective).unwrap(),
                loaded.config
            );
        }
    }
}

#[tokio::test]
async fn product_default_partial_limits_and_explicit_disabled_survive_both_file_formats() {
    let directory = tempfile::tempdir().unwrap();
    for extension in ["json", "toml"] {
        for disabled in [false, true] {
            let extra = match (extension, disabled) {
                ("json", false) => {
                    r#",
                    "cost_observation":{"max_queued_samples":17,"max_samples_per_update":11},
                    "output":{"max_queued_events_per_request":19}"#
                }
                ("json", true) => {
                    r#",
                    "cost_observation":{"max_queued_samples":17,"max_samples_per_update":11,
                        "live_structured_calibration":{"kind":"disabled"}},
                    "output":{"transport":"legacy","max_queued_events_per_request":19}"#
                }
                ("toml", false) => {
                    r#"
[cost_observation]
max_queued_samples = 17
max_samples_per_update = 11
[output]
max_queued_events_per_request = 19
"#
                }
                ("toml", true) => {
                    r#"
[cost_observation]
max_queued_samples = 17
max_samples_per_update = 11
[cost_observation.live_structured_calibration]
kind = "disabled"
[output]
transport = "legacy"
max_queued_events_per_request = 19
"#
                }
                _ => unreachable!(),
            };
            let path = directory.path().join(format!("policy.{extension}"));
            tokio::fs::write(&path, minimal_wire(extension, extra))
                .await
                .unwrap();
            // Config-file owned selection shares the same typed boundary as
            // the command-line path exercised by the run/serve test above.
            let loaded = load(None, Some(&path)).await.unwrap().unwrap();
            assert_eq!(loaded.config.cost_observation.max_queued_samples.get(), 17);
            assert_eq!(
                loaded.config.cost_observation.max_samples_per_update.get(),
                11
            );
            assert_eq!(loaded.config.output.max_queued_events_per_request.get(), 19);
            assert_eq!(
                loaded
                    .config
                    .cost_observation
                    .live_structured_calibration
                    .is_disabled(),
                disabled
            );
            assert_eq!(
                loaded.config.output.transport,
                if disabled {
                    SloOutputTransport::Legacy
                } else {
                    SloOutputTransport::Credited
                }
            );
            let saved = if extension == "json" {
                serde_json::to_string(&loaded.config).unwrap()
            } else {
                toml::to_string(&loaded.config).unwrap()
            };
            tokio::fs::write(&path, saved).await.unwrap();
            let reloaded = load(Some(&path), None).await.unwrap().unwrap();
            assert_eq!(reloaded.config, loaded.config);
        }
    }
}

#[tokio::test]
async fn product_default_shared_loader_rejects_unknown_nested_options() {
    let directory = tempfile::tempdir().unwrap();
    for extension in ["json", "toml"] {
        let extra = if extension == "json" {
            r#","cost_observation":{"live_structured_calibration":{
                "kind":"automatic_v1","settings":{"maximum_owner":7}}}"#
        } else {
            r#"
[cost_observation.live_structured_calibration]
kind = "automatic_v1"
[cost_observation.live_structured_calibration.settings]
maximum_owner = 7
"#
        };
        let path = directory.path().join(format!("policy.{extension}"));
        tokio::fs::write(&path, minimal_wire(extension, extra))
            .await
            .unwrap();
        assert!(load(Some(&path), None).await.is_err());
    }
}
