use super::*;
use clap::{CommandFactory, Parser, Subcommand};
use ferrum_types::{EngineConfig, ServiceSloConfig, SloAttainmentTargets, SloLatencyBudgets};
use std::{num::NonZeroU64, path::PathBuf};

mod product_defaults;
mod reuse;

#[derive(Parser)]
struct TestCli {
    #[command(subcommand)]
    command: TestCommand,
}

#[derive(Subcommand)]
enum TestCommand {
    Run(crate::commands::run::RunCommand),
    Serve(crate::commands::serve::ServeCommand),
}

impl TestCommand {
    fn config_path(&self) -> Option<&Path> {
        match self {
            Self::Run(command) => command.slo_config.as_deref(),
            Self::Serve(command) => command.slo_config.as_deref(),
        }
    }

    async fn execute(self, config: crate::config::CliConfig) -> Result<()> {
        match self {
            Self::Run(command) => crate::commands::run::execute(command, config).await,
            Self::Serve(command) => crate::commands::serve::execute(command, config).await,
        }
    }
}

fn policy(mode: SloMode) -> SloConfig {
    SloConfig {
        mode,
        default_service_class: Some("interactive".to_owned()),
        services: vec![ServiceSloConfig {
            id: "interactive".to_owned(),
            server_token_commit: SloLatencyBudgets {
                ttft_ms: NonZeroU64::new(500).unwrap(),
                tpot_ms: NonZeroU64::new(40).unwrap(),
                itl_ms: NonZeroU64::new(80).unwrap(),
            },
            attainment: SloAttainmentTargets::default(),
            client_visible: None,
        }],
        cost_profile: (mode == SloMode::Enforce).then(|| PathBuf::from("profiles/cost.json")),
        ..Default::default()
    }
}

#[tokio::test]
async fn stage_ablation_run_and_serve_preserve_the_declared_policy_and_digest() {
    use ferrum_types::{SloExperimentStageV1 as Stage, SloLiveStructuredCalibration};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("policy.json");
    for stage in [
        Stage::ControlledAdaptiveBaseline,
        Stage::SingleWave,
        Stage::OutputIsolation,
        Stage::DeadlineOnly,
        Stage::CostCandidates,
        Stage::Complete,
    ] {
        let mut policy = policy(SloMode::Enforce);
        policy.experiment_stage = Some(stage);
        policy.output.transport = stage.output_transport();
        policy.cost_profile = None;
        if policy.execution_policy().cost_observation {
            policy.cost_observation =
                ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
            policy.cost_observation.live_structured_calibration =
                SloLiveStructuredCalibration::AutomaticV1 {
                    settings: Default::default(),
                };
        }
        let contents = serde_json::to_vec(&policy).unwrap();
        tokio::fs::write(&path, &contents).await.unwrap();
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
            let mut snapshot = RuntimeConfigSnapshot::default();
            loaded.apply_to_snapshot(&mut snapshot);
            let mut engine = EngineConfig::default();
            engine.apply_runtime_config_snapshot(&snapshot).unwrap();
            assert_eq!(engine.scheduler.slo, policy);
            assert!(engine.scheduler.slo.cost_profile.is_none());
            assert!(engine.scheduler.slo.prefill_reference.is_none());
            assert!(engine.slo_cost_profile_receipt.is_none());
            assert_eq!(
                crate::runtime_env::runtime_snapshot_value(
                    &snapshot,
                    SLO_CONFIG_DIGEST_RUNTIME_KEY
                ),
                Some(format!("sha256:{:x}", Sha256::digest(&contents)).as_str())
            );
        }
    }
}

#[tokio::test]
async fn reference_path_is_resolved_relative_to_the_policy_for_both_entrypoints() {
    let directory = tempfile::tempdir().unwrap();
    let mut policy = policy(SloMode::Observe);
    policy.prefill_reference = Some(ferrum_types::SloPrefillReferenceConfig {
        artifact_path: "reference/calibration.json".into(),
        expected_protocol_sha256: [7; 32],
        limits: Default::default(),
    });
    let path = directory.path().join("policy.json");
    tokio::fs::write(&path, serde_json::to_vec(&policy).unwrap())
        .await
        .unwrap();
    for cli_owned in [true, false] {
        let loaded = if cli_owned {
            load(Some(&path), None).await
        } else {
            load(None, Some(&path)).await
        }
        .unwrap()
        .unwrap();
        assert_eq!(
            loaded.config.prefill_reference.unwrap().artifact_path,
            directory
                .path()
                .canonicalize()
                .unwrap()
                .join("reference/calibration.json")
        );
    }
}

#[tokio::test]
async fn live_calibration_paths_follow_the_shared_policy_without_opening_artifacts() {
    use ferrum_types::{SloCostObservationConfig, SloLiveStructuredCalibration};
    use std::num::NonZeroUsize;

    let directory = tempfile::tempdir().unwrap();
    let base = directory.path().canonicalize().unwrap();
    let declaration = base.join("calibration/declaration.json");
    let evidence = base.join("calibration/evidence");
    for extension in ["json", "toml"] {
        for absolute_declaration in [false, true] {
            let mut policy = policy(SloMode::Observe);
            policy.cost_observation = SloCostObservationConfig::structured_whole_wave_v2();
            policy.cost_observation.live_structured_calibration =
                SloLiveStructuredCalibration::ServiceWindowsV1 {
                    declaration: if absolute_declaration {
                        declaration.clone()
                    } else {
                        "calibration/declaration.json".into()
                    },
                    evidence_directory: if absolute_declaration {
                        "calibration/evidence".into()
                    } else {
                        evidence.clone()
                    },
                    maximum_generations: NonZeroUsize::new(4).unwrap(),
                    maximum_source_bytes: NonZeroU64::new(1 << 28).unwrap(),
                    maximum_retained_numeric_bytes: NonZeroUsize::new(1 << 26).unwrap(),
                };
            let contents = if extension == "json" {
                serde_json::to_string(&policy).unwrap()
            } else {
                toml::to_string(&policy).unwrap()
            };
            let path = base.join(format!("policy.{extension}"));
            tokio::fs::write(&path, &contents).await.unwrap();
            for command in ["run", "serve"] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test",
                    "--slo-config",
                    path.to_str().unwrap(),
                ])
                .unwrap();
                let loaded = load(parsed.command.config_path(), None)
                    .await
                    .unwrap()
                    .unwrap();
                let SloLiveStructuredCalibration::ServiceWindowsV1 {
                    declaration: resolved_declaration,
                    evidence_directory: resolved_evidence,
                    ..
                } = &loaded.config.cost_observation.live_structured_calibration
                else {
                    panic!("live calibration policy lost")
                };
                assert_eq!(resolved_declaration, &declaration);
                assert_eq!(resolved_evidence, &evidence);
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
            }
        }
    }
    assert!(
        !base.join("calibration").exists(),
        "loading policy must not open declaration or create evidence output"
    );
}

#[tokio::test]
async fn automatic_calibration_partial_settings_follow_shared_run_serve_policy_without_opening_storage(
) {
    use ferrum_types::{
        SloAutomaticCalibrationDiagnosticsV1, SloAutomaticCalibrationSettingsV1,
        SloCostObservationConfig, SloLiveStructuredCalibration,
    };
    use std::num::NonZeroUsize;

    let directory = tempfile::tempdir().unwrap();
    let base = directory.path().canonicalize().unwrap();
    let diagnostic_path = base.join("calibration/diagnostics");
    let persisted = |path| SloAutomaticCalibrationDiagnosticsV1::Directory {
        directory: path,
        maximum_source_bytes: NonZeroU64::new(1024).unwrap(),
        maximum_total_bytes: NonZeroU64::new(4096).unwrap(),
        maximum_retained_generations: NonZeroUsize::new(2).unwrap(),
    };
    for diagnostics in [
        SloAutomaticCalibrationDiagnosticsV1::MemoryOnly,
        persisted("calibration/diagnostics".into()),
        persisted(diagnostic_path.clone()),
    ] {
        let mut policy = policy(SloMode::Observe);
        policy.cost_observation = SloCostObservationConfig::structured_whole_wave_v2();
        policy.cost_observation.live_structured_calibration =
            SloLiveStructuredCalibration::AutomaticV1 {
                settings: SloAutomaticCalibrationSettingsV1 {
                    diagnostics: diagnostics.clone(),
                    ..Default::default()
                },
            };
        let expected = match diagnostics {
            SloAutomaticCalibrationDiagnosticsV1::MemoryOnly => {
                SloAutomaticCalibrationDiagnosticsV1::MemoryOnly
            }
            SloAutomaticCalibrationDiagnosticsV1::Directory { .. } => {
                persisted(diagnostic_path.clone())
            }
        };
        for extension in ["json", "toml"] {
            // A normal user adds diagnostics without restating the complete
            // automatic policy. Do not serialize route_population here: that
            // would conceal a field default overriding the automatic default.
            let contents = if extension == "json" {
                let mut wire = serde_json::to_value(&policy).unwrap();
                wire["cost_observation"]["live_structured_calibration"]["settings"]
                    .as_object_mut()
                    .unwrap()
                    .retain(|key, _| key == "diagnostics");
                serde_json::to_string(&wire).unwrap()
            } else {
                let mut wire: toml::Value =
                    toml::from_str(&toml::to_string(&policy).unwrap()).unwrap();
                wire.get_mut("cost_observation")
                    .unwrap()
                    .get_mut("live_structured_calibration")
                    .unwrap()
                    .get_mut("settings")
                    .unwrap()
                    .as_table_mut()
                    .unwrap()
                    .retain(|key, _| key == "diagnostics");
                toml::to_string(&wire).unwrap()
            };
            let path = base.join(format!("policy.{extension}"));
            tokio::fs::write(&path, &contents).await.unwrap();
            for command in ["run", "serve"] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test",
                    "--slo-config",
                    path.to_str().unwrap(),
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
                assert_eq!(settings.diagnostics, expected);
                assert_eq!(
                    settings.route_population,
                    ferrum_types::SloCalibrationRoutePopulationV1::WarmOrGraphDisabledWithNoSubmissionV2
                );
                assert!(loaded.config.cost_profile.is_none());
                assert!(loaded.config.prefill_reference.is_none());
                let mut snapshot = RuntimeConfigSnapshot::default();
                loaded.apply_to_snapshot(&mut snapshot);
                let resolved = crate::startup::resolve_config(
                    snapshot,
                    ferrum_types::ModelCapabilities::unknown(),
                    ferrum_types::HardwareCapabilities::unknown(),
                    ferrum_types::WorkloadProfile::serving_default(),
                    ferrum_types::ExecutionResourceAuthority::PlanRuntime,
                    match command {
                        "run" => crate::startup::StartupUsage::SingleRequest,
                        "serve" => crate::startup::StartupUsage::PersistentServing,
                        _ => unreachable!(),
                    },
                )
                .unwrap();
                let snapshot = resolved.runtime_config;
                let mut engine = EngineConfig::default();
                engine.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(engine.scheduler.slo, loaded.config);
                let prefix_enabled = command == "serve";
                assert_eq!(engine.runtime.prefix_state_cache_enabled, prefix_enabled);
                let prefix = snapshot
                    .entries
                    .iter()
                    .find(|entry| entry.key == "FERRUM_PREFIX_CACHE")
                    .unwrap();
                assert_eq!(
                    prefix.effective_value,
                    if prefix_enabled { "1" } else { "0" }
                );
                assert_eq!(prefix.source, RuntimeConfigSource::Default);
                assert!(engine.slo_cost_profile_receipt.is_none());
                assert_eq!(
                    crate::runtime_env::runtime_snapshot_value(
                        &snapshot,
                        SLO_CONFIG_DIGEST_RUNTIME_KEY
                    ),
                    Some(format!("sha256:{:x}", Sha256::digest(contents.as_bytes())).as_str())
                );
                assert!(
                    !base.join("calibration").exists(),
                    "loading automatic policy must not create or open diagnostic storage"
                );
            }
        }
    }
}

#[tokio::test]
async fn export_paths_are_owned_by_the_shared_run_serve_policy_and_original_hash_is_preserved() {
    let directory = tempfile::tempdir().unwrap();
    let mut policy = policy(SloMode::Observe);
    policy.cost_observation.profile_export = Some(ferrum_types::SloCostProfileExportConfig {
        path: PathBuf::from("capture/cost.json"),
        observations_path: PathBuf::from("capture/raw.jsonl"),
        declared_clock_max_error_ns: Some(1000),
        ..Default::default()
    });
    for extension in ["json", "toml"] {
        let path = directory.path().join(format!("policy.{extension}"));
        let text = if extension == "json" {
            serde_json::to_string(&policy).unwrap()
        } else {
            toml::to_string(&policy).unwrap()
        };
        tokio::fs::write(&path, &text).await.unwrap();
        let loaded = load(Some(&path), None).await.unwrap().unwrap();
        let export = loaded
            .config
            .cost_observation
            .profile_export
            .as_ref()
            .unwrap();
        let base = directory.path().canonicalize().unwrap();
        assert_eq!(export.path, base.join("capture/cost.json"));
        assert_eq!(export.observations_path, base.join("capture/raw.jsonl"));
        let mut snapshot = RuntimeConfigSnapshot::default();
        loaded.apply_to_snapshot(&mut snapshot);
        assert_eq!(
            crate::runtime_env::runtime_snapshot_value(&snapshot, SLO_CONFIG_DIGEST_RUNTIME_KEY),
            Some(format!("sha256:{:x}", Sha256::digest(text.as_bytes())).as_str())
        );
        let mut engine = EngineConfig::default();
        engine.apply_runtime_config_snapshot(&snapshot).unwrap();
        assert_eq!(
            engine
                .scheduler
                .slo
                .cost_observation
                .profile_export
                .as_ref(),
            Some(export)
        );
        assert!(
            !base.join("capture").exists(),
            "loading policy must not create output files"
        );
    }
}

#[test]
fn slo_config_flag_is_optional_and_documented_for_both_entrypoints() {
    let mut cli = TestCli::command();
    for command in ["run", "serve"] {
        let default = TestCli::try_parse_from(["ferrum", command, "test-model"]).unwrap();
        assert!(default.command.config_path().is_none());
        let parsed = TestCli::try_parse_from([
            "ferrum",
            command,
            "test-model",
            "--slo-config",
            "interactive.toml",
        ])
        .unwrap();
        assert_eq!(
            parsed.command.config_path(),
            Some(Path::new("interactive.toml"))
        );
        let help = cli
            .find_subcommand_mut(command)
            .unwrap()
            .render_long_help()
            .to_string();
        assert!(help.contains("--slo-config <PATH>"));
        assert!(help.contains("runtime.slo_config"));
        assert!(help.contains("No environment override"));
        let help = help.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(help.contains("automatic calibration or imported profiles"));
        assert!(help.contains("SLO coverage remains unknown until qualified"));
    }
}

#[tokio::test]
async fn absent_slo_file_leaves_existing_configuration_off() {
    assert!(load(None, None).await.unwrap().is_none());
    let config: crate::config::CliConfig = toml::from_str("").unwrap();
    assert!(config.runtime.slo_config.is_none());
}

#[tokio::test]
async fn slo_loader_preserves_content_source_and_effective_config_for_json_and_toml() {
    let directory = tempfile::tempdir().unwrap();
    let mut policy = policy(SloMode::Enforce);
    policy.admission.time_policy = ferrum_types::SloTimeAdmissionPolicy::RequireSlo;
    for extension in ["json", "toml"] {
        let path = directory.path().join(format!("policy.{extension}"));
        let contents = if extension == "json" {
            serde_json::to_string_pretty(&policy).unwrap()
        } else {
            toml::to_string(&policy).unwrap()
        };
        tokio::fs::write(&path, &contents).await.unwrap();
        let loaded = load(None, Some(&path)).await.unwrap().unwrap();
        let expected_profile = directory
            .path()
            .canonicalize()
            .unwrap()
            .join("profiles/cost.json");
        assert_eq!(
            loaded.config.cost_profile.as_deref(),
            Some(expected_profile.as_path())
        );
        assert!(loaded
            .runtime_entries()
            .iter()
            .all(|entry| entry.source == RuntimeConfigSource::ConfigFile));
        let snapshot =
            RuntimeConfigSnapshot::from_entries(loaded.runtime_entries().iter().cloned());
        let digest =
            crate::runtime_env::runtime_snapshot_value(&snapshot, SLO_CONFIG_DIGEST_RUNTIME_KEY)
                .unwrap();
        assert_eq!(
            digest,
            format!("sha256:{:x}", Sha256::digest(contents.as_bytes()))
        );
        assert_eq!(
            crate::runtime_env::runtime_snapshot_value(&snapshot, SLO_CONFIG_PATH_RUNTIME_KEY),
            path.canonicalize().unwrap().to_str(),
        );
        let mut engine = EngineConfig::default();
        engine.apply_runtime_config_snapshot(&snapshot).unwrap();
        assert_eq!(engine.scheduler.slo, loaded.config);
        assert!(engine.scheduler.slo.services[0].client_visible.is_none());
        assert!(crate::runtime_env::materialize_runtime_env_effective(&snapshot).is_empty());
        assert!(
            crate::runtime_env::materialize_runtime_env_defaults(loaded.runtime_entries())
                .is_empty()
        );
    }
}

#[tokio::test]
async fn explicit_slo_cli_file_overrides_config_file_without_opening_the_latter() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cli.json");
    tokio::fs::write(
        &path,
        serde_json::to_vec(&policy(SloMode::Observe)).unwrap(),
    )
    .await
    .unwrap();
    let loaded = load(Some(&path), Some(&directory.path().join("absent.toml")))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded.config.mode, SloMode::Observe);
    assert!(loaded
        .runtime_entries()
        .iter()
        .all(|entry| entry.source == RuntimeConfigSource::Cli));
    let mut snapshot = RuntimeConfigSnapshot::from_env_vars([
        ("FERRUM_SLO_CONFIG", "ignored.toml"),
        (SLO_CONFIG_RUNTIME_KEY, "{\"mode\":\"off\"}"),
    ]);
    loaded.apply_to_snapshot(&mut snapshot);
    let mut engine = EngineConfig::default();
    engine.apply_runtime_config_snapshot(&snapshot).unwrap();
    assert_eq!(engine.scheduler.slo.mode, SloMode::Observe);
}

#[tokio::test]
async fn slo_content_changes_are_visible_even_when_the_path_is_unchanged() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("policy.json");
    let mut policy = policy(SloMode::Observe);
    tokio::fs::write(&path, serde_json::to_vec(&policy).unwrap())
        .await
        .unwrap();
    let first = load(Some(&path), None).await.unwrap().unwrap();
    policy.services[0].server_token_commit.itl_ms = NonZeroU64::new(90).unwrap();
    tokio::fs::write(&path, serde_json::to_vec(&policy).unwrap())
        .await
        .unwrap();
    let second = load(Some(&path), None).await.unwrap().unwrap();
    let first = RuntimeConfigSnapshot::from_entries(first.entries);
    let second = RuntimeConfigSnapshot::from_entries(second.entries);
    for key in [SLO_CONFIG_RUNTIME_KEY, SLO_CONFIG_DIGEST_RUNTIME_KEY] {
        assert_ne!(
            crate::runtime_env::runtime_snapshot_value(&first, key),
            crate::runtime_env::runtime_snapshot_value(&second, key),
        );
    }
    assert_eq!(
        crate::runtime_env::runtime_snapshot_value(&first, SLO_CONFIG_PATH_RUNTIME_KEY),
        crate::runtime_env::runtime_snapshot_value(&second, SLO_CONFIG_PATH_RUNTIME_KEY),
    );
}

#[tokio::test]
async fn malformed_or_incomplete_slo_file_is_not_silently_disabled() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("policy.toml");
    for invalid in [
        "mode = 'observ'",
        "mode = 'observe'",
        "unknown_mode = 'observe'",
        "mode =",
    ] {
        tokio::fs::write(&path, invalid).await.unwrap();
        assert!(load(Some(&path), None).await.is_err());
    }
    assert!(load(Some(Path::new("")), None).await.is_err());
    assert!(load(Some(&directory.path().join("missing.toml")), None)
        .await
        .is_err());
}

#[tokio::test]
async fn enabled_slo_policy_rejects_synthetic_run_and_serve_before_model_resolution() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("policy.json");
    tokio::fs::write(
        &path,
        serde_json::to_vec(&policy(SloMode::Observe)).unwrap(),
    )
    .await
    .unwrap();
    for command in ["run", "serve"] {
        let parsed = TestCli::try_parse_from([
            "ferrum",
            command,
            "synthetic/no-weight",
            "--profile-detail",
            "basic",
        ])
        .unwrap();
        let mut config = crate::config::CliConfig::default();
        config.runtime.slo_config = Some(path.clone());
        let error = parsed.command.execute(config).await.unwrap_err();
        assert!(error.to_string().contains("real language-model execution"));
    }
}

#[test]
fn slo_enforce_requires_native_authority_while_observe_keeps_legacy_coverage() {
    for mode in [SloMode::Off, SloMode::Observe, SloMode::Enforce] {
        let loaded = LoadedSloConfig {
            config: policy(mode),
            entries: Vec::new(),
        };
        loaded
            .validate_authority(ExecutionResourceAuthority::PlanRuntime)
            .unwrap();
        assert_eq!(
            loaded
                .validate_authority(ExecutionResourceAuthority::LegacyEngine)
                .is_ok(),
            mode != SloMode::Enforce
        );
        assert_eq!(
            loaded.validate_real_execution(true).is_ok(),
            mode == SloMode::Off
        );
    }
}

#[tokio::test]
async fn required_query_observation_uses_shared_run_serve_config_and_relative_output() {
    let directory = tempfile::tempdir().unwrap();
    let mut policy = policy(SloMode::Observe);
    policy.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    policy.required_query_observation =
        ferrum_types::SloRequiredQueryObservationConfig::StructuredRequiredV1 {
            path: "queries.jsonl".into(),
            limits: Default::default(),
        };
    let path = directory.path().join("policy.json");
    tokio::fs::write(&path, serde_json::to_vec(&policy).unwrap())
        .await
        .unwrap();
    for command in ["run", "serve"] {
        let parsed = TestCli::try_parse_from([
            "ferrum",
            command,
            "test",
            "--slo-config",
            path.to_str().unwrap(),
        ])
        .unwrap();
        let loaded = load(parsed.command.config_path(), None)
            .await
            .unwrap()
            .unwrap();
        let ferrum_types::SloRequiredQueryObservationConfig::StructuredRequiredV1 {
            path: observed,
            ..
        } = loaded.config.required_query_observation
        else {
            panic!("enabled observation lost")
        };
        assert_eq!(
            observed,
            directory
                .path()
                .canonicalize()
                .unwrap()
                .join("queries.jsonl")
        );
        assert!(
            !observed.exists(),
            "loading a policy does not open the writer"
        );
    }
    policy.mode = SloMode::Off;
    assert!(policy
        .validate()
        .unwrap_err()
        .contains("Observe or Enforce"));
    policy.mode = SloMode::Observe;
    policy.cost_observation = Default::default();
    assert!(policy.validate().unwrap_err().contains("Structured V2"));
}

#[tokio::test]
async fn uncalibrated_required_query_observation_uses_shared_run_serve_policy() {
    let directory = tempfile::tempdir().unwrap();
    let mut policy = policy(SloMode::Observe);
    policy.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    policy.prefill_reference = Some(ferrum_types::SloPrefillReferenceConfig {
        artifact_path: "reference.json".into(),
        expected_protocol_sha256: [7; 32],
        limits: Default::default(),
    });
    policy.required_query_observation =
        ferrum_types::SloRequiredQueryObservationConfig::StructuredUncalibratedV1 {
            path: "root-queries.jsonl".into(),
            limits: Default::default(),
        };
    let path = directory.path().join("policy.toml");
    tokio::fs::write(&path, toml::to_string(&policy).unwrap())
        .await
        .unwrap();
    for command in ["run", "serve"] {
        let parsed = TestCli::try_parse_from([
            "ferrum",
            command,
            "test",
            "--slo-config",
            path.to_str().unwrap(),
        ])
        .unwrap();
        let loaded = load(parsed.command.config_path(), None)
            .await
            .unwrap()
            .unwrap();
        let ferrum_types::SloRequiredQueryObservationConfig::StructuredUncalibratedV1 {
            path: observed,
            ..
        } = &loaded.config.required_query_observation
        else {
            panic!("uncalibrated diagnostic mode lost")
        };
        assert_eq!(
            observed,
            &directory
                .path()
                .canonicalize()
                .unwrap()
                .join("root-queries.jsonl")
        );
        assert!(!observed.exists());
        assert_eq!(loaded.config.mode, SloMode::Observe);
        assert!(loaded.config.cost_profile.is_none());
        assert_eq!(
            loaded.config.prefill_reference.unwrap().artifact_path,
            directory
                .path()
                .canonicalize()
                .unwrap()
                .join("reference.json")
        );
    }
}

#[tokio::test]
async fn global_residual_strategy_run_serve_config_reaches_effective_engine_without_manual_profiles(
) {
    use ferrum_types::{
        SloAutomaticCalibrationNumericalStrategyV1 as Strategy, SloAutomaticCalibrationSettingsV1,
        SloCostObservationConfig, SloLiveStructuredCalibration,
    };
    let directory = tempfile::tempdir().unwrap();
    let mut policy = policy(SloMode::Enforce);
    policy.cost_profile = None;
    policy.cost_observation = SloCostObservationConfig::structured_whole_wave_v2();
    policy.cost_observation.live_structured_calibration =
        SloLiveStructuredCalibration::AutomaticV1 {
            settings: SloAutomaticCalibrationSettingsV1 {
                numerical_strategy: Strategy::IdentifiedFitGlobalResidualV1,
                ..Default::default()
            },
        };
    for extension in ["json", "toml"] {
        let contents = if extension == "json" {
            serde_json::to_string(&policy).unwrap()
        } else {
            toml::to_string(&policy).unwrap()
        };
        let path = directory.path().join(format!("policy.{extension}"));
        tokio::fs::write(&path, &contents).await.unwrap();
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
            let mut snapshot = RuntimeConfigSnapshot::default();
            loaded.apply_to_snapshot(&mut snapshot);
            let wire =
                crate::runtime_env::runtime_snapshot_value(&snapshot, SLO_CONFIG_RUNTIME_KEY)
                    .unwrap();
            let effective: SloConfig = serde_json::from_str(wire).unwrap();
            assert_eq!(effective, policy);
            let mut engine = EngineConfig::default();
            engine.apply_runtime_config_snapshot(&snapshot).unwrap();
            assert_eq!(engine.scheduler.slo, effective);
            assert!(engine.scheduler.slo.cost_profile.is_none());
            assert!(engine.scheduler.slo.prefill_reference.is_none());
            assert!(engine.slo_cost_profile_receipt.is_none());
        }
    }
}
