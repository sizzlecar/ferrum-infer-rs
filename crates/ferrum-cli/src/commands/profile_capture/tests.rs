use super::*;
use clap::{CommandFactory, Parser, Subcommand};
use ferrum_types::{EngineConfig, RuntimeConfigEffect};

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
    fn frame_limit(&self) -> Option<NonZeroU32> {
        match self {
            Self::Run(command) => command.profile_max_frames_per_request,
            Self::Serve(command) => command.profile_max_frames_per_request,
        }
    }

    async fn execute(self, config: crate::config::CliConfig) -> Result<()> {
        match self {
            Self::Run(command) => crate::commands::run::execute(command, config).await,
            Self::Serve(command) => crate::commands::serve::execute(command, config).await,
        }
    }
}

#[test]
fn frame_limit_cli_accepts_only_positive_u32_for_run_and_serve() {
    for command in ["run", "serve"] {
        assert!(TestCli::try_parse_from(["ferrum", command, "test-model"])
            .unwrap()
            .command
            .frame_limit()
            .is_none());
        for value in ["1", "4294967295"] {
            let parsed = TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--profile-max-frames-per-request",
                value,
            ])
            .unwrap();
            assert_eq!(
                parsed.command.frame_limit().unwrap().get().to_string(),
                value
            );
        }
        for invalid in ["0", "-1", "4294967296", "unlimited"] {
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--profile-max-frames-per-request",
                invalid,
            ])
            .is_err());
        }
        assert!(TestCli::try_parse_from([
            "ferrum",
            command,
            "test-model",
            "--profile-max-frames-per-request",
            "2",
            "--observability-vertical-slice-out",
            "artifacts",
        ])
        .is_err());
    }
}

#[test]
fn frame_limit_help_explains_scope_and_configuration_for_both_entrypoints() {
    let mut cli = TestCli::command();
    for command in ["run", "serve"] {
        let help = cli
            .find_subcommand_mut(command)
            .unwrap()
            .render_long_help()
            .to_string();
        assert!(help.contains("--profile-max-frames-per-request <N>"));
        assert!(help.contains("Limits diagnostics only"));
        assert!(help.contains("runtime.profile_max_frames_per_request"));
        assert!(help.contains("No environment override"));
    }
}

#[test]
fn frame_limit_requires_an_all_frame_mode_and_execution_event_sink() {
    let environment = RuntimeConfigSnapshot::default();
    let limit = NonZeroU32::new(2);
    let sink = Some(Path::new("profile.jsonl"));
    for detail in [
        ObservabilityProfileDetail::Resource,
        ObservabilityProfileDetail::Kernel,
        ObservabilityProfileDetail::Replay,
        ObservabilityProfileDetail::Verify,
        ObservabilityProfileDetail::Full,
    ] {
        assert!(validate_requested(limit, detail, sink, None, &environment).is_ok());
        assert!(validate_requested(limit, detail, None, sink, &environment).is_ok());
        assert!(validate_requested(limit, detail, None, None, &environment).is_err());
    }
    for detail in [
        ObservabilityProfileDetail::Off,
        ObservabilityProfileDetail::Basic,
        ObservabilityProfileDetail::Latency,
        ObservabilityProfileDetail::Debug,
    ] {
        assert!(validate_requested(limit, detail, sink, None, &environment).is_err());
        assert!(validate_requested(None, detail, None, None, &environment).is_ok());
    }
}

#[test]
fn frame_limit_accepts_existing_sink_aliases_but_rejects_empty_effective_paths() {
    let limit = NonZeroU32::new(2);
    for key in ["FERRUM_PROFILE_JSONL", "FERRUM_SCHEDULER_TRACE_JSONL"] {
        let environment = RuntimeConfigSnapshot::from_env_vars([(key, "profile.jsonl")]);
        assert!(validate_requested(
            limit,
            ObservabilityProfileDetail::Full,
            None,
            None,
            &environment,
        )
        .is_ok());
        let environment = RuntimeConfigSnapshot::from_env_vars([(key, "")]);
        assert!(validate_requested(
            limit,
            ObservabilityProfileDetail::Full,
            None,
            None,
            &environment,
        )
        .is_err());
    }
    let environment =
        RuntimeConfigSnapshot::from_env_vars([("FERRUM_PROFILE_JSONL", "profile.jsonl")]);
    assert!(validate_requested(
        limit,
        ObservabilityProfileDetail::Full,
        Some(Path::new("")),
        None,
        &environment,
    )
    .is_err());
}

#[test]
fn frame_limit_requires_native_resource_authority_only_when_requested() {
    assert!(
        validate_authority(NonZeroU32::new(2), ExecutionResourceAuthority::PlanRuntime).is_ok()
    );
    assert!(
        validate_authority(NonZeroU32::new(2), ExecutionResourceAuthority::LegacyEngine).is_err()
    );
    assert!(validate_authority(None, ExecutionResourceAuthority::LegacyEngine).is_ok());
}

#[test]
fn frame_limit_config_and_cli_keep_precedence_provenance_and_no_env_override() {
    let config: crate::config::CliConfig =
        toml::from_str("[runtime]\nprofile_max_frames_per_request = 7\n").unwrap();
    for (requested, expected, source) in [
        (None, 7, RuntimeConfigSource::ConfigFile),
        (NonZeroU32::new(3), 3, RuntimeConfigSource::Cli),
    ] {
        let environment = RuntimeConfigSnapshot::from_env_vars([
            (PROFILE_MAX_FRAMES_PER_REQUEST_CONFIG_KEY, "99"),
            ("FERRUM_PROFILE_MAX_FRAMES_PER_REQUEST", "99"),
        ]);
        let mut cli = Vec::new();
        push_cli_entry(&mut cli, requested);
        let snapshot = crate::commands::serve::merge_runtime_config_sources(
            config.runtime.runtime_config_entries(),
            environment,
            cli,
        );
        let entry = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == PROFILE_MAX_FRAMES_PER_REQUEST_CONFIG_KEY)
            .unwrap();
        assert_eq!(entry.source, source);
        assert_eq!(entry.affects, [RuntimeConfigEffect::Diagnostics]);
        let mut engine = EngineConfig::default();
        engine.apply_runtime_config_snapshot(&snapshot).unwrap();
        assert_eq!(
            engine.runtime.profile_max_frames_per_request.unwrap().get(),
            expected
        );
        let serialized = serde_json::to_value(&snapshot).unwrap();
        let restored: RuntimeConfigSnapshot = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored, snapshot);
    }
    assert!(toml::from_str::<crate::config::CliConfig>(
        "[runtime]\nprofile_max_frames_per_request = 0\n"
    )
    .is_err());
}

#[test]
fn frame_limit_is_not_materialized_into_process_environment() {
    let before = std::env::var_os(PROFILE_MAX_FRAMES_PER_REQUEST_CONFIG_KEY);
    let mut entries = Vec::new();
    push_cli_entry(&mut entries, NonZeroU32::new(3));
    assert!(crate::runtime_env::materialize_runtime_env_defaults(&entries).is_empty());
    assert!(crate::runtime_env::materialize_runtime_env_effective(
        &RuntimeConfigSnapshot::from_entries(entries)
    )
    .is_empty());
    assert_eq!(
        std::env::var_os(PROFILE_MAX_FRAMES_PER_REQUEST_CONFIG_KEY),
        before
    );
}

#[tokio::test]
async fn configured_frame_limit_rejects_synthetic_execution_before_writing_artifacts() {
    for command in ["run", "serve"] {
        let directory = tempfile::tempdir().unwrap();
        let sink = directory.path().join("profile.jsonl");
        let parsed = TestCli::try_parse_from([
            "ferrum",
            command,
            "synthetic/no-weight",
            "--profile-detail",
            "full",
            "--profile-jsonl",
            sink.to_str().unwrap(),
        ])
        .unwrap();
        let mut config = crate::config::CliConfig::default();
        config.runtime.profile_max_frames_per_request = NonZeroU32::new(2);
        let error = parsed.command.execute(config).await.unwrap_err();
        assert!(error.to_string().contains("native plan runtime"));
        assert!(!sink.exists());
    }
}
