use super::*;
use clap::Parser;

#[derive(Parser)]
struct RunCli {
    #[command(flatten)]
    command: crate::commands::run::RunCommand,
}

#[derive(Parser)]
struct ServeCli {
    #[command(flatten)]
    command: crate::commands::serve::ServeCommand,
}

async fn synthetic_entrypoint(
    run: bool,
    path: &Path,
    config: CliConfig,
) -> ferrum_execution::ExecutorBuildResult<()> {
    let args = [
        "ferrum",
        "synthetic/no-weight",
        "--profile-detail",
        "basic",
        "--profile-jsonl",
        path.to_str().unwrap(),
    ];
    if run {
        crate::commands::run::execute(RunCli::try_parse_from(args).unwrap().command, config).await
    } else {
        crate::commands::serve::execute(ServeCli::try_parse_from(args).unwrap().command, config)
            .await
    }
}

#[tokio::test]
async fn profile_journal_config_reaches_run_serve_artifacts_and_engine_settings() {
    let directory = tempfile::tempdir().unwrap();
    let config_path = directory.path().join("config.toml");
    fs::write(&config_path, "[profile_journal]\nqueue_capacity = 2\nmax_batch_records = 2048\nmax_record_bytes = 2097152\nmax_batch_bytes = 8388608\npayload_capacity_bytes = 16777216\n").await.unwrap();
    let config = CliConfig::load(&config_path).await.unwrap();
    assert_eq!(config.profile_journal.queue_capacity, 2);
    assert_eq!(config.profile_journal.max_batch_records, 2048);
    config.save(&config_path).await.unwrap();
    assert_eq!(
        CliConfig::load(&config_path).await.unwrap().profile_journal,
        config.profile_journal
    );
    for run in [true, false] {
        let mut engine = ferrum_types::EngineConfig::default();
        config.apply_profile_journal_config(&mut engine).unwrap();
        assert_eq!(engine.runtime.profile_journal, config.profile_journal);
        let output = directory
            .path()
            .join(if run { "run.jsonl" } else { "serve.jsonl" });
        // The standalone synthetic command owns its writer. A second live
        // owner is intentionally forbidden by the real ownership protocol.
        synthetic_entrypoint(run, &output, config.clone())
            .await
            .unwrap();
        let events: Vec<ferrum_types::FerrumProfileEvent> = fs::read_to_string(&output)
            .await
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(!events.is_empty());
        for event in events {
            event.validate().unwrap();
        }
        // A real tiny budget must govern the product writer, rather than only
        // survive configuration parsing. Basic reports transport loss without
        // turning a synthetic inference-free command into an execution error.
        let limited_output = output.with_extension("limited.jsonl");
        let mut limited = config.clone();
        limited.profile_journal.max_record_bytes = 1;
        synthetic_entrypoint(run, &limited_output, limited)
            .await
            .unwrap();
        assert!(
            !limited_output.exists(),
            "bounded tool preflight rejects before opening a writer"
        );
    }
}

#[tokio::test]
async fn profile_journal_invalid_config_fails_before_run_or_serve_artifacts() {
    let directory = tempfile::tempdir().unwrap();
    let config_path = directory.path().join("invalid.toml");
    for content in [
        "[profile_journal]\nqueue_capacity = 0\n",
        "[profile_journal]\nmax_batch_bytes = 1\n",
        "[profile_journal]\nunlimited = true\n",
    ] {
        fs::write(&config_path, content).await.unwrap();
        assert!(CliConfig::load(&config_path).await.is_err());
    }
    for run in [true, false] {
        let output = directory
            .path()
            .join(if run { "run.jsonl" } else { "serve.jsonl" });
        let mut config = CliConfig::default();
        config.profile_journal.queue_capacity = 0;
        let error = synthetic_entrypoint(run, &output, config)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("profile_journal.queue_capacity"));
        assert!(!output.exists());
    }
}
