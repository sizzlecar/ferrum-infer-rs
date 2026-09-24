//! Exercise main -> commands::run::execute -> real CPU PlanRuntime -> credited
//! output and retained profile. No executor injection or helper-only shortcut.
#[path = "credited_run_entrypoint/model.rs"]
mod model;

use serde_json::Value;
use std::{fs, path::Path, process::Stdio, time::Duration};
use tokio::process::Command;

fn command(root: &Path, model: &Path, format: &str) -> Command {
    fs::write(root.join("ferrum.toml"), "").unwrap();
    let policy = root.join("credited.toml");
    fs::write(
        &policy,
        "mode = \"off\"\n[output]\ntransport = \"credited\"\n",
    )
    .unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_ferrum"));
    command
        .current_dir(root)
        .stdin(Stdio::null())
        .kill_on_drop(true);
    // Isolate the child from local tuning and credentials. All model files are
    // local; the dead endpoint prevents an accidental download fallback.
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("FERRUM_")
            || matches!(name.as_ref(), "HF_TOKEN" | "HUGGING_FACE_HUB_TOKEN")
        {
            command.env_remove(key);
        }
    }
    command
        .env("HF_HOME", root.join("cache"))
        .env("HF_ENDPOINT", "http://127.0.0.1:9")
        .env("NO_COLOR", "1")
        .args(["run"])
        .arg(model)
        .args([
            "--backend",
            "cpu",
            "--prompt",
            "hello",
            "--output-format",
            format,
            "--max-tokens",
            "3",
            "--temperature",
            "0",
            "--top-k",
            "0",
            "--top-p",
            "1",
            "--repeat-penalty",
            "1",
            "--disable-thinking",
            "--kv-capacity",
            "16",
            "--max-model-len",
            "16",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "8",
            "--disable-reusable-execution",
            "--slo-config",
        ])
        .arg(policy);
    command
}

async fn output(mut command: Command) -> std::process::Output {
    tokio::time::timeout(Duration::from_secs(45), command.output())
        .await
        .expect("real CLI timed out; child killed on drop")
        .expect("launch actual ferrum binary")
}

async fn completes_with_profile(detail: &str) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let model = root.join("model");
    model::write(&model);
    let profile = root.join("profile.jsonl");
    let mut command = command(root, &model, "text");
    command
        .args(["--profile-detail", detail, "--profile-jsonl"])
        .arg(&profile);
    let result = output(command).await;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout)
            .unwrap()
            .split_whitespace()
            .collect::<Vec<_>>(),
        ["hello", "hello", "hello"]
    );
    let records: Vec<Value> = fs::read_to_string(&profile)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    // This sink also receives the executor's vnext.* lifecycle/native records.
    // Its total line count depends on profile detail and execution, not on the
    // number of completed requests. Validate every record, then identify the
    // unique terminal product record by its declared semantic phase.
    for record in &records {
        serde_json::from_value::<ferrum_types::FerrumProfileEvent>(record.clone())
            .unwrap()
            .validate()
            .unwrap();
    }
    let terminal: Vec<_> = records
        .iter()
        .filter(|record| record["phase"] == "credited_generation")
        .collect();
    assert_eq!(
        terminal.len(),
        1,
        "exactly one terminal profile for this one-shot request"
    );
    let record = terminal[0];
    let request_id = record["request_id"].as_str().unwrap();
    assert!(!request_id.is_empty());
    let execution_id = record["attributes"]["execution_request_id"]
        .as_str()
        .unwrap();
    assert_eq!(execution_id, format!("request.product.{request_id}"));
    assert!(
        records.iter().any(|event| {
            event["phase"]
                .as_str()
                .is_some_and(|phase| phase.starts_with("vnext."))
                && event["attributes"]["execution_request_id"].as_str() == Some(execution_id)
        }),
        "terminal evidence must join the same actual executor request"
    );
    assert_eq!(record["status"], "ok");
    assert_eq!(record["entrypoint"], "run");
    assert_eq!(record["phase"], "credited_generation");
    assert_eq!(record["attributes"]["output_transport"], "credited");
    assert_eq!(record["attributes"]["profile_detail"], detail);
    assert_eq!(record["attributes"]["completion_token_count"], 3);
    assert_eq!(record["attributes"]["finish_reason"], "length");
    assert_eq!(record["attributes"]["engine_token_commit_count"], 3);
    assert_eq!(record["attributes"]["itl_source"], "engine_token_commit");
    assert_eq!(record["attributes"]["itl_interval_count"], 2);
    let commits = record["attributes"]["engine_token_commit_nanos_since_request_start"]
        .as_array()
        .unwrap();
    assert_eq!(commits.len(), 3);
    assert!(commits
        .windows(2)
        .all(|pair| pair[0].as_u64().unwrap() <= pair[1].as_u64().unwrap()));
    assert!(!record["attributes"]["engine_decode_stage_intervals"]
        .as_array()
        .unwrap()
        .is_empty());
    assert_eq!(record["attributes"]["engine_decode_stages_complete"], true);
    assert_eq!(
        record["attributes"]["engine_decode_stage_intervals_omitted"],
        0
    );
}

#[tokio::test]
async fn credited_run_entrypoint_latency_profile_completes_real_inference() {
    completes_with_profile("latency").await;
}

#[tokio::test]
async fn credited_run_entrypoint_kernel_profile_completes_real_inference() {
    completes_with_profile("kernel").await;
}

#[tokio::test]
async fn credited_run_entrypoint_keeps_unsupported_artifact_guards() {
    for arguments in [
        vec!["--request-dump-dir", "dump"],
        vec!["--memory-profile-jsonl", "memory.jsonl"],
        vec!["--scheduler-trace-jsonl", "scheduler.jsonl"],
        vec!["--profile-detail", "full", "--profile-jsonl", "full.jsonl"],
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut command = command(
            directory.path(),
            &directory.path().join("unresolved-model"),
            "text",
        );
        command.args(&arguments);
        let result = output(command).await;
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("require separate output contracts")
        );
        for artifact in ["dump", "memory.jsonl", "scheduler.jsonl", "full.jsonl"] {
            assert!(!directory.path().join(artifact).exists());
        }
    }
}

#[tokio::test]
async fn credited_run_entrypoint_profile_cannot_select_synthetic_execution() {
    let directory = tempfile::tempdir().unwrap();
    let mut command = command(directory.path(), Path::new("synthetic/no-weight"), "text");
    command.args([
        "--profile-detail",
        "latency",
        "--profile-jsonl",
        "profile.jsonl",
    ]);
    let result = output(command).await;
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("requires real inference"));
    assert!(!directory.path().join("profile.jsonl").exists());
}

#[tokio::test]
async fn credited_run_entrypoint_profile_keeps_text_only_contract() {
    let directory = tempfile::tempdir().unwrap();
    let mut command = command(
        directory.path(),
        &directory.path().join("unresolved-model"),
        "jsonl",
    );
    command.args([
        "--profile-detail",
        "latency",
        "--profile-jsonl",
        "profile.jsonl",
    ]);
    let result = output(command).await;
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr)
        .contains("requires --prompt and --output-format text"));
    assert!(!directory.path().join("profile.jsonl").exists());
}
