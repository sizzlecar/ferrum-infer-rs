use super::*;
use clap::Parser;
use ferrum_bench_core::release_regression::{
    model_tasks::{ModelCheck, ModelRunCapacity},
    ExecutionTarget, ModelProfile,
};
use ferrum_types::ModelOutputProtocol;

fn task(backend: Backend, id: &str) -> ExpectedModelRun {
    ExpectedModelRun {
        profile: ModelProfile {
            gguf: None,
            reasoning_protocol: ferrum_types::ModelReasoningProtocol::PromptOpened,
            id: id.into(),
            model: "org/model".into(),
            target: ExecutionTarget {
                architecture: "qwen3".into(),
                protocol: ModelOutputProtocol::Text,
                precision: "bf16".into(),
                backend,
                execution_path: "safetensors".into(),
            },
            available: true,
            estimate: None,
        },
        binary_sha256: "a".repeat(64),
        version: "0.8.8".into(),
        checks: vec![ModelCheck::Basic, ModelCheck::Tools],
        disable_thinking: true,
        use_default_backend: true,
        max_tokens: 512,
        runtime_capacity: None,
        reasoning_alias_replay: true,
        stop_prompt: "quotes ' $(literal)\nline".into(),
    }
}
fn args(directory: &Path) -> LocalArgs {
    LocalArgs {
        tasks: directory.join("tasks.json"),
        backend: LocalBackend::Metal,
        ferrum_bin: directory.join("ferrum"),
        runner: directory.join("model_regression"),
        runner_sha256: "b".repeat(64),
        report_dir: directory.join("reports"),
        task_timeout_secs: 20,
        request_timeout_secs: 10,
    }
}
#[test]
fn local_backend_cli_rejects_cuda() {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: LocalArgs,
    }
    for backend in ["cpu", "metal", "cuda"] {
        let result = Cli::try_parse_from([
            "local",
            "--tasks",
            "tasks.json",
            "--backend",
            backend,
            "--ferrum-bin",
            "ferrum",
            "--runner",
            "runner",
            "--runner-sha256",
            &"a".repeat(64),
            "--report-dir",
            "reports",
            "--task-timeout-secs",
            "20",
        ]);
        if backend == "cuda" {
            assert_eq!(
                result.err().unwrap().kind(),
                clap::error::ErrorKind::InvalidValue
            );
        } else {
            assert!(result.is_ok());
        }
    }
}
#[test]
fn local_selection_retains_other_backends_and_rejects_unsupported_tasks() {
    let metal = task(Backend::Metal, "metal");
    let cuda = task(Backend::Cuda, "cuda");
    let cpu = task(Backend::Cpu, "cpu");
    let (selected, other, _) = select(
        PreparedTasks {
            schema_version: 1,
            expectations: vec![cuda, cpu, metal.clone()],
            unsupported_obligations: vec![],
            remaining_plan_gaps: vec![],
        },
        LocalBackend::Metal,
    )
    .unwrap();
    assert_eq!(selected, vec![metal]);
    assert_eq!(other, vec!["cuda", "cpu"]);
    assert!(select(
        PreparedTasks {
            schema_version: 1,
            expectations: vec![],
            unsupported_obligations: vec![7],
            remaining_plan_gaps: vec![]
        },
        LocalBackend::Cpu
    )
    .is_err());
    assert!(select(
        PreparedTasks {
            schema_version: 1,
            expectations: vec![task(Backend::Cpu, "same"), task(Backend::Metal, "same")],
            unsupported_obligations: vec![],
            remaining_plan_gaps: vec![]
        },
        LocalBackend::Cpu
    )
    .is_err());
}
#[test]
fn local_runner_arguments_preserve_task_flags_without_a_shell() {
    let directory = tempfile::tempdir().unwrap();
    let input = args(directory.path());
    let mut expected = task(Backend::Metal, "metal");
    expected.profile.gguf = Some(ferrum_bench_core::release_regression::GgufSourceProfile {
        filename: "weights/model.gguf".into(),
        semantic_source: format!("author/model@{}", "a".repeat(40)),
        tokenizer_source: None,
    });
    let words = runner_arguments(
        &input,
        &expected,
        &directory.path().join("expected.json"),
        &directory.path().join("report"),
    );
    let after = |flag| {
        words
            .windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].to_str().unwrap())
    };
    assert_eq!(after("--backend"), Some("metal"));
    assert_eq!(after("--model"), Some(expected.profile.model.as_str()));
    assert_eq!(after("--gguf-file"), Some("weights/model.gguf"));
    assert_eq!(after("--stop-prompt"), Some(expected.stop_prompt.as_str()));
    assert_eq!(after("--checks"), Some("basic,tools"));
    assert_eq!(after("--max-tokens"), Some("512"));
    assert_eq!(after("--run-timeout-secs"), Some("20"));
    assert_eq!(after("--request-timeout-secs"), Some("10"));
    assert_eq!(after("--context-tokens"), None);
    assert_eq!(after("--max-num-seqs"), None);
    for flag in [
        "--disable-thinking",
        "--use-default-backend",
        "--reasoning-alias-replay",
    ] {
        assert!(words.iter().any(|word| word == flag));
    }
    let mut explicit = expected;
    explicit.disable_thinking = false;
    explicit.use_default_backend = false;
    explicit.reasoning_alias_replay = false;
    explicit.runtime_capacity = Some(ModelRunCapacity {
        context_tokens: 2048,
        max_num_seqs: 1,
    });
    let words = runner_arguments(&input, &explicit, Path::new("task"), Path::new("report"));
    for (flag, value) in [("--context-tokens", "2048"), ("--max-num-seqs", "1")] {
        assert!(words
            .windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value));
    }
    for flag in [
        "--disable-thinking",
        "--use-default-backend",
        "--reasoning-alias-replay",
    ] {
        assert!(!words.iter().any(|word| word == flag));
    }
}
#[test]
fn local_capacity_preflight_rejects_an_exhausted_context_or_empty_sequence_pool() {
    let mut expected = task(Backend::Metal, "metal");
    expected.runtime_capacity = Some(ModelRunCapacity {
        context_tokens: 2048,
        max_num_seqs: 1,
    });
    let document = |task| PreparedTasks {
        schema_version: 1,
        expectations: vec![task],
        unsupported_obligations: vec![],
        remaining_plan_gaps: vec![],
    };
    assert_eq!(
        select(document(expected.clone()), LocalBackend::Metal)
            .unwrap()
            .0,
        vec![expected.clone()]
    );
    for (context_tokens, max_num_seqs) in [(512, 1), (0, 1), (2048, 0)] {
        expected.runtime_capacity = Some(ModelRunCapacity {
            context_tokens,
            max_num_seqs,
        });
        assert!(select(document(expected.clone()), LocalBackend::Metal).is_err());
    }
}
fn inputs(directory: &Path, tasks: Vec<ExpectedModelRun>) -> LocalArgs {
    let mut input = args(directory);
    fs::write(&input.ferrum_bin, b"binary input fixture, never executed").unwrap();
    fs::write(&input.runner, b"runner input fixture, never executed").unwrap();
    input.runner_sha256 = hash(&input.runner).unwrap();
    write_json(&input.tasks,&json!({"schema_version":1,"expectations":tasks,"unsupported_obligations":[],"remaining_plan_gaps":[]})).unwrap();
    input
}
#[tokio::test]
async fn local_no_matching_tasks_are_not_reported_as_executed_and_directory_is_fresh() {
    let directory = tempfile::tempdir().unwrap();
    let input = inputs(directory.path(), vec![task(Backend::Cuda, "cuda")]);
    execute(input).await.unwrap();
    let report = read_json(&directory.path().join("reports/execution.json")).unwrap();
    assert_eq!(report["status"], "not_required");
    assert_eq!(report["other_profiles"], json!(["cuda"]));
    assert_eq!(report["release_approved"], false);
    let input = inputs(directory.path(), vec![]);
    assert!(execute(input).await.unwrap_err().contains("fresh"));
}
#[tokio::test]
async fn local_changed_binary_or_runner_bytes_fail_before_a_child_starts() {
    let directory = tempfile::tempdir().unwrap();
    let mut input = inputs(directory.path(), vec![task(Backend::Metal, "metal")]);
    input.runner_sha256 = "c".repeat(64);
    assert!(execute(input).await.unwrap_err().contains("runner SHA-256"));
    let input = inputs(directory.path(), vec![task(Backend::Metal, "metal")]);
    assert!(execute(input).await.unwrap_err().contains("Ferrum SHA-256"));
    assert!(!directory.path().join("reports").exists());
}
#[test]
fn local_missing_or_incomplete_passed_report_cannot_satisfy_task() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("report.json");
    let expected = task(Backend::Metal, "metal");
    assert!(checked_report(&path, &expected).is_err());
    write_json(&path, &json!({"status":"passed"})).unwrap();
    assert!(checked_report(&path, &expected)
        .unwrap_err()
        .contains("schema_version"));
}

// Child/grandchild are the same Rust test executable. No shell test scripts,
// model loading, SSH, or hardware dependencies are involved.
#[test]
fn local_process_fixture() {
    let Ok(mode) = std::env::var("FERRUM_LOCAL_PROCESS_FIXTURE") else {
        return;
    };
    let directory = PathBuf::from(std::env::var_os("FERRUM_LOCAL_PROCESS_DIRECTORY").unwrap());
    if mode == "server" {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        fs::write(
            directory.join("address"),
            listener.local_addr().unwrap().to_string(),
        )
        .unwrap();
        std::thread::sleep(Duration::from_secs(30));
        drop(listener);
        return;
    }
    if mode == "failure" {
        panic!("controlled model runner failure");
    }
    let _child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "local::tests::local_process_fixture",
            "--nocapture",
        ])
        .env("FERRUM_LOCAL_PROCESS_FIXTURE", "server")
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !directory.join("address").exists() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(5));
    }
    if mode == "timeout" {
        std::thread::sleep(Duration::from_secs(30));
    }
    // In success mode deliberately leave the descendant alive. This models a
    // runner exiting without its normal Process::Drop cleanup.
}
#[cfg(unix)]
#[tokio::test]
async fn local_runner_process_group_cleans_descendant_on_timeout_and_normal_exit() {
    for mode in ["timeout", "success", "failure"] {
        let directory = tempfile::tempdir().unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "local::tests::local_process_fixture",
                "--nocapture",
            ])
            .env("FERRUM_LOCAL_PROCESS_FIXTURE", mode)
            .env("FERRUM_LOCAL_PROCESS_DIRECTORY", directory.path());
        let result = run_command(
            command,
            &directory.path().join("runner"),
            Duration::from_secs(2),
        )
        .await;
        match mode {
            "timeout" => assert!(result.unwrap_err().contains("deadline")),
            "failure" => assert!(result.unwrap_err().contains("exited")),
            _ => result.unwrap(),
        }
        if mode != "failure" {
            let address = fs::read_to_string(directory.path().join("address"))
                .expect("real descendant had started listening")
                .parse()
                .unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while std::net::TcpStream::connect_timeout(&address, Duration::from_millis(50)).is_ok()
            {
                assert!(
                    std::time::Instant::now() < deadline,
                    "descendant still holds listening socket after group cleanup"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}
