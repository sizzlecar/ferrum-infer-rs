use super::*;
use ferrum_bench_core::release_regression::{
    model_tasks::{ModelCheck, ModelRunCapacity},
    ExecutionTarget, ModelProfile,
};
use ferrum_types::ModelOutputProtocol;

pub(crate) fn args() -> ExecuteArgs {
    ExecuteArgs {
        tasks: "tasks.json".into(),
        archive: "cuda.tar.gz".into(),
        archive_sha256: "a".repeat(64),
        ferrum_bin: "ferrum".into(),
        runner: "model_regression".into(),
        runner_sha256: "b".repeat(64),
        report_dir: "reports".into(),
        repository_id: 1,
        run_id: 2,
        attempt: 1,
        disk_gib: 300,
        min_free_disk_gib: 200,
        min_cpu_ram_mb: 64000,
        max_hourly_usd: 0.75,
        max_network_usd_per_gb: 0.004,
        lease_secs: 7200,
        bootstrap_secs: 900,
        task_timeout_secs: 1800,
        max_create_attempts: 1,
    }
}
fn task(backend: Backend, id: &str) -> ExpectedModelRun {
    ExpectedModelRun {
        profile: ModelProfile {
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
        checks: vec![ModelCheck::Basic, ModelCheck::Stop],
        disable_thinking: true,
        use_default_backend: true,
        max_tokens: 512,
        runtime_capacity: None,
        reasoning_alias_replay: false,
        stop_prompt: "alpha 'beta' $(never-run)\nnext".into(),
    }
}
#[test]
fn cloud_ownership_requires_exact_namespace_ids_and_expiry() {
    let owner = Ownership {
        repository: 12,
        run: 34,
        attempt: 2,
        expiry: 999,
        nonce: "a".repeat(32),
    };
    let label = owner.label();
    assert_eq!(Ownership::parse(&label), Some(owner));
    for invalid in [
        format!("other-{label}"),
        format!("{label}:extra"),
        label.replace(":r12:", ":r012:"),
        label.replace(":a2:", ":a0:"),
        label.replace(":x999:", ":xno:"),
        label.replace(":n", ":nG"),
    ] {
        assert!(Ownership::parse(&invalid).is_none(), "{invalid}");
    }
}
#[test]
fn cloud_task_filter_preserves_cuda_options_and_does_not_substitute_metal() {
    let cuda = task(Backend::Cuda, "cuda");
    let metal = task(Backend::Metal, "metal");
    let (selected, _, other) = cuda_tasks(PreparedTasks {
        schema_version: 1,
        expectations: vec![metal, cuda.clone()],
        unsupported_obligations: vec![],
        remaining_plan_gaps: vec![],
    })
    .unwrap();
    assert_eq!(selected, vec![cuda]);
    assert_eq!(other, vec!["metal"]);
    assert!(cuda_tasks(PreparedTasks {
        schema_version: 1,
        expectations: vec![],
        unsupported_obligations: vec![7],
        remaining_plan_gaps: vec![]
    })
    .is_err());
    assert!(cuda_tasks(PreparedTasks {
        schema_version: 1,
        expectations: vec![task(Backend::Cuda, "same"), task(Backend::Metal, "same")],
        unsupported_obligations: vec![],
        remaining_plan_gaps: vec![]
    })
    .is_err());
}
#[test]
fn cloud_runner_arguments_preserve_expected_semantics_and_quote_shell_data() {
    let mut task = task(Backend::Cuda, "cuda");
    task.checks.push(ModelCheck::Tools);
    task.reasoning_alias_replay = true;
    let words = ssh::runner_command(
        &task,
        "/workspace/release",
        "/workspace/task.json",
        "/workspace/report",
        1800,
    );
    let after = |flag| {
        words
            .windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
    };
    assert_eq!(after("--model"), Some("org/model"));
    assert_eq!(after("--stop-prompt"), Some(task.stop_prompt.as_str()));
    assert_eq!(after("--checks"), Some("basic,stop,tools"));
    assert_eq!(after("--max-tokens"), Some("512"));
    // Cold loading shares the whole task's existing limit; request and server
    // startup limits keep their independent defaults and are not inflated.
    assert_eq!(after("--run-timeout-secs"), Some("1800"));
    assert_eq!(words[3], "1800s");
    assert_eq!(after("--startup-timeout-secs"), None);
    assert_eq!(after("--request-timeout-secs"), None);
    assert_eq!(after("--context-tokens"), None);
    assert_eq!(after("--max-num-seqs"), None);
    for flag in [
        "--disable-thinking",
        "--use-default-backend",
        "--reasoning-alias-replay",
    ] {
        assert!(words.iter().any(|word| word == flag));
    }
    assert_eq!(ssh::quote("a'b $(x)\n"), "'a'\\''b $(x)\n'");
    task.runtime_capacity = Some(ModelRunCapacity {
        context_tokens: 2048,
        max_num_seqs: 1,
    });
    let words = ssh::runner_command(&task, "/workspace/release", "task", "report", 1800);
    for (flag, value) in [("--context-tokens", "2048"), ("--max-num-seqs", "1")] {
        assert!(words
            .windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value));
    }
}
#[test]
fn cloud_capacity_preflight_rejects_an_exhausted_context_before_rental() {
    let mut expected = task(Backend::Cuda, "cuda");
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
        cuda_tasks(document(expected.clone())).unwrap().0,
        vec![expected.clone()]
    );
    for (context_tokens, max_num_seqs) in [(512, 1), (0, 1), (2048, 0)] {
        expected.runtime_capacity = Some(ModelRunCapacity {
            context_tokens,
            max_num_seqs,
        });
        assert!(cuda_tasks(document(expected.clone())).is_err());
    }
}
#[test]
fn cloud_device_rejects_nominal_or_wrong_gpu_capacity_and_backend() {
    assert!(
        ssh::validate_device("NVIDIA RTX 6000 Ada Generation, 49140, 8.9, 580.95.05\n").is_ok()
    );
    for wrong in [
        "NVIDIA RTX 4090, 49140, 8.9, 580.95.05",
        "NVIDIA L40S, 20000, 8.9, 580.95.05",
        "NVIDIA L40S, 49140, 8.6, 580.95.05",
        "NVIDIA L40S, 49140, 8.9, 535.1.1",
        "NVIDIA L40S, 49140, 8.9, 580.95.05\nNVIDIA L40S, 49140, 8.9, 580.95.05",
    ] {
        assert!(ssh::validate_device(wrong).is_err(), "{wrong}");
    }
}
#[test]
fn cloud_price_limits_and_create_count_are_explicit() {
    let mut input = args();
    assert!(validate_args(&input).is_ok());
    input.max_create_attempts = 2;
    assert!(validate_args(&input).is_ok());
    input.max_create_attempts = 3;
    assert!(validate_args(&input).is_ok());
    input.max_create_attempts = 4;
    assert!(validate_args(&input).is_err());
    input.max_create_attempts = 0;
    assert!(validate_args(&input).is_err());
    input = args();
    input.max_hourly_usd = f64::NAN;
    assert!(validate_args(&input).is_err());
    input = args();
    input.bootstrap_secs = input.lease_secs;
    assert!(validate_args(&input).is_err());
}

#[tokio::test]
async fn cloud_controlled_ssh_success_cannot_hide_missing_or_incomplete_report() {
    fn no_evidence(program: &str, arguments: &[std::ffi::OsString]) -> Result<String, String> {
        if program == "ssh" {
            assert!(arguments.iter().any(|value| value == "root@ssh5.vast.ai"));
            assert!(arguments.iter().any(|value| value == "12122"));
            assert!(arguments
                .last()
                .unwrap()
                .to_string_lossy()
                .contains("'--expected-task'"));
        }
        Ok(String::new())
    }
    fn invalid_report(program: &str, arguments: &[std::ffi::OsString]) -> Result<String, String> {
        no_evidence(program, arguments)?;
        if program == "scp" && arguments.iter().any(|value| value == "-r") {
            let report = PathBuf::from(arguments.last().unwrap()).join("report-0");
            fs::create_dir(&report).unwrap();
            write_json(&report.join("report.json"), &json!({"status":"passed"})).unwrap();
        }
        Ok(String::new())
    }
    let instance = api::Instance {
        id: 1,
        label: None,
        actual_status: Some("running".into()),
        ssh_host: Some("ssh5.vast.ai".into()),
        ssh_port: Some(12122),
    };
    for (fixture, expected) in [
        (
            no_evidence as fn(&str, &[std::ffi::OsString]) -> Result<String, String>,
            "missing task report",
        ),
        (invalid_report, "schema"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut input = args();
        input.report_dir = directory.path().into();
        let mut remote = ssh::Remote::new(
            &instance,
            &directory.path().join("test-key"),
            "/workspace/test".into(),
        )
        .unwrap();
        remote.fixture = Some(fixture);
        let error = remote
            .run_task(&input, &task(Backend::Cuda, "cuda"), 0)
            .await
            .unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}

// The child fixture is the Rust test executable itself, not an added shell or
// Python script. Normal test discovery performs no child behavior.
#[test]
fn cloud_transport_child_fixture() {
    let Ok(mode) = std::env::var("FERRUM_CLOUD_TRANSPORT_FIXTURE") else {
        return;
    };
    match mode.as_str() {
        "failure" => panic!("controlled SSH subprocess failure"),
        "timeout" => {
            fs::write(
                std::env::var("FERRUM_CLOUD_TRANSPORT_PID").unwrap(),
                std::process::id().to_string(),
            )
            .unwrap();
            std::thread::sleep(Duration::from_secs(30));
        }
        _ => panic!("unknown fixture mode"),
    }
}
#[tokio::test]
async fn cloud_subprocess_failure_and_timeout_are_errors_and_timeout_reaps_child() {
    for mode in ["failure", "timeout"] {
        let directory = tempfile::tempdir().unwrap();
        let pid_file = directory.path().join("pid");
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "cloud::tests::cloud_transport_child_fixture",
                "--nocapture",
            ])
            .env("FERRUM_CLOUD_TRANSPORT_FIXTURE", mode)
            .env("FERRUM_CLOUD_TRANSPORT_PID", &pid_file);
        let error = ssh::command_with(
            command,
            "controlled-ssh",
            &directory.path().join(mode),
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        if mode == "failure" {
            assert!(error.contains("failed"));
        } else {
            assert!(error.contains("timed out"));
            let pid =
                fs::read_to_string(&pid_file).expect("child entered Rust fixture before timeout");
            assert!(
                pid.trim().parse::<u32>().is_ok_and(|value| value > 0),
                "child fixture omitted a valid process ID"
            );
            #[cfg(unix)]
            {
                let status = std::process::Command::new("kill")
                    .args(["-0", pid.trim()])
                    .stderr(std::process::Stdio::null())
                    .status()
                    .unwrap();
                assert!(!status.success(), "timed-out child still exists");
            }
        }
    }
}
