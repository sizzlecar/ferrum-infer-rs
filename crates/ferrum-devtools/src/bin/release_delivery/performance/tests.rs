use super::source::{FilePin, SourceManifest};
use super::*;
use clap::Parser;
use ferrum_bench_core::release_regression::{Backend, ExecutionTarget};
use ferrum_types::{ModelOutputProtocol, ModelReasoningProtocol};
use sha2::{Digest, Sha256};
fn profile() -> ModelProfile {
    ModelProfile {
        gguf: None,
        id: "legacy-metal".into(),
        model: "llama3.1:8b-q4_k_m".into(),
        target: ExecutionTarget {
            architecture: "llama_dense".into(),
            protocol: ModelOutputProtocol::Text,
            precision: "gguf-q4_k_m".into(),
            backend: Backend::Metal,
            execution_path: "legacy-model-executor".into(),
        },
        available: true,
        estimate: None,
        reasoning_protocol: ModelReasoningProtocol::None,
    }
}
fn pin(path: PathBuf, bytes: &[u8]) -> FilePin {
    FilePin {
        path,
        bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(bytes)),
    }
}
fn fixture(root: &Path) -> SourceManifest {
    let weight = b"small pinned weight fixture; never loaded by a model";
    let directory = root.join("sidecars");
    fs::create_dir(&directory).unwrap();
    fs::write(root.join("weight.gguf"), weight).unwrap();
    fs::write(directory.join("tokenizer.json"), b"{}").unwrap();
    fs::write(directory.join("tokenizer_config.json"), b"{}").unwrap();
    SourceManifest {
        schema_version: 1,
        profile: profile(),
        gguf: pin(root.join("weight.gguf"), weight),
        tokenizer_dir: directory,
        sidecars: vec![
            pin("tokenizer.json".into(), b"{}"),
            pin("tokenizer_config.json".into(), b"{}"),
        ],
    }
}
fn args(root: &Path) -> PerformanceArgs {
    PerformanceArgs {
        baseline_bin: root.join("baseline"),
        baseline_sha256: "a".repeat(64),
        baseline_version: "0.8.7".into(),
        candidate_bin: root.join("candidate"),
        candidate_sha256: "b".repeat(64),
        candidate_version: "0.8.8".into(),
        client_bin: root.join("client"),
        client_sha256: "c".repeat(64),
        client_version: "0.8.8".into(),
        source_manifest: root.join("source.json"),
        expected_task: root.join("task.json"),
        input_tokens: 512,
        output_tokens: 128,
        measured_requests: 4,
        warmup_requests: 2,
        repeats: 3,
        seed: 7,
        max_model_len: 2048,
        runtime_memory_budget_bytes: 8 * 1024 * 1024 * 1024,
        startup_timeout_secs: 60,
        request_timeout_secs: 30,
        task_timeout_secs: 600,
        ttft_max_relative_increase: 0.1,
        tpot_max_relative_increase: 0.1,
        report_dir: root.join("reports"),
    }
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}
#[test]
fn pinned_source_rejects_alias_traversal_missing_metadata_and_profile_substitution() {
    let root = tempfile::tempdir().unwrap();
    let mut source = fixture(root.path());
    source.validate(&profile(), deadline()).unwrap();
    let original = source.gguf.path.clone();
    source.gguf.path = "llama3.1:8b-q4_k_m".into();
    assert!(source
        .validate(&profile(), deadline())
        .unwrap_err()
        .contains("absolute local"));
    source.gguf.path = original;
    source.sidecars[0].path = "../tokenizer.json".into();
    assert!(source
        .validate(&profile(), deadline())
        .unwrap_err()
        .contains("metadata name"));
    source.sidecars[0].path = "tokenizer.json".into();
    let removed = source.sidecars.pop().unwrap();
    assert!(source
        .validate(&profile(), deadline())
        .unwrap_err()
        .contains("tokenizer_config"));
    source.sidecars.push(removed);
    let mut substituted = profile();
    substituted.target.backend = Backend::Cuda;
    assert!(source.validate(&substituted, deadline()).is_err());
}
#[test]
#[cfg(unix)]
fn pinned_bundle_reuses_weight_blob_isolates_metadata_and_detects_mutation() {
    let root = tempfile::tempdir().unwrap();
    let source = fixture(root.path());
    fs::write(
        source.tokenizer_dir.join("unregistered.jinja"),
        b"must never enter the bundle",
    )
    .unwrap();
    let bundle = source
        .prepare(&root.path().join("bundle"), deadline())
        .unwrap();
    assert!(fs::symlink_metadata(&bundle.gguf)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        fs::canonicalize(&bundle.gguf).unwrap(),
        fs::canonicalize(&source.gguf.path).unwrap()
    );
    assert!(!bundle.tokenizer_dir.join("unregistered.jinja").exists());
    source.verify_bundle(&bundle, deadline()).unwrap();
    fs::write(bundle.tokenizer_dir.join("tokenizer.json"), b"[]").unwrap();
    assert!(source
        .verify_bundle(&bundle, deadline())
        .unwrap_err()
        .contains("digest"));
    fs::write(bundle.tokenizer_dir.join("tokenizer.json"), b"{}").unwrap();
    fs::write(bundle.tokenizer_dir.join("extra.json"), b"{}").unwrap();
    assert!(source
        .verify_bundle(&bundle, deadline())
        .unwrap_err()
        .contains("unregistered"));
}
#[test]
fn performance_limits_and_ci_workload_are_required_before_any_model_work() {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: PerformanceArgs,
    }
    use clap::CommandFactory;
    Cli::command().debug_assert();
    for name in ["ttft_max_relative_increase", "tpot_max_relative_increase"] {
        let command = Cli::command();
        let argument = command
            .get_arguments()
            .find(|a| a.get_id().as_str() == name)
            .unwrap();
        assert!(argument.is_required_set());
        assert!(argument.get_default_values().is_empty());
    }
    let mut input = args(Path::new("/unused"));
    input.validate().unwrap();
    input.ttft_max_relative_increase = f64::NAN;
    assert!(input.validate().is_err());
    input.ttft_max_relative_increase = 0.1;
    input.repeats = 2;
    assert!(input.validate().is_err());
    input.repeats = 3;
    input.input_tokens = u32::MAX;
    assert!(input.validate().is_err());
}
#[tokio::test]
async fn invalid_execution_retains_failed_receipt_and_never_reuses_report_directory() {
    let root = tempfile::tempdir().unwrap();
    let mut input = args(root.path());
    input.repeats = 0;
    assert!(execute(input).await.unwrap_err().contains("workload"));
    let report: PerformanceReport =
        read_json(&root.path().join("reports/performance.json")).unwrap();
    assert!(matches!(report.status, Status::Invalid));
    assert!(report.completed_phases.is_empty());
    assert!(report.error.is_some());
    assert!(!report.release_approved);
    assert!(execute(args(root.path()))
        .await
        .unwrap_err()
        .contains("fresh"));
}
#[test]
fn child_commands_bind_same_local_files_and_fixed_generation_workload() {
    let input = args(Path::new("/unused"));
    let bundle = source::Bundle {
        gguf: "/fixed/model.gguf".into(),
        tokenizer_dir: "/fixed".into(),
    };
    let words = process::client_arguments(&input, &bundle, 1234, Path::new("/report/bench.json"));
    let value = |flag: &str| {
        words
            .windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].to_string_lossy().into_owned())
    };
    assert_eq!(value("--tokenizer"), Some("/fixed".into()));
    assert_eq!(value("--seed"), Some("7".into()));
    assert_eq!(value("--random-output-len"), Some("128".into()));
    assert_eq!(value("--target-backend"), Some("metal".into()));
    assert!(words.iter().any(|w| w == "--ignore-eos"));
    assert!(words.iter().any(|w| w == "--fail-on-error"));
    let server = process::server_arguments(&input, &bundle, 1234);
    assert_eq!(server[1], "/fixed/model.gguf");
    assert!(server.iter().any(|w| w == "--runtime-memory-budget-bytes"));
    for flag in [
        "--kv-capacity",
        "--max-model-len",
        "--max-num-batched-tokens",
    ] {
        let capacity = server.windows(2).find(|w| w[0] == flag).unwrap();
        assert_eq!(capacity[1], input.max_model_len.to_string().as_str());
    }
}

#[tokio::test]
#[cfg(unix)]
async fn failed_server_start_is_invalid_and_reaped_before_return() {
    let root = tempfile::tempdir().unwrap();
    let input = args(root.path());
    let bundle = source::Bundle {
        gguf: root.path().join("model.gguf"),
        tokenizer_dir: root.path().to_owned(),
    };
    let phase = root.path().join("phase");
    fs::create_dir(&phase).unwrap();
    let error = process::phase(
        &input,
        Path::new("/usr/bin/false"),
        Path::new("/usr/bin/false"),
        "0.8.7",
        &bundle,
        &phase,
        deadline(),
    )
    .await
    .unwrap_err();
    assert!(error.contains("exited before readiness"), "{error}");
    let receipt: serde_json::Value = read_json(&phase.join("execution.json")).unwrap();
    assert_eq!(receipt["client_completed_successfully"], false);
    assert_eq!(receipt["cleanup_completed"], true);
    assert!(!phase.join("bench.json").exists());
    let pid = receipt["server_pid"].as_i64().unwrap() as i32;
    // SAFETY: signal zero only checks the child PID just reaped by phase().
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}

fn replay_fixture(directory: &Path) -> ExpectedPerformanceRun {
    use ferrum_bench_core::release_regression::performance::ExpectedPerformanceRun;
    fn mac_wire_arguments(args: Vec<std::ffi::OsString>) -> serde_json::Value {
        json!(args
            .iter()
            .map(|arg| json!({"Unix": arg.to_str().unwrap().as_bytes()}))
            .collect::<Vec<_>>())
    }
    // Preserve the actual POSIX spelling a Mac worker writes; the replay host
    // must not manufacture different command evidence with its own path joins.
    let mut input = PerformanceArgs {
        baseline_bin: "/missing-mac-worker/baseline".into(),
        candidate_bin: "/missing-mac-worker/candidate".into(),
        client_bin: "/missing-mac-worker/client".into(),
        source_manifest: "/missing-mac-worker/source.json".into(),
        expected_task: "/missing-mac-worker/task.json".into(),
        report_dir: "/missing-mac-worker/reports".into(),
        ..args(directory)
    };
    input.input_tokens = 32;
    input.output_tokens = 8;
    input.measured_requests = 2;
    input.warmup_requests = 1;
    input.max_model_len = 128;
    let mut source = fixture(directory);
    source.gguf.path = "/missing-mac-worker/snapshot/model.gguf".into();
    source.tokenizer_dir = "/missing-mac-worker/tokenizer".into();
    let expected = ExpectedPerformanceRun {
        schema_version: 1,
        profile: profile(),
        baseline_version: input.baseline_version.clone(),
        baseline_sha256: input.baseline_sha256.clone(),
        candidate_version: input.candidate_version.clone(),
        candidate_sha256: input.candidate_sha256.clone(),
        client_version: input.client_version.clone(),
        client_sha256: input.client_sha256.clone(),
        policy: input.policy(),
        source: source.identity().unwrap(),
        obligations: vec![9],
    };
    expected.validate().unwrap();
    write_json(&directory.join("inputs.json"), &input).unwrap();
    write_json(&directory.join("expected-task.json"), &expected).unwrap();
    write_json(&directory.join("source-manifest.json"), &source).unwrap();
    let bundle = source::Bundle {
        gguf: "/missing-mac-worker/bundle/model.gguf".into(),
        tokenizer_dir: "/missing-mac-worker/bundle".into(),
    };
    write_json(
        &directory.join("bound-source.json"),
        &json!({"gguf_canonical":"/missing-mac-worker/blob","bundle":bundle}),
    )
    .unwrap();
    let mut binaries = serde_json::Map::new();
    for (label, path, sha, version) in [
        (
            "baseline",
            &input.baseline_bin,
            &input.baseline_sha256,
            &input.baseline_version,
        ),
        (
            "candidate",
            &input.candidate_bin,
            &input.candidate_sha256,
            &input.candidate_version,
        ),
        (
            "client",
            &input.client_bin,
            &input.client_sha256,
            &input.client_version,
        ),
    ] {
        let stdout = format!("ferrum {version}\n");
        binaries.insert(label.into(),json!({"requested":path,"canonical":path,"sha256":sha,"version":version,"version_exit_code":0,"version_stdout":stdout}));
        fs::write(
            directory.join(format!("{label}-version.stdout.log")),
            stdout,
        )
        .unwrap();
    }
    write_json(&directory.join("binaries.json"), &binaries).unwrap();
    let baseline = compare::fixture_report([1.0; 3]);
    let candidate = compare::fixture_report([1.02; 3]);
    for (phase, path, sha, version, measurement) in [
        (
            "baseline-a",
            &input.baseline_bin,
            &input.baseline_sha256,
            &input.baseline_version,
            &baseline,
        ),
        (
            "baseline-b",
            &input.baseline_bin,
            &input.baseline_sha256,
            &input.baseline_version,
            &baseline,
        ),
        (
            "candidate",
            &input.candidate_bin,
            &input.candidate_sha256,
            &input.candidate_version,
            &candidate,
        ),
    ] {
        let location = directory.join(phase);
        fs::create_dir(&location).unwrap();
        write_json(&location.join("bench.json"), measurement).unwrap();
        write_json(&location.join("checks.json"),&json!({"source_before":expected.source,"source_after":expected.source,
            "server_sha256_before":sha,"server_sha256_after":sha,"client_sha256_before":input.client_sha256,"client_sha256_after":input.client_sha256})).unwrap();
        write_json(&location.join("health.json"),&json!({"status":"healthy","version":version,"auto_config":{
            "hardware_capabilities":{"backend":"Metal"},"selected_kv_capacity":input.max_model_len,
            "selected_max_model_len":input.max_model_len,"selected_max_sequences":1,
            "selected_max_batched_tokens":input.max_model_len
        }})).unwrap();
        write_json(&location.join("execution.json"),&json!({"server_pid":123,"client_exit_code":0,"client_cleanup_completed":true,
            "client_completed_successfully":true,"cleanup_completed":true,"error":null,"cleanup_error":null})).unwrap();
        let worker_report = format!("/missing-mac-worker/reports/{phase}/bench.json");
        write_json(&location.join("commands.json"),&json!({"port":1234,"server":{"program":path,"args":mac_wire_arguments(process::server_arguments(&input,&bundle,1234))},
            "client":{"program":input.client_bin,"args":mac_wire_arguments(process::client_arguments(&input,&bundle,1234,Path::new(&worker_report)))}})).unwrap();
    }
    let comparison = compare::compare(
        &baseline,
        &baseline,
        &candidate,
        &input.workload(),
        &input.limits(),
    )
    .unwrap();
    let report = PerformanceReport {
        schema_version: 1,
        status: Status::Passed,
        profile: Some(profile()),
        comparison: Some(comparison),
        calibration_stable: Some(true),
        completed_phases: vec!["baseline-a".into(), "baseline-b".into(), "candidate".into()],
        elapsed_seconds: 1.0,
        error: None,
        release_approved: false,
    };
    write_json(&directory.join("performance.json"), &report).unwrap();
    expected
}
#[test]
fn performance_replay_uses_original_measurements_without_opening_remote_model_paths() {
    let root = tempfile::tempdir().unwrap();
    let expected = replay_fixture(root.path());
    let commands: serde_json::Value =
        read_json(&root.path().join("candidate/commands.json")).unwrap();
    let client_args = commands["client"]["args"].as_array().unwrap();
    assert_eq!(
        client_args
            .windows(2)
            .find(|args| args[0] == json!({"Unix": "--out".as_bytes()}))
            .unwrap()[1],
        json!({"Unix": "/missing-mac-worker/reports/candidate/bench.json".as_bytes()})
    );
    verify_evidence(&expected, root.path()).unwrap();
    write_json(
        &root.path().join("candidate/bench.json"),
        &compare::fixture_report([1.5; 3]),
    )
    .unwrap();
    assert!(verify_evidence(&expected, root.path())
        .unwrap_err()
        .contains("raw benchmark"));
}

#[test]
fn performance_replay_rejects_invalid_remote_paths_and_mismatched_bundle() {
    for (file, pointer, replacement, reason) in [
        (
            "bound-source.json",
            "/bundle/tokenizer_dir",
            "relative/bundle",
            "POSIX worker path",
        ),
        (
            "bound-source.json",
            "/bundle/gguf",
            "/missing-mac-worker/bundle/../model.gguf",
            "POSIX worker path",
        ),
        (
            "bound-source.json",
            "/gguf_canonical",
            "/missing-mac-worker/../blob",
            "invalid bound source paths",
        ),
        (
            "bound-source.json",
            "/bundle/gguf",
            "/different-bundle/model.gguf",
            "invalid bound source paths",
        ),
        (
            "binaries.json",
            "/baseline/canonical",
            "relative/baseline",
            "POSIX worker path",
        ),
        (
            "binaries.json",
            "/candidate/canonical",
            "/missing-mac-worker/../candidate",
            "POSIX worker path",
        ),
        (
            "binaries.json",
            "/client/canonical",
            "C:\\worker\\client.exe",
            "POSIX worker path",
        ),
        (
            "inputs.json",
            "/report_dir",
            "relative/reports",
            "POSIX worker path",
        ),
        (
            "inputs.json",
            "/report_dir",
            "/missing-mac-worker/../reports",
            "POSIX worker path",
        ),
        (
            "inputs.json",
            "/report_dir",
            "/missing-mac-worker\\reports",
            "POSIX worker path",
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let expected = replay_fixture(root.path());
        let path = root.path().join(file);
        let mut value: serde_json::Value = read_json(&path).unwrap();
        *value.pointer_mut(pointer).unwrap() = json!(replacement);
        write_json(&path, &value).unwrap();
        let error = verify_evidence(&expected, root.path()).unwrap_err();
        assert!(
            error.contains(reason),
            "{file}{pointer} = {replacement:?}: {error}"
        );
    }
}

#[test]
fn performance_replay_rejects_changed_policy_source_command_and_missing_measurement() {
    for mutation in [
        "policy",
        "source",
        "command",
        "worker_argument_encoding",
        "mixed_worker_argument_encoding",
        "missing",
        "summary",
        "capacity",
    ] {
        let root = tempfile::tempdir().unwrap();
        let expected = replay_fixture(root.path());
        match mutation {
            "capacity" => {
                let path = root.path().join("candidate/health.json");
                let mut value: serde_json::Value = read_json(&path).unwrap();
                value["auto_config"]["selected_kv_capacity"] = json!(32);
                write_json(&path, &value).unwrap();
            }
            "summary" => {
                let path = root.path().join("performance.json");
                let mut value: serde_json::Value = read_json(&path).unwrap();
                value["comparison"]["candidate_relative"]["ttft"]["mean"] = json!(0.005);
                write_json(&path, &value).unwrap();
            }
            "policy" => {
                let path = root.path().join("inputs.json");
                let mut value: serde_json::Value = read_json(&path).unwrap();
                value["seed"] = json!(8);
                write_json(&path, &value).unwrap();
            }
            "source" => {
                let path = root.path().join("candidate/checks.json");
                let mut value: serde_json::Value = read_json(&path).unwrap();
                value["source_after"]["gguf"]["sha256"] = json!("d".repeat(64));
                write_json(&path, &value).unwrap();
            }
            "command" => {
                let path = root.path().join("candidate/commands.json");
                let mut value: serde_json::Value = read_json(&path).unwrap();
                value["server"]["args"][3] = json!("cpu");
                write_json(&path, &value).unwrap();
            }
            "worker_argument_encoding" | "mixed_worker_argument_encoding" => {
                let path = root.path().join("candidate/commands.json");
                let mut value: serde_json::Value = read_json(&path).unwrap();
                let args = value["client"]["args"].as_array_mut().unwrap();
                for arg in args {
                    let bytes: Vec<u8> = serde_json::from_value(arg["Unix"].clone()).unwrap();
                    let text = String::from_utf8(bytes).unwrap();
                    *arg = json!({"Windows": text.encode_utf16().collect::<Vec<_>>()});
                    if mutation == "mixed_worker_argument_encoding" {
                        break;
                    }
                }
                write_json(&path, &value).unwrap();
            }
            "missing" => fs::remove_file(root.path().join("candidate/bench.json")).unwrap(),
            _ => unreachable!(),
        }
        assert!(
            verify_evidence(&expected, root.path()).is_err(),
            "accepted {mutation}"
        );
    }
}

#[tokio::test]
#[cfg(unix)]
async fn preflight_failure_does_not_leave_weight_links_in_uploaded_reports() {
    let root = tempfile::tempdir().unwrap();
    let input = args(root.path());
    let source = fixture(root.path());
    let expected = ExpectedPerformanceRun {
        schema_version: 1,
        profile: profile(),
        baseline_version: input.baseline_version.clone(),
        baseline_sha256: input.baseline_sha256.clone(),
        candidate_version: input.candidate_version.clone(),
        candidate_sha256: input.candidate_sha256.clone(),
        client_version: input.client_version.clone(),
        client_sha256: input.client_sha256.clone(),
        policy: input.policy(),
        source: source.identity().unwrap(),
        obligations: vec![9],
    };
    write_json(&input.source_manifest, &source).unwrap();
    write_json(&input.expected_task, &expected).unwrap();
    assert!(execute(input)
        .await
        .unwrap_err()
        .contains("resolve baseline binary"));
    let bound: serde_json::Value =
        read_json(&root.path().join("reports/bound-source.json")).unwrap();
    let directory = PathBuf::from(bound["bundle"]["tokenizer_dir"].as_str().unwrap());
    assert!(
        !directory.exists(),
        "owned temporary model directory was not removed"
    );
    assert!(
        !root.path().join("reports/model").exists(),
        "uploadable evidence must not include model links"
    );
    assert!(
        source.gguf.path.is_file(),
        "cleanup must retain the cached weight source"
    );
}
