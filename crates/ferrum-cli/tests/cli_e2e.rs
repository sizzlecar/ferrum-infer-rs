use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX_EPOCH")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ferrum-cli-{prefix}-{nanos}"));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn ferrum_bin() -> PathBuf {
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_ferrum") {
        return PathBuf::from(bin);
    }

    let current = std::env::current_exe().expect("failed to get current test executable path");
    let target_debug_dir = current
        .parent()
        .and_then(|p| p.parent())
        .expect("failed to locate target/debug directory from test executable");

    let mut bin = target_debug_dir.join("ferrum");
    if cfg!(windows) {
        bin.set_extension("exe");
    }

    assert!(bin.exists(), "ferrum binary not found at {}", bin.display());
    bin
}

fn base_cmd(cwd: &Path) -> Command {
    let mut cmd = Command::new(ferrum_bin());
    cmd.current_dir(cwd);
    cmd.env("NO_COLOR", "1");
    cmd
}

fn run(cmd: &mut Command) -> Output {
    cmd.output().expect("failed to run ferrum command")
}

fn assert_serve_usage(cmd: &Command, stdout: &str) {
    // Clap uses the invoked executable's filename, including .exe on Windows.
    let binary_name = Path::new(cmd.get_program())
        .file_name()
        .and_then(|name| name.to_str())
        .expect("ferrum executable has a UTF-8 filename");
    let expected = format!("Usage: {binary_name} serve [OPTIONS] [MODEL]");
    assert!(
        stdout.lines().any(|line| line == expected),
        "expected help usage line {expected:?}, got:\n{stdout}"
    );
}

fn configure_temp_env(cmd: &mut Command, temp_root: &Path) {
    cmd.env("TMPDIR", temp_root);
    cmd.env("TMP", temp_root);
    cmd.env("TEMP", temp_root);
}

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dirs");
    }
    fs::write(path, bytes).expect("write file");
}

fn create_cached_model(hf_home: &Path, model_id: &str, complete: bool, blob_size: usize) {
    let repo_dir = hf_home
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")));
    let snapshot_dir = repo_dir.join("snapshots").join("test-rev");
    let blobs_dir = repo_dir.join("blobs");

    fs::create_dir_all(&snapshot_dir).expect("create snapshot dir");
    fs::create_dir_all(&blobs_dir).expect("create blobs dir");

    write_file(&blobs_dir.join("blob.bin"), &vec![7u8; blob_size]);
    if complete {
        write_file(&snapshot_dir.join("model.safetensors"), b"stub");
        write_file(&snapshot_dir.join("config.json"), b"{}");
        write_file(&snapshot_dir.join("tokenizer.json"), b"{}");
    } else {
        write_file(&snapshot_dir.join("tokenizer.json"), b"{}");
    }
}

#[test]
fn cli_help_lists_commands() {
    let workspace = TempDirGuard::new("help-root");
    let mut cmd = base_cmd(workspace.path());
    cmd.arg("--help");
    let output = run(&mut cmd);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage: ferrum"));
    assert!(stdout.contains("run"));
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("stop"));
    assert!(stdout.contains("pull"));
    assert!(stdout.contains("list"));
    assert!(stdout.contains("doctor"));
    assert!(!stdout.contains("embed"));
    assert!(!stdout.contains("transcribe"));
    assert!(!stdout.contains("tts"));
}

#[test]
fn subcommand_help_smoke() {
    let workspace = TempDirGuard::new("help-subcommands");
    for sub in ["run", "serve", "pull", "list", "stop", "doctor"] {
        let mut cmd = base_cmd(workspace.path());
        cmd.arg(sub).arg("--help");
        let output = run(&mut cmd);
        assert!(output.status.success(), "{sub} --help should succeed");
    }
}

#[test]
fn retired_modal_commands_and_options_are_rejected() {
    let workspace = TempDirGuard::new("retired-modal-commands");
    for sub in ["embed", "e", "transcribe", "t", "tts"] {
        let mut cmd = base_cmd(workspace.path());
        cmd.arg(sub);
        let output = run(&mut cmd);
        assert!(!output.status.success(), "{sub} must not start a task");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand"),
            "{sub} must be rejected by command parsing"
        );
    }

    let mut cmd = base_cmd(workspace.path());
    cmd.args(["serve", "--tts-slots", "2"]);
    let output = run(&mut cmd);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unexpected argument") && stderr.contains("--tts-slots"));
}

#[test]
fn retired_runtime_config_is_rejected_before_run_or_serve() {
    let workspace = TempDirGuard::new("retired-runtime-config");
    for (key, value) in [
        ("moe_graph", "false"),
        ("moe_batch_threshold", "4"),
        ("whole_prompt_prefix_cache", "false"),
        ("prefix_cache_enabled", "false"),
        ("spec_draft", "\"\""),
        ("spec_n", "0"),
    ] {
        write_file(
            &workspace.path().join("ferrum.toml"),
            format!("[runtime]\n{key} = {value}\n").as_bytes(),
        );
        for command in ["run", "serve"] {
            let output = run(base_cmd(workspace.path()).arg(command));
            assert!(!output.status.success(), "{command} must reject {key}");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("unknown field") && stderr.contains(key),
                "{stderr}"
            );
        }
    }
}

#[test]
fn retired_scheduler_config_is_rejected_before_run_or_serve() {
    let workspace = TempDirGuard::new("retired-scheduler-config");
    for (key, value) in [
        ("policy", "\"Priority\""),
        ("scheduler_type", "\"fifo\""),
        ("enable_load_balancing", "false"),
        ("fair_share_weights", "{}"),
        ("enable_sla_enforcement", "false"),
    ] {
        write_file(
            &workspace.path().join("ferrum.toml"),
            format!("[scheduler]\n{key} = {value}\n").as_bytes(),
        );
        for command in ["run", "serve"] {
            let output = run(base_cmd(workspace.path()).arg(command));
            assert!(!output.status.success(), "{command} must reject {key}");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("unknown field") && stderr.contains(key),
                "{stderr}"
            );
        }
    }
}

#[test]
fn retired_execution_environment_is_rejected_before_model_selection() {
    let workspace = TempDirGuard::new("retired-execution-env");
    let keys = [
        "FERRUM_WHOLE_PROMPT_PREFIX_CACHE",
        "FERRUM_SPEC_DRAFT",
        "FERRUM_SPEC_N",
    ];
    for key in keys {
        for value in ["", "false", "0", "1"] {
            for command in ["run", "serve", "bench"] {
                let mut cmd = base_cmd(workspace.path());
                for removed in keys {
                    cmd.env_remove(removed);
                }
                // No explicit model is supplied: the retired option must be
                // reported before resolving a source or initializing a backend.
                let output = run(cmd.arg(command).env(key, value));
                assert!(!output.status.success(), "{command} must reject {key}");
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(
                    stderr.contains(key) && stderr.contains("retired"),
                    "{stderr}"
                );
            }
        }
    }
}

#[test]
fn retired_speculative_serve_options_are_rejected_by_parser() {
    let workspace = TempDirGuard::new("retired-speculative-options");
    for (option, value) in [("--spec-draft", "draft-model"), ("--spec-tokens", "4")] {
        let output = run(base_cmd(workspace.path()).args(["serve", option, value]));
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("unexpected argument") && stderr.contains(option),
            "{stderr}"
        );
    }
}

#[test]
fn list_reports_empty_cache() {
    let workspace = TempDirGuard::new("list-empty");
    let hf_home = workspace.path().join("hf-cache");
    fs::create_dir_all(&hf_home).expect("create hf cache dir");

    let mut cmd = base_cmd(workspace.path());
    cmd.arg("list").env("HF_HOME", &hf_home);
    let output = run(&mut cmd);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No models downloaded yet."));
}

#[test]
fn list_shows_ready_and_incomplete_models() {
    let workspace = TempDirGuard::new("list-models");
    let hf_home = workspace.path().join("hf-cache");
    fs::create_dir_all(&hf_home).expect("create hf cache dir");

    create_cached_model(&hf_home, "Acme/ReadyModel", true, 1024);
    create_cached_model(&hf_home, "Acme/IncompleteModel", false, 512);

    let mut cmd = base_cmd(workspace.path());
    cmd.arg("list").env("HF_HOME", &hf_home);
    let output = run(&mut cmd);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("NAME"));
    assert!(stdout.contains("Acme/ReadyModel"));
    assert!(stdout.contains("Acme/IncompleteModel"));
    assert!(stdout.contains("ready"));
    assert!(stdout.contains("incomplete"));
}

#[test]
fn list_redirected_output_is_plain_text_even_with_force_color() {
    let workspace = TempDirGuard::new("list-plain-text");
    let hf_home = workspace.path().join("hf-cache");
    create_cached_model(&hf_home, "Acme/ReadyModel", true, 1024);
    create_cached_model(&hf_home, "Acme/IncompleteModel", false, 512);

    // Test each command in a separate process rather than mutating global
    // environment or color switches in a parallel Rust test process.
    for no_color in [None, Some("1")] {
        let mut cmd = base_cmd(workspace.path());
        cmd.arg("list")
            .env("HF_HOME", &hf_home)
            .env_remove("NO_COLOR")
            .env_remove("CLICOLOR")
            .env("CLICOLOR_FORCE", "1")
            .env("TERM", "xterm-256color");
        if let Some(value) = no_color {
            cmd.env("NO_COLOR", value);
        }
        let output = run(&mut cmd);
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            !stdout.contains('\u{1b}'),
            "stdout contains ANSI: {stdout:?}"
        );
        assert!(
            !stderr.contains('\u{1b}'),
            "stderr contains ANSI: {stderr:?}"
        );
        assert!(stdout.contains("Acme/ReadyModel"));
        assert!(stdout.contains("Acme/IncompleteModel"));
        assert!(stdout.contains("ready"));
        assert!(stdout.contains("incomplete"));
    }
}

#[test]
fn serve_accepts_positional_model_argument() {
    let workspace = TempDirGuard::new("serve-positional-model");

    let mut cmd = base_cmd(workspace.path());
    cmd.arg("serve").arg("Acme/MissingModel").arg("--help");
    let output = run(&mut cmd);
    assert!(
        output.status.success(),
        "serve positional model should parse successfully"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_serve_usage(&cmd, &stdout);
    assert!(
        !stderr.contains("unexpected argument"),
        "serve positional model should parse successfully"
    );
}

#[test]
fn serve_accepts_model_flag_argument() {
    let workspace = TempDirGuard::new("serve-flag-model");

    let mut cmd = base_cmd(workspace.path());
    cmd.arg("serve")
        .arg("--model")
        .arg("Acme/MissingModel")
        .arg("--help");
    let output = run(&mut cmd);
    assert!(
        output.status.success(),
        "serve --model should parse successfully"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_serve_usage(&cmd, &stdout);
    assert!(
        !stderr.contains("unexpected argument"),
        "serve --model should parse successfully"
    );
}

#[test]
fn stop_without_pid_file_is_ok() {
    let workspace = TempDirGuard::new("stop-no-pid");
    let mut cmd = base_cmd(workspace.path());
    configure_temp_env(&mut cmd, workspace.path());
    cmd.arg("stop");

    let output = run(&mut cmd);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No running server found."));
}

#[test]
fn stop_removes_stale_pid_file() {
    let workspace = TempDirGuard::new("stop-stale-pid");
    let pid_file = workspace.path().join("ferrum.pid");
    write_file(&pid_file, b"999999\n");

    let mut cmd = base_cmd(workspace.path());
    configure_temp_env(&mut cmd, workspace.path());
    cmd.arg("stop");

    let output = run(&mut cmd);
    assert!(output.status.success());
    assert!(
        !pid_file.exists(),
        "stale pid file should be removed after stop command"
    );
}
