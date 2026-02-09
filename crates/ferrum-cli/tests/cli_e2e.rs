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
    assert!(stdout.contains("embed"));
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("stop"));
    assert!(stdout.contains("pull"));
    assert!(stdout.contains("list"));
}

#[test]
fn subcommand_help_smoke() {
    let workspace = TempDirGuard::new("help-subcommands");
    for sub in ["run", "embed", "serve", "pull", "list", "stop"] {
        let mut cmd = base_cmd(workspace.path());
        cmd.arg(sub).arg("--help");
        let output = run(&mut cmd);
        assert!(output.status.success(), "{sub} --help should succeed");
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
