//! Execute prepared CPU/Metal tasks with the existing real-model runner.
use clap::{Args, ValueEnum};
use ferrum_bench_core::release_regression::{
    model_tasks::{verify_model_options, verify_model_report, ExpectedModelRun},
    Backend, Gap,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::process::Stdio;
use std::{
    collections::BTreeSet,
    ffi::OsString,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LocalBackend {
    Cpu,
    Metal,
}
impl LocalBackend {
    fn backend(self) -> Backend {
        match self {
            Self::Cpu => Backend::Cpu,
            Self::Metal => Backend::Metal,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug, Args)]
pub struct LocalArgs {
    #[arg(long)]
    pub tasks: PathBuf,
    #[arg(long, value_enum)]
    pub backend: LocalBackend,
    #[arg(long)]
    pub ferrum_bin: PathBuf,
    #[arg(long)]
    pub runner: PathBuf,
    #[arg(long)]
    pub runner_sha256: String,
    /// Must not exist. Each task keeps a separate original runner report directory.
    #[arg(long)]
    pub report_dir: PathBuf,
    #[arg(long)]
    pub task_timeout_secs: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedTasks {
    schema_version: u32,
    expectations: Vec<ExpectedModelRun>,
    unsupported_obligations: Vec<usize>,
    remaining_plan_gaps: Vec<Gap>,
}
fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", path.display()))
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", path.display()))
}
fn hash(path: &Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn timestamp() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| error.to_string())
}
fn select(
    document: PreparedTasks,
    backend: LocalBackend,
) -> Result<(Vec<ExpectedModelRun>, Vec<String>, Vec<Gap>), String> {
    if document.schema_version != 1 || !document.unsupported_obligations.is_empty() {
        return Err("unsupported model task schema or obligations".into());
    }
    let mut selected = Vec::new();
    let mut other = Vec::new();
    let mut seen = BTreeSet::new();
    for task in document.expectations {
        if !seen.insert(task.profile.id.clone()) {
            return Err("duplicate prepared profile ID".into());
        }
        if task.profile.target.backend != backend.backend() {
            other.push(task.profile.id);
            continue;
        }
        if !task.profile.available {
            return Err(format!(
                "selected profile {} is unavailable",
                task.profile.id
            ));
        }
        verify_model_options(&task,&json!({"profile_id":task.profile.id,"model":task.profile.model,"backend":backend.name(),
            "gguf_file":task.profile.gguf.as_ref().map(|gguf|&gguf.filename),
            "stop_prompt":task.stop_prompt,"checks":task.checks,"max_tokens":task.max_tokens,
            "context_tokens":task.runtime_capacity.as_ref().map(|capacity|capacity.context_tokens),
            "max_num_seqs":task.runtime_capacity.as_ref().map(|capacity|capacity.max_num_seqs),
            "disable_thinking":task.disable_thinking,"use_default_backend":task.use_default_backend,"reasoning_alias_replay":task.reasoning_alias_replay}))
            .map_err(|issues|issues.join("; "))?;
        selected.push(task);
    }
    Ok((selected, other, document.remaining_plan_gaps))
}
fn runner_arguments(
    args: &LocalArgs,
    task: &ExpectedModelRun,
    expected: &Path,
    report: &Path,
) -> Vec<OsString> {
    let mut words = vec![
        "--ferrum-bin".into(),
        args.ferrum_bin.as_os_str().into(),
        "--model".into(),
        task.profile.model.clone().into(),
        "--backend".into(),
        args.backend.name().into(),
        "--profile-id".into(),
        task.profile.id.clone().into(),
        "--expected-task".into(),
        expected.as_os_str().into(),
        "--report-dir".into(),
        report.as_os_str().into(),
        "--checks".into(),
        task.checks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
            .into(),
        "--max-tokens".into(),
        task.max_tokens.to_string().into(),
        "--stop-prompt".into(),
        task.stop_prompt.clone().into(),
    ];
    if let Some(gguf) = &task.profile.gguf {
        words.extend(["--gguf-file".into(), gguf.filename.clone().into()]);
    }
    if let Some(capacity) = &task.runtime_capacity {
        words.extend([
            "--context-tokens".into(),
            capacity.context_tokens.to_string().into(),
            "--max-num-seqs".into(),
            capacity.max_num_seqs.to_string().into(),
        ]);
    }
    if task.disable_thinking {
        words.push("--disable-thinking".into());
    }
    if task.use_default_backend {
        words.push("--use-default-backend".into());
    }
    if task.reasoning_alias_replay {
        words.push("--reasoning-alias-replay".into());
    }
    words
}

// model_regression's Process::Drop kills and waits for Ferrum on normal Rust
// teardown. Killing the runner externally skips that Drop, so its descendants
// must inherit a separate process group owned by this controller.
#[cfg(unix)]
pub(super) struct ProcessGroup {
    pub(super) child: tokio::process::Child,
    pub(super) id: i32,
    pub(super) armed: bool,
}
#[cfg(unix)]
impl ProcessGroup {
    fn kill(&self) -> Result<(), String> {
        // SAFETY: id is the positive PID of our freshly spawned group leader;
        // the negative value addresses only that group, never the controller.
        let result = unsafe { libc::kill(-self.id, libc::SIGKILL) };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(format!("kill runner process group: {error}"))
        }
    }
    pub(super) async fn cleanup(&mut self) -> Result<(), String> {
        let killed = self.kill();
        let waited = tokio::time::timeout(Duration::from_secs(5), self.child.wait())
            .await
            .map_err(|_| "runner did not reap after process-group kill".to_string())
            .and_then(|value| value.map_err(|error| format!("reap runner: {error}")));
        if killed.is_ok() && waited.is_ok() {
            self.armed = false;
        }
        killed.and(waited.map(|_| ()))
    }
}
#[cfg(unix)]
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.kill();
            let _ = self.child.start_kill();
        }
    }
}

#[cfg(unix)]
pub(super) async fn run_command(
    mut command: Command,
    log: &Path,
    timeout: Duration,
) -> Result<(), String> {
    use std::os::unix::process::CommandExt;
    use tokio::signal::unix::{signal, SignalKind};
    // Register before spawning model work, rather than after a termination signal.
    let mut terminate = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
    let mut interrupt = signal(SignalKind::interrupt()).map_err(|error| error.to_string())?;
    command.as_std_mut().process_group(0);
    command
        .env_remove("VAST_API_KEY")
        .stdin(Stdio::null())
        .stdout(
            fs::File::create(log.with_extension("stdout.log"))
                .map_err(|error| error.to_string())?,
        )
        .stderr(
            fs::File::create(log.with_extension("stderr.log"))
                .map_err(|error| error.to_string())?,
        )
        .kill_on_drop(true);
    let child = command
        .spawn()
        .map_err(|error| format!("spawn model runner: {error}"))?;
    let id = child
        .id()
        .and_then(|id| i32::try_from(id).ok())
        .filter(|id| *id > 0)
        .ok_or("model runner omitted process-group ID")?;
    let mut group = ProcessGroup {
        child,
        id,
        armed: true,
    };
    let result = tokio::select! {
        waited=tokio::time::timeout(timeout,group.child.wait())=>match waited {
            Ok(Ok(status)) if status.success()=>Ok(()),
            Ok(Ok(status))=>Err(format!("model runner exited {status}; raw logs retained")),
            Ok(Err(error))=>Err(format!("wait model runner: {error}")),
            Err(_)=>Err("model runner task deadline exceeded".into()),
        },
        _=terminate.recv()=>Err("model runner interrupted by SIGTERM".into()),
        _=interrupt.recv()=>Err("model runner interrupted by SIGINT".into()),
    };
    // Also removes an accidentally orphaned model after an otherwise successful
    // runner exit. SIGKILL cannot invoke the runner's own Drop implementations.
    let cleanup = group.cleanup().await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => Err(format!("{error}; {cleanup}")),
    }
}
#[cfg(not(unix))]
pub(super) async fn run_command(
    _command: Command,
    _log: &Path,
    _timeout: Duration,
) -> Result<(), String> {
    Err("local model execution requires Unix process-group cleanup".into())
}

fn checked_report(path: &Path, task: &ExpectedModelRun) -> Result<Value, String> {
    let report = read_json(path)?;
    verify_model_report(task, &report).map_err(|issues| issues.join("; "))?;
    Ok(report)
}

pub async fn execute(mut args: LocalArgs) -> Result<(), String> {
    if args.task_timeout_secs == 0
        || tokio::time::Instant::now()
            .checked_add(Duration::from_secs(args.task_timeout_secs))
            .is_none()
    {
        return Err("task timeout must be positive and representable".into());
    }
    let document: PreparedTasks = serde_json::from_value(read_json(&args.tasks)?)
        .map_err(|error| format!("read model tasks: {error}"))?;
    let (selected, other, gaps) = select(document, args.backend)?;
    args.ferrum_bin = fs::canonicalize(&args.ferrum_bin)
        .map_err(|error| format!("resolve staged Ferrum: {error}"))?;
    args.runner = fs::canonicalize(&args.runner)
        .map_err(|error| format!("resolve staged runner: {error}"))?;
    let binary_sha = hash(&args.ferrum_bin)?;
    let runner_sha = hash(&args.runner)?;
    if args.runner_sha256.len() != 64 || !runner_sha.eq_ignore_ascii_case(&args.runner_sha256) {
        return Err("staged runner SHA-256 mismatch".into());
    }
    if selected
        .iter()
        .any(|task| !binary_sha.eq_ignore_ascii_case(&task.binary_sha256))
    {
        return Err("staged Ferrum SHA-256 differs from selected task".into());
    }
    if selected
        .windows(2)
        .any(|pair| pair[0].version != pair[1].version)
    {
        return Err("selected tasks disagree on candidate version".into());
    }
    fs::create_dir(&args.report_dir)
        .map_err(|error| format!("create fresh local report directory: {error}"))?;
    args.report_dir = fs::canonicalize(&args.report_dir).map_err(|error| error.to_string())?;
    let started = timestamp()?;
    let mut completed = Vec::new();
    let mut failure = None;
    for (index, task) in selected.iter().enumerate() {
        let outcome=async {
            let expected=args.report_dir.join(format!("task-{index}.json"));
            let report_dir=args.report_dir.join(format!("report-{index}"));
            write_json(&expected,task)?;
            let arguments=runner_arguments(&args,task,&expected,&report_dir);
            write_json(&args.report_dir.join(format!("command-{index}.json")),&json!({"program":args.runner,"args":arguments,"task_timeout_secs":args.task_timeout_secs}))?;
            let mut command=Command::new(&args.runner);command.args(&arguments);
            let process=run_command(command,&args.report_dir.join(format!("runner-{index}")),Duration::from_secs(args.task_timeout_secs)).await;
            let report=checked_report(&report_dir.join("report.json"),task);
            match (process,report) {
                (Ok(()),Ok(_))=>Ok(()),
                (Err(error),Ok(_)) | (Ok(()),Err(error))=>Err(error),
                (Err(process),Err(report))=>Err(format!("{process}; report: {report}")),
            }
        }.await;
        match outcome {
            Ok(()) => completed.push(task.profile.id.clone()),
            Err(error) => {
                failure = Some(format!("profile {}: {error}", task.profile.id));
                break;
            }
        }
    }
    let status = if failure.is_some() {
        "failed"
    } else if selected.is_empty() {
        "not_required"
    } else {
        "passed"
    };
    write_json(
        &args.report_dir.join("execution.json"),
        &json!({"schema_version":1,"status":status,"backend":args.backend.name(),
        "binary_sha256":binary_sha,"runner_sha256":runner_sha,"started_at":started,"finished_at":timestamp()?,
        "profiles":selected.iter().map(|task|&task.profile.id).collect::<Vec<_>>(),"completed_profiles":completed,
        "other_profiles":other,"remaining_plan_gaps":gaps,"error":failure,"release_approved":false}),
    )?;
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
#[path = "local_tests.rs"]
mod tests;
