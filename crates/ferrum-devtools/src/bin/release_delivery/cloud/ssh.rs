//! Bounded subprocess transport. Model assertions stay in model_regression.
use super::{api::Instance, write_json, ExecuteArgs};
use ferrum_bench_core::release_regression::model_tasks::{verify_model_report, ExpectedModelRun};
use serde_json::Value;
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::process::Command;

pub(super) fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
fn shell_command(words: &[String]) -> String {
    words
        .iter()
        .map(|word| quote(word))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) async fn command(
    program: &str,
    args: &[OsString],
    log: &Path,
    timeout: Duration,
) -> Result<String, String> {
    let mut child_command = Command::new(program);
    child_command.args(args);
    command_with(child_command, program, log, timeout).await
}

pub(super) async fn command_with(
    mut child_command: Command,
    program: &str,
    log: &Path,
    timeout: Duration,
) -> Result<String, String> {
    let stdout = fs::File::create(log.with_extension("stdout.log")).map_err(|e| e.to_string())?;
    let stderr = fs::File::create(log.with_extension("stderr.log")).map_err(|e| e.to_string())?;
    let mut child = child_command
        .env_remove("VAST_API_KEY")
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("start {program}: {e}"))?;
    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(result) => result.map_err(|e| format!("wait {program}: {e}"))?,
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(format!("{program} timed out; see {}", log.display()));
        }
    };
    if !status.success() {
        return Err(format!(
            "{program} failed ({status}); see {}",
            log.display()
        ));
    }
    let path = log.with_extension("stdout.log");
    if fs::metadata(&path).map_err(|e| e.to_string())?.len() > 1024 * 1024 {
        return Err(format!(
            "{program} control output exceeds 1 MiB; raw log preserved"
        ));
    }
    fs::read_to_string(path).map_err(|e| e.to_string())
}

pub(super) async fn keypair(directory: &Path) -> Result<PathBuf, String> {
    let key = directory.join("id_ed25519");
    command(
        "ssh-keygen",
        &[
            "-q".into(),
            "-t".into(),
            "ed25519".into(),
            "-N".into(),
            "".into(),
            "-C".into(),
            "ferrum-release-ephemeral".into(),
            "-f".into(),
            key.clone().into_os_string(),
        ],
        &directory.join("keygen"),
        Duration::from_secs(15),
    )
    .await?;
    Ok(key)
}

pub(super) struct Remote {
    host: String,
    port: u16,
    key: PathBuf,
    known_hosts: PathBuf,
    pub directory: String,
    #[cfg(test)]
    pub fixture: Option<fn(&str, &[OsString]) -> Result<String, String>>,
}
impl Remote {
    pub fn new(instance: &Instance, key: &Path, directory: String) -> Result<Self, String> {
        let host = instance
            .ssh_host
            .clone()
            .ok_or("Vast instance omitted ssh_host")?;
        if host.is_empty()
            || host.starts_with('-')
            || !host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-'))
        {
            return Err("Vast SSH hostname is not a safe DNS/IP host".into());
        }
        let port = instance
            .ssh_port
            .filter(|port| *port > 0)
            .ok_or("Vast instance omitted SSH port")?;
        Ok(Self {
            host,
            port,
            key: key.into(),
            known_hosts: key.with_file_name("known_hosts"),
            directory,
            #[cfg(test)]
            fixture: None,
        })
    }
    fn options(&self) -> Vec<OsString> {
        vec![
            "-i".into(),
            self.key.clone().into_os_string(),
            "-o".into(),
            "BatchMode=yes".into(),
            "-o".into(),
            "IdentitiesOnly=yes".into(),
            "-o".into(),
            "ConnectTimeout=10".into(),
            "-o".into(),
            "ServerAliveInterval=15".into(),
            "-o".into(),
            "ServerAliveCountMax=2".into(),
            "-o".into(),
            "StrictHostKeyChecking=accept-new".into(),
            "-o".into(),
            format!("UserKnownHostsFile={}", self.known_hosts.display()).into(),
        ]
    }
    async fn transport(
        &self,
        program: &str,
        args: &[OsString],
        log: &Path,
        timeout: Duration,
    ) -> Result<String, String> {
        #[cfg(test)]
        if let Some(fixture) = self.fixture {
            return fixture(program, args);
        }
        command(program, args, log, timeout).await
    }
    pub async fn exec(
        &self,
        words: &[String],
        log: &Path,
        timeout: Duration,
    ) -> Result<String, String> {
        let mut args = self.options();
        args.extend([
            "-T".into(),
            "-p".into(),
            self.port.to_string().into(),
            format!("root@{}", self.host).into(),
            shell_command(words).into(),
        ]);
        self.transport("ssh", &args, log, timeout).await
    }
    async fn copy(
        &self,
        local: &Path,
        remote: &str,
        download: bool,
        log: &Path,
    ) -> Result<(), String> {
        let mut args = self.options();
        args.extend(["-P".into(), self.port.to_string().into()]);
        let endpoint: OsString = format!("root@{}:{remote}", self.host).into();
        if download {
            args.extend(["-r".into(), "--".into(), endpoint, local.as_os_str().into()]);
        } else {
            args.extend(["--".into(), local.as_os_str().into(), endpoint]);
        }
        self.transport("scp", &args, log, Duration::from_secs(300))
            .await
            .map(|_| ())
    }
    pub async fn prepare(
        &self,
        args: &ExecuteArgs,
        runner_sha: &str,
        binary_sha: &str,
        version: &str,
    ) -> Result<(), String> {
        let logs = &args.report_dir;
        let gpu = self
            .exec(
                &[
                    "nvidia-smi".into(),
                    "--query-gpu=name,memory.total,compute_cap,driver_version".into(),
                    "--format=csv,noheader,nounits".into(),
                ],
                &logs.join("device"),
                Duration::from_secs(30),
            )
            .await?;
        validate_device(&gpu)?;
        let memory = self
            .exec(
                &[
                    "awk".into(),
                    "/MemAvailable:/ {print $2}".into(),
                    "/proc/meminfo".into(),
                ],
                &logs.join("memory"),
                Duration::from_secs(30),
            )
            .await?;
        let available_kib = memory
            .trim()
            .parse::<u64>()
            .map_err(|_| "cannot read host available RAM")?;
        if available_kib.saturating_mul(1024) < args.min_cpu_ram_mb.saturating_mul(1_000_000) {
            return Err("insufficient available host RAM".into());
        }
        // MemAvailable is host-visible; it does not prove a model fits a container
        // limit or VRAM. Actual runner failures remain mandatory failures.
        self.exec(
            &["mkdir".into(), "-p".into(), self.directory.clone()],
            &logs.join("mkdir"),
            Duration::from_secs(30),
        )
        .await?;
        let available = self
            .exec(
                &["df".into(), "-Pk".into(), self.directory.clone()],
                &logs.join("disk"),
                Duration::from_secs(30),
            )
            .await?;
        let free_kib = available
            .lines()
            .last()
            .and_then(|line| line.split_whitespace().nth(3))
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or("cannot read task disk availability")?;
        // Image/base occupancy is real: require a reviewable separate free-space floor.
        if free_kib < args.min_free_disk_gib.saturating_mul(1024 * 1024) {
            return Err("instance has insufficient free model/report disk space".into());
        }
        for (local, name, digest) in [
            (&args.ferrum_bin, "ferrum", binary_sha),
            (&args.runner, "model_regression", runner_sha),
        ] {
            let path = format!("{}/{name}", self.directory);
            self.copy(local, &path, false, &logs.join(format!("upload-{name}")))
                .await?;
            let hash = self
                .exec(
                    &["sha256sum".into(), path.clone()],
                    &logs.join(format!("hash-{name}")),
                    Duration::from_secs(30),
                )
                .await?;
            if hash.split_whitespace().next() != Some(digest) {
                return Err(format!("uploaded {name} digest differs"));
            }
            self.exec(
                &["chmod".into(), "700".into(), path.clone()],
                &logs.join(format!("chmod-{name}")),
                Duration::from_secs(30),
            )
            .await?;
            let dependencies = self
                .exec(
                    &["ldd".into(), path.clone()],
                    &logs.join(format!("ldd-{name}")),
                    Duration::from_secs(30),
                )
                .await?;
            if dependencies.contains("not found") {
                return Err(format!("{name} has unresolved runtime libraries"));
            }
            self.exec(
                &[path, "--help".into()],
                &logs.join(format!("help-{name}")),
                Duration::from_secs(30),
            )
            .await?;
        }
        let actual_version = self
            .exec(
                &[format!("{}/ferrum", self.directory), "--version".into()],
                &logs.join("ferrum-version"),
                Duration::from_secs(30),
            )
            .await?;
        if actual_version.trim() != format!("ferrum {version}") {
            return Err("staged Ferrum version differs from prepared tasks".into());
        }
        Ok(())
    }
    pub async fn run_task(
        &self,
        args: &ExecuteArgs,
        task: &ExpectedModelRun,
        index: usize,
    ) -> Result<Value, String> {
        let task_path = args.report_dir.join(format!("task-{index}.json"));
        write_json(&task_path, task)?;
        let remote_task = format!("{}/task-{index}.json", self.directory);
        self.copy(
            &task_path,
            &remote_task,
            false,
            &args.report_dir.join(format!("upload-task-{index}")),
        )
        .await?;
        let remote_report = format!("{}/report-{index}", self.directory);
        let words = runner_command(
            task,
            &self.directory,
            &remote_task,
            &remote_report,
            args.task_timeout_secs,
        );
        let run = self
            .exec(
                &words,
                &args.report_dir.join(format!("run-{index}")),
                Duration::from_secs(args.task_timeout_secs.saturating_add(30)),
            )
            .await;
        // Recover evidence even on runner failure; absent evidence is never success.
        let collection = self
            .copy(
                &args.report_dir,
                &remote_report,
                true,
                &args.report_dir.join(format!("collect-{index}")),
            )
            .await;
        collection?;
        let report_path = args.report_dir.join(format!("report-{index}/report.json"));
        let report: Value = serde_json::from_slice(
            &fs::read(&report_path).map_err(|e| format!("missing task report: {e}"))?,
        )
        .map_err(|e| format!("invalid task report: {e}"))?;
        run?;
        verify_model_report(task, &report).map_err(|issues| issues.join("; "))?;
        Ok(report)
    }
}

pub(super) fn runner_command(
    task: &ExpectedModelRun,
    directory: &str,
    task_file: &str,
    report: &str,
    timeout_secs: u64,
) -> Vec<String> {
    let mut words = vec![
        "timeout".into(),
        "--signal=TERM".into(),
        "--kill-after=15s".into(),
        format!("{timeout_secs}s"),
        format!("{directory}/model_regression"),
        "--ferrum-bin".into(),
        format!("{directory}/ferrum"),
        "--model".into(),
        task.profile.model.clone(),
        "--backend".into(),
        "cuda".into(),
        "--expected-task".into(),
        task_file.into(),
        "--profile-id".into(),
        task.profile.id.clone(),
        "--report-dir".into(),
        report.into(),
        "--checks".into(),
        task.checks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
        "--max-tokens".into(),
        task.max_tokens.to_string(),
        "--stop-prompt".into(),
        task.stop_prompt.clone(),
    ];
    if let Some(capacity) = &task.runtime_capacity {
        words.extend([
            "--context-tokens".into(),
            capacity.context_tokens.to_string(),
            "--max-num-seqs".into(),
            capacity.max_num_seqs.to_string(),
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

pub(super) fn validate_device(text: &str) -> Result<(), String> {
    let rows: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if rows.len() != 1 {
        return Err("expected exactly one visible GPU".into());
    }
    let fields: Vec<_> = rows[0].split(',').map(str::trim).collect();
    if fields.len() != 4 {
        return Err("invalid nvidia-smi device record".into());
    }
    let memory = fields[1].parse::<u64>().map_err(|_| "invalid GPU memory")?;
    if !matches!(
        fields[0],
        "NVIDIA RTX 6000 Ada Generation" | "NVIDIA L40" | "NVIDIA L40S"
    ) || !(45000..=50000).contains(&memory)
        || fields[2] != "8.9"
    {
        return Err("actual device does not match native 48 GB sm89 execution requirement".into());
    }
    let driver = fields[3]
        .split('.')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "invalid GPU driver")?;
    if driver.len() != 3 || driver.as_slice() < [550, 54, 14].as_slice() {
        return Err("actual CUDA driver is older than 550.54.14".into());
    }
    Ok(())
}
