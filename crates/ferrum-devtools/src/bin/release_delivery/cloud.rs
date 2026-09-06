//! One-provider, one-GPU execution of already prepared model tasks.
//! Quotes/deadlines are bounded policy inputs, not a provider-enforced invoice cap.
#[path = "cloud/api.rs"]
mod api;
#[path = "cloud/reaper.rs"]
mod reaper;
#[path = "cloud/ssh.rs"]
mod ssh;
pub use reaper::{reap, ReapArgs};

use clap::Args;
use ferrum_bench_core::release_regression::{
    model_tasks::{verify_model_options, verify_model_reports, ExpectedModelRun},
    Backend, Gap,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::time::Instant;

const IMAGE: &str =
    "vastai/base-image@sha256:d7544432c3fe9951e13b740c42ddfcec3ec71961d158da7f88495015e9f564ad";

#[derive(Debug, Args)]
pub struct ExecuteArgs {
    #[arg(long)]
    pub tasks: PathBuf,
    /// Archive whose expected SHA comes from trusted staging metadata.
    #[arg(long)]
    pub archive: PathBuf,
    #[arg(long)]
    pub archive_sha256: String,
    /// Binary extracted from that archive; caller must verify archive membership.
    #[arg(long)]
    pub ferrum_bin: PathBuf,
    #[arg(long)]
    pub runner: PathBuf,
    #[arg(long)]
    pub runner_sha256: String,
    /// Must not exist. Includes raw reports, sanitized lifecycle and SSH logs.
    #[arg(long)]
    pub report_dir: PathBuf,
    #[arg(long)]
    pub repository_id: u64,
    #[arg(long)]
    pub run_id: u64,
    #[arg(long)]
    pub attempt: u32,
    #[arg(long)]
    pub disk_gib: u64,
    #[arg(long)]
    pub min_free_disk_gib: u64,
    #[arg(long)]
    pub min_cpu_ram_mb: u64,
    #[arg(long)]
    pub max_hourly_usd: f64,
    #[arg(long)]
    pub max_network_usd_per_gb: f64,
    /// Total execution deadline, including creation/pull/downloads, before cleanup.
    #[arg(long)]
    pub lease_secs: u64,
    #[arg(long)]
    pub bootstrap_secs: u64,
    #[arg(long)]
    pub task_timeout_secs: u64,
    /// First implementation permits exactly one create request, never retries it.
    #[arg(long)]
    pub max_create_attempts: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedTasks {
    schema_version: u32,
    expectations: Vec<ExpectedModelRun>,
    unsupported_obligations: Vec<usize>,
    remaining_plan_gaps: Vec<Gap>,
}
#[derive(Debug, PartialEq, Eq)]
struct Ownership {
    repository: u64,
    run: u64,
    attempt: u32,
    expiry: u64,
    nonce: String,
}
impl Ownership {
    fn label(&self) -> String {
        format!(
            "ferrum-rel:1:r{}:w{}:a{}:x{}:n{}",
            self.repository, self.run, self.attempt, self.expiry, self.nonce
        )
    }
    fn parse(label: &str) -> Option<Self> {
        let parts: Vec<_> = label.split(':').collect();
        if parts.len() != 7 || parts[0] != "ferrum-rel" || parts[1] != "1" {
            return None;
        }
        fn number<T: std::str::FromStr>(s: &str, prefix: char) -> Option<T> {
            let s = s.strip_prefix(prefix)?;
            if s.is_empty() || s.starts_with('0') || !s.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            s.parse().ok()
        }
        let nonce = parts[6].strip_prefix('n')?;
        if nonce.len() != 32
            || !nonce
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return None;
        }
        Some(Self {
            repository: number(parts[2], 'r')?,
            run: number(parts[3], 'w')?,
            attempt: number(parts[4], 'a')?,
            expiry: number(parts[5], 'x')?,
            nonce: nonce.into(),
        })
    }
}
fn now() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_secs())
        .map_err(|e| e.to_string())
}
fn sha256(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut bytes = [0u8; 65536];
    loop {
        let count = file.read(&mut bytes).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&bytes[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn check_digest(path: &Path, expected: &str) -> Result<String, String> {
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid input SHA-256 digest".into());
    }
    let actual = sha256(path)?;
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(format!("input digest mismatch: {}", path.display()));
    }
    Ok(actual)
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {}: {e}", path.display()))
}
fn cuda_tasks(
    document: PreparedTasks,
) -> Result<(Vec<ExpectedModelRun>, Vec<Gap>, Vec<String>), String> {
    if document.schema_version != 1 || !document.unsupported_obligations.is_empty() {
        return Err(
            "unsupported model task schema or obligations; do not allocate hardware".into(),
        );
    }
    let mut ids = BTreeSet::new();
    let mut cuda = Vec::new();
    let mut other = Vec::new();
    for task in document.expectations {
        if !ids.insert(task.profile.id.clone()) {
            return Err("duplicate expected profile ID".into());
        }
        if task.profile.target.backend != Backend::Cuda {
            other.push(task.profile.id);
            continue;
        }
        if !task.profile.available {
            return Err(
                "CUDA task profile is unavailable; do not replace its execution target".into(),
            );
        }
        verify_model_options(&task,&json!({"profile_id":task.profile.id,"model":task.profile.model,"backend":"cuda",
            "stop_prompt":task.stop_prompt,"disable_thinking":task.disable_thinking,"use_default_backend":task.use_default_backend,
            "context_tokens":task.runtime_capacity.as_ref().map(|capacity|capacity.context_tokens),
            "max_num_seqs":task.runtime_capacity.as_ref().map(|capacity|capacity.max_num_seqs),
            "reasoning_alias_replay":task.reasoning_alias_replay,"max_tokens":task.max_tokens,"checks":task.checks}))
            .map_err(|issues|issues.join("; "))?;
        cuda.push(task);
    }
    Ok((cuda, document.remaining_plan_gaps, other))
}
fn validate_args(args: &ExecuteArgs) -> Result<(), String> {
    if args.repository_id == 0
        || args.run_id == 0
        || args.attempt == 0
        || args.max_create_attempts != 1
    {
        return Err("positive repo/run/attempt and max-create-attempts=1 are required".into());
    }
    if args.disk_gib == 0
        || args.min_free_disk_gib == 0
        || args.min_free_disk_gib > args.disk_gib
        || args.min_cpu_ram_mb == 0
        || args.lease_secs == 0
        || args.bootstrap_secs == 0
        || args.bootstrap_secs >= args.lease_secs
        || args.task_timeout_secs == 0
    {
        return Err(
            "explicit positive capacity/deadline limits are required; bootstrap must fit lease"
                .into(),
        );
    }
    if !args.max_hourly_usd.is_finite()
        || args.max_hourly_usd <= 0.0
        || !args.max_network_usd_per_gb.is_finite()
        || args.max_network_usd_per_gb < 0.0
    {
        return Err("invalid explicit hourly/network price limits".into());
    }
    for seconds in [args.lease_secs, args.bootstrap_secs, args.task_timeout_secs] {
        if Instant::now()
            .checked_add(Duration::from_secs(seconds.saturating_add(30)))
            .is_none()
        {
            return Err("deadline exceeds platform timer range".into());
        }
    }
    Ok(())
}

struct PrivateKey(PathBuf);
impl Drop for PrivateKey {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
        let _ = fs::remove_file(self.0.with_extension("pub"));
    }
}
async fn interrupted() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                tokio::select! { _=signal.recv()=>{}, _=tokio::signal::ctrl_c()=>{} }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// CREATE is never retried. A lost response is reconciled by the exact ownership
/// label, and any discovered IDs are registered for cleanup before further work.
async fn acquire(
    client: &api::Client,
    offer: u64,
    args: &ExecuteArgs,
    label: &str,
    owned: &mut Vec<u64>,
) -> Result<u64, String> {
    match client.create(offer, args.disk_gib, label, IMAGE).await {
        Ok(id) => {
            owned.push(id);
            Ok(id)
        }
        Err(create_error) => {
            for attempt in 0..3 {
                match client.instances().await {
                    Ok(instances) => {
                        for instance in instances
                            .into_iter()
                            .filter(|instance| instance.label.as_deref() == Some(label))
                        {
                            if !owned.contains(&instance.id) {
                                owned.push(instance.id);
                            }
                        }
                        match owned.as_slice() {
                            [id]=>return Ok(*id),
                            []=>{},
                            _=>return Err("ambiguous create produced multiple owned instances; cleaning up all, no execution".into()),
                        }
                    }
                    Err(_) => {}
                }
                if attempt < 2 {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
            Err(format!("{create_error}; no unique instance reconciled; intent retained and expiry reaper required"))
        }
    }
}
async fn destroy_confirm(client: &api::Client, id: u64) -> Result<(), String> {
    let destruction = client.destroy(id).await;
    for attempt in 0..6 {
        match client.instances().await {
            Ok(instances) if !instances.iter().any(|instance| instance.id == id) => return Ok(()),
            Ok(_) => {}
            Err(_) => {}
        }
        if attempt < 5 {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    }
    Err(format!(
        "instance {id} destruction not confirmed{}",
        destruction
            .err()
            .map(|e| format!(": {e}"))
            .unwrap_or_default()
    ))
}

async fn run_lease(
    client: &api::Client,
    args: &ExecuteArgs,
    tasks: &[ExpectedModelRun],
    key: &Path,
    label: &str,
    owned: &mut Vec<u64>,
    runner_sha: &str,
    binary_sha: &str,
) -> Result<Vec<Value>, String> {
    let offers = client.offers(args).await?;
    write_json(&args.report_dir.join("offers.json"), &offers)?;
    let offer = offers
        .first()
        .ok_or("no verified native 48 GB sm89 offer within explicit resource/price limits")?;
    write_json(
        &args.report_dir.join("create-intent.json"),
        &json!({"schema_version":1,"label":label,"offer":offer,"image":IMAGE,
        "disk_gib":args.disk_gib,"created_at":now()?,"lease_secs":args.lease_secs,"max_create_attempts":1}),
    )?;
    let boot = Instant::now() + Duration::from_secs(args.bootstrap_secs);
    let remote = tokio::time::timeout_at(boot, async {
        let id = acquire(client, offer.id, args, label, owned).await?;
        write_json(
            &args.report_dir.join("lease.json"),
            &json!({"instance_id":id,"label":label,"image":IMAGE}),
        )?;
        let public_key =
            fs::read_to_string(key.with_extension("pub")).map_err(|e| e.to_string())?;
        client.attach(id, public_key.trim()).await?;
        let directory = format!(
            "/workspace/ferrum-release-{}-{}-{}",
            args.repository_id, args.run_id, args.attempt
        );
        loop {
            if let Some(instance) = client.instance(id).await? {
                if instance.label.as_deref() != Some(label) {
                    return Err("instance ownership label changed".into());
                }
                if instance.actual_status.as_deref() == Some("running") {
                    if let Ok(remote) = ssh::Remote::new(&instance, key, directory.clone()) {
                        if remote
                            .exec(
                                &["true".into()],
                                &args.report_dir.join("ssh-ready"),
                                Duration::from_secs(15),
                            )
                            .await
                            .is_ok()
                        {
                            break Ok::<_, String>(remote);
                        }
                    }
                }
            } else {
                return Err("created instance disappeared before execution".into());
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    })
    .await
    .map_err(|_| "instance bootstrap/SSH deadline exceeded".to_string())??;
    remote
        .prepare(args, runner_sha, binary_sha, &tasks[0].version)
        .await?;
    let mut reports = Vec::new();
    for (index, task) in tasks.iter().enumerate() {
        reports.push(remote.run_task(args, task, index).await?);
    }
    verify_model_reports(tasks, &reports).map_err(|issues| issues.join("; "))?;
    Ok(reports)
}

pub async fn execute(args: ExecuteArgs) -> Result<(), String> {
    validate_args(&args)?;
    let document: PreparedTasks =
        serde_json::from_slice(&fs::read(&args.tasks).map_err(|e| e.to_string())?)
            .map_err(|e| format!("read prepared model tasks: {e}"))?;
    let (tasks, gaps, other_profiles) = cuda_tasks(document)?;
    let archive_sha = check_digest(&args.archive, &args.archive_sha256)?;
    let runner_sha = check_digest(&args.runner, &args.runner_sha256)?;
    let binary_sha = sha256(&args.ferrum_bin)?;
    if tasks
        .iter()
        .any(|task| !task.binary_sha256.eq_ignore_ascii_case(&binary_sha))
    {
        return Err("CUDA expectations do not identify the staged input binary".into());
    }
    if tasks
        .windows(2)
        .any(|pair| pair[0].version != pair[1].version)
    {
        return Err("CUDA tasks disagree on candidate version".into());
    }
    fs::create_dir(&args.report_dir)
        .map_err(|e| format!("create fresh cloud report directory: {e}"))?;
    if tasks.is_empty() {
        return write_json(
            &args.report_dir.join("execution.json"),
            &json!({"schema_version":1,"status":"not_required","cuda_tasks":0,"other_profiles":other_profiles,"remaining_plan_gaps":gaps,"release_approved":false}),
        );
    }
    let client = api::Client::new(
        std::env::var("VAST_API_KEY").map_err(|_| "VAST_API_KEY is required on the controller")?,
    )?;
    let private = args.report_dir.join("private");
    fs::create_dir(&private).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&private, fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
    }
    let _private_key = PrivateKey(private.join("id_ed25519"));
    let key = ssh::keypair(&private).await?;
    let ownership = Ownership {
        repository: args.repository_id,
        run: args.run_id,
        attempt: args.attempt,
        expiry: now()?
            .checked_add(args.lease_secs)
            .ok_or("lease timestamp overflow")?,
        nonce: format!("{:032x}", rand::random::<u128>()),
    };
    let label = ownership.label();
    let mut owned = Vec::new();
    let started = now()?;
    let result = {
        let execution = tokio::time::timeout(
            Duration::from_secs(args.lease_secs),
            run_lease(
                &client,
                &args,
                &tasks,
                &key,
                &label,
                &mut owned,
                &runner_sha,
                &binary_sha,
            ),
        );
        tokio::pin!(execution);
        tokio::select! {
            value=&mut execution=>value.map_err(|_|"total lease execution deadline exceeded".to_string()).and_then(|value|value),
            _=interrupted()=>Err("controller interrupted; recovering evidence already collected and cleaning up".into()),
        }
    };
    // No early `?` after creation: cleanup is attempted for success and failure.
    let mut cleanup_errors = Vec::new();
    if owned.is_empty() && args.report_dir.join("create-intent.json").exists() {
        // Cancellation can drop a pending CREATE response before acquire records
        // its ID. Reconcile again, without another allocation request.
        match client.instances().await {
            Ok(instances) => owned.extend(
                instances
                    .into_iter()
                    .filter(|instance| instance.label.as_deref() == Some(&label))
                    .map(|instance| instance.id),
            ),
            Err(error) => {
                cleanup_errors.push(format!("post-interruption reconciliation failed: {error}"))
            }
        }
        if owned.is_empty() {
            cleanup_errors.push("create outcome remains unconfirmed; exact expiry label requires the independent reaper".into());
        }
    }
    for id in &owned {
        if let Err(error) =
            tokio::time::timeout(Duration::from_secs(120), destroy_confirm(&client, *id))
                .await
                .map_err(|_| format!("instance {id} cleanup timed out"))
                .and_then(|value| value)
        {
            cleanup_errors.push(error);
        }
    }
    let _ = fs::remove_file(&key);
    let _ = fs::remove_file(key.with_extension("pub"));
    let error = result.as_ref().err().cloned();
    let passed = error.is_none() && cleanup_errors.is_empty();
    write_json(
        &args.report_dir.join("execution.json"),
        &json!({"schema_version":1,"status":if passed {"passed"} else {"failed"},
        "label":label,"instance_ids":owned,"archive_sha256":archive_sha,"runner_sha256":runner_sha,"binary_sha256":binary_sha,
        "started_at":started,"finished_at":now()?,"profiles":tasks.iter().map(|task|&task.profile.id).collect::<Vec<_>>(),
        "other_profiles":other_profiles,"remaining_plan_gaps":gaps,"error":error,"cleanup_errors":cleanup_errors,"release_approved":false}),
    )?;
    if !passed {
        return Err(format!(
            "cloud execution or cleanup failed; see {}",
            args.report_dir.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "cloud/tests.rs"]
mod tests;
