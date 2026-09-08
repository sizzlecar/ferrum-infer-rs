use super::{evidence, layout::Selection, model_io, process_identity, save, Args};
use anyhow::{ensure, Context, Result};
use ferrum_bench_core::release_regression::model_basic::ARITHMETIC_PROMPT;
use serde_json::{json, Value};
use std::{
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

fn command(args: &Args, selected: &Selection, entry: &str) -> Result<Command> {
    let mut command = Command::new(&selected.launcher_path);
    command.args([
        entry,
        args.model
            .as_ref()
            .context("live upgrade requires --model")?
            .to_str()
            .context("non-Unicode model path")?,
    ]);
    command.args([
        "--kv-capacity",
        &args.context_tokens.to_string(),
        "--max-model-len",
        &args.context_tokens.to_string(),
        "--max-num-seqs",
        "1",
    ]);
    command.current_dir(&args.report_dir).env("NO_COLOR", "1");
    // Match model_regression's standard default-backend invocation policy.
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("FERRUM_") {
            command.env_remove(key);
        }
    }
    Ok(command)
}

fn spawn(args: &Args, mut command: Command, name: &str) -> Result<Owned> {
    save(
        &args.report_dir,
        &format!("{name}.command.json"),
        &json!({"program":command.get_program(),"args":command.get_args().collect::<Vec<_>>(),"cwd":args.report_dir,"environment_policy":"remove_inherited_ferrum_overrides"}),
    )?;
    let child = command
        .stdin(Stdio::null())
        .stdout(fs::File::create(
            args.report_dir.join(format!("{name}.stdout.log")),
        )?)
        .stderr(fs::File::create(
            args.report_dir.join(format!("{name}.stderr.log")),
        )?)
        .spawn()?;
    let owned = Owned {
        child,
        core: None,
        cleanup_path: args.report_dir.join(format!("{name}.cleanup.json")),
    };
    save(
        &args.report_dir,
        &format!("{name}.process.json"),
        &json!({"launcher_pid":owned.child.id(),"started_at":chrono::Utc::now().to_rfc3339()}),
    )?;
    Ok(owned)
}

struct Owned {
    child: Child,
    core: Option<process_identity::Process>,
    cleanup_path: PathBuf,
}
impl Owned {
    fn alive(&mut self) -> Result<()> {
        ensure!(
            self.child.try_wait()?.is_none(),
            "owned launcher exited during upgrade"
        );
        if let Some(core) = &self.core {
            core.alive()?;
        }
        Ok(())
    }
    fn observe(&mut self, expected: &Path) -> Result<bool> {
        self.alive()?;
        if self.core.is_none() {
            self.core = process_identity::child(self.child.id(), expected)?;
        }
        Ok(self.core.is_some())
    }
    fn stop(&mut self) -> Result<Value> {
        if self.child.try_wait()?.is_none() {
            self.child.kill()?;
        }
        let status = self.child.wait()?;
        if let Some(core) = &self.core {
            core.wait_for_exit(10_000)?;
        }
        Ok(
            json!({"launcher_pid":self.child.id(),"launcher_exit_code":status.code(),"core":self.core.as_ref().map(|c|&c.identity),"core_terminated":self.core.as_ref().map(|_|true),"method":"terminate_owned_launcher_and_wait_for_job_children"}),
        )
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        let result = self.stop();
        let evidence = match result {
            Ok(value) => value,
            Err(error) => json!({"launcher_pid":self.child.id(),"error":format!("{error:#}")}),
        };
        if let Ok(bytes) = serde_json::to_vec_pretty(&evidence) {
            let _ = fs::write(&self.cleanup_path, bytes);
        }
    }
}

pub struct Session {
    owned: Owned,
    runtime: tokio::runtime::Runtime,
    client: reqwest::Client,
    url: String,
    report_dir: PathBuf,
    max_tokens: u32,
    version: String,
    timeout: Duration,
}

async fn infer(
    client: &reqwest::Client,
    url: &str,
    directory: &Path,
    name: &str,
    max_tokens: u32,
) -> Result<Value> {
    let body = json!({"model":"upgrade-regression","messages":[{"role":"user","content":ARITHMETIC_PROMPT}],"max_tokens":max_tokens,"temperature":0,"seed":7,"stream":false});
    save(directory, &format!("{name}.request.json"), &body)?;
    let started = chrono::Utc::now().to_rfc3339();
    let response = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    let text = response.text().await?;
    fs::write(directory.join(format!("{name}.response.txt")), &text)?;
    ensure!(status.is_success(), "old session HTTP {status}");
    let observed = model_io::chat(&text)?;
    ensure!(
        observed["usage"]["completion_tokens"]
            .as_u64()
            .is_some_and(|n| n <= u64::from(max_tokens)),
        "old inference exceeded the requested token budget"
    );
    Ok(
        json!({"started_at":started,"finished_at":chrono::Utc::now().to_rfc3339(),"http_status":status.as_u16(),"observation":observed}),
    )
}

impl Session {
    pub fn start(args: &Args, selected: &Selection, version: &str) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let mut command = command(args, selected, "serve")?;
        command.args([
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--served-model-name",
            "upgrade-regression",
        ]);
        let owned = spawn(args, command, "old-serve")?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(args.timeout_secs))
            .build()?;
        let mut session = Self {
            owned,
            runtime,
            client,
            url: format!("http://127.0.0.1:{port}"),
            report_dir: args.report_dir.clone(),
            max_tokens: args.max_tokens,
            version: version.into(),
            timeout: Duration::from_secs(args.timeout_secs),
        };
        let start = Instant::now();
        loop {
            session.owned.observe(&selected.core_path)?;
            let health = session.runtime.block_on(model_io::health(
                &session.client,
                &session.url,
                Duration::from_secs(2),
            ));
            if let Ok(health) = health {
                ensure!(
                    health["status"] == "healthy"
                        && health["version"] == version
                        && health["auto_config"]["hardware_capabilities"]["backend"] == "cuda",
                    "old server health/version/backend mismatch"
                );
                ensure!(
                    session.owned.core.is_some(),
                    "healthy service is not the owned launcher's versioned core"
                );
                save(&args.report_dir, "old-serve.health.json", &health)?;
                break;
            }
            ensure!(
                start.elapsed() < Duration::from_secs(args.startup_timeout_secs),
                "old serve startup timed out"
            );
            thread::sleep(Duration::from_millis(200));
        }
        Ok(session)
    }

    pub fn proof(&mut self, name: &str) -> Result<Value> {
        self.owned.alive()?;
        let observed = self.runtime.block_on(infer(
            &self.client,
            &self.url,
            &self.report_dir,
            name,
            self.max_tokens,
        ))?;
        self.owned.alive()?;
        let health =
            self.runtime
                .block_on(model_io::health(&self.client, &self.url, self.timeout))?;
        ensure!(
            health["version"] == self.version && health["status"] == "healthy",
            "old process stopped serving its original version"
        );
        Ok(
            json!({"launcher_pid":self.owned.child.id(),"core":self.owned.core.as_ref().map(|c|&c.identity),"inference":observed,"health":health}),
        )
    }

    pub fn while_upgrading(&mut self, action: impl FnOnce() -> Result<()>) -> Result<Value> {
        self.owned.alive()?;
        let (url, directory, budget, timeout) = (
            self.url.clone(),
            self.report_dir.clone(),
            self.max_tokens,
            self.timeout,
        );
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || -> Result<Vec<Value>> {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            // Connections in self.client belong to the paused main-thread runtime.
            // Keep the upgrade worker's connection pool on its own active runtime.
            let client = reqwest::Client::builder()
                .no_proxy()
                .timeout(timeout)
                .build()?;
            let mut observations = Vec::new();
            while !worker_stop.load(Ordering::Acquire) {
                let result = runtime.block_on(infer(
                    &client,
                    &url,
                    &directory,
                    &format!("during-upgrade-{}", observations.len()),
                    budget,
                ));
                if observations.is_empty() {
                    let _ = sender.send(result.is_ok());
                }
                observations.push(result?);
                thread::sleep(Duration::from_millis(250));
            }
            Ok(observations)
        });
        let ready = receiver.recv_timeout(self.timeout);
        let action = if ready == Ok(true) {
            action()
        } else {
            Err(anyhow::anyhow!(
                "old inference did not start before upgrade: {ready:?}"
            ))
        };
        stop.store(true, Ordering::Release);
        let observed = worker
            .join()
            .map_err(|_| anyhow::anyhow!("upgrade inference worker panicked"))?;
        let installer: Value = fs::read(self.report_dir.join("upgrade.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or(Value::Null);
        let evidence = json!({"installer":installer,"installer_succeeded":action.is_ok(),"core":self.owned.core.as_ref().map(|c|&c.identity),"inference_error":observed.as_ref().err().map(|e|format!("{e:#}")),"observations":observed.as_ref().ok()});
        save(&self.report_dir, "during-upgrade.json", &evidence)?;
        action?;
        let observations = observed?;
        model_io::require_overlap(
            &observations,
            installer["started_at"]
                .as_str()
                .context("installer start missing")?,
            installer["finished_at"]
                .as_str()
                .context("installer finish missing")?,
        )?;
        self.owned.alive()?;
        Ok(evidence)
    }

    pub fn stop(mut self) -> Result<Value> {
        let result = self.owned.stop()?;
        save(&self.report_dir, "old-serve.cleanup.json", &result)?;
        Ok(result)
    }
}

pub fn new_run(args: &Args, selected: &Selection) -> Result<Value> {
    let mut command = command(args, selected, "run")?;
    command.args([
        "--prompt",
        ARITHMETIC_PROMPT,
        "--output-format",
        "jsonl",
        "--temperature",
        "0",
        "--seed",
        "7",
        "--max-tokens",
        &args.max_tokens.to_string(),
    ]);
    let mut owned = spawn(args, command, "new-run")?;
    let start = Instant::now();
    loop {
        if let Some(status) = owned.child.try_wait()? {
            ensure!(status.success(), "new run failed: {status}");
            break;
        }
        if owned.core.is_none() {
            owned.observe(&selected.core_path)?;
        }
        ensure!(
            start.elapsed()
                < Duration::from_secs(args.startup_timeout_secs.saturating_add(args.timeout_secs)),
            "new run timed out"
        );
        thread::sleep(Duration::from_millis(50));
    }
    let core = owned
        .core
        .as_ref()
        .context("new run did not execute an observed versioned core child")?;
    core.wait_for_exit(10_000)?;
    ensure!(
        evidence::digest(&selected.core_path)? == selected.core,
        "new core bytes changed during inference"
    );
    let observed = model_io::run(
        &fs::read_to_string(args.report_dir.join("new-run.stdout.log"))?,
        args.model
            .as_ref()
            .unwrap()
            .to_str()
            .context("model path")?,
    )?;
    ensure!(
        observed["assistant"]["usage"]["completion_tokens"]
            .as_u64()
            .is_some_and(|n| n <= u64::from(args.max_tokens)),
        "new run exceeded the requested token budget"
    );
    Ok(
        json!({"launcher_pid":owned.child.id(),"core":core.identity,"core_sha256":selected.core.sha256,"observation":observed}),
    )
}
