use super::{identity, kv, write_json, Args};
use anyhow::{bail, ensure, Context, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub(super) enum ApiEndpoint {
    ChatCompletions,
    Responses,
}

impl ApiEndpoint {
    fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Responses => "/v1/responses",
        }
    }
}

struct Process {
    child: Child,
    stdout: PathBuf,
    stderr: PathBuf,
}

// Only the child command is changed. Do not clear unrelated runtime, cache,
// authentication, or loader configuration, and never record environment values.
fn isolate_product_environment(
    command: &mut Command,
    use_default_backend: bool,
    inherited_keys: impl IntoIterator<Item = OsString>,
) -> Vec<String> {
    let mut removed = Vec::new();
    if use_default_backend {
        for key in inherited_keys {
            if key.to_string_lossy().starts_with("FERRUM_") {
                command.env_remove(&key);
                removed.push(key.to_string_lossy().into_owned());
            }
        }
    }
    removed.sort();
    removed.dedup();
    removed
}

impl Process {
    fn spawn(args: &Args, name: &str, mut argv: Vec<String>, stdin: Option<&str>) -> Result<Self> {
        let stdout = args.report_dir.join(format!("{name}.stdout.txt"));
        let stderr = args.report_dir.join(format!("{name}.stderr.txt"));
        if args.checks.contains(&super::Check::Observability)
            && matches!(argv.first().map(String::as_str), Some("run" | "serve"))
        {
            argv.extend([
                "--profile-detail".into(),
                "debug".into(),
                "--profile-jsonl".into(),
                args.report_dir
                    .join(format!("{name}.profile.jsonl"))
                    .to_string_lossy()
                    .into_owned(),
                "--scheduler-trace-jsonl".into(),
                args.report_dir
                    .join(format!("{name}.scheduler.jsonl"))
                    .to_string_lossy()
                    .into_owned(),
            ]);
        }
        let mut command = Command::new(&args.ferrum_bin);
        let removed_environment_keys = isolate_product_environment(
            &mut command,
            args.use_default_backend,
            std::env::vars_os().map(|(key, _)| key),
        );
        write_json(
            args.report_dir.join(format!("{name}.command.json")),
            &json!({
                "program": args.ferrum_bin, "args": argv, "cwd": args.report_dir,
                "environment_overrides": {"NO_COLOR": "1"},
                "removed_environment_keys": removed_environment_keys
            }),
        )?;
        if let Some(input) = stdin {
            fs::write(args.report_dir.join(format!("{name}.stdin.txt")), input)?;
        }
        let child = command
            .args(argv)
            .current_dir(&args.report_dir)
            .env("NO_COLOR", "1")
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::from(File::create(&stdout)?))
            .stderr(Stdio::from(File::create(&stderr)?))
            .spawn()
            .with_context(|| format!("spawn {}", args.ferrum_bin.display()))?;
        // Own the child before any fallible setup or health polling.
        let mut process = Self {
            child,
            stdout,
            stderr,
        };
        if let Some(input) = stdin {
            process
                .child
                .stdin
                .take()
                .context("child stdin")?
                .write_all(input.as_bytes())?;
        }
        Ok(process)
    }

    async fn wait(&mut self, timeout: Duration) -> Result<String> {
        let status = self.wait_status(timeout).await?;
        ensure!(
            status.success(),
            "child exited {status}; stderr: {}",
            self.stderr.display()
        );
        fs::read_to_string(&self.stdout).context("read child UTF-8 stdout")
    }

    async fn wait_status(&mut self, timeout: Duration) -> Result<ExitStatus> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            ensure!(
                started.elapsed() < timeout,
                "child exceeded {timeout:?}; logs: {}, {}",
                self.stdout.display(),
                self.stderr.display()
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    #[cfg(unix)]
    async fn terminate_and_wait(&mut self, timeout: Duration) -> Result<Value> {
        let mut evidence = json!({
            "method": "SIGTERM", "child_pid": self.child.id(),
            "signal_sent": false, "exit_code": null,
            "stdout": self.stdout, "stderr": self.stderr,
            "timeout_ms": timeout.as_millis()
        });
        let result = async {
            if let Some(status) = self.child.try_wait()? {
                evidence["exit_code"] = json!(status.code());
                bail!("server exited before graceful shutdown was requested: {status}");
            }
            let pid = libc::pid_t::try_from(self.child.id()).context("child PID exceeds pid_t")?;
            // The original child is still owned and unreaped. It cannot have
            // its PID reused while this handle awaits its actual exit.
            if unsafe { libc::kill(pid, libc::SIGTERM) } != 0 {
                return Err(std::io::Error::last_os_error())
                    .context("signal original server child");
            }
            evidence["signal_sent"] = json!(true);
            let status = self.wait_status(timeout).await?;
            evidence["exit_code"] = json!(status.code());
            ensure!(
                status.success(),
                "server graceful shutdown exited {status}; stderr: {}",
                self.stderr.display()
            );
            Ok(())
        }
        .await;
        match result {
            Ok(()) => Ok(evidence),
            Err(error) => Err(super::case_failure(evidence, error)),
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return;
        }
        // Failed setup, timed-out observation, and unsupported-platform
        // cleanup keep the original child owner until it is reaped. This is
        // not graceful-shutdown evidence.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(super) async fn run(
    args: &Args,
    name: &str,
    mut argv: Vec<String>,
    stdin: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let configuration_evidence = if argv.first().is_some_and(|entrypoint| entrypoint == "run")
        && (args.runtime_capacity().is_some()
            || args.kv_dtype.is_some()
            || args.sequence_fit_policy.is_some()
            || args.max_num_batched_tokens.is_some()
            || identity::requires_source_evidence(args)?)
    {
        let path = args
            .report_dir
            .join(format!("{name}.effective-config.json"));
        argv.extend([
            "--effective-config-json".into(),
            path.to_string_lossy().into_owned(),
        ]);
        Some(path)
    } else {
        None
    };
    let stdout = Process::spawn(args, name, argv, stdin)?
        .wait(timeout)
        .await?;
    if let Some(path) = configuration_evidence {
        // This is the configuration written by the same product process that
        // generated stdout, not a separate inspection or a copy of task inputs.
        let config: Value = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )
        .context("parse run effective configuration")?;
        if let Some(capacity) = args.runtime_capacity() {
            capacity
                .verify_health(&json!({"auto_config": &config}))
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("{name} actual run capacity"))?;
        }
        identity::validate_source_config(args, &config)
            .with_context(|| format!("{name} actual source selection"))?;
        super::capacity::validate_effective_policy(args, &config)
            .with_context(|| format!("{name} effective admission policy"))?;
        kv::validate_storage(args.kv_dtype, &config)
            .with_context(|| format!("{name} actual KV storage"))?;
    }
    Ok(stdout)
}

pub(super) struct Server<'a> {
    pub args: &'a Args,
    pub health: Value,
    pub source_identity: Value,
    url: String,
    client: Client,
    _process: Process,
}

impl<'a> Server<'a> {
    #[cfg(unix)]
    pub async fn shutdown(mut self) -> Result<Value> {
        self._process
            .terminate_and_wait(Duration::from_secs(self.args.request_timeout_secs))
            .await
    }

    pub async fn start(args: &'a Args) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let mut argv = args.common_args("serve");
        argv.extend([
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            "--served-model-name".into(),
            "regression-model".into(),
        ]);
        if args.kv_dtype.is_some()
            || args.sequence_fit_policy.is_some()
            || args.max_num_batched_tokens.is_some()
            || identity::requires_source_evidence(args)?
        {
            argv.extend([
                "--effective-config-json".into(),
                args.report_dir
                    .join("serve.effective-config.json")
                    .to_string_lossy()
                    .into_owned(),
            ]);
        }
        let mut process = Process::spawn(args, "serve", argv, None)?;
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(args.request_timeout_secs))
            .build()?;
        let url = format!("http://127.0.0.1:{port}");
        let started = Instant::now();
        let timeout = Duration::from_secs(args.startup_timeout_secs);
        loop {
            if let Some(status) = process.child.try_wait()? {
                bail!(
                    "server exited before readiness: {status}; see {}",
                    process.stderr.display()
                );
            }
            ensure!(
                started.elapsed() < timeout,
                "server startup exceeded {timeout:?}; see {}",
                process.stderr.display()
            );
            let remaining = timeout.saturating_sub(started.elapsed());
            if let Ok(response) = client
                .get(format!("{url}/health"))
                .timeout(remaining.min(Duration::from_secs(2)))
                .send()
                .await
            {
                let status = response.status();
                if let Ok(text) = response.text().await {
                    fs::write(args.report_dir.join("serve.health.txt"), &text)?;
                    if status.is_success() {
                        let health = serde_json::from_str(&text).context("parse /health")?;
                        identity::validate_serve(args, &health)?;
                        kv::validate_storage(args.kv_dtype, &health)
                            .context("serve actual KV storage from /health")?;
                        if args.kv_dtype.is_some() {
                            let config: Value = serde_json::from_slice(&fs::read(
                                args.report_dir.join("serve.effective-config.json"),
                            )?)?;
                            kv::validate_storage(args.kv_dtype, &config)
                                .context("serve actual KV storage from effective config")?;
                        }
                        let source_identity = identity::source_evidence(args, "serve")?;
                        return Ok(Self {
                            args,
                            health,
                            source_identity,
                            url,
                            client,
                            _process: process,
                        });
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    pub async fn request(&self, name: &str, body: &Value) -> Result<String> {
        self.request_to(name, body, ApiEndpoint::ChatCompletions)
            .await
    }

    pub async fn request_to(
        &self,
        name: &str,
        body: &Value,
        endpoint: ApiEndpoint,
    ) -> Result<String> {
        write_json(
            self.args.report_dir.join(format!("{name}.request.json")),
            body,
        )?;
        let started = Instant::now();
        let mut metadata = json!({"path": endpoint.path()});
        let outcome = async {
            let mut response = self
                .client
                .post(format!("{}{}", self.url, endpoint.path()))
                .json(body)
                .send()
                .await?;
            let status = response.status();
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("")
                .to_owned();
            metadata["status"] = json!(status.as_u16());
            metadata["content_type"] = json!(content_type);
            let mut raw = File::create(self.args.report_dir.join(format!("{name}.response.txt")))?;
            let mut bytes = Vec::new();
            while let Some(chunk) = response.chunk().await? {
                if bytes.is_empty() && !chunk.is_empty() {
                    // Transport timing only: one chunk need not contain a text token.
                    metadata["first_body_chunk_ms"] =
                        json!(started.elapsed().as_secs_f64() * 1000.0);
                }
                // Keep partial SSE evidence even if a later read times out.
                raw.write_all(&chunk)?;
                bytes.extend_from_slice(&chunk);
                metadata["response_bytes"] = json!(bytes.len());
            }
            let text = String::from_utf8(bytes).context("response is not valid UTF-8")?;
            ensure!(status.is_success(), "{name} returned HTTP {status}: {text}");
            ensure!(
                if body["stream"] == true {
                    content_type.starts_with("text/event-stream")
                } else {
                    content_type.starts_with("application/json")
                },
                "unexpected content type {content_type}"
            );
            Ok::<_, anyhow::Error>(text)
        }
        .await;
        metadata["elapsed_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        metadata["outcome"] = json!(if outcome.is_ok() { "success" } else { "failed" });
        if let Err(error) = &outcome {
            metadata["error"] = json!(format!("{error:#}"));
        }
        write_json(
            self.args.report_dir.join(format!("{name}.http.json")),
            &metadata,
        )?;
        outcome
    }

    pub async fn health_snapshot(&self, name: &str) -> Result<Value> {
        let response = self
            .client
            .get(format!("{}/health", self.url))
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        fs::write(self.args.report_dir.join(format!("{name}.json")), &text)?;
        ensure!(status.is_success(), "health returned {status}: {text}");
        let health: Value = serde_json::from_str(&text).context("parse health snapshot")?;
        kv::validate_storage(self.args.kv_dtype, &health)?;
        Ok(health)
    }

    pub async fn models(&self) -> Result<Value> {
        let response = self
            .client
            .get(format!("{}/v1/models", self.url))
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        fs::write(self.args.report_dir.join("serve.models.json"), &text)?;
        ensure!(status.is_success(), "models returned {status}: {text}");
        let models: Value = serde_json::from_str(&text)?;
        ensure!(
            models["data"]
                .as_array()
                .is_some_and(|items| items.iter().any(|model| model["id"] == "regression-model")),
            "served alias absent from /v1/models"
        );
        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn graceful_shutdown_observes_original_child_exit_and_preserves_failures() {
        const PROBE: &str = "MODEL_REGRESSION_SHUTDOWN_PROBE";
        const DIRECTORY: &str = "MODEL_REGRESSION_SHUTDOWN_DIRECTORY";
        if let Ok(mode) = std::env::var(PROBE) {
            let directory = PathBuf::from(std::env::var_os(DIRECTORY).unwrap());
            let mut signal =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
            fs::write(directory.join("ready"), b"ready").unwrap();
            signal.recv().await.expect("receive original child SIGTERM");
            fs::write(directory.join("received"), std::process::id().to_string()).unwrap();
            match mode.as_str() {
                "success" => std::process::exit(0),
                "failure" => std::process::exit(17),
                "timeout" => std::future::pending::<()>().await,
                _ => panic!("unknown shutdown probe"),
            }
            unreachable!();
        }

        for mode in ["success", "failure", "timeout"] {
            let directory = tempfile::tempdir().unwrap();
            let stdout = directory.path().join("stdout");
            let stderr = directory.path().join("stderr");
            let child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "process::tests::graceful_shutdown_observes_original_child_exit_and_preserves_failures",
                ])
                .env(PROBE, mode)
                .env(DIRECTORY, directory.path())
                .stdin(Stdio::null())
                .stdout(File::create(&stdout).unwrap())
                .stderr(File::create(&stderr).unwrap())
                .spawn().unwrap();
            let mut process = Process {
                child,
                stdout,
                stderr,
            };
            let original_pid = process.child.id();
            let started = Instant::now();
            while !directory.path().join("ready").exists() {
                assert!(
                    process.child.try_wait().unwrap().is_none(),
                    "probe exited before readiness"
                );
                assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "probe readiness timeout"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let timeout = if mode == "timeout" {
                Duration::from_secs(1)
            } else {
                Duration::from_secs(10)
            };
            let result = process.terminate_and_wait(timeout).await;
            let evidence = if mode == "success" {
                let evidence = result.unwrap();
                assert_eq!(evidence["exit_code"], 0);
                assert!(process.child.try_wait().unwrap().unwrap().success());
                evidence
            } else {
                let error = result.unwrap_err();
                let failure = error.downcast_ref::<super::super::CaseFailure>().unwrap();
                if mode == "failure" {
                    assert_eq!(failure.evidence["exit_code"], 17);
                    assert_eq!(process.child.try_wait().unwrap().unwrap().code(), Some(17));
                } else {
                    assert!(failure.evidence["exit_code"].is_null());
                    assert!(process.child.try_wait().unwrap().is_none());
                    assert!(error.to_string().contains("exceeded"));
                }
                failure.evidence.clone()
            };
            assert_eq!(evidence["child_pid"], original_pid);
            assert_eq!(evidence["method"], "SIGTERM");
            assert_eq!(evidence["signal_sent"], true);
            // A zero-test subprocess must not count as a successful shutdown.
            // The child itself acknowledges receipt of the real signal.
            let received = directory.path().join("received");
            let started = Instant::now();
            while !received.exists() {
                assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "signal receipt timeout"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert_eq!(
                fs::read_to_string(received).unwrap(),
                original_pid.to_string()
            );
            // Timeout cleanup still belongs to Process::drop, not the report.
            drop(process);
        }
    }

    #[test]
    fn default_backend_isolates_only_child_product_environment() {
        const PROBE: &str = "MODEL_REGRESSION_ENV_PROBE";
        const PRODUCT: &str = "FERRUM_ENV_ISOLATION_FIXTURE";
        const SECRET: &str = "synthetic-fixture-value";
        const VERIFIED_EXIT: i32 = 23;
        if let Ok(mode) = std::env::var(PROBE) {
            if mode == "default" {
                assert!(std::env::vars_os()
                    .all(|(key, _)| !key.to_string_lossy().starts_with("FERRUM_")));
            } else {
                assert_eq!(mode, "explicit");
                assert_eq!(std::env::var(PRODUCT).unwrap(), SECRET);
            }
            for (key, value) in [
                ("PATH", "retained-path"),
                ("CUDA_VISIBLE_DEVICES", "7"),
                ("HF_HOME", "retained-cache"),
                ("HF_TOKEN", "synthetic-token"),
            ] {
                assert_eq!(std::env::var(key).unwrap(), value);
            }
            // Distinguishes execution of the probe from a test harness matching
            // no test. A failed assertion exits differently before this point.
            std::process::exit(VERIFIED_EXIT);
        }
        let original = std::env::var_os(PRODUCT);
        for use_default in [true, false] {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command.args([
                "--exact",
                "process::tests::default_backend_isolates_only_child_product_environment",
            ]);
            command
                .env(PROBE, if use_default { "default" } else { "explicit" })
                .env(PRODUCT, SECRET)
                .env("PATH", "retained-path")
                .env("CUDA_VISIBLE_DEVICES", "7")
                .env("HF_HOME", "retained-cache")
                .env("HF_TOKEN", "synthetic-token");
            let keys = std::env::vars_os()
                .map(|(key, _)| key)
                .chain(std::iter::once(OsString::from(PRODUCT)));
            let removed = isolate_product_environment(&mut command, use_default, keys);
            if use_default {
                assert!(removed.iter().any(|key| key == PRODUCT));
                assert!(removed.iter().all(|key| key.starts_with("FERRUM_")));
                assert!(!serde_json::to_string(&removed).unwrap().contains(SECRET));
            } else {
                assert!(removed.is_empty());
            }
            let output = command.output().unwrap();
            assert_eq!(
                output.status.code(),
                Some(VERIFIED_EXIT),
                "child environment probe failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(std::env::var_os(PRODUCT), original);
        }
    }
}
