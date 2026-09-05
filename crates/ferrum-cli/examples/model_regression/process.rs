use super::{write_json, Args};
use anyhow::{bail, ensure, Context, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Process {
    child: Child,
    stdout: PathBuf,
    stderr: PathBuf,
}

impl Process {
    fn spawn(args: &Args, name: &str, argv: Vec<String>, stdin: Option<&str>) -> Result<Self> {
        let stdout = args.report_dir.join(format!("{name}.stdout.txt"));
        let stderr = args.report_dir.join(format!("{name}.stderr.txt"));
        write_json(
            args.report_dir.join(format!("{name}.command.json")),
            &json!({
                "program": args.ferrum_bin, "args": argv, "cwd": args.report_dir,
                "environment_overrides": {"NO_COLOR": "1"}
            }),
        )?;
        if let Some(input) = stdin {
            fs::write(args.report_dir.join(format!("{name}.stdin.txt")), input)?;
        }
        let child = Command::new(&args.ferrum_bin)
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
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait()? {
                ensure!(
                    status.success(),
                    "child exited {status}; stderr: {}",
                    self.stderr.display()
                );
                return fs::read_to_string(&self.stdout).context("read child UTF-8 stdout");
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
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(super) async fn run(
    args: &Args,
    name: &str,
    argv: Vec<String>,
    stdin: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    Process::spawn(args, name, argv, stdin)?.wait(timeout).await
}

pub(super) struct Server<'a> {
    pub args: &'a Args,
    pub health: Value,
    url: String,
    client: Client,
    _process: Process,
}

impl<'a> Server<'a> {
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
                        return Ok(Self {
                            args,
                            health,
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
        write_json(
            self.args.report_dir.join(format!("{name}.request.json")),
            body,
        )?;
        let mut response = self
            .client
            .post(format!("{}/v1/chat/completions", self.url))
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
        write_json(
            self.args.report_dir.join(format!("{name}.http.json")),
            &json!({"status": status.as_u16(), "content_type": content_type}),
        )?;
        let mut raw = File::create(self.args.report_dir.join(format!("{name}.response.txt")))?;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            // Keep partial SSE evidence even if a later read times out.
            raw.write_all(&chunk)?;
            bytes.extend_from_slice(&chunk);
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
        Ok(text)
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
