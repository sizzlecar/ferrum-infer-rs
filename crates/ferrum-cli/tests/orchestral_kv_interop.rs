//! Opt-in, real Orchestral → Ferrum SSE/tool/session acceptance.
//!
//! FERRUM_ORCHESTRAL_INTEROP_CONFIG names a JSON file with ferrum_bin,
//! orchestral_bin, model, backend (metal/cuda), report_dir (new directory),
//! context_tokens and max_tokens. Optional: semantic_source, tokenizer_source,
//! runtime_memory_budget_bytes, disable_thinking (default true),
//! long_context_records (default 0), timeouts {startup_secs, turn_secs, http_secs}.
//! Run this target with --ignored --test-threads=1 on the selected GPU host.
//! Reports retain only journals created by this test; no user's Orchestral home
//! is inspected. The real model, not a scripted HTTP fixture, chooses tool calls.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[path = "orchestral_kv_interop/journal.rs"]
mod journal;
#[path = "orchestral_kv_interop/process.rs"]
mod process;
#[path = "orchestral_kv_interop/tests.rs"]
mod tests;

const MODEL_ALIAS: &str = "ferrum-interop";
const SESSION: &str = "kv-interop-session";
const DATA_FILE: &str = "shipment.json";

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Metal,
    Cuda,
}

impl Backend {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct Timeouts {
    startup_secs: u64,
    turn_secs: u64,
    http_secs: u64,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            startup_secs: 600,
            turn_secs: 600,
            http_secs: 10,
        }
    }
}

fn yes() -> bool {
    true
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Configuration {
    ferrum_bin: PathBuf,
    orchestral_bin: PathBuf,
    model: PathBuf,
    semantic_source: Option<PathBuf>,
    tokenizer_source: Option<PathBuf>,
    backend: Backend,
    report_dir: PathBuf,
    context_tokens: u32,
    max_tokens: u32,
    runtime_memory_budget_bytes: Option<u64>,
    #[serde(default = "yes")]
    disable_thinking: bool,
    #[serde(default)]
    long_context_records: u32,
    #[serde(default)]
    timeouts: Timeouts,
}

impl Configuration {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.max_tokens > 0 && self.context_tokens > self.max_tokens,
            "positive output budget must fit the context"
        );
        ensure!(
            self.runtime_memory_budget_bytes != Some(0),
            "memory budget must be positive"
        );
        ensure!(
            self.timeouts.startup_secs > 0
                && self.timeouts.turn_secs > 0
                && self.timeouts.http_secs > 0,
            "timeouts must be positive"
        );
        ensure!(
            self.long_context_records == 0 || self.long_context_records >= 3,
            "optional record retrieval needs at least three records"
        );
        for path in [&self.ferrum_bin, &self.orchestral_bin, &self.model]
            .into_iter()
            .chain(self.semantic_source.iter())
            .chain(self.tokenizer_source.iter())
        {
            ensure!(
                path.is_absolute() && path.exists(),
                "input must be an existing absolute local path: {}",
                path.display()
            );
        }
        ensure!(
            self.ferrum_bin.is_file() && self.orchestral_bin.is_file(),
            "executables must be files"
        );
        ensure!(self.report_dir.is_absolute(), "report_dir must be absolute");
        Ok(())
    }

    fn ferrum_command(&self, dtype: &str, port: u16, directory: &Path) -> Command {
        let mut command = Command::new(&self.ferrum_bin);
        command
            .current_dir(directory)
            .arg("serve")
            .arg(&self.model)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--backend",
                self.backend.as_str(),
                "--kv-dtype",
                dtype,
                "--served-model-name",
                MODEL_ALIAS,
                "--max-model-len",
                &self.context_tokens.to_string(),
                "--max-num-seqs",
                "1",
            ]);
        if self.disable_thinking {
            command.arg("--disable-thinking");
        }
        for (flag, path) in [
            ("--semantic-source", &self.semantic_source),
            ("--tokenizer-source", &self.tokenizer_source),
        ] {
            if let Some(path) = path {
                command.arg(flag).arg(path);
            }
        }
        if let Some(bytes) = self.runtime_memory_budget_bytes {
            command.args(["--runtime-memory-budget-bytes", &bytes.to_string()]);
        }
        // Explicit product options define this run, not inherited tuning knobs.
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("FERRUM_") {
                command.env_remove(key);
            }
        }
        command
    }

    fn orchestral_config(&self, endpoint: &str, directory: &Path) -> Value {
        json!({
            "version": 1,
            "agent": {
                "backend": "openai", "model_profile": "interop", "max_model_steps": 6,
                "max_tool_calls": 8, "max_context_tokens": self.context_tokens,
                "reserved_output_tokens": self.max_tokens,
                "input_requests_enabled": false,
                "project_instructions": {"enabled": false},
                "model_retry": {"max_retries": 0}, "compaction": {"enabled": false}
            },
            "providers": {
                "default_backend": "openai", "default_model": "interop",
                "backends": [{"name": "openai", "kind": "openai", "endpoint": endpoint,
                    "config": {"auth": "none", "stream_idle_timeout_secs": self.timeouts.turn_secs}}],
                "models": [{"name": "interop", "backend": "openai", "model": MODEL_ALIAS,
                    "temperature": 0.0, "max_tokens": self.max_tokens}]
            },
            "tools": {"exec": {"enabled": false, "allow_host_execution": false}},
            "mcp": {"enabled": false}, "skills": {"enabled": false, "auto_discover": false},
            "journal": {"backend": "filesystem", "root_dir": directory.join("journals")},
            "artifacts": {"backend": "filesystem", "root_dir": directory.join("artifacts")},
            "observability": {"log_file": directory.join("orchestral.log")}
        })
    }

    fn orchestral_command(&self, endpoint: &str, directory: &Path, prompt: &str) -> Command {
        let mut command = Command::new(&self.orchestral_bin);
        command
            .current_dir(directory.join("sandbox"))
            .env("ORCHESTRAL_HOME", directory.join("orchestral-home"))
            .args([
                "--base-url",
                endpoint,
                "--no-auth",
                "--model-profile",
                "interop",
                "--model",
                MODEL_ALIAS,
                "--session-id",
                SESSION,
                "--input-mode",
                "none",
                "--no-mcp",
                "--no-skills",
                "--temperature",
                "0",
                "--config",
            ])
            .arg(directory.join("orchestral.yaml"))
            .arg("--cwd")
            .arg(directory.join("sandbox"))
            .arg(prompt);
        // The local endpoint is the only provider. Do not inherit alternate homes,
        // API tokens, credential files or tracing exporters into the test agent.
        for (key, _) in std::env::vars_os() {
            let name = key.to_string_lossy();
            if (name.starts_with("ORCHESTRAL_") && name != "ORCHESTRAL_HOME")
                || name.contains("API_KEY")
                || name.starts_with("GOOGLE_")
                || name.starts_with("OPENAI_")
                || name.starts_with("OTEL_")
            {
                command.env_remove(key);
            }
        }
        command
    }
}

#[derive(Serialize)]
struct Facts {
    tracking_code: String,
    parcel_count: u32,
    retrieval_code: String,
}

impl Facts {
    fn new() -> Self {
        Self {
            tracking_code: format!("cargo-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]),
            parcel_count: 17,
            retrieval_code: format!("route-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]),
        }
    }
    fn answer(&self) -> Value {
        json!({"tracking_code": self.tracking_code, "parcel_count": self.parcel_count})
    }
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn health_evidence(health: &Value, dtype: &str) -> Result<()> {
    ensure!(health["status"] == "healthy", "server reports unhealthy");
    let format = match dtype {
        "fp16" => ferrum_types::KvStorageFormat::F16,
        "int8" => ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
        _ => anyhow::bail!("unknown dtype"),
    };
    let kv = &health["kv_storage"];
    ensure!(
        kv["source"] == "resolved_model_plan"
            && kv["requested"] == json!(format)
            && kv["selected"] == json!(format),
        "actual KV storage does not match {dtype}: {kv}"
    );
    ensure!(
        health["scheduler"]["failed_requests"].as_u64() == Some(0),
        "request failures: {}",
        health["scheduler"]
    );
    Ok(())
}

async fn get_health(client: &reqwest::Client, base: &str) -> Result<Value> {
    Ok(client
        .get(format!("{base}/health"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

async fn run_format(config: &Configuration, dtype: &str, facts: &Facts) -> Result<Value> {
    let directory = config.report_dir.join(dtype);
    fs::create_dir(&directory)?;
    fs::create_dir(directory.join("sandbox"))?;
    fs::create_dir(directory.join("orchestral-home"))?;
    write_json(directory.join("sandbox").join(DATA_FILE), &facts.answer())?;
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    let base = format!("http://127.0.0.1:{port}");
    let endpoint = format!("{base}/v1");
    fs::write(
        directory.join("orchestral.yaml"),
        serde_yaml::to_string(&config.orchestral_config(&endpoint, &directory))?,
    )?;
    let mut server = process::OwnedChild::spawn(
        config.ferrum_command(dtype, port, &directory),
        &directory,
        "ferrum",
    )?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(config.timeouts.http_secs))
        .build()?;
    let started = Instant::now();
    let before = loop {
        ensure!(
            server.try_wait()?.is_none(),
            "Ferrum exited before health readiness; see ferrum.stderr.log"
        );
        if let Ok(health) = get_health(&client, &base).await {
            break health;
        }
        ensure!(
            started.elapsed() < Duration::from_secs(config.timeouts.startup_secs),
            "Ferrum startup timed out"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    write_json(directory.join("health-before.json"), &before)?;
    health_evidence(&before, dtype)?;
    let result: Result<Vec<Value>> = async {
    let prompt = format!("Use the file_read tool to read {DATA_FILE} in this workspace. Report its tracking_code and parcel_count fields with their exact values. Do not modify files or run commands.");
    let mut turns = Vec::new();
    let first = run_turn(config, &endpoint, &directory, 1, &prompt, &facts.answer()).await?;
    let snapshot = journal::Snapshot::read(&directory.join("journals"))?;
    snapshot.validate_file_read(DATA_FILE, &facts.answer())?;
    turns.push(first);
    // A second process must reconstruct the same session; the source no longer
    // exists, so another read cannot substitute for conversation continuity.
    fs::remove_file(directory.join("sandbox").join(DATA_FILE))?;
    let tools_before = snapshot.tool_exchange_count();
    turns.push(run_turn(config, &endpoint, &directory, 2,
        "Without using any tools or reading any files, repeat the tracking_code and parcel_count from the previous turn. Use the values already in our conversation.", &facts.answer()).await?);
    let snapshot = journal::Snapshot::read(&directory.join("journals"))?;
    ensure!(
        snapshot.tool_exchange_count() == tools_before,
        "recall turn invoked a tool instead of using session history"
    );
    if config.long_context_records > 0 {
        let selected = config.long_context_records / 2;
        let mut prompt =
            String::from("Read these records supplied in this message; do not use tools.\n");
        for index in 0..config.long_context_records {
            let suffix = if index == selected {
                facts.retrieval_code.clone()
            } else {
                format!("unrelated-{index:06}")
            };
            prompt.push_str(&format!("record {index}: dispatch_code={suffix}; region=west; condition=packed; audit=pending.\n"));
        }
        prompt.push_str(&format!(
            "Report the record number {selected} and its exact dispatch_code value."
        ));
        turns.push(
            run_turn(
                config,
                &endpoint,
                &directory,
                3,
                &prompt,
                &json!({"record": selected, "dispatch_code": facts.retrieval_code}),
            )
            .await?,
        );
        ensure!(
            journal::Snapshot::read(&directory.join("journals"))?.tool_exchange_count()
                == tools_before,
            "record retrieval unexpectedly invoked a tool"
        );
    }
    Ok(turns)
    }.await;
    // Preserve server state on failed turns too, without hiding the first error.
    let final_health = get_health(&client, &base).await;
    match &final_health {
        Ok(health) => write_json(directory.join("health-after.json"), health)?,
        Err(error) => write_json(
            directory.join("health-after-error.json"),
            &json!({"error": format!("{error:#}")}),
        )?,
    }
    let turns = result?;
    let after = final_health?;
    health_evidence(&after, dtype)?;
    ensure!(
        after["scheduler"]["successful_requests"]
            .as_u64()
            .context("missing successful request count")?
            > before["scheduler"]["successful_requests"]
                .as_u64()
                .context("missing initial request count")?,
        "no real model requests completed"
    );
    ensure!(
        after["engine"]["active_requests"] == 0 && after["engine"]["queued_requests"] == 0,
        "requests remain active after completed agent turns"
    );
    journal::Snapshot::read(&directory.join("journals"))?.validate_turn_count(turns.len())?;
    Ok(
        json!({"kv_dtype": dtype, "status": "passed", "turns": turns, "health": after,
        "long_context": if config.long_context_records > 0 { "executed" } else { "not_requested" },
        "protocol": "real Orchestral OpenAI adapter SSE plus canonical tool/session/run journals"}),
    )
}

async fn run_turn(
    config: &Configuration,
    endpoint: &str,
    directory: &Path,
    index: usize,
    prompt: &str,
    expected: &Value,
) -> Result<Value> {
    let label = format!("turn-{index}");
    let before = journal::Snapshot::read(&directory.join("journals"))?.run_ids();
    fs::write(directory.join(format!("{label}.prompt.txt")), prompt)?;
    let output = process::run(
        config.orchestral_command(endpoint, directory, prompt),
        directory,
        &label,
        Duration::from_secs(config.timeouts.turn_secs),
    )
    .await?;
    ensure!(
        output.success,
        "Orchestral {label} failed; see {label}.stderr.log"
    );
    let snapshot = journal::Snapshot::read(&directory.join("journals"))?;
    let delivery = snapshot.new_delivery(&before)?;
    journal::validate_answer(&delivery, expected)?;
    let stdout = fs::read_to_string(directory.join(format!("{label}.stdout.log")))?;
    journal::validate_answer(&stdout, expected)?;
    Ok(
        json!({"turn": index, "seconds": output.seconds, "delivery": delivery, "expected": expected}),
    )
}

#[tokio::test]
#[ignore = "requires explicit local Ferrum/Orchestral binaries, model and GPU"]
async fn real_orchestral_file_read_and_session_recall_with_both_kv_formats() -> Result<()> {
    let path = std::env::var_os("FERRUM_ORCHESTRAL_INTEROP_CONFIG")
        .context("FERRUM_ORCHESTRAL_INTEROP_CONFIG is required")?;
    let config: Configuration = serde_json::from_slice(&fs::read(path)?)?;
    config.validate()?;
    fs::create_dir(&config.report_dir)
        .context("report_dir must be a new directory; previous evidence is never overwritten")?;
    write_json(config.report_dir.join("configuration.json"), &config)?;
    let identity = process::identities(&config.ferrum_bin, &config.orchestral_bin)?;
    write_json(config.report_dir.join("binary-identities.json"), &identity)?;
    let facts = Facts::new();
    write_json(config.report_dir.join("expected-facts.json"), &facts)?;
    let mut reports = Vec::new();
    for dtype in ["fp16", "int8"] {
        let result = run_format(&config, dtype, &facts).await;
        let report = match result {
            Ok(report) => report,
            Err(error) => {
                json!({"kv_dtype": dtype, "status": "failed", "error": format!("{error:#}")})
            }
        };
        write_json(
            config.report_dir.join(format!("{dtype}-report.json")),
            &report,
        )?;
        reports.push(report);
    }
    let unchanged = identity == process::identities(&config.ferrum_bin, &config.orchestral_bin)?;
    let passed = unchanged && reports.iter().all(|r| r["status"] == "passed");
    write_json(
        config.report_dir.join("report.json"),
        &json!({"passed": passed, "binaries_unchanged": unchanged,
        "cases": reports, "scope": "local agent tool/session interoperability; no performance or all-model qualification"}),
    )?;
    ensure!(
        passed,
        "paired interoperability failed; inspect {}",
        config.report_dir.join("report.json").display()
    );
    Ok(())
}
