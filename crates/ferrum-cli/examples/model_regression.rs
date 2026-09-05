//! Exercise a staged Ferrum binary with a real model, independently of this
//! runner's accelerator features. Run with --help for explicit test inputs.

#[path = "model_regression/cases.rs"]
mod cases;
#[path = "model_regression/process.rs"]
mod process;
#[path = "model_regression/protocol.rs"]
mod protocol;

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::future::Future;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum Check {
    Basic,
    Stop,
    Structured,
    Tools,
}

#[derive(Debug, Parser, Serialize)]
#[command(about = "Real-model regression against an explicitly selected staged binary")]
struct Args {
    #[arg(long)]
    ferrum_bin: PathBuf,
    /// Release alias, Hugging Face repository, local directory, or GGUF file.
    #[arg(long)]
    model: String,
    #[arg(long, value_parser = ["cpu", "metal", "cuda"])]
    backend: String,
    /// A new or empty directory; raw requests, responses and child logs stay here.
    #[arg(long)]
    report_dir: PathBuf,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "basic")]
    checks: Vec<Check>,
    /// Forward the public flag to both entrypoints. Otherwise keep model defaults.
    #[arg(long)]
    disable_thinking: bool,
    /// Also replay actual, nonempty tool-call reasoning through reasoning_content.
    #[arg(long)]
    reasoning_alias_replay: bool,
    /// Per-generation test output budget; does not change context or concurrency.
    #[arg(long, default_value = "512", value_parser = clap::value_parser!(u32).range(1..))]
    max_tokens: u32,
    #[arg(long, default_value = "600", value_parser = clap::value_parser!(u64).range(1..))]
    startup_timeout_secs: u64,
    #[arg(long, default_value = "300", value_parser = clap::value_parser!(u64).range(1..))]
    request_timeout_secs: u64,
    /// Includes model loading and all REPL turns in a run subprocess.
    #[arg(long, default_value = "1200", value_parser = clap::value_parser!(u64).range(1..))]
    run_timeout_secs: u64,
    /// Descriptive source/revision and precision labels; never inferred from names.
    #[arg(long)]
    source_label: Option<String>,
    #[arg(long)]
    precision_label: Option<String>,
}

impl Args {
    fn common_args(&self, entrypoint: &str) -> Vec<String> {
        let mut args = vec![
            entrypoint.into(),
            self.model.clone(),
            "--backend".into(),
            self.backend.clone(),
        ];
        if self.disable_thinking {
            args.push("--disable-thinking".into());
        }
        args
    }
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    fs::write(path.as_ref(), serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.as_ref().display()))
}

fn binary_sha256(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

struct Report {
    path: PathBuf,
    document: Value,
    failed: bool,
}

impl Report {
    fn save(&self) -> Result<()> {
        write_json(&self.path, &self.document)
    }

    fn record(&mut self, name: &str, elapsed: Duration, result: Result<Value>) -> Result<()> {
        let case = match result {
            Ok(evidence) => json!({"case": name, "status": "passed", "evidence": evidence}),
            Err(error) => {
                self.failed = true;
                json!({"case": name, "status": "failed", "error": format!("{error:#}")})
            }
        };
        eprintln!("{name}: {}", case["status"]);
        if let Some(error) = case.get("error") {
            eprintln!("  {error}");
        }
        let mut case = case;
        case["elapsed_ms"] = json!(elapsed.as_millis());
        self.document["cases"]
            .as_array_mut()
            .expect("report cases")
            .push(case);
        self.save()
    }
}

async fn record(
    report: &mut Report,
    name: &str,
    future: impl Future<Output = Result<Value>>,
) -> Result<()> {
    let started = Instant::now();
    let result = future.await;
    report.record(name, started.elapsed(), result)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();
    ensure!(!args.model.trim().is_empty(), "--model must not be empty");
    ensure!(
        args.checks
            .iter()
            .enumerate()
            .all(|(index, check)| !args.checks[..index].contains(check)),
        "each check may be selected only once"
    );
    ensure!(
        !args.reasoning_alias_replay || args.checks.contains(&Check::Tools),
        "--reasoning-alias-replay requires --checks tools (or a list including tools)"
    );
    args.ferrum_bin = fs::canonicalize(&args.ferrum_bin).context("resolve staged binary")?;
    if Path::new(&args.model).exists() {
        args.model = fs::canonicalize(&args.model)?
            .to_str()
            .context("model path is not UTF-8")?
            .to_owned();
    }
    fs::create_dir_all(&args.report_dir)?;
    ensure!(
        fs::read_dir(&args.report_dir)?.next().is_none(),
        "report directory must be empty to preserve prior evidence"
    );
    args.report_dir = fs::canonicalize(&args.report_dir)?;
    let digest = binary_sha256(&args.ferrum_bin)?;
    let mut report = Report {
        path: args.report_dir.join("report.json"),
        document: json!({
            "schema_version": 1, "status": "running", "options": args,
            "binary_sha256": digest, "started_at": chrono::Utc::now().to_rfc3339(),
            "sampling": {"temperature": 0, "seed": 7, "max_tokens": args.max_tokens},
            "capacity_overrides": [], "cases": []
        }),
        failed: false,
    };
    report.save()?;
    record(&mut report, "binary-version", async {
        let version = process::run(
            &args,
            "version",
            vec!["--version".into()],
            None,
            Duration::from_secs(args.request_timeout_secs),
        )
        .await?;
        ensure!(
            !version.trim().is_empty(),
            "binary returned an empty version"
        );
        Ok(json!({"version": version.trim()}))
    })
    .await?;
    if args.checks.contains(&Check::Basic) {
        record(&mut report, "run-basic", cases::run_basic(&args)).await?;
    }
    if args.checks.contains(&Check::Stop) {
        record(&mut report, "run-stop", cases::run_stop(&args)).await?;
    }
    let started = Instant::now();
    match process::Server::start(&args).await {
        Ok(server) => {
            report.record(
                "serve-startup",
                started.elapsed(),
                Ok(server.health.clone()),
            )?;
            for check in &args.checks {
                match check {
                    Check::Basic => {
                        record(&mut report, "serve-basic", cases::serve_basic(&server)).await?
                    }
                    Check::Stop => {
                        record(&mut report, "serve-stop", cases::serve_stop(&server)).await?
                    }
                    Check::Structured => {
                        record(&mut report, "serve-structured", cases::structured(&server)).await?
                    }
                    Check::Tools => {
                        record(&mut report, "serve-tools", cases::tools(&server)).await?
                    }
                }
            }
        }
        Err(error) => {
            report.record("serve-startup", started.elapsed(), Err(error))?;
            report.document["unexecuted_serve_checks"] = json!(args.checks);
        }
    }
    let final_digest = binary_sha256(&args.ferrum_bin)?;
    report.record(
        "binary-unchanged",
        Duration::ZERO,
        if final_digest == digest {
            Ok(json!({"sha256": final_digest}))
        } else {
            Err(anyhow::anyhow!("staged binary changed during regression"))
        },
    )?;
    report.document["status"] = json!(if report.failed { "failed" } else { "passed" });
    report.document["finished_at"] = json!(chrono::Utc::now().to_rfc3339());
    report.save()?;
    ensure!(
        !report.failed,
        "regression failed; see {}",
        report.path.display()
    );
    Ok(())
}
