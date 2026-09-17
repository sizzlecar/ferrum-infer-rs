//! Bounded fixed-history F16/INT8 quality comparison using the product binary.
//! Passing certifies the supplied finite sample and budgets, not general quality.

#[path = "kv_teacher_regression/evidence.rs"]
mod evidence;
#[path = "kv_teacher_regression/process.rs"]
mod process;
#[path = "kv_teacher_regression/scoring.rs"]
mod scoring;
#[cfg(test)]
#[path = "kv_teacher_regression/tests.rs"]
mod tests;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use ferrum_bench_core::release_regression::model_sources::pinned_hf_source;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Parser, Serialize)]
#[command(about = "Measure paired F16/INT8 KV on one complete canonical teacher history")]
struct Args {
    #[arg(long)]
    ferrum_bin: PathBuf,
    #[arg(long)]
    checkpoint_diff_bin: PathBuf,
    /// Existing local model path or owner/repository@40-hex-commit.
    #[arg(long)]
    model: String,
    /// Repository-relative GGUF file when using a pinned HF source.
    #[arg(long)]
    gguf_file: Option<String>,
    #[arg(long)]
    semantic_source: Option<PathBuf>,
    #[arg(long)]
    tokenizer_source: Option<PathBuf>,
    #[arg(long, value_parser = ["metal", "cuda"])]
    backend: String,
    /// UTF-8 plain text, passed unchanged as the one-shot user prompt.
    #[arg(long)]
    prompt_file: PathBuf,
    /// New or empty directory. Failures retain commands, logs and checkpoints.
    #[arg(long)]
    report_dir: PathBuf,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    context_tokens: u32,
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..))]
    max_tokens: u32,
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    runtime_memory_budget_bytes: Option<u64>,
    #[arg(long, default_value_t = 1200, value_parser = clap::value_parser!(u64).range(1..))]
    timeout_secs: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    disable_thinking: bool,
    #[command(flatten)]
    budgets: scoring::Budgets,
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    fs::write(path.as_ref(), serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.as_ref().display()))
}

fn binary_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

impl Args {
    fn normalize(&mut self) -> Result<()> {
        self.budgets.validate()?;
        ensure!(
            self.context_tokens > self.max_tokens,
            "output budget must fit the context"
        );
        ensure!(
            self.max_tokens as usize <= ferrum_types::MAX_VNEXT_TEACHER_FORCED_TOKENS,
            "output budget exceeds the teacher-capture limit"
        );
        self.ferrum_bin = fs::canonicalize(&self.ferrum_bin).context("resolve Ferrum binary")?;
        self.checkpoint_diff_bin = fs::canonicalize(&self.checkpoint_diff_bin)
            .context("resolve checkpoint_diff binary")?;
        self.prompt_file = fs::canonicalize(&self.prompt_file).context("resolve prompt file")?;
        if Path::new(&self.model).exists() {
            self.model = fs::canonicalize(&self.model)?
                .to_string_lossy()
                .into_owned();
            ensure!(
                self.gguf_file.is_none(),
                "--gguf-file requires a pinned HF model"
            );
        } else {
            ensure!(
                pinned_hf_source(&self.model)
                    .map_err(anyhow::Error::msg)?
                    .is_some(),
                "model must be a local path or immutable owner/repository@40-hex-commit"
            );
            if let Some(filename) = &self.gguf_file {
                ferrum_types::validate_gguf_filename(filename).map_err(anyhow::Error::msg)?;
            }
        }
        for source in [&mut self.semantic_source, &mut self.tokenizer_source] {
            if let Some(path) = source {
                *path = fs::canonicalize(&*path).context("resolve explicit metadata path")?;
            }
        }
        fs::create_dir_all(&self.report_dir)?;
        ensure!(
            fs::read_dir(&self.report_dir)?.next().is_none(),
            "report directory must be empty"
        );
        self.report_dir = fs::canonicalize(&self.report_dir)?;
        Ok(())
    }

    fn run_args(
        &self,
        name: &str,
        dtype: &str,
        prompt: &str,
        teacher: Option<&Path>,
    ) -> Vec<String> {
        let mut words = vec!["run".into(), self.model.clone()];
        for (flag, value) in [
            ("--backend", self.backend.clone()),
            ("--kv-dtype", dtype.into()),
            ("--prompt", prompt.into()),
            ("--max-tokens", self.max_tokens.to_string()),
            ("--seed", self.seed.to_string()),
            ("--temperature", "0".into()),
            ("--top-k", "0".into()),
            ("--top-p", "1".into()),
            ("--min-p", "0".into()),
            ("--presence-penalty", "0".into()),
            ("--repeat-penalty", "1".into()),
            ("--output-format", "jsonl".into()),
            ("--kv-capacity", self.context_tokens.to_string()),
            ("--max-model-len", self.context_tokens.to_string()),
            ("--max-num-seqs", "1".into()),
            ("--sequence-fit-policy", "full-input-must-fit".into()),
            (
                "--effective-config-json",
                self.report_dir
                    .join(format!("{name}.effective-config.json"))
                    .to_string_lossy()
                    .into_owned(),
            ),
        ] {
            words.extend([flag.into(), value]);
        }
        words.push("--no-context-shift".into());
        if self.disable_thinking {
            words.push("--disable-thinking".into());
        }
        for (flag, path) in [
            ("--semantic-source", &self.semantic_source),
            ("--tokenizer-source", &self.tokenizer_source),
        ] {
            if let Some(path) = path {
                words.extend([flag.into(), path.to_string_lossy().into_owned()]);
            }
        }
        if let Some(filename) = &self.gguf_file {
            words.extend(["--gguf-file".into(), filename.clone()]);
        }
        if let Some(budget) = self.runtime_memory_budget_bytes {
            words.extend(["--runtime-memory-budget-bytes".into(), budget.to_string()]);
        }
        if let Some(teacher) = teacher {
            words.extend([
                "--vnext-checkpoint-dir".into(),
                self.report_dir
                    .join(format!("{name}.checkpoints"))
                    .to_string_lossy()
                    .into_owned(),
                "--vnext-checkpoint-product-output".into(),
                "--vnext-checkpoint-teacher-token-file".into(),
                teacher.to_string_lossy().into_owned(),
            ]);
        }
        words
    }
}

async fn measure(args: &Args, report: &mut Value) -> Result<()> {
    let prompt = fs::read_to_string(&args.prompt_file).context("read UTF-8 prompt")?;
    ensure!(!prompt.trim().is_empty(), "prompt must not be empty");
    fs::write(args.report_dir.join("prompt.txt"), &prompt)?;
    report["prompt_sha256"] = json!(format!("{:x}", Sha256::digest(prompt.as_bytes())));
    // An explicit product configuration keeps attention arithmetic comparable.
    fs::write(
        args.report_dir.join("ferrum.toml"),
        "[runtime]\nattention_policy = \"portable\"\n",
    )?;
    let timeout = Duration::from_secs(args.timeout_secs);
    let stdout = process::run(
        &args.ferrum_bin,
        &args.run_args("seed", "fp16", &prompt, None),
        &args.report_dir,
        "seed",
        timeout,
    )
    .await?;
    let seed = evidence::parse_generation(&stdout, args, true)?;
    let seed_config = evidence::read_config(args, "seed", ferrum_types::KvStorageFormat::F16)?;
    let teacher = args.report_dir.join("teacher.json");
    write_json(
        &teacher,
        &json!({"schema_version":1,"encoding":"u32-le","token_ids":seed.tokens}),
    )?;
    report["seed"] = json!({"usage":seed.usage,"token_ids_sha256":evidence::token_digest(&seed.tokens),"effective_config":"seed.effective-config.json"});
    for (name, dtype, format) in [
        ("fp16", "fp16", ferrum_types::KvStorageFormat::F16),
        (
            "int8",
            "int8",
            ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
        ),
    ] {
        let stdout = process::run(
            &args.ferrum_bin,
            &args.run_args(name, dtype, &prompt, Some(&teacher)),
            &args.report_dir,
            name,
            timeout,
        )
        .await?;
        let observed = evidence::parse_generation(&stdout, args, false)?;
        ensure!(
            observed.usage == seed.usage,
            "{name} teacher run usage differs from complete seed usage"
        );
        let config = evidence::read_config(args, name, format)?;
        ensure!(
            config["resolution_evidence"] == seed_config["resolution_evidence"],
            "{name} used different model, metadata, tokenizer or template sources"
        );
        if name == "fp16" {
            ensure!(
                config["numerical_execution"]["selected_profile"]
                    == seed_config["numerical_execution"]["selected_profile"],
                "F16 capture selected a different numerical profile from the seed"
            );
        }
        report[name] = json!({"usage":observed.usage,"effective_config":format!("{name}.effective-config.json")});
    }
    report["source_identity_verified"] = json!(true);
    report["seed_history_complete"] = json!(true);
    let output = args.report_dir.join("checkpoint-diff.json");
    let words = [
        "--reference-dir".to_owned(),
        args.report_dir
            .join("fp16.checkpoints")
            .to_string_lossy()
            .into_owned(),
        "--candidate-dir".to_owned(),
        args.report_dir
            .join("int8.checkpoints")
            .to_string_lossy()
            .into_owned(),
        "--output".to_owned(),
        output.to_string_lossy().into_owned(),
    ];
    process::run(
        &args.checkpoint_diff_bin,
        &words,
        &args.report_dir,
        "checkpoint-diff",
        timeout,
    )
    .await?;
    let comparison = serde_json::from_slice(&fs::read(output)?)?;
    let score = scoring::evaluate(&comparison, &seed.tokens, args.budgets)?;
    let passed = score["passed"] == true;
    report["score"] = score;
    ensure!(
        passed,
        "fixed-history sample exceeds the explicitly selected quality budgets"
    );
    Ok(())
}

async fn run(mut args: Args) -> Result<()> {
    args.normalize()?;
    let before = json!({"ferrum":binary_sha256(&args.ferrum_bin)?,"checkpoint_diff":binary_sha256(&args.checkpoint_diff_bin)?});
    let mut report = json!({"schema_version":1,"status":"running","args":args,
        "runner_version":env!("CARGO_PKG_VERSION"),
        "started_at":chrono::Utc::now().to_rfc3339(),"binary_sha256_before":before,
        "release_approved":false,"scope":"one_fixed_history_sample",
        "limitation":"Finite sample only; passing does not establish general model quality or performance."});
    let path = args.report_dir.join("report.json");
    write_json(&path, &report)?;
    let result = measure(&args, &mut report).await;
    let after = (|| -> Result<Value> {
        Ok(json!({"ferrum":binary_sha256(&args.ferrum_bin)?,
        "checkpoint_diff":binary_sha256(&args.checkpoint_diff_bin)?}))
    })();
    let integrity = after.as_ref().is_ok_and(|after| after == &before);
    report["binary_sha256_after"] = match after {
        Ok(value) => value,
        Err(error) => json!({"error":format!("{error:#}")}),
    };
    report["binary_integrity_verified"] = json!(integrity);
    report["status"] = json!(if result.is_ok() && integrity {
        "passed"
    } else {
        "failed"
    });
    report["finished_at"] = json!(chrono::Utc::now().to_rfc3339());
    if let Err(error) = &result {
        report["error"] = json!(format!("{error:#}"));
    }
    write_json(&path, &report)?;
    result?;
    ensure!(
        integrity,
        "an executable changed or became unreadable during measurement"
    );
    println!("{}", path.display());
    Ok(())
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    match run(Args::parse()).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("kv teacher regression: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
