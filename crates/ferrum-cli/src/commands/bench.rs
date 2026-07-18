//! Bench command — single-process CLI benchmark (no HTTP).
//!
//! Same forward path as `ferrum run` (batch=1 by default, no HTTP). Emits
//! the canonical `BenchReport` schema from `ferrum-bench-core`. This is
//! the "Level 1" command in PLAYBOOK § 1 — single-user feel of the engine.
//!
//! Modes:
//!   ferrum bench qwen3:4b                              # default: sequential, 5 rounds
//!   ferrum bench qwen3:4b --concurrency 4              # concurrent (tests batch decode)
//!   ferrum bench qwen3:4b --max-tokens 1024            # long decode (tests flash decode)
//!   ferrum bench qwen3:4b --long-context               # 2k prompt + 256 decode
//!   ferrum bench qwen3:4b --concurrency 8 --max-tokens 64  # throughput stress
//!
//! Phase 0 additions:
//!   --n-repeats N    independent runs (≥3 unlocks stddev + CI95)
//!   --goodput ...    SLO triple for goodput computation
//!   --output json    emit canonical BenchReport JSON
//!   --out PATH       write JSON to file (else stdout)

use crate::config::CliConfig;
use chrono::Utc;
use clap::Args;
use colored::*;
use ferrum_bench_core::{
    compute_metrics, BenchReport, Env, ItlEvidenceSource, OutputTokenCountSource,
    QualityIssueCounts, RequestItlEvidence, RequestRecord, RunRecord, Scenario, Slo,
};
use ferrum_types::{InferenceRequest, Priority, RequestId, Result, SamplingParams};
use futures::StreamExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use uuid::Uuid;

#[derive(Args)]
pub struct BenchCommand {
    /// Model name (e.g., qwen3:0.6b, qwen3:4b)
    #[arg(default_value = "qwen3:0.6b")]
    pub model: String,

    /// Sequential inference passes per run.
    #[arg(long, default_value = "5")]
    pub rounds: usize,

    /// Max tokens per request
    #[arg(long, default_value = "128")]
    pub max_tokens: u32,

    /// Backend: auto, cpu, cuda, metal
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// Prompt to use
    #[arg(long, default_value = "Explain the theory of relativity in detail.")]
    pub prompt: String,

    /// Number of concurrent requests (>1 tests batch decode).
    #[arg(long, default_value = "1")]
    pub concurrency: usize,

    /// Long-context mode: use a ~2k token prompt (tests flash decode / paged KV).
    #[arg(long)]
    pub long_context: bool,

    /// KV cache element dtype. Accepts `fp16`, `bf16`, `int8`, `fp8`.
    #[arg(long, value_name = "DTYPE")]
    pub kv_dtype: Option<String>,

    // ─── Phase 0 additions (canonical schema) ─────────────────────
    /// Independent repeats of the whole bench. ≥ 3 unlocks CI95
    /// (PLAYBOOK § 0.4). Default 1 — emits mean only.
    #[arg(long, default_value_t = 1)]
    pub n_repeats: u32,

    /// SLO triple for goodput. Format: `ttft:500 tpot:50 e2el:30000`.
    #[arg(long, value_parser = super::bench_serve::parse_slo)]
    pub goodput: Option<Slo>,

    /// Output format: `json` (canonical BenchReport) or `human` (summary
    /// table, default — matches pre-Phase-0 behaviour).
    #[arg(long, default_value = "human")]
    pub output: String,

    /// Output file path (json mode).
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Override `env.hw_id` (defaults to auto-detected).
    #[arg(long)]
    pub hw_id: Option<String>,

    /// Override `env.commit_sha` (defaults to `git rev-parse --short HEAD`).
    #[arg(long)]
    pub commit_sha: Option<String>,
}

fn validate_command(cmd: &BenchCommand) -> Result<()> {
    if cmd.n_repeats == 0 {
        return Err(ferrum_types::FerrumError::model("--n-repeats must be > 0"));
    }
    if cmd.rounds == 0 {
        return Err(ferrum_types::FerrumError::model("--rounds must be > 0"));
    }
    if cmd.concurrency == 0 {
        return Err(ferrum_types::FerrumError::model(
            "--concurrency must be > 0",
        ));
    }
    measured_request_count(cmd)?;
    if cmd.goodput.is_some_and(|slo| !slo.is_valid()) {
        return Err(ferrum_types::FerrumError::model(
            "--goodput values must be positive finite numbers",
        ));
    }
    Ok(())
}

fn measured_request_count(cmd: &BenchCommand) -> Result<u32> {
    let count = cmd
        .rounds
        .checked_mul(cmd.concurrency)
        .ok_or_else(|| ferrum_types::FerrumError::model("rounds * concurrency overflow"))?;
    u32::try_from(count).map_err(|_| {
        ferrum_types::FerrumError::model("rounds * concurrency exceeds report capacity")
    })
}

pub async fn execute(cmd: BenchCommand, config: CliConfig) -> Result<()> {
    validate_command(&cmd)?;
    let cache_dir = crate::source_resolver::hf_cache_dir(&config);
    let resolved = crate::source_resolver::resolve_model_source(
        &cmd.model,
        &cache_dir,
        crate::source_resolver::DownloadPolicy::AutoDownload,
        None,
    )
    .await?;
    let source = resolved.source;
    let model_id = crate::source_resolver::public_model_id(&source);
    eprintln!("{}", format!("Ferrum Benchmark - {}", model_id).bold());
    eprintln!("{}", "=".repeat(60).dimmed());

    let engine_model_path = source.local_path.to_string_lossy().to_string();

    let device = super::run::select_device(&cmd.backend)?;
    let backend_str = format!("{:?}", device).to_lowercase();
    eprintln!("{} {:?}", "Device:".dimmed(), device);
    let runtime_config = ferrum_types::RuntimeConfigSnapshot::capture_current();

    #[cfg(feature = "cuda")]
    {
        let graph_mode =
            crate::runtime_env::runtime_snapshot_value(&runtime_config, "FERRUM_CUDA_GRAPH")
                .is_some();
        if !graph_mode {
            if let Ok(d) = candle_core::Device::new_cuda(0) {
                if let Ok(cd) = d.as_cuda_device() {
                    let name = cd.cuda_stream().context().name().unwrap_or_default();
                    eprintln!("GPU 0: {name}");
                }
            }
        }
        let tp = crate::runtime_env::runtime_snapshot_value(&runtime_config, "FERRUM_TP")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                candle_core::cuda_backend::cudarc::driver::CudaContext::device_count()
                    .map(|n| n as usize)
                    .unwrap_or(1)
            });
        if tp > 1 {
            eprintln!("Tensor Parallel: TP={tp}");
        }
    }

    let mut engine_config = ferrum_types::EngineConfig::default();
    engine_config.model.model_id = ferrum_types::ModelId::new(model_id.clone());
    engine_config.backend.device = device;
    engine_config.backend.backend_options.insert(
        "model_path".to_string(),
        serde_json::Value::String(engine_model_path),
    );
    engine_config.scheduler.policy = ferrum_types::SchedulingPolicy::ContinuousBatch;
    engine_config
        .apply_runtime_config_snapshot(&runtime_config)
        .map_err(ferrum_types::FerrumError::config)?;
    let effective_kv_dtype = cmd
        .kv_dtype
        .as_deref()
        .or_else(|| crate::runtime_env::runtime_snapshot_value(&runtime_config, "FERRUM_KV_DTYPE"));
    super::run::apply_kv_dtype_override(&mut engine_config, effective_kv_dtype)?;
    let engine = ferrum_engine::create_default_engine(engine_config).await?;

    let prompt = if cmd.long_context {
        generate_long_prompt()
    } else {
        cmd.prompt.clone()
    };

    let mode_str = if cmd.concurrency > 1 {
        format!("concurrent({})", cmd.concurrency)
    } else if cmd.long_context {
        "long-context".to_string()
    } else {
        "sequential".to_string()
    };

    eprintln!(
        "{}",
        format!(
            "Config: {} rounds × {} repeats, {} max_tokens, mode={}, prompt_len=~{}chars",
            cmd.rounds,
            cmd.n_repeats,
            cmd.max_tokens,
            mode_str,
            prompt.len()
        )
        .dimmed()
    );
    if cmd.n_repeats < 3 {
        eprintln!(
            "{}",
            "  [warn] n_repeats < 3 — emitting mean only, no stddev/CI95 (PLAYBOOK § 0.4)".yellow()
        );
    }
    eprintln!("{}", "=".repeat(60).dimmed());

    // Warmup (discarded — once per process, not per repeat).
    eprintln!("{}", "Warmup...".dimmed());
    let warmup = run_single(&*engine, &model_id, "Hello", 16).await?;
    if !warmup.success {
        return Err(ferrum_types::FerrumError::model(
            "benchmark warmup request failed",
        ));
    }
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // ─── n_repeats × rounds run loop ──────────────────────────────
    let mut runs: Vec<RunRecord> = Vec::with_capacity(cmd.n_repeats as usize);
    for repeat_idx in 0..cmd.n_repeats {
        eprintln!(
            "{}",
            format!("Repeat {}/{}", repeat_idx + 1, cmd.n_repeats).bold()
        );
        let run = if cmd.concurrency > 1 {
            run_concurrent_round(&*engine, &model_id, &prompt, &cmd).await?
        } else {
            run_sequential_round(&*engine, &model_id, &prompt, &cmd).await?
        };
        eprintln!(
            "  {} requests · {:.1}s · {:.1} tok/s",
            run.records.len(),
            run.duration_s,
            run.records
                .iter()
                .map(|r| r.output_tokens as f64)
                .sum::<f64>()
                / run.duration_s
        );
        runs.push(run);
    }

    // ─── Aggregate + emit ─────────────────────────────────────────
    let scenario = if cmd.concurrency > 1 {
        // Closed-loop with K workers (concurrency).
        Scenario::ClosedLoop
    } else {
        Scenario::Cli
    };
    let concurrency = if cmd.concurrency > 1 {
        Some(cmd.concurrency as u32)
    } else {
        None
    };

    let env = build_env(&cmd);
    let slo = cmd.goodput.unwrap_or_else(Slo::unbounded);
    let prompt_len = u32::try_from(prompt.len()).map_err(|_| {
        ferrum_types::FerrumError::model("benchmark prompt length exceeds report capacity")
    })?;
    let report = compute_metrics(
        model_id.clone(),
        backend_str,
        scenario,
        concurrency,
        None,
        prompt_len, // n_prompt char approx — true token count requires tokenizer
        cmd.max_tokens,
        0, // CLI bench: process-level warmup only (not per-repeat)
        slo,
        runs,
        env,
    );

    emit_then_enforce_bench_report(&cmd, &report, &mode_str)
}

// ── Run loops (one round = N sequential or concurrent passes) ───────

async fn run_sequential_round(
    engine: &(dyn ferrum_interfaces::engine::LlmInferenceEngine + Send + Sync),
    model_id: &str,
    prompt: &str,
    cmd: &BenchCommand,
) -> Result<RunRecord> {
    let expected_requests = measured_request_count(cmd)?;
    let mut records = Vec::with_capacity(cmd.rounds);
    let start = Instant::now();
    for _ in 0..cmd.rounds {
        match run_single(engine, model_id, prompt, cmd.max_tokens).await {
            Ok(record) => records.push(record),
            Err(error) => {
                eprintln!("  request start error: {error}");
                let mut quality = QualityIssueCounts::default();
                quality.malformed_stream = 1;
                records.push(failed_bench_record(quality));
            }
        }
        // Let engine finish cleanup between rounds.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    let duration_s = start.elapsed().as_secs_f64();
    Ok(RunRecord {
        records,
        expected_requests,
        duration_s,
        warmup: Default::default(),
    })
}

async fn run_concurrent_round(
    engine: &(dyn ferrum_interfaces::engine::LlmInferenceEngine + Send + Sync),
    model_id: &str,
    prompt: &str,
    cmd: &BenchCommand,
) -> Result<RunRecord> {
    let expected_requests = measured_request_count(cmd)?;
    let mut records = Vec::with_capacity(expected_requests as usize);
    let start = Instant::now();
    for _ in 0..cmd.rounds {
        let mut handles = Vec::with_capacity(cmd.concurrency);
        for _ in 0..cmd.concurrency {
            let request = make_request(model_id, prompt, cmd.max_tokens);
            match engine.infer_stream(request).await {
                Ok(stream) => handles.push(tokio::spawn(collect_stream(stream))),
                Err(error) => {
                    eprintln!("  request start error: {error}");
                    let mut quality = QualityIssueCounts::default();
                    quality.malformed_stream = 1;
                    records.push(failed_bench_record(quality));
                }
            }
        }
        for handle in handles {
            match handle.await {
                Ok(Ok(r)) => records.push(r),
                Ok(Err(e)) => {
                    eprintln!("  request error: {e}");
                    let mut quality = QualityIssueCounts::default();
                    quality.malformed_stream = 1;
                    records.push(failed_bench_record(quality));
                }
                Err(e) => {
                    eprintln!("  join error: {e}");
                    let mut quality = QualityIssueCounts::default();
                    quality.panic = 1;
                    records.push(failed_bench_record(quality));
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    let duration_s = start.elapsed().as_secs_f64();
    Ok(RunRecord {
        records,
        expected_requests,
        duration_s,
        warmup: Default::default(),
    })
}

fn failed_bench_record(quality_issues: QualityIssueCounts) -> RequestRecord {
    RequestRecord {
        success: false,
        ttft_ms: 0.0,
        e2e_ms: 0.0,
        input_tokens: 0,
        output_tokens: 0,
        output_token_count_source: OutputTokenCountSource::None,
        itl_evidence: RequestItlEvidence::failed(ItlEvidenceSource::EngineTokenEvents),
        quality_issues,
        itl_ms: vec![],
    }
}

// ── Single-stream collection ────────────────────────────────────────

fn make_request(model_id: &str, prompt: &str, max_tokens: u32) -> InferenceRequest {
    InferenceRequest {
        id: RequestId(Uuid::new_v4()),
        model_id: ferrum_types::ModelId(model_id.to_string()),
        prompt: prompt.to_string(),
        sampling_params: SamplingParams {
            max_tokens: max_tokens as usize,
            temperature: 0.0, // greedy — matches PLAYBOOK § 0.5 L3 determinism contract
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_sequences: vec![
                "<|im_end|>".to_string(),
                "</s>".to_string(),
                "<|endoftext|>".to_string(),
            ],
            ..Default::default()
        },
        stream: true,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: Utc::now(),
        api_request: None,
        metadata: HashMap::new(),
    }
}

async fn run_single(
    engine: &(dyn ferrum_interfaces::engine::LlmInferenceEngine + Send + Sync),
    model_id: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<RequestRecord> {
    let request = make_request(model_id, prompt, max_tokens);
    let stream = engine.infer_stream(request).await?;
    Ok(collect_stream(stream).await?)
}

async fn collect_stream(
    mut stream: std::pin::Pin<
        Box<
            dyn futures::Stream<
                    Item = std::result::Result<
                        ferrum_types::StreamChunk,
                        ferrum_types::FerrumError,
                    >,
                > + Send,
        >,
    >,
) -> Result<RequestRecord> {
    let start = Instant::now();
    let mut token_count: u32 = 0;
    let mut first_token_time: Option<Instant> = None;
    let mut last_token_time: Option<Instant> = None;
    let mut itl_ms: Vec<f64> = Vec::new();
    let mut quality_issues = QualityIssueCounts::default();

    let mut got_finish = false;
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                if chunk.token.is_some() {
                    let now = Instant::now();
                    if first_token_time.is_none() {
                        first_token_time = Some(now);
                    } else if let Some(prev) = last_token_time {
                        itl_ms.push((now - prev).as_secs_f64() * 1000.0);
                    }
                    last_token_time = Some(now);
                    token_count = token_count
                        .checked_add(1)
                        .expect("benchmark output token count overflow");
                }
                if chunk.finish_reason.is_some() {
                    got_finish = true;
                    break;
                }
            }
            Err(_) => {
                quality_issues.malformed_stream = 1;
                break;
            }
        }
    }
    if !got_finish {
        quality_issues.missing_done = 1;
        if token_count > 0 {
            eprintln!(
                "  [warn] stream ended without finish_reason ({} tokens)",
                token_count
            );
        }
    }
    if token_count == 0 {
        quality_issues.zero_output_tokens = 1;
    }

    let e2e_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_token_time
        .map(|t| t.duration_since(start).as_secs_f64() * 1000.0)
        .unwrap_or(e2e_ms);

    let success = token_count > 0 && got_finish && quality_issues.request_error_count() == 0;
    let observed_intervals =
        u32::try_from(itl_ms.len()).expect("benchmark ITL interval count overflow");
    Ok(RequestRecord {
        success,
        ttft_ms,
        e2e_ms,
        input_tokens: 0, // CLI bench doesn't tokenize; left as 0
        output_tokens: token_count,
        output_token_count_source: if token_count > 0 {
            OutputTokenCountSource::StreamChunks
        } else {
            OutputTokenCountSource::None
        },
        itl_evidence: RequestItlEvidence::engine(success, token_count, observed_intervals),
        quality_issues,
        itl_ms,
    })
}

// ── Env construction ────────────────────────────────────────────────

fn build_env(cmd: &BenchCommand) -> Env {
    let commit_sha = cmd
        .commit_sha
        .clone()
        .or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    #[allow(unused_mut)]
    let mut features: Vec<String> = Vec::new();
    #[cfg(feature = "metal")]
    features.push("metal".to_string());
    #[cfg(feature = "cuda")]
    features.push("cuda".to_string());

    let mut env = Env::capture_minimal(commit_sha, features);
    if let Some(hw) = cmd.hw_id.clone() {
        env.hw_id = hw;
    }
    env
}

// ── Output formatters ───────────────────────────────────────────────

fn print_human_summary(report: &BenchReport, cmd: &BenchCommand, mode_str: &str) {
    eprintln!();
    eprintln!("{}", "=".repeat(60));
    eprintln!("{}", format!("BENCHMARK RESULTS ({})", mode_str).bold());
    eprintln!("{}", "=".repeat(60));
    eprintln!("Model:             {}", report.model);
    eprintln!("Backend:           {}", report.backend);
    eprintln!("Rounds:            {}", cmd.rounds);
    eprintln!("Repeats:           {}", report.n_repeats);
    eprintln!("Max tokens/req:    {}", cmd.max_tokens);
    if let Some(c) = report.concurrency {
        eprintln!("Concurrency:       {}", c);
    }
    eprintln!("{}", "-".repeat(60));
    let fmt = |s: &ferrum_bench_core::ScalarStats| -> String {
        if report.n_repeats >= 3 {
            format!("{:.1} ± {:.1}", s.mean, s.ci95_hw)
        } else {
            format!("{:.1}", s.mean)
        }
    };
    eprintln!(
        "TTFT_ms      p50={}  p95={}  p99={}",
        fmt(&report.ttft_ms.p50),
        fmt(&report.ttft_ms.p95),
        fmt(&report.ttft_ms.p99)
    );
    eprintln!(
        "TPOT_ms      p50={}  p95={}  p99={}",
        fmt(&report.tpot_ms.p50),
        fmt(&report.tpot_ms.p95),
        fmt(&report.tpot_ms.p99)
    );
    if report.has_complete_itl_evidence() {
        eprintln!(
            "ITL_ms       p50={}  p95={}  p99={}",
            fmt(&report.itl_ms.p50),
            fmt(&report.itl_ms.p95),
            fmt(&report.itl_ms.p99)
        );
    } else {
        eprintln!("ITL_ms       unavailable");
    }
    eprintln!("Output thr   {} tok/s", fmt(&report.output_throughput_tps));
    if cmd.goodput.is_some() {
        eprintln!("Goodput      {} req/s", fmt(&report.goodput_rps));
    }
    eprintln!("env_hash:    {}", report.env_hash);
    eprintln!("{}", "=".repeat(60));
}

fn emit_json(cmd: &BenchCommand, report: &BenchReport) -> Result<()> {
    let pretty = serde_json::to_string_pretty(report).expect("serialize");
    if let Some(out) = cmd.out.as_ref() {
        std::fs::write(out, &pretty).map_err(|e| {
            ferrum_types::FerrumError::model(format!("write {}: {e}", out.display()))
        })?;
        eprintln!("\n→ wrote {}", out.display());
    } else {
        println!("{}", pretty);
    }
    Ok(())
}

fn emit_then_enforce_bench_report(
    cmd: &BenchCommand,
    report: &BenchReport,
    mode_str: &str,
) -> Result<()> {
    match cmd.output.as_str() {
        "human" => print_human_summary(report, cmd, mode_str),
        "json" => emit_json(cmd, report)?,
        other => {
            return Err(ferrum_types::FerrumError::model(format!(
                "unknown --output '{other}': allowed values are human, json"
            )))
        }
    }

    // PLAYBOOK § 1.5: Rust's `static` items don't run Drop on program
    // exit, so the global TraceWriter's flush-on-drop never fires. Call
    // it explicitly here. No-op when FERRUM_TRACE_OUT is unset.
    ferrum_bench_core::trace::flush_global_trace();
    enforce_bench_error_policy(report)
}

fn enforce_bench_error_policy(report: &BenchReport) -> Result<()> {
    let errored = report
        .errored_per_run
        .iter()
        .try_fold(0_u64, |total, value| total.checked_add(*value as u64))
        .ok_or_else(|| ferrum_types::FerrumError::model("benchmark error count overflow"))?;
    if errored > 0 {
        return Err(ferrum_types::FerrumError::model(format!(
            "benchmark measured requests failed: {errored}"
        )));
    }
    Ok(())
}

/// Generate a ~2k token prompt for long-context benchmarking.
fn generate_long_prompt() -> String {
    let base = "The history of artificial intelligence is a fascinating journey through decades of research, breakthroughs, and setbacks. From the early days of symbolic AI in the 1950s, through the AI winters, to the modern era of deep learning and large language models, the field has undergone remarkable transformations. ";
    let mut prompt = String::with_capacity(8192);
    prompt.push_str("Please provide a comprehensive analysis of the following text, identifying key themes, patterns, and insights:\n\n");
    for _ in 0..25 {
        prompt.push_str(base);
    }
    prompt.push_str("\n\nNow analyze the above text in detail:");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command() -> BenchCommand {
        BenchCommand {
            model: "test".to_string(),
            rounds: 3,
            max_tokens: 4,
            backend: "cpu".to_string(),
            prompt: "hello".to_string(),
            concurrency: 2,
            long_context: false,
            kv_dtype: None,
            n_repeats: 1,
            goodput: None,
            output: "json".to_string(),
            out: None,
            hw_id: None,
            commit_sha: None,
        }
    }

    #[test]
    fn validates_expected_request_count_and_numeric_inputs() {
        let mut cmd = command();
        assert_eq!(measured_request_count(&cmd).unwrap(), 6);
        assert_eq!(
            measured_request_count(&cmd).unwrap() as usize,
            cmd.rounds * cmd.concurrency
        );
        assert!(validate_command(&cmd).is_ok());
        cmd.concurrency = 1;
        assert_eq!(measured_request_count(&cmd).unwrap() as usize, cmd.rounds);
        assert!(validate_command(&cmd).is_ok());
        cmd.n_repeats = 0;
        assert!(validate_command(&cmd).is_err());
        cmd.n_repeats = 1;
        cmd.goodput = Some(Slo {
            ttft_p99_ms: f64::INFINITY,
            ..Slo::default()
        });
        assert!(validate_command(&cmd).is_err());
    }

    #[test]
    fn failed_bench_record_is_explicit_error_evidence() {
        let mut quality = QualityIssueCounts::default();
        quality.panic = 1;
        let record = failed_bench_record(quality);
        assert!(!record.success);
        assert_eq!(record.quality_issues.panic, 1);
        assert_eq!(
            record.output_token_count_source,
            OutputTokenCountSource::None
        );
        assert_eq!(
            record.itl_evidence.source,
            ItlEvidenceSource::EngineTokenEvents
        );
    }

    #[test]
    fn failed_bench_report_is_written_before_nonzero_result() {
        let out = std::env::temp_dir().join(format!(
            "ferrum-bench-failed-report-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&out);
        let mut quality = QualityIssueCounts::default();
        quality.malformed_stream = 1;
        let report = compute_metrics(
            "test".to_string(),
            "cpu".to_string(),
            Scenario::Cli,
            None,
            None,
            1,
            1,
            0,
            Slo::default(),
            vec![RunRecord {
                records: vec![failed_bench_record(quality)],
                expected_requests: 1,
                duration_s: 1.0,
                warmup: Default::default(),
            }],
            Env::default(),
        );
        let mut cmd = command();
        cmd.out = Some(out.clone());
        let error = emit_then_enforce_bench_report(&cmd, &report, "test")
            .expect_err("failed measured request must return nonzero");
        assert!(error.to_string().contains("measured requests failed"));
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(json["errored_per_run"], serde_json::json!([1]));
        assert_eq!(json["repeat_metrics"][0]["itl_ineligible_requests"], 1);
        let _ = std::fs::remove_file(out);
    }
}
