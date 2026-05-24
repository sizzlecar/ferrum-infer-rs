//! Tokenizer-aware HTTP bench client — emits the canonical schema
//! defined in `crates/ferrum-bench-core` and gated by
//! `docs/bench/PLAYBOOK.md` § 7.
//!
//! Scenarios (PLAYBOOK § 2):
//!   - **Closed-loop** — `--concurrency K`, K workers in tight send→wait
//!     loop. Headline = throughput. The "capacity knee" measurement.
//!   - **Concurrency sweep** — `--concurrency-sweep 1,4,16,32`, runs N
//!     closed-loop cells back-to-back to find the knee.
//!   - **Open-loop** — `--request-rate R`, Poisson(R) arrivals. The
//!     ONLY scenario in which goodput is meaningful (§ 0.4).
//!
//! Each cell runs `--n-repeats` independent times; the per-run percentiles
//! are aggregated with mean + sample stddev + Student-t 95% CI half-width.
//! Cells where `n_repeats < 3` emit `mean` only (PLAYBOOK § 0.4 contract).

use clap::Args;
use colored::*;
use ferrum_bench_core::{
    arrivals::poisson_arrival_times, compute_metrics, BenchReport, Env, RequestRecord, RunRecord,
    Scenario, Slo,
};
use ferrum_types::Result;
use rand::Rng;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio_stream::StreamExt;

use crate::config::CliConfig;

#[derive(Args, Clone)]
pub struct BenchServeCommand {
    /// Base URL of the ferrum (or other OpenAI-compatible) server.
    #[arg(long)]
    pub base_url: String,

    /// Model identifier sent in request body's `model` field.
    /// Use the local path so `vllm bench serve` numbers compare 1:1.
    #[arg(long)]
    pub model: String,

    /// Path to the model directory containing `tokenizer.json`. Used to
    /// generate exact-length random token sequences.
    #[arg(long)]
    pub tokenizer: PathBuf,

    // ─── Workload selection (pick one mode) ────────────────────────
    /// Closed-loop concurrency (single cell). Default when no other
    /// mode is given. Alias: `--max-concurrency` (legacy vLLM naming).
    #[arg(long, default_value_t = 32, alias = "max-concurrency")]
    pub concurrency: u32,

    /// Closed-loop concurrency sweep. Overrides `--concurrency`. E.g.
    /// `--concurrency-sweep 1,4,16,32` runs four closed-loop cells.
    #[arg(long, value_delimiter = ',')]
    pub concurrency_sweep: Vec<u32>,

    /// Open-loop arrival rate (req/s, Poisson). When set, overrides
    /// the closed-loop modes — this is the goodput-relevant scenario.
    #[arg(long)]
    pub request_rate: Option<f64>,

    // ─── Dataset ───────────────────────────────────────────────────
    /// Dataset: `random` (tokenizer-aware), `sharegpt` (load from JSONL),
    /// `shared-prefix` (1024-tok shared prefix + unique suffix).
    /// PLAYBOOK § 2 Scenario A (sharegpt) / Scenario C (shared-prefix).
    #[arg(long, default_value = "random")]
    pub dataset: String,

    /// Number of *tokens* per random prompt (`--dataset random` only).
    #[arg(long, default_value_t = 256)]
    pub random_input_len: usize,

    /// Max output tokens per request.
    #[arg(long, default_value_t = 128)]
    pub random_output_len: usize,

    /// Path to a ShareGPT-format JSONL file (`--dataset sharegpt`).
    /// Each line should be a `{"conversations": [{"from": "...", "value":
    /// "..."}, ...]}` object (HF anon8231489123/ShareGPT_Vicuna format).
    #[arg(long)]
    pub sharegpt_path: Option<PathBuf>,

    /// Shared prefix length in *tokens* (`--dataset shared-prefix` only).
    #[arg(long, default_value_t = 1024)]
    pub shared_prefix_len: usize,

    /// Per-request unique suffix length in *tokens* (`--dataset shared-prefix`).
    #[arg(long, default_value_t = 64)]
    pub shared_suffix_len: usize,

    // ─── Run shape ─────────────────────────────────────────────────
    /// Total prompts sent per run (warmup is counted separately).
    #[arg(long, default_value_t = 100)]
    pub num_prompts: u32,

    /// Warmup requests sent before measurement begins each run.
    /// Discarded from the metrics; PLAYBOOK § 0.3 mandates ≥ 10 for
    /// committed reports.
    #[arg(long, default_value_t = 10)]
    pub warmup_requests: u32,

    /// Independent repeats per cell. PLAYBOOK § 0.4: ≥ 3 unlocks
    /// stddev + CI95; n < 3 emits mean only.
    #[arg(long, default_value_t = 1)]
    pub n_repeats: u32,

    /// SLO triple for goodput. Format: `ttft:500 tpot:50 e2el:30000`
    /// (or comma-separated). Goodput is reported only when all three
    /// are set.
    #[arg(long, value_parser = parse_slo)]
    pub goodput: Option<Slo>,

    /// Per-request HTTP timeout in seconds.
    #[arg(long, default_value_t = 600.0)]
    pub timeout: f64,

    // ─── Output ────────────────────────────────────────────────────
    /// Output format: `json` (BenchReport), `jsonl` (append one
    /// `BenchReport` per line — used by `scripts/compare-commits.sh`),
    /// `md` (human-readable markdown).
    #[arg(long, default_value = "json")]
    pub output: String,

    /// Output file path. For `jsonl`, the file is opened append-mode so
    /// repeated invocations against the same DB accumulate rows.
    /// Alias: `--result-file` (legacy).
    #[arg(long, alias = "result-file")]
    pub out: Option<PathBuf>,

    // ─── Env / parity ──────────────────────────────────────────────
    /// Override `env.hw_id` (defaults to auto-detected).
    #[arg(long)]
    pub hw_id: Option<String>,

    /// Override `env.commit_sha` (defaults to `git rev-parse --short HEAD`).
    #[arg(long)]
    pub commit_sha: Option<String>,

    /// Tag string written into the report's `model` field's suffix —
    /// useful for the `bench_vs_vllm.sh` script which tags ferrum/vllm.
    #[arg(long)]
    pub tag: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────
// SLO parser
// ─────────────────────────────────────────────────────────────────────

pub(super) fn parse_slo(s: &str) -> std::result::Result<Slo, String> {
    let mut ttft: Option<f64> = None;
    let mut tpot: Option<f64> = None;
    let mut e2el: Option<f64> = None;
    for tok in s.split(|c: char| c == ',' || c.is_whitespace()) {
        if tok.is_empty() {
            continue;
        }
        let (k, v) = tok
            .split_once(':')
            .ok_or_else(|| format!("bad SLO token '{tok}', expected key:value"))?;
        let v: f64 = v
            .parse()
            .map_err(|e| format!("bad SLO value '{v}': {e}"))?;
        match k {
            "ttft" => ttft = Some(v),
            "tpot" => tpot = Some(v),
            "e2el" | "e2e" => e2el = Some(v),
            other => return Err(format!("unknown SLO key '{other}'")),
        }
    }
    Ok(Slo {
        ttft_p99_ms: ttft.ok_or("missing ttft in --goodput")?,
        tpot_p99_ms: tpot.ok_or("missing tpot in --goodput")?,
        e2e_p99_ms: e2el.ok_or("missing e2el in --goodput")?,
    })
}

// ─────────────────────────────────────────────────────────────────────
// OpenAI SSE chunk types
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Option<Vec<OpenAiStreamChoice>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: Option<OpenAiStreamDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────
// Single-request streamer
// ─────────────────────────────────────────────────────────────────────

async fn stream_one(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: String,
    input_tokens: u32,
    max_tokens: usize,
    timeout_s: f64,
) -> RequestRecord {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": true,
        "temperature": 0.0,
    });
    let start = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut last_token_time: Option<Instant> = None;
    let mut output_tokens: u32 = 0;
    let mut itl_ms: Vec<f64> = Vec::new();

    let resp = match client
        .post(format!("{}/v1/chat/completions", base_url))
        .json(&body)
        .timeout(Duration::from_secs_f64(timeout_s))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[err] post: {}", e);
            return failed_record(input_tokens, start);
        }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let txt = resp.text().await.unwrap_or_default();
        eprintln!("[err] http {}: {}", status, &txt[..txt.len().min(200)]);
        return failed_record(input_tokens, start);
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[err] stream: {}", e);
                break;
            }
        };
        buf.push_str(&String::from_utf8_lossy(&chunk));
        loop {
            let Some(nl) = buf.find('\n') else { break };
            let line = buf[..nl].trim().to_string();
            buf.drain(..=nl);
            if !line.starts_with("data:") {
                continue;
            }
            let payload = line.trim_start_matches("data:").trim();
            if payload == "[DONE]" {
                let e2e_ms = start.elapsed().as_secs_f64() * 1000.0;
                let ttft_ms = first_token_time
                    .map(|t| t.duration_since(start).as_secs_f64() * 1000.0)
                    .unwrap_or(e2e_ms);
                return RequestRecord {
                    success: output_tokens > 0,
                    ttft_ms,
                    e2e_ms,
                    input_tokens,
                    output_tokens,
                    itl_ms,
                };
            }
            if let Ok(c) = serde_json::from_str::<OpenAiStreamChunk>(payload) {
                if let Some(choices) = c.choices {
                    if let Some(first) = choices.into_iter().next() {
                        if let Some(delta) = first.delta {
                            if delta.content.is_some() {
                                let now = Instant::now();
                                if first_token_time.is_none() {
                                    first_token_time = Some(now);
                                } else if let Some(prev) = last_token_time {
                                    itl_ms.push((now - prev).as_secs_f64() * 1000.0);
                                }
                                last_token_time = Some(now);
                                output_tokens += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    // Stream ended without [DONE] sentinel — still report what we got.
    let e2e_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_token_time
        .map(|t| t.duration_since(start).as_secs_f64() * 1000.0)
        .unwrap_or(e2e_ms);
    RequestRecord {
        success: output_tokens > 0,
        ttft_ms,
        e2e_ms,
        input_tokens,
        output_tokens,
        itl_ms,
    }
}

fn failed_record(input_tokens: u32, start: Instant) -> RequestRecord {
    RequestRecord {
        success: false,
        ttft_ms: 0.0,
        e2e_ms: start.elapsed().as_secs_f64() * 1000.0,
        input_tokens,
        output_tokens: 0,
        itl_ms: vec![],
    }
}

// ─────────────────────────────────────────────────────────────────────
// Dataset generation
// ─────────────────────────────────────────────────────────────────────

/// Draw a random token sequence of exact length `n_tokens`.
fn gen_random_prompt(tok: &tokenizers::Tokenizer, n_tokens: usize, rng: &mut impl Rng) -> String {
    let vocab_size = tok.get_vocab_size(false) as u32;
    let lo: u32 = 256.min(vocab_size.saturating_sub(1));
    let hi: u32 = vocab_size.saturating_sub(1);
    let ids: Vec<u32> = (0..n_tokens).map(|_| rng.random_range(lo..=hi)).collect();
    tok.decode(&ids, false)
        .unwrap_or_else(|_| "hello world ".repeat(n_tokens / 2))
}

fn build_prompts(
    cmd: &BenchServeCommand,
    tok: &tokenizers::Tokenizer,
    rng: &mut impl Rng,
    count: usize,
) -> Result<Vec<String>> {
    match cmd.dataset.as_str() {
        "random" => Ok((0..count)
            .map(|_| gen_random_prompt(tok, cmd.random_input_len, rng))
            .collect()),
        "shared-prefix" => Ok(gen_shared_prefix_prompts(
            tok,
            count,
            cmd.shared_prefix_len,
            cmd.shared_suffix_len,
            rng,
        )),
        "sharegpt" => {
            let p = cmd.sharegpt_path.as_ref().ok_or_else(|| {
                ferrum_types::FerrumError::model(
                    "--dataset sharegpt requires --sharegpt-path PATH",
                )
            })?;
            load_sharegpt_prompts(p, count, rng)
        }
        other => Err(ferrum_types::FerrumError::model(format!(
            "unknown --dataset '{}': allowed values are random, sharegpt, shared-prefix",
            other
        ))),
    }
}

/// Generate `count` prompts that all share a 1024-token (or whatever
/// `shared_prefix_len` is) prefix, with a unique random suffix per
/// request. The shared prefix is sampled ONCE from the tokenizer's
/// mid-vocab range and decoded into a UTF-8 string the server can
/// re-tokenize back to ~the same length.
///
/// Used by PLAYBOOK § 2 Scenario C (prefix-cache thundering herd):
/// closed-loop ShareGPT can't reproduce this because each ShareGPT
/// conversation has a different prefix.
fn gen_shared_prefix_prompts(
    tok: &tokenizers::Tokenizer,
    count: usize,
    prefix_len: usize,
    suffix_len: usize,
    rng: &mut impl Rng,
) -> Vec<String> {
    let vocab_size = tok.get_vocab_size(false) as u32;
    let lo: u32 = 256.min(vocab_size.saturating_sub(1));
    let hi: u32 = vocab_size.saturating_sub(1);
    // Shared prefix sampled once.
    let prefix_ids: Vec<u32> = (0..prefix_len).map(|_| rng.random_range(lo..=hi)).collect();
    let prefix = tok
        .decode(&prefix_ids, false)
        .unwrap_or_else(|_| "hello world ".repeat(prefix_len / 2));
    (0..count)
        .map(|_| {
            let suffix_ids: Vec<u32> =
                (0..suffix_len).map(|_| rng.random_range(lo..=hi)).collect();
            let suffix = tok
                .decode(&suffix_ids, false)
                .unwrap_or_else(|_| "x ".repeat(suffix_len / 2));
            // Insert a newline so the prefix is a clear boundary — helps
            // server-side prefix-cache hashing key on the same prefix
            // even when suffixes differ.
            format!("{prefix}\n{suffix}")
        })
        .collect()
}

/// Load up to `count` user-turn prompts from a ShareGPT-format JSONL.
///
/// Accepts either:
///   - HF `anon8231489123/ShareGPT_Vicuna` format: each line is
///     `{"id": "...", "conversations": [{"from": "human"/"gpt", "value": "..."}]}`
///   - vLLM-style: `{"input": "..."}` per line
///
/// Picks the first `human` turn per conversation (matches vLLM's
/// benchmark_serving.py heuristic). If `count` < records available,
/// randomly samples; if `count` > available, cycles with replacement.
fn load_sharegpt_prompts(
    path: &std::path::Path,
    count: usize,
    rng: &mut impl Rng,
) -> Result<Vec<String>> {
    use std::io::BufRead;
    let f = std::fs::File::open(path).map_err(|e| {
        ferrum_types::FerrumError::model(format!("open sharegpt {}: {e}", path.display()))
    })?;
    let mut prompts: Vec<String> = Vec::new();
    for (idx, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| {
            ferrum_types::FerrumError::model(format!("read line {idx} of {}: {e}", path.display()))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[warn] sharegpt line {idx}: skip (parse error: {e})");
                continue;
            }
        };
        // Try ShareGPT-Vicuna format first.
        let prompt: Option<String> = v
            .get("conversations")
            .and_then(|c| c.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find(|t| t.get("from").and_then(|f| f.as_str()) == Some("human"))
                    .and_then(|t| t.get("value").and_then(|x| x.as_str()).map(|s| s.to_string()))
            })
            // Fallback: simple {"input": "..."} format.
            .or_else(|| {
                v.get("input")
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string())
            });
        if let Some(p) = prompt {
            if !p.is_empty() {
                prompts.push(p);
            }
        }
    }
    if prompts.is_empty() {
        return Err(ferrum_types::FerrumError::model(format!(
            "sharegpt {}: no usable prompts found",
            path.display()
        )));
    }
    // Sample `count` with replacement (cycles deterministically for the
    // given seed via rng).
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let idx = rng.random_range(0..prompts.len());
        out.push(prompts[idx].clone());
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────
// Scenario runners
// ─────────────────────────────────────────────────────────────────────

struct RunContext {
    client: Arc<reqwest::Client>,
    base_url: Arc<String>,
    model: Arc<String>,
    max_out: usize,
    input_tokens: u32,
    timeout_s: f64,
}

/// Closed-loop: K workers in a tight loop. Warmup prompts run sequentially
/// at full concurrency (just to load caches), then the measurement window
/// begins.
async fn run_closed_loop(
    ctx: &RunContext,
    prompts: Vec<String>,
    warmup_requests: u32,
    concurrency: u32,
) -> RunRecord {
    let n_warmup = warmup_requests as usize;
    let total = prompts.len();
    assert!(
        total > n_warmup,
        "num_prompts ({total}) must exceed warmup_requests ({n_warmup})"
    );

    // Warmup window — fire and discard.
    {
        let sem = Arc::new(Semaphore::new(concurrency as usize));
        let mut handles = Vec::new();
        for prompt in prompts.iter().take(n_warmup) {
            let permit = sem.clone().acquire_owned().await.expect("semaphore");
            let ctx_c = ctx.clone_inner();
            let p = prompt.clone();
            handles.push(tokio::spawn(async move {
                let _g = permit;
                stream_one(
                    &ctx_c.client,
                    &ctx_c.base_url,
                    &ctx_c.model,
                    p,
                    ctx_c.input_tokens,
                    ctx_c.max_out,
                    ctx_c.timeout_s,
                )
                .await
            }));
        }
        for h in handles {
            let _ = h.await;
        }
    }

    // Measurement window.
    let sem = Arc::new(Semaphore::new(concurrency as usize));
    let start = Instant::now();
    let mut handles = Vec::with_capacity(total - n_warmup);
    for prompt in prompts.into_iter().skip(n_warmup) {
        let permit = sem.clone().acquire_owned().await.expect("semaphore");
        let ctx_c = ctx.clone_inner();
        handles.push(tokio::spawn(async move {
            let _g = permit;
            stream_one(
                &ctx_c.client,
                &ctx_c.base_url,
                &ctx_c.model,
                prompt,
                ctx_c.input_tokens,
                ctx_c.max_out,
                ctx_c.timeout_s,
            )
            .await
        }));
    }
    let mut records = Vec::with_capacity(handles.len());
    for h in handles {
        if let Ok(r) = h.await {
            records.push(r);
        }
    }
    let duration_s = start.elapsed().as_secs_f64();
    RunRecord { records, duration_s }
}

/// Open-loop: Poisson(rate) arrivals. The arrival schedule is fixed
/// before sending so that slow responses don't push later arrivals.
async fn run_open_loop(
    ctx: &RunContext,
    prompts: Vec<String>,
    warmup_requests: u32,
    rate: f64,
) -> RunRecord {
    let n_warmup = warmup_requests as usize;
    let total = prompts.len();
    assert!(total > n_warmup);

    // Warmup: send a few sequentially to load caches.
    for prompt in prompts.iter().take(n_warmup) {
        let _ = stream_one(
            &ctx.client,
            &ctx.base_url,
            &ctx.model,
            prompt.clone(),
            ctx.input_tokens,
            ctx.max_out,
            ctx.timeout_s,
        )
        .await;
    }

    // Pre-compute arrival schedule (re-zeroed after warmup).
    let mut rng = rand::rng();
    let measurement_count = total - n_warmup;
    let schedule = poisson_arrival_times(rate, measurement_count, &mut rng);

    let start = Instant::now();
    let mut handles = Vec::with_capacity(measurement_count);
    for (i, prompt) in prompts.into_iter().skip(n_warmup).enumerate() {
        let target = schedule[i];
        let now = start.elapsed().as_secs_f64();
        if target > now {
            tokio::time::sleep(Duration::from_secs_f64(target - now)).await;
        }
        let ctx_c = ctx.clone_inner();
        handles.push(tokio::spawn(async move {
            stream_one(
                &ctx_c.client,
                &ctx_c.base_url,
                &ctx_c.model,
                prompt,
                ctx_c.input_tokens,
                ctx_c.max_out,
                ctx_c.timeout_s,
            )
            .await
        }));
    }
    let mut records = Vec::with_capacity(handles.len());
    for h in handles {
        if let Ok(r) = h.await {
            records.push(r);
        }
    }
    let duration_s = start.elapsed().as_secs_f64();
    RunRecord { records, duration_s }
}

impl RunContext {
    fn clone_inner(&self) -> Self {
        Self {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            max_out: self.max_out,
            input_tokens: self.input_tokens,
            timeout_s: self.timeout_s,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Env construction
// ─────────────────────────────────────────────────────────────────────

fn build_env(cmd: &BenchServeCommand, features: Vec<String>) -> Env {
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

    let mut env = Env::capture_minimal(commit_sha, features);
    if let Some(hw) = cmd.hw_id.clone() {
        env.hw_id = hw;
    }
    env
}

fn detect_features() -> Vec<String> {
    let mut v = Vec::new();
    #[cfg(feature = "metal")]
    v.push("metal".to_string());
    #[cfg(feature = "cuda")]
    v.push("cuda".to_string());
    #[cfg(feature = "vllm-marlin")]
    v.push("vllm-marlin".to_string());
    #[cfg(feature = "vllm-moe-marlin")]
    v.push("vllm-moe-marlin".to_string());
    #[cfg(feature = "vllm-paged-attn-v2")]
    v.push("vllm-paged-attn-v2".to_string());
    #[cfg(feature = "triton-kernels")]
    v.push("triton-kernels".to_string());
    v.sort();
    v.dedup();
    v
}

// ─────────────────────────────────────────────────────────────────────
// Cell execution + multi-cell driver
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Cell {
    Closed(u32),
    Open(f64),
}

async fn execute_cell(cmd: &BenchServeCommand, ctx: &RunContext, cell: Cell) -> Result<BenchReport> {
    let backend = if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
        "cuda"
    } else if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cpu"
    };

    let tokenizer_path = cmd.tokenizer.join("tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        ferrum_types::FerrumError::model(format!(
            "Load tokenizer at {}: {}",
            tokenizer_path.display(),
            e
        ))
    })?;
    let mut rng = rand::rng();

    let total_prompts = (cmd.num_prompts + cmd.warmup_requests) as usize;

    let mut runs: Vec<RunRecord> = Vec::with_capacity(cmd.n_repeats as usize);
    for repeat_idx in 0..cmd.n_repeats {
        let prompts = build_prompts(cmd, &tok, &mut rng, total_prompts)?;
        eprintln!(
            "{}",
            format!(
                "  cell {} — repeat {}/{} ({} prompts including {} warmup)",
                cell_label(cell),
                repeat_idx + 1,
                cmd.n_repeats,
                total_prompts,
                cmd.warmup_requests
            )
            .dimmed()
        );
        let run = match cell {
            Cell::Closed(c) => run_closed_loop(ctx, prompts, cmd.warmup_requests, c).await,
            Cell::Open(r) => run_open_loop(ctx, prompts, cmd.warmup_requests, r).await,
        };
        eprintln!(
            "    {} completed / {} errored / {:.1}s",
            run.n_completed(),
            run.n_errored(),
            run.duration_s
        );
        runs.push(run);
    }

    let env = build_env(cmd, detect_features());
    let slo = cmd.goodput.unwrap_or(Slo {
        ttft_p99_ms: f64::INFINITY,
        tpot_p99_ms: f64::INFINITY,
        e2e_p99_ms: f64::INFINITY,
    });
    let model_field = match &cmd.tag {
        Some(t) => format!("{}#{}", cmd.model, t),
        None => cmd.model.clone(),
    };

    let (scenario, concurrency, request_rate) = match cell {
        Cell::Closed(c) => (Scenario::ClosedLoop, Some(c), None),
        Cell::Open(r) => (Scenario::OpenLoop, None, Some(r)),
    };

    Ok(compute_metrics(
        model_field,
        backend.to_string(),
        scenario,
        concurrency,
        request_rate,
        cmd.random_input_len as u32,
        cmd.random_output_len as u32,
        cmd.warmup_requests,
        slo,
        runs,
        env,
    ))
}

fn cell_label(cell: Cell) -> String {
    match cell {
        Cell::Closed(c) => format!("closed_loop c={c}"),
        Cell::Open(r) => format!("open_loop rate={r}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Top-level entry
// ─────────────────────────────────────────────────────────────────────

pub async fn execute(cmd: BenchServeCommand, _cfg: CliConfig) -> Result<()> {
    eprintln!(
        "{}",
        format!(
            "ferrum bench-serve — dataset={} num_prompts={} warmup={} n_repeats={}",
            cmd.dataset, cmd.num_prompts, cmd.warmup_requests, cmd.n_repeats
        )
        .dimmed()
    );
    if cmd.n_repeats < 3 {
        eprintln!(
            "{}",
            "  [warn] n_repeats < 3 — emitting mean only, no stddev/CI95 (PLAYBOOK § 0.4)".yellow()
        );
    }

    // Build cells from CLI flags. Order: request_rate > concurrency_sweep > concurrency.
    let cells: Vec<Cell> = if let Some(rate) = cmd.request_rate {
        vec![Cell::Open(rate)]
    } else if !cmd.concurrency_sweep.is_empty() {
        cmd.concurrency_sweep.iter().copied().map(Cell::Closed).collect()
    } else {
        vec![Cell::Closed(cmd.concurrency)]
    };

    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .build()
            .map_err(|e| ferrum_types::FerrumError::model(format!("reqwest client: {e}")))?,
    );
    let ctx = RunContext {
        client,
        base_url: Arc::new(cmd.base_url.clone()),
        model: Arc::new(cmd.model.clone()),
        max_out: cmd.random_output_len,
        input_tokens: cmd.random_input_len as u32,
        timeout_s: cmd.timeout,
    };

    let mut reports: Vec<BenchReport> = Vec::with_capacity(cells.len());
    for cell in cells {
        eprintln!("{}", format!("→ {}", cell_label(cell)).bold());
        let r = execute_cell(&cmd, &ctx, cell).await?;
        emit_summary_line(&r);
        reports.push(r);
    }

    // Emit final report.
    match cmd.output.as_str() {
        "json" => emit_json(&cmd, &reports)?,
        "jsonl" => emit_jsonl(&cmd, &reports)?,
        "md" => emit_markdown(&cmd, &reports)?,
        other => {
            return Err(ferrum_types::FerrumError::model(format!(
                "unknown --output '{other}': allowed values are json, jsonl, md"
            )))
        }
    }
    Ok(())
}

fn emit_jsonl(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    use std::io::Write as _;
    let out_path = cmd.out.as_ref().ok_or_else(|| {
        ferrum_types::FerrumError::model("--output jsonl requires --out PATH (append-mode log)")
    })?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .map_err(|e| ferrum_types::FerrumError::model(format!("open {}: {e}", out_path.display())))?;
    for r in reports {
        let line = serde_json::to_string(r).expect("serialize");
        writeln!(f, "{}", line)
            .map_err(|e| ferrum_types::FerrumError::model(format!("write jsonl: {e}")))?;
    }
    eprintln!(
        "\n→ appended {} record(s) to {}",
        reports.len(),
        out_path.display()
    );
    Ok(())
}

fn emit_markdown(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    use ferrum_bench_core::report::{render_single, render_sweep};
    let md = if reports.len() == 1 {
        render_single(&reports[0])
    } else {
        render_sweep(reports)
    };
    if let Some(out) = cmd.out.as_ref() {
        std::fs::write(out, &md)
            .map_err(|e| ferrum_types::FerrumError::model(format!("write {}: {e}", out.display())))?;
        eprintln!("\n→ wrote {}", out.display());
    } else {
        println!("{}", md);
    }
    Ok(())
}

fn emit_summary_line(r: &BenchReport) {
    let scenario_str = match r.scenario {
        Scenario::ClosedLoop => format!("c={}", r.concurrency.unwrap_or(0)),
        Scenario::OpenLoop => format!("rate={}", r.request_rate.unwrap_or(0.0)),
        Scenario::SharedPrefix => "shared_prefix".to_string(),
        Scenario::Cli => "cli".to_string(),
    };
    let ci = if r.n_repeats >= 3 {
        format!(" (n_repeats={}, ± = ci95_hw)", r.n_repeats)
    } else {
        format!(" (n_repeats={}, no CI)", r.n_repeats)
    };
    eprintln!("    {} {}{}", "summary".bold(), scenario_str, ci);
    fmt_metric("TTFT_ms ", &r.ttft_ms, r.n_repeats);
    fmt_metric("TPOT_ms ", &r.tpot_ms, r.n_repeats);
    fmt_metric("ITL_ms  ", &r.itl_ms, r.n_repeats);
    let thr = &r.output_throughput_tps;
    let good = &r.goodput_rps;
    if r.n_repeats >= 3 {
        eprintln!(
            "      throughput      {:.1} ± {:.1} tok/s",
            thr.mean, thr.ci95_hw
        );
        eprintln!(
            "      goodput         {:.2} ± {:.2} req/s",
            good.mean, good.ci95_hw
        );
    } else {
        eprintln!("      throughput      {:.1} tok/s", thr.mean);
        eprintln!("      goodput         {:.2} req/s", good.mean);
    }
}

fn fmt_metric(name: &str, m: &ferrum_bench_core::MetricSet, n_repeats: u32) {
    if n_repeats >= 3 {
        eprintln!(
            "      {} p50={:.1}±{:.1}  p95={:.1}±{:.1}  p99={:.1}±{:.1}",
            name, m.p50.mean, m.p50.ci95_hw, m.p95.mean, m.p95.ci95_hw, m.p99.mean, m.p99.ci95_hw
        );
    } else {
        eprintln!(
            "      {} p50={:.1}  p95={:.1}  p99={:.1}",
            name, m.p50.mean, m.p95.mean, m.p99.mean
        );
    }
}

fn emit_json(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    let value = if reports.len() == 1 {
        serde_json::to_value(&reports[0]).expect("serialize")
    } else {
        serde_json::to_value(reports).expect("serialize")
    };
    let pretty = serde_json::to_string_pretty(&value).expect("pretty");
    if let Some(out) = cmd.out.as_ref() {
        std::fs::write(out, &pretty)
            .map_err(|e| ferrum_types::FerrumError::model(format!("write {}: {e}", out.display())))?;
        eprintln!("\n→ wrote {}", out.display());
    } else {
        println!("{}", pretty);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slo_parses_space_separated() {
        let s = parse_slo("ttft:500 tpot:50 e2el:30000").unwrap();
        assert_eq!(s.ttft_p99_ms, 500.0);
        assert_eq!(s.tpot_p99_ms, 50.0);
        assert_eq!(s.e2e_p99_ms, 30000.0);
    }

    #[test]
    fn slo_parses_comma_separated() {
        let s = parse_slo("ttft:500,tpot:50,e2el:30000").unwrap();
        assert_eq!(s.ttft_p99_ms, 500.0);
    }

    #[test]
    fn slo_accepts_e2e_alias() {
        let s = parse_slo("ttft:1 tpot:2 e2e:3").unwrap();
        assert_eq!(s.e2e_p99_ms, 3.0);
    }

    #[test]
    fn slo_rejects_missing_key() {
        assert!(parse_slo("ttft:500 tpot:50").is_err());
    }

    #[test]
    fn slo_rejects_unknown_key() {
        assert!(parse_slo("ttft:500 tpot:50 e2el:30000 bogus:1").is_err());
    }
}
