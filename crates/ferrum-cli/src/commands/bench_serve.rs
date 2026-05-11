//! Tokenizer-aware HTTP bench client — apples-to-apples with
//! `vllm bench serve --dataset-name random`.
//!
//! Why: external benchmarks (PR #102's reported 391 tok/s at c=32 on
//! Qwen3-30B-A3B) used `vllm bench serve` which draws random *tokens*
//! (256 tokens per prompt by default) via the model's tokenizer. A
//! generic httpx-based client that draws random *words* of English
//! noise produces prompts that tokenize to ~1.4× more tokens, which
//! inflates prefill work and stalls continuous-batching decode iters
//! — at c=32, that's a ~25% throughput gap vs the published number on
//! the same engine binary.
//!
//! This subcommand uses the model's own `tokenizer.json` (already
//! loaded by `ferrum serve`) to draw exact-length random token
//! sequences, then POSTs them via `reqwest` async with `stream=true`
//! to `/v1/chat/completions` and measures the same metrics vllm's
//! bench prints.
//!
//! Usage:
//!   ferrum bench-serve --base-url http://127.0.0.1:8800 \
//!     --model /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
//!     --random-input-len 256 --random-output-len 128 \
//!     --num-prompts 128 --max-concurrency 32

use clap::Args;
use colored::*;
use ferrum_types::Result;
use rand::Rng;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tokio_stream::StreamExt;

use crate::config::CliConfig;

#[derive(Args)]
pub struct BenchServeCommand {
    /// Base URL of the ferrum (or other OpenAI-compatible) server.
    #[arg(long)]
    pub base_url: String,

    /// Model identifier sent in the request body (`model` field).
    /// Most servers accept any string here; use the local path so
    /// `vllm bench serve`'s numbers can compare 1:1.
    #[arg(long)]
    pub model: String,

    /// Path to the model directory containing `tokenizer.json`.
    /// Used to generate exact-length random token sequences.
    #[arg(long)]
    pub tokenizer: PathBuf,

    /// Number of *tokens* per prompt (tokenized).
    #[arg(long, default_value_t = 256)]
    pub random_input_len: usize,

    /// Max output tokens per request.
    #[arg(long, default_value_t = 128)]
    pub random_output_len: usize,

    /// Total prompts to send.
    #[arg(long, default_value_t = 128)]
    pub num_prompts: usize,

    /// Max in-flight concurrent requests.
    #[arg(long, default_value_t = 32)]
    pub max_concurrency: usize,

    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 600.0)]
    pub timeout: f64,

    /// Write JSON result to file.
    #[arg(long)]
    pub result_file: Option<PathBuf>,
}

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

struct RequestResult {
    success: bool,
    ttft_ms: f64,
    e2e_ms: f64,
    output_tokens: usize,
}

/// Draw a random token sequence of exact length `n_tokens` from
/// the tokenizer's vocab. We pick mid-range tokens (skip special
/// tokens at low IDs) and decode to a UTF-8 string the server can
/// re-tokenize back to ~the same length.
fn gen_random_prompt(tok: &tokenizers::Tokenizer, n_tokens: usize, rng: &mut impl Rng) -> String {
    let vocab_size = tok.get_vocab_size(false) as u32;
    // Skip the first 256 IDs (typically special tokens / control chars).
    let lo: u32 = 256.min(vocab_size.saturating_sub(1));
    let hi: u32 = vocab_size.saturating_sub(1);
    let ids: Vec<u32> = (0..n_tokens).map(|_| rng.random_range(lo..=hi)).collect();
    tok.decode(&ids, false)
        .unwrap_or_else(|_| "hello world ".repeat(n_tokens / 2))
}

async fn stream_one(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: String,
    max_tokens: usize,
    timeout_s: f64,
) -> RequestResult {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": true,
        "temperature": 0.0,
    });
    let start = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut output_tokens: usize = 0;

    let resp = match client
        .post(format!("{}/v1/chat/completions", base_url))
        .json(&body)
        .timeout(std::time::Duration::from_secs_f64(timeout_s))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[err] post: {}", e);
            return RequestResult {
                success: false,
                ttft_ms: 0.0,
                e2e_ms: 0.0,
                output_tokens: 0,
            };
        }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let txt = resp.text().await.unwrap_or_default();
        eprintln!("[err] http {}: {}", status, &txt[..txt.len().min(200)]);
        return RequestResult {
            success: false,
            ttft_ms: 0.0,
            e2e_ms: 0.0,
            output_tokens: 0,
        };
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
                let e2e = start.elapsed().as_secs_f64() * 1000.0;
                let ttft = first_token_time
                    .map(|t| t.duration_since(start).as_secs_f64() * 1000.0)
                    .unwrap_or(e2e);
                return RequestResult {
                    success: true,
                    ttft_ms: ttft,
                    e2e_ms: e2e,
                    output_tokens,
                };
            }
            if let Ok(c) = serde_json::from_str::<OpenAiStreamChunk>(payload) {
                if let Some(choices) = c.choices {
                    if let Some(first) = choices.into_iter().next() {
                        if let Some(delta) = first.delta {
                            if delta.content.is_some() {
                                if first_token_time.is_none() {
                                    first_token_time = Some(Instant::now());
                                }
                                output_tokens += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    // Stream ended without [DONE] sentinel — still report what we got.
    let e2e = start.elapsed().as_secs_f64() * 1000.0;
    let ttft = first_token_time
        .map(|t| t.duration_since(start).as_secs_f64() * 1000.0)
        .unwrap_or(e2e);
    RequestResult {
        success: output_tokens > 0,
        ttft_ms: ttft,
        e2e_ms: e2e,
        output_tokens,
    }
}

fn pct(xs: &mut Vec<f64>, q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((xs.len() - 1) as f64 * q).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

pub async fn execute(cmd: BenchServeCommand, _cfg: CliConfig) -> Result<()> {
    println!(
        "{}",
        format!(
            "ferrum bench-serve — concurrency={} num_prompts={} input_len={} output_len={}",
            cmd.max_concurrency, cmd.num_prompts, cmd.random_input_len, cmd.random_output_len
        )
        .dimmed()
    );

    let tokenizer_path = cmd.tokenizer.join("tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        ferrum_types::FerrumError::model(format!(
            "Load tokenizer at {}: {}",
            tokenizer_path.display(),
            e
        ))
    })?;

    // Generate prompts upfront so wall-clock is dominated by HTTP only.
    let mut rng = rand::rng();
    let prompts: Vec<String> = (0..cmd.num_prompts)
        .map(|_| gen_random_prompt(&tok, cmd.random_input_len, &mut rng))
        .collect();

    let sem = Arc::new(Semaphore::new(cmd.max_concurrency));
    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(cmd.max_concurrency * 2)
            .build()
            .map_err(|e| ferrum_types::FerrumError::model(format!("reqwest client: {e}")))?,
    );

    let base_url = Arc::new(cmd.base_url.clone());
    let model = Arc::new(cmd.model.clone());
    let bench_start = Instant::now();

    let mut handles = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let permit = sem.clone().acquire_owned().await.expect("semaphore closed");
        let client = client.clone();
        let base_url = base_url.clone();
        let model = model.clone();
        let max_out = cmd.random_output_len;
        let tmo = cmd.timeout;
        handles.push(tokio::spawn(async move {
            let _p = permit;
            stream_one(&client, &base_url, &model, prompt, max_out, tmo).await
        }));
    }

    let mut results: Vec<RequestResult> = Vec::with_capacity(handles.len());
    for h in handles {
        if let Ok(r) = h.await {
            results.push(r);
        }
    }
    let bench_wall_s = bench_start.elapsed().as_secs_f64();

    let successful: Vec<&RequestResult> = results.iter().filter(|r| r.success).collect();
    let total_output_tokens: usize = successful.iter().map(|r| r.output_tokens).sum();
    let output_throughput = if bench_wall_s > 0.0 {
        total_output_tokens as f64 / bench_wall_s
    } else {
        0.0
    };
    let mut ttfts: Vec<f64> = successful.iter().map(|r| r.ttft_ms).collect();
    let mut tpots: Vec<f64> = successful
        .iter()
        .filter(|r| r.output_tokens >= 2)
        .map(|r| (r.e2e_ms - r.ttft_ms) / (r.output_tokens - 1) as f64)
        .collect();
    let mean_ttft = if ttfts.is_empty() {
        0.0
    } else {
        ttfts.iter().sum::<f64>() / ttfts.len() as f64
    };
    let mean_tpot = if tpots.is_empty() {
        0.0
    } else {
        tpots.iter().sum::<f64>() / tpots.len() as f64
    };
    let p99_ttft = pct(&mut ttfts, 0.99);
    let p99_tpot = pct(&mut tpots, 0.99);

    let summary = serde_json::json!({
        "model": &cmd.model,
        "num_prompts": cmd.num_prompts,
        "max_concurrency": cmd.max_concurrency,
        "random_input_len": cmd.random_input_len,
        "random_output_len": cmd.random_output_len,
        "completed": successful.len(),
        "failed": results.len() - successful.len(),
        "duration_s": (bench_wall_s * 1000.0).round() / 1000.0,
        "total_output_tokens": total_output_tokens,
        "output_throughput": (output_throughput * 100.0).round() / 100.0,
        "mean_ttft_ms": (mean_ttft * 100.0).round() / 100.0,
        "p99_ttft_ms": (p99_ttft * 100.0).round() / 100.0,
        "mean_tpot_ms": (mean_tpot * 100.0).round() / 100.0,
        "p99_tpot_ms": (p99_tpot * 100.0).round() / 100.0,
    });
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());

    if let Some(p) = cmd.result_file.as_ref() {
        std::fs::write(p, serde_json::to_string_pretty(&summary).unwrap())
            .map_err(|e| ferrum_types::FerrumError::model(format!("write result: {e}")))?;
    }
    Ok(())
}
