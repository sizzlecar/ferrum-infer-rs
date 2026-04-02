//! Bench command - Throughput and latency benchmarking
//!
//! Modes:
//!   ferrum bench qwen3:4b                              # default: sequential, 5 rounds
//!   ferrum bench qwen3:4b --concurrency 4              # concurrent requests (tests batch decode)
//!   ferrum bench qwen3:4b --max-tokens 1024            # long decode (tests flash decode)
//!   ferrum bench qwen3:4b --long-context               # 2k prompt + 256 decode
//!   ferrum bench qwen3:4b --concurrency 8 --max-tokens 64  # throughput stress test

use crate::config::CliConfig;
use chrono::Utc;
use clap::Args;
use colored::*;
use ferrum_models::HfDownloader;
use ferrum_types::{InferenceRequest, Priority, RequestId, Result, SamplingParams};
use futures::StreamExt;
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

#[derive(Args)]
pub struct BenchCommand {
    /// Model name (e.g., qwen3:0.6b, qwen3:4b)
    #[arg(default_value = "qwen3:0.6b")]
    pub model: String,

    /// Number of benchmark rounds
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

    /// Number of concurrent requests (>1 tests batch decode)
    #[arg(long, default_value = "1")]
    pub concurrency: usize,

    /// Long-context mode: use a ~2k token prompt (tests flash decode / paged KV)
    #[arg(long)]
    pub long_context: bool,
}

pub async fn execute(cmd: BenchCommand, config: CliConfig) -> Result<()> {
    let model_id = super::run::resolve_model_alias(&cmd.model);
    eprintln!("{}", format!("Ferrum Benchmark - {}", model_id).bold());
    eprintln!("{}", "=".repeat(60).dimmed());

    // Find or download model
    let cache_dir = super::run::get_hf_cache_dir(&config);
    let source = match super::run::find_cached_model(&cache_dir, &model_id) {
        Some(source) => source,
        None => {
            eprintln!("Downloading model...");
            let token = std::env::var("HF_TOKEN")
                .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
                .ok();
            let downloader = HfDownloader::new(cache_dir.clone(), token)?;
            let snapshot_path = downloader.download(&model_id, None).await?;
            let format = super::run::detect_format(&snapshot_path);
            ferrum_models::source::ResolvedModelSource {
                original: model_id.clone(),
                local_path: snapshot_path,
                format,
                from_cache: false,
            }
        }
    };

    unsafe {
        std::env::set_var(
            "FERRUM_MODEL_PATH",
            source.local_path.to_string_lossy().to_string(),
        );
    }

    let device = super::run::select_device(&cmd.backend);
    eprintln!("{} {:?}", "Device:".dimmed(), device);

    // Show GPU info
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = candle_core::Device::new_cuda(0) {
            if let Ok(cd) = d.as_cuda_device() {
                let name = cd.cuda_stream().context().device_name().unwrap_or_default();
                eprintln!("GPU 0: {name}");
            }
        }
        if let Ok(d) = candle_core::Device::new_cuda(1) {
            if let Ok(cd) = d.as_cuda_device() {
                let name = cd.cuda_stream().context().device_name().unwrap_or_default();
                eprintln!("GPU 1: {name}");
            }
        }
        if let Ok(tp) = std::env::var("FERRUM_TP") {
            eprintln!("Tensor Parallel: TP={tp}");
        }
    }

    // Create engine with ContinuousBatch scheduler (not Priority).
    // DefaultInferenceEngine (Priority) has stream lifecycle issues with bench.
    let mut engine_config = ferrum_engine::simple_engine_config(model_id.clone(), device);
    engine_config.scheduler.policy = ferrum_types::SchedulingPolicy::ContinuousBatch;
    let engine = ferrum_engine::create_mvp_engine(engine_config).await?;

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
            "Config: {} rounds, {} max_tokens, mode={}, prompt_len=~{}chars",
            cmd.rounds,
            cmd.max_tokens,
            mode_str,
            prompt.len()
        )
        .dimmed()
    );
    eprintln!("{}", "=".repeat(60).dimmed());

    // Warmup
    eprintln!("{}", "Warmup...".dimmed());
    let _ = run_single(&*engine, &model_id, "Hello", 16).await;
    // Let engine finish cleanup before starting benchmark rounds
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    if cmd.concurrency > 1 {
        run_concurrent_bench(&*engine, &model_id, &prompt, &cmd).await
    } else {
        run_sequential_bench(&*engine, &model_id, &prompt, &cmd).await
    }
}

// ── Sequential benchmark (existing behavior) ────────────────────────

async fn run_sequential_bench(
    engine: &(dyn ferrum_interfaces::InferenceEngine + Send + Sync),
    model_id: &str,
    prompt: &str,
    cmd: &BenchCommand,
) -> Result<()> {
    let mut total_tokens: usize = 0;
    let mut total_time_ms: f64 = 0.0;
    let mut ttft_ms_list: Vec<f64> = Vec::new();
    let mut tps_list: Vec<f64> = Vec::new();
    let mut tpot_ms_list: Vec<f64> = Vec::new();
    let mut decode_tps_list: Vec<f64> = Vec::new();

    for round in 1..=cmd.rounds {
        eprintln!("{}", format!("Round {}/{}...", round, cmd.rounds).dimmed());

        let result = run_single(engine, model_id, prompt, cmd.max_tokens).await?;
        let (tps, decode_tokens, tpot_ms, decode_tps) = compute_metrics(&result);

        total_tokens += result.token_count;
        total_time_ms += result.total_ms;
        ttft_ms_list.push(result.ttft_ms);
        tps_list.push(tps);
        if decode_tokens > 0 {
            tpot_ms_list.push(tpot_ms);
            decode_tps_list.push(decode_tps);
        }

        eprintln!(
            "  {} tokens in {:.1}ms ({:.1} tok/s, TTFT {:.1}ms, TPOT {:.2}ms, decode {:.1} tok/s)",
            result.token_count, result.total_ms, tps, result.ttft_ms, tpot_ms, decode_tps
        );

        // Let engine finish cleanup between rounds
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    print_summary(
        model_id,
        cmd,
        "sequential",
        total_tokens,
        total_time_ms,
        &tps_list,
        &ttft_ms_list,
        &tpot_ms_list,
        &decode_tps_list,
    );
    Ok(())
}

// ── Concurrent benchmark (tests batch decode) ───────────────────────

async fn run_concurrent_bench(
    engine: &(dyn ferrum_interfaces::InferenceEngine + Send + Sync),
    model_id: &str,
    prompt: &str,
    cmd: &BenchCommand,
) -> Result<()> {
    let mut total_tokens: usize = 0;
    let mut total_time_ms: f64 = 0.0;
    let mut all_ttft: Vec<f64> = Vec::new();
    let mut all_tpot: Vec<f64> = Vec::new();
    let mut round_tps_list: Vec<f64> = Vec::new();

    for round in 1..=cmd.rounds {
        eprintln!(
            "{}",
            format!(
                "Round {}/{} ({} concurrent)...",
                round, cmd.rounds, cmd.concurrency
            )
            .dimmed()
        );

        let round_start = Instant::now();

        // Launch N concurrent requests
        let mut handles = Vec::with_capacity(cmd.concurrency);
        for _ in 0..cmd.concurrency {
            let request = make_request(model_id, prompt, cmd.max_tokens);
            let stream = engine.infer_stream(request).await?;
            handles.push(tokio::spawn(collect_stream(stream)));
        }

        // Collect all results
        let mut round_tokens = 0usize;
        let mut round_results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => {
                    round_tokens += result.token_count;
                    round_results.push(result);
                }
                Ok(Err(e)) => eprintln!("  request error: {e}"),
                Err(e) => eprintln!("  join error: {e}"),
            }
        }

        // Let the engine finish cleaning up completed requests
        // (complete_request runs asynchronously after stream ends)
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let round_ms = round_start.elapsed().as_secs_f64() * 1000.0;
        let round_tps = if round_ms > 0.0 {
            round_tokens as f64 / (round_ms / 1000.0)
        } else {
            0.0
        };

        total_tokens += round_tokens;
        total_time_ms += round_ms;
        round_tps_list.push(round_tps);

        // Per-request metrics
        for r in &round_results {
            all_ttft.push(r.ttft_ms);
            let decode_tokens = r.token_count.saturating_sub(1);
            if decode_tokens > 0 {
                let tpot = (r.total_ms - r.ttft_ms) / decode_tokens as f64;
                all_tpot.push(tpot);
            }
        }

        let avg_ttft = round_results.iter().map(|r| r.ttft_ms).sum::<f64>()
            / round_results.len().max(1) as f64;

        eprintln!(
            "  {} requests, {} tokens in {:.1}ms ({:.1} tok/s total, avg TTFT {:.1}ms)",
            round_results.len(),
            round_tokens,
            round_ms,
            round_tps,
            avg_ttft
        );
    }

    // Summary
    let avg_tps = if !round_tps_list.is_empty() {
        round_tps_list.iter().sum::<f64>() / round_tps_list.len() as f64
    } else {
        0.0
    };
    let avg_ttft = if !all_ttft.is_empty() {
        all_ttft.iter().sum::<f64>() / all_ttft.len() as f64
    } else {
        0.0
    };
    let avg_tpot = if !all_tpot.is_empty() {
        all_tpot.iter().sum::<f64>() / all_tpot.len() as f64
    } else {
        0.0
    };

    eprintln!();
    eprintln!("{}", "=".repeat(60));
    eprintln!("{}", "BENCHMARK RESULTS (concurrent)".bold());
    eprintln!("{}", "=".repeat(60));
    eprintln!("Model:             {}", model_id);
    eprintln!(
        "Backend:           {:?}",
        super::run::select_device(&cmd.backend)
    );
    eprintln!("Rounds:            {}", cmd.rounds);
    eprintln!("Concurrency:       {}", cmd.concurrency);
    eprintln!("Max tokens/req:    {}", cmd.max_tokens);
    eprintln!("{}", "-".repeat(60));
    eprintln!(
        "Throughput (total): {:.1} tok/s avg ({:.1} min, {:.1} max)",
        avg_tps,
        round_tps_list.iter().cloned().fold(f64::INFINITY, f64::min),
        round_tps_list.iter().cloned().fold(0.0_f64, f64::max)
    );
    eprintln!(
        "TTFT:              {:.1}ms avg, {:.1}ms p99",
        avg_ttft,
        percentile(&all_ttft, 99.0)
    );
    eprintln!(
        "TPOT:              {:.2}ms avg, {:.2}ms p99",
        avg_tpot,
        percentile(&all_tpot, 99.0)
    );
    eprintln!("Total tokens:      {}", total_tokens);
    eprintln!("Total time:        {:.1}ms", total_time_ms);
    eprintln!("{}", "=".repeat(60));

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────

struct BenchResult {
    token_count: usize,
    ttft_ms: f64,
    total_ms: f64,
}

fn compute_metrics(r: &BenchResult) -> (f64, usize, f64, f64) {
    let tps = if r.total_ms > 0.0 {
        r.token_count as f64 / (r.total_ms / 1000.0)
    } else {
        0.0
    };
    let decode_tokens = r.token_count.saturating_sub(1);
    let decode_time_ms = r.total_ms - r.ttft_ms;
    let tpot_ms = if decode_tokens > 0 {
        decode_time_ms / decode_tokens as f64
    } else {
        0.0
    };
    let decode_tps = if decode_time_ms > 0.0 {
        decode_tokens as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    (tps, decode_tokens, tpot_ms, decode_tps)
}

fn make_request(model_id: &str, prompt: &str, max_tokens: u32) -> InferenceRequest {
    InferenceRequest {
        id: RequestId(Uuid::new_v4()),
        model_id: ferrum_types::ModelId(model_id.to_string()),
        prompt: prompt.to_string(),
        sampling_params: SamplingParams {
            max_tokens: max_tokens as usize,
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.1,
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
        metadata: HashMap::new(),
    }
}

async fn run_single(
    engine: &(dyn ferrum_interfaces::InferenceEngine + Send + Sync),
    model_id: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<BenchResult> {
    let request = make_request(model_id, prompt, max_tokens);
    let stream = engine.infer_stream(request).await?;
    collect_stream(stream).await
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
) -> Result<BenchResult> {
    let start = Instant::now();
    let mut token_count = 0usize;
    let mut first_token_time: Option<f64> = None;

    let mut got_finish = false;
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                if chunk.token.is_some() {
                    token_count += 1;
                    if first_token_time.is_none() {
                        first_token_time = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                if chunk.finish_reason.is_some() {
                    got_finish = true;
                    break;
                }
            }
            Err(_) => break,
        }
    }
    if !got_finish && token_count > 0 {
        eprintln!(
            "  [warn] stream ended without finish_reason ({} tokens)",
            token_count
        );
    }

    Ok(BenchResult {
        token_count,
        ttft_ms: first_token_time.unwrap_or(0.0),
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

fn print_summary(
    model_id: &str,
    cmd: &BenchCommand,
    mode: &str,
    total_tokens: usize,
    total_time_ms: f64,
    tps_list: &[f64],
    ttft_ms_list: &[f64],
    tpot_ms_list: &[f64],
    decode_tps_list: &[f64],
) {
    let avg_tps = if !tps_list.is_empty() {
        tps_list.iter().sum::<f64>() / tps_list.len() as f64
    } else {
        0.0
    };
    let min_tps = tps_list.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_tps = tps_list.iter().cloned().fold(0.0_f64, f64::max);
    let avg_ttft = if !ttft_ms_list.is_empty() {
        ttft_ms_list.iter().sum::<f64>() / ttft_ms_list.len() as f64
    } else {
        0.0
    };
    let avg_decode_tps = if !decode_tps_list.is_empty() {
        decode_tps_list.iter().sum::<f64>() / decode_tps_list.len() as f64
    } else {
        0.0
    };
    let avg_tpot = if !tpot_ms_list.is_empty() {
        tpot_ms_list.iter().sum::<f64>() / tpot_ms_list.len() as f64
    } else {
        0.0
    };

    eprintln!();
    eprintln!("{}", "=".repeat(60));
    eprintln!("{}", format!("BENCHMARK RESULTS ({})", mode).bold());
    eprintln!("{}", "=".repeat(60));
    eprintln!("Model:             {}", model_id);
    eprintln!(
        "Backend:           {:?}",
        super::run::select_device(&cmd.backend)
    );
    eprintln!("Rounds:            {}", cmd.rounds);
    eprintln!("Max tokens/round:  {}", cmd.max_tokens);
    eprintln!("{}", "-".repeat(60));
    eprintln!(
        "Throughput (e2e):  {:.1} tok/s avg ({:.1} min, {:.1} max)",
        avg_tps, min_tps, max_tps
    );
    eprintln!("Decode only:       {:.1} tok/s avg", avg_decode_tps);
    eprintln!(
        "TTFT:              {:.1}ms avg, {:.1}ms p99",
        avg_ttft,
        percentile(ttft_ms_list, 99.0)
    );
    eprintln!(
        "TPOT:              {:.2}ms avg, {:.2}ms p99",
        avg_tpot,
        percentile(tpot_ms_list, 99.0)
    );
    eprintln!("Total tokens:      {}", total_tokens);
    eprintln!("Total time:        {:.1}ms", total_time_ms);
    eprintln!("{}", "=".repeat(60));
}

/// Generate a ~2k token prompt for long-context benchmarking.
fn generate_long_prompt() -> String {
    let base = "The history of artificial intelligence is a fascinating journey through decades of research, breakthroughs, and setbacks. From the early days of symbolic AI in the 1950s, through the AI winters, to the modern era of deep learning and large language models, the field has undergone remarkable transformations. ";
    // Repeat to get ~2k tokens worth of text (~8k chars)
    let mut prompt = String::with_capacity(8192);
    prompt.push_str("Please provide a comprehensive analysis of the following text, identifying key themes, patterns, and insights:\n\n");
    for _ in 0..25 {
        prompt.push_str(base);
    }
    prompt.push_str("\n\nNow analyze the above text in detail:");
    prompt
}

fn percentile(data: &[f64], pct: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
