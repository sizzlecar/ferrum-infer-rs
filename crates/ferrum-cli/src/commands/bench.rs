//! Bench command - Throughput and latency benchmarking

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
    /// Model name (e.g., qwen3:0.6b)
    #[arg(default_value = "qwen3:0.6b")]
    pub model: String,

    /// Number of benchmark rounds
    #[arg(long, default_value = "5")]
    pub rounds: usize,

    /// Max tokens per round
    #[arg(long, default_value = "128")]
    pub max_tokens: u32,

    /// Backend: auto, cpu, cuda, metal
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// Prompt to use
    #[arg(long, default_value = "Explain the theory of relativity in detail.")]
    pub prompt: String,
}

pub async fn execute(cmd: BenchCommand, config: CliConfig) -> Result<()> {
    let model_id = super::run::resolve_model_alias(&cmd.model);
    eprintln!(
        "{}",
        format!("Ferrum Benchmark - {}", model_id).bold()
    );
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

    // Create engine
    let engine_config = ferrum_engine::simple_engine_config(model_id.clone(), device);
    let engine = ferrum_engine::create_mvp_engine(engine_config).await?;

    eprintln!(
        "{}",
        format!(
            "Config: {} rounds, {} max_tokens",
            cmd.rounds, cmd.max_tokens
        )
        .dimmed()
    );
    eprintln!("{}", "=".repeat(60).dimmed());

    // Warmup
    eprintln!("{}", "Warmup...".dimmed());
    let _ = run_single(&*engine, &model_id, "Hello", 16).await;

    // Sequential throughput benchmark
    let mut total_tokens: usize = 0;
    let mut total_time_ms: f64 = 0.0;
    let mut ttft_ms_list: Vec<f64> = Vec::new();
    let mut tps_list: Vec<f64> = Vec::new();

    for round in 1..=cmd.rounds {
        eprintln!(
            "{}",
            format!("Round {}/{}...", round, cmd.rounds).dimmed()
        );

        let start = Instant::now();
        let result = run_single(&*engine, &model_id, &cmd.prompt, cmd.max_tokens).await?;
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        let tps = if elapsed.as_secs_f64() > 0.0 {
            result.token_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        total_tokens += result.token_count;
        total_time_ms += elapsed_ms;
        ttft_ms_list.push(result.ttft_ms);
        tps_list.push(tps);

        eprintln!(
            "  {} tokens in {:.1}ms ({:.1} tok/s, TTFT {:.1}ms)",
            result.token_count, elapsed_ms, tps, result.ttft_ms
        );
    }

    // Summary
    eprintln!();
    eprintln!("{}", "=".repeat(60));
    eprintln!("{}", "BENCHMARK RESULTS".bold());
    eprintln!("{}", "=".repeat(60));
    eprintln!("Model:             {}", model_id);
    eprintln!("Backend:           {:?}", super::run::select_device(&cmd.backend));
    eprintln!("Rounds:            {}", cmd.rounds);
    eprintln!("Max tokens/round:  {}", cmd.max_tokens);
    eprintln!("{}", "-".repeat(60));

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
    let p99_ttft = percentile(&ttft_ms_list, 99.0);

    eprintln!(
        "Throughput:        {:.1} tok/s avg ({:.1} min, {:.1} max)",
        avg_tps, min_tps, max_tps
    );
    eprintln!(
        "TTFT:              {:.1}ms avg, {:.1}ms p99",
        avg_ttft, p99_ttft
    );
    eprintln!("Total tokens:      {}", total_tokens);
    eprintln!("Total time:        {:.1}ms", total_time_ms);
    eprintln!("{}", "=".repeat(60));

    Ok(())
}

struct BenchResult {
    token_count: usize,
    ttft_ms: f64,
}

async fn run_single(
    engine: &(dyn ferrum_interfaces::InferenceEngine + Send + Sync),
    model_id: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<BenchResult> {
    let request = InferenceRequest {
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
    };

    let start = Instant::now();
    let mut stream = engine.infer_stream(request).await?;
    let mut token_count = 0usize;
    let mut first_token_time: Option<f64> = None;

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
                    break;
                }
            }
            Err(_) => break,
        }
    }

    Ok(BenchResult {
        token_count,
        ttft_ms: first_token_time.unwrap_or(0.0),
    })
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
