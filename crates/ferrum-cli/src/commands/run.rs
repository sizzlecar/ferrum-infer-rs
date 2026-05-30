//! Run command - Interactive chat with a model (ollama-style)

use crate::config::CliConfig;
use chrono::Utc;
use clap::{Args, ValueEnum};
use colored::*;
use ferrum_models::source::{ModelFormat, ResolvedModelSource};
use ferrum_types::{
    FinishReason, InferenceRequest, Priority, RequestId, Result, RuntimeConfigSnapshot,
    SamplingParams,
};
use futures::StreamExt;
use std::collections::HashMap;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Output format for `ferrum run`. JSONL mode emits one record per event
/// (assistant generation result, user input, exit) on stdout — used by
/// integration tests and scripting. Text mode is the default interactive UX.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum OutputFormat {
    /// Streaming text on stdout, stats on stderr (default — interactive UX).
    #[default]
    Text,
    /// One JSON record per event on stdout (machine-readable; tests).
    Jsonl,
}

fn finish_reason_str(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::EOS => "eos",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Error => "error",
        FinishReason::ContentFilter => "content_filter",
    }
}

fn emit_jsonl_ready(model: &str, backend: &str) {
    let record = serde_json::json!({
        "event": "ready",
        "model": model,
        "backend": backend,
    });
    println!("{record}");
}

fn emit_jsonl_user(turn: usize, content: &str) {
    let record = serde_json::json!({
        "event": "user",
        "turn": turn,
        "content": content,
    });
    println!("{record}");
}

fn emit_jsonl_assistant(
    turn: usize,
    content: &str,
    finish_reason: Option<FinishReason>,
    n_tokens: usize,
    chunk_count: usize,
    ms: f64,
) {
    let record = serde_json::json!({
        "event": "assistant",
        "turn": turn,
        "content": content,
        "finish_reason": finish_reason.map(finish_reason_str),
        "n_tokens": n_tokens,
        "chunk_count": chunk_count,
        "ms": ms,
    });
    println!("{record}");
}

fn emit_jsonl_exit(reason: &str) {
    let record = serde_json::json!({
        "event": "exit",
        "reason": reason,
    });
    println!("{record}");
}

#[derive(Args)]
pub struct RunCommand {
    /// Model name (alias like `qwen3:8b`, HF repo id, or path to a `.gguf` file).
    /// When the argument is a `.gguf` path, ferrum routes to candle-transformers'
    /// quantized loaders for the M1 Max bench path (Qwen3 / Qwen3-MoE / Llama).
    #[arg(default_value = "tinyllama")]
    pub model: String,

    /// System prompt (interactive chat mode only).
    #[arg(long)]
    pub system: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "2048")]
    pub max_tokens: u32,

    /// Sampling temperature (0.0–2.0). 0.0 = greedy / argmax (deterministic,
    /// what you want for benchmarks). >0 = softmax sample with `--top-k`
    /// and `--top-p` filtering applied.
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Backend: auto, cpu, metal (default: auto)
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// One-shot prompt (skip interactive REPL). When supplied, ferrum runs a
    /// single prefill+decode and exits — useful for benchmarking and shell
    /// scripting. For `.gguf` paths, omitting this drops into the GGUF REPL.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Path to a HuggingFace `tokenizer.json` (only used for `.gguf` paths).
    /// If omitted, ferrum looks for `<gguf-stem>.tokenizer.json` and then
    /// `tokenizer.json` next to the `.gguf` file.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Bench mode: skip generated text output, print only timing summary.
    /// Implies one-shot (`--prompt` is required).
    #[arg(long)]
    pub bench_mode: bool,

    /// Top-K sampling cutoff (0 disables — keep all). Only the K highest-
    /// probability tokens compete in the softmax sample. Default 50, a
    /// conservative value that filters obvious garbage without flattening
    /// the distribution.
    #[arg(long, default_value = "50")]
    pub top_k: usize,

    /// Top-P (nucleus) sampling cutoff (0.0 disables, 1.0 keeps all).
    /// Smallest set of tokens whose cumulative probability exceeds P
    /// is kept; the rest are zeroed before sampling. Default 0.95.
    #[arg(long, default_value = "0.95")]
    pub top_p: f32,

    /// Repetition penalty applied to logits before sampling. >1 discourages
    /// repeats, <1 encourages, 1.0 disables. OpenAI uses 1.1 typically.
    #[arg(long, default_value = "1.0")]
    pub repeat_penalty: f32,

    /// Number of recent tokens that the repetition penalty considers.
    /// Smaller = local repeat avoidance only.
    #[arg(long, default_value = "64")]
    pub repeat_last_n: usize,

    /// Random seed for sampling (when temperature > 0). Default 42.
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Fraction of GPU memory ferrum is allowed to use (mirrors vLLM's
    /// `--gpu-memory-utilization`). Auto-sizes the KV pool: at 0.9
    /// ferrum will use ≤ 90 % of the GPU's reported total memory,
    /// reserving ~4 GB scratch + the weight bytes. Set to 1.0 for an
    /// exclusive GPU; leave at 0.9 if other processes share the card.
    #[arg(long, default_value = "0.9")]
    pub gpu_memory_utilization: f32,

    /// KV cache element dtype (Dim 5 polymorphism point). Accepts
    /// `fp16`, `bf16`, `int8`, `fp8`. Default `fp16`. INT8 / FP8
    /// require model wire-up; today only the kernel + type layer ships.
    /// Override via `FERRUM_KV_DTYPE` env var.
    #[arg(long, value_name = "DTYPE")]
    pub kv_dtype: Option<String>,

    /// Output format. `text` (default) — streaming text + stats UX.
    /// `jsonl` — one JSON record per event on stdout; used by tests and scripts.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub output_format: OutputFormat,
}

pub async fn execute(cmd: RunCommand, config: CliConfig) -> Result<()> {
    // Resolve graph-clean Qwen3-MoE defaults as typed entries first, then
    // materialize them only for legacy backend readers.
    let moe_graph_defaults = crate::runtime_env::moe_graph_default_entries(
        &ferrum_types::RuntimeConfigSnapshot::capture_current(),
        ferrum_types::RuntimeConfigSource::Default,
    );
    crate::runtime_env::materialize_runtime_env_defaults(&moe_graph_defaults);
    crate::runtime_env::warn_if_moe_graph_needs_unbuilt_vllm_moe(
        &ferrum_types::RuntimeConfigSnapshot::capture_current(),
    );

    // Resolve the model through the central source resolver. Handles
    // .gguf paths, local model dirs, HF cache hits, and HF download in
    // one entry; runs the chat-profile GPU autosize + (for GGUF) sets
    // the per-arch KV / MoE env-var defaults that `ferrum run` needs
    // for a single-user multi-turn REPL. The engine then picks up
    // either the safetensors path (via NativeSafetensorsLoader) or the
    // GGUF path (via gguf_engine_loader, routed by
    // `WeightFormat::detect()` inside `LlmExecutorFactory`).
    let cache_dir = get_hf_cache_dir(&config);
    let resolved = crate::source_resolver::resolve_model_source(
        &cmd.model,
        &cache_dir,
        crate::source_resolver::DownloadPolicy::AutoDownload,
        Some((
            crate::gpu_mem_autosize::AutoSizeProfile::Chat,
            cmd.gpu_memory_utilization,
        )),
    )
    .await?;
    let source = resolved.source;
    let model_id = source.original.clone();
    eprintln!("{}", format!("Loading {}...", model_id).dimmed());

    let engine_model_path = source.local_path.to_string_lossy().to_string();

    // Select device
    let device = select_device(&cmd.backend);
    let device_label = format!("{device:?}");
    eprintln!("{}", format!("Using {device_label} backend").dimmed());

    // Create engine. Big-model loads (15-60 GB safetensors) are slow on
    // first run — print a hint so users don't think it's frozen. Per-
    // layer INFO logs fire from the model loaders once parsing starts;
    // utils::setup_logging whitelists them at INFO level.
    eprintln!(
        "{}",
        "Loading weights to GPU... (30s+ for >10 GB models)".dimmed()
    );
    let load_start = std::time::Instant::now();
    let mut engine_config = ferrum_types::EngineConfig::default();
    engine_config.model.model_id = ferrum_types::ModelId::new(model_id.clone());
    engine_config.backend.device = device;
    engine_config.backend.backend_options.insert(
        "model_path".to_string(),
        serde_json::Value::String(engine_model_path),
    );
    let runtime_config = RuntimeConfigSnapshot::capture_current();
    engine_config
        .apply_runtime_config_snapshot(&runtime_config)
        .map_err(ferrum_types::FerrumError::config)?;
    let effective_kv_dtype = cmd
        .kv_dtype
        .as_deref()
        .or_else(|| crate::runtime_env::runtime_snapshot_value(&runtime_config, "FERRUM_KV_DTYPE"));
    apply_kv_dtype_override(&mut engine_config, effective_kv_dtype)?;
    let engine = ferrum_engine::create_default_engine(engine_config).await?;
    eprintln!(
        "{}",
        format!(
            "Model loaded in {:.1}s.",
            load_start.elapsed().as_secs_f64()
        )
        .dimmed()
    );

    // One-shot mode: --prompt supplied → run a single request and exit.
    // Matches the GGUF run_gguf_one_shot UX. Previously cmd.prompt was
    // documented as "one-shot for non-interactive runs" but the alias
    // path ignored it and dropped into REPL, which exits silently when
    // stdin is not a TTY.
    if let Some(one_shot) = cmd.prompt.clone() {
        let chat_prompt = build_chat_prompt(&[], &one_shot, cmd.system.as_deref(), &model_id);
        let model_lower = model_id.to_lowercase();
        let (default_top_p, default_top_k) = if model_lower.contains("qwen3") {
            (0.8, Some(20))
        } else {
            (0.9, None)
        };
        let request = InferenceRequest {
            id: RequestId(Uuid::new_v4()),
            model_id: ferrum_types::ModelId(model_id.clone()),
            prompt: chat_prompt,
            sampling_params: SamplingParams {
                max_tokens: cmd.max_tokens as usize,
                temperature: cmd.temperature,
                top_p: default_top_p,
                top_k: default_top_k,
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
            api_request: None,
            metadata: HashMap::new(),
        };
        let mut stream = engine.infer_stream(request).await?;
        let start = std::time::Instant::now();
        let mut tokens = 0usize;
        let mut content = String::new();
        let mut chunk_count = 0usize;
        let mut finish_reason: Option<FinishReason> = None;
        let format = cmd.output_format;
        let bench = cmd.bench_mode;
        while let Some(chunk) = stream.next().await {
            if let Ok(c) = chunk {
                if !c.text.is_empty() {
                    chunk_count += 1;
                    content.push_str(&c.text);
                    if format == OutputFormat::Text && !bench {
                        print!("{}", c.text);
                        io::stdout().flush().ok();
                    }
                }
                if c.token.is_some() {
                    tokens += 1;
                }
                if let Some(r) = c.finish_reason {
                    finish_reason = Some(r);
                    break;
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 {
            tokens as f64 / elapsed
        } else {
            0.0
        };
        match format {
            OutputFormat::Text => {
                if !bench {
                    println!();
                }
                eprintln!(
                    "{}",
                    format!("[{tokens} tokens, {tps:.1} tok/s, {elapsed:.1}s]").dimmed()
                );
            }
            OutputFormat::Jsonl => {
                emit_jsonl_assistant(
                    0,
                    &content,
                    finish_reason,
                    tokens,
                    chunk_count,
                    elapsed * 1000.0,
                );
            }
        }
        return Ok(());
    }

    // Print ready message
    let format = cmd.output_format;
    match format {
        OutputFormat::Text => {
            eprintln!();
            eprintln!("{}", "Ready. Type your message and press Enter.".green());
            eprintln!("{}", "Use /bye or Ctrl+D to exit.".dimmed());
            eprintln!();
        }
        OutputFormat::Jsonl => {
            emit_jsonl_ready(&model_id, &device_label);
        }
    }

    // Interactive loop
    let mut history: Vec<(String, String)> = Vec::new(); // (role, content)
    let generating = Arc::new(AtomicBool::new(false));
    let mut turn = 0usize;
    let mut exit_reason: &str = "eof";

    // If stdin is not a TTY (piped input), don't print prompts and just consume lines.
    // This enables: `printf "hi\n/bye\n" | ferrum run ...` for automation/profiling.
    let stdin_is_tty = io::stdin().is_terminal();
    let mut stdin = io::stdin().lock();

    loop {
        if stdin_is_tty {
            // Show prompt
            print!("{} ", ">>>".bright_green().bold());
            io::stdout().flush().unwrap();
        }

        // Read input
        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }
                if input == "/bye" || input == "exit" || input == "quit" {
                    exit_reason = match input {
                        "/bye" => "bye",
                        "exit" => "exit",
                        "quit" => "quit",
                        _ => "command",
                    };
                    break;
                }

                if format == OutputFormat::Jsonl {
                    emit_jsonl_user(turn, input);
                }

                // Build prompt with history
                let prompt = build_chat_prompt(&history, input, cmd.system.as_deref(), &model_id);

                // Model-specific sampling defaults
                let model_lower = model_id.to_lowercase();
                let (default_top_p, default_top_k) = if model_lower.contains("qwen3") {
                    // Qwen3 non-thinking mode: top_p=0.8, top_k=20
                    (0.8, Some(20))
                } else {
                    (0.9, None)
                };

                // Create request
                let request = InferenceRequest {
                    id: RequestId(Uuid::new_v4()),
                    model_id: ferrum_types::ModelId(model_id.clone()),
                    prompt,
                    sampling_params: SamplingParams {
                        max_tokens: cmd.max_tokens as usize,
                        temperature: cmd.temperature,
                        top_p: default_top_p,
                        top_k: default_top_k,
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
                    api_request: None,
                    metadata: HashMap::new(),
                };

                // Start generation
                generating.store(true, Ordering::SeqCst);
                let mut stream = engine.infer_stream(request).await?;
                let mut response = String::new();
                let start = std::time::Instant::now();
                let mut token_count = 0usize;
                let mut chunk_count = 0usize;
                let mut finish_reason: Option<FinishReason> = None;

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            if !chunk.text.is_empty() {
                                chunk_count += 1;
                                response.push_str(&chunk.text);
                                if format == OutputFormat::Text {
                                    print!("{}", chunk.text);
                                    io::stdout().flush().unwrap();
                                }
                            }
                            if chunk.token.is_some() {
                                token_count += 1;
                            }
                            if let Some(r) = chunk.finish_reason {
                                finish_reason = Some(r);
                                break;
                            }
                        }
                        Err(e) => {
                            if format == OutputFormat::Text {
                                eprintln!("\n{} {}", "Error:".red(), e);
                            } else {
                                let record = serde_json::json!({
                                    "event": "error",
                                    "turn": turn,
                                    "message": e.to_string(),
                                });
                                println!("{record}");
                            }
                            break;
                        }
                    }
                }

                generating.store(false, Ordering::SeqCst);

                let elapsed = start.elapsed();
                let elapsed_s = elapsed.as_secs_f64();
                let tps = if elapsed_s > 0.0 {
                    token_count as f64 / elapsed_s
                } else {
                    0.0
                };
                let clean_response = response.trim().to_string();

                match format {
                    OutputFormat::Text => {
                        println!();
                        eprintln!(
                            "{}",
                            format!("[{token_count} tokens, {tps:.1} tok/s, {elapsed_s:.1}s]")
                                .dimmed()
                        );
                        eprintln!();
                    }
                    OutputFormat::Jsonl => {
                        emit_jsonl_assistant(
                            turn,
                            &clean_response,
                            finish_reason,
                            token_count,
                            chunk_count,
                            elapsed_s * 1000.0,
                        );
                    }
                }

                // In non-interactive mode, don't wait for terminal formatting/spacing.
                if !stdin_is_tty {
                    io::stdout().flush().ok();
                    io::stderr().flush().ok();
                }

                // Add to history
                history.push(("user".to_string(), input.to_string()));
                if !clean_response.is_empty() {
                    history.push(("assistant".to_string(), clean_response));
                }

                // Limit history
                while history.len() > 10 {
                    history.remove(0);
                }
                turn += 1;
            }
            Err(e) => {
                eprintln!("{} {}", "Error reading input:".red(), e);
                exit_reason = "read_error";
                break;
            }
        }
    }

    match format {
        OutputFormat::Text => {
            eprintln!("{}", "Goodbye!".bright_yellow());
        }
        OutputFormat::Jsonl => {
            emit_jsonl_exit(exit_reason);
        }
    }
    Ok(())
}

pub fn resolve_model_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "tinyllama" | "tiny" => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        "qwen2.5:0.5b" | "qwen:0.5b" => "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        "qwen2.5:1.5b" | "qwen:1.5b" => "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        "qwen2.5:3b" | "qwen:3b" => "Qwen/Qwen2.5-3B-Instruct".to_string(),
        "qwen2.5:7b" | "qwen:7b" => "Qwen/Qwen2.5-7B-Instruct".to_string(),
        "qwen3:0.6b" => "Qwen/Qwen3-0.6B".to_string(),
        "qwen3:1.7b" => "Qwen/Qwen3-1.7B".to_string(),
        "qwen3:4b" => "Qwen/Qwen3-4B".to_string(),
        "qwen2.5:3b-gptq" | "qwen2.5-3b-instruct-gptq-int4" => {
            "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4".to_string()
        }
        "llama3.2:1b" => "meta-llama/Llama-3.2-1B-Instruct".to_string(),
        "llama3.2:3b" => "meta-llama/Llama-3.2-3B-Instruct".to_string(),
        "whisper-tiny" | "whisper:tiny" => "openai/whisper-tiny".to_string(),
        "whisper-base" | "whisper:base" => "openai/whisper-base".to_string(),
        "whisper-small" | "whisper:small" => "openai/whisper-small".to_string(),
        "whisper-medium" | "whisper:medium" => "openai/whisper-medium".to_string(),
        "whisper-large-v3" | "whisper:large-v3" => "openai/whisper-large-v3".to_string(),
        "whisper-turbo" | "whisper:turbo" | "whisper-large-v3-turbo" => {
            "openai/whisper-large-v3-turbo".to_string()
        }
        _ => name.to_string(),
    }
}

pub fn get_hf_cache_dir(config: &CliConfig) -> PathBuf {
    // Check environment variable first
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }

    // Use config value
    let configured = shellexpand::tilde(&config.models.download.hf_cache_dir).to_string();
    PathBuf::from(configured)
}

pub fn find_cached_model(cache_dir: &PathBuf, model_id: &str) -> Option<ResolvedModelSource> {
    let repo_dir = cache_dir
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")));
    let snapshots_dir = repo_dir.join("snapshots");

    // Try refs/main first
    let ref_main = repo_dir.join("refs").join("main");
    if let Ok(rev) = std::fs::read_to_string(&ref_main) {
        let rev = rev.trim();
        if !rev.is_empty() {
            let snapshot = snapshots_dir.join(rev);
            if snapshot.exists() {
                let format = detect_format(&snapshot);
                if format != ModelFormat::Unknown {
                    return Some(ResolvedModelSource {
                        original: model_id.to_string(),
                        local_path: snapshot,
                        format,
                        from_cache: true,
                    });
                }
            }
        }
    }

    // Fallback: first snapshot directory
    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let format = detect_format(&path);
                if format != ModelFormat::Unknown {
                    return Some(ResolvedModelSource {
                        original: model_id.to_string(),
                        local_path: path,
                        format,
                        from_cache: true,
                    });
                }
            }
        }
    }

    None
}

pub fn detect_format(path: &PathBuf) -> ModelFormat {
    if path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists()
    {
        ModelFormat::SafeTensors
    } else if path.join("pytorch_model.bin").exists() {
        ModelFormat::PyTorchBin
    } else {
        ModelFormat::Unknown
    }
}

/// Treat the model argument as a `.gguf` file path if it ends in `.gguf`
/// (case-insensitive) and the file actually exists. Anything else falls
/// through to the alias / HF repo path.
pub fn looks_like_gguf_path(model: &str) -> bool {
    let p = PathBuf::from(model);
    p.extension()
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
        && p.is_file()
}

/// Resolve a GGUF alias to a `(repo, filename)` pair if recognised. Returns
/// `None` for non-GGUF aliases — callers fall through to
/// [`resolve_model_alias`].
///
/// These map ergonomic aliases to the `<org>/<name>-GGUF` repos the
/// community publishes Q4_K_M quantizations under. The filename component
/// pins a specific quantization; users wanting other quants pass the path
/// directly or extend this table.
pub fn resolve_gguf_alias(name: &str) -> Option<(String, String)> {
    // Aliases verified by probing the HF API on 2026-05-01. Quantization
    // availability differs per repo — Qwen/Qwen3-{0.6B,1.7B}-GGUF only
    // host Q8_0; 4B / 8B / 30B-A3B host Q4_K_M.
    match name.to_lowercase().as_str() {
        // Group A bench targets — same models the bench scripts use for
        // single-request PP/TG comparison vs llama.cpp / mistral.rs.
        "qwen3:8b-q4_k_m" => Some((
            "Qwen/Qwen3-8B-GGUF".to_string(),
            "Qwen3-8B-Q4_K_M.gguf".to_string(),
        )),
        "qwen3:4b-q4_k_m" => Some((
            "Qwen/Qwen3-4B-GGUF".to_string(),
            "Qwen3-4B-Q4_K_M.gguf".to_string(),
        )),
        "qwen3:1.7b" | "qwen3:1.7b-q8_0" => Some((
            "Qwen/Qwen3-1.7B-GGUF".to_string(),
            "Qwen3-1.7B-Q8_0.gguf".to_string(),
        )),
        "qwen3:0.6b-gguf" | "qwen3:0.6b-q8_0" => Some((
            "Qwen/Qwen3-0.6B-GGUF".to_string(),
            "Qwen3-0.6B-Q8_0.gguf".to_string(),
        )),
        "qwen3-moe:30b-a3b-q4_k_m" | "qwen3:30b-a3b-q4_k_m" => Some((
            "Qwen/Qwen3-30B-A3B-GGUF".to_string(),
            "Qwen3-30B-A3B-Q4_K_M.gguf".to_string(),
        )),
        "llama3.1:8b-q4_k_m" => Some((
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string(),
        )),
        "llama3.2:3b-q4_k_m" => Some((
            "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
        )),
        "llama3.2:1b-q4_k_m" => Some((
            "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string(),
        )),
        _ => None,
    }
}

/// For GGUF aliases whose repo lacks a tokenizer.json, return the sibling
/// safetensors repo where the tokenizer should be pulled from. Convention:
/// strip a trailing `-GGUF` from the repo name. Returns `None` for repos
/// that already host their own tokenizer (e.g. bartowski/*).
pub fn tokenizer_sibling_repo(gguf_repo: &str) -> Option<String> {
    if let Some(stripped) = gguf_repo.strip_suffix("-GGUF") {
        Some(stripped.to_string())
    } else {
        None
    }
}

/// Locate a previously-pulled GGUF file in the HF cache.
///
/// Mirrors `find_cached_model` but returns the path to the specific
/// `.gguf` file (not a directory). Looks up `refs/main` to find the
/// active snapshot, falls back to the first snapshot containing the
/// requested file. Returns `None` if neither finds it.
pub fn find_cached_gguf(cache_dir: &PathBuf, repo: &str, filename: &str) -> Option<PathBuf> {
    let repo_dir = cache_dir
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")));
    let snapshots_dir = repo_dir.join("snapshots");

    let ref_main = repo_dir.join("refs").join("main");
    if let Ok(rev) = std::fs::read_to_string(&ref_main) {
        let rev = rev.trim();
        if !rev.is_empty() {
            let candidate = snapshots_dir.join(rev).join(filename);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
        for entry in entries.flatten() {
            let candidate = entry.path().join(filename);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

pub fn select_device(backend: &str) -> ferrum_types::Device {
    match backend.to_lowercase().as_str() {
        "cpu" => ferrum_types::Device::CPU,
        "metal" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return ferrum_types::Device::Metal;
            }
            #[allow(unreachable_code)]
            {
                eprintln!("Metal not available, falling back to CPU");
                ferrum_types::Device::CPU
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                return ferrum_types::Device::CUDA(0);
            }
            #[allow(unreachable_code)]
            {
                eprintln!("CUDA not available, falling back to CPU");
                ferrum_types::Device::CPU
            }
        }
        "auto" | _ => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return ferrum_types::Device::Metal;
            }
            #[cfg(feature = "cuda")]
            {
                return ferrum_types::Device::CUDA(0);
            }
            #[allow(unreachable_code)]
            ferrum_types::Device::CPU
        }
    }
}

fn build_chat_prompt(
    history: &[(String, String)],
    user_input: &str,
    system: Option<&str>,
    model_id: &str,
) -> String {
    // Detect model type and use appropriate template
    let model_lower = model_id.to_lowercase();

    if model_lower.contains("qwen") {
        // Qwen ChatML format (Qwen2, Qwen2.5, Qwen3)
        let mut prompt = String::new();
        if let Some(sys) = system {
            prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
        }
        for (role, content) in history {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
        prompt.push_str(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_input
        ));
        // Qwen3: disable thinking mode by inserting empty think block
        if model_lower.contains("qwen3") {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        prompt
    } else if model_lower.contains("llama") && model_lower.contains("3") {
        // Llama 3 format
        let mut prompt = String::new();
        prompt.push_str("<|begin_of_text|>");
        if let Some(sys) = system {
            prompt.push_str(&format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                sys
            ));
        }
        for (role, content) in history {
            prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                role, content
            ));
        }
        prompt.push_str(&format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            user_input
        ));
        prompt
    } else {
        // TinyLlama / generic chat format
        let sys = system.unwrap_or("You are a helpful assistant.");
        let mut prompt = format!("<|system|>\n{}</s>\n", sys);
        for (role, content) in history {
            let tag = if role == "user" { "user" } else { "assistant" };
            prompt.push_str(&format!("<|{}|>\n{}</s>\n", tag, content));
        }
        prompt.push_str(&format!("<|user|>\n{}</s>\n<|assistant|>\n", user_input));
        prompt
    }
}

/// Apply the resolved `--kv-dtype` / runtime-config override to an engine
/// config, validating early. Default is FP16 (the production-validated path on
/// every backend); selecting INT8 / FP8 is rejected with a helpful message
/// until model integration ships.
pub fn apply_kv_dtype_override(
    engine_config: &mut ferrum_types::EngineConfig,
    raw: Option<&str>,
) -> ferrum_types::Result<()> {
    use ferrum_types::KvCacheDtype;
    let Some(raw) = raw else {
        // No override → keep config default (FP16).
        return Ok(());
    };
    let parsed = KvCacheDtype::parse(raw).ok_or_else(|| {
        ferrum_types::FerrumError::config(format!(
            "Unknown --kv-dtype value '{}'. Accepts: fp16, bf16, int8, fp8.",
            raw
        ))
    })?;
    match parsed {
        KvCacheDtype::Fp16 => {
            engine_config.kv_cache.dtype = KvCacheDtype::Fp16;
            Ok(())
        }
        KvCacheDtype::Int8 => {
            // Dim 5 PR C: end-to-end INT8 KV path on CUDA via
            // LlamaFamilyModel<CudaBackend, KvInt8>. Registry rejects
            // (CPU/Metal, Int8) and (CUDA Qwen3-MoE, Int8) with helpful
            // messages.
            engine_config.kv_cache.dtype = KvCacheDtype::Int8;
            Ok(())
        }
        KvCacheDtype::Fp8 => Err(ferrum_types::FerrumError::unsupported(
            "FP8 KV cache: kernels not yet implemented. Tracked as PR D.",
        )),
        KvCacheDtype::Bf16 => Err(ferrum_types::FerrumError::unsupported(
            "BF16 KV cache: marker only, no backend impl ships yet.",
        )),
    }
}
