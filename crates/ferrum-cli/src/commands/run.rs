//! Run command - Interactive chat with a model (ollama-style)

use crate::config::CliConfig;
use chrono::Utc;
use clap::{Args, ValueEnum};
use colored::*;
use console::{measure_text_width, Key, Term};
use ferrum_models::source::{ModelFormat, ResolvedModelSource};
use ferrum_server::chat_template::{ChatTemplateOptions, ModelChatTemplate, PromptMessage};
use ferrum_types::{
    FerrumConfigBuilder, FerrumError, FinishReason, InferenceRequest, ModelCapabilities, Priority,
    RequestId, ResolvedFerrumConfig, Result, RuntimeConfigEntry, RuntimeConfigSnapshot,
    RuntimeConfigSource, SamplingParams, WorkloadProfile,
};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::io::{self, BufRead, IsTerminal, Write};
#[cfg(unix)]
use std::mem;
#[cfg(unix)]
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const RUN_INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY: &str = "ferrum_initial_forbidden_token_texts";
const THINK_START_TAG: &str = "<think>";
const THINK_END_TAG: &str = "</think>";

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

fn run_request_metadata(
    prompt: &str,
    chat_template_options: &ChatTemplateOptions,
) -> HashMap<String, serde_json::Value> {
    let mut metadata = HashMap::new();
    if !has_unclosed_thinking_block(prompt) {
        let mut forbidden = vec![serde_json::Value::String(THINK_END_TAG.to_string())];
        if chat_template_options.enable_thinking == Some(false) {
            forbidden.push(serde_json::Value::String(THINK_START_TAG.to_string()));
        }
        metadata.insert(
            RUN_INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY.to_string(),
            serde_json::Value::Array(forbidden),
        );
    }
    metadata
}

fn has_unclosed_thinking_block(prompt: &str) -> bool {
    match (prompt.rfind(THINK_START_TAG), prompt.rfind(THINK_END_TAG)) {
        (Some(start), Some(end)) => start > end,
        (Some(_), None) => true,
        _ => false,
    }
}

#[derive(Debug, Clone)]
struct RunPromptPlan {
    prompt: String,
    sampling_params: SamplingParams,
    prompt_tokens: Option<usize>,
    kv_capacity: Option<usize>,
    dropped_history_messages: usize,
    dropped_history_turns: usize,
    max_tokens_clamped_from: Option<usize>,
}

struct RunBudget {
    tokenizer: Option<tokenizers::Tokenizer>,
    kv_capacity: Option<usize>,
    #[cfg(test)]
    prompt_token_counter: Option<fn(&str) -> usize>,
}

impl RunBudget {
    fn from_source(source_path: &Path, snapshot: &RuntimeConfigSnapshot) -> Self {
        let tokenizer = discover_run_tokenizer_path(source_path)
            .and_then(|path| tokenizers::Tokenizer::from_file(path).ok());
        let kv_capacity =
            crate::runtime_env::runtime_snapshot_value(snapshot, "FERRUM_KV_CAPACITY")
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0);
        Self {
            tokenizer,
            kv_capacity,
            #[cfg(test)]
            prompt_token_counter: None,
        }
    }

    fn prompt_tokens(&self, prompt: &str) -> Option<usize> {
        #[cfg(test)]
        if let Some(counter) = self.prompt_token_counter {
            return Some(counter(prompt));
        }
        self.tokenizer
            .as_ref()
            .and_then(|tok| tok.encode(prompt, true).ok())
            .map(|encoding| encoding.len())
    }
}

fn display_response_text(text: &str) -> String {
    text.trim().to_string()
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
    #[arg(long, default_value = "1024")]
    pub max_tokens: u32,

    /// Enable model reasoning for chat templates that support it.
    #[arg(long, conflicts_with = "disable_thinking")]
    pub enable_thinking: bool,

    /// Disable model reasoning for chat templates that support it.
    #[arg(long, conflicts_with = "enable_thinking")]
    pub disable_thinking: bool,

    /// Disable CLI context shift. By default, `ferrum run` keeps the REPL
    /// alive by dropping the oldest history before shrinking this turn's output
    /// budget, then clamps only when the current turn itself leaves too little
    /// room in KV.
    #[arg(long)]
    pub no_context_shift: bool,

    /// Sampling temperature (0.0–2.0). 0.0 = greedy / argmax (deterministic,
    /// what you want for benchmarks). >0 = softmax sample with `--top-k`
    /// and `--top-p` filtering applied.
    #[arg(long, default_value = "0.0")]
    pub temperature: f32,

    /// Backend: auto, cpu, metal (default: auto)
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// CUDA GPU ids to use, comma-separated. Multi-GPU requests fail until
    /// the real layer-split loader is implemented.
    #[arg(long, value_name = "IDS")]
    pub gpu_devices: Option<String>,

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

    /// Random seed for sampling (when temperature > 0). Omit for non-deterministic chat.
    #[arg(long)]
    pub seed: Option<u64>,

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

    /// Write resolved startup runtime config JSON and exit artifacts.
    #[arg(long)]
    pub effective_config_json: Option<PathBuf>,

    /// Write one auto-config decision JSON record per line.
    #[arg(long)]
    pub decision_trace_jsonl: Option<PathBuf>,

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

    // Select device before model resolution so CPU runs do not materialize
    // GPU/Metal chat-profile defaults such as paged KV.
    let mut device = select_device(&cmd.backend);
    let gpu_selection =
        crate::gpu_devices::resolve_cuda_gpu_devices(cmd.gpu_devices.as_deref(), &device)?;
    if let Some(selection) = &gpu_selection {
        device = selection.primary_device();
        eprintln!(
            "{} {} ({})",
            "CUDA GPUs:".dimmed(),
            selection.selected_csv(),
            selection.selected_distributed_strategy
        );
    }
    let autosize = run_autosize_for_device(&device, cmd.gpu_memory_utilization);

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
        autosize,
    )
    .await?;
    let source = resolved.source;
    let model_id = source.original.clone();
    let model_definition_for_config = load_run_model_definition(&source).await?;
    let model_chat_template = crate::source_resolver::load_model_chat_template(&source.local_path);
    let chat_template_options = build_chat_template_options(&cmd, model_chat_template.as_ref());
    eprintln!("{}", format!("Loading {}...", model_id).dimmed());

    let engine_model_path = source.local_path.to_string_lossy().to_string();

    let device_label = format!("{device:?}");
    eprintln!("{}", format!("Using {device_label} backend").dimmed());
    let metal_moe_entries = crate::source_resolver::metal_gguf_moe_correctness_entries(
        &source.local_path,
        &device,
        &ferrum_types::RuntimeConfigSnapshot::capture_current(),
        ferrum_types::RuntimeConfigSource::Default,
    );
    crate::runtime_env::materialize_runtime_env_defaults(&metal_moe_entries);

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
    engine_config.backend.device = device.clone();
    engine_config.scheduler.policy = ferrum_types::SchedulingPolicy::ContinuousBatch;
    engine_config.backend.backend_options.insert(
        "model_path".to_string(),
        serde_json::Value::String(engine_model_path),
    );
    let runtime_config = RuntimeConfigSnapshot::capture_current();
    if let Some(selection) = &gpu_selection {
        selection.insert_backend_options(&mut engine_config.backend.backend_options);
    }
    let effective_runtime_config = run_effective_runtime_config(
        &runtime_config,
        cmd.kv_dtype.as_deref(),
        gpu_selection.as_ref(),
    );
    let startup_auto_config = run_startup_auto_config(
        &device,
        model_definition_for_config.as_ref(),
        effective_runtime_config,
    )?;
    crate::commands::serve::write_startup_config_artifacts(
        &startup_auto_config,
        cmd.effective_config_json.as_deref(),
        cmd.decision_trace_jsonl.as_deref(),
    )?;
    let run_budget = RunBudget::from_source(&source.local_path, &runtime_config);
    engine_config
        .apply_runtime_config_snapshot(&runtime_config)
        .map_err(ferrum_types::FerrumError::config)?;
    if runtime_config_bool(&runtime_config, "FERRUM_METAL_PAGED_KV").unwrap_or(false) {
        engine_config.kv_cache.cache_type = ferrum_types::KvCacheType::Paged;
    }
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
        let format = cmd.output_format;
        let plan = build_run_prompt_plan(
            &[],
            &one_shot,
            cmd.system.as_deref(),
            &model_id,
            model_chat_template.as_ref(),
            &chat_template_options,
            &cmd,
            &run_budget,
        )?;
        maybe_warn_context_shift(&plan, format);
        let metadata = run_request_metadata(&plan.prompt, &chat_template_options);
        let request = InferenceRequest {
            id: RequestId(Uuid::new_v4()),
            model_id: ferrum_types::ModelId(model_id.clone()),
            prompt: plan.prompt,
            sampling_params: plan.sampling_params,
            stream: false,
            priority: Priority::Normal,
            client_id: None,
            session_id: None,
            created_at: Utc::now(),
            api_request: None,
            metadata,
        };
        let start = std::time::Instant::now();
        let response = engine.infer(request).await?;
        let tokens = response.tokens.len();
        let content = display_response_text(&response.text);
        let chunk_count = usize::from(!content.is_empty());
        let finish_reason = Some(response.finish_reason);
        let bench = cmd.bench_mode;
        if format == OutputFormat::Text && !bench {
            print!("{}", content);
            io::stdout().flush().ok();
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
        engine.shutdown().await?;
        return Ok(());
    }

    // Print ready message
    let format = cmd.output_format;
    match format {
        OutputFormat::Text => {
            eprintln!();
            eprintln!("{}", "Ready. Type your message and press Enter.".green());
            eprintln!(
                "{}",
                "Use /clear to reset history; /bye or Ctrl+D to exit.".dimmed()
            );
            eprintln!();
        }
        OutputFormat::Jsonl => {
            emit_jsonl_ready(&model_id, &device_label);
        }
    }

    // Interactive loop
    let mut history: Vec<(String, String)> = Vec::new(); // (role, content)
    let mut turn = 0usize;
    let mut exit_reason: &str = "eof";

    // If stdin is not a TTY (piped input), don't print prompts and just consume lines.
    // This enables: `printf "hi\n/bye\n" | ferrum run ...` for automation/profiling.
    let stdin_handle = io::stdin();
    let stdin_is_tty = stdin_handle.is_terminal();
    let term = stdin_is_tty.then(Term::stdout);
    let mut stdin = stdin_handle.lock();

    loop {
        if stdin_is_tty {
            // Show prompt
            print!("{} ", ">>>".bright_green().bold());
            io::stdout().flush().unwrap();
        }

        // Read input
        let mut input = String::new();
        match read_repl_input_line(term.as_ref(), &mut stdin, &mut input) {
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
                if input == "/clear" {
                    history.clear();
                    match format {
                        OutputFormat::Text => {
                            eprintln!("{}", "History cleared.".dimmed());
                        }
                        OutputFormat::Jsonl => {
                            let record = serde_json::json!({
                                "event": "clear",
                                "turn": turn,
                            });
                            println!("{record}");
                        }
                    }
                    continue;
                }

                if format == OutputFormat::Jsonl {
                    emit_jsonl_user(turn, input);
                }

                let plan = build_run_prompt_plan(
                    &history,
                    input,
                    cmd.system.as_deref(),
                    &model_id,
                    model_chat_template.as_ref(),
                    &chat_template_options,
                    &cmd,
                    &run_budget,
                )?;
                maybe_warn_context_shift(&plan, format);
                let metadata = run_request_metadata(&plan.prompt, &chat_template_options);
                // Create request
                let request = InferenceRequest {
                    id: RequestId(Uuid::new_v4()),
                    model_id: ferrum_types::ModelId(model_id.clone()),
                    prompt: plan.prompt,
                    sampling_params: plan.sampling_params,
                    stream: format == OutputFormat::Text,
                    priority: Priority::Normal,
                    client_id: None,
                    session_id: None,
                    created_at: Utc::now(),
                    api_request: None,
                    metadata,
                };

                let start = std::time::Instant::now();
                let trace_tokens = crate::runtime_env::runtime_snapshot_value(
                    &runtime_config,
                    "FERRUM_RUN_TRACE_TOKENS",
                )
                .is_some();
                let (clean_response, finish_reason, token_count, chunk_count) = match format {
                    OutputFormat::Text => {
                        let mut first_token_indicator = start_first_token_indicator(stdin_is_tty);
                        let mut stream = match engine.infer_stream(request).await {
                            Ok(stream) => stream,
                            Err(e) => {
                                clear_first_token_indicator(&mut first_token_indicator);
                                return Err(e);
                            }
                        };
                        let mut raw_response_text = String::new();
                        let mut finish_reason = None;
                        let mut token_count = 0usize;
                        let mut chunk_count = 0usize;
                        while let Some(chunk) = stream.next().await {
                            let chunk = match chunk {
                                Ok(chunk) => chunk,
                                Err(e) => {
                                    clear_first_token_indicator(&mut first_token_indicator);
                                    return Err(e);
                                }
                            };
                            if first_token_indicator.is_some()
                                && (!chunk.text.is_empty()
                                    || chunk.token.is_some()
                                    || chunk.finish_reason.is_some())
                            {
                                clear_first_token_indicator(&mut first_token_indicator);
                            }
                            if trace_tokens {
                                if let Some(token) = chunk.token {
                                    eprintln!(
                                        "[run-token-trace] turn={turn} token={} text={:?}",
                                        token.get(),
                                        chunk.text
                                    );
                                }
                            }
                            if !chunk.text.is_empty() {
                                raw_response_text.push_str(&chunk.text);
                                print!("{}", chunk.text);
                                io::stdout().flush().ok();
                                chunk_count += 1;
                            }
                            if chunk.token.is_some() {
                                token_count += 1;
                            }
                            if let Some(usage) = chunk.usage.as_ref() {
                                token_count = usage.completion_tokens;
                            }
                            if chunk.finish_reason.is_some() {
                                finish_reason = chunk.finish_reason;
                            }
                        }
                        clear_first_token_indicator(&mut first_token_indicator);
                        (
                            display_response_text(&raw_response_text),
                            finish_reason,
                            token_count,
                            chunk_count,
                        )
                    }
                    OutputFormat::Jsonl => {
                        let response = engine.infer(request).await?;
                        if trace_tokens {
                            let tokens = response
                                .tokens
                                .iter()
                                .map(|token| token.get())
                                .collect::<Vec<_>>();
                            eprintln!("[run-token-trace] turn={turn} tokens={tokens:?}");
                        }
                        let content = display_response_text(&response.text);
                        let chunk_count = usize::from(!content.is_empty());
                        (
                            content,
                            Some(response.finish_reason),
                            response.tokens.len(),
                            chunk_count,
                        )
                    }
                };

                let elapsed = start.elapsed();
                let elapsed_s = elapsed.as_secs_f64();
                let tps = if elapsed_s > 0.0 {
                    token_count as f64 / elapsed_s
                } else {
                    0.0
                };

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
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {
                exit_reason = "interrupt";
                break;
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
    engine.shutdown().await?;
    Ok(())
}

#[cfg(unix)]
fn read_repl_input_line<R: BufRead + AsRawFd>(
    term: Option<&Term>,
    stdin: &mut R,
    input: &mut String,
) -> io::Result<usize> {
    if let Some(term) = term {
        let Some(line) = read_tty_input_line(term, stdin)? else {
            return Ok(0);
        };
        input.push_str(&line);
        return Ok(input.len().max(1));
    }
    stdin.read_line(input)
}

#[cfg(not(unix))]
fn read_repl_input_line<R: BufRead>(
    term: Option<&Term>,
    stdin: &mut R,
    input: &mut String,
) -> io::Result<usize> {
    if let Some(term) = term {
        let Some(line) = read_tty_input_line(term)? else {
            return Ok(0);
        };
        input.push_str(&line);
        return Ok(input.len().max(1));
    }
    stdin.read_line(input)
}

#[cfg(unix)]
struct RawModeGuard {
    fd: RawFd,
    original: libc::termios,
}

#[cfg(unix)]
impl RawModeGuard {
    fn new(fd: RawFd) -> io::Result<Self> {
        let mut termios = mem::MaybeUninit::uninit();
        if unsafe { libc::tcgetattr(fd, termios.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let original = unsafe { termios.assume_init() };
        let mut raw = original;
        unsafe { libc::cfmakeraw(&mut raw) };
        raw.c_oflag = original.c_oflag;
        if unsafe { libc::tcsetattr(fd, libc::TCSADRAIN, &raw) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { fd, original })
    }
}

#[cfg(unix)]
impl Drop for RawModeGuard {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSADRAIN, &self.original);
        }
    }
}

#[cfg(unix)]
fn read_tty_input_line<R: AsRawFd>(term: &Term, stdin: &mut R) -> io::Result<Option<String>> {
    let _raw_mode = RawModeGuard::new(stdin.as_raw_fd())?;
    let mut chars: Vec<char> = Vec::new();
    loop {
        match term.read_key_raw()? {
            Key::Backspace => {
                if let Some(ch) = chars.pop() {
                    let width = measure_text_width(&ch.to_string());
                    if width > 0 {
                        term.clear_chars(width)?;
                    }
                    term.flush()?;
                }
            }
            Key::Char('\u{4}') => {
                term.write_str("\n")?;
                term.flush()?;
                if chars.is_empty() {
                    return Ok(None);
                }
                break;
            }
            Key::CtrlC | Key::Char('\u{3}') => {
                term.write_str("^C\n")?;
                term.flush()?;
                return Err(io::Error::new(io::ErrorKind::Interrupted, "interrupted"));
            }
            Key::Enter => {
                term.write_str("\n")?;
                term.flush()?;
                break;
            }
            Key::Char(ch) if !ch.is_ascii_control() => {
                chars.push(ch);
                term.write_str(&ch.to_string())?;
                term.flush()?;
            }
            _ => {}
        }
    }
    Ok(Some(chars.into_iter().collect()))
}

#[cfg(not(unix))]
fn read_tty_input_line(term: &Term) -> io::Result<Option<String>> {
    let mut chars: Vec<char> = Vec::new();
    loop {
        match term.read_key()? {
            Key::Backspace => {
                if let Some(ch) = chars.pop() {
                    let width = measure_text_width(&ch.to_string());
                    if width > 0 {
                        term.clear_chars(width)?;
                    }
                    term.flush()?;
                }
            }
            Key::Char('\u{4}') => {
                term.write_str("\n")?;
                term.flush()?;
                if chars.is_empty() {
                    return Ok(None);
                }
                break;
            }
            Key::CtrlC | Key::Char('\u{3}') => {
                term.write_str("^C\n")?;
                term.flush()?;
                return Err(io::Error::new(io::ErrorKind::Interrupted, "interrupted"));
            }
            Key::Enter => {
                term.write_str("\n")?;
                term.flush()?;
                break;
            }
            Key::Char(ch) if !ch.is_ascii_control() => {
                chars.push(ch);
                term.write_str(&ch.to_string())?;
                term.flush()?;
            }
            _ => {}
        }
    }
    Ok(Some(chars.into_iter().collect()))
}

fn start_first_token_indicator(enabled: bool) -> Option<ProgressBar> {
    if !enabled {
        return None;
    }
    let progress = ProgressBar::new_spinner();
    let style = ProgressStyle::with_template("{spinner} Working ({elapsed})")
        .unwrap_or_else(|_| ProgressStyle::default_spinner());
    progress.set_style(style);
    progress.enable_steady_tick(std::time::Duration::from_millis(120));
    progress.tick();
    Some(progress)
}

fn clear_first_token_indicator(progress: &mut Option<ProgressBar>) {
    if let Some(progress) = progress.take() {
        progress.finish_and_clear();
    }
}

fn runtime_config_bool(snapshot: &RuntimeConfigSnapshot, key: &str) -> Option<bool> {
    crate::runtime_env::runtime_snapshot_value(snapshot, key).map(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "1" | "true" | "yes" | "on"
        )
    })
}

fn run_autosize_for_device(
    device: &ferrum_types::Device,
    gpu_memory_utilization: f32,
) -> Option<(crate::gpu_mem_autosize::AutoSizeProfile, f32)> {
    match device {
        ferrum_types::Device::CPU => None,
        _ => Some((
            crate::gpu_mem_autosize::AutoSizeProfile::Chat,
            gpu_memory_utilization,
        )),
    }
}

fn build_sampling_params(cmd: &RunCommand) -> SamplingParams {
    let greedy = cmd.temperature <= 0.0;
    SamplingParams {
        max_tokens: cmd.max_tokens as usize,
        temperature: cmd.temperature,
        top_p: if greedy { 1.0 } else { cmd.top_p },
        top_k: if greedy || cmd.top_k == 0 {
            None
        } else {
            Some(cmd.top_k)
        },
        repetition_penalty: cmd.repeat_penalty,
        stop_sequences: vec![
            "<|im_end|>".to_string(),
            "</s>".to_string(),
            "<|endoftext|>".to_string(),
        ],
        seed: cmd.seed,
        ..Default::default()
    }
}

fn build_chat_template_options(
    cmd: &RunCommand,
    model_template: Option<&ModelChatTemplate>,
) -> ChatTemplateOptions {
    let mut options = ChatTemplateOptions::default_for_template(model_template);
    if cmd.enable_thinking {
        options.enable_thinking = Some(true);
    } else if cmd.disable_thinking {
        options.enable_thinking = Some(false);
    }
    options
}

fn build_run_prompt_plan(
    history: &[(String, String)],
    user_input: &str,
    system: Option<&str>,
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
    chat_template_options: &ChatTemplateOptions,
    cmd: &RunCommand,
    budget: &RunBudget,
) -> Result<RunPromptPlan> {
    let base_sampling = build_sampling_params(cmd);

    if cmd.no_context_shift {
        let prompt = build_chat_prompt(
            history,
            user_input,
            system,
            model_id,
            model_template,
            chat_template_options,
        );
        let prompt_tokens = budget.prompt_tokens(&prompt);
        if !fits_kv_budget(&base_sampling, prompt_tokens, budget.kv_capacity) {
            return Err(FerrumError::invalid_request(format!(
                "This model context is limited to {} tokens, but this turn needs {} input tokens + {} output tokens. Reduce --max-tokens, use /clear, or shorten the prompt.",
                budget.kv_capacity.unwrap_or(0),
                prompt_tokens.unwrap_or(0),
                base_sampling.max_tokens,
            )));
        }

        return Ok(RunPromptPlan {
            prompt,
            sampling_params: base_sampling,
            prompt_tokens,
            kv_capacity: budget.kv_capacity,
            dropped_history_messages: 0,
            dropped_history_turns: 0,
            max_tokens_clamped_from: None,
        });
    }

    let mut history_start = 0usize;
    loop {
        let prompt = build_chat_prompt(
            &history[history_start..],
            user_input,
            system,
            model_id,
            model_template,
            chat_template_options,
        );
        let prompt_tokens = budget.prompt_tokens(&prompt);

        let Some(kv_capacity) = budget.kv_capacity else {
            return Ok(RunPromptPlan {
                prompt,
                sampling_params: base_sampling,
                prompt_tokens,
                kv_capacity: None,
                dropped_history_messages: history_start,
                dropped_history_turns: count_user_turns(&history[..history_start]),
                max_tokens_clamped_from: None,
            });
        };
        let Some(prompt_tokens) = prompt_tokens else {
            return Ok(RunPromptPlan {
                prompt,
                sampling_params: base_sampling,
                prompt_tokens: None,
                kv_capacity: Some(kv_capacity),
                dropped_history_messages: history_start,
                dropped_history_turns: count_user_turns(&history[..history_start]),
                max_tokens_clamped_from: None,
            });
        };

        if prompt_tokens < kv_capacity {
            let remaining = kv_capacity - prompt_tokens;
            if base_sampling.max_tokens > remaining && history_start < history.len() {
                history_start = next_context_shift_history_start(history, history_start);
                continue;
            }
            let mut sampling_params = base_sampling.clone();
            let max_tokens_clamped_from = if sampling_params.max_tokens > remaining {
                let old = sampling_params.max_tokens;
                sampling_params.max_tokens = remaining;
                Some(old)
            } else {
                None
            };
            return Ok(RunPromptPlan {
                prompt,
                sampling_params,
                prompt_tokens: Some(prompt_tokens),
                kv_capacity: Some(kv_capacity),
                dropped_history_messages: history_start,
                dropped_history_turns: count_user_turns(&history[..history_start]),
                max_tokens_clamped_from,
            });
        }

        if history_start >= history.len() {
            return Err(FerrumError::invalid_request(format!(
                "This model context is limited to {kv_capacity} tokens, but the current turn needs {prompt_tokens} input tokens before generation. Use a shorter prompt or increase KV capacity.",
            )));
        }
        history_start = next_context_shift_history_start(history, history_start);
    }
}

fn next_context_shift_history_start(history: &[(String, String)], start: usize) -> usize {
    if start + 1 < history.len()
        && history[start].0 == "user"
        && history[start + 1].0 == "assistant"
    {
        start + 2
    } else {
        start + 1
    }
}

fn count_user_turns(history: &[(String, String)]) -> usize {
    history.iter().filter(|(role, _)| role == "user").count()
}

fn maybe_warn_context_shift(plan: &RunPromptPlan, format: OutputFormat) {
    if format != OutputFormat::Text {
        return;
    }
    if plan.dropped_history_messages == 0 && plan.max_tokens_clamped_from.is_none() {
        return;
    }

    let prompt_tokens = plan
        .prompt_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "?".to_string());
    let kv_capacity = plan
        .kv_capacity
        .map(|value| value.to_string())
        .unwrap_or_else(|| "?".to_string());
    let mut parts = Vec::new();
    if plan.dropped_history_messages > 0 {
        parts.push(format!(
            "dropped {} old message(s) / {} turn(s)",
            plan.dropped_history_messages, plan.dropped_history_turns
        ));
    }
    if let Some(old) = plan.max_tokens_clamped_from {
        parts.push(format!(
            "max_tokens {} -> {}",
            old, plan.sampling_params.max_tokens
        ));
    }

    eprintln!(
        "{}",
        format!(
            "[context-shift] {} (prompt_tokens={}, kv_capacity={})",
            parts.join("; "),
            prompt_tokens,
            kv_capacity
        )
        .dimmed()
    );
}

fn fits_kv_budget(
    base: &SamplingParams,
    prompt_tokens: Option<usize>,
    kv_capacity: Option<usize>,
) -> bool {
    let (Some(prompt_tokens), Some(kv_capacity)) = (prompt_tokens, kv_capacity) else {
        return true;
    };
    prompt_tokens < kv_capacity && prompt_tokens + base.max_tokens <= kv_capacity
}

fn discover_run_tokenizer_path(source_path: &Path) -> Option<PathBuf> {
    if source_path.is_file()
        && source_path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    {
        return ferrum_models::gguf_engine_loader::auto_discover_tokenizer_path(source_path);
    }
    let tokenizer = source_path.join("tokenizer.json");
    tokenizer.is_file().then_some(tokenizer)
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
    model_template: Option<&ModelChatTemplate>,
    chat_template_options: &ChatTemplateOptions,
) -> String {
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(PromptMessage::new("system", sys));
    }
    for (role, content) in history {
        messages.push(PromptMessage::new(role, content));
    }
    messages.push(PromptMessage::new("user", user_input));
    ferrum_server::chat_template::render_prompt_messages_with_options(
        &messages,
        model_id,
        model_template,
        chat_template_options,
    )
}

async fn load_run_model_definition(
    source: &ResolvedModelSource,
) -> Result<Option<ferrum_models::ModelDefinition>> {
    if source.format != ModelFormat::SafeTensors {
        return Ok(None);
    }
    let mut config_manager = ferrum_models::ConfigManager::new();
    Ok(Some(
        config_manager.load_from_path(&source.local_path).await?,
    ))
}

fn run_effective_runtime_config(
    runtime_config: &RuntimeConfigSnapshot,
    kv_dtype: Option<&str>,
    gpu_selection: Option<&crate::gpu_devices::GpuDeviceSelection>,
) -> RuntimeConfigSnapshot {
    let mut snapshot = runtime_config.clone();
    if let Some(value) = kv_dtype.filter(|value| !value.trim().is_empty()) {
        snapshot.upsert_entry(RuntimeConfigEntry::new(
            "FERRUM_KV_DTYPE",
            value.to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(selection) = gpu_selection {
        for entry in selection.runtime_config_entries() {
            snapshot.upsert_entry(entry);
        }
    }
    snapshot
}

fn run_startup_auto_config(
    device: &ferrum_types::Device,
    model_definition: Option<&ferrum_models::ModelDefinition>,
    runtime_config: RuntimeConfigSnapshot,
) -> Result<ResolvedFerrumConfig> {
    let model = model_definition
        .map(crate::commands::serve::model_capabilities_from_definition)
        .unwrap_or_else(ModelCapabilities::unknown);
    let hardware = crate::commands::serve::hardware_capabilities_for_device(device);
    let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
    FerrumConfigBuilder::new(runtime_config)
        .with_model_capabilities(model)
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .map_err(|err| ferrum_types::FerrumError::config(format!("invalid auto config: {err}")))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            max_tokens,
            ..SamplingParams::default()
        }
    }

    fn test_run_cmd() -> RunCommand {
        RunCommand {
            model: "tinyllama".to_string(),
            system: None,
            max_tokens: 1024,
            no_context_shift: false,
            enable_thinking: false,
            disable_thinking: false,
            temperature: 0.0,
            backend: "auto".to_string(),
            gpu_devices: None,
            prompt: None,
            tokenizer: None,
            bench_mode: false,
            top_k: 50,
            top_p: 0.95,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: None,
            gpu_memory_utilization: 0.9,
            kv_dtype: None,
            effective_config_json: None,
            decision_trace_jsonl: None,
            output_format: OutputFormat::Text,
        }
    }

    fn whitespace_budget(kv_capacity: usize) -> RunBudget {
        RunBudget {
            tokenizer: None,
            kv_capacity: Some(kv_capacity),
            prompt_token_counter: Some(|prompt| prompt.split_whitespace().count()),
        }
    }

    fn default_template_options() -> ChatTemplateOptions {
        ChatTemplateOptions::default()
    }

    #[test]
    fn run_effective_runtime_config_records_cli_kv_dtype() {
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let effective = run_effective_runtime_config(&snapshot, Some("int8"), None);
        let entry = effective
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_KV_DTYPE")
            .expect("missing kv dtype entry");
        assert_eq!(entry.effective_value, "int8");
        assert_eq!(entry.source, RuntimeConfigSource::Cli);
    }

    #[test]
    fn run_effective_runtime_config_records_gpu_device_selection() {
        let selection = crate::gpu_devices::GpuDeviceSelection {
            raw_cli_value: "1".to_string(),
            requested_gpu_devices: vec![1],
            selected_gpu_devices: vec![1],
            selected_distributed_strategy: "single_gpu".to_string(),
        };
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let effective = run_effective_runtime_config(&snapshot, None, Some(&selection));
        let entry = |key: &str| {
            effective
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_BACKEND").effective_value, "cuda");
        assert_eq!(entry("FERRUM_REQUESTED_GPU_DEVICES").effective_value, "1");
        assert_eq!(entry("FERRUM_SELECTED_GPU_DEVICES").effective_value, "1");
        assert_eq!(
            entry("FERRUM_SELECTED_DISTRIBUTED_STRATEGY").effective_value,
            "single_gpu"
        );
    }

    #[test]
    fn run_startup_auto_config_renders_effective_config_schema() {
        let resolved = run_startup_auto_config(
            &ferrum_types::Device::CPU,
            None,
            RuntimeConfigSnapshot::from_entries(Vec::new()),
        )
        .expect("auto config");
        let doc = resolved.effective_config_document();
        assert_eq!(doc["schema_version"], 1);
        assert!(doc["entries"].is_array());
        assert!(doc["model_capabilities"].is_object());
        assert!(doc["hardware_capabilities"].is_object());
        assert!(doc["workload_profile"].is_object());
        assert!(doc["decisions"].is_array());
    }

    #[test]
    fn run_metadata_forbids_initial_thinking_close_without_open_block() {
        let metadata =
            run_request_metadata("<|im_start|>assistant\n", &ChatTemplateOptions::default());
        let forbidden = metadata
            .get(RUN_INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY)
            .and_then(|value| value.as_array())
            .expect("initial forbidden token texts");
        assert_eq!(
            forbidden,
            &[serde_json::Value::String(THINK_END_TAG.to_string())]
        );
    }

    #[test]
    fn run_metadata_forbids_initial_thinking_start_when_template_disables_thinking() {
        let metadata = run_request_metadata(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            &ChatTemplateOptions {
                enable_thinking: Some(false),
            },
        );
        let forbidden = metadata
            .get(RUN_INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY)
            .and_then(|value| value.as_array())
            .expect("initial forbidden token texts");
        assert_eq!(
            forbidden,
            &[
                serde_json::Value::String(THINK_END_TAG.to_string()),
                serde_json::Value::String(THINK_START_TAG.to_string()),
            ]
        );
    }

    #[test]
    fn run_metadata_allows_thinking_close_when_prompt_opened_block() {
        let metadata = run_request_metadata(
            "<|im_start|>assistant\n<think>\n",
            &ChatTemplateOptions::default(),
        );
        assert!(!metadata.contains_key(RUN_INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY));
    }

    #[test]
    fn cli_display_preserves_thinking_markers() {
        assert_eq!(
            display_response_text("<think>\nreasoning\n</think>\n\n最终答案"),
            "<think>\nreasoning\n</think>\n\n最终答案"
        );
    }

    #[test]
    fn cli_display_preserves_orphan_think_close() {
        assert_eq!(
            display_response_text("</think>\n\n你好！很高兴见到你。"),
            "</think>\n\n你好！很高兴见到你。"
        );
    }

    #[test]
    fn default_run_temperature_is_greedy() {
        let cmd = test_run_cmd();
        assert_eq!(build_sampling_params(&cmd).temperature, 0.0);
        assert_eq!(build_sampling_params(&cmd).max_tokens, 1024);
    }

    #[test]
    fn cpu_run_skips_gpu_chat_autosize_defaults() {
        assert!(run_autosize_for_device(&ferrum_types::Device::CPU, 0.9).is_none());
    }

    #[cfg(any(all(target_os = "macos", feature = "metal"), feature = "cuda"))]
    #[test]
    fn accelerator_run_keeps_chat_autosize_defaults() {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let device = ferrum_types::Device::Metal;
        #[cfg(all(feature = "cuda", not(all(target_os = "macos", feature = "metal"))))]
        let device = ferrum_types::Device::CUDA(0);

        let autosize = run_autosize_for_device(&device, 0.75);
        assert_eq!(
            autosize,
            Some((crate::gpu_mem_autosize::AutoSizeProfile::Chat, 0.75))
        );
    }

    #[test]
    fn context_shift_clamps_output_to_remaining_kv_budget() {
        let cmd = test_run_cmd();
        let budget = whitespace_budget(64);
        let options = default_template_options();
        let plan = build_run_prompt_plan(
            &[],
            "demo",
            None,
            "tinyllama",
            None,
            &options,
            &cmd,
            &budget,
        )
        .unwrap();

        let prompt_tokens = plan.prompt_tokens.unwrap();
        assert!(prompt_tokens < 64);
        assert_eq!(plan.max_tokens_clamped_from, Some(1024));
        assert_eq!(plan.sampling_params.max_tokens, 64 - prompt_tokens);
    }

    #[test]
    fn context_shift_drops_oldest_history_until_prompt_fits() {
        let cmd = test_run_cmd();
        let budget = whitespace_budget(64);
        let long = std::iter::repeat_n("old", 80).collect::<Vec<_>>().join(" ");
        let history = vec![
            ("user".to_string(), long.clone()),
            ("assistant".to_string(), long),
        ];
        let options = default_template_options();
        let plan = build_run_prompt_plan(
            &history,
            "demo",
            None,
            "tinyllama",
            None,
            &options,
            &cmd,
            &budget,
        )
        .unwrap();

        assert_eq!(plan.dropped_history_messages, 2);
        assert_eq!(plan.dropped_history_turns, 1);
        assert!(plan.prompt_tokens.unwrap() < 64);
    }

    #[test]
    fn context_shift_drops_history_before_clamping_output_budget() {
        let mut cmd = test_run_cmd();
        cmd.max_tokens = 32;
        let budget = whitespace_budget(64);
        let medium = std::iter::repeat_n("old", 12).collect::<Vec<_>>().join(" ");
        let history = vec![
            ("user".to_string(), medium.clone()),
            ("assistant".to_string(), medium),
        ];
        let options = default_template_options();
        let plan = build_run_prompt_plan(
            &history,
            "demo",
            None,
            "tinyllama",
            None,
            &options,
            &cmd,
            &budget,
        )
        .unwrap();

        assert_eq!(plan.dropped_history_messages, 2);
        assert_eq!(plan.dropped_history_turns, 1);
        assert_eq!(plan.max_tokens_clamped_from, None);
        assert_eq!(plan.sampling_params.max_tokens, 32);
    }

    #[test]
    fn kv_budget_accepts_request_inside_capacity() {
        assert!(fits_kv_budget(&default_params(512), Some(64), Some(2048)));
    }

    #[test]
    fn kv_budget_rejects_output_past_capacity() {
        assert!(!fits_kv_budget(&default_params(2048), Some(64), Some(2048)));
    }

    #[test]
    fn kv_budget_rejects_prompt_at_capacity() {
        assert!(!fits_kv_budget(&default_params(1), Some(2048), Some(2048)));
    }
}
