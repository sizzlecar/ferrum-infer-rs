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
    RuntimeConfigSource, SamplingParams, WorkloadProfile, DEFAULT_CHAT_REPETITION_PENALTY,
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

#[cfg(test)]
use crate::source_resolver::tokenizer_sibling_repo;

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
    prompt_token_ids: Option<Vec<u32>>,
    prompt_tokens: Option<usize>,
    kv_capacity: Option<usize>,
    dropped_history_messages: usize,
    dropped_history_turns: usize,
    max_tokens_clamped_from: Option<usize>,
}

#[derive(Debug, Clone)]
struct RunPromptTokenization {
    token_ids: Option<Vec<u32>>,
    token_count: Option<usize>,
}

struct RunBudget {
    tokenizer: Option<tokenizers::Tokenizer>,
    kv_capacity: Option<usize>,
    #[cfg(test)]
    prompt_token_id_mapper: Option<fn(&str) -> Vec<u32>>,
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
            prompt_token_id_mapper: None,
            #[cfg(test)]
            prompt_token_counter: None,
        }
    }

    fn prompt_tokenization(&self, prompt: &str) -> RunPromptTokenization {
        #[cfg(test)]
        if let Some(mapper) = self.prompt_token_id_mapper {
            let token_ids = mapper(prompt);
            return RunPromptTokenization {
                token_count: Some(token_ids.len()),
                token_ids: Some(token_ids),
            };
        }
        #[cfg(test)]
        if let Some(counter) = self.prompt_token_counter {
            return RunPromptTokenization {
                token_ids: None,
                token_count: Some(counter(prompt)),
            };
        }
        if let Some(encoding) = self
            .tokenizer
            .as_ref()
            .and_then(|tok| tok.encode(prompt, true).ok())
        {
            let token_ids = encoding.get_ids().to_vec();
            return RunPromptTokenization {
                token_count: Some(token_ids.len()),
                token_ids: Some(token_ids),
            };
        }
        RunPromptTokenization {
            token_ids: None,
            token_count: None,
        }
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

    /// Stop generation when this text appears. Can be provided multiple times.
    #[arg(long, value_name = "TEXT")]
    pub stop: Vec<String>,

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

    /// Backend: auto, cpu, metal, cuda (default: auto)
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// Enable the explicit CPU/FP32 Qwen3.5/Qwen3.6 reference executor for W3
    /// correctness bring-up. This is not a release-performance path.
    #[arg(long)]
    pub qwen35_reference: bool,

    /// CUDA GPU ids to use, comma-separated. Multi-GPU requests select
    /// layer-split for supported Llama-family safetensors models.
    #[arg(long, value_name = "IDS")]
    pub gpu_devices: Option<String>,

    /// Layer-split decode pipeline mode for multi-GPU CUDA runs.
    #[arg(long, value_enum)]
    pub layer_split_pipeline_mode: Option<crate::layer_split_pipeline::LayerSplitPipelineModeArg>,

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
    /// repeats, <1 encourages, 1.0 disables. Defaults to 1.1 (OpenAI/llama.cpp
    /// standard) because the chat default is greedy (temperature 0): greedy
    /// with no penalty deterministically locks into token loops on some
    /// inputs (the "2D/3D 2D/3D..." degeneration). Pass `--repeat-penalty 1.0`
    /// for an unpenalized greedy baseline.
    #[arg(long, default_value_t = DEFAULT_CHAT_REPETITION_PENALTY)]
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

    /// Exact device-wide memory budget available to runtime weights and
    /// dynamic resources. This is a typed capacity ceiling shared by `run`
    /// and `serve`; omit it to use the normal pressure-threshold policy.
    #[arg(long, value_name = "BYTES")]
    pub runtime_memory_budget_bytes: Option<std::num::NonZeroUsize>,

    /// vLLM-compatible alias for `FERRUM_MAX_MODEL_LEN`.
    #[arg(long, value_name = "N")]
    pub max_model_len: Option<usize>,

    /// vLLM-compatible alias for `FERRUM_PAGED_MAX_SEQS`.
    #[arg(long, value_name = "N")]
    pub max_num_seqs: Option<usize>,

    /// vLLM-compatible alias for `FERRUM_MAX_BATCHED_TOKENS`.
    #[arg(long, value_name = "N")]
    pub max_num_batched_tokens: Option<usize>,

    /// Enable legacy Llama/Gemma batched decode CUDA graph replay.
    #[arg(long, conflicts_with = "disable_batched_graph")]
    pub batched_graph: bool,

    /// Disable legacy Llama/Gemma batched decode CUDA graph replay.
    #[arg(long, conflicts_with = "batched_graph")]
    pub disable_batched_graph: bool,

    /// Enable Llama/Gemma unified decode CUDA graph replay.
    #[arg(long, conflicts_with = "disable_unified_graph")]
    pub unified_graph: bool,

    /// Disable Llama/Gemma unified decode CUDA graph replay.
    #[arg(long, conflicts_with = "unified_graph")]
    pub disable_unified_graph: bool,

    /// Capture only Llama/Gemma unified transformer layers in CUDA graph replay.
    #[arg(long, conflicts_with = "disable_unified_graph_layers_only")]
    pub unified_graph_layers_only: bool,

    /// Disable layers-only unified CUDA graph capture scope.
    #[arg(long, conflicts_with = "unified_graph_layers_only")]
    pub disable_unified_graph_layers_only: bool,

    /// Capture unified layers plus final packing; leave lm_head eager.
    #[arg(long, conflicts_with = "disable_unified_graph_lm_head_eager")]
    pub unified_graph_lm_head_eager: bool,

    /// Disable lm-head-eager unified CUDA graph capture scope.
    #[arg(long, conflicts_with = "unified_graph_lm_head_eager")]
    pub disable_unified_graph_lm_head_eager: bool,

    /// KV cache element dtype (Dim 5 polymorphism point). Accepts
    /// `fp16`, `bf16`, `int8`, `fp8`. Default `fp16`. INT8 / FP8
    /// require model wire-up; today only the kernel + type layer ships.
    /// Override via `FERRUM_KV_DTYPE` env var.
    #[arg(long, value_name = "DTYPE")]
    pub kv_dtype: Option<String>,

    /// Per-sequence KV token capacity (`FERRUM_KV_CAPACITY`).
    #[arg(long, value_name = "N")]
    pub kv_capacity: Option<usize>,

    /// KV block budget (`FERRUM_KV_MAX_BLOCKS`).
    #[arg(long, value_name = "N")]
    pub kv_max_blocks: Option<usize>,

    /// Write resolved startup runtime config JSON and exit artifacts.
    #[arg(long)]
    pub effective_config_json: Option<PathBuf>,

    /// Write one auto-config decision JSON record per line.
    #[arg(long)]
    pub decision_trace_jsonl: Option<PathBuf>,

    /// Generate a synthetic/no-weight observability vertical-slice artifact and exit.
    #[arg(long, value_name = "DIR")]
    pub observability_vertical_slice_out: Option<PathBuf>,

    /// Write product observability profile events to this JSONL path.
    #[arg(long, value_name = "PATH")]
    pub profile_jsonl: Option<PathBuf>,

    /// Product observability detail level.
    #[arg(long, value_enum, default_value_t = crate::observability_product::ProfileDetailArg::Off)]
    pub profile_detail: crate::observability_product::ProfileDetailArg,

    /// Write product memory profile events to this JSONL path.
    #[arg(long, value_name = "PATH")]
    pub memory_profile_jsonl: Option<PathBuf>,

    /// Write scheduler/admission trace events to this JSONL path.
    #[arg(long, value_name = "PATH")]
    pub scheduler_trace_jsonl: Option<PathBuf>,

    /// Write a sanitized request/replay bundle to this directory.
    #[arg(long, value_name = "DIR")]
    pub request_dump_dir: Option<PathBuf>,

    /// Product observability sampling rate for resource lifecycle events.
    #[arg(long, default_value_t = crate::observability_product::default_profile_sample_rate())]
    pub profile_sample_rate: f64,

    /// Output format. `text` (default) — streaming text + stats UX.
    /// `jsonl` — one JSON record per event on stdout; used by tests and scripts.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub output_format: OutputFormat,
}

pub async fn execute(cmd: RunCommand, config: CliConfig) -> Result<()> {
    if let Some(out_dir) = cmd.observability_vertical_slice_out.as_ref() {
        crate::observability_vertical_slice::write_observability_vertical_slice(
            ferrum_types::ProfileEntrypoint::Run,
            out_dir,
        )?;
        println!(
            "OBSERVABILITY VERTICAL SLICE ARTIFACT: {}",
            out_dir.display()
        );
        return Ok(());
    }
    let product_observability = crate::observability_product::ProductObservabilityConfig::new(
        ferrum_types::ProfileEntrypoint::Run,
        &cmd.model,
        cmd.profile_jsonl.as_ref(),
        cmd.profile_detail,
        cmd.memory_profile_jsonl.as_ref(),
        cmd.scheduler_trace_jsonl.as_ref(),
        cmd.request_dump_dir.as_ref(),
        cmd.profile_sample_rate,
    );
    let memory_sampler = crate::memory_profile::ProcessMemorySampler;
    let product_memory_enabled = product_observability.enabled();
    let process_start_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let process_start_memory = process_start_sample
        .clone()
        .map(crate::memory_profile::ProcessMemoryObservation::from_sample);
    if product_observability.synthetic_no_weight_enabled() {
        let written = crate::observability_product::write_synthetic_product_observability(
            &product_observability,
        )?;
        println!(
            "OBSERVABILITY PRODUCT ARTIFACTS: {}",
            written
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        return Ok(());
    }

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
    let mut device = select_device(&cmd.backend)?;
    let mut gpu_selection =
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
    let backend_initialized_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let backend_initialized_memory = process_memory_observation_between(
        process_start_sample.clone(),
        backend_initialized_sample.clone(),
    );
    let mut startup_cli_runtime_entries =
        run_startup_cli_runtime_entries(&cmd, gpu_selection.as_ref());
    materialize_run_cli_runtime_entries(&startup_cli_runtime_entries);
    let autosize = run_autosize_for_device(&device, cmd.gpu_memory_utilization);

    // Resolve the model through the central source resolver. Handles
    // .gguf paths, local model dirs, HF cache hits, and HF download in
    // one entry; runs the chat-profile GPU autosize + (for GGUF) sets
    // the per-arch KV / MoE env-var defaults that `ferrum run` needs
    // for a single-user multi-turn REPL. The engine then picks up
    // either the safetensors path (via NativeSafetensorsLoader) or the
    // GGUF path (via gguf_engine_loader, routed by
    // `WeightFormat::detect()` inside `LlmExecutorFactory`).
    let cache_dir = crate::source_resolver::hf_cache_dir(&config);
    let resolved = crate::source_resolver::resolve_model_source(
        &cmd.model,
        &cache_dir,
        crate::source_resolver::DownloadPolicy::AutoDownload,
        autosize,
    )
    .await?;
    let source = resolved.source;
    let model_id = crate::source_resolver::public_model_id(&source);
    let model_definition_for_config = load_run_model_definition(&source).await?;
    if let (Some(selection), Some(definition)) =
        (gpu_selection.as_mut(), model_definition_for_config.as_ref())
    {
        if selection.apply_model_layer_count(definition.num_hidden_layers)? {
            if let Some(plan) = selection.selected_layer_split_plan.as_deref() {
                eprintln!("{}", format!("CUDA layer split plan: {plan}").dimmed());
            }
            startup_cli_runtime_entries =
                run_startup_cli_runtime_entries(&cmd, gpu_selection.as_ref());
            materialize_run_cli_runtime_entries(&startup_cli_runtime_entries);
        }
    }
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
    if cmd.qwen35_reference {
        engine_config.backend.backend_options.insert(
            "qwen35_reference".to_string(),
            serde_json::Value::Bool(true),
        );
    }
    let runtime_config = RuntimeConfigSnapshot::capture_current();
    if let Some(selection) = &gpu_selection {
        selection.insert_backend_options(&mut engine_config.backend.backend_options);
    }
    crate::layer_split_pipeline::insert_backend_option_from_runtime(
        &runtime_config,
        &mut engine_config.backend.backend_options,
    )?;
    let effective_runtime_config =
        run_effective_runtime_config(&runtime_config, &startup_cli_runtime_entries);
    let startup_auto_config = run_startup_auto_config(
        &device,
        model_definition_for_config.as_ref(),
        crate::commands::serve::model_weight_bytes_from_path(&source.local_path),
        effective_runtime_config,
    )?;
    // Apply the resolved auto-config knobs the same way `serve` does. Without
    // this, `ferrum run` ignored the resolved config (e.g. the CUDA GPTQ-MoE
    // fast path FERRUM_VLLM_MOE / MOE_DEVICE_ROUTE) and fell back to the slow
    // host-route MoE — ~9.7 vs ~59 tok/s on a 4090 for Qwen3-30B-A3B.
    crate::runtime_env::materialize_runtime_env_effective(&startup_auto_config.runtime_config);
    crate::commands::serve::write_startup_config_artifacts(
        &startup_auto_config,
        cmd.effective_config_json.as_deref(),
        cmd.decision_trace_jsonl.as_deref(),
    )?;
    let run_budget = RunBudget::from_source(&source.local_path, &runtime_config);
    engine_config
        .apply_runtime_config_snapshot(&startup_auto_config.runtime_config)
        .map_err(ferrum_types::FerrumError::config)?;
    if runtime_config_bool(&startup_auto_config.runtime_config, "FERRUM_PAGED_KV")
        .or_else(|| {
            runtime_config_bool(&startup_auto_config.runtime_config, "FERRUM_METAL_PAGED_KV")
        })
        .unwrap_or(false)
    {
        engine_config.kv_cache.cache_type = ferrum_types::KvCacheType::Paged;
    }
    let effective_kv_dtype = cmd
        .kv_dtype
        .as_deref()
        .or_else(|| crate::runtime_env::runtime_snapshot_value(&runtime_config, "FERRUM_KV_DTYPE"));
    apply_kv_dtype_override(&mut engine_config, effective_kv_dtype)?;
    let engine = ferrum_engine::create_default_engine(engine_config).await?;
    let model_loaded_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let model_loaded_memory = process_memory_observation_between(
        backend_initialized_sample
            .clone()
            .or_else(|| process_start_sample.clone()),
        model_loaded_sample.clone(),
    );
    let model_loaded_duration_us = load_start
        .elapsed()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX);
    let profile_run_done_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let profile_run_done_memory = process_memory_observation_between(
        model_loaded_sample.clone(),
        profile_run_done_sample.clone(),
    );
    let cache_allocated_status = if product_memory_enabled {
        Some(engine.status().await)
    } else {
        None
    };
    let cache_allocated_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let cache_allocated_memory = process_memory_observation_between(
        profile_run_done_sample
            .clone()
            .or_else(|| model_loaded_sample.clone()),
        cache_allocated_sample.clone(),
    );
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
        let prompt_chars = plan.prompt.chars().count();
        let request = InferenceRequest {
            id: RequestId(Uuid::new_v4()),
            model_id: ferrum_types::ModelId(model_id.clone()),
            prompt: plan.prompt,
            sampling_params: plan.sampling_params.clone(),
            stream: false,
            priority: Priority::Normal,
            client_id: None,
            session_id: None,
            created_at: Utc::now(),
            api_request: None,
            metadata,
        };
        let profile_request_id = request.id.to_string();
        let memory_before = product_observability
            .enabled()
            .then(|| memory_sampler.sample())
            .flatten();
        let start = std::time::Instant::now();
        let response = match engine.infer(request).await {
            Ok(response) => response,
            Err(err) => {
                let memory_after = product_memory_enabled
                    .then(|| memory_sampler.sample())
                    .flatten();
                let memory = process_memory_observation_between(memory_before, memory_after);
                let elapsed = start.elapsed().as_secs_f64();
                if let Err(observability_err) =
                    crate::observability_product::write_actual_run_failure_observability(
                        &product_observability,
                        &crate::observability_product::ActualRunFailureObservation {
                            request_id: profile_request_id,
                            duration_us: (elapsed * 1_000_000.0).max(0.0) as u64,
                            sampling_params: plan.sampling_params.clone(),
                            prompt_token_ids: plan.prompt_token_ids.clone(),
                            prompt_token_count: plan.prompt_tokens,
                            prompt_chars,
                            failure_kind: err.observability_failure_kind().to_string(),
                            error_kind: err.observability_error_kind().to_string(),
                            error_message: err.to_string(),
                            memory,
                            memory_stages: actual_run_memory_stages(
                                product_memory_enabled,
                                process_start_memory.clone(),
                                backend_initialized_memory.clone(),
                                model_loaded_memory.clone(),
                                model_loaded_duration_us,
                                profile_run_done_memory.clone(),
                                cache_allocated_memory.clone(),
                                cache_allocated_status.clone(),
                                None,
                            ),
                        },
                    )
                {
                    eprintln!("failed to write run failure observability: {observability_err}");
                }
                return Err(err);
            }
        };
        let memory_after = product_memory_enabled
            .then(|| memory_sampler.sample())
            .flatten();
        let memory = process_memory_observation_between(memory_before, memory_after.clone());
        let tokens = response.tokens.len();
        let output_token_ids = response
            .tokens
            .iter()
            .map(|token| token.get())
            .collect::<Vec<_>>();
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
        let shutdown_result = engine.shutdown().await;
        let shutdown_after = product_memory_enabled
            .then(|| memory_sampler.sample())
            .flatten();
        let shutdown_memory = process_memory_observation_between(
            memory_after
                .clone()
                .or_else(|| model_loaded_sample.clone())
                .or_else(|| backend_initialized_sample.clone())
                .or_else(|| process_start_sample.clone()),
            shutdown_after,
        );
        crate::observability_product::write_actual_run_observability(
            &product_observability,
            &crate::observability_product::ActualRunObservation {
                request_id: profile_request_id,
                duration_us: (elapsed * 1_000_000.0).max(0.0) as u64,
                sampling_params: plan.sampling_params.clone(),
                prompt_token_ids: plan.prompt_token_ids.clone(),
                prompt_token_count: plan.prompt_tokens,
                output_tokens: tokens,
                output_token_ids,
                chunk_count,
                finish_reason: finish_reason.map(finish_reason_str).map(str::to_string),
                prompt_chars,
                response_chars: content.chars().count(),
                response_text: content.clone(),
                memory,
                memory_stages: actual_run_memory_stages(
                    product_memory_enabled,
                    process_start_memory.clone(),
                    backend_initialized_memory.clone(),
                    model_loaded_memory.clone(),
                    model_loaded_duration_us,
                    profile_run_done_memory.clone(),
                    cache_allocated_memory.clone(),
                    cache_allocated_status.clone(),
                    shutdown_memory,
                ),
            },
        )?;
        shutdown_result?;
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

fn process_memory_observation_between(
    before: Option<crate::memory_profile::ProcessMemorySample>,
    after: Option<crate::memory_profile::ProcessMemorySample>,
) -> Option<crate::memory_profile::ProcessMemoryObservation> {
    after.map(|after| crate::memory_profile::ProcessMemoryObservation::from_samples(before, after))
}

fn actual_run_memory_stages(
    enabled: bool,
    process_start_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    backend_initialized_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    model_loaded_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    model_loaded_duration_us: u64,
    profile_run_done_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    cache_allocated_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    cache_allocated_status: Option<ferrum_types::EngineStatus>,
    shutdown_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
) -> Vec<crate::observability_product::ActualMemoryStageObservation> {
    if !enabled {
        return Vec::new();
    }
    let profile_run_done = crate::observability_product::ActualMemoryStageObservation::new(
        "actual_run_profile_run_done",
        "profile_run_done",
        None,
        profile_run_done_memory,
    )
    .with_profile_run_status(
        false,
        "not_configured",
        "product_basic_profile_does_not_execute_extra_warmup",
    );
    let mut cache_allocated = crate::observability_product::ActualMemoryStageObservation::new(
        "actual_run_cache_allocated",
        "cache_allocated",
        None,
        cache_allocated_memory,
    );
    if let Some(status) = cache_allocated_status.as_ref() {
        cache_allocated = cache_allocated.with_engine_cache_status(status);
    }
    vec![
        crate::observability_product::ActualMemoryStageObservation::new(
            "actual_run_process_start",
            "process_start",
            None,
            process_start_memory,
        ),
        crate::observability_product::ActualMemoryStageObservation::new(
            "actual_run_backend_initialized",
            "backend_initialized",
            None,
            backend_initialized_memory,
        ),
        crate::observability_product::ActualMemoryStageObservation::new(
            "actual_run_model_loaded",
            "model_loaded",
            Some(model_loaded_duration_us),
            model_loaded_memory,
        ),
        profile_run_done,
        cache_allocated,
        crate::observability_product::ActualMemoryStageObservation::new(
            "actual_run_shutdown",
            "shutdown",
            None,
            shutdown_memory,
        ),
    ]
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
    let mut stop_sequences = vec![
        "<|im_end|>".to_string(),
        "</s>".to_string(),
        "<|endoftext|>".to_string(),
    ];
    stop_sequences.extend(cmd.stop.iter().filter(|stop| !stop.is_empty()).cloned());
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
        stop_sequences,
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
        )?;
        let prompt_tokenization = budget.prompt_tokenization(&prompt);
        let prompt_tokens = prompt_tokenization.token_count;
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
            prompt_token_ids: prompt_tokenization.token_ids,
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
        )?;
        let prompt_tokenization = budget.prompt_tokenization(&prompt);
        let prompt_tokens = prompt_tokenization.token_count;

        let Some(kv_capacity) = budget.kv_capacity else {
            return Ok(RunPromptPlan {
                prompt,
                sampling_params: base_sampling,
                prompt_token_ids: prompt_tokenization.token_ids,
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
                prompt_token_ids: prompt_tokenization.token_ids,
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
                prompt_token_ids: prompt_tokenization.token_ids,
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

pub fn select_device(backend: &str) -> Result<ferrum_types::Device> {
    match backend.trim().to_lowercase().as_str() {
        "cpu" => Ok(ferrum_types::Device::CPU),
        "metal" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return Ok(ferrum_types::Device::Metal);
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Err(FerrumError::config(
                    "requested backend 'metal' but this ferrum binary was not built with Metal support; use --backend auto/cpu or build with the metal feature",
                ))
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                return Ok(ferrum_types::Device::CUDA(0));
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(FerrumError::config(
                    "requested backend 'cuda' but this ferrum binary was not built with CUDA support; use --backend auto/cpu or build with the cuda feature",
                ))
            }
        }
        "auto" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return Ok(ferrum_types::Device::Metal);
            }
            #[cfg(feature = "cuda")]
            {
                return Ok(ferrum_types::Device::CUDA(0));
            }
            #[allow(unreachable_code)]
            Ok(ferrum_types::Device::CPU)
        }
        other => Err(FerrumError::config(format!(
            "unknown backend {other:?}; expected one of: auto, cpu, metal, cuda"
        ))),
    }
}

fn build_chat_prompt(
    history: &[(String, String)],
    user_input: &str,
    system: Option<&str>,
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
    chat_template_options: &ChatTemplateOptions,
) -> Result<String> {
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
    cli_runtime_entries: &[RuntimeConfigEntry],
) -> RuntimeConfigSnapshot {
    let mut snapshot = runtime_config.clone();
    for entry in cli_runtime_entries {
        snapshot.upsert_entry(entry.clone());
    }
    snapshot
}

fn run_startup_cli_runtime_entries(
    cmd: &RunCommand,
    gpu_selection: Option<&crate::gpu_devices::GpuDeviceSelection>,
) -> Vec<RuntimeConfigEntry> {
    let mut entries = Vec::new();
    crate::runtime_env::push_cli_runtime_entry(
        &mut entries,
        "FERRUM_KV_DTYPE",
        cmd.kv_dtype.as_deref(),
    );
    crate::runtime_env::push_cli_runtime_usize(&mut entries, "FERRUM_KV_CAPACITY", cmd.kv_capacity);
    crate::runtime_env::push_cli_runtime_usize(
        &mut entries,
        "FERRUM_KV_MAX_BLOCKS",
        cmd.kv_max_blocks,
    );
    crate::runtime_env::push_cli_runtime_usize(
        &mut entries,
        "FERRUM_MAX_MODEL_LEN",
        cmd.max_model_len,
    );
    crate::runtime_env::push_cli_runtime_usize(
        &mut entries,
        "FERRUM_PAGED_MAX_SEQS",
        cmd.max_num_seqs,
    );
    crate::runtime_env::push_cli_runtime_usize(
        &mut entries,
        "FERRUM_MAX_BATCHED_TOKENS",
        cmd.max_num_batched_tokens,
    );
    crate::runtime_env::push_cli_runtime_usize(
        &mut entries,
        "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES",
        cmd.runtime_memory_budget_bytes
            .map(std::num::NonZeroUsize::get),
    );
    if let Some(enabled) = bool_cli_override(cmd.batched_graph, cmd.disable_batched_graph) {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_BATCHED_GRAPH",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(enabled) = bool_cli_override(cmd.unified_graph, cmd.disable_unified_graph) {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_UNIFIED_GRAPH",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(enabled) = bool_cli_override(
        cmd.unified_graph_layers_only,
        cmd.disable_unified_graph_layers_only,
    ) {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(enabled) = bool_cli_override(
        cmd.unified_graph_lm_head_eager,
        cmd.disable_unified_graph_lm_head_eager,
    ) {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    crate::layer_split_pipeline::push_cli_runtime_entry(
        &mut entries,
        cmd.layer_split_pipeline_mode,
    );
    if let Some(path) = &cmd.profile_jsonl {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PROFILE_JSONL",
            path.to_string_lossy().to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(path) = &cmd.scheduler_trace_jsonl {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_SCHEDULER_TRACE_JSONL",
            path.to_string_lossy().to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
    if cmd.profile_jsonl.is_some() || cmd.scheduler_trace_jsonl.is_some() {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PROFILE_ENTRYPOINT",
            "run",
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(selection) = gpu_selection {
        entries.extend(selection.runtime_config_entries());
    }
    entries
}

fn bool_cli_override(enable: bool, disable: bool) -> Option<bool> {
    if enable {
        Some(true)
    } else if disable {
        Some(false)
    } else {
        None
    }
}

fn materialize_run_cli_runtime_entries(entries: &[RuntimeConfigEntry]) {
    if entries.is_empty() {
        return;
    }
    crate::runtime_env::materialize_runtime_env_effective(&RuntimeConfigSnapshot::from_entries(
        entries.to_vec(),
    ));
}

fn run_startup_auto_config(
    device: &ferrum_types::Device,
    model_definition: Option<&ferrum_models::ModelDefinition>,
    model_weight_bytes: Option<u64>,
    runtime_config: RuntimeConfigSnapshot,
) -> Result<ResolvedFerrumConfig> {
    let hardware = crate::commands::serve::hardware_capabilities_for_device(device);
    let model = model_definition
        .map(|definition| {
            crate::commands::serve::model_capabilities_from_definition_with_weight_bytes_for_hardware(
                definition,
                model_weight_bytes,
                &hardware,
            )
        })
        .unwrap_or_else(ModelCapabilities::unknown);
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
    use ferrum_types::RuntimeConfigSource;

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
            stop: Vec::new(),
            no_context_shift: false,
            enable_thinking: false,
            disable_thinking: false,
            temperature: 0.0,
            backend: "auto".to_string(),
            qwen35_reference: false,
            gpu_devices: None,
            layer_split_pipeline_mode: None,
            prompt: None,
            tokenizer: None,
            bench_mode: false,
            top_k: 50,
            top_p: 0.95,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: None,
            gpu_memory_utilization: 0.9,
            runtime_memory_budget_bytes: None,
            max_model_len: None,
            max_num_seqs: None,
            max_num_batched_tokens: None,
            batched_graph: false,
            disable_batched_graph: false,
            unified_graph: false,
            disable_unified_graph: false,
            unified_graph_layers_only: false,
            disable_unified_graph_layers_only: false,
            unified_graph_lm_head_eager: false,
            disable_unified_graph_lm_head_eager: false,
            kv_dtype: None,
            kv_capacity: None,
            kv_max_blocks: None,
            effective_config_json: None,
            decision_trace_jsonl: None,
            observability_vertical_slice_out: None,
            profile_jsonl: None,
            profile_detail: crate::observability_product::ProfileDetailArg::Off,
            memory_profile_jsonl: None,
            scheduler_trace_jsonl: None,
            request_dump_dir: None,
            profile_sample_rate: crate::observability_product::default_profile_sample_rate(),
            output_format: OutputFormat::Text,
        }
    }

    fn whitespace_budget(kv_capacity: usize) -> RunBudget {
        RunBudget {
            tokenizer: None,
            kv_capacity: Some(kv_capacity),
            prompt_token_id_mapper: None,
            prompt_token_counter: Some(|prompt| prompt.split_whitespace().count()),
        }
    }

    fn mapped_token_budget(kv_capacity: usize) -> RunBudget {
        RunBudget {
            tokenizer: None,
            kv_capacity: Some(kv_capacity),
            prompt_token_id_mapper: Some(|prompt| {
                prompt
                    .split_whitespace()
                    .enumerate()
                    .map(|(index, _)| (index + 1) as u32)
                    .collect()
            }),
            prompt_token_counter: None,
        }
    }

    fn default_template_options() -> ChatTemplateOptions {
        ChatTemplateOptions::default()
    }

    #[test]
    fn run_effective_runtime_config_records_cli_kv_dtype() {
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let mut cmd = test_run_cmd();
        cmd.kv_dtype = Some("int8".to_string());
        let cli_entries = run_startup_cli_runtime_entries(&cmd, None);
        let effective = run_effective_runtime_config(&snapshot, &cli_entries);
        let entry = effective
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_KV_DTYPE")
            .expect("missing kv dtype entry");
        assert_eq!(entry.effective_value, "int8");
        assert_eq!(entry.source, RuntimeConfigSource::Cli);
    }

    #[test]
    fn run_effective_runtime_config_records_memory_budget() {
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let mut cmd = test_run_cmd();
        cmd.runtime_memory_budget_bytes = std::num::NonZeroUsize::new(12_345);

        let effective =
            run_effective_runtime_config(&snapshot, &run_startup_cli_runtime_entries(&cmd, None));
        let entry = effective
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES")
            .expect("missing runtime memory budget entry");

        assert_eq!(entry.effective_value, "12345");
        assert_eq!(entry.source, RuntimeConfigSource::Cli);
    }

    #[test]
    fn run_effective_runtime_config_records_observability_paths() {
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let mut cmd = test_run_cmd();
        cmd.profile_jsonl = Some(PathBuf::from("/tmp/run-profile.jsonl"));
        cmd.scheduler_trace_jsonl = Some(PathBuf::from("/tmp/run-scheduler.jsonl"));

        let cli_entries = run_startup_cli_runtime_entries(&cmd, None);
        let effective = run_effective_runtime_config(&snapshot, &cli_entries);
        let entry = |key: &str| {
            effective
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key} entry"))
        };

        assert_eq!(
            entry("FERRUM_PROFILE_JSONL").effective_value,
            "/tmp/run-profile.jsonl"
        );
        assert_eq!(
            entry("FERRUM_SCHEDULER_TRACE_JSONL").effective_value,
            "/tmp/run-scheduler.jsonl"
        );
        assert_eq!(entry("FERRUM_PROFILE_ENTRYPOINT").effective_value, "run");
        assert!(entry("FERRUM_PROFILE_ENTRYPOINT")
            .affects
            .contains(&ferrum_types::RuntimeConfigEffect::Diagnostics));
    }

    #[test]
    fn run_effective_runtime_config_records_layer_split_pipeline_mode() {
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let mut cmd = test_run_cmd();
        cmd.layer_split_pipeline_mode =
            Some(crate::layer_split_pipeline::LayerSplitPipelineModeArg::Batch);
        let cli_entries = run_startup_cli_runtime_entries(&cmd, None);
        let effective = run_effective_runtime_config(&snapshot, &cli_entries);
        let entry = effective
            .entries
            .iter()
            .find(|entry| entry.key == crate::layer_split_pipeline::LAYER_SPLIT_PIPELINE_MODE_KEY)
            .expect("missing layer split pipeline mode entry");
        assert_eq!(entry.effective_value, "batch");
        assert_eq!(entry.source, RuntimeConfigSource::Cli);
    }

    #[test]
    fn run_effective_runtime_config_records_batched_graph_flag() {
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let mut cmd = test_run_cmd();
        cmd.batched_graph = true;
        cmd.unified_graph = true;
        cmd.unified_graph_layers_only = true;
        cmd.unified_graph_lm_head_eager = true;
        let cli_entries = run_startup_cli_runtime_entries(&cmd, None);
        let effective = run_effective_runtime_config(&snapshot, &cli_entries);
        let entry = |key: &str| {
            effective
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key} entry"))
        };
        assert_eq!(entry("FERRUM_BATCHED_GRAPH").effective_value, "1");
        assert_eq!(
            entry("FERRUM_BATCHED_GRAPH").source,
            RuntimeConfigSource::Cli
        );
        assert_eq!(entry("FERRUM_UNIFIED_GRAPH").effective_value, "1");
        assert_eq!(
            entry("FERRUM_UNIFIED_GRAPH").source,
            RuntimeConfigSource::Cli
        );
        assert_eq!(
            entry("FERRUM_UNIFIED_GRAPH_LAYERS_ONLY").effective_value,
            "1"
        );
        assert_eq!(
            entry("FERRUM_UNIFIED_GRAPH_LAYERS_ONLY").source,
            RuntimeConfigSource::Cli
        );
        assert_eq!(
            entry("FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER").effective_value,
            "1"
        );
        assert_eq!(
            entry("FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER").source,
            RuntimeConfigSource::Cli
        );
    }

    #[test]
    fn run_effective_runtime_config_records_gpu_device_selection() {
        let selection = crate::gpu_devices::GpuDeviceSelection {
            raw_cli_value: "1".to_string(),
            requested_gpu_devices: vec![1],
            selected_gpu_devices: vec![1],
            cuda_device_count: 2,
            selected_distributed_strategy: "single_gpu".to_string(),
            selected_layer_split_plan: None,
            selected_layer_split_stages: None,
        };
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let cmd = test_run_cmd();
        let cli_entries = run_startup_cli_runtime_entries(&cmd, Some(&selection));
        let effective = run_effective_runtime_config(&snapshot, &cli_entries);
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
    fn run_effective_runtime_config_records_cli_runtime_limits() {
        let mut cmd = test_run_cmd();
        cmd.kv_capacity = Some(2048);
        cmd.kv_max_blocks = Some(4096);
        cmd.max_model_len = Some(8192);
        cmd.max_num_seqs = Some(8);
        cmd.max_num_batched_tokens = Some(1024);
        let snapshot = RuntimeConfigSnapshot::from_entries(Vec::new());
        let cli_entries = run_startup_cli_runtime_entries(&cmd, None);
        let effective = run_effective_runtime_config(&snapshot, &cli_entries);
        let entry = |key: &str| {
            effective
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_KV_CAPACITY").effective_value, "2048");
        assert_eq!(entry("FERRUM_KV_MAX_BLOCKS").effective_value, "4096");
        assert_eq!(entry("FERRUM_MAX_MODEL_LEN").effective_value, "8192");
        assert_eq!(entry("FERRUM_PAGED_MAX_SEQS").effective_value, "8");
        assert_eq!(entry("FERRUM_MAX_BATCHED_TOKENS").effective_value, "1024");
        assert_eq!(
            entry("FERRUM_MAX_MODEL_LEN").source,
            RuntimeConfigSource::Cli
        );
    }

    #[test]
    fn run_effective_runtime_config_applies_recurrent_state_slots_to_engine_config() {
        let cmd = test_run_cmd();
        let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
            "FERRUM_RECURRENT_STATE_MAX_SLOTS",
            "16",
            RuntimeConfigSource::ConfigFile,
        )]);
        let cli_entries = run_startup_cli_runtime_entries(&cmd, None);
        let effective = run_effective_runtime_config(&snapshot, &cli_entries);
        let mut engine_config = ferrum_types::EngineConfig::default();

        engine_config
            .apply_runtime_config_snapshot(&effective)
            .expect("run effective runtime config should apply");

        assert_eq!(engine_config.runtime.recurrent_state_max_slots, Some(16));
    }

    #[test]
    fn run_startup_auto_config_renders_effective_config_schema() {
        let resolved = run_startup_auto_config(
            &ferrum_types::Device::CPU,
            None,
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
    fn unknown_backend_is_rejected() {
        let err = select_device("not-a-backend").expect_err("unknown backend must fail");
        assert!(
            err.to_string().contains("unknown backend"),
            "unexpected error: {err}"
        );
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    #[test]
    fn explicit_metal_backend_without_compiled_support_is_rejected() {
        let err = select_device("metal").expect_err("unsupported explicit Metal must fail");
        assert!(
            err.to_string()
                .contains("requested backend 'metal' but this ferrum binary was not built"),
            "unexpected error: {err}"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn explicit_cuda_backend_without_compiled_support_is_rejected() {
        let err = select_device("cuda").expect_err("unsupported explicit CUDA must fail");
        assert!(
            err.to_string()
                .contains("requested backend 'cuda' but this ferrum binary was not built"),
            "unexpected error: {err}"
        );
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
                ..Default::default()
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
    fn chat_default_applies_repetition_penalty() {
        // The chat default is greedy (temperature 0). Greedy with NO repetition
        // penalty deterministically locks into token loops on some inputs (the
        // "2D/3D 2D/3D..." degeneration a user hit). The clap default must carry
        // a penalty (OpenAI/llama.cpp standard 1.1) so the out-of-box chat does
        // not loop. Parsing the real CLI (not the struct-literal test fixture)
        // is what pins the actual default users get.
        use clap::Parser;
        #[derive(Parser)]
        struct TestCli {
            #[command(flatten)]
            run: RunCommand,
        }
        let parsed = TestCli::parse_from(["ferrum", "qwen3:0.6b"]);
        assert!(
            parsed.run.repeat_penalty > 1.0,
            "chat default repeat_penalty must discourage repeats, got {}",
            parsed.run.repeat_penalty
        );
        assert!(
            build_sampling_params(&parsed.run).repetition_penalty > 1.0,
            "build_sampling_params must propagate the default penalty"
        );
    }

    #[test]
    fn run_parses_explicit_qwen35_reference_flag() {
        use clap::Parser;

        #[derive(Parser)]
        struct TestCli {
            #[command(flatten)]
            run: RunCommand,
        }

        let parsed = TestCli::parse_from(["ferrum", "qwen3.5", "--qwen35-reference"]);

        assert!(parsed.run.qwen35_reference);
    }

    #[test]
    fn run_sampling_params_include_cli_stop_sequences() {
        let mut cmd = test_run_cmd();
        cmd.stop = vec!["\n".to_string(), String::new(), "END".to_string()];
        let params = build_sampling_params(&cmd);
        assert!(params.stop_sequences.contains(&"\n".to_string()));
        assert!(params.stop_sequences.contains(&"END".to_string()));
        assert!(!params.stop_sequences.contains(&String::new()));
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
    fn run_prompt_plan_retains_prompt_token_ids_for_observability() {
        let cmd = test_run_cmd();
        let budget = mapped_token_budget(64);
        let options = default_template_options();
        let plan = build_run_prompt_plan(
            &[],
            "demo prompt",
            None,
            "tinyllama",
            None,
            &options,
            &cmd,
            &budget,
        )
        .unwrap();

        let token_ids = plan
            .prompt_token_ids
            .as_ref()
            .expect("prompt token ids should be retained");
        assert_eq!(plan.prompt_tokens, Some(token_ids.len()));
        assert!(!token_ids.is_empty());
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

    #[test]
    fn sibling_repo_strips_gguf_suffix_by_default() {
        assert_eq!(
            tokenizer_sibling_repo("Qwen/Qwen3-0.6B-GGUF").as_deref(),
            Some("Qwen/Qwen3-0.6B")
        );
        assert_eq!(tokenizer_sibling_repo("Qwen/Qwen3-0.6B"), None);
    }

    #[test]
    fn sibling_repo_explicit_mappings_beat_strip_convention() {
        // bartowski/* have no safetensors mirrors; stripping `-GGUF`
        // would point at repos that don't exist.
        assert_eq!(
            tokenizer_sibling_repo("bartowski/Qwen2.5-Coder-32B-Instruct-GGUF").as_deref(),
            Some("Qwen/Qwen2.5-Coder-32B-Instruct")
        );
        // mistralai upstream ships tekken-format tokenizers only.
        assert_eq!(
            tokenizer_sibling_repo("bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF")
                .as_deref(),
            Some("unsloth/Mistral-Small-3.2-24B-Instruct-2506")
        );
        // meta-llama upstream is gated; unsloth mirror is not.
        assert_eq!(
            tokenizer_sibling_repo("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF").as_deref(),
            Some("unsloth/Meta-Llama-3.1-8B-Instruct")
        );
    }
}
