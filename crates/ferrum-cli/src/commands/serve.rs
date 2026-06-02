//! Serve command - Start the HTTP inference server

use crate::config::CliConfig;
use crate::runtime_env::runtime_snapshot_value;
use clap::Args;
use colored::*;
use ferrum_bench_core::{ProfileMetadata, ProfileSinkConfig};
use ferrum_models::source::ModelFormat;
use ferrum_server::{AxumServer, HttpServer, ServerConfig};
use ferrum_types::{
    CompiledKernelFeatures, FerrumConfigBuilder, HardwareCapabilities, ModelCapabilities,
    MoeCapabilities, ResolvedFerrumConfig, Result, RuntimeConfigEntry, RuntimeConfigSnapshot,
    RuntimeConfigSource, WorkloadProfile, M3_QWEN3_30B_A3B_INT4_PRESET,
};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use tokio::signal;

#[derive(Args)]
pub struct ServeCommand {
    /// Model to serve (default: from config)
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    /// Model to serve (default: from config)
    #[arg(
        short = 'm',
        long = "model",
        value_name = "MODEL",
        conflicts_with = "model"
    )]
    pub model_option: Option<String>,

    /// Host to bind to
    #[arg(long)]
    pub host: Option<String>,

    /// Port to listen on
    #[arg(short, long)]
    pub port: Option<u16>,

    /// Number of TTS concurrent slots (default: 2)
    #[arg(long, default_value = "2")]
    pub tts_slots: usize,

    /// Speculative decoding: draft model id (same family as target).
    /// Example: `--spec-draft qwen3:0.6b` when serving `qwen3:4b`.
    /// The draft model must share the tokenizer + vocabulary.
    #[arg(long, value_name = "MODEL")]
    pub spec_draft: Option<String>,

    /// Number of speculative tokens per draft forward pass (default: 4).
    /// Only active when --spec-draft is set.
    #[arg(long, default_value = "4")]
    pub spec_tokens: usize,

    /// Fraction of GPU memory ferrum is allowed to use (mirrors vLLM's
    /// `--gpu-memory-utilization`). Auto-sizes the KV pool to fit
    /// weights + scratch + KV inside `total_mem * util`. Default 0.9.
    /// Set 1.0 for an exclusive GPU; lower if you share the card.
    #[arg(long, default_value = "0.9")]
    pub gpu_memory_utilization: f32,

    /// KV cache element dtype (Dim 5 polymorphism point). Accepts
    /// `fp16`, `bf16`, `int8`, `fp8`. Default `fp16`. INT8 / FP8
    /// require model wire-up; today only the kernel + type layer ships.
    /// Override via `FERRUM_KV_DTYPE` env var.
    #[arg(long, value_name = "DTYPE")]
    pub kv_dtype: Option<String>,

    /// Named startup/runtime preset, for example
    /// `m3_qwen3_30b_a3b_int4`.
    #[arg(long, value_name = "PRESET")]
    pub runtime_preset: Option<String>,

    /// Write the startup effective runtime config JSON artifact.
    #[arg(long, value_name = "PATH")]
    pub effective_config_json: Option<PathBuf>,

    /// Write the startup auto-config decision trace JSONL artifact.
    #[arg(long, value_name = "PATH")]
    pub decision_trace_jsonl: Option<PathBuf>,

    /// Write native structured profile events to this JSONL path.
    #[arg(long, value_name = "PATH")]
    pub profile_jsonl: Option<PathBuf>,

    /// Git commit stamped into native structured profile events.
    #[arg(long, value_name = "SHA")]
    pub profile_commit_sha: Option<String>,

    /// Runtime environment hash stamped into native structured profile events.
    #[arg(long, value_name = "SHA256")]
    pub profile_env_hash: Option<String>,

    /// Model label stamped into native structured profile events.
    #[arg(long, value_name = "MODEL")]
    pub profile_model: Option<String>,

    /// Concurrency stamped into native structured profile events.
    #[arg(long, value_name = "N")]
    pub profile_concurrency: Option<u32>,

    /// Runtime flags/config JSON object embedded in native profile events.
    #[arg(long, value_name = "JSON")]
    pub profile_runtime_flags_json: Option<String>,
}

pub async fn execute(cmd: ServeCommand, config: CliConfig) -> Result<()> {
    let ServeCommand {
        model,
        model_option,
        host,
        port,
        tts_slots,
        spec_draft,
        spec_tokens,
        gpu_memory_utilization,
        kv_dtype,
        runtime_preset,
        effective_config_json,
        decision_trace_jsonl,
        profile_jsonl,
        profile_commit_sha,
        profile_env_hash,
        profile_model,
        profile_concurrency,
        profile_runtime_flags_json,
    } = cmd;

    // Print banner
    print_banner();

    // Resolve model
    let model_name = model
        .or(model_option)
        .or(config.models.default_model.clone())
        .unwrap_or_else(|| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    // GGUF short-circuit: if the user passed a `.gguf` file path directly OR
    // an alias resolving to a GGUF (e.g. `qwen3:8b-q4_k_m`), look up the
    // file in the HF cache (or accept the path) and skip
    // `ConfigManager::load_from_path` (which expects an HF safetensors
    // directory). The engine's LlmExecutorFactory +
    // HuggingFaceTokenizerFactory both detect `.gguf` and route to
    // GgufLoader + sibling-tokenizer auto-discovery.
    let cache_dir_for_gguf = get_hf_cache_dir(&config);
    let gguf_path: Option<PathBuf> = if super::run::looks_like_gguf_path(&model_name) {
        Some(PathBuf::from(&model_name))
    } else if let Some((repo, filename)) = super::run::resolve_gguf_alias(&model_name) {
        match super::run::find_cached_gguf(&cache_dir_for_gguf, &repo, &filename) {
            Some(p) => Some(p),
            None => {
                eprintln!(
                    "{} GGUF alias '{}' not in cache. Run: ferrum pull {}",
                    "Error:".red().bold(),
                    model_name,
                    model_name
                );
                return Err(ferrum_types::FerrumError::model("GGUF model not found"));
            }
        }
    } else {
        None
    };

    // Local safetensors directory passthrough: if `--model` is a path to
    // a directory containing `config.json` (the canonical HF safetensors
    // layout), use it directly without going through the HF cache lookup.
    // Lets bench scripts / tooling point at any locally-staged model
    // (e.g. `--local-dir` pulls or symlinked snapshots) without having
    // to mimic the `models--owner--repo/snapshots/<sha>/` cache layout.
    let local_dir_path: Option<PathBuf> = if gguf_path.is_none() {
        let p = PathBuf::from(&model_name);
        if p.is_dir() && p.join("config.json").is_file() {
            Some(p)
        } else {
            None
        }
    } else {
        None
    };

    let config_runtime_entries = config.runtime.runtime_config_entries();
    let configured_runtime_preset = runtime_preset
        .as_deref()
        .map(|preset| (preset, RuntimeConfigSource::Cli))
        .or_else(|| {
            config
                .runtime
                .preset
                .as_deref()
                .map(|preset| (preset, RuntimeConfigSource::ConfigFile))
        });
    let selected_runtime_preset_name =
        configured_runtime_preset.map(|(preset, _source)| preset.to_string());
    let preset_runtime_entries = match configured_runtime_preset {
        Some((preset, source)) => runtime_preset_entries(preset, source)?,
        None => Vec::new(),
    };
    let mut non_env_runtime_entries = preset_runtime_entries;
    non_env_runtime_entries.extend(config_runtime_entries);
    let mut non_env_runtime_entries =
        RuntimeConfigSnapshot::from_entries(non_env_runtime_entries).entries;
    let mut materialized_runtime_keys =
        crate::runtime_env::materialize_runtime_env_defaults(&non_env_runtime_entries);

    let autosize_env_before = RuntimeConfigSnapshot::capture_current();
    // GPU-memory auto-sizing: scale FERRUM_KV_MAX_BLOCKS so weights +
    // KV pool + 4 GB scratch reserve fit in `total_gpu_mem * gpu_util`.
    // Skipped on Mac / when nvidia-smi is missing; respects user-set
    // FERRUM_KV_MAX_BLOCKS overrides.
    if let Some(p) = local_dir_path.as_ref() {
        crate::gpu_mem_autosize::apply_auto_size(p, gpu_memory_utilization);
    }
    let autosize_runtime_entries = runtime_entries_added_by_snapshot(
        &autosize_env_before,
        &RuntimeConfigSnapshot::capture_current(),
        SERVE_AUTOSIZE_RUNTIME_KEYS,
        RuntimeConfigSource::MemoryProfile,
    );
    materialized_runtime_keys.extend(
        autosize_runtime_entries
            .iter()
            .map(|entry| entry.key.clone()),
    );
    non_env_runtime_entries.extend(autosize_runtime_entries);
    non_env_runtime_entries = RuntimeConfigSnapshot::from_entries(non_env_runtime_entries).entries;
    materialized_runtime_keys.sort();
    materialized_runtime_keys.dedup();

    let model_id = if let Some(p) = gguf_path.as_ref() {
        // Use the GGUF stem as the OpenAI model id — the user sees this
        // back in /v1/models responses + chat completion `model` field.
        p.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| model_name.clone())
    } else if let Some(p) = local_dir_path.as_ref() {
        // Local safetensors dir → use the dir name as the public id.
        p.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| model_name.clone())
    } else {
        resolve_model_alias(&model_name)
    };
    println!("{} {}", "Model:".dimmed(), model_id.cyan());

    let host = host.unwrap_or_else(|| config.server.host.clone());
    let port = port.unwrap_or(config.server.port);
    let kv_runtime_snapshot = RuntimeConfigSnapshot::capture_current();
    let env_kv_dtype = runtime_snapshot_value(&kv_runtime_snapshot, "FERRUM_KV_DTYPE");
    let effective_kv_dtype = resolve_effective_kv_dtype(
        kv_dtype.as_deref(),
        env_kv_dtype,
        config.runtime.kv_dtype.as_deref(),
    );

    let source: ferrum_models::source::ResolvedModelSource = if let Some(p) = gguf_path.clone() {
        println!("{} {}", "Path:".dimmed(), p.display());
        ferrum_models::source::ResolvedModelSource {
            original: model_name.clone(),
            local_path: p,
            format: ModelFormat::Unknown, // GGUF — handled by engine
            from_cache: true,
        }
    } else if let Some(p) = local_dir_path.clone() {
        // Local safetensors directory passed via --model.
        println!("{} {}", "Path:".dimmed(), p.display());
        ferrum_models::source::ResolvedModelSource {
            original: model_name.clone(),
            local_path: p,
            format: ModelFormat::SafeTensors,
            from_cache: false,
        }
    } else {
        // Find cached model
        let cache_dir = get_hf_cache_dir(&config);
        match crate::source_resolver::find_cached_model(&cache_dir, &model_id) {
            Some(source) => {
                println!("{} {}", "Path:".dimmed(), source.local_path.display());
                source
            }
            None => {
                eprintln!(
                    "{} Model '{}' not found. Run: ferrum pull {}",
                    "Error:".red().bold(),
                    model_id,
                    model_name
                );
                return Err(ferrum_types::FerrumError::model("Model not found"));
            }
        }
    };

    let engine_model_path = source.local_path.to_string_lossy().to_string();

    // Speculative decoding draft model: resolve the draft path and pass it
    // through EngineConfig backend options. Validates that the draft model is
    // actually cached before the target load kicks in.
    let mut engine_spec_draft_path = None;
    if let Some(ref draft_name) = spec_draft {
        if gguf_path.is_some() {
            return Err(ferrum_types::FerrumError::unsupported(
                "Speculative decoding is not yet wired through the GGUF path",
            ));
        }
        let draft_id = resolve_model_alias(draft_name);
        println!("{} {}", "Draft model:".dimmed(), draft_id.cyan());
        let cache_dir = get_hf_cache_dir(&config);
        let draft_source = crate::source_resolver::find_cached_model(&cache_dir, &draft_id)
            .ok_or_else(|| {
                eprintln!(
                    "{} Draft model '{}' not in HF cache. Run: ferrum pull {}",
                    "Error:".red().bold(),
                    draft_id,
                    draft_name
                );
                ferrum_types::FerrumError::model("Draft model not found")
            })?;
        engine_spec_draft_path = Some(draft_source.local_path.to_string_lossy().to_string());
        println!(
            "{} {} tokens / verify pass",
            "Speculative decoding:".dimmed(),
            spec_tokens
        );
    }

    // Select device
    let device = select_device();
    println!("{} {:?}", "Device:".dimmed(), device);
    let metal_moe_entries = crate::source_resolver::metal_gguf_moe_correctness_entries(
        &source.local_path,
        &device,
        &RuntimeConfigSnapshot::capture_current(),
        RuntimeConfigSource::Default,
    );
    if !metal_moe_entries.is_empty() {
        non_env_runtime_entries.extend(metal_moe_entries.clone());
        non_env_runtime_entries =
            RuntimeConfigSnapshot::from_entries(non_env_runtime_entries).entries;
        materialized_runtime_keys.extend(crate::runtime_env::materialize_runtime_env_defaults(
            &metal_moe_entries,
        ));
        materialized_runtime_keys.sort();
        materialized_runtime_keys.dedup();
    }

    // Detect architecture to choose engine type. For GGUF we skip
    // ConfigManager::load_from_path (which expects HF safetensors layout)
    // and route directly to the continuous-batching LLM engine — the
    // engine's LlmExecutorFactory uses WeightFormat::detect() to route GGUF.
    println!();
    let model_definition: Option<ferrum_models::ModelDefinition> = if gguf_path.is_some() {
        None
    } else {
        let mut config_manager = ferrum_models::ConfigManager::new();
        Some(config_manager.load_from_path(&source.local_path).await?)
    };
    let arch_for_dispatch = model_definition
        .as_ref()
        .map(|model_def| model_def.architecture);
    let mut selected_runtime_preset_name = selected_runtime_preset_name;
    if selected_runtime_preset_name.is_none()
        && is_m3_qwen3_30b_a3b(arch_for_dispatch, model_definition.as_ref())
    {
        selected_runtime_preset_name = Some(M3_QWEN3_30B_A3B_INT4_PRESET.to_string());
        let mut inferred_entries =
            runtime_preset_entries(M3_QWEN3_30B_A3B_INT4_PRESET, RuntimeConfigSource::Default)?;
        inferred_entries.extend(non_env_runtime_entries);
        non_env_runtime_entries = RuntimeConfigSnapshot::from_entries(inferred_entries).entries;
        materialized_runtime_keys.extend(crate::runtime_env::materialize_runtime_env_defaults(
            &non_env_runtime_entries,
        ));
        materialized_runtime_keys.sort();
        materialized_runtime_keys.dedup();
    }

    // Preserve the historical non-preset serve default for Qwen3-MoE, but
    // route it through the typed startup snapshot instead of a hidden
    // process-wide env mutation. M3 explicit or model-inferred presets have
    // already materialized the same graph-clean defaults above.
    if selected_runtime_preset_name.is_none()
        && arch_for_dispatch == Some(ferrum_models::Architecture::Qwen3Moe)
    {
        let current_runtime = merge_runtime_config_sources(
            non_env_runtime_entries.clone(),
            RuntimeConfigSnapshot::capture_current(),
            Vec::new(),
        );
        let mut legacy_entries = crate::runtime_env::moe_graph_default_entries(
            &current_runtime,
            RuntimeConfigSource::Default,
        );
        legacy_entries.extend(non_env_runtime_entries);
        non_env_runtime_entries = RuntimeConfigSnapshot::from_entries(legacy_entries).entries;
        materialized_runtime_keys.extend(crate::runtime_env::materialize_runtime_env_defaults(
            &non_env_runtime_entries,
        ));
        materialized_runtime_keys.sort();
        materialized_runtime_keys.dedup();
    }

    let startup_cli_runtime_entries = serve_cli_runtime_entries(
        kv_dtype.as_deref(),
        profile_jsonl.as_ref(),
        profile_commit_sha.as_deref(),
        profile_env_hash.as_deref(),
        profile_model.as_deref(),
        profile_concurrency,
        profile_runtime_flags_json.as_deref(),
    );
    let startup_auto_config = startup_auto_config(
        &device,
        arch_for_dispatch,
        model_definition.as_ref(),
        selected_runtime_preset_name.as_deref(),
        non_env_runtime_entries,
        materialized_runtime_keys,
        startup_cli_runtime_entries,
    )?;
    write_startup_config_artifacts(
        &startup_auto_config,
        effective_config_json.as_deref(),
        decision_trace_jsonl.as_deref(),
    )?;
    configure_profile_sink(
        profile_jsonl,
        ProfileSinkCliFields {
            commit_sha: profile_commit_sha,
            env_hash: profile_env_hash,
            model: profile_model,
            concurrency: profile_concurrency,
            runtime_flags_json: profile_runtime_flags_json,
        },
        &startup_auto_config,
        &model_id,
    )?;

    let server = match arch_for_dispatch {
        Some(ferrum_models::Architecture::Clip) => {
            println!("{}", "Initializing CLIP embedding engine...".dimmed());
            let candle_device = candle_core::Device::Cpu;
            let executor = ferrum_models::ClipModelExecutor::from_path(
                &source.local_path.to_string_lossy(),
                candle_device,
                candle_core::DType::F32,
            )?;
            let tokenizer = crate::commands::embed::load_tokenizer(&source.local_path)?;
            let mut engine_config = ferrum_types::EngineConfig::default();
            engine_config.model.model_id = ferrum_types::ModelId::new(model_id.clone());
            engine_config.backend.device = device;
            let engine: Arc<dyn ferrum_engine::EmbedEngine + Send + Sync> = Arc::new(
                ferrum_engine::embedding_engine::EmbeddingEngine::new(executor, engine_config)
                    .with_tokenizer(tokenizer),
            );
            AxumServer::from_embed(engine)
        }
        Some(ferrum_models::Architecture::Whisper) => {
            println!("{}", "Initializing Whisper ASR engine...".dimmed());
            let candle_device = to_candle_device(&device);
            let executor = ferrum_models::WhisperModelExecutor::from_path(
                &source.local_path.to_string_lossy(),
                candle_device,
                candle_core::DType::F32,
            )?;
            let mut engine_config = ferrum_types::EngineConfig::default();
            engine_config.model.model_id = ferrum_types::ModelId::new(model_id.clone());
            engine_config.backend.device = device;
            let engine: Arc<dyn ferrum_engine::TranscribeEngine + Send + Sync> = Arc::new(
                ferrum_engine::transcription_engine::TranscriptionEngine::new(
                    executor,
                    engine_config,
                ),
            );
            AxumServer::from_transcribe(engine)
        }
        Some(ferrum_models::Architecture::Qwen3TTS) => {
            let n_slots = tts_slots.max(1);
            println!(
                "{} ({} slot{})",
                "Initializing Qwen3-TTS engine...".dimmed(),
                n_slots,
                if n_slots > 1 { "s" } else { "" }
            );
            let model_path = source.local_path.to_string_lossy().to_string();
            let mut executors = Vec::with_capacity(n_slots);
            for i in 0..n_slots {
                let candle_device = to_candle_device(&device);
                let executor = ferrum_models::TtsModelExecutor::from_path(
                    &model_path,
                    candle_device,
                    candle_core::DType::F32,
                )?;
                if i == 0 {
                    println!("  Slot 0 loaded");
                } else {
                    println!("  Slot {} loaded", i);
                }
                executors.push(executor);
            }
            let engine: Arc<dyn ferrum_engine::TtsEngine + Send + Sync> =
                Arc::new(ferrum_engine::tts_engine::TtsService::new_multi(
                    executors,
                    ferrum_types::ModelId(model_id.clone()),
                ));
            AxumServer::from_tts(engine)
        }
        _ => {
            println!(
                "{}",
                "Initializing engine (continuous batching)...".dimmed()
            );
            let mut engine_config = ferrum_types::EngineConfig::default();
            engine_config.model.model_id = ferrum_types::ModelId::new(model_id.clone());
            engine_config.backend.device = device;
            engine_config.scheduler.policy = ferrum_types::SchedulingPolicy::ContinuousBatch;
            engine_config
                .apply_runtime_config_snapshot(&startup_auto_config.runtime_config)
                .map_err(ferrum_types::FerrumError::config)?;
            engine_config.kv_cache.cache_type = ferrum_types::KvCacheType::Paged;
            engine_config.backend.backend_options.insert(
                "model_path".to_string(),
                serde_json::Value::String(engine_model_path.clone()),
            );
            if let Some(draft_path) = engine_spec_draft_path.as_ref() {
                engine_config.backend.backend_options.insert(
                    "spec_draft".to_string(),
                    serde_json::Value::String(draft_path.clone()),
                );
                engine_config.backend.backend_options.insert(
                    "spec_n".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(spec_tokens)),
                );
            }
            super::run::apply_kv_dtype_override(&mut engine_config, effective_kv_dtype)?;
            let engine: Arc<dyn ferrum_engine::LlmInferenceEngine + Send + Sync> =
                Arc::from(ferrum_engine::create_default_engine(engine_config).await?);
            AxumServer::from_llm(engine)
        }
    }
    .with_auto_config(startup_auto_config);

    // Create server config
    let server_config = ServerConfig {
        host: host.clone(),
        port,
        ..Default::default()
    };

    println!();
    println!(
        "{} {} {}",
        "🚀".green(),
        "Server running at".green().bold(),
        format!("http://{}:{}", host, port).cyan().bold()
    );
    println!();
    println!("Endpoints:");
    println!("  POST /v1/chat/completions      - OpenAI-compatible chat");
    println!("  POST /v1/audio/transcriptions  - Speech-to-text (Whisper)");
    println!("  POST /v1/audio/speech          - Text-to-speech (TTS)");
    println!("  POST /v1/embeddings            - Text/image embeddings");
    println!("  GET  /v1/models                - List models");
    println!("  GET  /health                   - Health check");
    println!();
    println!("{}", "Press Ctrl+C to stop.".dimmed());
    println!();

    // Write PID file for stop command
    let pid_file = std::env::temp_dir().join("ferrum.pid");
    std::fs::write(&pid_file, std::process::id().to_string()).ok();

    // Start server with graceful shutdown
    tokio::select! {
        result = server.start(&server_config) => {
            if let Err(e) = result {
                eprintln!("{} Server error: {}", "Error:".red().bold(), e);
            }
        }
        _ = signal::ctrl_c() => {
            println!();
            println!("{}", "Shutting down...".yellow());
        }
    }

    // Clean up PID file
    std::fs::remove_file(&pid_file).ok();

    // PLAYBOOK § 1.5: Rust statics don't drop on exit, so the global
    // TraceWriter's buffered events would be lost. Force-flush on
    // ctrl_c-driven shutdown (matches bench / bench-serve exit paths).
    ferrum_bench_core::trace::flush_global_trace();
    ferrum_bench_core::flush_global_profile();

    Ok(())
}

fn print_banner() {
    println!();
    println!("{}", "  ______                            ".bright_red());
    println!("{}", " |  ____|                           ".bright_red());
    println!("{}", " | |__ ___ _ __ _ __ _   _ _ __ ___  ".bright_red());
    println!("{}", " |  __/ _ \\ '__| '__| | | | '_ ` _ \\ ".bright_red());
    println!("{}", " | | |  __/ |  | |  | |_| | | | | | ".bright_red());
    println!("{}", " |_|  \\___|_|  |_|   \\__,_|_| |_| |_|".bright_red());
    println!();
    println!("   {}", "🦀 Rust LLM Inference Server".bright_cyan().bold());
    println!(
        "   {}",
        format!("Version {}", env!("CARGO_PKG_VERSION")).dimmed()
    );
    println!();
}

fn resolve_model_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "tinyllama" | "tiny" => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        "qwen2.5:0.5b" | "qwen:0.5b" => "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        "qwen2.5:1.5b" | "qwen:1.5b" => "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        "qwen2.5:3b" | "qwen:3b" => "Qwen/Qwen2.5-3B-Instruct".to_string(),
        "qwen2.5:7b" | "qwen:7b" => "Qwen/Qwen2.5-7B-Instruct".to_string(),
        "qwen3:0.6b" => "Qwen/Qwen3-0.6B".to_string(),
        "qwen3:1.7b" => "Qwen/Qwen3-1.7B".to_string(),
        "qwen3:4b" => "Qwen/Qwen3-4B".to_string(),
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
        "qwen3-tts" | "tts" | "tts:0.6b" => "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
        "tts:1.7b" | "qwen3-tts:1.7b" => "Qwen/Qwen3-TTS-12Hz-1.7B-Base".to_string(),
        _ => name.to_string(),
    }
}

fn get_hf_cache_dir(config: &CliConfig) -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }
    let configured = shellexpand::tilde(&config.models.download.hf_cache_dir).to_string();
    PathBuf::from(configured)
}

fn startup_auto_config(
    device: &ferrum_types::Device,
    architecture: Option<ferrum_models::Architecture>,
    model_definition: Option<&ferrum_models::ModelDefinition>,
    runtime_preset: Option<&str>,
    non_env_runtime_entries: Vec<RuntimeConfigEntry>,
    materialized_runtime_keys: Vec<String>,
    cli_runtime_entries: Vec<RuntimeConfigEntry>,
) -> Result<ResolvedFerrumConfig> {
    let mut env_snapshot = RuntimeConfigSnapshot::capture_current();
    env_snapshot = remove_materialized_config_env_entries(env_snapshot, &materialized_runtime_keys);
    let runtime_config =
        merge_runtime_config_sources(non_env_runtime_entries, env_snapshot, cli_runtime_entries);
    let model = model_definition
        .map(model_capabilities_from_definition)
        .unwrap_or_else(ModelCapabilities::unknown);
    let hardware = hardware_capabilities_for_device(device);
    let workload = match runtime_preset {
        Some(M3_QWEN3_30B_A3B_INT4_PRESET) => WorkloadProfile::m3_qwen3_30b_a3b_int4(),
        Some(other) => {
            return Err(ferrum_types::FerrumError::config(format!(
                "unknown runtime preset: {other}"
            )));
        }
        None if is_m3_qwen3_30b_a3b(architecture, model_definition) => {
            WorkloadProfile::m3_qwen3_30b_a3b_int4()
        }
        None => WorkloadProfile::serving_default(),
    };

    FerrumConfigBuilder::new(runtime_config)
        .with_model_capabilities(model)
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .map_err(|err| ferrum_types::FerrumError::config(format!("invalid auto config: {err}")))
}

fn merge_runtime_config_sources(
    config_file_entries: Vec<RuntimeConfigEntry>,
    env_snapshot: RuntimeConfigSnapshot,
    cli_entries: Vec<RuntimeConfigEntry>,
) -> RuntimeConfigSnapshot {
    let mut runtime_config = RuntimeConfigSnapshot::from_entries(config_file_entries);
    for entry in env_snapshot.entries {
        runtime_config.upsert_entry(entry);
    }
    for entry in cli_entries {
        runtime_config.upsert_entry(entry);
    }
    runtime_config
}

fn remove_materialized_config_env_entries(
    mut env_snapshot: RuntimeConfigSnapshot,
    materialized_config_runtime_keys: &[String],
) -> RuntimeConfigSnapshot {
    env_snapshot
        .entries
        .retain(|entry| !materialized_config_runtime_keys.contains(&entry.key));
    env_snapshot
}

const SERVE_AUTOSIZE_RUNTIME_KEYS: &[&str] = &[
    "FERRUM_MAX_BATCHED_TOKENS",
    "FERRUM_KV_MAX_BLOCKS",
    "FERRUM_PAGED_MAX_SEQS",
    "FERRUM_KV_CAPACITY",
];

fn runtime_entries_added_by_snapshot(
    before: &RuntimeConfigSnapshot,
    after: &RuntimeConfigSnapshot,
    keys: &[&str],
    source: RuntimeConfigSource,
) -> Vec<RuntimeConfigEntry> {
    keys.iter()
        .filter_map(|key| {
            let before_value = runtime_snapshot_value(before, key);
            let after_value = runtime_snapshot_value(after, key);
            match (before_value, after_value) {
                (None, Some(value)) => Some(RuntimeConfigEntry::new(*key, value, source)),
                _ => None,
            }
        })
        .collect()
}

fn runtime_preset_entries(
    preset: &str,
    source: RuntimeConfigSource,
) -> Result<Vec<RuntimeConfigEntry>> {
    let pairs: &[(&str, &str)] = match preset {
        M3_QWEN3_30B_A3B_INT4_PRESET => &[
            ("FERRUM_BACKEND", "cuda"),
            ("FERRUM_MOE_DEVICE_ROUTE", "1"),
            ("FERRUM_MOE_STREAMS", "4"),
            ("FERRUM_GREEDY_ARGMAX", "1"),
            ("FERRUM_KV_MAX_BLOCKS", "2048"),
            ("FERRUM_PAGED_MAX_SEQS", "32"),
            ("FERRUM_MOE_GRAPH", "1"),
            ("FERRUM_VLLM_MOE", "1"),
            ("FERRUM_VLLM_MOE_PAIR_IDS", "1"),
            ("FERRUM_USE_VLLM_PAGED_ATTN", "1"),
            ("FERRUM_PREFIX_CACHE", "0"),
        ],
        other => {
            return Err(ferrum_types::FerrumError::config(format!(
                "unknown runtime preset: {other}"
            )));
        }
    };
    Ok(pairs
        .iter()
        .map(|(key, value)| RuntimeConfigEntry::new(*key, *value, source))
        .collect())
}

fn serve_cli_runtime_entries(
    kv_dtype: Option<&str>,
    profile_jsonl: Option<&PathBuf>,
    profile_commit_sha: Option<&str>,
    profile_env_hash: Option<&str>,
    profile_model: Option<&str>,
    profile_concurrency: Option<u32>,
    profile_runtime_flags_json: Option<&str>,
) -> Vec<RuntimeConfigEntry> {
    let mut entries = Vec::new();
    push_cli_runtime_entry(&mut entries, "FERRUM_KV_DTYPE", kv_dtype);
    if let Some(path) = profile_jsonl {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PROFILE_JSONL",
            path.to_string_lossy().to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
    push_cli_runtime_entry(
        &mut entries,
        "FERRUM_PROFILE_COMMIT_SHA",
        profile_commit_sha,
    );
    push_cli_runtime_entry(&mut entries, "FERRUM_PROFILE_ENV_HASH", profile_env_hash);
    push_cli_runtime_entry(&mut entries, "FERRUM_PROFILE_MODEL", profile_model);
    if let Some(concurrency) = profile_concurrency {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PROFILE_CONCURRENCY",
            concurrency.to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
    push_cli_runtime_entry(
        &mut entries,
        "FERRUM_PROFILE_RUNTIME_FLAGS_JSON",
        profile_runtime_flags_json,
    );
    entries
}

fn resolve_effective_kv_dtype<'a>(
    cli_arg: Option<&'a str>,
    env_value: Option<&'a str>,
    config_file_value: Option<&'a str>,
) -> Option<&'a str> {
    cli_arg.or(env_value).or(config_file_value)
}

fn push_cli_runtime_entry(entries: &mut Vec<RuntimeConfigEntry>, key: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        entries.push(RuntimeConfigEntry::new(
            key,
            value.to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
}

fn write_startup_config_artifacts(
    auto_config: &ResolvedFerrumConfig,
    effective_config_json: Option<&std::path::Path>,
    decision_trace_jsonl: Option<&std::path::Path>,
) -> Result<()> {
    if let Some(path) = effective_config_json {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| ferrum_types::FerrumError::io(err.to_string()))?;
        }
        let bytes = serde_json::to_vec_pretty(&auto_config.effective_config_document())
            .map_err(|err| ferrum_types::FerrumError::serialization(err.to_string()))?;
        std::fs::write(path, [bytes.as_slice(), b"\n"].concat())
            .map_err(|err| ferrum_types::FerrumError::io(err.to_string()))?;
    }
    if let Some(path) = decision_trace_jsonl {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| ferrum_types::FerrumError::io(err.to_string()))?;
        }
        let trace = auto_config
            .decision_trace_jsonl()
            .map_err(|err| ferrum_types::FerrumError::serialization(err.to_string()))?;
        std::fs::write(path, trace)
            .map_err(|err| ferrum_types::FerrumError::io(err.to_string()))?;
    }
    Ok(())
}

struct ProfileSinkCliFields {
    commit_sha: Option<String>,
    env_hash: Option<String>,
    model: Option<String>,
    concurrency: Option<u32>,
    runtime_flags_json: Option<String>,
}

fn configure_profile_sink(
    profile_jsonl: Option<PathBuf>,
    fields: ProfileSinkCliFields,
    auto_config: &ResolvedFerrumConfig,
    model_id: &str,
) -> Result<()> {
    let Some(path) = profile_jsonl else {
        return Ok(());
    };

    let runtime_flags = match fields.runtime_flags_json {
        Some(json) => {
            let value = serde_json::from_str::<serde_json::Value>(&json).map_err(|err| {
                ferrum_types::FerrumError::config(format!(
                    "invalid --profile-runtime-flags-json: {err}"
                ))
            })?;
            if !value.is_object() {
                return Err(ferrum_types::FerrumError::config(
                    "--profile-runtime-flags-json must be a JSON object",
                ));
            }
            value
        }
        None => auto_config.effective_config_document(),
    };

    let env_hash = match fields.env_hash {
        Some(value) if value.starts_with("sha256:") => value,
        Some(_) => {
            return Err(ferrum_types::FerrumError::config(
                "--profile-env-hash must start with sha256:",
            ))
        }
        None => auto_config.runtime_env_hash(),
    };

    let metadata = ProfileMetadata {
        commit_sha: fields.commit_sha.filter(|value| !value.trim().is_empty()),
        env_hash,
        model: fields
            .model
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| model_id.to_string()),
        concurrency: fields
            .concurrency
            .filter(|value| *value > 0)
            .unwrap_or_else(|| auto_config.workload_profile.target_concurrency.max(1) as u32),
        runtime_flags,
    };
    let profile_config = ProfileSinkConfig::enabled(path, metadata);
    ferrum_bench_core::configure_global_profile(profile_config.clone())
        .map_err(|err| ferrum_types::FerrumError::io(err.to_string()))?;
    ferrum_kernels::configure_native_profile_sink(&profile_config)
        .map_err(|err| ferrum_types::FerrumError::io(err.to_string()))?;
    Ok(())
}

fn model_capabilities_from_definition(
    definition: &ferrum_models::ModelDefinition,
) -> ModelCapabilities {
    let architecture = match definition.architecture {
        ferrum_models::Architecture::Qwen3Moe => "qwen3_moe",
        ferrum_models::Architecture::Qwen3 => "qwen3",
        ferrum_models::Architecture::Qwen2 => "qwen2",
        ferrum_models::Architecture::Llama => "llama",
        ferrum_models::Architecture::Mistral => "mistral",
        ferrum_models::Architecture::Phi => "phi",
        ferrum_models::Architecture::GPT2 => "gpt2",
        ferrum_models::Architecture::Bert => "bert",
        ferrum_models::Architecture::Clip => "clip",
        ferrum_models::Architecture::Whisper => "whisper",
        ferrum_models::Architecture::Qwen3TTS => "qwen3_tts",
        ferrum_models::Architecture::Unknown => "unknown",
    }
    .to_string();
    let head_dim = definition
        .extra_params
        .get("head_dim")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .or_else(|| {
            (definition.num_attention_heads > 0)
                .then_some(definition.hidden_size / definition.num_attention_heads)
        });
    let moe = if definition.architecture == ferrum_models::Architecture::Qwen3Moe {
        let num_experts = definition
            .extra_params
            .get("num_experts")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize;
        let experts_per_token = definition
            .extra_params
            .get("num_experts_per_tok")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize;
        Some(MoeCapabilities {
            num_experts,
            experts_per_token,
            moe_intermediate_size: definition
                .extra_params
                .get("moe_intermediate_size")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize),
        })
    } else {
        None
    };

    ModelCapabilities {
        architecture,
        quantization: quantization_from_definition(definition),
        moe,
        max_context_len: Some(definition.max_position_embeddings),
        num_hidden_layers: Some(definition.num_hidden_layers),
        head_dim,
        kv_heads: definition.num_key_value_heads,
        estimated_weight_bytes: estimated_weight_bytes_from_definition(definition),
        supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
        graph_safe_moe: definition.architecture == ferrum_models::Architecture::Qwen3Moe,
    }
}

fn quantization_from_definition(definition: &ferrum_models::ModelDefinition) -> Option<String> {
    let quant = definition.extra_params.get("quantization_config")?;
    let method = quant
        .get("quant_method")
        .or_else(|| quant.get("type"))
        .and_then(|value| value.as_str())
        .unwrap_or("quantized");
    let bits = quant.get("bits").and_then(|value| value.as_u64());
    Some(match bits {
        Some(bits) => format!("{method}_int{bits}"),
        None => method.to_string(),
    })
}

fn estimated_weight_bytes_from_definition(
    definition: &ferrum_models::ModelDefinition,
) -> Option<u64> {
    let params = definition.to_model_info("__auto_config").num_parameters;
    if params == 0 {
        return None;
    }
    let quant = definition.extra_params.get("quantization_config");
    let bits_per_param = quant
        .and_then(|quant| quant.get("bits"))
        .and_then(|value| value.as_u64())
        .filter(|bits| *bits > 0)
        .unwrap_or(16);
    Some(params.saturating_mul(bits_per_param).div_ceil(8))
}

fn hardware_capabilities_for_device(device: &ferrum_types::Device) -> HardwareCapabilities {
    let features = compiled_kernel_features();
    match device {
        ferrum_types::Device::CUDA(id) => {
            cuda_hardware_capabilities(features, probe_cuda_device(*id as usize))
        }
        ferrum_types::Device::ROCm(_) => HardwareCapabilities {
            backend: "rocm".to_string(),
            supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
            supported_kv_dtypes: vec!["fp16".to_string()],
            compiled_features: features,
            ..HardwareCapabilities::unknown()
        },
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ferrum_types::Device::Metal => HardwareCapabilities {
            backend: "metal".to_string(),
            supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
            supported_kv_dtypes: vec!["fp16".to_string()],
            compiled_features: features,
            ..HardwareCapabilities::unknown()
        },
        ferrum_types::Device::CPU => HardwareCapabilities {
            backend: "cpu".to_string(),
            supported_dtypes: vec!["fp32".to_string()],
            supported_kv_dtypes: vec!["fp16".to_string()],
            compiled_features: features,
            ..HardwareCapabilities::unknown()
        },
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CudaDeviceProbe {
    name: Option<String>,
    cuda_runtime: Option<String>,
    compute_capability: Option<String>,
    vram_bytes: Option<u64>,
    sm_count: Option<u32>,
}

fn cuda_hardware_capabilities(
    features: CompiledKernelFeatures,
    probe: CudaDeviceProbe,
) -> HardwareCapabilities {
    HardwareCapabilities {
        backend: "cuda".to_string(),
        cuda_runtime: probe.cuda_runtime,
        compute_capability: probe.compute_capability,
        vram_bytes: probe.vram_bytes,
        sm_count: probe.sm_count,
        supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
        supported_kv_dtypes: vec![
            "fp16".to_string(),
            "bf16".to_string(),
            "int8".to_string(),
            "fp8".to_string(),
        ],
        graph_support: cfg!(feature = "cuda"),
        compiled_features: features,
    }
}

fn probe_cuda_device(device_id: usize) -> CudaDeviceProbe {
    let mut probe = run_nvidia_smi_query(device_id, "name,compute_cap,memory.total")
        .and_then(|output| parse_nvidia_smi_gpu_query(&output))
        .unwrap_or_default();
    probe.cuda_runtime = probe_cuda_runtime_version();
    probe.sm_count = run_nvidia_smi_query(device_id, "multiprocessor_count")
        .and_then(|output| parse_first_u32(&output))
        .or_else(|| probe.name.as_deref().and_then(infer_sm_count_from_gpu_name));
    probe
}

fn run_nvidia_smi_query(device_id: usize, query: &str) -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args([
            format!("--query-gpu={query}"),
            "--format=csv,noheader,nounits".to_string(),
            "-i".to_string(),
            device_id.to_string(),
        ])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).to_string())
}

fn probe_cuda_runtime_version() -> Option<String> {
    run_command_stdout("nvcc", &["--version"])
        .and_then(|output| parse_nvcc_cuda_release(&output))
        .or_else(|| {
            run_command_stdout("nvidia-smi", &[])
                .and_then(|output| parse_nvidia_smi_cuda_version(&output))
        })
}

fn run_command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).to_string())
}

fn parse_nvidia_smi_gpu_query(output: &str) -> Option<CudaDeviceProbe> {
    let line = output.lines().find(|line| !line.trim().is_empty())?;
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() < 3 {
        return None;
    }
    let name = non_empty_probe_value(fields[0]).map(str::to_string);
    let compute_capability = non_empty_probe_value(fields[1]).map(str::to_string);
    let vram_bytes = parse_memory_total_bytes(fields[2]);
    Some(CudaDeviceProbe {
        name,
        compute_capability,
        vram_bytes,
        ..CudaDeviceProbe::default()
    })
}

fn parse_memory_total_bytes(raw: &str) -> Option<u64> {
    let lower = raw.trim().to_ascii_lowercase();
    let numeric = lower
        .trim_end_matches("mib")
        .trim_end_matches("mb")
        .trim_end_matches("gib")
        .trim_end_matches("gb")
        .trim();
    let value = numeric.parse::<f64>().ok()?;
    let multiplier = if lower.contains("gib") || lower.contains("gb") {
        1024.0 * 1024.0 * 1024.0
    } else {
        1024.0 * 1024.0
    };
    Some((value * multiplier).round() as u64)
}

fn parse_first_u32(output: &str) -> Option<u32> {
    output
        .lines()
        .find_map(|line| non_empty_probe_value(line)?.parse::<u32>().ok())
}

fn parse_nvcc_cuda_release(output: &str) -> Option<String> {
    let marker = "release ";
    let start = output.find(marker)? + marker.len();
    parse_version_prefix(&output[start..])
}

fn parse_nvidia_smi_cuda_version(output: &str) -> Option<String> {
    let marker = "CUDA Version:";
    let start = output.find(marker)? + marker.len();
    parse_version_prefix(output[start..].trim())
}

fn parse_version_prefix(raw: &str) -> Option<String> {
    let version = raw
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
        .collect::<String>();
    (!version.is_empty()).then_some(version)
}

fn non_empty_probe_value(raw: &str) -> Option<&str> {
    let value = raw.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("n/a") {
        None
    } else {
        Some(value)
    }
}

fn infer_sm_count_from_gpu_name(name: &str) -> Option<u32> {
    let normalized = name.to_ascii_lowercase();
    if normalized.contains("rtx 4090") {
        Some(128)
    } else {
        None
    }
}

fn compiled_kernel_features() -> CompiledKernelFeatures {
    CompiledKernelFeatures {
        cuda: cfg!(feature = "cuda"),
        vllm_paged_attn: cfg!(feature = "vllm-paged-attn-v2"),
        vllm_moe_marlin: cfg!(feature = "vllm-moe-marlin"),
        cuda_graph: cfg!(feature = "cuda"),
        greedy_argmax: cfg!(feature = "cuda") || cfg!(feature = "metal"),
        fa2_source: cfg!(feature = "fa2-source"),
        fa2_direct_ffi: cfg!(feature = "cuda"),
    }
}

fn is_m3_qwen3_30b_a3b(
    architecture: Option<ferrum_models::Architecture>,
    model_definition: Option<&ferrum_models::ModelDefinition>,
) -> bool {
    if architecture != Some(ferrum_models::Architecture::Qwen3Moe) {
        return false;
    }
    let Some(definition) = model_definition else {
        return false;
    };
    let num_experts = definition
        .extra_params
        .get("num_experts")
        .and_then(|value| value.as_u64());
    let experts_per_token = definition
        .extra_params
        .get("num_experts_per_tok")
        .and_then(|value| value.as_u64());
    definition.hidden_size == 2048
        && definition.num_hidden_layers >= 40
        && definition.num_key_value_heads == Some(4)
        && num_experts == Some(128)
        && experts_per_token == Some(8)
}

// `find_cached_model` and `detect_format` previously lived here as forks
// of the `run.rs` versions. They moved to `crate::source_resolver` so the
// HF cache walk + format detection have a single source of truth across
// `run` / `serve` / `bench`. Use `crate::source_resolver::find_cached_model`
// / `crate::source_resolver::detect_format` directly.

fn select_device() -> ferrum_types::Device {
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

fn to_candle_device(device: &ferrum_types::Device) -> candle_core::Device {
    match device {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        ferrum_types::Device::Metal => {
            candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
        }
        #[cfg(feature = "cuda")]
        ferrum_types::Device::CUDA(id) => {
            candle_core::Device::new_cuda(*id as usize).unwrap_or(candle_core::Device::Cpu)
        }
        _ => candle_core::Device::Cpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_cli_runtime_entries_are_cli_sourced_and_classified() {
        let entries = serve_cli_runtime_entries(
            Some("int8"),
            Some(&PathBuf::from("/tmp/profile.jsonl")),
            Some("abc123"),
            Some("sha256:test"),
            Some("Qwen/Qwen3-30B-A3B-GPTQ-Int4"),
            Some(32),
            Some("{\"schema_version\":1}"),
        );
        let snapshot = RuntimeConfigSnapshot::from_entries(entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_KV_DTYPE").effective_value, "int8");
        assert_eq!(entry("FERRUM_KV_DTYPE").source, RuntimeConfigSource::Cli);
        assert!(entry("FERRUM_KV_DTYPE")
            .affects
            .contains(&ferrum_types::RuntimeConfigEffect::Correctness));
        assert_eq!(
            entry("FERRUM_PROFILE_JSONL").effective_value,
            "/tmp/profile.jsonl"
        );
        assert_eq!(
            entry("FERRUM_PROFILE_ENV_HASH").effective_value,
            "sha256:test"
        );
        assert_eq!(entry("FERRUM_PROFILE_CONCURRENCY").effective_value, "32");
        assert!(entry("FERRUM_PROFILE_JSONL")
            .affects
            .contains(&ferrum_types::RuntimeConfigEffect::Diagnostics));
    }

    #[test]
    fn serve_runtime_snapshot_prefers_cli_over_config_file() {
        let config_entries = crate::config::RuntimeCliConfig {
            kv_dtype: Some("fp16".to_string()),
            ..Default::default()
        }
        .runtime_config_entries();
        let cli_entries =
            serve_cli_runtime_entries(Some("int8"), None, None, None, None, None, None);

        let snapshot = merge_runtime_config_sources(
            config_entries,
            RuntimeConfigSnapshot::default(),
            cli_entries,
        );
        let kv = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_KV_DTYPE")
            .unwrap();
        assert_eq!(kv.effective_value, "int8");
        assert_eq!(kv.source, RuntimeConfigSource::Cli);
    }

    #[test]
    fn autosize_snapshot_diff_marks_only_new_values_as_memory_profile() {
        let before = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new("FERRUM_KV_MAX_BLOCKS", "2048", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_MOE_GRAPH", "1", RuntimeConfigSource::Env),
        ]);
        let after = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new("FERRUM_KV_MAX_BLOCKS", "2048", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_MOE_GRAPH", "1", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new(
                "FERRUM_MAX_BATCHED_TOKENS",
                "2048",
                RuntimeConfigSource::Env,
            ),
            RuntimeConfigEntry::new("FERRUM_PAGED_MAX_SEQS", "32", RuntimeConfigSource::Env),
        ]);

        let entries = runtime_entries_added_by_snapshot(
            &before,
            &after,
            SERVE_AUTOSIZE_RUNTIME_KEYS,
            RuntimeConfigSource::MemoryProfile,
        );
        let snapshot = RuntimeConfigSnapshot::from_entries(entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(snapshot.entries.len(), 2);
        assert_eq!(
            entry("FERRUM_MAX_BATCHED_TOKENS").source,
            RuntimeConfigSource::MemoryProfile
        );
        assert_eq!(entry("FERRUM_PAGED_MAX_SEQS").effective_value, "32");
        assert!(snapshot
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_KV_MAX_BLOCKS"));
    }

    #[test]
    fn materialized_autosize_entries_keep_memory_profile_source() {
        let autosize_entries = vec![RuntimeConfigEntry::new(
            "FERRUM_MAX_BATCHED_TOKENS",
            "2048",
            RuntimeConfigSource::MemoryProfile,
        )];
        let materialized_keys = autosize_entries
            .iter()
            .map(|entry| entry.key.clone())
            .collect::<Vec<_>>();
        let env_snapshot = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new(
                "FERRUM_MAX_BATCHED_TOKENS",
                "2048",
                RuntimeConfigSource::Env,
            ),
            RuntimeConfigEntry::new("FERRUM_KV_DTYPE", "fp16", RuntimeConfigSource::Env),
        ]);
        let env_snapshot = remove_materialized_config_env_entries(env_snapshot, &materialized_keys);
        let snapshot = merge_runtime_config_sources(autosize_entries, env_snapshot, Vec::new());
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(
            entry("FERRUM_MAX_BATCHED_TOKENS").source,
            RuntimeConfigSource::MemoryProfile
        );
        assert_eq!(entry("FERRUM_KV_DTYPE").source, RuntimeConfigSource::Env);
    }

    #[test]
    fn nvidia_smi_gpu_query_parser_extracts_cuda_hardware_fields() {
        let probe = parse_nvidia_smi_gpu_query("NVIDIA GeForce RTX 4090, 8.9, 24564\n").unwrap();

        assert_eq!(probe.name.as_deref(), Some("NVIDIA GeForce RTX 4090"));
        assert_eq!(probe.compute_capability.as_deref(), Some("8.9"));
        assert_eq!(probe.vram_bytes, Some(24564 * 1024 * 1024));
    }

    #[test]
    fn nvidia_smi_gpu_query_parser_handles_units_and_empty_values() {
        let probe = parse_nvidia_smi_gpu_query("N/A, N/A, 24 GiB\n").unwrap();

        assert_eq!(probe.name, None);
        assert_eq!(probe.compute_capability, None);
        assert_eq!(probe.vram_bytes, Some(24 * 1024 * 1024 * 1024));
    }

    #[test]
    fn cuda_runtime_version_parsers_accept_nvcc_and_nvidia_smi_output() {
        let nvcc = "Cuda compilation tools, release 12.8, V12.8.93";
        let smi = "| NVIDIA-SMI 570.86.15    Driver Version: 570.86.15    CUDA Version: 12.8 |";

        assert_eq!(parse_nvcc_cuda_release(nvcc).as_deref(), Some("12.8"));
        assert_eq!(parse_nvidia_smi_cuda_version(smi).as_deref(), Some("12.8"));
        assert_eq!(parse_first_u32("128\n").unwrap(), 128);
        assert_eq!(
            infer_sm_count_from_gpu_name("NVIDIA GeForce RTX 4090"),
            Some(128)
        );
    }

    #[test]
    fn cuda_hardware_capabilities_uses_runtime_probe_values() {
        let hardware = cuda_hardware_capabilities(
            CompiledKernelFeatures {
                cuda: true,
                cuda_graph: true,
                ..CompiledKernelFeatures::default()
            },
            CudaDeviceProbe {
                name: Some("NVIDIA GeForce RTX 4090".to_string()),
                cuda_runtime: Some("12.8".to_string()),
                compute_capability: Some("8.9".to_string()),
                vram_bytes: Some(24 * 1024 * 1024 * 1024),
                sm_count: Some(128),
            },
        );

        assert_eq!(hardware.backend, "cuda");
        assert_eq!(hardware.cuda_runtime.as_deref(), Some("12.8"));
        assert_eq!(hardware.compute_capability.as_deref(), Some("8.9"));
        assert_eq!(hardware.vram_bytes, Some(24 * 1024 * 1024 * 1024));
        assert_eq!(hardware.sm_count, Some(128));
        assert!(hardware.supported_kv_dtypes.contains(&"int8".to_string()));
        assert!(hardware.compiled_features.cuda);
    }

    #[test]
    fn m3_runtime_preset_entries_are_cli_sourced_defaults() {
        let entries =
            runtime_preset_entries(M3_QWEN3_30B_A3B_INT4_PRESET, RuntimeConfigSource::Cli).unwrap();
        let snapshot = RuntimeConfigSnapshot::from_entries(entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_BACKEND").effective_value, "cuda");
        assert_eq!(entry("FERRUM_MOE_GRAPH").effective_value, "1");
        assert_eq!(entry("FERRUM_VLLM_MOE").effective_value, "1");
        assert_eq!(entry("FERRUM_VLLM_MOE_PAIR_IDS").effective_value, "1");
        assert_eq!(entry("FERRUM_PREFIX_CACHE").effective_value, "0");
        assert_eq!(entry("FERRUM_BACKEND").source, RuntimeConfigSource::Cli);
        assert_eq!(snapshot.entries.len(), 11);
    }

    #[test]
    fn m3_runtime_preset_entries_can_be_default_sourced_for_model_inference() {
        let entries =
            runtime_preset_entries(M3_QWEN3_30B_A3B_INT4_PRESET, RuntimeConfigSource::Default)
                .unwrap();
        let snapshot = RuntimeConfigSnapshot::from_entries(entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_MOE_GRAPH").effective_value, "1");
        assert_eq!(entry("FERRUM_VLLM_MOE").effective_value, "1");
        assert_eq!(
            entry("FERRUM_MOE_GRAPH").source,
            RuntimeConfigSource::Default
        );
        assert_eq!(
            entry("FERRUM_VLLM_MOE").source,
            RuntimeConfigSource::Default
        );
    }

    #[test]
    fn qwen3_moe_serve_defaults_are_typed_default_entries() {
        let entries = crate::runtime_env::moe_graph_default_entries(
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );
        let snapshot = RuntimeConfigSnapshot::from_entries(entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_MOE_GRAPH").effective_value, "1");
        assert_eq!(
            entry("FERRUM_MOE_GRAPH").source,
            RuntimeConfigSource::Default
        );
        #[cfg(feature = "vllm-moe-marlin")]
        {
            assert_eq!(entry("FERRUM_VLLM_MOE").effective_value, "1");
            assert_eq!(entry("FERRUM_VLLM_MOE_PAIR_IDS").effective_value, "1");
            assert_eq!(snapshot.entries.len(), 3);
        }
        #[cfg(not(feature = "vllm-moe-marlin"))]
        assert_eq!(snapshot.entries.len(), 1);
    }

    #[test]
    fn qwen3_moe_serve_defaults_keep_config_file_overrides() {
        let config_entries = crate::config::RuntimeCliConfig {
            moe_graph: Some(false),
            vllm_moe: Some(false),
            ..Default::default()
        }
        .runtime_config_entries();
        let current = RuntimeConfigSnapshot::from_entries(config_entries.clone());
        let mut entries =
            crate::runtime_env::moe_graph_default_entries(&current, RuntimeConfigSource::Default);
        entries.extend(config_entries);
        let snapshot = RuntimeConfigSnapshot::from_entries(entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_MOE_GRAPH").effective_value, "0");
        assert_eq!(
            entry("FERRUM_MOE_GRAPH").source,
            RuntimeConfigSource::ConfigFile
        );
        assert_eq!(entry("FERRUM_VLLM_MOE").effective_value, "0");
        assert_eq!(
            entry("FERRUM_VLLM_MOE").source,
            RuntimeConfigSource::ConfigFile
        );
    }

    #[test]
    fn model_inferred_m3_preset_keeps_config_file_overrides() {
        let mut inferred_entries =
            runtime_preset_entries(M3_QWEN3_30B_A3B_INT4_PRESET, RuntimeConfigSource::Default)
                .unwrap();
        inferred_entries.extend(
            crate::config::RuntimeCliConfig {
                prefix_cache: Some(true),
                kv_max_blocks: Some(4096),
                ..Default::default()
            }
            .runtime_config_entries(),
        );
        let snapshot = RuntimeConfigSnapshot::from_entries(inferred_entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_PREFIX_CACHE").effective_value, "1");
        assert_eq!(
            entry("FERRUM_PREFIX_CACHE").source,
            RuntimeConfigSource::ConfigFile
        );
        assert_eq!(entry("FERRUM_KV_MAX_BLOCKS").effective_value, "4096");
        assert_eq!(
            entry("FERRUM_KV_MAX_BLOCKS").source,
            RuntimeConfigSource::ConfigFile
        );
        assert_eq!(
            entry("FERRUM_MOE_GRAPH").source,
            RuntimeConfigSource::Default
        );
    }

    #[test]
    fn materialized_inferred_m3_entries_keep_default_source() {
        let inferred_entries =
            runtime_preset_entries(M3_QWEN3_30B_A3B_INT4_PRESET, RuntimeConfigSource::Default)
                .unwrap();
        let materialized_keys = inferred_entries
            .iter()
            .map(|entry| entry.key.clone())
            .collect::<Vec<_>>();
        let env_snapshot = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new("FERRUM_MOE_GRAPH", "1", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_VLLM_MOE", "1", RuntimeConfigSource::Env),
        ]);
        let env_snapshot = remove_materialized_config_env_entries(env_snapshot, &materialized_keys);
        let snapshot = merge_runtime_config_sources(inferred_entries, env_snapshot, Vec::new());
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(
            entry("FERRUM_MOE_GRAPH").source,
            RuntimeConfigSource::Default
        );
        assert_eq!(
            entry("FERRUM_VLLM_MOE").source,
            RuntimeConfigSource::Default
        );
    }

    #[test]
    fn runtime_config_fields_override_preset_defaults_before_env() {
        let preset_entries =
            runtime_preset_entries(M3_QWEN3_30B_A3B_INT4_PRESET, RuntimeConfigSource::Cli).unwrap();
        let config_entries = crate::config::RuntimeCliConfig {
            prefix_cache: Some(true),
            kv_max_blocks: Some(4096),
            ..Default::default()
        }
        .runtime_config_entries();
        let mut non_env_entries = preset_entries;
        non_env_entries.extend(config_entries);
        let snapshot = RuntimeConfigSnapshot::from_entries(non_env_entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(entry("FERRUM_PREFIX_CACHE").effective_value, "1");
        assert_eq!(
            entry("FERRUM_PREFIX_CACHE").source,
            RuntimeConfigSource::ConfigFile
        );
        assert_eq!(entry("FERRUM_KV_MAX_BLOCKS").effective_value, "4096");
        assert_eq!(
            entry("FERRUM_KV_MAX_BLOCKS").source,
            RuntimeConfigSource::ConfigFile
        );
        assert_eq!(entry("FERRUM_VLLM_MOE").effective_value, "1");
        assert_eq!(entry("FERRUM_VLLM_MOE").source, RuntimeConfigSource::Cli);
    }

    #[test]
    fn serve_runtime_snapshot_prefers_env_over_config_file() {
        let config_entries = crate::config::RuntimeCliConfig {
            kv_dtype: Some("fp16".to_string()),
            ..Default::default()
        }
        .runtime_config_entries();
        let env_snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
            "FERRUM_KV_DTYPE",
            "int8",
            RuntimeConfigSource::Env,
        )]);

        let snapshot = merge_runtime_config_sources(config_entries, env_snapshot, Vec::new());
        let kv = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_KV_DTYPE")
            .unwrap();
        assert_eq!(kv.effective_value, "int8");
        assert_eq!(kv.source, RuntimeConfigSource::Env);
    }

    #[test]
    fn materialized_config_env_entries_keep_config_file_source() {
        let config_entries = crate::config::RuntimeCliConfig {
            prefix_cache: Some(true),
            ..Default::default()
        }
        .runtime_config_entries();
        let env_snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
            "FERRUM_PREFIX_CACHE",
            "1",
            RuntimeConfigSource::Env,
        )]);
        let env_snapshot = remove_materialized_config_env_entries(
            env_snapshot,
            &[String::from("FERRUM_PREFIX_CACHE")],
        );

        let snapshot = merge_runtime_config_sources(config_entries, env_snapshot, Vec::new());
        let prefix_cache = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_PREFIX_CACHE")
            .unwrap();
        assert_eq!(prefix_cache.effective_value, "1");
        assert_eq!(prefix_cache.source, RuntimeConfigSource::ConfigFile);
    }

    #[test]
    fn serve_runtime_snapshot_prefers_cli_over_env() {
        let env_snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
            "FERRUM_KV_DTYPE",
            "int8",
            RuntimeConfigSource::Env,
        )]);
        let cli_entries =
            serve_cli_runtime_entries(Some("bf16"), None, None, None, None, None, None);

        let snapshot = merge_runtime_config_sources(Vec::new(), env_snapshot, cli_entries);
        let kv = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_KV_DTYPE")
            .unwrap();
        assert_eq!(kv.effective_value, "bf16");
        assert_eq!(kv.source, RuntimeConfigSource::Cli);
    }

    #[test]
    fn effective_kv_dtype_precedence_is_cli_env_config() {
        assert_eq!(
            resolve_effective_kv_dtype(Some("bf16"), Some("int8"), Some("fp16")),
            Some("bf16")
        );
        assert_eq!(
            resolve_effective_kv_dtype(None, Some("int8"), Some("fp16")),
            Some("int8")
        );
        assert_eq!(
            resolve_effective_kv_dtype(None, None, Some("fp16")),
            Some("fp16")
        );
        assert_eq!(resolve_effective_kv_dtype(None, None, None), None);
    }
}
