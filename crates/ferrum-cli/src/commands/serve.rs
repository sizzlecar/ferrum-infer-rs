//! Serve command - Start the HTTP inference server

use crate::config::CliConfig;
use crate::runtime_env::runtime_snapshot_value;
use clap::Args;
use colored::*;
use ferrum_bench_core::{ProfileMetadata, ProfileSinkConfig};
use ferrum_models::source::ModelFormat;
use ferrum_server::{AxumServer, HttpServer, ServerConfig};
use ferrum_types::{
    CompiledKernelFeatures, CompiledNativeOperatorArtifact, FerrumConfigBuilder,
    HardwareCapabilities, ModelCapabilities, MoeCapabilities, ResolvedFerrumConfig, Result,
    RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource, WorkloadProfile,
    M3_QWEN3_30B_A3B_INT4_PRESET, QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET,
};
use std::path::Path;
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

    /// Backend: auto, cpu, metal, cuda.
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

    /// Layer-split decode pipeline mode for multi-GPU CUDA serving.
    #[arg(long, value_enum)]
    pub layer_split_pipeline_mode: Option<crate::layer_split_pipeline::LayerSplitPipelineModeArg>,

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

    /// Exact device-wide memory budget available to runtime weights and
    /// dynamic resources. This is the same typed ceiling used by `run`.
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

    /// Prefer prefilling until this many requests are active before early decodes.
    #[arg(long, value_name = "N")]
    pub scheduler_prefill_first_until_active: Option<usize>,

    /// Cap per-request scheduler prefill chunks before they enter the engine.
    #[arg(long, value_name = "N")]
    pub scheduler_prefill_step_chunk: Option<usize>,

    /// Cap prefill chunks while decode requests are active.
    #[arg(long, value_name = "N")]
    pub scheduler_active_decode_prefill_chunk: Option<usize>,

    /// Enable prefix caching (`FERRUM_PREFIX_CACHE=1`).
    #[arg(
        long,
        conflicts_with_all = ["no_enable_prefix_caching", "disable_prefix_cache"]
    )]
    pub enable_prefix_caching: bool,

    /// Disable prefix caching (`FERRUM_PREFIX_CACHE=0`).
    #[arg(long, conflicts_with = "enable_prefix_cache")]
    pub no_enable_prefix_caching: bool,

    /// Enable prefix cache (`FERRUM_PREFIX_CACHE=1`).
    #[arg(
        long,
        conflicts_with_all = ["no_enable_prefix_caching", "disable_prefix_cache"]
    )]
    pub enable_prefix_cache: bool,

    /// Disable prefix cache (`FERRUM_PREFIX_CACHE=0`).
    #[arg(long, conflicts_with_all = ["enable_prefix_caching", "enable_prefix_cache"])]
    pub disable_prefix_cache: bool,

    /// Session cache mode (`off` or `memory`).
    #[arg(long, value_name = "MODE", value_parser = ["off", "memory"])]
    pub session_cache: Option<String>,

    /// Maximum in-memory session cache entries.
    #[arg(long, value_name = "N")]
    pub session_cache_max_entries: Option<usize>,

    /// Approximate maximum tokens retained per session.
    #[arg(long, value_name = "N")]
    pub session_cache_max_tokens: Option<usize>,

    /// KV cache element dtype (Dim 5 polymorphism point). Accepts
    /// `fp16`, `bf16`, `int8`, `fp8`. Default `fp16`. INT8 / FP8
    /// require model wire-up; today only the kernel + type layer ships.
    /// Override via `FERRUM_KV_DTYPE` env var.
    #[arg(long, value_name = "DTYPE")]
    pub kv_dtype: Option<String>,

    /// Per-sequence KV token capacity (`FERRUM_KV_CAPACITY`).
    #[arg(long, value_name = "N")]
    pub kv_capacity: Option<usize>,

    /// Global KV block budget (`FERRUM_KV_MAX_BLOCKS`).
    #[arg(long, value_name = "N")]
    pub kv_max_blocks: Option<usize>,

    /// Use GPU argmax for greedy decoding (`FERRUM_GREEDY_ARGMAX=1`).
    #[arg(long, conflicts_with = "disable_greedy_argmax")]
    pub greedy_argmax: bool,

    /// Disable GPU argmax for greedy decoding (`FERRUM_GREEDY_ARGMAX=0`).
    #[arg(long, conflicts_with = "greedy_argmax")]
    pub disable_greedy_argmax: bool,

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

    /// Generate a synthetic/no-weight observability vertical-slice artifact and exit.
    #[arg(long, value_name = "DIR")]
    pub observability_vertical_slice_out: Option<PathBuf>,

    /// Write native structured profile events to this JSONL path.
    #[arg(long, value_name = "PATH")]
    pub profile_jsonl: Option<PathBuf>,

    /// Product observability detail level.
    #[arg(long, value_enum, default_value_t = crate::observability_product::ProfileDetailArg::Off)]
    pub profile_detail: crate::observability_product::ProfileDetailArg,

    /// Write product memory profile events to this JSONL path.
    #[arg(long, value_name = "PATH")]
    pub memory_profile_jsonl: Option<PathBuf>,

    /// Write scheduler iteration trace events to this JSONL path.
    #[arg(long, value_name = "PATH")]
    pub scheduler_trace_jsonl: Option<PathBuf>,

    /// Write a sanitized request/replay bundle to this directory.
    #[arg(long, value_name = "DIR")]
    pub request_dump_dir: Option<PathBuf>,

    /// Product observability sampling rate for resource lifecycle events.
    #[arg(long, default_value_t = crate::observability_product::default_profile_sample_rate())]
    pub profile_sample_rate: f64,

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

    /// Startup-loaded LoRA adapter, formatted as NAME=PATH. May be repeated.
    #[arg(long = "lora", value_name = "NAME=PATH")]
    pub lora: Vec<String>,

    /// Public model id template for LoRA adapters. Supports <base> and <name>.
    #[arg(long, value_name = "TEMPLATE", default_value = "<base>:<name>")]
    pub lora_model_id_template: String,
}

pub async fn execute(cmd: ServeCommand, config: CliConfig) -> Result<()> {
    let ServeCommand {
        model,
        model_option,
        host,
        port,
        tts_slots,
        backend,
        qwen35_reference,
        gpu_devices,
        layer_split_pipeline_mode,
        spec_draft,
        spec_tokens,
        gpu_memory_utilization,
        runtime_memory_budget_bytes,
        max_model_len,
        max_num_seqs,
        max_num_batched_tokens,
        scheduler_prefill_first_until_active,
        scheduler_prefill_step_chunk,
        scheduler_active_decode_prefill_chunk,
        enable_prefix_caching,
        no_enable_prefix_caching,
        enable_prefix_cache,
        disable_prefix_cache,
        session_cache,
        session_cache_max_entries,
        session_cache_max_tokens,
        kv_dtype,
        kv_capacity,
        kv_max_blocks,
        greedy_argmax,
        disable_greedy_argmax,
        batched_graph,
        disable_batched_graph,
        unified_graph,
        disable_unified_graph,
        unified_graph_layers_only,
        disable_unified_graph_layers_only,
        unified_graph_lm_head_eager,
        disable_unified_graph_lm_head_eager,
        runtime_preset,
        effective_config_json,
        decision_trace_jsonl,
        observability_vertical_slice_out,
        profile_jsonl,
        profile_detail,
        memory_profile_jsonl,
        scheduler_trace_jsonl,
        request_dump_dir,
        profile_sample_rate,
        profile_commit_sha,
        profile_env_hash,
        profile_model,
        profile_concurrency,
        profile_runtime_flags_json,
        lora,
        lora_model_id_template,
    } = cmd;

    if let Some(out_dir) = observability_vertical_slice_out.as_ref() {
        crate::observability_vertical_slice::write_observability_vertical_slice(
            ferrum_types::ProfileEntrypoint::Serve,
            out_dir,
        )?;
        println!(
            "OBSERVABILITY VERTICAL SLICE ARTIFACT: {}",
            out_dir.display()
        );
        return Ok(());
    }

    // Resolve model
    let model_name = model
        .or(model_option)
        .or(config.models.default_model.clone())
        .unwrap_or_else(|| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());
    let serve_start = std::time::Instant::now();
    let product_observability = crate::observability_product::ProductObservabilityConfig::new(
        ferrum_types::ProfileEntrypoint::Serve,
        &model_name,
        profile_jsonl.as_ref(),
        profile_detail,
        memory_profile_jsonl.as_ref(),
        scheduler_trace_jsonl.as_ref(),
        request_dump_dir.as_ref(),
        profile_sample_rate,
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

    // Select the requested device before model/cache resolution. Explicit
    // backend requests must fail closed instead of doing model work and then
    // silently running on CPU.
    let mut device = super::run::select_device(&backend)?;
    let mut gpu_selection =
        crate::gpu_devices::resolve_cuda_gpu_devices(gpu_devices.as_deref(), &device)?;
    if let Some(selection) = &gpu_selection {
        device = selection.primary_device();
        println!(
            "{} {} ({})",
            "CUDA GPUs:".dimmed(),
            selection.selected_csv(),
            selection.selected_distributed_strategy
        );
    }
    let backend_initialized_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let backend_initialized_memory = serve_process_memory_observation_between(
        process_start_sample.clone(),
        backend_initialized_sample.clone(),
    );

    // Print banner
    print_banner();

    // `run` and `serve` share one source decision. This covers exact GGUF
    // aliases/files, local model directories, HF cache hits, and resumable HF
    // download without an entrypoint-specific fallback chain.
    let cache_dir = crate::source_resolver::hf_cache_dir(&config);
    let resolved = crate::source_resolver::resolve_model_source(
        &model_name,
        &cache_dir,
        crate::source_resolver::DownloadPolicy::AutoDownload,
        None,
    )
    .await?;
    let original_source = resolved.original_source;
    let source = resolved.source;
    let model_id = crate::source_resolver::public_model_id(&source);
    let gguf_path = (source.format == ModelFormat::GGUF).then(|| source.local_path.clone());
    println!("{} {}", "Model:".dimmed(), model_id.cyan());
    println!("{} {}", "Path:".dimmed(), source.local_path.display());

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

    let lora_specs = parse_lora_specs(&lora)?;
    let startup_lora_adapters = if lora_specs.is_empty() {
        Vec::new()
    } else {
        ferrum_models::load_startup_lora_adapters(
            &model_id,
            Some(&lora_model_id_template),
            &lora_specs,
        )?
    };
    for adapter in &startup_lora_adapters {
        println!(
            "{} {} -> {} ({})",
            "LoRA:".dimmed(),
            adapter.name.cyan(),
            adapter.public_model_id.cyan(),
            adapter.path.display()
        );
    }

    let host = host.unwrap_or_else(|| config.server.host.clone());
    let port = port.unwrap_or(config.server.port);
    let kv_runtime_snapshot = RuntimeConfigSnapshot::capture_current();
    let env_kv_dtype = runtime_snapshot_value(&kv_runtime_snapshot, "FERRUM_KV_DTYPE");
    let effective_kv_dtype = resolve_effective_kv_dtype(
        kv_dtype.as_deref(),
        env_kv_dtype,
        config.runtime.kv_dtype.as_deref(),
    );

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
        let draft_id = crate::source_resolver::resolve_model_alias(draft_name);
        println!("{} {}", "Draft model:".dimmed(), draft_id.cyan());
        let cache_dir = crate::source_resolver::hf_cache_dir(&config);
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

    println!("{} {:?}", "Device:".dimmed(), device);
    let serve_profile_entries = crate::source_resolver::serve_profile_runtime_entries(
        &source.local_path,
        &device,
        &RuntimeConfigSnapshot::capture_current(),
        RuntimeConfigSource::Default,
    );
    if !serve_profile_entries.is_empty() {
        non_env_runtime_entries.extend(serve_profile_entries.clone());
        non_env_runtime_entries =
            RuntimeConfigSnapshot::from_entries(non_env_runtime_entries).entries;
        materialized_runtime_keys.extend(crate::runtime_env::materialize_runtime_env_defaults(
            &serve_profile_entries,
        ));
        materialized_runtime_keys.sort();
        materialized_runtime_keys.dedup();
    }
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
    // Materialize the multi-GPU layer-split plan. The safetensors path
    // gets the layer count from ModelDefinition; the GGUF path reads it
    // from the file header — without this the placeholder plan
    // (`layers=auto`) reaches the engine and is rejected.
    let model_layer_count = if let Some(definition) = model_definition.as_ref() {
        Some(definition.num_hidden_layers)
    } else if let (Some(selection), Some(p)) = (gpu_selection.as_ref(), gguf_path.as_ref()) {
        if selection.selected_layer_split_plan.is_some() {
            Some(ferrum_models::gguf_config::gguf_num_layers(p)?)
        } else {
            None
        }
    } else {
        None
    };
    if let (Some(selection), Some(layer_count)) = (gpu_selection.as_mut(), model_layer_count) {
        if selection.apply_model_layer_count(layer_count)? {
            if let Some(plan) = selection.selected_layer_split_plan.as_deref() {
                println!("{}", format!("CUDA layer split plan: {plan}").dimmed());
            }
        }
    }
    let mut selected_runtime_preset_name = selected_runtime_preset_name;
    if selected_runtime_preset_name.is_none() {
        let inferred_preset = infer_runtime_preset_for_startup(
            arch_for_dispatch,
            model_definition.as_ref(),
            gpu_selection.as_ref(),
        );
        if let Some(preset) = inferred_preset {
            selected_runtime_preset_name = Some(preset.to_string());
            let mut inferred_entries =
                runtime_preset_entries(preset, RuntimeConfigSource::Default)?;
            inferred_entries.extend(non_env_runtime_entries);
            non_env_runtime_entries = RuntimeConfigSnapshot::from_entries(inferred_entries).entries;
            materialized_runtime_keys.extend(crate::runtime_env::materialize_runtime_env_defaults(
                &non_env_runtime_entries,
            ));
            materialized_runtime_keys.sort();
            materialized_runtime_keys.dedup();
        }
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

    let mut startup_cli_runtime_entries = serve_cli_runtime_entries(
        kv_dtype.as_deref(),
        kv_capacity,
        kv_max_blocks,
        max_model_len,
        max_num_seqs,
        max_num_batched_tokens,
        runtime_memory_budget_bytes.map(std::num::NonZeroUsize::get),
        scheduler_prefill_first_until_active,
        scheduler_prefill_step_chunk,
        scheduler_active_decode_prefill_chunk,
        greedy_argmax_cli_override(greedy_argmax, disable_greedy_argmax),
        prefix_cache_cli_override(
            enable_prefix_caching,
            no_enable_prefix_caching,
            enable_prefix_cache,
            disable_prefix_cache,
        ),
        session_cache.as_deref(),
        session_cache_max_entries,
        session_cache_max_tokens,
        profile_jsonl.as_ref(),
        scheduler_trace_jsonl.as_ref(),
        profile_commit_sha.as_deref(),
        profile_env_hash.as_deref(),
        profile_model.as_deref(),
        profile_concurrency,
        profile_runtime_flags_json.as_deref(),
        layer_split_pipeline_mode,
    );
    if let Some(enabled) = batched_graph_cli_override(batched_graph, disable_batched_graph) {
        startup_cli_runtime_entries.push(RuntimeConfigEntry::new(
            "FERRUM_BATCHED_GRAPH",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(enabled) = batched_graph_cli_override(unified_graph, disable_unified_graph) {
        startup_cli_runtime_entries.push(RuntimeConfigEntry::new(
            "FERRUM_UNIFIED_GRAPH",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(enabled) =
        batched_graph_cli_override(unified_graph_layers_only, disable_unified_graph_layers_only)
    {
        startup_cli_runtime_entries.push(RuntimeConfigEntry::new(
            "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(enabled) = batched_graph_cli_override(
        unified_graph_lm_head_eager,
        disable_unified_graph_lm_head_eager,
    ) {
        startup_cli_runtime_entries.push(RuntimeConfigEntry::new(
            "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(selection) = &gpu_selection {
        startup_cli_runtime_entries.extend(selection.runtime_config_entries());
    }
    if !startup_cli_runtime_entries.is_empty() {
        crate::runtime_env::materialize_runtime_env_effective(
            &RuntimeConfigSnapshot::from_entries(startup_cli_runtime_entries.clone()),
        );
    }
    let autosize_env_before = RuntimeConfigSnapshot::capture_current();
    // GPU-memory auto-sizing must run after the model source resolves.
    // HF-cache models are not `local_dir_path`, but they still need the same
    // KV block sizing as direct local safetensors directories.
    crate::gpu_mem_autosize::apply_auto_size(&source.local_path, gpu_memory_utilization);
    let autosize_runtime_entries = runtime_entries_changed_by_snapshot(
        &autosize_env_before,
        &RuntimeConfigSnapshot::capture_current(),
        SERVE_AUTOSIZE_RUNTIME_KEYS,
        RuntimeConfigSource::MemoryProfile,
    );
    let autosize_runtime_keys: Vec<String> = autosize_runtime_entries
        .iter()
        .map(|entry| entry.key.clone())
        .collect();
    startup_cli_runtime_entries.retain(|entry| !autosize_runtime_keys.contains(&entry.key));
    materialized_runtime_keys.extend(
        autosize_runtime_entries
            .iter()
            .map(|entry| entry.key.clone()),
    );
    non_env_runtime_entries.extend(autosize_runtime_entries);
    non_env_runtime_entries = RuntimeConfigSnapshot::from_entries(non_env_runtime_entries).entries;
    materialized_runtime_keys.sort();
    materialized_runtime_keys.dedup();
    let startup_auto_config = startup_auto_config(
        &device,
        arch_for_dispatch,
        model_definition.as_ref(),
        model_weight_bytes_from_path(&source.local_path),
        selected_runtime_preset_name.as_deref(),
        non_env_runtime_entries,
        materialized_runtime_keys,
        startup_cli_runtime_entries,
    )?;
    crate::runtime_env::materialize_runtime_env_effective(&startup_auto_config.runtime_config);
    write_startup_config_artifacts(
        &startup_auto_config,
        effective_config_json.as_deref(),
        decision_trace_jsonl.as_deref(),
    )?;
    let native_profile_jsonl = if product_observability.unified_product_profile_enabled() {
        None
    } else {
        profile_jsonl.clone()
    };
    configure_profile_sink(
        native_profile_jsonl,
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

    let lora_server_models: Vec<ferrum_server::LoraAdapterModel> = startup_lora_adapters
        .iter()
        .map(|adapter| {
            ferrum_server::LoraAdapterModel::new(
                adapter.name.clone(),
                adapter.public_model_id.clone(),
                adapter.path.display().to_string(),
            )
        })
        .collect();

    let mut cache_allocated_status = None;
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
            engine_config.model.source = Some(original_source);
            engine_config.sampling.default_params = ferrum_server::default_chat_sampling_params();
            engine_config.backend.device = device;
            if let Some(selection) = &gpu_selection {
                selection.insert_backend_options(&mut engine_config.backend.backend_options);
            }
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
            if let Some(selection) = &gpu_selection {
                selection.insert_backend_options(&mut engine_config.backend.backend_options);
            }
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
            engine_config.kv_cache.cache_type = serve_kv_cache_type_for_device(&device);
            engine_config.backend.device = device;
            engine_config.scheduler.policy = ferrum_types::SchedulingPolicy::ContinuousBatch;
            engine_config
                .apply_runtime_config_snapshot(&startup_auto_config.runtime_config)
                .map_err(ferrum_types::FerrumError::config)?;
            engine_config.backend.backend_options.insert(
                "model_path".to_string(),
                serde_json::Value::String(engine_model_path.clone()),
            );
            if qwen35_reference {
                engine_config.backend.backend_options.insert(
                    "qwen35_reference".to_string(),
                    serde_json::Value::Bool(true),
                );
            }
            if let Some(selection) = &gpu_selection {
                selection.insert_backend_options(&mut engine_config.backend.backend_options);
            }
            crate::layer_split_pipeline::insert_backend_option_from_runtime(
                &startup_auto_config.runtime_config,
                &mut engine_config.backend.backend_options,
            )?;
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
            if product_memory_enabled {
                cache_allocated_status = Some(engine.status().await);
            }
            AxumServer::from_llm(engine).with_prompt_template(
                crate::source_resolver::load_model_chat_template(&source.local_path),
            )
        }
    }
    .with_auto_config(startup_auto_config);
    let model_loaded_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let model_loaded_memory = serve_process_memory_observation_between(
        backend_initialized_sample
            .clone()
            .or_else(|| process_start_sample.clone()),
        model_loaded_sample.clone(),
    );
    let model_loaded_duration_us = serve_start
        .elapsed()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX);
    let profile_run_done_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let profile_run_done_memory = serve_process_memory_observation_between(
        model_loaded_sample.clone(),
        profile_run_done_sample.clone(),
    );
    let cache_allocated_sample = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let cache_allocated_memory = serve_process_memory_observation_between(
        profile_run_done_sample
            .clone()
            .or_else(|| model_loaded_sample.clone()),
        cache_allocated_sample.clone(),
    );
    let server = if lora_server_models.is_empty() {
        server
    } else {
        server.with_lora_adapters(model_id.clone(), lora_server_models)
    };
    crate::observability_product::write_actual_serve_startup_observability(
        &product_observability,
        model_loaded_duration_us,
        model_loaded_memory.clone(),
        actual_serve_startup_memory_stages(
            product_memory_enabled,
            process_start_memory.clone(),
            backend_initialized_memory.clone(),
            profile_run_done_memory.clone(),
            cache_allocated_memory.clone(),
            cache_allocated_status.clone(),
        ),
    )?;

    // Create server config
    let server_config = ServerConfig {
        host: host.clone(),
        port,
        request_dump_dir: request_dump_dir.clone(),
        profile_jsonl: product_observability
            .unified_product_profile_enabled()
            .then(|| profile_jsonl.clone())
            .flatten(),
        memory_profile_jsonl: product_observability
            .unified_product_profile_enabled()
            .then(|| memory_profile_jsonl.clone())
            .flatten(),
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
        _ = serve_shutdown_signal() => {
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
    let shutdown_after = product_memory_enabled
        .then(|| memory_sampler.sample())
        .flatten();
    let shutdown_memory = serve_process_memory_observation_between(
        model_loaded_sample
            .clone()
            .or_else(|| backend_initialized_sample.clone())
            .or_else(|| process_start_sample.clone()),
        shutdown_after,
    );
    crate::observability_product::append_actual_serve_memory_stage_observability(
        &product_observability,
        crate::observability_product::ActualMemoryStageObservation::new(
            "actual_serve_shutdown",
            "shutdown",
            None,
            shutdown_memory,
        ),
    )?;

    Ok(())
}

async fn serve_shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! {
                    _ = signal::ctrl_c() => {}
                    _ = terminate.recv() => {}
                }
            }
            Err(_) => {
                let _ = signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = signal::ctrl_c().await;
    }
}

fn serve_process_memory_observation_between(
    before: Option<crate::memory_profile::ProcessMemorySample>,
    after: Option<crate::memory_profile::ProcessMemorySample>,
) -> Option<crate::memory_profile::ProcessMemoryObservation> {
    after.map(|after| crate::memory_profile::ProcessMemoryObservation::from_samples(before, after))
}

fn actual_serve_startup_memory_stages(
    enabled: bool,
    process_start_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    backend_initialized_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    profile_run_done_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    cache_allocated_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
    cache_allocated_status: Option<ferrum_types::EngineStatus>,
) -> Vec<crate::observability_product::ActualMemoryStageObservation> {
    if !enabled {
        return Vec::new();
    }
    let profile_run_done = crate::observability_product::ActualMemoryStageObservation::new(
        "actual_serve_profile_run_done",
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
        "actual_serve_cache_allocated",
        "cache_allocated",
        None,
        cache_allocated_memory,
    );
    if let Some(status) = cache_allocated_status.as_ref() {
        cache_allocated = cache_allocated.with_engine_cache_status(status);
    }
    vec![
        crate::observability_product::ActualMemoryStageObservation::new(
            "actual_serve_process_start",
            "process_start",
            None,
            process_start_memory,
        ),
        crate::observability_product::ActualMemoryStageObservation::new(
            "actual_serve_backend_initialized",
            "backend_initialized",
            None,
            backend_initialized_memory,
        ),
        profile_run_done,
        cache_allocated,
    ]
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

fn parse_lora_specs(values: &[String]) -> Result<Vec<ferrum_models::StartupLoraSpec>> {
    let mut specs = Vec::with_capacity(values.len());
    for value in values {
        let (name, path) = value.split_once('=').ok_or_else(|| {
            ferrum_types::FerrumError::config(format!(
                "invalid --lora value {value:?}; expected NAME=PATH"
            ))
        })?;
        if name.is_empty() || path.is_empty() {
            return Err(ferrum_types::FerrumError::config(format!(
                "invalid --lora value {value:?}; expected non-empty NAME=PATH"
            )));
        }
        specs.push(ferrum_models::StartupLoraSpec {
            name: name.to_string(),
            path: PathBuf::from(shellexpand::tilde(path).to_string()),
        });
    }
    Ok(specs)
}

fn startup_auto_config(
    device: &ferrum_types::Device,
    architecture: Option<ferrum_models::Architecture>,
    model_definition: Option<&ferrum_models::ModelDefinition>,
    model_weight_bytes: Option<u64>,
    runtime_preset: Option<&str>,
    non_env_runtime_entries: Vec<RuntimeConfigEntry>,
    materialized_runtime_keys: Vec<String>,
    cli_runtime_entries: Vec<RuntimeConfigEntry>,
) -> Result<ResolvedFerrumConfig> {
    let mut env_snapshot = RuntimeConfigSnapshot::capture_current();
    env_snapshot = remove_materialized_config_env_entries(env_snapshot, &materialized_runtime_keys);
    let runtime_config =
        merge_runtime_config_sources(non_env_runtime_entries, env_snapshot, cli_runtime_entries);
    let hardware = hardware_capabilities_for_device(device);
    let model = model_definition
        .map(|definition| {
            model_capabilities_from_definition_with_weight_bytes_for_hardware(
                definition,
                model_weight_bytes,
                &hardware,
            )
        })
        .unwrap_or_else(ModelCapabilities::unknown);
    let workload = match runtime_preset {
        Some(M3_QWEN3_30B_A3B_INT4_PRESET) => WorkloadProfile::m3_qwen3_30b_a3b_int4(),
        Some(QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET) => {
            WorkloadProfile::qwen25_72b_gptq_int4_2x4090_layer_split()
        }
        Some(other) => {
            return Err(ferrum_types::FerrumError::config(format!(
                "unknown runtime preset: {other}"
            )));
        }
        None => match infer_runtime_preset_for_startup(architecture, model_definition, None) {
            Some(M3_QWEN3_30B_A3B_INT4_PRESET) => WorkloadProfile::m3_qwen3_30b_a3b_int4(),
            _ => WorkloadProfile::serving_default_for_hardware(&hardware),
        },
    };

    FerrumConfigBuilder::new(runtime_config)
        .with_model_capabilities(model)
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .map_err(|err| ferrum_types::FerrumError::config(format!("invalid auto config: {err}")))
}

pub(crate) fn merge_runtime_config_sources(
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

pub(crate) const SERVE_AUTOSIZE_RUNTIME_KEYS: &[&str] = &[
    "FERRUM_MAX_BATCHED_TOKENS",
    "FERRUM_KV_MAX_BLOCKS",
    "FERRUM_PAGED_MAX_SEQS",
    "FERRUM_KV_CAPACITY",
];

pub(crate) fn runtime_entries_changed_by_snapshot(
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
                (Some(before), Some(after)) if before != after => {
                    Some(RuntimeConfigEntry::new(*key, after, source))
                }
                _ => None,
            }
        })
        .collect()
}

pub(crate) fn runtime_preset_entries(
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
            ("FERRUM_KV_CAPACITY", "512"),
            ("FERRUM_MOE_GRAPH", "0"),
            ("FERRUM_VLLM_MOE", "1"),
            ("FERRUM_VLLM_MOE_PAIR_IDS", "1"),
            ("FERRUM_USE_VLLM_PAGED_ATTN", "1"),
            ("FERRUM_PREFIX_CACHE", "0"),
        ],
        QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET => &[
            ("FERRUM_BACKEND", "cuda"),
            ("FERRUM_LAYER_SPLIT_PIPELINE_MODE", "batch"),
            ("FERRUM_MAX_MODEL_LEN", "4096"),
            ("FERRUM_KV_MAX_BLOCKS", "1024"),
            ("FERRUM_KV_CAPACITY", "1024"),
            ("FERRUM_PAGED_MAX_SEQS", "16"),
            ("FERRUM_MAX_BATCHED_TOKENS", "1536"),
            ("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE", "16"),
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
    kv_capacity: Option<usize>,
    kv_max_blocks: Option<usize>,
    max_model_len: Option<usize>,
    max_num_seqs: Option<usize>,
    max_num_batched_tokens: Option<usize>,
    runtime_memory_budget_bytes: Option<usize>,
    scheduler_prefill_first_until_active: Option<usize>,
    scheduler_prefill_step_chunk: Option<usize>,
    scheduler_active_decode_prefill_chunk: Option<usize>,
    greedy_argmax: Option<bool>,
    prefix_cache: Option<bool>,
    session_cache: Option<&str>,
    session_cache_max_entries: Option<usize>,
    session_cache_max_tokens: Option<usize>,
    profile_jsonl: Option<&PathBuf>,
    scheduler_trace_jsonl: Option<&PathBuf>,
    profile_commit_sha: Option<&str>,
    profile_env_hash: Option<&str>,
    profile_model: Option<&str>,
    profile_concurrency: Option<u32>,
    profile_runtime_flags_json: Option<&str>,
    layer_split_pipeline_mode: Option<crate::layer_split_pipeline::LayerSplitPipelineModeArg>,
) -> Vec<RuntimeConfigEntry> {
    let mut entries = Vec::new();
    push_cli_runtime_entry(&mut entries, "FERRUM_KV_DTYPE", kv_dtype);
    push_cli_runtime_usize(&mut entries, "FERRUM_KV_CAPACITY", kv_capacity);
    push_cli_runtime_usize(&mut entries, "FERRUM_KV_MAX_BLOCKS", kv_max_blocks);
    push_cli_runtime_usize(&mut entries, "FERRUM_MAX_MODEL_LEN", max_model_len);
    push_cli_runtime_usize(&mut entries, "FERRUM_PAGED_MAX_SEQS", max_num_seqs);
    push_cli_runtime_usize(
        &mut entries,
        "FERRUM_MAX_BATCHED_TOKENS",
        max_num_batched_tokens,
    );
    push_cli_runtime_usize(
        &mut entries,
        "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES",
        runtime_memory_budget_bytes,
    );
    push_cli_runtime_usize(
        &mut entries,
        "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE",
        scheduler_prefill_first_until_active,
    );
    push_cli_runtime_usize(
        &mut entries,
        "FERRUM_SCHED_PREFILL_STEP_CHUNK",
        scheduler_prefill_step_chunk,
    );
    push_cli_runtime_usize(
        &mut entries,
        "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK",
        scheduler_active_decode_prefill_chunk,
    );
    if let Some(enabled) = greedy_argmax {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_GREEDY_ARGMAX",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(enabled) = prefix_cache {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PREFIX_CACHE_REQUESTED",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PREFIX_CACHE_PRODUCT",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PREFIX_CACHE",
            if enabled { "1" } else { "0" },
            RuntimeConfigSource::Cli,
        ));
    }
    push_cli_runtime_entry(&mut entries, "FERRUM_SESSION_CACHE", session_cache);
    push_cli_runtime_usize(
        &mut entries,
        "FERRUM_SESSION_CACHE_MAX_ENTRIES",
        session_cache_max_entries,
    );
    push_cli_runtime_usize(
        &mut entries,
        "FERRUM_SESSION_CACHE_MAX_TOKENS",
        session_cache_max_tokens,
    );
    if let Some(path) = profile_jsonl {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PROFILE_JSONL",
            path.to_string_lossy().to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
    if let Some(path) = scheduler_trace_jsonl {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_SCHEDULER_TRACE_JSONL",
            path.to_string_lossy().to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
    if profile_jsonl.is_some() || scheduler_trace_jsonl.is_some() {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PROFILE_ENTRYPOINT",
            "serve",
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
    crate::layer_split_pipeline::push_cli_runtime_entry(&mut entries, layer_split_pipeline_mode);
    entries
}

fn prefix_cache_cli_override(
    enable_vllm: bool,
    disable_vllm: bool,
    enable_product: bool,
    disable_product: bool,
) -> Option<bool> {
    if enable_vllm || enable_product {
        Some(true)
    } else if disable_vllm || disable_product {
        Some(false)
    } else {
        None
    }
}

fn greedy_argmax_cli_override(enable: bool, disable: bool) -> Option<bool> {
    if enable {
        Some(true)
    } else if disable {
        Some(false)
    } else {
        None
    }
}

fn batched_graph_cli_override(enable: bool, disable: bool) -> Option<bool> {
    if enable {
        Some(true)
    } else if disable {
        Some(false)
    } else {
        None
    }
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

fn push_cli_runtime_usize(entries: &mut Vec<RuntimeConfigEntry>, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        entries.push(RuntimeConfigEntry::new(
            key,
            value.to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
}

fn serve_kv_cache_type_for_device(device: &ferrum_types::Device) -> ferrum_types::KvCacheType {
    match device {
        ferrum_types::Device::CPU => ferrum_types::KvCacheType::Contiguous,
        _ => ferrum_types::KvCacheType::Paged,
    }
}

pub(crate) fn write_startup_config_artifacts(
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

#[cfg(test)]
pub(crate) fn model_capabilities_from_definition_with_weight_bytes(
    definition: &ferrum_models::ModelDefinition,
    model_weight_bytes: Option<u64>,
) -> ModelCapabilities {
    model_capabilities_from_definition_with_weight_bytes_and_recurrent_state_dtype(
        definition,
        model_weight_bytes,
        ferrum_types::DataType::FP32,
    )
}

pub(crate) fn model_capabilities_from_definition_with_weight_bytes_for_hardware(
    definition: &ferrum_models::ModelDefinition,
    model_weight_bytes: Option<u64>,
    hardware: &HardwareCapabilities,
) -> ModelCapabilities {
    model_capabilities_from_definition_with_weight_bytes_and_recurrent_state_dtype(
        definition,
        model_weight_bytes,
        qwen35_recurrent_state_dtype_for_hardware(hardware),
    )
}

fn model_capabilities_from_definition_with_weight_bytes_and_recurrent_state_dtype(
    definition: &ferrum_models::ModelDefinition,
    model_weight_bytes: Option<u64>,
    recurrent_state_dtype: ferrum_types::DataType,
) -> ModelCapabilities {
    let architecture = match definition.architecture {
        ferrum_models::Architecture::Qwen3Moe => "qwen3_moe",
        ferrum_models::Architecture::Qwen35Moe => "qwen3_5_moe",
        ferrum_models::Architecture::Gemma3 => "gemma3",
        ferrum_models::Architecture::Qwen35 => "qwen3_5",
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
    let moe = if matches!(
        definition.architecture,
        ferrum_models::Architecture::Qwen3Moe | ferrum_models::Architecture::Qwen35Moe
    ) {
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
    let recurrent_state_bytes_per_sequence =
        recurrent_state_bytes_per_sequence_from_definition(definition, recurrent_state_dtype);

    ModelCapabilities {
        architecture,
        quantization: quantization_from_definition(definition),
        moe,
        max_context_len: Some(definition.max_position_embeddings),
        num_hidden_layers: Some(definition.num_hidden_layers),
        head_dim,
        kv_heads: definition.num_key_value_heads,
        estimated_weight_bytes: model_weight_bytes
            .filter(|value| *value > 0)
            .or_else(|| estimated_weight_bytes_from_definition(definition)),
        recurrent_state_bytes_per_sequence,
        supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
        graph_safe_moe: false,
    }
}

fn qwen35_recurrent_state_dtype_for_hardware(
    hardware: &HardwareCapabilities,
) -> ferrum_types::DataType {
    if hardware.backend.eq_ignore_ascii_case("cuda") && hardware.compiled_features.cuda {
        ferrum_types::DataType::FP16
    } else {
        ferrum_types::DataType::FP32
    }
}

fn recurrent_state_bytes_per_sequence_from_definition(
    definition: &ferrum_models::ModelDefinition,
    dtype: ferrum_types::DataType,
) -> Option<u64> {
    if !matches!(
        definition.architecture,
        ferrum_models::Architecture::Qwen35 | ferrum_models::Architecture::Qwen35Moe
    ) {
        return None;
    }
    Some(
        ferrum_models::qwen35_config::Qwen35TextConfig::from_model_definition(definition)
            .ok()?
            .recurrent_state_bytes_per_slot(dtype)
            .ok()?,
    )
}

pub(crate) fn model_weight_bytes_from_path(path: &Path) -> Option<u64> {
    if path.is_file() {
        return std::fs::metadata(path)
            .ok()
            .map(|metadata| metadata.len())
            .filter(|value| *value > 0);
    }
    if !path.is_dir() {
        return None;
    }
    let mut total = 0u64;
    for entry in std::fs::read_dir(path).ok()?.flatten() {
        let entry_path = entry.path();
        let is_weight = entry_path
            .extension()
            .and_then(|value| value.to_str())
            .map(|ext| ext == "safetensors" || ext == "bin")
            .unwrap_or(false);
        if !is_weight {
            continue;
        }
        if let Ok(metadata) = std::fs::metadata(&entry_path) {
            total = total.saturating_add(metadata.len());
        }
    }
    (total > 0).then_some(total)
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
    let params = estimated_total_parameters_from_definition(definition)?;
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

fn estimated_total_parameters_from_definition(
    definition: &ferrum_models::ModelDefinition,
) -> Option<u64> {
    let dense_params = definition.to_model_info("__auto_config").num_parameters;
    if !matches!(
        definition.architecture,
        ferrum_models::Architecture::Qwen3Moe | ferrum_models::Architecture::Qwen35Moe
    ) {
        return Some(dense_params);
    }

    let hidden = definition.hidden_size as u128;
    let layers = definition.num_hidden_layers as u128;
    let vocab = definition.vocab_size as u128;
    let num_experts = definition
        .extra_params
        .get("num_experts")
        .and_then(|value| value.as_u64())? as u128;
    let moe_intermediate = definition
        .extra_params
        .get("moe_intermediate_size")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            (definition.intermediate_size > 0).then_some(definition.intermediate_size as u64)
        })? as u128;
    let shared_intermediate = definition
        .extra_params
        .get("shared_expert_intermediate_size")
        .and_then(|value| value.as_u64())
        .unwrap_or(0) as u128;

    let embedding_params = vocab.saturating_mul(hidden);
    let lm_head_params = embedding_params;
    let attention_params = layers
        .saturating_mul(4)
        .saturating_mul(hidden)
        .saturating_mul(hidden);
    let norm_params = layers.saturating_mul(2).saturating_mul(hidden);
    let router_params = layers.saturating_mul(hidden).saturating_mul(num_experts);
    let expert_params = layers
        .saturating_mul(num_experts)
        .saturating_mul(3)
        .saturating_mul(hidden)
        .saturating_mul(moe_intermediate);
    let shared_expert_params = layers
        .saturating_mul(3)
        .saturating_mul(hidden)
        .saturating_mul(shared_intermediate);
    let total = embedding_params
        .saturating_add(lm_head_params)
        .saturating_add(attention_params)
        .saturating_add(norm_params)
        .saturating_add(router_params)
        .saturating_add(expert_params)
        .saturating_add(shared_expert_params);
    Some(total.min(u64::MAX as u128) as u64)
}

pub(crate) fn hardware_capabilities_for_device(
    device: &ferrum_types::Device,
) -> HardwareCapabilities {
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
    let fa2_native = ferrum_kernels::native_ops::compiled_fa2_native_operator_artifact();
    CompiledKernelFeatures {
        cuda: cfg!(feature = "cuda"),
        vllm_paged_attn: cfg!(feature = "vllm-paged-attn-v2"),
        vllm_moe_marlin: cfg!(feature = "vllm-moe-marlin"),
        cuda_graph: cfg!(feature = "cuda"),
        greedy_argmax: cfg!(feature = "cuda") || cfg!(feature = "metal"),
        fa2_source: false,
        fa2_direct_ffi: cfg!(feature = "cuda"),
        fa2_native_operator_artifact: fa2_native.is_some(),
        fa2_native_operator_artifact_metadata: fa2_native.map(|artifact| {
            CompiledNativeOperatorArtifact {
                manifest_path: artifact.manifest_path.to_string(),
                artifact_path: artifact.artifact_path.to_string(),
                source_package_sha256: artifact.source_package_sha256.to_string(),
                inputs_sha256: artifact.inputs_sha256.to_string(),
                binary_sha256: artifact.binary_sha256.to_string(),
            }
        }),
    }
}

#[derive(Clone, Copy)]
struct RuntimePresetInferenceRule {
    preset: &'static str,
    architecture: ferrum_models::Architecture,
    quantization: Option<&'static str>,
    exact_hidden_size: Option<usize>,
    min_hidden_size: Option<usize>,
    exact_hidden_layers: Option<usize>,
    min_hidden_layers: Option<usize>,
    kv_heads: Option<usize>,
    num_experts: Option<u64>,
    experts_per_token: Option<u64>,
    distributed_strategy: Option<&'static str>,
    gpu_count: Option<usize>,
}

const RUNTIME_PRESET_INFERENCE_RULES: &[RuntimePresetInferenceRule] = &[
    RuntimePresetInferenceRule {
        preset: M3_QWEN3_30B_A3B_INT4_PRESET,
        architecture: ferrum_models::Architecture::Qwen3Moe,
        quantization: None,
        exact_hidden_size: Some(2048),
        min_hidden_size: None,
        exact_hidden_layers: None,
        min_hidden_layers: Some(40),
        kv_heads: Some(4),
        num_experts: Some(128),
        experts_per_token: Some(8),
        distributed_strategy: None,
        gpu_count: None,
    },
    RuntimePresetInferenceRule {
        preset: QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET,
        architecture: ferrum_models::Architecture::Qwen2,
        quantization: Some("gptq_int4"),
        exact_hidden_size: None,
        min_hidden_size: Some(8192),
        exact_hidden_layers: Some(80),
        min_hidden_layers: None,
        kv_heads: Some(8),
        num_experts: None,
        experts_per_token: None,
        distributed_strategy: Some("layer_split"),
        gpu_count: Some(2),
    },
];

fn infer_runtime_preset_for_startup(
    architecture: Option<ferrum_models::Architecture>,
    model_definition: Option<&ferrum_models::ModelDefinition>,
    gpu_selection: Option<&crate::gpu_devices::GpuDeviceSelection>,
) -> Option<&'static str> {
    let definition = model_definition?;
    RUNTIME_PRESET_INFERENCE_RULES
        .iter()
        .find(|rule| rule.matches(architecture, definition, gpu_selection))
        .map(|rule| rule.preset)
}

impl RuntimePresetInferenceRule {
    fn matches(
        &self,
        architecture: Option<ferrum_models::Architecture>,
        definition: &ferrum_models::ModelDefinition,
        gpu_selection: Option<&crate::gpu_devices::GpuDeviceSelection>,
    ) -> bool {
        if architecture != Some(self.architecture) {
            return false;
        }
        if self.quantization.is_some()
            && quantization_from_definition(definition).as_deref() != self.quantization
        {
            return false;
        }
        if self
            .exact_hidden_size
            .is_some_and(|value| definition.hidden_size != value)
        {
            return false;
        }
        if self
            .min_hidden_size
            .is_some_and(|value| definition.hidden_size < value)
        {
            return false;
        }
        if self
            .exact_hidden_layers
            .is_some_and(|value| definition.num_hidden_layers != value)
        {
            return false;
        }
        if self
            .min_hidden_layers
            .is_some_and(|value| definition.num_hidden_layers < value)
        {
            return false;
        }
        if self
            .kv_heads
            .is_some_and(|value| definition.num_key_value_heads != Some(value))
        {
            return false;
        }
        if self.num_experts.is_some()
            && definition
                .extra_params
                .get("num_experts")
                .and_then(|value| value.as_u64())
                != self.num_experts
        {
            return false;
        }
        if self.experts_per_token.is_some()
            && definition
                .extra_params
                .get("num_experts_per_tok")
                .and_then(|value| value.as_u64())
                != self.experts_per_token
        {
            return false;
        }
        if self.distributed_strategy.is_some() || self.gpu_count.is_some() {
            let Some(selection) = gpu_selection else {
                return false;
            };
            if self
                .distributed_strategy
                .is_some_and(|value| selection.selected_distributed_strategy != value)
            {
                return false;
            }
            if self
                .gpu_count
                .is_some_and(|value| selection.selected_gpu_devices.len() != value)
            {
                return false;
            }
        }
        true
    }
}

// `find_cached_model` and `detect_format` previously lived here as forks
// of the `run.rs` versions. They moved to `crate::source_resolver` so the
// HF cache walk + format detection have a single source of truth across
// `run` / `serve` / `bench`. Use `crate::source_resolver::find_cached_model`
// / `crate::source_resolver::detect_format` directly.

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

    const QWEN35_ARTIFACT_ROOT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/goals/model-coverage-2026-06-12/artifacts/",
        "w3_hf_config_probe_20260617T131209Z_f97c1d6f"
    );

    #[test]
    fn serve_parses_explicit_qwen35_reference_flag() {
        use clap::Parser;

        #[derive(Parser)]
        struct TestCli {
            #[command(flatten)]
            serve: ServeCommand,
        }

        let parsed = TestCli::parse_from(["ferrum", "--model", "qwen3.5", "--qwen35-reference"]);

        assert!(parsed.serve.qwen35_reference);
    }

    #[test]
    fn serve_cli_runtime_entries_are_cli_sourced_and_classified() {
        let entries = serve_cli_runtime_entries(
            Some("int8"),
            Some(1024),
            Some(4096),
            Some(4096),
            Some(64),
            Some(2048),
            Some(12_345),
            Some(8),
            Some(16),
            Some(24),
            Some(true),
            Some(false),
            Some("memory"),
            Some(16),
            Some(1024),
            Some(&PathBuf::from("/tmp/profile.jsonl")),
            Some(&PathBuf::from("/tmp/scheduler-trace.jsonl")),
            Some("abc123"),
            Some("sha256:test"),
            Some("Qwen/Qwen3-30B-A3B-GPTQ-Int4"),
            Some(32),
            Some("{\"schema_version\":1}"),
            Some(crate::layer_split_pipeline::LayerSplitPipelineModeArg::Batch),
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
        assert_eq!(entry("FERRUM_MAX_MODEL_LEN").effective_value, "4096");
        assert_eq!(entry("FERRUM_KV_CAPACITY").effective_value, "1024");
        assert_eq!(entry("FERRUM_KV_MAX_BLOCKS").effective_value, "4096");
        assert_eq!(
            entry("FERRUM_KV_MAX_BLOCKS").source,
            RuntimeConfigSource::Cli
        );
        assert_eq!(entry("FERRUM_PAGED_MAX_SEQS").effective_value, "64");
        assert_eq!(entry("FERRUM_MAX_BATCHED_TOKENS").effective_value, "2048");
        assert_eq!(
            entry("FERRUM_RUNTIME_MEMORY_BUDGET_BYTES").effective_value,
            "12345"
        );
        assert!(entry("FERRUM_RUNTIME_MEMORY_BUDGET_BYTES")
            .affects
            .contains(&ferrum_types::RuntimeConfigEffect::Memory));
        assert!(entry("FERRUM_RUNTIME_MEMORY_BUDGET_BYTES")
            .affects
            .contains(&ferrum_types::RuntimeConfigEffect::Correctness));
        assert_eq!(
            entry("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE").effective_value,
            "8"
        );
        assert_eq!(
            entry("FERRUM_SCHED_PREFILL_STEP_CHUNK").effective_value,
            "16"
        );
        assert_eq!(
            entry("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK").effective_value,
            "24"
        );
        assert_eq!(entry("FERRUM_GREEDY_ARGMAX").effective_value, "1");
        assert_eq!(entry("FERRUM_PREFIX_CACHE").effective_value, "0");
        assert_eq!(entry("FERRUM_SESSION_CACHE").effective_value, "memory");
        assert_eq!(
            entry("FERRUM_SESSION_CACHE_MAX_ENTRIES").effective_value,
            "16"
        );
        assert_eq!(
            entry("FERRUM_SESSION_CACHE_MAX_TOKENS").effective_value,
            "1024"
        );
        assert_eq!(
            entry("FERRUM_MAX_MODEL_LEN").source,
            RuntimeConfigSource::Cli
        );
        assert_eq!(
            entry("FERRUM_PROFILE_JSONL").effective_value,
            "/tmp/profile.jsonl"
        );
        assert_eq!(
            entry("FERRUM_SCHEDULER_TRACE_JSONL").effective_value,
            "/tmp/scheduler-trace.jsonl"
        );
        assert_eq!(entry("FERRUM_PROFILE_ENTRYPOINT").effective_value, "serve");
        assert_eq!(
            entry("FERRUM_PROFILE_ENV_HASH").effective_value,
            "sha256:test"
        );
        assert_eq!(entry("FERRUM_PROFILE_CONCURRENCY").effective_value, "32");
        assert_eq!(
            entry(crate::layer_split_pipeline::LAYER_SPLIT_PIPELINE_MODE_KEY).effective_value,
            "batch"
        );
        assert!(entry("FERRUM_PROFILE_JSONL")
            .affects
            .contains(&ferrum_types::RuntimeConfigEffect::Diagnostics));
        assert!(entry("FERRUM_SCHEDULER_TRACE_JSONL")
            .affects
            .contains(&ferrum_types::RuntimeConfigEffect::Diagnostics));
    }

    #[test]
    fn cpu_serve_uses_contiguous_kv_cache() {
        assert!(matches!(
            serve_kv_cache_type_for_device(&ferrum_types::Device::CPU),
            ferrum_types::KvCacheType::Contiguous
        ));
    }

    #[cfg(any(all(target_os = "macos", feature = "metal"), feature = "cuda"))]
    #[test]
    fn accelerator_serve_uses_paged_kv_cache() {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let device = ferrum_types::Device::Metal;
        #[cfg(all(feature = "cuda", not(all(target_os = "macos", feature = "metal"))))]
        let device = ferrum_types::Device::CUDA(0);

        assert!(matches!(
            serve_kv_cache_type_for_device(&device),
            ferrum_types::KvCacheType::Paged
        ));
    }

    #[test]
    fn serve_runtime_snapshot_prefers_cli_over_config_file() {
        let config_entries = crate::config::RuntimeCliConfig {
            kv_dtype: Some("fp16".to_string()),
            ..Default::default()
        }
        .runtime_config_entries();
        let cli_entries = serve_cli_runtime_entries(
            Some("int8"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

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
    fn serve_runtime_snapshot_applies_recurrent_state_slots_to_engine_config() {
        let config_entries = crate::config::RuntimeCliConfig {
            recurrent_state_max_slots: Some(16),
            ..Default::default()
        }
        .runtime_config_entries();
        let snapshot = merge_runtime_config_sources(
            config_entries,
            RuntimeConfigSnapshot::default(),
            Vec::new(),
        );
        let entry = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_RECURRENT_STATE_MAX_SLOTS")
            .expect("missing recurrent state slot entry");
        let mut engine_config = ferrum_types::EngineConfig::default();

        engine_config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("serve runtime config should apply to engine config");

        assert_eq!(entry.effective_value, "16");
        assert_eq!(entry.source, RuntimeConfigSource::ConfigFile);
        assert_eq!(engine_config.runtime.recurrent_state_max_slots, Some(16));
    }

    #[test]
    fn vllm_compat_runtime_flags_follow_existing_precedence() {
        let config_entries = crate::config::RuntimeCliConfig {
            max_model_len: Some(1024),
            paged_max_seqs: Some(2),
            max_batched_tokens: Some(128),
            prefix_cache: Some(false),
            ..Default::default()
        }
        .runtime_config_entries();
        let env_snapshot = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new("FERRUM_MAX_MODEL_LEN", "2048", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_PAGED_MAX_SEQS", "4", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_MAX_BATCHED_TOKENS", "256", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_PREFIX_CACHE", "1", RuntimeConfigSource::Env),
        ]);

        let env_over_config =
            merge_runtime_config_sources(config_entries.clone(), env_snapshot.clone(), Vec::new());
        fn entry<'a>(snapshot: &'a RuntimeConfigSnapshot, key: &str) -> &'a RuntimeConfigEntry {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        }
        assert_eq!(
            entry(&env_over_config, "FERRUM_MAX_MODEL_LEN").effective_value,
            "2048"
        );
        assert_eq!(
            entry(&env_over_config, "FERRUM_PREFIX_CACHE").source,
            RuntimeConfigSource::Env
        );

        let cli_entries = serve_cli_runtime_entries(
            None,
            Some(1024),
            None,
            Some(4096),
            Some(8),
            Some(512),
            None,
            Some(8),
            Some(16),
            Some(32),
            Some(false),
            Some(false),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let cli_over_env = merge_runtime_config_sources(config_entries, env_snapshot, cli_entries);
        assert_eq!(
            entry(&cli_over_env, "FERRUM_MAX_MODEL_LEN").effective_value,
            "4096"
        );
        assert_eq!(
            entry(&cli_over_env, "FERRUM_KV_CAPACITY").effective_value,
            "1024"
        );
        assert_eq!(
            entry(&cli_over_env, "FERRUM_PAGED_MAX_SEQS").effective_value,
            "8"
        );
        assert_eq!(
            entry(&cli_over_env, "FERRUM_MAX_BATCHED_TOKENS").effective_value,
            "512"
        );
        assert_eq!(
            entry(&cli_over_env, "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK").effective_value,
            "32"
        );
        assert_eq!(
            entry(&cli_over_env, "FERRUM_PREFIX_CACHE").effective_value,
            "0"
        );
        assert_eq!(
            entry(&cli_over_env, "FERRUM_PREFIX_CACHE").source,
            RuntimeConfigSource::Cli
        );
    }

    #[test]
    fn autosize_snapshot_diff_marks_new_and_changed_values_as_memory_profile() {
        let before = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new("FERRUM_KV_MAX_BLOCKS", "2048", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_MOE_GRAPH", "1", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_KV_CAPACITY", "512", RuntimeConfigSource::Cli),
        ]);
        let after = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new("FERRUM_KV_MAX_BLOCKS", "2048", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_MOE_GRAPH", "1", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_KV_CAPACITY", "256", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new(
                "FERRUM_MAX_BATCHED_TOKENS",
                "2048",
                RuntimeConfigSource::Env,
            ),
            RuntimeConfigEntry::new("FERRUM_PAGED_MAX_SEQS", "32", RuntimeConfigSource::Env),
        ]);

        let entries = runtime_entries_changed_by_snapshot(
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

        assert_eq!(snapshot.entries.len(), 3);
        assert_eq!(
            entry("FERRUM_MAX_BATCHED_TOKENS").source,
            RuntimeConfigSource::MemoryProfile
        );
        assert_eq!(entry("FERRUM_PAGED_MAX_SEQS").effective_value, "32");
        assert_eq!(entry("FERRUM_KV_CAPACITY").effective_value, "256");
        assert_eq!(
            entry("FERRUM_KV_CAPACITY").source,
            RuntimeConfigSource::MemoryProfile
        );
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
        assert_eq!(entry("FERRUM_MOE_GRAPH").effective_value, "0");
        assert_eq!(entry("FERRUM_VLLM_MOE").effective_value, "1");
        assert_eq!(entry("FERRUM_VLLM_MOE_PAIR_IDS").effective_value, "1");
        assert_eq!(entry("FERRUM_KV_CAPACITY").effective_value, "512");
        assert_eq!(entry("FERRUM_PREFIX_CACHE").effective_value, "0");
        assert_eq!(entry("FERRUM_BACKEND").source, RuntimeConfigSource::Cli);
        assert_eq!(snapshot.entries.len(), 12);
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

        assert_eq!(entry("FERRUM_MOE_GRAPH").effective_value, "0");
        assert_eq!(entry("FERRUM_VLLM_MOE").effective_value, "1");
        assert_eq!(entry("FERRUM_KV_CAPACITY").effective_value, "512");
        assert_eq!(
            entry("FERRUM_MOE_GRAPH").source,
            RuntimeConfigSource::Default
        );
        assert_eq!(
            entry("FERRUM_VLLM_MOE").source,
            RuntimeConfigSource::Default
        );
    }

    fn qwen25_72b_gptq_definition() -> ferrum_models::ModelDefinition {
        let mut definition = ferrum_models::ModelDefinition {
            architecture: ferrum_models::Architecture::Qwen2,
            hidden_size: 8192,
            num_hidden_layers: 80,
            num_key_value_heads: Some(8),
            ..Default::default()
        };
        definition.extra_params = serde_json::json!({
            "quantization_config": {
                "bits": 4,
                "quant_method": "gptq"
            }
        });
        definition
    }

    fn qwen35_moe_reference_definition() -> ferrum_models::ModelDefinition {
        let raw = std::fs::read_to_string(format!(
            "{QWEN35_ARTIFACT_ROOT}/moe_shared_expert_reference.config.json"
        ))
        .expect("read Qwen3.5 MoE reference config");
        let root: serde_json::Value =
            serde_json::from_str(&raw).expect("parse Qwen3.5 MoE reference config");
        let text = root
            .get("text_config")
            .and_then(|value| value.as_object())
            .expect("Qwen3.5 reference config should include text_config");
        let mut extra_params = root
            .as_object()
            .expect("Qwen3.5 reference config root should be an object")
            .clone();
        for (key, value) in text {
            extra_params.insert(key.clone(), value.clone());
        }

        ferrum_models::ModelDefinition {
            architecture: ferrum_models::Architecture::Qwen35Moe,
            hidden_size: 2048,
            intermediate_size: 512,
            num_hidden_layers: 40,
            num_attention_heads: 16,
            num_key_value_heads: Some(2),
            max_position_embeddings: 262144,
            vocab_size: 248320,
            extra_params: serde_json::Value::Object(extra_params),
            ..Default::default()
        }
    }

    #[test]
    fn qwen35_moe_model_capabilities_preserve_moe_shape() {
        let definition = qwen35_moe_reference_definition();

        let capabilities = model_capabilities_from_definition_with_weight_bytes(&definition, None);

        assert_eq!(capabilities.architecture, "qwen3_5_moe");
        assert_eq!(capabilities.head_dim, Some(256));
        assert_eq!(capabilities.num_hidden_layers, Some(40));
        let moe = capabilities.moe.expect("Qwen3.5 MoE should be marked MoE");
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.experts_per_token, 8);
        assert_eq!(moe.moe_intermediate_size, Some(512));
        assert!(
            capabilities.estimated_weight_bytes.unwrap() > 14 * 1024 * 1024 * 1024,
            "Qwen3.5 MoE weight estimate must account for all resident experts, not only active params"
        );
        assert_eq!(
            capabilities.recurrent_state_bytes_per_sequence,
            Some(30 * (8192 * 3 + 32 * 128 * 128) * 4)
        );
    }

    #[test]
    fn qwen35_moe_cuda_model_capabilities_use_cuda_recurrent_state_dtype() {
        let definition = qwen35_moe_reference_definition();
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());

        let capabilities = model_capabilities_from_definition_with_weight_bytes_for_hardware(
            &definition,
            None,
            &hardware,
        );

        assert_eq!(
            capabilities.recurrent_state_bytes_per_sequence,
            Some(30 * (8192 * 3 + 32 * 128 * 128) * 2)
        );
    }

    #[test]
    fn model_capabilities_prefer_measured_weight_bytes_from_model_source() {
        let mut definition = ferrum_models::ModelDefinition {
            architecture: ferrum_models::Architecture::Qwen35Moe,
            hidden_size: 2048,
            intermediate_size: 512,
            num_hidden_layers: 40,
            num_attention_heads: 16,
            num_key_value_heads: Some(2),
            max_position_embeddings: 262144,
            ..Default::default()
        };
        definition.extra_params = serde_json::json!({
            "head_dim": 256,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "quantization_config": {
                "bits": 4,
                "quant_method": "gptq"
            }
        });

        let capabilities =
            model_capabilities_from_definition_with_weight_bytes(&definition, Some(19_123_456_789));

        assert_eq!(capabilities.estimated_weight_bytes, Some(19_123_456_789));
    }

    #[test]
    fn model_weight_bytes_from_path_sums_local_weight_files() {
        let dir = std::env::temp_dir().join(format!(
            "ferrum-weight-bytes-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp model dir");
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), vec![0u8; 7])
            .expect("write safetensors shard");
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), vec![0u8; 11])
            .expect("write safetensors shard");
        std::fs::write(dir.join("tokenizer.json"), vec![0u8; 101]).expect("write non-weight file");

        let result = model_weight_bytes_from_path(&dir);
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(result, Some(18));
    }

    fn two_gpu_layer_split_selection() -> crate::gpu_devices::GpuDeviceSelection {
        crate::gpu_devices::GpuDeviceSelection {
            raw_cli_value: "0,1".to_string(),
            requested_gpu_devices: vec![0, 1],
            selected_gpu_devices: vec![0, 1],
            cuda_device_count: 2,
            selected_distributed_strategy: "layer_split".to_string(),
            selected_layer_split_plan: Some(
                "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79".to_string(),
            ),
            selected_layer_split_stages: None,
        }
    }

    #[test]
    fn qwen25_layer_split_runtime_preset_entries_are_default_sourced() {
        let entries = runtime_preset_entries(
            QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET,
            RuntimeConfigSource::Default,
        )
        .unwrap();
        let snapshot = RuntimeConfigSnapshot::from_entries(entries);
        let entry = |key: &str| {
            snapshot
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };

        assert_eq!(
            entry(crate::layer_split_pipeline::LAYER_SPLIT_PIPELINE_MODE_KEY).effective_value,
            "batch"
        );
        assert_eq!(entry("FERRUM_MAX_MODEL_LEN").effective_value, "4096");
        assert_eq!(entry("FERRUM_KV_MAX_BLOCKS").effective_value, "1024");
        assert_eq!(entry("FERRUM_KV_CAPACITY").effective_value, "1024");
        assert_eq!(entry("FERRUM_PAGED_MAX_SEQS").effective_value, "16");
        assert_eq!(entry("FERRUM_MAX_BATCHED_TOKENS").effective_value, "1536");
        assert_eq!(
            entry("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE").effective_value,
            "16"
        );
        assert_eq!(
            entry("FERRUM_PAGED_MAX_SEQS").source,
            RuntimeConfigSource::Default
        );
    }

    #[test]
    fn runtime_preset_inference_uses_capability_rules() {
        let definition = qwen25_72b_gptq_definition();
        let selection = two_gpu_layer_split_selection();
        assert_eq!(
            infer_runtime_preset_for_startup(
                Some(ferrum_models::Architecture::Qwen2),
                Some(&definition),
                Some(&selection),
            ),
            Some(QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET)
        );

        let mut one_gpu = selection.clone();
        one_gpu.selected_gpu_devices = vec![0];
        one_gpu.selected_distributed_strategy = "single_gpu".to_string();
        assert_eq!(
            infer_runtime_preset_for_startup(
                Some(ferrum_models::Architecture::Qwen2),
                Some(&definition),
                Some(&one_gpu),
            ),
            None
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

        assert_eq!(entry("FERRUM_MOE_GRAPH").effective_value, "0");
        assert_eq!(
            entry("FERRUM_MOE_GRAPH").source,
            RuntimeConfigSource::Default
        );
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
        let cli_entries = serve_cli_runtime_entries(
            Some("bf16"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

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
    fn prefix_cache_vllm_and_product_aliases_resolve_identically() {
        assert_eq!(
            prefix_cache_cli_override(true, false, false, false),
            Some(true)
        );
        assert_eq!(
            prefix_cache_cli_override(false, false, true, false),
            Some(true)
        );
        assert_eq!(
            prefix_cache_cli_override(false, true, false, false),
            Some(false)
        );
        assert_eq!(
            prefix_cache_cli_override(false, false, false, true),
            Some(false)
        );

        let enabled_entries = serve_cli_runtime_entries(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            prefix_cache_cli_override(true, false, false, false),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let product_enabled_entries = serve_cli_runtime_entries(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            prefix_cache_cli_override(false, false, true, false),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let disabled_entries = serve_cli_runtime_entries(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            prefix_cache_cli_override(false, true, false, false),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let product_disabled_entries = serve_cli_runtime_entries(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            prefix_cache_cli_override(false, false, false, true),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(enabled_entries, product_enabled_entries);
        assert_eq!(disabled_entries, product_disabled_entries);
    }

    #[test]
    fn batched_graph_cli_override_records_flag_state() {
        assert_eq!(batched_graph_cli_override(true, false), Some(true));
        assert_eq!(batched_graph_cli_override(false, true), Some(false));
        assert_eq!(batched_graph_cli_override(false, false), None);
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
