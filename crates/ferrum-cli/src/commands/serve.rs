//! Serve command - Start the HTTP inference server

use crate::config::CliConfig;
use clap::Args;
use colored::*;
use ferrum_interfaces::InferenceEngine;
use ferrum_models::source::ModelFormat;
use ferrum_server::{AxumServer, HttpServer, ServerConfig};
use ferrum_types::Result;
use std::path::PathBuf;
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
}

pub async fn execute(cmd: ServeCommand, config: CliConfig) -> Result<()> {
    let ServeCommand {
        model,
        model_option,
        host,
        port,
    } = cmd;

    // Print banner
    print_banner();

    // Resolve model
    let model_name = model
        .or(model_option)
        .or(config.models.default_model.clone())
        .unwrap_or_else(|| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    let model_id = resolve_model_alias(&model_name);
    println!("{} {}", "Model:".dimmed(), model_id.cyan());

    let host = host.unwrap_or_else(|| config.server.host.clone());
    let port = port.unwrap_or(config.server.port);

    // Find cached model
    let cache_dir = get_hf_cache_dir(&config);
    let source = match find_cached_model(&cache_dir, &model_id) {
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
    };

    // Set model path for engine
    std::env::set_var(
        "FERRUM_MODEL_PATH",
        source.local_path.to_string_lossy().to_string(),
    );

    // Select device
    let device = select_device();
    println!("{} {:?}", "Device:".dimmed(), device);

    // Detect architecture to choose engine type
    println!();
    let mut config_manager = ferrum_models::ConfigManager::new();
    let model_def = config_manager.load_from_path(&source.local_path).await?;

    let engine: Arc<dyn InferenceEngine + Send + Sync> = match model_def.architecture {
        ferrum_models::Architecture::Clip => {
            println!("{}", "Initializing CLIP embedding engine...".dimmed());
            let candle_device = candle_core::Device::Cpu;
            let executor = ferrum_models::ClipModelExecutor::from_path(
                &source.local_path.to_string_lossy(),
                candle_device,
                candle_core::DType::F32,
            )?;
            let tokenizer = crate::commands::embed::load_tokenizer(&source.local_path)?;
            let engine_config = ferrum_engine::simple_engine_config(model_id.clone(), device);
            Arc::new(
                ferrum_engine::embedding_engine::EmbeddingEngine::new(executor, engine_config)
                    .with_tokenizer(tokenizer),
            )
        }
        ferrum_models::Architecture::Whisper => {
            println!("{}", "Initializing Whisper ASR engine...".dimmed());
            let candle_device = to_candle_device(&device);
            let executor = ferrum_models::WhisperModelExecutor::from_path(
                &source.local_path.to_string_lossy(),
                candle_device,
                candle_core::DType::F32,
            )?;
            let engine_config = ferrum_engine::simple_engine_config(model_id.clone(), device);
            Arc::new(
                ferrum_engine::transcription_engine::TranscriptionEngine::new(
                    executor,
                    engine_config,
                ),
            )
        }
        ferrum_models::Architecture::Qwen3TTS => {
            println!("{}", "Initializing Qwen3-TTS engine...".dimmed());
            let candle_device = to_candle_device(&device);
            let executor = ferrum_models::TtsModelExecutor::from_path(
                &source.local_path.to_string_lossy(),
                candle_device,
                candle_core::DType::F32,
            )?;
            Arc::new(ferrum_engine::tts_engine::TtsEngine::new(
                executor,
                ferrum_types::ModelId(model_id.clone()),
            ))
        }
        _ => {
            println!(
                "{}",
                "Initializing engine (continuous batching)...".dimmed()
            );
            let mut engine_config = ferrum_engine::simple_engine_config(model_id.clone(), device);
            engine_config.scheduler.policy = ferrum_types::SchedulingPolicy::ContinuousBatch;
            engine_config.kv_cache.cache_type = ferrum_types::KvCacheType::Paged;
            let engine = ferrum_engine::create_mvp_engine(engine_config).await?;
            Arc::from(engine)
        }
    };

    // Create server config
    let server_config = ServerConfig {
        host: host.clone(),
        port,
        ..Default::default()
    };

    // Create server with engine
    let server = AxumServer::new(engine);

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

fn find_cached_model(
    cache_dir: &PathBuf,
    model_id: &str,
) -> Option<ferrum_models::source::ResolvedModelSource> {
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
                    return Some(ferrum_models::source::ResolvedModelSource {
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
                    return Some(ferrum_models::source::ResolvedModelSource {
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

fn detect_format(path: &PathBuf) -> ModelFormat {
    if path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists()
    {
        ModelFormat::SafeTensors
    } else if path.join("pytorch_model.bin").exists() {
        ModelFormat::PyTorchBin
    } else {
        ModelFormat::Unknown
    }
}

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
