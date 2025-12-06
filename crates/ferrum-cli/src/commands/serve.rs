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
    #[arg(short, long)]
    pub model: Option<String>,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "11434")]
    pub port: u16,
}

pub async fn execute(cmd: ServeCommand, config: CliConfig) -> Result<()> {
    // Print banner
    print_banner();

    // Resolve model
    let model_name = cmd
        .model
        .or(config.models.default_model.clone())
        .unwrap_or_else(|| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    let model_id = resolve_model_alias(&model_name);
    println!("{} {}", "Model:".dimmed(), model_id.cyan());

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

    // Create engine
    println!();
    println!("{}", "Initializing engine...".dimmed());
    let engine_config = ferrum_engine::simple_engine_config(model_id.clone(), device);
    let engine = ferrum_engine::create_mvp_engine(engine_config).await?;
    // Convert Box<dyn InferenceEngine> to Arc<dyn InferenceEngine>
    let engine: Arc<dyn InferenceEngine + Send + Sync> = Arc::from(engine);

    // Create server config
    let server_config = ServerConfig {
        host: cmd.host.clone(),
        port: cmd.port,
        ..Default::default()
    };

    // Create server with engine
    let server = AxumServer::new(engine);

    println!();
    println!(
        "{} {} {}",
        "🚀".green(),
        "Server running at".green().bold(),
        format!("http://{}:{}", cmd.host, cmd.port).cyan().bold()
    );
    println!();
    println!("Endpoints:");
    println!("  POST /v1/chat/completions  - OpenAI-compatible chat");
    println!("  GET  /v1/models            - List models");
    println!("  GET  /health               - Health check");
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
    println!(
        "{}",
        " |  __/ _ \\ '__| '__| | | | '_ ` _ \\ ".bright_red()
    );
    println!("{}", " | | |  __/ |  | |  | |_| | | | | | ".bright_red());
    println!(
        "{}",
        " |_|  \\___|_|  |_|   \\__,_|_| |_| |_|".bright_red()
    );
    println!();
    println!(
        "   {}",
        "🦀 Rust LLM Inference Server".bright_cyan().bold()
    );
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
        "llama3.2:1b" => "meta-llama/Llama-3.2-1B-Instruct".to_string(),
        "llama3.2:3b" => "meta-llama/Llama-3.2-3B-Instruct".to_string(),
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

    #[allow(unreachable_code)]
    ferrum_types::Device::CPU
}
