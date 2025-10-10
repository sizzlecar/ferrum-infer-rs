//! Server command implementation

use crate::{config::CliConfig, output::OutputFormat};
use clap::Args;
use colored::*;
use ferrum_types::Result;
use ferrum_models::ModelSourceResolver;
use ferrum_server::{traits::HttpServer, types::ServerConfig, AxumServer};
use std::sync::Arc;
use tracing::info;

#[derive(Args)]
pub struct ServeCommand {
    /// Server host to bind
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Server port to bind
    #[arg(short, long, default_value = "8000")]
    pub port: u16,

    /// Number of worker threads
    #[arg(short, long)]
    pub workers: Option<usize>,

    /// Model to load on startup
    #[arg(short, long)]
    pub model: Option<String>,

    /// Enable hot reload
    #[arg(long)]
    pub hot_reload: bool,

    /// Enable development mode
    #[arg(long)]
    pub dev: bool,

    /// Backend to use (cpu, metal, cuda:0)
    #[arg(long, default_value = "auto")]
    pub backend: String,
}

pub async fn execute(cmd: ServeCommand, _config: CliConfig, _format: OutputFormat) -> Result<()> {
    println!("{} Starting Ferrum inference server...", "🚀".bright_blue());
    println!("Host: {}", cmd.host.cyan());
    println!("Port: {}", cmd.port.to_string().cyan());

    let model_name = cmd
        .model
        .unwrap_or_else(|| "TinyLlama-1.1B-Chat-v1.0".to_string());
    println!("Model: {}", model_name.cyan());

    if cmd.hot_reload {
        println!("{} Hot reload enabled", "🔥".yellow());
    }

    if cmd.dev {
        println!("{} Development mode enabled", "🛠️".yellow());
    }

    // Enhanced model resolution and loading
    println!(
        "{} Resolving model: {}",
        "🔍".bright_blue(),
        model_name.cyan()
    );

    let registry = ferrum_models::DefaultModelRegistry::with_defaults();
    let resolved_id = registry.resolve_model_id(&model_name);

    if resolved_id != model_name {
        println!(
            "{} Resolved alias '{}' to: {}",
            "🔗".bright_blue(),
            model_name.cyan(),
            resolved_id.yellow()
        );
    }

    // Create model source resolver
    let source_config = ferrum_models::ModelSourceConfig::default();
    let resolver = ferrum_models::DefaultModelSourceResolver::new(source_config);

    // Resolve model source
    let source = match resolver.resolve(&resolved_id, None).await {
        Ok(source) => {
            println!(
                "{} Model resolved: {}",
                "✅".bright_green(),
                source.local_path.display()
            );
            source
        }
        Err(e) => {
            println!(
                "{} Failed to resolve model '{}': {}",
                "❌".bright_red(),
                resolved_id,
                e
            );
            println!(
                "\nTip: Use 'ferrum models --download {}' to download the model",
                resolved_id
            );
            return Err(e);
        }
    };

    // Load model configuration
    println!("{} Loading model configuration...", "⚙️".bright_blue());
    let mut config_manager = ferrum_models::ConfigManager::new();
    let model_config = match config_manager.load_from_source(&source).await {
        Ok(config) => {
            println!(
                "{} Configuration loaded: {} ({})",
                "✅".bright_green(),
                format!("{:?}", config.architecture).yellow(),
                format!(
                    "{} vocab, {} layers",
                    config.vocab_size, config.num_hidden_layers
                )
                .cyan()
            );
            config
        }
        Err(e) => {
            println!(
                "{} Warning: Failed to load configuration, using defaults: {}",
                "⚠️".yellow(),
                e
            );
            // Use default configuration
            ferrum_models::ModelDefinition {
                architecture: ferrum_models::Architecture::Llama,
                hidden_size: 4096,
                intermediate_size: 11008,
                vocab_size: 32000,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                num_key_value_heads: None,
                max_position_embeddings: 2048,
                rope_theta: Some(10000.0),
                rope_scaling: None,
                norm_type: ferrum_models::NormType::RMSNorm,
                norm_eps: 1e-6,
                attention_config: ferrum_models::AttentionConfig {
                    attention_bias: false,
                    sliding_window: None,
                },
                activation: ferrum_models::Activation::SiLU,
                extra_params: serde_json::Value::Object(serde_json::Map::new()),
            }
        }
    };

    // Create engine configuration with model-aware settings
    let device = match cmd.backend.as_str() {
        "auto" => {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                if cfg!(feature = "metal") {
                    println!(
                        "{} Auto-detected Metal backend for Apple GPU",
                        "🔥".yellow()
                    );
                    ferrum_types::Device::Metal
                } else {
                    println!("{} Auto-detected CPU backend", "💻".blue());
                    ferrum_types::Device::CPU
                }
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                println!("{} Auto-detected CPU backend", "💻".blue());
                ferrum_types::Device::CPU
            }
        }
        "cpu" => ferrum_types::Device::CPU,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        "metal" => ferrum_types::Device::Metal,
        backend if backend.starts_with("cuda:") => {
            let device_id = backend[5..].parse().unwrap_or(0);
            ferrum_types::Device::CUDA(device_id)
        }
        backend => {
            println!("{} Using {} backend", "⚙️".blue(), backend.cyan());
            ferrum_types::Device::CPU
        }
    };
    
    let mut engine_config = ferrum_engine::simple_engine_config(resolved_id.clone(), device);

    // Initialize engine
    println!("{} Initializing inference engine...", "⚙️".yellow());
    let engine = ferrum_engine::create_mvp_engine(engine_config).await?;
    println!("{} Engine initialized successfully", "✅".green());

    // Create server configuration
    let server_config = ServerConfig {
        host: cmd.host.clone(),
        port: cmd.port,
        max_connections: 1000,
        request_timeout: std::time::Duration::from_secs(300), // 5 minutes
        keep_alive_timeout: std::time::Duration::from_secs(60),
        enable_tls: false,
        tls_cert_path: None,
        tls_key_path: None,
        cors: None,        // Simplified for MVP
        compression: None, // Simplified for MVP
        auth: None,        // No auth for MVP
        api_version: ferrum_server::types::ApiVersion::V1,
    };

    // Create and start server
    println!("{} Starting HTTP server...", "🌐".blue());
    let server = AxumServer::new(Arc::from(engine));

    // This will block until server shuts down
    server.start(&server_config).await?;

    Ok(())
}
