//! Server command implementation

use crate::{config::CliConfig, output::OutputFormat};
use clap::Args;
use colored::*;
use ferrum_core::Result;
use ferrum_server::{traits::HttpServer, AxumServer, types::ServerConfig};
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
}

pub async fn execute(cmd: ServeCommand, _config: CliConfig, _format: OutputFormat) -> Result<()> {
    println!("{} Starting Ferrum inference server...", "🚀".bright_blue());
    println!("Host: {}", cmd.host.cyan());
    println!("Port: {}", cmd.port.to_string().cyan());

    let model_id = cmd.model.unwrap_or_else(|| "TinyLlama-1.1B-Chat-v1.0".to_string());
    println!("Model: {}", model_id.cyan());

    if cmd.hot_reload {
        println!("{} Hot reload enabled", "🔥".yellow());
    }

    if cmd.dev {
        println!("{} Development mode enabled", "🛠️".yellow());
    }

    // Create engine configuration
    let engine_config = ferrum_engine::EngineConfig {
        max_batch_size: 32,
        max_sequence_length: 2048,
        num_gpu_blocks: 512,
        block_size: 16,
        enable_continuous_batching: false, // Simplified for MVP
        enable_prefix_caching: false,
        gpu_memory_fraction: 0.9,
        scheduling_interval_ms: 10,
        model_id: model_id.clone(),
        device: "cpu".to_string(), // Use CPU for MVP
    };

    // Initialize engine
    println!("{} Initializing inference engine...", "⚙️".yellow());
    let engine = Arc::new(ferrum_engine::create_mvp_engine(engine_config).await?);
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
        cors: None, // Simplified for MVP
        compression: None, // Simplified for MVP
        auth: None, // No auth for MVP
        api_version: ferrum_server::types::ApiVersion::V1,
    };

    // Create and start server
    println!("{} Starting HTTP server...", "🌐".blue());
    let server = AxumServer::new(engine);
    
    // This will block until server shuts down
    server.start(&server_config).await?;

    Ok(())
}
