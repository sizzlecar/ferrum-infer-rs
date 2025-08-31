//! Server command implementation

use crate::{config::CliConfig, output::OutputFormat};
use clap::Args;
use colored::*;
use ferrum_core::{Device, Result};
use ferrum_engine::CandleBackend;
use ferrum_server::{AxumServer, ServerConfig};
use std::sync::Arc;
use tracing::{info, instrument};

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
        request_timeout_ms: 300000, // 5 minutes
        enable_cors: true,
        cors_origins: vec!["*".to_string()],
        enable_compression: true,
        compression_level: 6,
        enable_auth: false,
        auth_secret: None,
        enable_rate_limiting: false,
        rate_limit_requests_per_minute: 60,
        enable_metrics: true,
        metrics_path: "/metrics".to_string(),
        enable_health_check: true,
        health_check_path: "/health".to_string(),
        max_request_size_bytes: 10 * 1024 * 1024, // 10MB
        read_timeout_ms: 30000,
        write_timeout_ms: 30000,
        graceful_shutdown_timeout_ms: 30000,
    };

    // Create and start server
    println!("{} Starting HTTP server...", "🌐".blue());
    let server = AxumServer::new(engine);
    
    // This will block until server shuts down
    server.start(&server_config).await?;

    Ok(())
}
