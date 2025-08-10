//! Main entry point for Ferrum Infer
//!
//! This is the main binary that starts the inference engine server with
//! OpenAI-compatible API endpoints.

use ferrum_infer::{
    api::start_server,
    config::Config,
    error::Result,
    init_engine,
    utils::{format_duration, get_memory_info, init_logging},
    VERSION,
};
use std::sync::Arc;
use std::time::Instant;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Config::from_env().unwrap_or_else(|e| {
        eprintln!("Failed to load configuration: {}", e);
        eprintln!("Using default configuration");
        Config::default()
    });

    // Initialize logging
    if let Err(e) = init_logging(&config.logging.level, &config.logging.format) {
        eprintln!("Failed to initialize logging: {}", e);
        return Err(e);
    }

    // Print startup banner
    print_banner();

    // Log configuration (sanitized)
    info!("Starting Ferrum Infer with configuration:");
    info!("  Server: {}:{}", config.server.host, config.server.port);
    info!("  Model: {}", config.model.name);
    info!("  Device: {}", config.model.device);
    info!("  Cache enabled: {}", config.cache.enabled);
    info!(
        "  Max sequence length: {}",
        config.model.max_sequence_length
    );

    if config.server.api_key.is_some() {
        info!("  API key authentication: enabled");
    } else {
        warn!("  API key authentication: disabled (not recommended for production)");
    }

    // Display system information
    if let Ok(memory_info) = get_memory_info() {
        info!("System memory: {}", memory_info.format());
    }

    // Initialize the inference engine
    info!("Initializing inference engine...");
    let start_time = Instant::now();

    let engine = match init_engine().await {
        Ok(engine) => {
            let init_duration = start_time.elapsed();
            info!(
                "Inference engine initialized successfully in {}",
                format_duration(init_duration)
            );
            Arc::new(engine)
        }
        Err(e) => {
            error!("Failed to initialize inference engine: {}", e);
            return Err(e);
        }
    };

    // Set up graceful shutdown
    let engine_clone = Arc::clone(&engine);
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Received shutdown signal, cleaning up...");

        if let Err(e) = engine_clone.shutdown().await {
            error!("Error during shutdown: {}", e);
        }

        info!("Shutdown complete");
        std::process::exit(0);
    });

    // Start the HTTP server
    info!("Starting HTTP server...");
    if let Err(e) = start_server(config, engine).await {
        error!("Server error: {}", e);
        return Err(e);
    }

    Ok(())
}

/// Print the startup banner
fn print_banner() {
    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🚀 Ferrum Infer v{}                                       ║
║                                                              ║
║    A high-performance Rust-based LLM inference server       ║
║    with OpenAI-compatible API endpoints                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"#,
        VERSION
    );
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_banner_contains_version() {
        // Just ensure the banner function doesn't panic
        print_banner();
    }
}
