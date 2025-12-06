//! Ferrum CLI - Ollama-style command line interface for LLM inference
//!
//! Commands:
//! - serve: Start the inference server
//! - run: Run a model and start interactive chat
//! - stop: Stop the running server
//! - pull: Download a model
//! - list: List downloaded models

use clap::{Parser, Subcommand};
use colored::*;
use ferrum_cli::{commands::*, config::CliConfig, utils::setup_logging};
use std::process;

#[derive(Parser)]
#[command(name = "ferrum")]
#[command(about = "Ferrum - Fast LLM Inference Engine")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    long_about = "A high-performance LLM inference engine with Metal/CUDA acceleration.\n\nExamples:\n  ferrum run tinyllama        # Start chat with TinyLlama\n  ferrum pull qwen2.5:7b      # Download Qwen 2.5 7B model\n  ferrum list                 # Show downloaded models\n  ferrum serve                # Start HTTP server"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a model and start interactive chat
    #[command(visible_alias = "r")]
    Run(run::RunCommand),

    /// Start the inference HTTP server
    Serve(serve::ServeCommand),

    /// Stop the running server
    Stop(stop::StopCommand),

    /// Download a model from HuggingFace Hub
    Pull(pull::PullCommand),

    /// List downloaded models
    #[command(visible_alias = "ls")]
    List(list::ListCommand),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Setup logging
    if let Err(e) = setup_logging(cli.verbose, false) {
        eprintln!("{} Failed to setup logging: {}", "Error:".red().bold(), e);
        process::exit(1);
    }

    // Load configuration (create default if not exists)
    let config = match CliConfig::load("ferrum.toml").await {
        Ok(config) => config,
        Err(e) => {
            if cli.verbose {
                eprintln!("{} Config: {}", "⚠️".yellow(), e);
            }
            CliConfig::default()
        }
    };

    // Execute command
    let result = match cli.command {
        Commands::Run(cmd) => run::execute(cmd, config).await,
        Commands::Serve(cmd) => serve::execute(cmd, config).await,
        Commands::Stop(cmd) => stop::execute(cmd).await,
        Commands::Pull(cmd) => pull::execute(cmd, config).await,
        Commands::List(cmd) => list::execute(cmd, config).await,
    };

    if let Err(e) = result {
        eprintln!("{} {}", "Error:".red().bold(), e);
        process::exit(1);
    }
}
