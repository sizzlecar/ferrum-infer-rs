//! Ferrum CLI - Command line interface for LLM inference
//!
//! This binary provides various commands for testing, benchmarking,
//! and managing the Ferrum inference framework.

use clap::{Parser, Subcommand};
use colored::*;
use ferrum_cli::{
    commands::*,
    config::CliConfig,
    output::OutputFormat as CliOutputFormat,
    utils::{print_banner, setup_logging},
};
use std::process;

#[derive(Parser)]
#[command(name = "ferrum")]
#[command(about = "Ferrum LLM Inference CLI")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(long_about = "A command-line interface for the Ferrum LLM inference framework")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Configuration file path
    #[arg(short, long, default_value = "ferrum.toml")]
    config: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Quiet mode (only errors)
    #[arg(short, long)]
    quiet: bool,

    /// Output format
    #[arg(long, default_value = "pretty")]
    format: CliOutputFormat,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve(ServeCommand),

    /// Test inference with a model
    Infer(InferCommand),

    /// List available models
    Models(ModelsCommand),

    /// Run benchmarks
    Benchmark(BenchmarkCommand),

    /// Validate configuration
    Config(ConfigCommand),

    /// Health check operations
    Health(HealthCommand),

    /// Cache management operations
    Cache(CacheCommand),

    /// Development and debugging tools
    Dev(DevCommand),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Setup logging based on verbosity
    setup_logging(cli.verbose, cli.quiet).unwrap_or_else(|e| {
        eprintln!("{} Failed to setup logging: {}", "Error:".red().bold(), e);
        process::exit(1);
    });

    // Load configuration
    let config = match CliConfig::load(&cli.config).await {
        Ok(config) => config,
        Err(e) => {
            eprintln!("{} Failed to load config: {}", "Error:".red().bold(), e);
            process::exit(1);
        }
    };

    // Print banner for interactive commands
    if !cli.quiet && matches!(cli.command, Commands::Serve(_) | Commands::Dev(_)) {
        print_banner();
    }

    // Execute command
    let result = match cli.command {
        Commands::Serve(cmd) => serve::execute(cmd, config, cli.format).await,
        Commands::Infer(cmd) => infer::execute(cmd, config, cli.format).await,
        Commands::Models(cmd) => models::execute(cmd, config, cli.format).await,
        Commands::Benchmark(cmd) => benchmark::execute(cmd, config, cli.format).await,
        Commands::Config(cmd) => config_cmd::execute(cmd, config, cli.format).await,
        Commands::Health(cmd) => health::execute(cmd, config, cli.format).await,
        Commands::Cache(cmd) => cache::execute(cmd, config, cli.format).await,
        Commands::Dev(cmd) => dev::execute(cmd, config, cli.format).await,
    };

    // Handle result
    match result {
        Ok(_) => {
            if !cli.quiet {
                println!("{}", "✅ Command completed successfully".green().bold());
            }
        }
        Err(e) => {
            eprintln!("{} {}", "Error:".red().bold(), e);
            process::exit(1);
        }
    }
}
