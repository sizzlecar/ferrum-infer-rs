//! Ferrum CLI - Ollama-style command line interface for LLM inference
//!
//! Commands:
//! - serve: Start the inference server
//! - run: Run a model and start interactive chat
//! - stop: Stop the running server
//! - pull: Download a model
//! - list: List downloaded models
//! - doctor: Inspect a binary and model source without downloading weights

use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use colored::*;
use ferrum_cli::{commands::*, config::CliConfig, utils::setup_logging};
use std::{future::Future, pin::Pin, process};

#[derive(Parser)]
#[command(name = "ferrum")]
#[command(about = "Ferrum - Fast LLM Inference Engine")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    long_about = "A high-performance LLM inference engine with Metal/CUDA acceleration.\n\nExamples:\n  ferrum doctor                                                   # Inspect this binary\n  ferrum run qwen3.5:4b-q4_k_m --disable-thinking                # Metal chat\n  ferrum run qwen3.5:4b --disable-thinking                       # CUDA chat\n  ferrum serve --model qwen3.5:4b --disable-thinking --port 8000 # OpenAI-compatible server\n  ferrum list                                                     # Show downloaded models"
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
    /// Run a model in interactive chat or with a one-shot prompt
    #[command(visible_alias = "r")]
    Run(run::RunCliCommand),

    /// Benchmark model throughput and latency
    #[command(hide = true)]
    Bench(bench::BenchCommand),

    /// HTTP serve-side bench with tokenizer-aware random prompts
    /// (apples-to-apples vs `vllm bench serve --dataset-name random`)
    #[command(hide = true)]
    BenchServe(bench_serve::BenchServeCommand),

    /// Benchmark image HTTP endpoints with a fixed prompt/seed/dimension dataset.
    BenchImages(bench_images::BenchImagesCommand),

    /// Prepare image FP8 weights once for reuse by run and serve (CPU only).
    PrepareImageWeights(prepare_image_weights::PrepareImageWeightsCommand),

    /// Validate and replay a request replay bundle without starting HTTP.
    #[command(hide = true)]
    ReplayBundle(replay_bundle::ReplayBundleCommand),

    /// Collect bitwise CUDA vNext evidence for the release model matrix.
    #[command(hide = true)]
    VnextDeterminism(vnext_determinism::VNextDeterminismCommand),

    /// Start the inference HTTP server
    Serve(serve::ServeCliCommand),

    /// Stop the running server
    Stop(stop::StopCommand),

    /// Download a model from HuggingFace Hub
    Pull(pull::PullCommand),

    /// List downloaded models
    #[command(visible_alias = "ls")]
    List(list::ListCommand),

    /// Inspect the binary, backend, cache, and a model source without downloading it
    Doctor(doctor::DoctorCommand),
}

#[tokio::main]
async fn main() {
    ferrum_cli::terminal::initialize();
    let definition = Cli::command();
    let matches = definition.clone().get_matches();
    if let Err(error) = image_product::validate_cli_matches(&definition, &matches) {
        eprintln!("{} {}", "Error:".red().bold(), error);
        process::exit(2);
    }
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());

    // Setup logging
    let suppress_chat_template_warnings = matches!(cli.command, Commands::Run(_)) && !cli.verbose;
    if let Err(e) = setup_logging(cli.verbose, false, suppress_chat_template_warnings) {
        eprintln!("{} Failed to setup logging: {}", "Error:".red().bold(), e);
        process::exit(1);
    }

    // Load the optional local configuration without creating files.
    let config = match CliConfig::load("ferrum.toml").await {
        Ok(config) => config,
        Err(e) => {
            // A missing optional file already resolves to defaults in load().
            // An explicit invalid configuration must never become a different
            // execution policy by silently falling back to those defaults.
            eprintln!("{} Config: {}", "Error:".red().bold(), e);
            process::exit(1);
        }
    };

    if let Err(e) = command_future(cli.command, config).await {
        if let Some(cleanup) = e.cleanup() {
            // This timeout bounds CLI observation only. The original service
            // retains its owner independently; unknown work is never released
            // or reported as retired because this wait ended.
            match tokio::time::timeout(std::time::Duration::from_secs(30), cleanup.wait()).await {
                Ok(report) => {
                    tracing::error!(phase = ?report.phase(), "failed construction resource retirement")
                }
                Err(_) => tracing::error!(
                    "failed construction retirement still pending; original custody retained"
                ),
            }
        }
        eprintln!("{} {}", "Error:".red().bold(), e);
        process::exit(1);
    }
}

// Construct command futures outside main's poll frame. In unoptimized builds,
// keeping every command inline adds large construction temporaries to the same
// stack as model initialization, exceeding Windows' default main-thread stack.
fn command_future(
    command: Commands,
    config: CliConfig,
) -> Pin<Box<dyn Future<Output = Result<(), image_product::CommandError>>>> {
    match command {
        Commands::Run(cmd) => Box::pin(run::execute_cli(cmd, config)),
        Commands::Bench(cmd) => {
            Box::pin(async move { bench::execute(cmd, config).await.map_err(Into::into) })
        }
        Commands::BenchServe(cmd) => {
            Box::pin(async move { bench_serve::execute(cmd, config).await.map_err(Into::into) })
        }
        Commands::BenchImages(cmd) => {
            Box::pin(async move { bench_images::execute(cmd).await.map_err(Into::into) })
        }
        Commands::PrepareImageWeights(cmd) => Box::pin(async move {
            tokio::task::spawn_blocking(move || prepare_image_weights::execute(cmd))
                .await
                .map_err(|error| image_product::ImageProductError::Configuration(error.into()))?
                .map_err(|error| image_product::ImageProductError::Configuration(error).into())
        }),
        Commands::ReplayBundle(cmd) => Box::pin(async move {
            replay_bundle::execute(cmd, config)
                .await
                .map_err(Into::into)
        }),
        Commands::VnextDeterminism(cmd) => {
            Box::pin(async move { vnext_determinism::execute(cmd).await.map_err(Into::into) })
        }
        Commands::Serve(cmd) => Box::pin(async move {
            let compatibility =
                ferrum_cli::config::load_interleaved_system_coalescing("ferrum.toml").await;
            match compatibility {
                Ok(configured) => serve::execute_cli(cmd, config, configured)
                    .await
                    .map_err(Into::into),
                Err(error) => Err(error.into()),
            }
        }),
        Commands::Stop(cmd) => {
            Box::pin(async move { stop::execute(cmd).await.map_err(Into::into) })
        }
        Commands::Pull(cmd) => {
            Box::pin(async move { pull::execute(cmd, config).await.map_err(Into::into) })
        }
        Commands::List(cmd) => {
            Box::pin(async move { list::execute(cmd, config).await.map_err(Into::into) })
        }
        Commands::Doctor(cmd) => {
            Box::pin(async move { doctor::execute(cmd, config).await.map_err(Into::into) })
        }
    }
}
