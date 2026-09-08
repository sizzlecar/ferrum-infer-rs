//! Ferrum CLI - Ollama-style command line interface for LLM inference
//!
//! Commands:
//! - serve: Start the inference server
//! - run: Run a model and start interactive chat
//! - stop: Stop the running server
//! - pull: Download a model
//! - list: List downloaded models
//! - doctor: Inspect a binary and model source without downloading weights

use clap::{Parser, Subcommand};
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
    Run(run::RunCommand),

    /// Benchmark model throughput and latency
    #[command(hide = true)]
    Bench(bench::BenchCommand),

    /// HTTP serve-side bench with tokenizer-aware random prompts
    /// (apples-to-apples vs `vllm bench serve --dataset-name random`)
    #[command(hide = true)]
    BenchServe(bench_serve::BenchServeCommand),

    /// Validate and replay a request replay bundle without starting HTTP.
    #[command(hide = true)]
    ReplayBundle(replay_bundle::ReplayBundleCommand),

    /// Collect bitwise CUDA vNext evidence for the release model matrix.
    #[command(hide = true)]
    VnextDeterminism(vnext_determinism::VNextDeterminismCommand),

    /// Generate text embeddings using BERT models
    #[command(visible_alias = "e", hide = true)]
    Embed(embed::EmbedCommand),

    /// Transcribe audio files using Whisper models
    #[command(visible_alias = "t", hide = true)]
    Transcribe(transcribe::TranscribeCommand),

    /// Text-to-speech synthesis using Qwen3-TTS models
    #[command(hide = true)]
    Tts(tts::TtsCommand),

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
    let cli = Cli::parse();

    // Setup logging
    let suppress_chat_template_warnings = matches!(cli.command, Commands::Run(_)) && !cli.verbose;
    if let Err(e) = setup_logging(cli.verbose, false, suppress_chat_template_warnings) {
        eprintln!("{} Failed to setup logging: {}", "Error:".red().bold(), e);
        process::exit(1);
    }

    // Load the optional local configuration without creating files.
    let (config, config_loaded) = match CliConfig::load("ferrum.toml").await {
        Ok(config) => (config, true),
        Err(e) => {
            if cli.verbose {
                eprintln!("{} Config: {}", "⚠️".yellow(), e);
            }
            (CliConfig::default(), false)
        }
    };

    if let Err(e) = command_future(cli.command, config, config_loaded).await {
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
    config_loaded: bool,
) -> Pin<Box<dyn Future<Output = ferrum_types::Result<()>>>> {
    match command {
        Commands::Run(cmd) => Box::pin(run::execute(cmd, config)),
        Commands::Bench(cmd) => Box::pin(bench::execute(cmd, config)),
        Commands::BenchServe(cmd) => Box::pin(bench_serve::execute(cmd, config)),
        Commands::ReplayBundle(cmd) => Box::pin(replay_bundle::execute(cmd, config)),
        Commands::VnextDeterminism(cmd) => Box::pin(vnext_determinism::execute(cmd)),
        Commands::Embed(cmd) => Box::pin(embed::execute(cmd, config)),
        Commands::Transcribe(cmd) => Box::pin(transcribe::execute(cmd, config)),
        Commands::Tts(cmd) => Box::pin(tts::execute(cmd, config)),
        Commands::Serve(cmd) => Box::pin(async move {
            let compatibility = if config_loaded {
                ferrum_cli::config::load_interleaved_system_coalescing("ferrum.toml").await
            } else {
                Ok(true)
            };
            match compatibility {
                Ok(configured) => serve::execute_cli(cmd, config, configured).await,
                Err(error) => Err(error),
            }
        }),
        Commands::Stop(cmd) => Box::pin(stop::execute(cmd)),
        Commands::Pull(cmd) => Box::pin(pull::execute(cmd, config)),
        Commands::List(cmd) => Box::pin(list::execute(cmd, config)),
        Commands::Doctor(cmd) => Box::pin(doctor::execute(cmd, config)),
    }
}
