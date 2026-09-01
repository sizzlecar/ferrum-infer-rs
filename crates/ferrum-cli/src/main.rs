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
use std::process;

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
    Serve(serve::ServeCommand),

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
        Commands::Bench(cmd) => bench::execute(cmd, config).await,
        Commands::BenchServe(cmd) => bench_serve::execute(cmd, config).await,
        Commands::ReplayBundle(cmd) => replay_bundle::execute(cmd, config).await,
        Commands::VnextDeterminism(cmd) => vnext_determinism::execute(cmd).await,
        Commands::Embed(cmd) => embed::execute(cmd, config).await,
        Commands::Transcribe(cmd) => transcribe::execute(cmd, config).await,
        Commands::Tts(cmd) => tts::execute(cmd, config).await,
        Commands::Serve(cmd) => serve::execute(cmd, config).await,
        Commands::Stop(cmd) => stop::execute(cmd).await,
        Commands::Pull(cmd) => pull::execute(cmd, config).await,
        Commands::List(cmd) => list::execute(cmd, config).await,
        Commands::Doctor(cmd) => doctor::execute(cmd, config).await,
    };

    if let Err(e) = result {
        eprintln!("{} {}", "Error:".red().bold(), e);
        process::exit(1);
    }
}
