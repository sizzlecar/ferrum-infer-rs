//! A fresh process per image: CLI wall time, never warm HTTP serving throughput.
use clap::Parser;
use std::{path::PathBuf, process::ExitCode};

#[cfg(unix)]
#[path = "image_process_bench/config.rs"]
mod config;
#[cfg(unix)]
#[path = "../process_group.rs"]
mod process_group;
#[cfg(unix)]
#[path = "image_process_bench/runner.rs"]
mod runner;
#[cfg(all(test, unix))]
#[path = "image_process_bench/tests.rs"]
mod tests;

#[derive(Clone, Debug, Parser)]
#[command(about = "Benchmark bounded fresh CLI image processes, not a warm model server")]
struct Args {
    /// JSON: program/SHA/cwd/args/env/inputs and model/precision declaration.
    #[arg(long)]
    command_template: PathBuf,
    /// Existing ImageDataset JSON; every repetition uses the same ordered samples.
    #[arg(long)]
    dataset: PathBuf,
    /// Must not exist. Raw stdout/stderr, PNGs and request records are retained.
    #[arg(long)]
    report_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    #[arg(long, default_value_t = 2)]
    repetitions: usize,
    #[arg(long, default_value_t = 1)]
    warmup_repetitions: usize,
    #[arg(long, default_value_t = 1800)]
    timeout_seconds: u64,
    /// On cancellation/timeout, allow TERM this long before owned-group cleanup.
    #[arg(long, default_value_t = 10)]
    termination_grace_seconds: u64,
    #[arg(long, default_value_t = 1000)]
    maximum_total_requests: usize,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    #[cfg(not(unix))]
    {
        let _ = args;
        eprintln!("image_process_bench requires Unix process-group ownership and cleanup");
        ExitCode::FAILURE
    }
    #[cfg(unix)]
    {
        match runner::execute(args).await {
            Ok(report) if report.succeeded() => ExitCode::SUCCESS,
            Ok(_) => {
                eprintln!("image process benchmark failed; raw records and report.json retained");
                ExitCode::FAILURE
            }
            Err(error) => {
                eprintln!("image process benchmark: {error:#}");
                ExitCode::FAILURE
            }
        }
    }
}
