//! Offline descriptive re-evaluation; no assertion of predeclared thresholds.
use clap::Parser;
use ferrum_bench_core::slo_offline::{
    read_bounded, reevaluate_json, write_new, MAX_CONFIG_BYTES, MAX_REPORT_BYTES,
};
use std::{path::PathBuf, process::ExitCode};

#[derive(Parser)]
#[command(
    about = "Recover last-visible client SLO metrics from retained benchmark request records (posthoc, not an original sidecar)"
)]
struct Args {
    /// Canonical BenchReport JSON, or an array of cells.
    #[arg(long)]
    report: PathBuf,
    /// Explicit SloClientVisibleConfig JSON. Prior declaration is not inferred.
    #[arg(long)]
    slo_client_config: PathBuf,
    /// New output file; existing files and aliases are rejected.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = (|| {
        let report = read_bounded(&args.report, MAX_REPORT_BYTES)?;
        let config = read_bounded(&args.slo_client_config, MAX_CONFIG_BYTES)?;
        let evaluation = reevaluate_json(&report, &config)?;
        write_new(&args.out, &evaluation)
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("offline SLO re-evaluation failed: {error}");
            ExitCode::FAILURE
        }
    }
}
