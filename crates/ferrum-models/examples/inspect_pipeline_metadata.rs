//! Inspect a local component-pipeline metadata package without opening weights.
//!
//! Usage: cargo run -p ferrum-models --example inspect_pipeline_metadata -- DIRECTORY

use std::io::Write;
use std::path::PathBuf;

use ferrum_models::pipeline_source::{PipelineSourceBundle, PipelineSourceLimits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let directory = arguments.next().ok_or(
        "usage: inspect_pipeline_metadata DIRECTORY (metadata preflight only; no weights opened)",
    )?;
    if directory == "--help" || directory == "-h" {
        println!("usage: inspect_pipeline_metadata DIRECTORY\nMetadata preflight only; no weights are opened or validated.");
        return Ok(());
    }
    if arguments.next().is_some() {
        return Err("expected exactly one local metadata directory".into());
    }

    let bundle =
        PipelineSourceBundle::open(PathBuf::from(directory), PipelineSourceLimits::default())?;
    let mut output = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut output, bundle.manifest())?;
    writeln!(output)?;
    Ok(())
}
