//! Summarize completed product native timing traces without duplicating shared work.
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

#[path = "native_profile/summary.rs"]
mod summary;

#[derive(Parser)]
#[command(about = "Summarize physical native commands and their separate subwork intervals")]
struct Args {
    /// Completed product profile JSONL. Partial or malformed JSON lines are errors.
    #[arg(long)]
    input: PathBuf,
    /// New JSON report; existing evidence is never overwritten.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = File::open(&args.input).context("open input profile")?;
    let report = summary::summarize(BufReader::new(input))?;
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .context("create new summary")?;
    let mut writer = BufWriter::new(output);
    serde_json::to_writer_pretty(
        &mut writer,
        &serde_json::json!({"input": args.input, "summary": report}),
    )?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
