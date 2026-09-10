//! Run registered Rust contract assertions from a completed Cargo no-run build.
use clap::Parser;
use ferrum_bench_core::release_regression::contracts::{
    contract_groups, parse_compiler_artifacts, run_contract_checks, verify_contract_report,
    ContractRunOptions,
};
use ferrum_bench_core::release_regression::{backend_contracts, Backend};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(about = "Execute exact registered Rust contracts; a skipped or missing test cannot pass")]
struct Args {
    /// Run device correctness assertions instead of the shared CPU contract suite.
    #[arg(long, value_parser = ["cpu", "metal", "cuda"])]
    backend: Option<String>,
    /// JSONL from a completed cargo test --no-run --message-format=json build.
    #[arg(long)]
    artifacts: PathBuf,
    /// A new report file. Raw harness output is stored in an adjacent new .logs directory.
    #[arg(long)]
    output: PathBuf,
    /// Registered group IDs; omit to execute the full selected contract registry.
    #[arg(long, value_delimiter = ',')]
    group: Vec<String>,
    /// Deadline for each listing and assertion process, including process startup.
    #[arg(long, default_value = "120", value_parser = clap::value_parser!(u64).range(1..))]
    timeout_secs: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let backend = args.backend.as_deref().map(|backend| match backend {
        "cpu" => Backend::Cpu,
        "metal" => Backend::Metal,
        "cuda" => Backend::Cuda,
        _ => unreachable!("clap validates backend"),
    });
    let groups = backend.map_or_else(contract_groups, backend_contracts::contract_groups);
    let selected: BTreeSet<_> = args.group.iter().collect();
    if selected.len() != args.group.len() {
        return Err("duplicate --group selection".into());
    }
    for id in &selected {
        if !groups.iter().any(|group| &group.id == *id) {
            return Err(format!("unknown contract group {id}").into());
        }
    }
    let groups: Vec<_> = groups
        .into_iter()
        .filter(|group| selected.is_empty() || selected.contains(&group.id))
        .collect();
    let artifacts = parse_compiler_artifacts(&fs::read_to_string(&args.artifacts)?)?;
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let report = run_contract_checks(
        &artifacts,
        &groups,
        &ContractRunOptions {
            log_dir: args.output.with_extension("logs"),
            timeout: Duration::from_secs(args.timeout_secs),
        },
    )?;
    let mut writer = BufWriter::new(output);
    if let Some(backend) = backend {
        serde_json::to_writer_pretty(
            &mut writer,
            &backend_contracts::BackendContractReport {
                schema_version: 1,
                backend,
                execution: report.clone(),
            },
        )?;
    } else {
        serde_json::to_writer_pretty(&mut writer, &report)?;
    }
    writer.write_all(b"\n")?;
    writer.flush()?;
    verify_contract_report(&groups, &report).map_err(|issues| issues.join("; "))?;
    Ok(())
}
