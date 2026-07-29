use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ferrum_native_ops_builder::{
    assemble_native_operator_set, package_native_operator, NativeOperatorPackageRequest,
    NativeOperatorSetRequest,
};

#[derive(Debug, Parser)]
#[command(name = "ferrum-native-ops-builder")]
#[command(about = "Build-boundary tooling for Ferrum native operator artifacts")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Package {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        input_archive: PathBuf,
        #[arg(long)]
        g03_catalog: PathBuf,
        #[arg(long)]
        abi_contract: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "cc")]
        cc: String,
        #[arg(long, default_value = "ar")]
        ar: String,
    },
    AssembleSet {
        #[arg(long, required = true)]
        receipt: Vec<PathBuf>,
        #[arg(long)]
        compute_capability: String,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("ferrum-native-ops-builder: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> ferrum_native_ops_builder::Result<()> {
    match cli.command {
        Command::Package {
            spec,
            source_root,
            input_archive,
            g03_catalog,
            abi_contract,
            out,
            cc,
            ar,
        } => {
            package_native_operator(&NativeOperatorPackageRequest {
                spec_path: spec,
                source_root,
                input_archive,
                g03_catalog_path: g03_catalog,
                abi_contract_path: abi_contract,
                output_dir: out.clone(),
                cc,
                ar,
            })?;
            println!(
                "FERRUM NATIVE OPERATOR PACKAGE READY: {}",
                out.join("package.receipt.json").display()
            );
        }
        Command::AssembleSet {
            receipt,
            compute_capability,
            out,
        } => {
            assemble_native_operator_set(&NativeOperatorSetRequest {
                receipt_paths: receipt,
                output_lock_path: out.clone(),
                compute_capability,
            })?;
            println!("FERRUM NATIVE OPERATOR SET READY: {}", out.display());
        }
    }
    Ok(())
}
