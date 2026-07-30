use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ferrum_native_ops_builder::{
    assemble_native_operator_set, lock_native_operator_source_definition,
    materialize_native_operator_package_spec, package_native_operator,
    run_native_operator_source_build, NativeOperatorPackageRequest,
    NativeOperatorPackageSpecRequest, NativeOperatorSetRequest, NativeOperatorSourceBuildRequest,
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
    LockSource {
        #[arg(long)]
        definition: PathBuf,
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    SourceBuild {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        compute_capability: String,
        #[arg(long)]
        builder_sha: String,
        #[arg(long)]
        nvcc: PathBuf,
        #[arg(long)]
        cuda_toolkit_root: PathBuf,
        #[arg(long)]
        ccbin: PathBuf,
        #[arg(long)]
        ar: PathBuf,
        #[arg(long, default_value_t = 4)]
        nvcc_threads: u32,
        #[arg(long)]
        object_cache: PathBuf,
        #[arg(long)]
        plan_only: bool,
        #[arg(long)]
        out: PathBuf,
    },
    MaterializePackageSpec {
        #[arg(long)]
        definition: PathBuf,
        #[arg(long)]
        g03_catalog: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Package {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        license_root: PathBuf,
        #[arg(long)]
        source_build_receipt: PathBuf,
        #[arg(long)]
        source_build_plan: PathBuf,
        #[arg(long)]
        g03_catalog: PathBuf,
        #[arg(long)]
        abi_contract: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        cc: PathBuf,
        #[arg(long)]
        ar: PathBuf,
    },
    AssembleSet {
        #[arg(long, required = true)]
        receipt: Vec<PathBuf>,
        #[arg(long = "receipt-sha256", required = true)]
        receipt_sha256: Vec<String>,
        #[arg(long)]
        g03_catalog_sha256: String,
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
        Command::LockSource {
            definition,
            source_root,
            out,
        } => {
            lock_native_operator_source_definition(&definition, &source_root, &out)?;
            println!("FERRUM NATIVE SOURCE LOCK READY: {}", out.display());
        }
        Command::SourceBuild {
            plan,
            source_root,
            compute_capability,
            builder_sha,
            nvcc,
            cuda_toolkit_root,
            ccbin,
            ar,
            nvcc_threads,
            object_cache,
            plan_only,
            out,
        } => {
            run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
                plan_path: plan,
                source_root,
                output_dir: out.clone(),
                compute_capability,
                builder_sha,
                nvcc_path: nvcc,
                cuda_toolkit_root,
                ccbin_path: ccbin,
                ar_path: ar,
                nvcc_threads,
                object_cache_dir: object_cache,
                plan_only,
            })?;
            let kind = if plan_only { "PLAN" } else { "BUILD" };
            println!(
                "FERRUM NATIVE SOURCE {kind} READY: {}",
                out.join("source-build.receipt.json").display()
            );
        }
        Command::MaterializePackageSpec {
            definition,
            g03_catalog,
            out,
        } => {
            materialize_native_operator_package_spec(&NativeOperatorPackageSpecRequest {
                definition_path: definition,
                g03_catalog_path: g03_catalog,
                output_path: out.clone(),
            })?;
            println!("FERRUM NATIVE PACKAGE SPEC READY: {}", out.display());
        }
        Command::Package {
            spec,
            source_root,
            license_root,
            source_build_receipt,
            source_build_plan,
            g03_catalog,
            abi_contract,
            out,
            cc,
            ar,
        } => {
            package_native_operator(&NativeOperatorPackageRequest {
                spec_path: spec,
                source_root,
                license_root,
                source_build_receipt_path: source_build_receipt,
                source_build_plan_path: source_build_plan,
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
            receipt_sha256,
            g03_catalog_sha256,
            compute_capability,
            out,
        } => {
            assemble_native_operator_set(&NativeOperatorSetRequest {
                receipt_paths: receipt,
                expected_receipt_sha256: receipt_sha256,
                expected_g03_catalog_sha256: g03_catalog_sha256,
                output_lock_path: out.clone(),
                compute_capability,
            })?;
            println!("FERRUM NATIVE OPERATOR SET READY: {}", out.display());
        }
    }
    Ok(())
}
