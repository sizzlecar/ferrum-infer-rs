//! Prepare coordinated versions and describe staged assets. This tool does not
//! publish packages, create releases or authorize unexecuted runtime checks.
#[path = "release_candidate/workspace.rs"]
mod workspace;

use clap::{Args, Parser, Subcommand, ValueEnum};
use ferrum_bench_core::release_candidate::staging::{self, AbiInput, Backend, CandidateInput};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(about = "Prepare release versions and staged metadata; no publication permission")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Update the coordinated workspace and internal dependencies for a formal version.
    Prepare {
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        #[arg(long)]
        version: String,
        /// Print the changed file list without writing the proposed edits.
        #[arg(long)]
        dry_run: bool,
    },
    /// Validate the candidate inputs and all actual workspace member versions.
    Verify {
        #[command(flatten)]
        candidate: CandidateArgs,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        /// Require advancement from this previous formal version (without v).
        #[arg(long)]
        previous_version: Option<String>,
    },
    /// Generate new adjacent checksum, version, dependency and ABI records.
    Manifest {
        #[command(flatten)]
        candidate: CandidateArgs,
        #[arg(long, value_enum)]
        backend: BackendArg,
        #[arg(long)]
        target_triple: String,
        #[arg(long)]
        asset: PathBuf,
        #[arg(long)]
        binary: PathBuf,
        #[arg(long)]
        dependencies: PathBuf,
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,
        #[arg(long, value_delimiter = ',')]
        cargo_features: Vec<String>,
        #[arg(long)]
        cuda_compute_capability: Option<String>,
        #[arg(long)]
        cuda_toolkit_image: Option<String>,
    },
}

#[derive(Args)]
struct CandidateArgs {
    /// Formal binary/package version; RC is only a candidate tag.
    #[arg(long)]
    version: String,
    #[arg(long)]
    candidate_sha: String,
    #[arg(long)]
    candidate_tag: String,
    #[arg(long)]
    staging_label: String,
    #[arg(long, default_value = "local")]
    workflow_run_id: String,
    #[arg(long, default_value = "1")]
    workflow_run_attempt: String,
}

impl CandidateArgs {
    fn input(self) -> CandidateInput {
        CandidateInput {
            version: self.version,
            release_candidate_sha: self.candidate_sha,
            release_candidate_tag: self.candidate_tag,
            staging_label: self.staging_label,
            workflow_run_id: self.workflow_run_id,
            workflow_run_attempt: self.workflow_run_attempt,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    Cpu,
    Metal,
    Cuda,
}

impl From<BackendArg> for Backend {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Cpu => Self::Cpu,
            BackendArg::Metal => Self::Metal,
            BackendArg::Cuda => Self::Cuda,
        }
    }
}

fn filename(path: &Path) -> Result<&str, String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("expected a UTF-8 file name: {}", path.display()))
}

fn json_bytes(value: &serde_json::Value) -> Result<Vec<u8>, String> {
    let mut result = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    result.push(b'\n');
    Ok(result)
}

fn write_new_outputs(directory: &Path, outputs: Vec<(String, Vec<u8>)>) -> Result<(), String> {
    if !directory.is_dir() {
        return Err(format!(
            "output directory does not exist: {}",
            directory.display()
        ));
    }
    // Reserve all destinations before writing. Existing records are never
    // overwritten; on failure remove only files reserved by this invocation.
    let mut reserved = Vec::new();
    let result = (|| {
        for (name, bytes) in outputs {
            let path = directory.join(name);
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| format!("create {}: {e}", path.display()))?;
            reserved.push((path, file, bytes));
        }
        for (path, file, bytes) in &mut reserved {
            file.write_all(bytes)
                .and_then(|()| file.sync_all())
                .map_err(|e| format!("write {}: {e}", path.display()))?;
        }
        Ok(())
    })();
    if result.is_err() {
        for (path, file, _) in reserved {
            drop(file);
            let _ = fs::remove_file(path);
        }
    }
    result
}

fn execute(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Prepare {
            workspace,
            version,
            dry_run,
        } => workspace::prepare(&workspace, &version, dry_run),
        Command::Verify {
            candidate,
            workspace,
            previous_version,
        } => {
            let input = candidate.input();
            staging::validate_candidate(&input)?;
            if let Some(previous) = previous_version {
                staging::validate_version_progression(&previous, &input.version)?;
            }
            let metadata = workspace::metadata(&workspace)?;
            staging::validate_workspace_versions(&metadata, &input.version)?;
            println!(
                "Candidate inputs and workspace versions are consistent: {}",
                input.version
            );
            Ok(())
        }
        Command::Manifest {
            candidate,
            backend,
            target_triple,
            asset,
            binary,
            dependencies,
            output_dir,
            cargo_features,
            cuda_compute_capability,
            cuda_toolkit_image,
        } => {
            let input = candidate.input();
            staging::validate_candidate(&input)?;
            let abi = AbiInput {
                backend: backend.into(),
                target_triple,
                cargo_features,
                cuda_compute_capability,
                cuda_toolkit_image,
            };
            let asset_name = filename(&asset)?;
            let read = |p: &Path| fs::read(p).map_err(|e| format!("read {}: {e}", p.display()));
            let audit = fs::read_to_string(&dependencies)
                .map_err(|e| format!("read dependency audit {}: {e}", dependencies.display()))?;
            let manifests = staging::generate_manifests(
                &input,
                &abi,
                asset_name,
                &read(&asset)?,
                &read(&binary)?,
                filename(&dependencies)?,
                &audit,
            )?;
            write_new_outputs(
                &output_dir,
                vec![
                    (
                        format!("{asset_name}.sha256"),
                        manifests.asset_checksum.into_bytes(),
                    ),
                    (
                        format!("{asset_name}.binary.sha256"),
                        manifests.binary_checksum.into_bytes(),
                    ),
                    (
                        format!("{asset_name}.version.json"),
                        json_bytes(&manifests.version)?,
                    ),
                    (
                        format!("{asset_name}.dependency.json"),
                        json_bytes(&manifests.dependency)?,
                    ),
                    (
                        format!("{asset_name}.abi.json"),
                        json_bytes(&manifests.abi)?,
                    ),
                ],
            )?;
            println!(
                "Wrote staged metadata for {asset_name}; runtime validation is still required"
            );
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match execute(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("release candidate: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_metadata_is_preserved_without_partial_new_records() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("asset.version.json"),
            "existing evidence",
        )
        .unwrap();
        let result = write_new_outputs(
            directory.path(),
            vec![
                ("asset.sha256".into(), b"new checksum".to_vec()),
                ("asset.version.json".into(), b"new evidence".to_vec()),
            ],
        );
        assert!(result.is_err());
        assert!(!directory.path().join("asset.sha256").exists());
        assert_eq!(
            fs::read_to_string(directory.path().join("asset.version.json")).unwrap(),
            "existing evidence"
        );
    }

    #[test]
    fn records_are_written_with_the_supplied_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let value = serde_json::json!({"version": "1.3.0", "runtime_validated": false});
        write_new_outputs(
            directory.path(),
            vec![
                ("asset.sha256".into(), b"checksum\n".to_vec()),
                ("asset.version.json".into(), json_bytes(&value).unwrap()),
            ],
        )
        .unwrap();
        assert_eq!(
            fs::read(directory.path().join("asset.sha256")).unwrap(),
            b"checksum\n"
        );
        let actual: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.path().join("asset.version.json")).unwrap())
                .unwrap();
        assert_eq!(actual, value);
    }
}
