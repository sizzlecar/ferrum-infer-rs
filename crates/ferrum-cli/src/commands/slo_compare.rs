//! Offline comparison of the same immutable benchmark evidence for JSON and
//! both publication languages. Loading and metric decisions live in bench-core.

use clap::Args;
use ferrum_bench_core::slo_comparison::{
    artifact::{compare_manifest, ArtifactComparisonReport, ArtifactLoadLimits},
    ComparisonStatus, MarkdownLanguage,
};
use ferrum_types::{FerrumError, Result};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

mod limits;

#[derive(Args, Debug, Clone)]
#[command(
    after_help = "Exit codes: 0 = requirements met under the frozen statistical method and declared assumptions, with raw pilot eligibility verified; 1 = load/config/output error; 2 = CLI usage error; 3 = requirements not met; 4 = evidence insufficient or descriptive-only. Bootstrap confidence bounds are approximate and scoped to the frozen workload. Outputs are written before exit 3/4; existing files are never overwritten."
)]
pub struct SloCompareCommand {
    /// Versioned comparison manifest with hash-pinned artifact references.
    pub manifest: PathBuf,

    /// Version 1 JSON evidence-reading limits; omitted fields keep defaults.
    /// Limits apply jointly to main and pilot evidence within hard ceilings.
    #[arg(long, value_name = "JSON")]
    pub limits_config: Option<PathBuf>,

    /// New JSON report file. Parent directory must exist.
    #[arg(long, value_name = "JSON")]
    pub out: PathBuf,

    /// New English Markdown table file, rendered from the same report.
    #[arg(long, value_name = "MARKDOWN")]
    pub markdown_en: PathBuf,

    /// New Chinese Markdown table file, rendered from the same report.
    #[arg(long, value_name = "MARKDOWN")]
    pub markdown_zh: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SloCompareExit {
    ProofPass = 0,
    RequirementsNotMet = 3,
    InsufficientEvidence = 4,
}

impl SloCompareExit {
    pub const fn code(self) -> i32 {
        self as i32
    }

    pub const fn from_status(status: ComparisonStatus) -> Self {
        match status {
            ComparisonStatus::ProofPass => Self::ProofPass,
            ComparisonStatus::Failed => Self::RequirementsNotMet,
            ComparisonStatus::ObservedPass
            | ComparisonStatus::Unknown
            | ComparisonStatus::Inconclusive => Self::InsufficientEvidence,
        }
    }
}

/// Does not terminate the process, making status/exit handling testable. The
/// binary maps Err to its normal code 1, and successful execution to this code.
pub fn execute(cmd: SloCompareCommand) -> Result<SloCompareExit> {
    let limits = limits::load(cmd.limits_config.as_deref())?;
    let report = compare_manifest(&cmd.manifest, &limits.evidence)
        .map_err(|error| FerrumError::config(format!("load SLO comparison evidence: {error}")))?;
    let outputs = prepare_outputs(&cmd, &report, limits.config_path.as_deref())?;
    write_outputs(&outputs)?;
    let exit = SloCompareExit::from_status(report.comparison.status);
    println!(
        "SLO comparison: {:?}. Wrote JSON, English Markdown and Chinese Markdown. Exit {}.",
        report.comparison.status,
        exit.code()
    );
    Ok(exit)
}

struct PreparedOutput {
    path: PathBuf,
    bytes: Vec<u8>,
}

// Like bench_serve's SLO-output validation, an absent leaf is resolved through
// an existing canonical parent. Missing ancestors are rejected before writing.
fn output_identity(path: &Path) -> Result<PathBuf> {
    if let Ok(canonical) = path.canonicalize() {
        return Ok(canonical);
    }
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent.canonicalize().map_err(|error| {
        FerrumError::config(format!(
            "comparison output parent {} must exist: {error}",
            parent.display()
        ))
    })?;
    let name = path
        .file_name()
        .ok_or_else(|| FerrumError::config("comparison output must name a file"))?;
    Ok(parent.join(name))
}

fn same_existing_file(first: &Path, second: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(first), Ok(second)) = (fs::metadata(first), fs::metadata(second)) {
            return first.dev() == second.dev() && first.ino() == second.ino();
        }
    }
    #[cfg(not(unix))]
    let _ = (first, second);
    false
}

fn validate_outputs(paths: &[&Path], inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut identities = Vec::new();
    for path in paths {
        let identity = output_identity(path)?;
        if identities
            .iter()
            .any(|previous: &PathBuf| previous == &identity || same_existing_file(previous, path))
        {
            return Err(FerrumError::config(
                "comparison JSON and Markdown outputs must be distinct files",
            ));
        }
        if inputs
            .iter()
            .any(|input| input == &identity || same_existing_file(input, path))
        {
            return Err(FerrumError::config(format!(
                "comparison output {} aliases an input artifact",
                path.display()
            )));
        }
        if fs::symlink_metadata(path).is_ok() {
            return Err(FerrumError::config(format!(
                "comparison output {} already exists; select a new file",
                path.display()
            )));
        }
        identities.push(identity);
    }
    Ok(identities)
}

fn prepare_outputs(
    cmd: &SloCompareCommand,
    report: &ArtifactComparisonReport,
    limits_config: Option<&Path>,
) -> Result<Vec<PreparedOutput>> {
    let manifest = cmd
        .manifest
        .canonicalize()
        .map_err(|error| FerrumError::config(format!("resolve comparison manifest: {error}")))?;
    let directory = manifest
        .parent()
        .ok_or_else(|| FerrumError::config("comparison manifest has no parent"))?;
    // verified_files are canonical paths relative to the canonical manifest's
    // parent. Preserve those names even if a source disappears after loading.
    let mut inputs: Vec<_> = report
        .verified_files
        .iter()
        .map(|file| directory.join(&file.path))
        .collect();
    inputs.push(manifest);
    if let Some(path) = limits_config {
        inputs.push(path.to_owned());
    }
    let paths = validate_outputs(&[&cmd.out, &cmd.markdown_en, &cmd.markdown_zh], &inputs)?;
    let mut json = serde_json::to_vec_pretty(report).map_err(|error| {
        FerrumError::serialization(format!("serialize comparison report: {error}"))
    })?;
    json.push(b'\n');
    let bytes = [
        json,
        report.to_markdown(MarkdownLanguage::English).into_bytes(),
        report.to_markdown(MarkdownLanguage::Chinese).into_bytes(),
    ];
    Ok(paths
        .into_iter()
        .zip(bytes)
        .map(|(path, bytes)| PreparedOutput { path, bytes })
        .collect())
}

fn output_error(
    operation: &str,
    path: &Path,
    error: std::io::Error,
    created: &[(&Path, File)],
) -> FerrumError {
    let created: Vec<_> = created
        .iter()
        .map(|(path, _)| path.display().to_string())
        .collect();
    FerrumError::model(format!("{operation} comparison output {}: {error}. New files retained from this invocation: {created:?}. Existing files were not overwritten.", path.display()))
}

fn write_outputs(outputs: &[PreparedOutput]) -> Result<()> {
    let mut created: Vec<(&Path, File)> = Vec::new();
    // Reserve every destination using create_new before writing any bytes.
    // On failure retain and name only the files created here. Deleting by path
    // could remove a different file substituted concurrently by another owner.
    for output in outputs {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output.path)
            .map_err(|error| output_error("create", &output.path, error, &created))?;
        created.push((&output.path, file));
    }
    for (index, output) in outputs.iter().enumerate() {
        if let Err(error) = created[index]
            .1
            .write_all(&output.bytes)
            .and_then(|()| created[index].1.flush())
        {
            return Err(output_error("write/flush", &output.path, error, &created));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
