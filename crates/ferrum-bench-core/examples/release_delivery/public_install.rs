//! Probe an executable after a real `cargo install` or Homebrew install/upgrade.
//!
//! The preceding workflow installation step establishes channel provenance.
//! `--channel` records that declaration; it cannot establish where executable
//! bytes came from. Passing a staged extraction does not test the public channel.
//! These four startup probes do not load models or repeat full runtime regression.
use super::installation::{probe, sha256, Observation};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    Cargo,
    Homebrew,
}
#[derive(Debug, Args)]
pub struct InstalledArgs {
    /// Executable produced by the immediately preceding channel installation step.
    #[arg(long)]
    pub binary: PathBuf,
    #[arg(long)]
    pub version: String,
    /// Caller attests which actual install/upgrade command produced this binary.
    #[arg(long, value_enum)]
    pub channel: Channel,
    /// New JSON report. Failures retain actual observations and available hashes.
    #[arg(long)]
    pub output: PathBuf,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Passed,
    Failed,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstalledReport {
    pub schema_version: u32,
    pub status: Status,
    pub channel: Channel,
    pub version: String,
    pub requested_binary: PathBuf,
    pub binary: Option<PathBuf>,
    pub binary_sha256: Option<String>,
    pub binary_sha256_after: Option<String>,
    pub observations: Vec<Observation>,
    pub error: Option<String>,
}
const COMMANDS: [&[&str]; 4] = [
    &["--version"],
    &["--help"],
    &["run", "--help"],
    &["serve", "--help"],
];
fn formal_version(version: &str) -> Result<(), String> {
    let parsed = semver::Version::parse(version)
        .map_err(|_| "installed target version is not semantic version")?;
    if !parsed.pre.is_empty() || !parsed.build.is_empty() || parsed.to_string() != version {
        return Err("installed target version must be formal".into());
    }
    Ok(())
}
fn verify_startup(version: &str, observations: &[Observation]) -> Result<(), String> {
    formal_version(version)?;
    if observations.len() != COMMANDS.len() {
        return Err("public installation startup probes are incomplete".into());
    }
    for (observation, arguments) in observations.iter().zip(COMMANDS) {
        if !observation
            .arguments
            .iter()
            .map(String::as_str)
            .eq(arguments.iter().copied())
            || observation.exit_code != Some(0)
            || observation.stdout.trim().is_empty()
        {
            return Err(format!(
                "public installation startup probe {arguments:?} failed or is missing"
            ));
        }
    }
    if observations[0].stdout.trim() != format!("ferrum {version}") {
        return Err("publicly installed binary reports a different version".into());
    }
    Ok(())
}
fn save(file: &mut fs::File, report: &InstalledReport) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    file.set_len(0).map_err(|error| error.to_string())?;
    file.write_all(&bytes).map_err(|error| error.to_string())?;
    file.flush().map_err(|error| error.to_string())
}
async fn observe(
    requested: &Path,
    report: &mut InstalledReport,
    file: &mut fs::File,
) -> Result<(), String> {
    formal_version(&report.version)?;
    let binary = fs::canonicalize(requested)
        .map_err(|error| format!("resolve installed binary: {error}"))?;
    report.binary = Some(binary.clone());
    report.binary_sha256 = Some(sha256(&binary)?);
    save(file, report)?;
    let mut errors = Vec::new();
    for arguments in COMMANDS {
        match probe(&binary, arguments).await {
            Ok(observation) => report.observations.push(observation),
            Err(error) => errors.push(format!("startup probe {arguments:?}: {error}")),
        }
        save(file, report)?;
    }
    match sha256(&binary) {
        Ok(after) => {
            if Some(&after) != report.binary_sha256.as_ref() {
                errors.push("installed binary changed during startup probes".into());
            }
            report.binary_sha256_after = Some(after);
        }
        Err(error) => errors.push(format!("hash installed binary after probes: {error}")),
    }
    if fs::canonicalize(requested).ok().as_ref() != Some(&binary) {
        errors.push("installed binary path changed during startup probes".into());
    }
    if let Err(error) = verify_startup(&report.version, &report.observations) {
        errors.push(error);
    }
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }
    Ok(())
}
pub async fn verify(args: InstalledArgs) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .map_err(|error| format!("create fresh installed-binary report: {error}"))?;
    let mut report = InstalledReport {
        schema_version: 1,
        status: Status::Failed,
        channel: args.channel,
        version: args.version,
        requested_binary: args.binary.clone(),
        binary: None,
        binary_sha256: None,
        binary_sha256_after: None,
        observations: Vec::new(),
        error: Some("installation startup verification has not finished".into()),
    };
    save(&mut file, &report)?;
    let result = observe(&args.binary, &mut report, &mut file).await;
    match &result {
        Ok(()) => {
            report.status = Status::Passed;
            report.error = None;
        }
        Err(error) => report.error = Some(error.clone()),
    }
    save(&mut file, &report)?;
    result
}

#[cfg(test)]
#[path = "public_install_tests.rs"]
mod tests;
