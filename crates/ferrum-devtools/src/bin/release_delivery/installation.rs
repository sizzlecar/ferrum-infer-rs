//! Verify the bytes inside a release archive, then probe those extracted bytes.
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::process::Command;

#[derive(Debug, Args)]
pub struct InspectArgs {
    #[arg(long)]
    pub abi: PathBuf,
    #[arg(long)]
    pub version: String,
    #[arg(long)]
    pub candidate_sha: String,
    /// New directory for the exact binary extracted from this archive.
    #[arg(long)]
    pub extract_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Verify/extract bytes on a host unable to execute this target. Records not_run.
    #[arg(long)]
    pub extract_only: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub arguments: Vec<String>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstallationReport {
    pub schema_version: u32,
    pub status: String,
    pub version: String,
    pub candidate_sha: String,
    pub asset_name: String,
    pub asset_sha256: String,
    pub binary_sha256: String,
    pub backend: String,
    pub target_triple: String,
    pub binary: PathBuf,
    pub observations: Vec<Observation>,
    pub error: Option<String>,
}
pub fn sha256(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut bytes = [0u8; 65536];
    loop {
        let count = file.read(&mut bytes).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&bytes[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn text(value: &Value, key: &str) -> Result<String, String> {
    value[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("metadata is missing {key}"))
}
fn plain_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(Component::Normal(_)))
        && parts.next().is_none()
        && !name.contains(['/', '\\'])
        && !name.starts_with('-')
}
fn validate_metadata(abi: &Value, version: &Value, expected: &InspectArgs) -> Result<(), String> {
    if abi["schema_version"] != 1 || version["schema_version"] != 1 {
        return Err("unsupported asset metadata schema".into());
    }
    let parsed = semver::Version::parse(&expected.version).map_err(|e| e.to_string())?;
    if !parsed.pre.is_empty()
        || !parsed.build.is_empty()
        || parsed.to_string() != expected.version
        || version["version"] != expected.version
    {
        return Err("archive version does not match the formal candidate version".into());
    }
    for key in [
        "asset_name",
        "asset_sha256",
        "binary_name",
        "binary_sha256",
        "release_candidate_sha",
    ] {
        text(abi, key)?;
        if abi[key] != version[key] {
            return Err(format!("asset metadata disagrees on {key}"));
        }
    }
    if abi["binary_name"] != "ferrum" {
        return Err("metadata must describe the ferrum archive entry".into());
    }
    if !valid_commit(&expected.candidate_sha) {
        return Err("candidate SHA must identify a complete Git commit".into());
    }
    if abi["release_candidate_sha"] != expected.candidate_sha {
        return Err("archive source differs from the planned candidate".into());
    }
    if !plain_name(&text(abi, "asset_name")?) {
        return Err("asset_name must be a filename in the staging directory".into());
    }
    for key in ["asset_sha256", "binary_sha256"] {
        let value = text(abi, key)?;
        if !valid_hex(&value, 64) {
            return Err(format!("invalid {key}"));
        }
    }
    validate_target(&text(abi, "backend")?, &text(abi, "target_triple")?)
}
fn valid_commit(value: &str) -> bool {
    valid_hex(value, 40) || valid_hex(value, 64)
}
fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn validate_target(backend: &str, target: &str) -> Result<(), String> {
    let parts: Vec<_> = target.split('-').collect();
    if parts.iter().any(|part| {
        part.is_empty() || !part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    }) {
        return Err("invalid installation target triple".into());
    }
    match (backend, parts.as_slice()) {
        ("cpu" | "cuda", [_, _, "linux", _]) | ("metal", [_, "apple", "darwin"]) => Ok(()),
        _ => Err("installation backend does not agree with its staged target platform".into()),
    }
}
async fn extract(archive: &Path, binary: &Path, expected_sha: &str) -> Result<(), String> {
    // Extract only the program to stdout: archive entries cannot write outside
    // the destination or create symlinks. A missing/duplicate entry changes its digest.
    let output = tokio::time::timeout(
        Duration::from_secs(120),
        Command::new("tar")
            .arg("-xOzf")
            .arg(archive)
            .args(["--", "ferrum"])
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| "archive extraction timed out")?
    .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("could not read ferrum from the archive".into());
    }
    let actual = format!("{:x}", Sha256::digest(&output.stdout));
    if output.stdout.is_empty() || actual != expected_sha {
        return Err("archive contains a different binary than its accepted digest".into());
    }
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o700);
    }
    let mut file = options.open(binary).map_err(|e| e.to_string())?;
    file.write_all(&output.stdout).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    Ok(())
}
pub(super) async fn probe(binary: &Path, arguments: &[&str]) -> Result<Observation, String> {
    let output = tokio::time::timeout(
        Duration::from_secs(30),
        Command::new(binary)
            .args(arguments)
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| format!("startup probe {arguments:?} timed out"))?
    .map_err(|e| e.to_string())?;
    Ok(Observation {
        arguments: arguments.iter().map(|s| (*s).into()).collect(),
        exit_code: output.status.code(),
        stdout: String::from_utf8(output.stdout).map_err(|e| e.to_string())?,
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}
pub fn verify_runtime(report: &InstallationReport) -> Result<(), String> {
    if report.schema_version != 1 || report.status != "passed" || report.error.is_some() {
        return Err("installation did not complete successfully".into());
    }
    let version =
        semver::Version::parse(&report.version).map_err(|_| "installation version is invalid")?;
    if !version.pre.is_empty()
        || !version.build.is_empty()
        || version.to_string() != report.version
        || !valid_commit(&report.candidate_sha)
        || !plain_name(&report.asset_name)
        || !valid_hex(&report.asset_sha256, 64)
        || !valid_hex(&report.binary_sha256, 64)
    {
        return Err("installation report lacks complete candidate/archive identity".into());
    }
    validate_target(&report.backend, &report.target_triple)?;
    let expected = [
        vec!["--version"],
        vec!["--help"],
        vec!["run", "--help"],
        vec!["serve", "--help"],
    ];
    if report.observations.len() != expected.len() {
        return Err("installation probes are incomplete".into());
    }
    for (observation, arguments) in report.observations.iter().zip(expected) {
        if observation.arguments != arguments
            || observation.exit_code != Some(0)
            || observation.stdout.trim().is_empty()
        {
            return Err(format!(
                "installation startup probe {arguments:?} failed or is missing"
            ));
        }
    }
    if report.observations[0].stdout.trim() != format!("ferrum {}", report.version) {
        return Err("installed program reports a different version".into());
    }
    Ok(())
}
pub async fn inspect(args: InspectArgs) -> Result<(), String> {
    let abi = read_json(&args.abi)?;
    let asset_name = text(&abi, "asset_name")?;
    if !plain_name(&asset_name) {
        return Err("invalid archive filename".into());
    }
    let directory = args.abi.parent().ok_or("ABI file has no parent")?;
    let version = read_json(&directory.join(format!("{asset_name}.version.json")))?;
    validate_metadata(&abi, &version, &args)?;
    let archive = directory.join(&asset_name);
    let asset_sha256 = sha256(&archive)?;
    if asset_sha256 != text(&abi, "asset_sha256")? {
        return Err("archive SHA-256 does not match staged metadata".into());
    }
    fs::create_dir(&args.extract_dir)
        .map_err(|e| format!("create new extraction directory: {e}"))?;
    let binary = fs::canonicalize(&args.extract_dir)
        .map_err(|e| e.to_string())?
        .join("ferrum");
    let binary_sha256 = text(&abi, "binary_sha256")?;
    let mut report = InstallationReport {
        schema_version: 1,
        status: "failed".into(),
        version: args.version.clone(),
        candidate_sha: args.candidate_sha.clone(),
        asset_name,
        asset_sha256,
        binary_sha256,
        backend: text(&abi, "backend")?,
        target_triple: text(&abi, "target_triple")?,
        binary,
        observations: Vec::new(),
        error: None,
    };
    let result = async {
        extract(&archive, &report.binary, &report.binary_sha256).await?;
        if sha256(&archive)? != report.asset_sha256 {
            return Err("archive changed during extraction".into());
        }
        if args.extract_only {
            report.status = "not_run".into();
            return Ok(());
        }
        for arguments in [
            &["--version"][..],
            &["--help"],
            &["run", "--help"],
            &["serve", "--help"],
        ] {
            if sha256(&report.binary)? != report.binary_sha256 {
                return Err("binary changed before startup probe".into());
            }
            report
                .observations
                .push(probe(&report.binary, arguments).await?);
            let observation = report
                .observations
                .last()
                .ok_or("startup observation missing")?;
            if observation.exit_code != Some(0) || observation.stdout.trim().is_empty() {
                return Err(format!("installation startup probe {arguments:?} failed"));
            }
            if arguments == ["--version"]
                && observation.stdout.trim() != format!("ferrum {}", report.version)
            {
                return Err("installed program reports a different version".into());
            }
        }
        report.status = "passed".into();
        verify_runtime(&report)?;
        if sha256(&report.binary)? != report.binary_sha256 {
            return Err("binary changed during inspection".into());
        }
        Ok::<(), String>(())
    }
    .await;
    if let Err(error) = &result {
        report.status = "failed".into();
        report.error = Some(error.clone());
    }
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)
        .map_err(|e| e.to_string())?;
    output
        .write_all(&serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    result
}

#[cfg(test)]
#[path = "installation_tests.rs"]
mod tests;
