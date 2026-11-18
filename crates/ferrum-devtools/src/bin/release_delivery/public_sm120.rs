//! Read-only verification of a bootstrap executed on public SM120 release bytes.
use super::{
    public_entry::EntryReport,
    public_install::{verify_startup, Channel, InstalledReport, Status},
};
use anyhow::{anyhow, ensure, Context, Result};
use chrono::{DateTime, FixedOffset};
use ferrum_bench_core::release_regression::{
    model_basic::{verify_basic_run, verify_basic_serve},
    model_tasks::run_backend,
    Backend,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const ARCHIVE: &str = "ferrum-linux-x86_64-cuda-sm120.tar.gz";
const INSTALLER: &str = "https://ferrum.pandaailabs.com/install.sh";

#[derive(Debug, clap::Args)]
pub struct Args {
    #[arg(long)]
    directory: PathBuf,
    #[arg(long)]
    version: String,
    #[arg(long)]
    source_sha: String,
}

#[derive(Deserialize)]
struct Evidence {
    schema_version: u32,
    release_id: u64,
    source_sha: String,
    version: String,
    archive_asset_id: u64,
    install_home_before: InstallHomeBefore,
    archive_transfer: ArchiveTransfer,
    install_started_at: DateTime<FixedOffset>,
    install_finished_at: DateTime<FixedOffset>,
    installer_args: Vec<String>,
    installer_exit_code: i32,
    installer_output: String,
    managed_archive_checksum: String,
    entry: EntryReport,
    installed: InstalledReport,
    model: Value,
}

#[derive(Deserialize)]
struct InstallHomeBefore {
    home: PathBuf,
    installer_root_exists: bool,
    launcher_exists: bool,
}

#[derive(Deserialize)]
struct ArchiveTransfer {
    requested_url: String,
    http_status: u16,
    exit_code: i32,
    download_bytes: u64,
}

fn checksum(line: &str, member: &str) -> Result<String> {
    let mut fields = line.split_whitespace();
    let hash = fields.next().context("missing SHA-256")?;
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            && fields.next() == Some(member)
            && fields.next().is_none(),
        "invalid checksum for {member}"
    );
    Ok(hash.to_owned())
}

fn timestamp(value: &Value) -> Result<DateTime<FixedOffset>> {
    Ok(DateTime::parse_from_rfc3339(
        value.as_str().context("missing timestamp")?,
    )?)
}

fn automatic(args: &[String], version: &str) -> bool {
    let mut args = args.iter().map(String::as_str);
    while let Some(arg) = args.next() {
        match arg {
            "--no-modify-path" => {}
            "--backend" if args.next() == Some("auto") => {}
            "--version"
                if args
                    .next()
                    .is_some_and(|v| v.strip_prefix('v').unwrap_or(v) == version) => {}
            _ => return false,
        }
    }
    true
}

fn verify_model(model: &Value, version: &str, binary: &Path, hash: &str) -> Result<()> {
    let options = &model["options"];
    ensure!(
        model["schema_version"] == 2
            && model["status"] == "passed"
            && model["binary_sha256"] == hash
            && options["ferrum_bin"].as_str().map(Path::new) == Some(binary)
            && options["backend"] == "cuda"
            && options["use_default_backend"] == true
            && model["environment_policy"] == "remove_inherited_ferrum_overrides",
        "model did not exercise the automatically selected installed executable"
    );
    let mut cases = BTreeMap::new();
    for case in model["cases"]
        .as_array()
        .context("missing raw model cases")?
    {
        let name = case["case"].as_str().context("unnamed model case")?;
        ensure!(
            case["status"] == "passed" && cases.insert(name, &case["evidence"]).is_none(),
            "failed or duplicate model case {name}"
        );
    }
    let observed = |name| {
        cases
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("missing model observation {name}"))
    };
    ensure!(
        observed("binary-version")?["version"] == format!("ferrum {version}")
            && observed("binary-unchanged")?["sha256"] == hash,
        "model run binary changed or reports a different version"
    );
    let run = observed("run-basic")?;
    let ready = &run["ready"];
    let health = observed("serve-startup")?;
    let device = &health["auto_config"]["hardware_capabilities"];
    ensure!(
        ready["event"] == "ready"
            && run_backend(&ready["backend"]) == Some(Backend::Cuda)
            && health["status"] == "healthy"
            && health["version"] == version
            && device["backend"] == "cuda"
            && device["compute_capability"] == "12.0",
        "missing actual CUDA run and SM120 serve observations"
    );
    let max = u32::try_from(
        options["max_tokens"]
            .as_u64()
            .context("missing model token budget")?,
    )?;
    let protocol = serde_json::from_value(ready["reasoning_protocol"].clone())?;
    verify_basic_run(run, protocol, max).map_err(anyhow::Error::msg)?;
    let protocol = serde_json::from_value(health["reasoning_protocol"].clone())?;
    verify_basic_serve(observed("serve-basic")?, protocol, max).map_err(anyhow::Error::msg)
}

fn verify_evidence(args: &Args) -> Result<()> {
    let read = |name| fs::read(args.directory.join(name));
    let release: Value = serde_json::from_slice(&read("release.json")?)?;
    let asset: Value = serde_json::from_slice(&read("evidence-asset.json")?)?;
    let bytes = read("evidence.json")?;
    ensure!(
        asset["digest"] == format!("sha256:{:x}", Sha256::digest(&bytes)),
        "evidence bytes differ from the approved GitHub asset digest"
    );
    let proof: Evidence = serde_json::from_slice(&bytes)?;
    ensure!(
        release["tag_name"] == format!("v{}", args.version)
            && release["target_commitish"] == args.source_sha
            && release["draft"] == false
            && release["prerelease"] == false
            && proof.schema_version == 1
            && proof.release_id != 0
            && release["id"] == proof.release_id
            && proof.version == args.version
            && proof.source_sha == args.source_sha,
        "evidence belongs to a different formal release or commit"
    );
    let published = timestamp(&release["published_at"])?;
    let created = timestamp(&asset["created_at"])?;
    let started = timestamp(&proof.model["started_at"])?;
    let finished = timestamp(&proof.model["finished_at"])?;
    // GitHub release/asset timestamps have one-second precision.
    ensure!(
        created > published
            && proof.install_started_at >= published
            && proof.install_finished_at >= proof.install_started_at
            && started >= proof.install_finished_at
            && finished >= started
            && finished.timestamp() <= created.timestamp(),
        "evidence was not collected after publication and before its upload"
    );
    let mut archives = release["assets"]
        .as_array()
        .context("release is missing assets")?
        .iter()
        .filter(|a| a["name"] == ARCHIVE);
    let archive = archives
        .next()
        .context("public release lacks SM120 archive")?;
    let archive_hash = checksum(&proof.managed_archive_checksum, ARCHIVE)?;
    ensure!(
        archives.next().is_none()
            && proof.archive_asset_id != 0
            && archive["id"] == proof.archive_asset_id
            && archive["state"] == "uploaded"
            && archive["digest"] == format!("sha256:{archive_hash}"),
        "installed archive differs from the official SM120 asset"
    );
    let transfer = &proof.archive_transfer;
    let official_path = format!("v{}/{ARCHIVE}", args.version);
    let origins = [
        "https://ferrum.pandaailabs.com/downloads",
        "https://github.com/sizzlecar/ferrum-infer-rs/releases/download",
    ];
    ensure!(
        origins
            .iter()
            .any(|base| transfer.requested_url == format!("{base}/{official_path}"))
            && transfer.http_status == 200
            && transfer.exit_code == 0
            && transfer.download_bytes > 0
            && archive["size"].as_u64() == Some(transfer.download_bytes),
        "bootstrap did not download the complete archive from the official release"
    );
    ensure!(
        proof
            .entry
            .matches_public_script(INSTALLER, &read("install.sh")?)
            && proof.installer_exit_code == 0
            && !proof.installer_output.is_empty()
            && automatic(&proof.installer_args, &args.version),
        "missing successful automatic installation from the deployed public script"
    );
    let binary_hash = checksum(&String::from_utf8(read("binary.sha256")?)?, "ferrum")?;
    let installed = &proof.installed;
    ensure!(
        installed.schema_version == 1
            && installed.channel == Channel::Bootstrap
            && installed.status == Status::Passed
            && installed.version == args.version
            && installed.error.is_none()
            && installed.binary_sha256.as_ref() == Some(&binary_hash)
            && installed.binary_sha256_after.as_ref() == Some(&binary_hash),
        "installed startup receipt does not match the public binary"
    );
    let before = &proof.install_home_before;
    ensure!(
        !before.installer_root_exists && !before.launcher_exists,
        "bootstrap HOME already contained an installation or launcher"
    );
    let home = &before.home;
    let binary = home
        .join(".local/share/ferrum/installer/releases")
        .join(format!("{}-cuda-{binary_hash}", args.version))
        .join("ferrum");
    ensure!(
        home.is_absolute()
            && installed.requested_binary == home.join(".local/bin/ferrum")
            && installed.binary.as_ref() == Some(&binary),
        "startup used an executable outside the active public bootstrap installation"
    );
    verify_startup(&args.version, &installed.observations).map_err(anyhow::Error::msg)?;
    verify_model(&proof.model, &args.version, &binary, &binary_hash)
}

pub fn verify(args: Args) -> Result<(), String> {
    verify_evidence(&args).map_err(|e| format!("{e:#}"))
}
