//! Channel publication follows the caller's candidate gate. Checkpoints are
//! resumable observations, never authorization or runtime correctness evidence.
use super::{AcceptedAsset, AcceptedRelease};
use clap::Args;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

#[path = "publish/github.rs"]
mod github;
#[path = "publish/registry.rs"]
mod registry;

#[derive(Args, Debug)]
pub struct PublishArgs {
    #[arg(skip)]
    pub repo: String,
    #[arg(long, default_value = "sizzlecar/homebrew-ferrum")]
    pub tap_repo: String,
    #[arg(long, default_value = "main")]
    pub tap_branch: String,
    /// Optional environment variable containing a token authorized for the tap.
    #[arg(long, default_value = "GITHUB_TOKEN")]
    pub tap_token_env: String,
    /// Durable checkpoints; must be outside the candidate workspace.
    #[arg(long)]
    pub state_dir: PathBuf,
    #[arg(long, default_value = "1.91.0")]
    pub cargo_toolchain: String,
    #[arg(long, default_value_t = 300)]
    pub index_timeout_secs: u64,
}

#[derive(Serialize, Deserialize)]
struct ChannelState {
    version: String,
    candidate_sha: String,
    crates: bool,
    github_release_and_tap: bool,
    installations: String,
}

pub async fn publish(args: PublishArgs, accepted: AcceptedRelease) -> Result<(), String> {
    validate_inputs(&args, &accepted)?;
    fs::create_dir_all(&args.state_dir).map_err(|e| format!("create publication state: {e}"))?;
    let _lock = StateLock::acquire(&args.state_dir)?;
    let mut state = ChannelState {
        version: accepted.version.clone(), candidate_sha: accepted.candidate_sha.clone(),
        crates: false, github_release_and_tap: false,
        installations: "pending: registry and Homebrew installation/startup verification must follow public publication".into(),
    };
    let state_path = args.state_dir.join("channels.json");
    // Saved booleans are outputs only. Every invocation re-reads the channels.
    write_json(&state_path, &state)?;
    registry::publish_crates(&args, &accepted).await?;
    state.crates = true;
    write_json(&state_path, &state)?;
    github::release_and_tap(&args, &accepted).await?;
    state.github_release_and_tap = true;
    write_json(&state_path, &state)?;
    println!("Publication channels reconciled for v{}. Delivery remains incomplete: installation/startup verification is pending. Receipt: {}", accepted.version, state_path.display());
    Ok(())
}

fn validate_inputs(args: &PublishArgs, accepted: &AcceptedRelease) -> Result<(), String> {
    let version =
        semver::Version::parse(&accepted.version).map_err(|_| "invalid release version")?;
    if !version.pre.is_empty()
        || !version.build.is_empty()
        || version.to_string() != accepted.version
    {
        return Err("publication requires a canonical stable semantic version".into());
    }
    if accepted.candidate_sha.len() != 40
        || !accepted
            .candidate_sha
            .bytes()
            .all(|c| c.is_ascii_hexdigit())
    {
        return Err("candidate SHA must identify a complete Git commit".into());
    }
    for repo in [&args.repo, &args.tap_repo] {
        let parts: Vec<_> = repo.split('/').collect();
        if parts.len() != 2 || parts.iter().any(|p| !safe_component(p)) {
            return Err("repository must be an owner/name pair".into());
        }
    }
    if !safe_component(&args.tap_branch)
        || !safe_component(&args.cargo_toolchain)
        || args.index_timeout_secs == 0
    {
        return Err("invalid branch/toolchain or zero index timeout".into());
    }
    if args.tap_token_env.is_empty()
        || !args
            .tap_token_env
            .bytes()
            .enumerate()
            .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || i > 0 && c.is_ascii_digit())
    {
        return Err("tap token must be an explicit environment variable name".into());
    }
    let workspace = accepted
        .workspace
        .canonicalize()
        .map_err(|e| format!("candidate workspace: {e}"))?;
    fs::create_dir_all(&args.state_dir).map_err(|e| format!("publication state directory: {e}"))?;
    let state = args.state_dir.canonicalize().map_err(|e| e.to_string())?;
    if state.starts_with(workspace) {
        return Err("publication state must be outside the candidate workspace".into());
    }
    let mut names = BTreeSet::new();
    if accepted.assets.is_empty() {
        return Err("accepted release has no assets".into());
    }
    for asset in &accepted.assets {
        if !safe_component(&asset.name) || !names.insert(&asset.name) {
            return Err("release assets require unique safe file names".into());
        }
        verified_asset_bytes(asset)?;
    }
    Ok(())
}

fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.starts_with('-')
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

pub(super) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn verified_asset_bytes(asset: &AcceptedAsset) -> Result<Vec<u8>, String> {
    let bytes =
        fs::read(&asset.path).map_err(|e| format!("read accepted asset {}: {e}", asset.name))?;
    if sha256(&bytes) != asset.sha256 {
        return Err(format!("accepted asset bytes changed: {}", asset.name));
    }
    Ok(bytes)
}

pub(super) fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let parent = path.parent().ok_or("state path has no parent")?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(file.as_file_mut(), value).map_err(|e| e.to_string())?;
    file.as_file_mut()
        .write_all(b"\n")
        .map_err(|e| e.to_string())?;
    file.as_file().sync_all().map_err(|e| e.to_string())?;
    file.persist(path)
        .map_err(|e| format!("persist publication state: {e}"))?;
    Ok(())
}

pub(super) fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent("ferrum-release-delivery")
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|_| "construct publication HTTP client".into())
}

struct StateLock(PathBuf);
impl StateLock {
    fn acquire(directory: &Path) -> Result<Self, String> {
        let path = directory.join("publication.lock");
        let mut file = fs::OpenOptions::new().create_new(true).write(true).open(&path)
            .map_err(|_| "publication state is locked; establish the previous controller has stopped before removing publication.lock")?;
        writeln!(file, "{}", std::process::id()).map_err(|e| e.to_string())?;
        Ok(Self(path))
    }
}
impl Drop for StateLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(test)]
#[path = "publish/test_http.rs"]
mod test_http;
