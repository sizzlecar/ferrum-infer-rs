//! Resolve a configured logical source in one existing Hugging Face cache.
//! No downloads, model execution, arbitrary snapshot selection or weight copies.
use super::super::source::{digest_file, FilePin, SourceManifest, SIDECARS};
use ferrum_bench_core::release_regression::ModelProfile;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tokio::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RepositorySelector {
    pub repository: String,
    pub revision: Option<String>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WeightSelector {
    pub repository: String,
    pub filename: String,
    pub revision: Option<String>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SourceSelector {
    pub model: String,
    pub gguf: WeightSelector,
    pub tokenizer: RepositorySelector,
}
#[derive(Debug, Serialize)]
pub(super) struct SnapshotEvidence {
    pub repository: String,
    pub revision: String,
    pub resolution: &'static str,
    pub snapshot: PathBuf,
}
#[derive(Debug, Serialize)]
pub(super) struct ResolvedSource {
    pub gguf: SnapshotEvidence,
    pub tokenizer: SnapshotEvidence,
}
pub(super) fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
pub(super) fn revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn repository_directory(cache: &Path, repository: &str) -> Result<PathBuf, String> {
    let parts: Vec<_> = repository.split('/').collect();
    if parts.len() != 2 || !parts.iter().all(|part| safe_component(part)) {
        return Err("HF repository must be an explicit owner/name, not a path or URL".into());
    }
    let hub = if cache.join("hub").is_dir() {
        cache.join("hub")
    } else {
        cache.to_owned()
    };
    let directory = hub.join(format!("models--{}--{}", parts[0], parts[1]));
    fs::canonicalize(&directory)
        .map_err(|e| format!("HF repository {repository} is not cached: {e}"))
}
fn complete(snapshot: &Path, required: &[&str]) -> bool {
    required
        .iter()
        .all(|name| fs::metadata(snapshot.join(name)).is_ok_and(|m| m.is_file() && m.len() > 0))
}
fn snapshot(
    cache: &Path,
    repository: &str,
    requested: Option<&str>,
    required: &[&str],
) -> Result<SnapshotEvidence, String> {
    let directory = repository_directory(cache, repository)?;
    let reference = directory.join("refs/main");
    let (selected, resolution) = if let Some(value) = requested {
        (value.to_owned(), "explicit_revision")
    } else if fs::symlink_metadata(&reference).is_ok() {
        (
            fs::read_to_string(&reference)
                .map_err(|e| format!("read HF main ref: {e}"))?
                .trim()
                .to_owned(),
            "cached_main_ref",
        )
    } else {
        let mut candidates = Vec::new();
        for entry in fs::read_dir(directory.join("snapshots"))
            .map_err(|e| format!("list HF snapshots: {e}"))?
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if revision(&name) && complete(&entry.path(), required) {
                candidates.push(name);
            }
        }
        if candidates.len() != 1 {
            return Err(format!("HF repository {repository} has {} complete snapshots and no main ref; specify an immutable revision", candidates.len()));
        }
        (candidates.remove(0), "unique_complete_snapshot")
    };
    if !revision(&selected) {
        return Err(format!(
            "HF repository {repository} requires a full immutable revision"
        ));
    }
    let path = directory.join("snapshots").join(&selected);
    if !complete(&path, required) {
        return Err(format!("selected HF snapshot {repository}@{selected} is incomplete; do not reuse a different revision"));
    }
    Ok(SnapshotEvidence {
        repository: repository.to_owned(),
        revision: selected,
        resolution,
        snapshot: fs::canonicalize(path).map_err(|e| e.to_string())?,
    })
}
pub(super) fn resolve(
    cache: &Path,
    selector: &SourceSelector,
    profile: &ModelProfile,
    deadline: Instant,
) -> Result<(SourceManifest, ResolvedSource), String> {
    if selector.model != profile.model {
        return Err("source selector model differs from the planned profile".into());
    }
    if !safe_component(&selector.gguf.filename) || !selector.gguf.filename.ends_with(".gguf") {
        return Err("GGUF selector requires one explicit relative .gguf filename".into());
    }
    let gguf = snapshot(
        cache,
        &selector.gguf.repository,
        selector.gguf.revision.as_deref(),
        &[&selector.gguf.filename],
    )?;
    let tokenizer = snapshot(
        cache,
        &selector.tokenizer.repository,
        selector.tokenizer.revision.as_deref(),
        &["tokenizer.json", "tokenizer_config.json"],
    )?;
    // Keep the immutable snapshot filename (.gguf), while digest_file follows HF blob symlinks.
    let path = gguf.snapshot.join(&selector.gguf.filename);
    let digest = digest_file(&path, deadline)?;
    let weight = FilePin {
        path,
        bytes: digest.bytes,
        sha256: digest.sha256,
    };
    let mut sidecars = Vec::new();
    for name in SIDECARS {
        let path = tokenizer.snapshot.join(name);
        if fs::symlink_metadata(&path).is_ok() {
            let digest = digest_file(&path, deadline)?;
            sidecars.push(FilePin {
                path: PathBuf::from(name),
                bytes: digest.bytes,
                sha256: digest.sha256,
            });
        }
    }
    let manifest = SourceManifest {
        schema_version: 1,
        profile: profile.clone(),
        gguf: weight,
        tokenizer_dir: tokenizer.snapshot.clone(),
        sidecars,
    };
    manifest.identity()?.validate()?;
    Ok((manifest, ResolvedSource { gguf, tokenizer }))
}
