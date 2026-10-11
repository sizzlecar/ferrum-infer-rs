//! Reproducible source materialization from a pinned bundle and local overrides.

use std::{collections::BTreeMap, fs, path::Path};

use ferrum_types::is_sha256_digest;
use serde::{Deserialize, Serialize};

use crate::{
    read_json, sha256_bytes, validate_relative_path, write_json, NativeOperatorBuilderError,
    NativeOperatorSourceFileLock, Result,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOperatorSourceOverlay {
    pub schema_version: u32,
    /// Path and digest of the unchanged bundle manifest in the repository.
    pub base_bundle_manifest: NativeOperatorSourceFileLock,
    pub files: Vec<NativeOperatorSourceOverlayFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOperatorSourceOverlayFile {
    pub path: String,
    pub base_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository_override: Option<NativeOperatorSourceFileLock>,
}

#[derive(Deserialize)]
struct BundleManifest {
    members: Vec<NativeOperatorSourceFileLock>,
}

fn invalid(message: impl Into<String>) -> NativeOperatorBuilderError {
    NativeOperatorBuilderError::Invalid(message.into())
}

fn checked_bytes(root: &Path, entry: &NativeOperatorSourceFileLock) -> Result<Vec<u8>> {
    validate_relative_path(&entry.path)?;
    if !is_sha256_digest(&entry.sha256) {
        return Err(invalid(format!("invalid source digest: {}", entry.path)));
    }
    let path = root.join(&entry.path);
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.clone(),
            source,
        })?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err(invalid(format!(
            "source file escapes its root: {}",
            entry.path
        )));
    }
    let bytes = fs::read(&canonical).map_err(|source| NativeOperatorBuilderError::Io {
        path: canonical,
        source,
    })?;
    if sha256_bytes(&bytes) != entry.sha256 {
        return Err(invalid(format!("source hash mismatch: {}", entry.path)));
    }
    Ok(bytes)
}

/// Materialize only declared members into a new directory. Neither source root
/// is modified; every baseline member is checked even when it is overridden.
/// The resulting directory is consumed by the existing lock/source-build API.
pub fn materialize_native_operator_source_overlay(
    spec_path: &Path,
    base_source_root: &Path,
    repository_root: &Path,
    output_root: &Path,
) -> Result<()> {
    let spec: NativeOperatorSourceOverlay = read_json(spec_path)?;
    if spec.schema_version != 1 || spec.files.is_empty() {
        return Err(invalid("unsupported or empty native source overlay"));
    }
    let canonicalize = |path: &Path| {
        path.canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: path.to_owned(),
                source,
            })
    };
    let base_root = canonicalize(base_source_root)?;
    let repository_root = canonicalize(repository_root)?;
    let output_parent = canonicalize(
        output_root
            .parent()
            .ok_or_else(|| invalid("source output has no parent"))?,
    )?;
    let output = output_parent.join(
        output_root
            .file_name()
            .ok_or_else(|| invalid("source output has no filename"))?,
    );
    if output.starts_with(&base_root) || output.starts_with(&repository_root) {
        return Err(invalid("source output must be outside both input roots"));
    }
    let manifest_bytes = checked_bytes(&repository_root, &spec.base_bundle_manifest)?;
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes).map_err(|source| {
        NativeOperatorBuilderError::Json {
            path: repository_root.join(&spec.base_bundle_manifest.path),
            source,
        }
    })?;
    let members = manifest
        .members
        .iter()
        .map(|entry| (entry.path.as_str(), entry.sha256.as_str()))
        .collect::<BTreeMap<_, _>>();
    if members.len() != manifest.members.len() {
        return Err(invalid("duplicate native bundle member"));
    }
    let mut previous = None;
    let mut materialized = Vec::with_capacity(spec.files.len());
    for entry in &spec.files {
        validate_relative_path(&entry.path)?;
        if previous.is_some_and(|path: &str| path >= entry.path.as_str()) {
            return Err(invalid("source overlay members must be sorted and unique"));
        }
        previous = Some(entry.path.as_str());
        if members.get(entry.path.as_str()).copied() != Some(entry.base_sha256.as_str()) {
            return Err(invalid(format!(
                "source overlay member differs from pinned bundle: {}",
                entry.path
            )));
        }
        let baseline = checked_bytes(
            &base_root,
            &NativeOperatorSourceFileLock {
                path: entry.path.clone(),
                sha256: entry.base_sha256.clone(),
            },
        )?;
        let bytes = match &entry.repository_override {
            Some(override_file) => checked_bytes(&repository_root, override_file)?,
            None => baseline,
        };
        materialized.push((entry.path.as_str(), bytes));
    }
    // All source validation precedes creation. create_dir fails if any existing
    // directory/file/symlink occupies the output, so none can be overwritten.
    fs::create_dir(output_root).map_err(|source| NativeOperatorBuilderError::Io {
        path: output_root.to_owned(),
        source,
    })?;
    for (relative, bytes) in materialized {
        let path = output_root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| NativeOperatorBuilderError::Io {
                path: parent.to_owned(),
                source,
            })?;
        }
        fs::write(&path, bytes)
            .map_err(|source| NativeOperatorBuilderError::Io { path, source })?;
    }
    write_json(&output_root.join("source-overlay.json"), &spec)?;
    Ok(())
}

#[cfg(test)]
mod tests;
