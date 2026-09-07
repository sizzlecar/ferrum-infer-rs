//! Cache selection and completeness for product model sources.

use ferrum_models::source::{inspect_cached_weights, CachedWeights, ResolvedModelSource};
use ferrum_types::{FerrumError, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
pub(crate) enum CacheRequirements {
    Repository,
    Product,
    ExplicitTokenizer,
    WeightsOnly,
}

impl CacheRequirements {
    fn metadata(self) -> &'static [&'static str] {
        match self {
            Self::Product => &["config.json", "tokenizer.json"],
            Self::ExplicitTokenizer => &["config.json"],
            Self::Repository | Self::WeightsOnly => &[],
        }
    }
}

pub(super) enum CachedModel {
    Ready(ResolvedModelSource),
    Missing(Option<String>),
}

impl CachedModel {
    pub(super) fn into_source(self) -> Option<ResolvedModelSource> {
        match self {
            Self::Ready(source) => Some(source),
            Self::Missing(_) => None,
        }
    }
}

pub(super) fn inspect_snapshot(
    path: &Path,
    requested_model: &str,
    requirements: CacheRequirements,
) -> Result<CachedModel> {
    let format = match inspect_cached_weights(path)? {
        CachedWeights::Ready(format) => format,
        CachedWeights::Absent => return Ok(CachedModel::Missing(None)),
        CachedWeights::Incomplete { reason } => return Ok(CachedModel::Missing(Some(reason))),
    };
    for filename in requirements.metadata() {
        let file = path.join(filename);
        match std::fs::metadata(&file) {
            Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {}
            Ok(metadata) if !metadata.is_file() => {
                return Err(FerrumError::model(format!(
                    "Cached model metadata {} is not a file",
                    file.display()
                )));
            }
            Ok(_) => {
                return Ok(CachedModel::Missing(Some(format!(
                    "Cached model metadata {} is empty",
                    file.display()
                ))));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(CachedModel::Missing(Some(format!(
                    "Cached model metadata {} is missing",
                    file.display()
                ))));
            }
            Err(error) => return Err(error.into()),
        }
    }
    // Explicit source overrides supply their own metadata and template.
    if matches!(
        requirements,
        CacheRequirements::Repository | CacheRequirements::Product
    ) {
        if let Some(reason) = ferrum_models::hf_download::inspect_cached_metadata_inventory(path)? {
            return Ok(CachedModel::Missing(Some(reason)));
        }
    }
    Ok(CachedModel::Ready(ResolvedModelSource {
        original: requested_model.to_owned(),
        local_path: path.to_owned(),
        format,
        from_cache: true,
    }))
}

/// A present main ref selects one revision, even when that revision needs
/// repair. Only a cache without a ref may use another available snapshot.
pub(crate) fn snapshot_candidates(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let snapshots = model_dir.join("snapshots");
    match std::fs::read_to_string(model_dir.join("refs/main")) {
        Ok(revision) => {
            let revision = revision.trim();
            if revision.is_empty()
                || matches!(revision, "." | "..")
                || revision.contains(['/', '\\'])
            {
                return Err(FerrumError::model("Invalid cached model refs/main"));
            }
            return Ok(vec![snapshots.join(revision)]);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let entries = match std::fs::read_dir(snapshots) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

pub(super) fn inspect_cached_model(
    cache_dir: &Path,
    model_id: &str,
    requirements: CacheRequirements,
) -> Result<CachedModel> {
    let model_dir = cache_dir
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")));
    let mut incomplete = None;
    let mut invalid = None;
    for snapshot in snapshot_candidates(&model_dir)? {
        match inspect_snapshot(&snapshot, model_id, requirements) {
            Ok(CachedModel::Ready(source)) => return Ok(CachedModel::Ready(source)),
            Ok(CachedModel::Missing(reason)) => {
                if incomplete.is_none() {
                    incomplete = reason;
                }
            }
            Err(error) => {
                if invalid.is_none() {
                    invalid = Some(error);
                }
            }
        }
    }
    if let Some(error) = invalid {
        return Err(error);
    }
    Ok(CachedModel::Missing(incomplete))
}

/// `list` checks weights and recorded transfers without assuming every model
/// uses tokenizer.json. GGUF repositories may carry only weights, with metadata
/// supplied by their curated alias from an independent repository.
pub(crate) fn model_cache_ready(model_dir: &Path) -> bool {
    snapshot_candidates(model_dir).is_ok_and(|snapshots| {
        snapshots.into_iter().any(|path| {
            if matches!(
                inspect_snapshot(&path, "cached model", CacheRequirements::Repository),
                Ok(CachedModel::Ready(_))
            ) && has_tokenizer_assets(&path)
                && nonempty_file(&path.join("config.json"))
            {
                return true;
            }
            std::fs::read_dir(&path).is_ok_and(|files| {
                files.flatten().any(|file| {
                    matches!(
                        inspect_cached_weights(&file.path()),
                        Ok(CachedWeights::Ready(
                            ferrum_models::source::ModelFormat::GGUF
                        ))
                    )
                })
            })
        })
    })
}

fn nonempty_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
}

fn has_tokenizer_assets(path: &Path) -> bool {
    nonempty_file(&path.join("tokenizer.json"))
        || nonempty_file(&path.join("tokenizer.model"))
        || (nonempty_file(&path.join("vocab.json")) && nonempty_file(&path.join("merges.txt")))
}

#[cfg(test)]
mod tests;
