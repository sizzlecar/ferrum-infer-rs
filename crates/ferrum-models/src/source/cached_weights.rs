use super::ModelFormat;
use ferrum_types::{FerrumError, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(crate) const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
pub(crate) const SAFETENSORS_SINGLE: &str = "model.safetensors";

/// Whether cached weights can be selected or need a recoverable transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CachedWeights {
    Ready(ModelFormat),
    Absent,
    Incomplete { reason: String },
}

/// Inspect the locally cached weight layout without loading tensor contents.
///
/// A canonical single Safetensors file takes precedence over an index. An
/// authoritative index must be valid and every referenced shard must be a
/// nonempty file. Missing or empty files can be repaired by downloading;
/// malformed indexes, invalid paths and unexpected IO failures return errors.
/// Config and tokenizer sources are separate concerns and need not be colocated
/// with the weights.
pub fn inspect_cached_weights(path: &Path) -> Result<CachedWeights> {
    if path.is_file()
        && path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        return inspect_weight_file(path, ModelFormat::GGUF);
    }

    let single = path.join(SAFETENSORS_SINGLE);
    if single.is_file() {
        return inspect_weight_file(&single, ModelFormat::SafeTensors);
    }

    let index = path.join(SAFETENSORS_INDEX);
    if cached_entry_exists(&index)? {
        let metadata = match std::fs::metadata(&index) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(CachedWeights::Incomplete {
                    reason: format!("Cached weight index {} is missing", index.display()),
                });
            }
            Err(error) => {
                return Err(FerrumError::io(format!(
                    "Failed to inspect cached weight index {}: {error}",
                    index.display()
                )));
            }
        };
        if !metadata.is_file() {
            return Err(FerrumError::model(format!(
                "Cached weight index {} is not a file",
                index.display()
            )));
        }
        let bytes = std::fs::read(&index).map_err(|error| {
            FerrumError::io(format!(
                "Failed to read cached weight index {}: {error}",
                index.display()
            ))
        })?;
        let shards = parse_safetensors_index(&bytes).map_err(|error| {
            FerrumError::model(format!(
                "Invalid cached weight index {}: {error}",
                index.display()
            ))
        })?;
        for shard in shards {
            let shard_path = path.join(&shard);
            match inspect_weight_file(&shard_path, ModelFormat::SafeTensors)? {
                CachedWeights::Incomplete { reason } => {
                    return Ok(CachedWeights::Incomplete {
                        reason: format!("{reason}; referenced by {}", index.display()),
                    });
                }
                CachedWeights::Ready(_) => {}
                CachedWeights::Absent => unreachable!("inspecting a required weight file"),
            }
        }
        return Ok(CachedWeights::Ready(ModelFormat::SafeTensors));
    }

    let pytorch = path.join("pytorch_model.bin");
    if pytorch.is_file() {
        return inspect_weight_file(&pytorch, ModelFormat::PyTorchBin);
    }

    for (candidate, format) in [
        (single, ModelFormat::SafeTensors),
        (pytorch, ModelFormat::PyTorchBin),
    ] {
        if cached_entry_exists(&candidate)? {
            return inspect_weight_file(&candidate, format);
        }
    }
    Ok(CachedWeights::Absent)
}

fn cached_entry_exists(path: &Path) -> Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(FerrumError::io(format!(
            "Failed to inspect cached path {}: {error}",
            path.display()
        ))),
    }
}

fn inspect_weight_file(path: &Path, format: ModelFormat) -> Result<CachedWeights> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CachedWeights::Incomplete {
                reason: format!("Cached weight {} is missing", path.display()),
            });
        }
        Err(error) => {
            return Err(FerrumError::io(format!(
                "Failed to inspect cached weight {}: {error}",
                path.display()
            )));
        }
    };
    if !metadata.is_file() {
        return Err(FerrumError::model(format!(
            "Cached weight {} is not a file",
            path.display()
        )));
    }
    if metadata.len() == 0 {
        return Ok(CachedWeights::Incomplete {
            reason: format!("Cached weight {} is empty", path.display()),
        });
    }
    Ok(CachedWeights::Ready(format))
}

#[derive(Deserialize)]
struct SafeTensorsIndex {
    weight_map: BTreeMap<String, String>,
}

/// Parse the same portable shard names for remote selection and local checks.
pub(crate) fn parse_safetensors_index(bytes: &[u8]) -> Result<BTreeSet<String>> {
    let index: SafeTensorsIndex = serde_json::from_slice(bytes)
        .map_err(|error| FerrumError::model(format!("Invalid {SAFETENSORS_INDEX}: {error}")))?;
    if index.weight_map.is_empty() {
        return Err(FerrumError::model(format!(
            "Invalid {SAFETENSORS_INDEX}: weight_map must not be empty"
        )));
    }

    let mut shards = BTreeSet::new();
    for path in index.weight_map.into_values() {
        if !valid_shard_path(&path) {
            return Err(FerrumError::model(format!(
                "Invalid shard path {path:?} in {SAFETENSORS_INDEX}: expected a portable relative .safetensors path without URL metacharacters"
            )));
        }
        shards.insert(path);
    }
    Ok(shards)
}

fn valid_shard_path(path: &str) -> bool {
    !path.is_empty()
        && path.ends_with(".safetensors")
        && !path
            .chars()
            .any(|c| c.is_control() || matches!(c, '\\' | '%' | '?' | '#' | ':'))
        && path
            .split('/')
            .all(|component| !matches!(component, "" | "." | ".."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    fn write_index(directory: &Path, shards: &[&str]) {
        let weight_map: BTreeMap<String, &str> = shards
            .iter()
            .enumerate()
            .map(|(index, shard)| (format!("weight_{index}"), *shard))
            .collect();
        fs::write(
            directory.join(SAFETENSORS_INDEX),
            serde_json::to_vec(&json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn cache_without_a_supported_weight_layout_is_not_ready() {
        let directory = tempdir().unwrap();
        assert_eq!(
            inspect_cached_weights(directory.path()).unwrap(),
            CachedWeights::Absent
        );
        fs::write(directory.path().join("config.json"), b"{}").unwrap();
        fs::write(directory.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(directory.path().join("orphan.safetensors"), b"weight").unwrap();
        assert_eq!(
            inspect_cached_weights(directory.path()).unwrap(),
            CachedWeights::Absent
        );
    }

    #[test]
    fn single_file_has_priority_without_reading_unused_index_or_tensor_data() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join(SAFETENSORS_SINGLE), b"weight").unwrap();
        fs::write(directory.path().join(SAFETENSORS_INDEX), b"invalid JSON").unwrap();
        assert_eq!(
            inspect_cached_weights(directory.path()).unwrap(),
            CachedWeights::Ready(ModelFormat::SafeTensors)
        );
    }

    #[test]
    fn index_requires_every_shard_to_be_a_file_and_accepts_nested_references() {
        let directory = tempdir().unwrap();
        let nested = "weights/second.safetensors";
        write_index(directory.path(), &["first.safetensors", nested, nested]);
        fs::write(directory.path().join("first.safetensors"), b"first").unwrap();

        let CachedWeights::Incomplete { reason } =
            inspect_cached_weights(directory.path()).unwrap()
        else {
            panic!("missing shard must be repairable");
        };
        assert!(reason.contains("second.safetensors"), "{reason}");
        assert!(reason.contains("missing"), "{reason}");

        let shard_path = directory.path().join(nested);
        fs::create_dir_all(&shard_path).unwrap();
        let error = inspect_cached_weights(directory.path()).unwrap_err();
        assert!(error.to_string().contains("not a file"), "{error}");

        fs::remove_dir(&shard_path).unwrap();
        fs::write(shard_path, b"second").unwrap();
        assert_eq!(
            inspect_cached_weights(directory.path()).unwrap(),
            CachedWeights::Ready(ModelFormat::SafeTensors)
        );
    }

    #[test]
    fn local_index_uses_strict_download_index_validation() {
        let directory = tempdir().unwrap();
        for index in [
            b"{".as_slice(),
            br#"{"weight_map":{}}"#,
            br#"{"weight_map":{"weight":7}}"#,
            br#"{"weight_map":{"weight":"../part.safetensors"}}"#,
            br#"{"weight_map":{"weight":"C:/part.safetensors"}}"#,
            br#"{"weight_map":{"weight":"weights/%2e%2e/part.safetensors"}}"#,
        ] {
            fs::write(directory.path().join(SAFETENSORS_INDEX), index).unwrap();
            let error = inspect_cached_weights(directory.path()).unwrap_err();
            assert!(error.to_string().contains("Invalid"), "{error}");
            assert!(error.to_string().contains(SAFETENSORS_INDEX), "{error}");
        }
    }

    #[test]
    fn a_directory_named_like_single_weights_does_not_hide_a_valid_index() {
        let directory = tempdir().unwrap();
        fs::create_dir(directory.path().join(SAFETENSORS_SINGLE)).unwrap();
        let error = inspect_cached_weights(directory.path()).unwrap_err();
        assert!(error.to_string().contains(SAFETENSORS_SINGLE), "{error}");
        write_index(directory.path(), &["part.safetensors"]);
        fs::write(directory.path().join("part.safetensors"), b"weight").unwrap();
        assert_eq!(
            inspect_cached_weights(directory.path()).unwrap(),
            CachedWeights::Ready(ModelFormat::SafeTensors)
        );
    }

    #[test]
    fn a_directory_named_like_the_index_is_not_ready() {
        let directory = tempdir().unwrap();
        fs::create_dir(directory.path().join(SAFETENSORS_INDEX)).unwrap();
        let error = inspect_cached_weights(directory.path()).unwrap_err();
        assert!(error.to_string().contains(SAFETENSORS_INDEX), "{error}");
        assert!(error.to_string().contains("not a file"), "{error}");
    }

    #[test]
    fn pytorch_single_file_and_explicit_gguf_remain_supported() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("pytorch_model.bin"), b"weight").unwrap();
        assert_eq!(
            inspect_cached_weights(directory.path()).unwrap(),
            CachedWeights::Ready(ModelFormat::PyTorchBin)
        );
        let gguf = directory.path().join("model.GGUF");
        fs::write(&gguf, b"weight").unwrap();
        assert_eq!(
            inspect_cached_weights(&gguf).unwrap(),
            CachedWeights::Ready(ModelFormat::GGUF)
        );
    }

    #[test]
    fn empty_finalized_weights_are_repairable() {
        let directory = tempdir().unwrap();
        for name in [SAFETENSORS_SINGLE, "pytorch_model.bin", "model.gguf"] {
            let file = directory.path().join(name);
            fs::write(&file, b"").unwrap();
            let source = if name.ends_with(".gguf") {
                file.as_path()
            } else {
                directory.path()
            };
            let CachedWeights::Incomplete { reason } = inspect_cached_weights(source).unwrap()
            else {
                panic!("empty {name} must be repairable");
            };
            assert!(reason.contains("empty"), "{reason}");
            fs::remove_file(file).unwrap();
        }
        write_index(directory.path(), &["part.safetensors"]);
        fs::write(directory.path().join("part.safetensors"), b"").unwrap();
        let CachedWeights::Incomplete { reason } =
            inspect_cached_weights(directory.path()).unwrap()
        else {
            panic!("empty shard must be repairable");
        };
        assert!(
            reason.contains("part.safetensors") && reason.contains("empty"),
            "{reason}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn hf_snapshot_links_may_point_to_shared_blobs_outside_the_snapshot() {
        use std::os::unix::fs::symlink;

        let cache = tempdir().unwrap();
        let snapshot = cache.path().join("snapshots/revision");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir(cache.path().join("blobs")).unwrap();
        fs::write(cache.path().join("blobs/etag"), b"weight").unwrap();
        write_index(&snapshot, &["part.safetensors"]);
        symlink("../../blobs/etag", snapshot.join("part.safetensors")).unwrap();
        assert_eq!(
            inspect_cached_weights(&snapshot).unwrap(),
            CachedWeights::Ready(ModelFormat::SafeTensors)
        );
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_symlinks_must_resolve_to_files() {
        use std::os::unix::fs::symlink;

        let directory = tempdir().unwrap();
        let blob = directory.path().join("blob");
        let shard = directory.path().join("part.safetensors");
        write_index(directory.path(), &["part.safetensors"]);
        symlink(&blob, &shard).unwrap();
        let CachedWeights::Incomplete { reason } =
            inspect_cached_weights(directory.path()).unwrap()
        else {
            panic!("broken shard link must be repairable");
        };
        assert!(reason.contains("part.safetensors"), "{reason}");

        fs::write(&blob, b"weight").unwrap();
        assert_eq!(
            inspect_cached_weights(directory.path()).unwrap(),
            CachedWeights::Ready(ModelFormat::SafeTensors)
        );

        fs::remove_file(directory.path().join(SAFETENSORS_INDEX)).unwrap();
        fs::remove_file(&blob).unwrap();
        symlink(&blob, directory.path().join(SAFETENSORS_SINGLE)).unwrap();
        let CachedWeights::Incomplete { reason } =
            inspect_cached_weights(directory.path()).unwrap()
        else {
            panic!("broken single-file link must be repairable");
        };
        assert!(reason.contains(SAFETENSORS_SINGLE), "{reason}");

        fs::remove_file(directory.path().join(SAFETENSORS_SINGLE)).unwrap();
        symlink(&blob, directory.path().join(SAFETENSORS_INDEX)).unwrap();
        let CachedWeights::Incomplete { reason } =
            inspect_cached_weights(directory.path()).unwrap()
        else {
            panic!("broken index link must be repairable");
        };
        assert!(reason.contains(SAFETENSORS_INDEX), "{reason}");
    }
}
