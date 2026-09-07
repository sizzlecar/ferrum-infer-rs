//! Remember sidecars selected by Ferrum without imposing them on other HF caches.

use super::{selection, HfFileInfo};
use ferrum_types::{FerrumError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

#[derive(Deserialize, Serialize)]
struct MetadataInventory {
    version: u32,
    files: BTreeMap<String, Option<u64>>,
}

/// Return the first recoverable missing/mismatched sidecar recorded by Ferrum.
///
/// No inventory means no additional requirements: external HF caches need not
/// contain optional tokenizer or template files. Completed inventories remain
/// on disk and permit cache hits without network access. Invalid inventories,
/// non-file entries, and unexpected IO errors are not download cache misses.
pub fn inspect_cached_metadata_inventory(snapshot: &Path) -> Result<Option<String>> {
    inspect_metadata_inventory(snapshot, None)
}

/// Inspect only previously recorded sidecars needed by this source consumer.
///
/// A metadata-only download may select fewer files than another operation on
/// the same repository. Missing files outside `filenames` do not require repair
/// by this consumer; malformed inventories still return their original error.
/// Unrecorded optional filenames are not new cache requirements.
pub fn inspect_cached_metadata_selection(
    snapshot: &Path,
    filenames: &[&str],
) -> Result<Option<String>> {
    for filename in filenames {
        validate_metadata_path(filename)?;
    }
    inspect_metadata_inventory(snapshot, Some(filenames))
}

fn inspect_metadata_inventory(
    snapshot: &Path,
    selection: Option<&[&str]>,
) -> Result<Option<String>> {
    let Some(directory) = inventory_directory(snapshot) else {
        return Ok(None);
    };
    let entries = match std::fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(io_error(
                "read metadata inventory directory",
                &directory,
                error,
            ))
        }
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            io_error("read metadata inventory directory entry", &directory, error)
        })?;
        let name = entry.file_name();
        if name
            .to_str()
            .is_some_and(|name| name.starts_with("sidecars-") && name.ends_with(".json"))
        {
            paths.push(entry.path());
        }
    }
    paths.sort();

    let mut missing = None;
    for path in paths {
        let metadata = std::fs::metadata(&path)
            .map_err(|error| io_error("inspect metadata inventory", &path, error))?;
        if !metadata.is_file() {
            return Err(FerrumError::model(format!(
                "Metadata inventory {} is not a file",
                path.display()
            )));
        }
        let bytes = std::fs::read(&path)
            .map_err(|error| io_error("read metadata inventory", &path, error))?;
        let inventory: MetadataInventory = serde_json::from_slice(&bytes).map_err(|error| {
            FerrumError::model(format!(
                "Invalid metadata inventory {}: {error}",
                path.display()
            ))
        })?;
        validate_inventory(&inventory).map_err(|error| {
            FerrumError::model(format!(
                "Invalid metadata inventory {}: {error}",
                path.display()
            ))
        })?;
        // Inspect every record even after a missing file, so corrupt inventories
        // or non-file entries cannot be hidden by directory iteration order.
        for (name, expected_size) in inventory.files {
            if selection.is_some_and(|filenames| !filenames.contains(&name.as_str())) {
                continue;
            }
            let file = snapshot.join(name);
            match std::fs::metadata(&file) {
                Ok(metadata) if !metadata.is_file() => {
                    return Err(FerrumError::model(format!(
                        "Recorded model metadata {} is not a file",
                        file.display()
                    )));
                }
                Ok(metadata) => {
                    if let Some(expected) = expected_size {
                        if metadata.len() != expected {
                            missing.get_or_insert_with(|| {
                                format!(
                                    "Cached model metadata {} has {} bytes, expected {expected}",
                                    file.display(),
                                    metadata.len()
                                )
                            });
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    missing.get_or_insert_with(|| {
                        format!("Cached model metadata {} is missing", file.display())
                    });
                }
                Err(error) => {
                    return Err(io_error("inspect recorded model metadata", &file, error));
                }
            }
        }
    }
    Ok(missing)
}

pub(super) async fn record_selected_metadata(
    snapshot: &Path,
    selected: &[&HfFileInfo],
) -> Result<()> {
    let mut files = BTreeMap::new();
    for file in selected {
        if file.file_type.as_deref() == Some("directory")
            || selection::is_weight_path(&file.path)
            || file.path.ends_with(".gguf")
        {
            continue;
        }
        if let Some(previous) = files.insert(file.path.clone(), file.size) {
            if previous != file.size {
                return Err(FerrumError::model(format!(
                    "Conflicting sizes for selected model metadata {:?}",
                    file.path
                )));
            }
        }
    }
    if files.is_empty() {
        return Ok(());
    }
    let inventory = MetadataInventory { version: 1, files };
    validate_inventory(&inventory)?;
    let directory = inventory_directory(snapshot).ok_or_else(|| {
        FerrumError::model(format!(
            "Cannot record model metadata outside an immutable HF snapshot: {}",
            snapshot.display()
        ))
    })?;
    let bytes = serde_json::to_vec(&inventory).map_err(|error| {
        FerrumError::model(format!("Failed to serialize metadata inventory: {error}"))
    })?;
    let destination = directory.join(format!("sidecars-{:x}.json", Sha256::digest(&bytes)));
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(|error| io_error("create metadata inventory directory", &directory, error))?;

    // Each writer owns its temporary file. Concurrent selections have distinct
    // destinations; equal selections atomically publish identical contents.
    let temporary = directory.join(format!(".sidecars-{}.tmp", uuid::Uuid::new_v4()));
    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .await
            .map_err(|error| io_error("create temporary metadata inventory", &temporary, error))?;
        file.write_all(&bytes)
            .await
            .map_err(|error| io_error("write metadata inventory", &temporary, error))?;
        file.sync_all()
            .await
            .map_err(|error| io_error("flush metadata inventory", &temporary, error))?;
        drop(file);
        tokio::fs::rename(&temporary, &destination)
            .await
            .map_err(|error| io_error("publish metadata inventory", &destination, error))
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&temporary).await;
    }
    result
}

fn inventory_directory(snapshot: &Path) -> Option<PathBuf> {
    let commit = snapshot.file_name()?.to_str()?;
    if commit.len() != 40 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let snapshots = snapshot.parent()?;
    if snapshots.file_name()? != "snapshots" {
        return None;
    }
    Some(snapshots.parent()?.join(".ferrum-downloads").join(commit))
}

fn validate_inventory(inventory: &MetadataInventory) -> Result<()> {
    if inventory.version != 1 || inventory.files.is_empty() {
        return Err(FerrumError::model(
            "Metadata inventory requires version 1 and a nonempty file selection",
        ));
    }
    for name in inventory.files.keys() {
        validate_metadata_path(name)?;
    }
    Ok(())
}

fn validate_metadata_path(name: &str) -> Result<()> {
    if name.is_empty()
        || name
            .chars()
            .any(|c| c.is_control() || matches!(c, '\\' | '%' | '?' | '#' | ':'))
        || name
            .split('/')
            .any(|component| matches!(component, "" | "." | ".."))
        || selection::is_weight_path(name)
        || name.ends_with(".gguf")
    {
        return Err(FerrumError::model(format!(
            "Invalid metadata path {name:?}: expected a portable relative sidecar path without URL metacharacters"
        )));
    }
    Ok(())
}

fn io_error(action: &str, path: &Path, error: std::io::Error) -> FerrumError {
    FerrumError::io(format!("Failed to {action} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const COMMIT: &str = "1234567890abcdef1234567890abcdef12345678";

    fn snapshot(cache: &Path) -> PathBuf {
        let snapshot = cache
            .join("hub/models--fixture--model/snapshots")
            .join(COMMIT);
        fs::create_dir_all(&snapshot).unwrap();
        snapshot
    }

    fn file(path: &str, size: Option<u64>) -> HfFileInfo {
        HfFileInfo {
            path: path.to_owned(),
            size,
            file_type: Some("file".to_owned()),
        }
    }

    fn put_inventory(snapshot: &Path, bytes: &[u8]) -> PathBuf {
        let directory = inventory_directory(snapshot).unwrap();
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("sidecars-test.json");
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn external_cache_without_inventory_does_not_require_optional_sidecars() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        fs::write(snapshot.join("config.json"), b"{}").unwrap();
        fs::write(snapshot.join("tokenizer.json"), b"{}").unwrap();
        assert_eq!(inspect_cached_metadata_inventory(&snapshot).unwrap(), None);
        assert_eq!(
            inspect_cached_metadata_inventory(cache.path()).unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn selected_optional_template_is_required_until_complete_and_record_can_remain() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        let template = file("chat_template.jinja", Some(4));
        let weights = file("auxiliary/model.safetensors", Some(100));
        record_selected_metadata(&snapshot, &[&weights, &template])
            .await
            .unwrap();
        let reason = inspect_cached_metadata_inventory(&snapshot)
            .unwrap()
            .unwrap();
        assert!(reason.contains("chat_template.jinja"), "{reason}");
        fs::write(snapshot.join("chat_template.jinja"), b"x").unwrap();
        let reason = inspect_cached_metadata_inventory(&snapshot)
            .unwrap()
            .unwrap();
        assert!(reason.contains("expected 4"), "{reason}");
        fs::write(snapshot.join("chat_template.jinja"), b"text").unwrap();
        assert_eq!(inspect_cached_metadata_inventory(&snapshot).unwrap(), None);
        assert!(inventory_directory(&snapshot).unwrap().is_dir());
        assert!(!snapshot.join("auxiliary/model.safetensors").exists());
    }

    #[tokio::test]
    async fn unknown_sizes_and_nested_metadata_use_file_presence() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        let tokenizer = file("nested/tokenizer_config.json", None);
        record_selected_metadata(&snapshot, &[&tokenizer])
            .await
            .unwrap();
        fs::create_dir_all(snapshot.join("nested")).unwrap();
        fs::write(snapshot.join(&tokenizer.path), b"{}").unwrap();
        assert_eq!(inspect_cached_metadata_inventory(&snapshot).unwrap(), None);
        fs::remove_file(snapshot.join(&tokenizer.path)).unwrap();
        fs::create_dir(snapshot.join(&tokenizer.path)).unwrap();
        assert!(inspect_cached_metadata_inventory(&snapshot).is_err());
    }

    #[tokio::test]
    async fn concurrent_distinct_selections_do_not_erase_each_other() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        let config = file("config.json", Some(2));
        let template = file("chat_template.jinja", Some(4));
        let first = [&config];
        let second = [&template];
        let (left, right, repeated) = tokio::join!(
            record_selected_metadata(&snapshot, &first),
            record_selected_metadata(&snapshot, &second),
            record_selected_metadata(&snapshot, &second),
        );
        left.unwrap();
        right.unwrap();
        repeated.unwrap();
        fs::write(snapshot.join("config.json"), b"{}").unwrap();
        let reason = inspect_cached_metadata_inventory(&snapshot)
            .unwrap()
            .unwrap();
        assert!(reason.contains("chat_template.jinja"), "{reason}");
        fs::write(snapshot.join("chat_template.jinja"), b"text").unwrap();
        assert_eq!(inspect_cached_metadata_inventory(&snapshot).unwrap(), None);
    }

    #[tokio::test]
    async fn metadata_inventory_is_independent_of_repository_weight_scope() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        let config = file("config.json", Some(2));
        let template = file("chat_template.jinja", Some(4));
        let weights = file("model.safetensors", Some(10));
        let auxiliary = file("encoder/model.bin", Some(20));
        record_selected_metadata(&snapshot, &[&config, &weights, &template])
            .await
            .unwrap();
        record_selected_metadata(&snapshot, &[&auxiliary, &template, &weights, &config])
            .await
            .unwrap();
        fs::write(snapshot.join("config.json"), b"{}").unwrap();
        fs::write(snapshot.join("chat_template.jinja"), b"text").unwrap();
        assert_eq!(inspect_cached_metadata_inventory(&snapshot).unwrap(), None);
    }

    #[tokio::test]
    async fn metadata_selection_repairs_its_template_without_requiring_other_sidecars() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        let config = file("config.json", Some(2));
        let template = file("chat_template.jinja", Some(4));
        let preprocessor = file("preprocessor_config.json", Some(2));
        record_selected_metadata(&snapshot, &[&config, &template, &preprocessor])
            .await
            .unwrap();
        fs::write(snapshot.join("config.json"), b"{}").unwrap();
        fs::write(snapshot.join("chat_template.jinja"), b"text").unwrap();
        let selected = [
            "config.json",
            "chat_template.jinja",
            "generation_config.json",
        ];
        assert_eq!(
            inspect_cached_metadata_selection(&snapshot, &selected).unwrap(),
            None
        );
        let reason = inspect_cached_metadata_inventory(&snapshot)
            .unwrap()
            .unwrap();
        assert!(reason.contains("preprocessor_config.json"), "{reason}");

        fs::remove_file(snapshot.join("chat_template.jinja")).unwrap();
        let reason = inspect_cached_metadata_selection(&snapshot, &selected)
            .unwrap()
            .unwrap();
        assert!(reason.contains("chat_template.jinja"), "{reason}");
        fs::write(snapshot.join("chat_template.jinja"), b"text").unwrap();
        assert_eq!(
            inspect_cached_metadata_selection(&snapshot, &selected).unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn metadata_selection_preserves_inventory_errors_after_a_missing_selected_file() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        let template = file("chat_template.jinja", Some(4));
        record_selected_metadata(&snapshot, &[&template])
            .await
            .unwrap();
        let selected = ["chat_template.jinja"];
        for bytes in [
            b"{".as_slice(),
            br#"{"version":1,"files":{"../unselected.json":2}}"#,
        ] {
            put_inventory(&snapshot, bytes);
            assert!(inspect_cached_metadata_selection(&snapshot, &selected).is_err());
        }
    }

    #[test]
    fn malformed_inventory_is_an_error_and_not_a_cache_miss() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        for bytes in [
            b"{".as_slice(),
            br#"{"version":2,"files":{"config.json":2}}"#,
            br#"{"version":1,"files":{}}"#,
            br#"{"version":1,"files":{"config.json":"two"}}"#,
        ] {
            put_inventory(&snapshot, bytes);
            assert!(inspect_cached_metadata_inventory(&snapshot).is_err());
        }
        let path = put_inventory(&snapshot, b"{}");
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(inspect_cached_metadata_inventory(&snapshot).is_err());
    }

    #[tokio::test]
    async fn inventory_rejects_unsafe_paths_and_weight_entries() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        for name in [
            "../config.json",
            "/config.json",
            "nested/../config.json",
            "nested\\config.json",
            "C:/config.json",
            "nested/%2e%2e/config.json",
            "config?query.json",
            "config#fragment.json",
            "config\0.json",
            "model.safetensors",
            "model.gguf",
        ] {
            let bytes = serde_json::to_vec(&MetadataInventory {
                version: 1,
                files: BTreeMap::from([(name.to_owned(), Some(2))]),
            })
            .unwrap();
            put_inventory(&snapshot, &bytes);
            let error = inspect_cached_metadata_inventory(&snapshot).unwrap_err();
            assert!(
                error.to_string().contains("Invalid metadata path"),
                "{error}"
            );
        }
        let unsafe_file = file("../config.json", None);
        assert!(record_selected_metadata(&snapshot, &[&unsafe_file])
            .await
            .is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn metadata_accepts_hf_blob_symlinks_and_detects_broken_links() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        let config = file("config.json", Some(2));
        record_selected_metadata(&snapshot, &[&config])
            .await
            .unwrap();
        let blobs = snapshot.parent().unwrap().parent().unwrap().join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        fs::write(blobs.join("config"), b"{}").unwrap();
        std::os::unix::fs::symlink("../../blobs/config", snapshot.join("config.json")).unwrap();
        assert_eq!(inspect_cached_metadata_inventory(&snapshot).unwrap(), None);
        fs::remove_file(blobs.join("config")).unwrap();
        let reason = inspect_cached_metadata_inventory(&snapshot)
            .unwrap()
            .unwrap();
        assert!(reason.contains("config.json"), "{reason}");
    }

    #[test]
    fn unpublished_temporary_inventory_is_not_read() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = snapshot(cache.path());
        let directory = inventory_directory(&snapshot).unwrap();
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join(".sidecars-interrupted.tmp"), b"{").unwrap();
        assert_eq!(inspect_cached_metadata_inventory(&snapshot).unwrap(), None);
    }
}
