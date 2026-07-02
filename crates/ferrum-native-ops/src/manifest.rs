//! Native operator manifest loading.

use std::fs;
use std::path::Path;

use ferrum_types::NativeOperatorManifest;

use crate::resolver::{NativeOperatorResolveError, Result};

pub fn load_manifest(path: impl AsRef<Path>) -> Result<NativeOperatorManifest> {
    let path = path.as_ref();
    let raw =
        fs::read_to_string(path).map_err(|source| NativeOperatorResolveError::ManifestRead {
            path: path.to_path_buf(),
            source,
        })?;
    let manifest: NativeOperatorManifest =
        serde_json::from_str(&raw).map_err(|source| NativeOperatorResolveError::ManifestJson {
            path: path.to_path_buf(),
            source,
        })?;
    manifest
        .validate()
        .map_err(NativeOperatorResolveError::ManifestInvalid)?;
    Ok(manifest)
}
