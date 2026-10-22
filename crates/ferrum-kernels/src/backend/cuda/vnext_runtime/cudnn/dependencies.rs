//! Explicit cuDNN JIT dependencies. No search directory or environment mutation.
//!
//! File identity is checked before native admission; the caller subsequently
//! obtains the existing opaque grant before load. Deployment files must remain
//! immutable between hashing, loading and use, just like the cuDNN bundle.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::mem::ManuallyDrop;
use std::path::PathBuf;

use libloading::Library;
use sha2::{Digest, Sha256};

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod linux;

const MAX_DEPENDENCIES: usize = 16;

pub(super) fn validate_paths(paths: &[PathBuf]) -> Result<(), String> {
    if paths.len() > MAX_DEPENDENCIES || paths.iter().any(|path| !path.is_absolute()) {
        return Err(format!(
            "cuDNN dependency_libraries requires at most {MAX_DEPENDENCIES} explicit absolute file paths"
        ));
    }
    if paths.iter().collect::<BTreeSet<_>>().len() != paths.len() {
        return Err("cuDNN dependency_libraries contains a duplicate path".into());
    }
    Ok(())
}

pub(super) struct Selection {
    paths: Vec<PathBuf>,
    fingerprint: [u8; 32],
}

impl Selection {
    pub(super) fn prepare(paths: &[PathBuf]) -> Result<Self, String> {
        validate_paths(paths)?;
        let mut canonical_paths = Vec::with_capacity(paths.len());
        let mut unique = BTreeSet::new();
        let mut digest = Sha256::new();
        digest.update(b"cudnn-explicit-dependencies-rtld-now-local-soname-v1");
        digest.update((paths.len() as u64).to_le_bytes());
        let mut chunk = [0_u8; 64 * 1024];
        for path in paths {
            let canonical = path
                .canonicalize()
                .map_err(|error| format!("cuDNN dependency {}: {error}", path.display()))?;
            if !unique.insert(canonical.clone()) {
                return Err(format!(
                    "cuDNN duplicate dependency target {}",
                    path.display()
                ));
            }
            let mut file = File::open(&canonical)
                .map_err(|error| format!("cuDNN dependency {}: {error}", canonical.display()))?;
            let metadata = file.metadata().map_err(|error| error.to_string())?;
            if !metadata.is_file() {
                return Err(format!(
                    "cuDNN dependency is not a file: {}",
                    canonical.display()
                ));
            }
            let path_bytes = canonical.as_os_str().as_encoded_bytes();
            digest.update((path_bytes.len() as u64).to_le_bytes());
            digest.update(path_bytes);
            digest.update(metadata.len().to_le_bytes());
            loop {
                let count = file.read(&mut chunk).map_err(|error| error.to_string())?;
                if count == 0 {
                    break;
                }
                digest.update(&chunk[..count]);
            }
            canonical_paths.push(canonical);
        }
        Ok(Self {
            paths: canonical_paths,
            fingerprint: digest.finalize().into(),
        })
    }

    pub(super) fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    pub(super) fn validate_loader_support(&self) -> Result<(), String> {
        if !self.paths.is_empty() && !cfg!(all(target_os = "linux", target_env = "gnu")) {
            return Err(
                "explicit cuDNN dependencies currently require the Linux GNU loader".into(),
            );
        }
        Ok(())
    }

    /// Caller must first retain capacity and the active CUDA context. Once any
    /// native initializer has run, failure does not prove cleanup: keep every
    /// successfully opened handle alongside the caller's fail-closed owners.
    pub(super) unsafe fn load(&self) -> Result<Vec<Library>, String> {
        let mut libraries = ManuallyDrop::new(Vec::with_capacity(self.paths.len()));
        for path in &self.paths {
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            libraries.push(unsafe { linux::load_checked(path) }?);
            #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
            return Err(format!(
                "explicit cuDNN dependency {} requires the Linux GNU loader",
                path.display()
            ));
        }
        // Vec drops elements in order. Release later-loaded dependants before
        // their earlier-loaded dependencies after the owning cuDNN library.
        libraries.reverse();
        Ok(unsafe { ManuallyDrop::take(&mut libraries) })
    }
}

#[cfg(test)]
mod tests;
