//! Opt-in sampling of allocations owned by the selected device runtime.

use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

mod cuda;
pub use cuda::{CudaMemoryDomains, CudaMemoryReading, CudaMemoryUnavailable};

/// Device allocation observations are sampled at this fixed interval.
/// A sampled high-water mark can miss peaks shorter than this interval.
pub const DEVICE_MEMORY_SAMPLE_INTERVAL_MS: u64 = 250;

/// Write device allocation observations from runtime initialization before
/// weight loading through shutdown. Device initialization itself is excluded.
/// This is independent of process RSS and the configured memory budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemorySamplingConfig {
    pub jsonl_path: PathBuf,
}

impl DeviceMemorySamplingConfig {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.jsonl_path.as_os_str().is_empty() {
            return Err("device memory JSONL path must not be empty".to_owned());
        }
        Ok(())
    }

    /// A capture must have its own output file. Other diagnostic writers may
    /// truncate or unlink their destination after the sampler has started.
    pub fn validate_output_paths<'a>(
        &self,
        output_paths: impl IntoIterator<Item = (&'a str, &'a Path)>,
    ) -> std::result::Result<(), String> {
        self.validate()?;
        let capture = normalize_output_path(&self.jsonl_path)?;
        for (label, path) in output_paths {
            if capture == normalize_output_path(path)? {
                return Err(format!(
                    "device-memory JSONL path {} conflicts with {label}; use separate output files",
                    self.jsonl_path.display()
                ));
            }
        }
        Ok(())
    }
}

// Resolve existing components before processing `..`: a symlink's parent is
// the parent of its target, not the lexical parent of the symlink name. Missing
// suffixes need no filesystem mutation to compare future output destinations.
fn normalize_output_path(path: &Path) -> std::result::Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve diagnostic output directory: {error}"))?
            .join(path)
    };
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                resolved.push(name);
                match std::fs::canonicalize(&resolved) {
                    Ok(canonical) => resolved = canonical,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(format!(
                            "resolve diagnostic output path {}: {error}",
                            path.display()
                        ));
                    }
                }
            }
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_memory_sampling_config_rejects_empty_path_and_round_trips() {
        let empty = DeviceMemorySamplingConfig {
            jsonl_path: PathBuf::new(),
        };
        assert!(empty.validate().is_err());
        let config = DeviceMemorySamplingConfig {
            jsonl_path: PathBuf::from("evidence/device-memory.jsonl"),
        };
        config.validate().unwrap();
        let restored: DeviceMemorySamplingConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn device_memory_sampling_defaults_off_in_existing_runtime_config() {
        let default = crate::RuntimeKnobs::default();
        assert!(default.device_memory_sampling.is_none());
        let mut json = serde_json::to_value(default).unwrap();
        json.as_object_mut()
            .unwrap()
            .remove("device_memory_sampling");
        let restored: crate::RuntimeKnobs = serde_json::from_value(json).unwrap();
        assert!(restored.device_memory_sampling.is_none());
    }

    #[test]
    fn device_memory_sampling_output_paths_reject_aliases_but_allow_distinct_files() {
        let config = DeviceMemorySamplingConfig {
            jsonl_path: PathBuf::from("evidence/device-memory.jsonl"),
        };
        let alias = std::env::current_dir()
            .unwrap()
            .join("evidence/nested/../device-memory.jsonl");
        let error = config
            .validate_output_paths([("profile", alias.as_path())])
            .unwrap_err();
        assert!(error.contains("conflicts with profile"));
        config
            .validate_output_paths([("profile", Path::new("evidence/profile.jsonl"))])
            .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn device_memory_sampling_output_paths_resolve_symlink_parents_before_dotdot() {
        let root = std::env::temp_dir().join(format!(
            "ferrum-device-memory-path-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("actual/nested")).unwrap();
        std::os::unix::fs::symlink(root.join("actual/nested"), root.join("alias")).unwrap();
        let config = DeviceMemorySamplingConfig {
            jsonl_path: root.join("actual/capture.jsonl"),
        };
        let alias = root.join("alias/../capture.jsonl");
        assert!(config
            .validate_output_paths([("scheduler", alias.as_path())])
            .is_err());
        assert!(!config.jsonl_path.exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
