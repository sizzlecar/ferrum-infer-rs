//! Deterministic component archives inside one immutable local model package.
use super::*;
use std::{fs, io::Read};

/// Standard names chosen by the component's declared library, not a model name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetensorsNaming {
    Transformers,
    Diffusers,
}
impl SafetensorsNaming {
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Transformers => ("model.safetensors", "model.safetensors.index.json"),
            Self::Diffusers => (
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.safetensors.index.json",
            ),
        }
    }
}

impl SafetensorsArchive {
    /// Opens exactly one standard single-file or indexed archive. Both present
    /// is an error; missing/invalid selection never falls back to another name.
    /// Every component, index and shard must resolve inside the canonical root.
    /// As with PipelineSourceBundle, the package must remain immutable while
    /// opening it; external HF-cache symlinks are not authorized by this API.
    /// Payloads are mapped, not copied or scanned for hashes.
    pub fn open_standard_component(
        root: impl AsRef<Path>,
        component_directory: &str,
        naming: SafetensorsNaming,
        maximum_index_bytes: u64,
    ) -> Result<Self> {
        if maximum_index_bytes == 0 || maximum_index_bytes >= isize::MAX as u64 {
            return Err(FerrumError::config(
                "weight index read limit must be positive and addressable",
            ));
        }
        if component_directory.is_empty()
            || component_directory.contains(['\\', ':'])
            || component_directory.chars().any(char::is_control)
            || component_directory
                .split('/')
                .any(|part| matches!(part, "" | "." | ".."))
        {
            return Err(FerrumError::model(
                "component directory must be a portable package-relative path",
            ));
        }
        let root = fs::canonicalize(root.as_ref())
            .map_err(|e| FerrumError::io(format!("resolve model root: {e}")))?;
        let directory = within_root(&root, &root.join(component_directory))?;
        if !root.is_dir() || !directory.is_dir() {
            return Err(FerrumError::model(
                "model root and component must be directories",
            ));
        }
        let (single_name, index_name) = naming.names();
        let single = directory.join(single_name);
        let index = directory.join(index_name);
        let shards = match (present(&single)?, present(&index)?) {
            (true, true) => return Err(FerrumError::model(format!(
                "ambiguous component archive: both {single_name} and {index_name} exist in {component_directory}"
            ))),
            (false, false) => return Err(FerrumError::model(format!(
                "missing component archive: expected {component_directory}/{single_name} or {component_directory}/{index_name}"
            ))),
            (true, false) => vec![(single_name.to_owned(), within_root(&root, &single)?)],
            (false, true) => {
                let index = within_root(&root, &index)?;
                let file = File::open(&index).map_err(|e| FerrumError::io(format!("open {index:?}: {e}")))?;
                if file.metadata().map_err(|e| FerrumError::io(e.to_string()))?.len() > maximum_index_bytes {
                    return Err(FerrumError::model(format!("weight index {index:?} exceeds its read limit")));
                }
                let mut bytes = Vec::new();
                file.take(maximum_index_bytes + 1).read_to_end(&mut bytes)
                    .map_err(|e| FerrumError::io(format!("read {index:?}: {e}")))?;
                if bytes.len() as u64 > maximum_index_bytes {
                    return Err(FerrumError::model(format!("weight index {index:?} exceeds its read limit")));
                }
                index_shards_from_bytes(&directory, &index, &bytes)?
                    .into_iter().map(|(name, path)| Ok((name, within_root(&root, &path)?)))
                    .collect::<Result<Vec<_>>>()?
            }
        };
        for (_, path) in &shards {
            if !path.is_file() {
                return Err(FerrumError::model(format!(
                    "safetensors shard {path:?} must be a regular file"
                )));
            }
        }
        Self::open_shards(shards)
    }
}

fn present(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true), // Dangling links are present and then fail resolution.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(FerrumError::io(format!("inspect {path:?}: {e}"))),
    }
}

fn within_root(root: &Path, path: &Path) -> Result<PathBuf> {
    let resolved =
        fs::canonicalize(path).map_err(|e| FerrumError::io(format!("resolve {path:?}: {e}")))?;
    if !resolved.starts_with(root) {
        return Err(FerrumError::model(format!(
            "component source {path:?} resolves outside its model root"
        )));
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests;
