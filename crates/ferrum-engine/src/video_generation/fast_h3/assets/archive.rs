use super::{FastH3ArchivePath, FastH3AssetLimits};
use anyhow::{ensure, Context, Result};
use ferrum_quantization::SafetensorsArchive;
use std::{collections::BTreeSet, fs::File, io::Read, path::Path};

pub(super) fn bounded_read(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    ensure!(
        maximum > 0 && maximum < isize::MAX as usize,
        "invalid file read capacity"
    );
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        file.metadata()?.len() <= maximum as u64,
        "{} exceeds metadata capacity",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= maximum,
        "{} grew beyond metadata capacity",
        path.display()
    );
    Ok(bytes)
}

/// Import extent, not a memory allocator, reservation or replacement ledger.
pub(super) struct ImportExtent {
    limits: FastH3AssetLimits,
    payload_bytes: u64,
}
impl ImportExtent {
    pub(super) fn new(limits: FastH3AssetLimits) -> Self {
        Self {
            limits,
            payload_bytes: 0,
        }
    }
    pub(super) fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub(super) fn open(&mut self, source: &FastH3ArchivePath) -> Result<SafetensorsArchive> {
        let (path, indexed) = match source {
            FastH3ArchivePath::File { path } => (path, false),
            FastH3ArchivePath::Index { path } => (path, true),
        };
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("archive needs a UTF-8 basename")?;
        portable_name(name)?;
        let directory = path.parent().context("archive lacks parent directory")?;
        let shards = if indexed {
            let bytes = bounded_read(path, self.limits.metadata.max_file_bytes)?;
            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
            let weights = value
                .get("weight_map")
                .and_then(|v| v.as_object())
                .context("archive index lacks weight_map")?;
            ensure!(
                !weights.is_empty() && weights.len() <= self.limits.maximum_tensors_per_archive,
                "index tensor inventory exceeds capacity"
            );
            let mut names = BTreeSet::new();
            for value in weights.values() {
                let name = value.as_str().context("shard name must be a string")?;
                portable_name(name)?;
                ensure!(
                    name.ends_with(".safetensors"),
                    "index must name safetensors shards"
                );
                names.insert(name.to_owned());
            }
            names
        } else {
            ensure!(
                name.ends_with(".safetensors"),
                "explicit archive must be a safetensors file"
            );
            BTreeSet::from([name.to_owned()])
        };
        ensure!(
            !shards.is_empty() && shards.len() <= self.limits.maximum_shards_per_archive,
            "archive shard count exceeds capacity"
        );
        let mut payload_bytes = self.payload_bytes;
        let mut tensors = 0usize;
        for shard in &shards {
            let mut file =
                File::open(directory.join(shard)).with_context(|| format!("open shard {shard}"))?;
            let mut prefix = [0; 8];
            file.read_exact(&mut prefix)?;
            let header_size = u64::from_le_bytes(prefix);
            ensure!(
                header_size > 0 && header_size <= self.limits.metadata.max_file_bytes as u64,
                "safetensors header exceeds metadata capacity"
            );
            let payload = file
                .metadata()?
                .len()
                .checked_sub(header_size + 8)
                .context("safetensors header exceeds file extent")?;
            payload_bytes = payload_bytes
                .checked_add(payload)
                .context("source payload overflow")?;
            ensure!(
                payload_bytes <= self.limits.maximum_total_source_bytes,
                "total checkpoint source bytes exceed capacity"
            );
            let mut bytes = vec![0; usize::try_from(header_size)?];
            file.read_exact(&mut bytes)?;
            let header: serde_json::Value = serde_json::from_slice(&bytes)?;
            let header = header
                .as_object()
                .context("safetensors header must be an object")?;
            let count = header
                .keys()
                .filter(|k| k.as_str() != "__metadata__")
                .count();
            tensors = tensors
                .checked_add(count)
                .context("tensor count overflow")?;
            ensure!(
                tensors <= self.limits.maximum_tensors_per_archive,
                "shard tensor inventory exceeds capacity"
            );
        }
        let archive = if indexed {
            SafetensorsArchive::open_with_index(directory, name)?
        } else {
            SafetensorsArchive::open_single_file(directory, name)?
        };
        ensure!(
            archive.tensor_count() == tensors,
            "mapped inventory differs from inspected immutable shards"
        );
        self.payload_bytes = payload_bytes;
        Ok(archive)
    }
}
fn portable_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && !matches!(name, "." | "..")
            && !name.contains(['/', '\\', ':'])
            && !name.chars().any(char::is_control),
        "archive names must be portable basenames"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
