use super::*;
use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

impl ArtifactLoadLimits {
    pub fn validate(&self) -> Result<(), ArtifactError> {
        let bytes = [
            self.max_manifest_bytes,
            self.max_file_bytes,
            self.max_total_bytes,
        ];
        let counts = [
            self.max_files,
            self.max_jsonl_records,
            self.max_cells,
            self.max_pairs_per_cell,
            self.max_requests_per_repeat,
            self.max_total_visible_gaps,
            self.max_memory_samples,
        ];
        if bytes.contains(&0)
            || counts.contains(&0)
            || self.max_manifest_bytes > 16 << 20
            || self.max_file_bytes > 256 << 20
            || self.max_total_bytes > 1 << 30
            || self.max_files > 4096
            || self.max_jsonl_records > 4096
            || self.max_cells > 1024
            || self.max_pairs_per_cell > 1024
            || self.max_requests_per_repeat > 1_000_000
            || self.max_total_visible_gaps > 16_000_000
            || self.max_memory_samples > 2_000_000
        {
            return Err(err(
                "artifact resource limits must be positive and within hard ceilings",
            ));
        }
        Ok(())
    }
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, ArtifactError> {
    if !std::fs::metadata(path)
        .map_err(|e| err(format!("metadata {}: {e}", path.display())))?
        .is_file()
    {
        return Err(err("artifact must be a regular file"));
    }
    let file = File::open(path).map_err(|e| err(format!("open {}: {e}", path.display())))?;
    let meta = file
        .metadata()
        .map_err(|e| err(format!("opened artifact metadata: {e}")))?;
    if !meta.is_file() || meta.len() > limit {
        return Err(err("artifact exceeds byte limit or is not regular"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| err(format!("read artifact: {e}")))?;
    if bytes.len() as u64 > limit || bytes.len() as u64 != meta.len() {
        return Err(err(
            "artifact changed size while reading or exceeds byte limit",
        ));
    }
    Ok(bytes)
}

pub(super) fn parse<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ArtifactError> {
    serde_json::from_slice(bytes).map_err(|e| err(format!("artifact JSON: {e}")))
}

pub(super) fn jsonl_rows(bytes: &[u8], max_records: usize) -> Result<Vec<&[u8]>, ArtifactError> {
    let mut rows = Vec::new();
    for line in bytes.split(|&byte| byte == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if rows.len() == max_records {
            return Err(err("JSONL record limit exceeded"));
        }
        rows.push(line);
    }
    if rows.is_empty() {
        return Err(err("artifact JSONL contains no records"));
    }
    Ok(rows)
}

pub(super) struct Reader<'a> {
    root: PathBuf,
    pub(super) limits: &'a ArtifactLoadLimits,
    total_bytes: u64,
    cache: BTreeMap<PathBuf, (Arc<Vec<u8>>, VerifiedArtifact)>,
    manifest: VerifiedArtifact,
}

impl<'a> Reader<'a> {
    pub(super) fn open(
        path: &Path,
        limits: &'a ArtifactLoadLimits,
    ) -> Result<(Self, ComparisonArtifactManifest), ArtifactError> {
        limits.validate()?;
        let path = path
            .canonicalize()
            .map_err(|e| err(format!("resolve manifest: {e}")))?;
        let root = path
            .parent()
            .ok_or_else(|| err("manifest has no parent"))?
            .to_owned();
        let bytes = read_bounded(&path, limits.max_manifest_bytes.min(limits.max_total_bytes))?;
        let manifest = VerifiedArtifact {
            path: PathBuf::from(
                path.file_name()
                    .ok_or_else(|| err("manifest has no name"))?,
            ),
            sha256: digest(&bytes),
            bytes: bytes.len() as u64,
        };
        let input = parse(&bytes)?;
        Ok((
            Self {
                root,
                limits,
                total_bytes: manifest.bytes,
                cache: BTreeMap::new(),
                manifest,
            },
            input,
        ))
    }

    pub(super) fn read(
        &mut self,
        reference: &ArtifactFileRef,
    ) -> Result<Arc<Vec<u8>>, ArtifactError> {
        if !valid_digest(&reference.sha256)
            || reference.bytes == 0
            || reference.bytes > self.limits.max_file_bytes
        {
            return Err(err("invalid artifact hash or declared byte size"));
        }
        if reference.path.as_os_str().is_empty()
            || reference
                .path
                .components()
                .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
        {
            return Err(err(
                "artifact path must stay relative to the manifest directory",
            ));
        }
        let path = self
            .root
            .join(&reference.path)
            .canonicalize()
            .map_err(|e| {
                err(format!(
                    "resolve artifact {}: {e}",
                    reference.path.display()
                ))
            })?;
        if !path.starts_with(&self.root) {
            return Err(err("artifact symlink escapes manifest directory"));
        }
        if let Some((bytes, verified)) = self.cache.get(&path) {
            if verified.bytes != reference.bytes
                || !same_digest(&verified.sha256, &reference.sha256)
            {
                return Err(err("conflicting references to one artifact"));
            }
            return Ok(Arc::clone(bytes));
        }
        if self.cache.len() >= self.limits.max_files {
            return Err(err("referenced artifact count limit exceeded"));
        }
        let total = self
            .total_bytes
            .checked_add(reference.bytes)
            .filter(|&n| n <= self.limits.max_total_bytes)
            .ok_or_else(|| err("total artifact byte limit exceeded"))?;
        let bytes = read_bounded(&path, reference.bytes)?;
        let hash = digest(&bytes);
        if bytes.len() as u64 != reference.bytes || !same_digest(&hash, &reference.sha256) {
            return Err(err(format!(
                "artifact size/SHA-256 mismatch: {}",
                reference.path.display()
            )));
        }
        let verified = VerifiedArtifact {
            path: path
                .strip_prefix(&self.root)
                .expect("checked containment")
                .to_owned(),
            sha256: hash,
            bytes: bytes.len() as u64,
        };
        let bytes = Arc::new(bytes);
        self.cache.insert(path, (Arc::clone(&bytes), verified));
        self.total_bytes = total;
        Ok(bytes)
    }

    pub(super) fn manifest_hash(&self) -> &str {
        &self.manifest.sha256
    }
    pub(super) fn verified_files(&self) -> Vec<VerifiedArtifact> {
        std::iter::once(self.manifest.clone())
            .chain(self.cache.values().map(|(_, file)| file.clone()))
            .collect()
    }
}
