use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrum_types::{FerrumError, Result};

use super::identity::MetadataFileDigest;

/// Explicit CPU read limits for metadata preflight. The tokenizer has a
/// separate ceiling because published vocabularies are larger than configs.
/// These bound retained input bytes; parser scratch space and allocation
/// capacity are additional host memory, so this is not an RSS admission limit.
#[derive(Debug, Clone, Copy)]
pub struct PipelineSourceLimits {
    configuration_bytes: u64,
    tokenizer_bytes: u64,
    total_bytes: u64,
}

impl PipelineSourceLimits {
    pub fn new(configuration_bytes: u64, tokenizer_bytes: u64, total_bytes: u64) -> Result<Self> {
        for bytes in [configuration_bytes, tokenizer_bytes, total_bytes] {
            // Reserve one byte for bounded oversize detection. Rust byte
            // allocations, including that sentinel, must fit isize::MAX.
            if bytes == 0 || bytes >= isize::MAX as u64 {
                return Err(FerrumError::config(
                    "pipeline metadata limits must fit non-zero host allocations",
                ));
            }
        }
        Ok(Self {
            configuration_bytes,
            tokenizer_bytes,
            total_bytes,
        })
    }

    pub const fn configuration_bytes(self) -> u64 {
        self.configuration_bytes
    }

    pub const fn tokenizer_bytes(self) -> u64 {
        self.tokenizer_bytes
    }

    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

impl Default for PipelineSourceLimits {
    fn default() -> Self {
        Self {
            configuration_bytes: 1024 * 1024,
            tokenizer_bytes: 32 * 1024 * 1024,
            total_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Exact bytes and digest consumed by parsing. This is not a path to reopen.
#[derive(Clone)]
pub struct RetainedMetadataFile {
    digest: MetadataFileDigest,
    bytes: Arc<[u8]>,
}

impl std::fmt::Debug for RetainedMetadataFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedMetadataFile")
            .field("digest", &self.digest)
            .finish()
    }
}

impl RetainedMetadataFile {
    pub fn path(&self) -> &str {
        self.digest.path()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn digest(&self) -> &MetadataFileDigest {
        &self.digest
    }
}

pub(super) struct MetadataReader {
    root: PathBuf,
    limits: PipelineSourceLimits,
    retained_bytes: u64,
}

impl MetadataReader {
    pub(super) fn new(root: &Path, limits: PipelineSourceLimits) -> Result<Self> {
        let root = fs::canonicalize(root).map_err(|error| {
            FerrumError::model(format!(
                "resolve pipeline metadata root {}: {error}",
                root.display()
            ))
        })?;
        if !root.is_dir() {
            return Err(FerrumError::model(
                "pipeline metadata root must be a directory",
            ));
        }
        Ok(Self {
            root,
            limits,
            retained_bytes: 0,
        })
    }

    pub(super) fn root(&self) -> &Path {
        &self.root
    }

    pub(super) fn read_optional(&mut self, relative: &str) -> Result<Option<RetainedMetadataFile>> {
        validate_relative_path(relative)?;
        match fs::symlink_metadata(self.root.join(relative)) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(FerrumError::model(format!(
                "inspect optional {relative}: {error}"
            ))),
            Ok(_) => self.read(relative, false).map(Some),
        }
    }

    pub(super) fn read(&mut self, relative: &str, tokenizer: bool) -> Result<RetainedMetadataFile> {
        validate_relative_path(relative)?;
        let path = fs::canonicalize(self.root.join(relative)).map_err(|error| {
            FerrumError::model(format!("required pipeline metadata {relative}: {error}"))
        })?;
        if !path.starts_with(&self.root) {
            return Err(FerrumError::model(format!(
                "pipeline metadata {relative} resolves outside its source root"
            )));
        }
        let metadata = fs::metadata(&path).map_err(|error| {
            FerrumError::model(format!("inspect pipeline metadata {relative}: {error}"))
        })?;
        if !metadata.is_file() {
            return Err(FerrumError::model(format!(
                "pipeline metadata {relative} must be a regular file"
            )));
        }
        let per_file = if tokenizer {
            self.limits.tokenizer_bytes
        } else {
            self.limits.configuration_bytes
        };
        let remaining = self
            .limits
            .total_bytes
            .checked_sub(self.retained_bytes)
            .ok_or_else(|| FerrumError::model("pipeline metadata size accounting overflow"))?;
        let limit = per_file.min(remaining);
        if metadata.len() > limit {
            return Err(FerrumError::model(format!(
                "pipeline metadata {relative} exceeds its remaining read limit ({limit} bytes)"
            )));
        }
        // The root is a local immutable package, not a hostile concurrently
        // mutated filesystem. Retaining the bytes prevents later path edits
        // from changing metadata between preflight and component preparation.
        let file = File::open(&path).map_err(|error| {
            FerrumError::model(format!("open pipeline metadata {relative}: {error}"))
        })?;
        if !file
            .metadata()
            .map_err(|error| FerrumError::model(format!("inspect opened {relative}: {error}")))?
            .is_file()
        {
            return Err(FerrumError::model(format!(
                "pipeline metadata {relative} must be a regular file"
            )));
        }
        let mut bytes = Vec::new();
        file.take(limit + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| {
                FerrumError::model(format!("read pipeline metadata {relative}: {error}"))
            })?;
        if bytes.is_empty() {
            return Err(FerrumError::model(format!(
                "required pipeline metadata {relative} is empty"
            )));
        }
        if bytes.len() as u64 > limit {
            return Err(FerrumError::model(format!(
                "pipeline metadata {relative} exceeds its remaining read limit ({limit} bytes)"
            )));
        }
        self.retained_bytes = self
            .retained_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| FerrumError::model("pipeline metadata size accounting overflow"))?;
        Ok(RetainedMetadataFile {
            digest: MetadataFileDigest::from_bytes(relative, &bytes)?,
            bytes: bytes.into(),
        })
    }
}

pub(super) fn validate_relative_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.contains(['\\', ':'])
        || path.split('/').any(|part| matches!(part, "" | "." | ".."))
        || Path::new(path).is_absolute()
    {
        return Err(FerrumError::model(format!(
            "invalid package-relative metadata path {path:?}"
        )));
    }
    Ok(())
}
