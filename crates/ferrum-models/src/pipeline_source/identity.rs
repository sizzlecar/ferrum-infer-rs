use ferrum_types::{FerrumError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::declaration::PipelineDeclaration;
use super::files::validate_relative_path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MetadataFileDigest {
    path: String,
    bytes: u64,
    sha256: String,
}

impl MetadataFileDigest {
    pub fn new(path: impl Into<String>, bytes: u64, sha256: impl Into<String>) -> Result<Self> {
        let path = path.into();
        let sha256 = sha256.into();
        validate_relative_path(&path)?;
        if bytes == 0 || !lower_hex(&sha256, 64) {
            return Err(FerrumError::model(format!(
                "metadata digest for {path:?} requires non-zero bytes and a lowercase SHA256"
            )));
        }
        Ok(Self {
            path,
            bytes,
            sha256,
        })
    }

    pub(super) fn from_bytes(path: &str, bytes: &[u8]) -> Result<Self> {
        Self::new(
            path,
            bytes.len() as u64,
            format!("{:x}", Sha256::digest(bytes)),
        )
    }

    pub fn path(&self) -> &str {
        &self.path
    }
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Identity of the consumed metadata only; weights and local paths are absent.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PipelineSourceManifest {
    schema_version: u32,
    declaration: PipelineDeclaration,
    files: Vec<MetadataFileDigest>,
    fingerprint: String,
}

impl PipelineSourceManifest {
    pub(super) fn new(
        declaration: PipelineDeclaration,
        files: Vec<MetadataFileDigest>,
    ) -> Result<Self> {
        let files = sorted_files(files)?;
        // Domain and version prevent interpreting a metadata digest as a model
        // weight identity, while omitting root paths makes copied bundles equal.
        let encoded =
            serde_json::to_vec(&("ferrum.pipeline-metadata", 1_u32, &declaration, &files))
                .map_err(|error| {
                    FerrumError::internal(format!("serialize pipeline metadata manifest: {error}"))
                })?;
        let fingerprint = format!("{:x}", Sha256::digest(encoded));
        Ok(Self {
            schema_version: 1,
            declaration,
            files,
            fingerprint,
        })
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }
    pub fn declaration(&self) -> &PipelineDeclaration {
        &self.declaration
    }
    pub fn files(&self) -> &[MetadataFileDigest] {
        &self.files
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

/// Independently supplied expectation for every metadata file a bundle consumes.
/// A repository/revision string alone never establishes source identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PipelineMetadataLock {
    schema_version: u32,
    repository_id: String,
    revision: String,
    files: Vec<MetadataFileDigest>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MetadataLockWire {
    schema_version: u32,
    repository_id: String,
    revision: String,
    files: Vec<MetadataDigestWire>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MetadataDigestWire {
    path: String,
    bytes: u64,
    sha256: String,
}

impl PipelineMetadataLock {
    pub fn new(
        repository_id: impl Into<String>,
        revision: impl Into<String>,
        files: Vec<MetadataFileDigest>,
    ) -> Result<Self> {
        let repository_id = repository_id.into();
        let revision = revision.into();
        let parts: Vec<_> = repository_id.split('/').collect();
        if parts.len() > 2
            || parts.iter().any(|part| {
                matches!(*part, "" | "." | "..")
                    || !part.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
                    })
            })
            || !lower_hex(&revision, 40)
        {
            return Err(FerrumError::model("pipeline metadata lock requires a repository id and full lowercase commit revision"));
        }
        Ok(Self {
            schema_version: 1,
            repository_id,
            revision,
            files: sorted_files(files)?,
        })
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let wire: MetadataLockWire = serde_json::from_slice(bytes).map_err(|error| {
            FerrumError::model(format!("invalid pipeline metadata lock: {error}"))
        })?;
        if wire.schema_version != 1 {
            return Err(FerrumError::unsupported(
                "unsupported pipeline metadata lock version",
            ));
        }
        let files = wire
            .files
            .into_iter()
            .map(|file| MetadataFileDigest::new(file.path, file.bytes, file.sha256))
            .collect::<Result<Vec<_>>>()?;
        Self::new(wire.repository_id, wire.revision, files)
    }

    pub fn repository_id(&self) -> &str {
        &self.repository_id
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub fn files(&self) -> &[MetadataFileDigest] {
        &self.files
    }

    pub(super) fn verify(&self, actual: &PipelineSourceManifest) -> Result<()> {
        if self.files.len() != actual.files.len()
            || self
                .files
                .iter()
                .zip(&actual.files)
                .any(|(expected, actual)| expected.path != actual.path)
        {
            return Err(FerrumError::model(
                "pipeline metadata lock must cover exactly the consumed metadata files",
            ));
        }
        for (expected, actual) in self.files.iter().zip(&actual.files) {
            if expected != actual {
                return Err(FerrumError::model(format!(
                    "pipeline metadata {} differs from pinned revision {} (size or SHA256 mismatch)",
                    actual.path, self.revision
                )));
            }
        }
        Ok(())
    }
}

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sorted_files(mut files: Vec<MetadataFileDigest>) -> Result<Vec<MetadataFileDigest>> {
    files.sort_by(|left, right| left.path.cmp(&right.path));
    if files.is_empty() || files.windows(2).any(|pair| pair[0].path == pair[1].path) {
        return Err(FerrumError::model(
            "pipeline metadata manifest requires non-empty, unique file identities",
        ));
    }
    Ok(files)
}
