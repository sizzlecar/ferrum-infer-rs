//! Content-addressed cache for native build outputs.
//!
//! Cargo assigns a different `OUT_DIR` to each profile and target identity.
//! Native CUDA outputs are substantially more expensive than the Rust leaf
//! that caused the profile change, so the build script uses this cache to move
//! only signature-identical artifacts across those boundaries.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const NATIVE_BUILD_ARTIFACT_CACHE_SCHEMA_VERSION: u32 = 1;
const ENTRY_LOCK_WAIT: Duration = Duration::from_secs(30);
const ENTRY_LOCK_POLL: Duration = Duration::from_millis(25);
static NEXT_TEMPORARY_FILE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBuildArtifactSpec {
    artifact_id: String,
    file_name: String,
    input_signature: String,
    input_signature_sha256: String,
}

impl NativeBuildArtifactSpec {
    pub fn new(
        artifact_id: impl Into<String>,
        file_name: impl Into<String>,
        input_signature: impl Into<String>,
    ) -> Result<Self, NativeBuildArtifactCacheError> {
        let artifact_id = artifact_id.into();
        let file_name = file_name.into();
        let input_signature = input_signature.into();
        validate_artifact_id(&artifact_id)?;
        validate_file_name(&file_name)?;
        if input_signature.is_empty() {
            return Err(NativeBuildArtifactCacheError::InvalidInputSignature);
        }
        let input_signature_sha256 = sha256_bytes(input_signature.as_bytes());
        Ok(Self {
            artifact_id,
            file_name,
            input_signature,
            input_signature_sha256,
        })
    }

    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn input_signature(&self) -> &str {
        &self.input_signature
    }

    pub fn input_signature_sha256(&self) -> &str {
        &self.input_signature_sha256
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NativeBuildArtifactCacheManifest {
    pub schema_version: u32,
    pub artifact_id: String,
    pub file_name: String,
    pub input_signature: String,
    pub input_signature_sha256: String,
    pub artifact_sha256: String,
    pub artifact_size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBuildArtifactCacheReceipt {
    pub cache_entry: PathBuf,
    pub artifact_path: PathBuf,
    pub manifest_path: PathBuf,
    pub artifact_sha256: String,
    pub artifact_size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeBuildArtifactLookup {
    Hit(NativeBuildArtifactCacheReceipt),
    Miss { reason: &'static str },
}

#[derive(Debug, Clone)]
pub struct NativeBuildArtifactCache {
    root: PathBuf,
}

impl NativeBuildArtifactCache {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, NativeBuildArtifactCacheError> {
        let root = root.into();
        if !root.is_absolute() {
            return Err(NativeBuildArtifactCacheError::CacheRootNotAbsolute(root));
        }
        fs::create_dir_all(&root).map_err(|source| {
            NativeBuildArtifactCacheError::CreateDirectory {
                path: root.clone(),
                source,
            }
        })?;
        let metadata = fs::symlink_metadata(&root).map_err(|source| {
            NativeBuildArtifactCacheError::Metadata {
                path: root.clone(),
                source,
            }
        })?;
        if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
            return Err(NativeBuildArtifactCacheError::CacheRootNotDirectory(root));
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn restore(
        &self,
        spec: &NativeBuildArtifactSpec,
        destination: impl AsRef<Path>,
    ) -> Result<NativeBuildArtifactLookup, NativeBuildArtifactCacheError> {
        let entry = self.entry_dir(spec);
        match self.validate_entry(spec, &entry)? {
            Some(receipt) => {
                let mut staged = stage_verified_copy(&receipt.artifact_path, destination.as_ref())?;
                if staged.sha256 != receipt.artifact_sha256 {
                    return Err(NativeBuildArtifactCacheError::ArtifactSha256Mismatch {
                        path: receipt.artifact_path,
                        expected: receipt.artifact_sha256,
                        actual: staged.sha256.clone(),
                    });
                }
                if staged.size_bytes != receipt.artifact_size_bytes {
                    return Err(NativeBuildArtifactCacheError::ArtifactSizeMismatch {
                        path: receipt.artifact_path,
                        expected: receipt.artifact_size_bytes,
                        actual: staged.size_bytes,
                    });
                }
                staged.commit(destination.as_ref())?;
                Ok(NativeBuildArtifactLookup::Hit(receipt))
            }
            None => Ok(NativeBuildArtifactLookup::Miss {
                reason: "entry-absent",
            }),
        }
    }

    pub fn publish(
        &self,
        spec: &NativeBuildArtifactSpec,
        source: impl AsRef<Path>,
    ) -> Result<NativeBuildArtifactCacheReceipt, NativeBuildArtifactCacheError> {
        let source = source.as_ref();
        let entry = self.entry_dir(spec);
        fs::create_dir_all(&entry).map_err(|source| {
            NativeBuildArtifactCacheError::CreateDirectory {
                path: entry.clone(),
                source,
            }
        })?;
        let _lock = EntryLock::acquire(&entry)?;
        let artifact_path = entry.join(&spec.file_name);
        let manifest_path = entry.join("manifest.json");
        let mut staged = stage_verified_copy(source, &artifact_path)?;
        let source_size = staged.size_bytes;
        let source_sha256 = staged.sha256.clone();

        if let Some(existing) = self.validate_entry(spec, &entry)? {
            if existing.artifact_sha256 != source_sha256
                || existing.artifact_size_bytes != source_size
            {
                return Err(NativeBuildArtifactCacheError::NondeterministicArtifact {
                    artifact_id: spec.artifact_id.clone(),
                    input_signature_sha256: spec.input_signature_sha256.clone(),
                    existing_sha256: existing.artifact_sha256,
                    candidate_sha256: source_sha256,
                });
            }
            return Ok(existing);
        }

        if artifact_path.exists() && !manifest_path.exists() {
            fs::remove_file(&artifact_path).map_err(|source| {
                NativeBuildArtifactCacheError::RemoveIncompleteEntry {
                    path: artifact_path.clone(),
                    source,
                }
            })?;
        }

        let manifest = NativeBuildArtifactCacheManifest {
            schema_version: NATIVE_BUILD_ARTIFACT_CACHE_SCHEMA_VERSION,
            artifact_id: spec.artifact_id.clone(),
            file_name: spec.file_name.clone(),
            input_signature: spec.input_signature.clone(),
            input_signature_sha256: spec.input_signature_sha256.clone(),
            artifact_sha256: source_sha256.clone(),
            artifact_size_bytes: source_size,
        };
        staged.commit(&artifact_path)?;
        atomic_write_json(&manifest_path, &manifest)?;

        self.validate_entry(spec, &entry)?.ok_or_else(|| {
            NativeBuildArtifactCacheError::PublishedEntryMissing {
                path: entry.clone(),
            }
        })
    }

    fn entry_dir(&self, spec: &NativeBuildArtifactSpec) -> PathBuf {
        self.root
            .join(&spec.artifact_id)
            .join(&spec.input_signature_sha256)
    }

    fn validate_entry(
        &self,
        spec: &NativeBuildArtifactSpec,
        entry: &Path,
    ) -> Result<Option<NativeBuildArtifactCacheReceipt>, NativeBuildArtifactCacheError> {
        let manifest_path = entry.join("manifest.json");
        let artifact_path = entry.join(&spec.file_name);
        let manifest_exists = manifest_path.exists();
        let artifact_exists = artifact_path.exists();
        if !manifest_exists && !artifact_exists {
            return Ok(None);
        }
        if !manifest_exists {
            return Ok(None);
        }
        if !artifact_exists {
            return Err(NativeBuildArtifactCacheError::EntryArtifactMissing {
                path: artifact_path,
            });
        }
        validate_regular_file(&manifest_path)?;
        validate_regular_file(&artifact_path)?;
        let raw = fs::read_to_string(&manifest_path).map_err(|source| {
            NativeBuildArtifactCacheError::Read {
                path: manifest_path.clone(),
                source,
            }
        })?;
        let manifest: NativeBuildArtifactCacheManifest =
            serde_json::from_str(&raw).map_err(|source| {
                NativeBuildArtifactCacheError::ManifestJson {
                    path: manifest_path.clone(),
                    source,
                }
            })?;
        validate_manifest(spec, &manifest, &manifest_path)?;
        let actual_size = fs::metadata(&artifact_path)
            .map_err(|source| NativeBuildArtifactCacheError::Metadata {
                path: artifact_path.clone(),
                source,
            })?
            .len();
        if actual_size != manifest.artifact_size_bytes {
            return Err(NativeBuildArtifactCacheError::ArtifactSizeMismatch {
                path: artifact_path,
                expected: manifest.artifact_size_bytes,
                actual: actual_size,
            });
        }
        let actual_sha256 = sha256_file(&artifact_path)?;
        if actual_sha256 != manifest.artifact_sha256 {
            return Err(NativeBuildArtifactCacheError::ArtifactSha256Mismatch {
                path: artifact_path,
                expected: manifest.artifact_sha256,
                actual: actual_sha256,
            });
        }
        Ok(Some(NativeBuildArtifactCacheReceipt {
            cache_entry: entry.to_path_buf(),
            artifact_path,
            manifest_path,
            artifact_sha256: actual_sha256,
            artifact_size_bytes: actual_size,
        }))
    }
}

#[derive(Debug, Error)]
pub enum NativeBuildArtifactCacheError {
    #[error("native build artifact id is invalid: {0:?}")]
    InvalidArtifactId(String),
    #[error("native build artifact file name is invalid: {0:?}")]
    InvalidFileName(String),
    #[error("native build artifact input signature must not be empty")]
    InvalidInputSignature,
    #[error("native build artifact cache root must be absolute: {0}")]
    CacheRootNotAbsolute(PathBuf),
    #[error("native build artifact cache root must be a real directory: {0}")]
    CacheRootNotDirectory(PathBuf),
    #[error("failed to create native build artifact directory {path}: {source}")]
    CreateDirectory { path: PathBuf, source: io::Error },
    #[error("failed to stat native build artifact {path}: {source}")]
    Metadata { path: PathBuf, source: io::Error },
    #[error("native build artifact must be a regular, non-symlink file: {0}")]
    NotRegularFile(PathBuf),
    #[error("failed to read native build artifact {path}: {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("failed to write native build artifact {path}: {source}")]
    Write { path: PathBuf, source: io::Error },
    #[error("failed to parse native build artifact manifest {path}: {source}")]
    ManifestJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("native build artifact manifest mismatch at {path}: {detail}")]
    ManifestMismatch { path: PathBuf, detail: String },
    #[error("native build artifact entry is missing its payload: {path}")]
    EntryArtifactMissing { path: PathBuf },
    #[error("native build artifact size mismatch at {path}: expected {expected}, got {actual}")]
    ArtifactSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("native build artifact sha256 mismatch at {path}: expected {expected}, got {actual}")]
    ArtifactSha256Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error(
        "native build artifact source changed while it was copied from {path}: copied {copied_sha256}, reread {reread_sha256}"
    )]
    SourceChangedDuringCopy {
        path: PathBuf,
        copied_sha256: String,
        reread_sha256: String,
    },
    #[error(
        "native build output is nondeterministic for {artifact_id}/{input_signature_sha256}: existing {existing_sha256}, candidate {candidate_sha256}"
    )]
    NondeterministicArtifact {
        artifact_id: String,
        input_signature_sha256: String,
        existing_sha256: String,
        candidate_sha256: String,
    },
    #[error("failed to acquire native build artifact entry lock {path}: {source}")]
    LockCreate { path: PathBuf, source: io::Error },
    #[error("timed out acquiring native build artifact entry lock: {0}")]
    LockTimeout(PathBuf),
    #[error("failed to remove native build artifact entry lock {path}: {source}")]
    LockRemove { path: PathBuf, source: io::Error },
    #[error("failed to remove incomplete native build artifact {path}: {source}")]
    RemoveIncompleteEntry { path: PathBuf, source: io::Error },
    #[error("published native build artifact entry is missing: {path}")]
    PublishedEntryMissing { path: PathBuf },
}

fn validate_artifact_id(value: &str) -> Result<(), NativeBuildArtifactCacheError> {
    let valid = !value.is_empty()
        && value.len() <= 128
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(NativeBuildArtifactCacheError::InvalidArtifactId(
            value.to_string(),
        ))
    }
}

fn validate_file_name(value: &str) -> Result<(), NativeBuildArtifactCacheError> {
    let path = Path::new(value);
    let valid = !value.is_empty()
        && value.len() <= 255
        && path.file_name().and_then(|name| name.to_str()) == Some(value)
        && value != "."
        && value != "..";
    if valid {
        Ok(())
    } else {
        Err(NativeBuildArtifactCacheError::InvalidFileName(
            value.to_string(),
        ))
    }
}

fn validate_manifest(
    spec: &NativeBuildArtifactSpec,
    manifest: &NativeBuildArtifactCacheManifest,
    path: &Path,
) -> Result<(), NativeBuildArtifactCacheError> {
    let mut mismatches = Vec::new();
    if manifest.schema_version != NATIVE_BUILD_ARTIFACT_CACHE_SCHEMA_VERSION {
        mismatches.push(format!(
            "schema_version expected {}, got {}",
            NATIVE_BUILD_ARTIFACT_CACHE_SCHEMA_VERSION, manifest.schema_version
        ));
    }
    if manifest.artifact_id != spec.artifact_id {
        mismatches.push(format!(
            "artifact_id expected {:?}, got {:?}",
            spec.artifact_id, manifest.artifact_id
        ));
    }
    if manifest.file_name != spec.file_name {
        mismatches.push(format!(
            "file_name expected {:?}, got {:?}",
            spec.file_name, manifest.file_name
        ));
    }
    if manifest.input_signature != spec.input_signature {
        mismatches.push("input_signature differs".to_string());
    }
    if manifest.input_signature_sha256 != spec.input_signature_sha256 {
        mismatches.push(format!(
            "input_signature_sha256 expected {}, got {}",
            spec.input_signature_sha256, manifest.input_signature_sha256
        ));
    }
    if sha256_bytes(manifest.input_signature.as_bytes()) != manifest.input_signature_sha256 {
        mismatches.push("manifest input_signature sha256 is invalid".to_string());
    }
    if manifest.artifact_sha256.len() != 64
        || !manifest
            .artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        mismatches.push("artifact_sha256 is not lowercase SHA256".to_string());
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(NativeBuildArtifactCacheError::ManifestMismatch {
            path: path.to_path_buf(),
            detail: mismatches.join("; "),
        })
    }
}

fn validate_regular_file(path: &Path) -> Result<(), NativeBuildArtifactCacheError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|source| NativeBuildArtifactCacheError::Metadata {
            path: path.to_path_buf(),
            source,
        })?;
    if metadata.file_type().is_file() && !metadata.file_type().is_symlink() {
        Ok(())
    } else {
        Err(NativeBuildArtifactCacheError::NotRegularFile(
            path.to_path_buf(),
        ))
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, NativeBuildArtifactCacheError> {
    let mut file = File::open(path).map_err(|source| NativeBuildArtifactCacheError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count =
            file.read(&mut buffer)
                .map_err(|source| NativeBuildArtifactCacheError::Read {
                    path: path.to_path_buf(),
                    source,
                })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn temporary_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    let sequence = NEXT_TEMPORARY_FILE.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(
        ".{file_name}.{}.{sequence}.tmp",
        std::process::id()
    ))
}

struct StagedCopy {
    path: PathBuf,
    sha256: String,
    size_bytes: u64,
    committed: bool,
}

impl StagedCopy {
    fn commit(&mut self, destination: &Path) -> Result<(), NativeBuildArtifactCacheError> {
        if destination.exists() {
            fs::remove_file(destination).map_err(|source| {
                NativeBuildArtifactCacheError::Write {
                    path: destination.to_path_buf(),
                    source,
                }
            })?;
        }
        fs::rename(&self.path, destination).map_err(|source| {
            NativeBuildArtifactCacheError::Write {
                path: destination.to_path_buf(),
                source,
            }
        })?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for StagedCopy {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn stage_verified_copy(
    source: &Path,
    destination: &Path,
) -> Result<StagedCopy, NativeBuildArtifactCacheError> {
    validate_regular_file(source)?;
    let parent = destination
        .parent()
        .ok_or_else(|| NativeBuildArtifactCacheError::Write {
            path: destination.to_path_buf(),
            source: io::Error::new(io::ErrorKind::InvalidInput, "destination has no parent"),
        })?;
    fs::create_dir_all(parent).map_err(|source| {
        NativeBuildArtifactCacheError::CreateDirectory {
            path: parent.to_path_buf(),
            source,
        }
    })?;
    let temporary = temporary_path(destination);
    let mut input =
        File::open(source).map_err(|source_error| NativeBuildArtifactCacheError::Read {
            path: source.to_path_buf(),
            source: source_error,
        })?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|source| NativeBuildArtifactCacheError::Write {
            path: temporary.clone(),
            source,
        })?;
    let mut copied_digest = Sha256::new();
    let mut size_bytes = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer).map_err(|source_error| {
            NativeBuildArtifactCacheError::Read {
                path: source.to_path_buf(),
                source: source_error,
            }
        })?;
        if count == 0 {
            break;
        }
        output.write_all(&buffer[..count]).map_err(|source_error| {
            NativeBuildArtifactCacheError::Write {
                path: temporary.clone(),
                source: source_error,
            }
        })?;
        copied_digest.update(&buffer[..count]);
        size_bytes = size_bytes
            .checked_add(count as u64)
            .expect("native artifact size overflow");
    }
    output
        .sync_all()
        .map_err(|source| NativeBuildArtifactCacheError::Write {
            path: temporary.clone(),
            source,
        })?;
    drop(output);

    input
        .seek(SeekFrom::Start(0))
        .map_err(|source_error| NativeBuildArtifactCacheError::Read {
            path: source.to_path_buf(),
            source: source_error,
        })?;
    let mut reread_digest = Sha256::new();
    loop {
        let count = input.read(&mut buffer).map_err(|source_error| {
            NativeBuildArtifactCacheError::Read {
                path: source.to_path_buf(),
                source: source_error,
            }
        })?;
        if count == 0 {
            break;
        }
        reread_digest.update(&buffer[..count]);
    }
    let copied_sha256 = format!("{:x}", copied_digest.finalize());
    let reread_sha256 = format!("{:x}", reread_digest.finalize());
    if copied_sha256 != reread_sha256 {
        let _ = fs::remove_file(&temporary);
        return Err(NativeBuildArtifactCacheError::SourceChangedDuringCopy {
            path: source.to_path_buf(),
            copied_sha256,
            reread_sha256,
        });
    }

    Ok(StagedCopy {
        path: temporary,
        sha256: copied_sha256,
        size_bytes,
        committed: false,
    })
}

fn atomic_write_json(
    path: &Path,
    manifest: &NativeBuildArtifactCacheManifest,
) -> Result<(), NativeBuildArtifactCacheError> {
    let bytes = serde_json::to_vec_pretty(manifest).map_err(|source| {
        NativeBuildArtifactCacheError::ManifestJson {
            path: path.to_path_buf(),
            source,
        }
    })?;
    let temporary = temporary_path(path);
    let mut cleanup = StagedCopy {
        path: temporary.clone(),
        sha256: String::new(),
        size_bytes: 0,
        committed: false,
    };
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|source| NativeBuildArtifactCacheError::Write {
            path: temporary.clone(),
            source,
        })?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|source| NativeBuildArtifactCacheError::Write {
            path: temporary.clone(),
            source,
        })?;
    cleanup.commit(path)
}

struct EntryLock {
    file: File,
    path: PathBuf,
}

impl EntryLock {
    #[cfg(unix)]
    fn acquire(entry: &Path) -> Result<Self, NativeBuildArtifactCacheError> {
        let path = entry.join("publish.lock");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|source| NativeBuildArtifactCacheError::LockCreate {
                path: path.clone(),
                source,
            })?;
        let started = Instant::now();
        loop {
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc == 0 {
                let mut guard = Self {
                    file,
                    path: path.clone(),
                };
                guard
                    .file
                    .set_len(0)
                    .and_then(|_| {
                        writeln!(guard.file, "pid={}", std::process::id())?;
                        guard.file.sync_all()
                    })
                    .map_err(|source| NativeBuildArtifactCacheError::LockCreate {
                        path: path.clone(),
                        source,
                    })?;
                return Ok(guard);
            }
            let source = io::Error::last_os_error();
            if source.kind() != io::ErrorKind::WouldBlock {
                return Err(NativeBuildArtifactCacheError::LockCreate { path, source });
            }
            if started.elapsed() >= ENTRY_LOCK_WAIT {
                return Err(NativeBuildArtifactCacheError::LockTimeout(path));
            }
            thread::sleep(ENTRY_LOCK_POLL);
        }
    }

    #[cfg(not(unix))]
    fn acquire(entry: &Path) -> Result<Self, NativeBuildArtifactCacheError> {
        let path = entry.join("publish.lock");
        let started = Instant::now();
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    let mut guard = Self {
                        file,
                        path: path.clone(),
                    };
                    writeln!(guard.file, "pid={}", std::process::id())
                        .and_then(|_| guard.file.sync_all())
                        .map_err(|source| NativeBuildArtifactCacheError::LockCreate {
                            path: path.clone(),
                            source,
                        })?;
                    return Ok(guard);
                }
                Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                    if started.elapsed() >= ENTRY_LOCK_WAIT {
                        return Err(NativeBuildArtifactCacheError::LockTimeout(path));
                    }
                    thread::sleep(ENTRY_LOCK_POLL);
                }
                Err(source) => {
                    return Err(NativeBuildArtifactCacheError::LockCreate { path, source });
                }
            }
        }
    }
}

impl Drop for EntryLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            if unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) } != 0 {
                eprintln!(
                    "failed to release native build artifact entry lock {}: {}",
                    self.path.display(),
                    io::Error::last_os_error()
                );
            }
        }
        #[cfg(not(unix))]
        if let Err(source) = fs::remove_file(&self.path) {
            if source.kind() != io::ErrorKind::NotFound {
                eprintln!(
                    "{}",
                    NativeBuildArtifactCacheError::LockRemove {
                        path: self.path.clone(),
                        source,
                    }
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ferrum-native-build-cache-{label}-{}-{sequence}",
                std::process::id()
            ));
            if path.exists() {
                fs::remove_dir_all(&path).unwrap();
            }
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn publish_and_restore_are_content_addressed() {
        let temp = TestDir::new("roundtrip");
        let cache = NativeBuildArtifactCache::new(temp.0.join("cache")).unwrap();
        let source = temp.0.join("libdemo.a");
        fs::write(&source, b"native-archive-v1").unwrap();
        let spec =
            NativeBuildArtifactSpec::new("static.demo", "libdemo.a", "flags=sm_89\ninput=abc")
                .unwrap();

        let published = cache.publish(&spec, &source).unwrap();
        let restored = temp.0.join("out/libdemo.a");
        let lookup = cache.restore(&spec, &restored).unwrap();

        assert_eq!(fs::read(&restored).unwrap(), b"native-archive-v1");
        assert_eq!(lookup, NativeBuildArtifactLookup::Hit(published.clone()));
        let manifest: NativeBuildArtifactCacheManifest =
            serde_json::from_str(&fs::read_to_string(published.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.input_signature, spec.input_signature());
        assert_eq!(manifest.artifact_sha256, published.artifact_sha256);
    }

    #[test]
    fn a_different_signature_is_a_cache_miss() {
        let temp = TestDir::new("signature-miss");
        let cache = NativeBuildArtifactCache::new(temp.0.join("cache")).unwrap();
        let source = temp.0.join("kernel.ptx");
        fs::write(&source, b"ptx-v1").unwrap();
        let first =
            NativeBuildArtifactSpec::new("core_ptx.kernel", "kernel.ptx", "source=one").unwrap();
        let second =
            NativeBuildArtifactSpec::new("core_ptx.kernel", "kernel.ptx", "source=two").unwrap();
        cache.publish(&first, &source).unwrap();

        assert_eq!(
            cache
                .restore(&second, temp.0.join("out/kernel.ptx"))
                .unwrap(),
            NativeBuildArtifactLookup::Miss {
                reason: "entry-absent"
            }
        );
    }

    #[test]
    fn corrupted_cache_entries_fail_closed() {
        let temp = TestDir::new("corrupt");
        let cache = NativeBuildArtifactCache::new(temp.0.join("cache")).unwrap();
        let source = temp.0.join("libdemo.a");
        fs::write(&source, b"native-archive-v1").unwrap();
        let spec = NativeBuildArtifactSpec::new("static.demo", "libdemo.a", "flags=sm_89").unwrap();
        let published = cache.publish(&spec, &source).unwrap();
        fs::write(&published.artifact_path, b"tampered").unwrap();

        let error = cache
            .restore(&spec, temp.0.join("out/libdemo.a"))
            .unwrap_err();
        assert!(matches!(
            error,
            NativeBuildArtifactCacheError::ArtifactSizeMismatch { .. }
                | NativeBuildArtifactCacheError::ArtifactSha256Mismatch { .. }
        ));
    }

    #[test]
    fn one_signature_cannot_publish_two_native_outputs() {
        let temp = TestDir::new("nondeterministic");
        let cache = NativeBuildArtifactCache::new(temp.0.join("cache")).unwrap();
        let first = temp.0.join("first.a");
        let second = temp.0.join("second.a");
        fs::write(&first, b"native-output-one").unwrap();
        fs::write(&second, b"native-output-two").unwrap();
        let spec = NativeBuildArtifactSpec::new("static.demo", "libdemo.a", "flags=sm_89").unwrap();
        cache.publish(&spec, &first).unwrap();

        assert!(matches!(
            cache.publish(&spec, &second),
            Err(NativeBuildArtifactCacheError::NondeterministicArtifact { .. })
        ));
    }

    #[test]
    fn artifact_identifiers_cannot_escape_the_cache_root() {
        assert!(matches!(
            NativeBuildArtifactSpec::new("../escape", "libdemo.a", "signature"),
            Err(NativeBuildArtifactCacheError::InvalidArtifactId(_))
        ));
        assert!(matches!(
            NativeBuildArtifactSpec::new("..", "libdemo.a", "signature"),
            Err(NativeBuildArtifactCacheError::InvalidArtifactId(_))
        ));
        assert!(matches!(
            NativeBuildArtifactSpec::new("static.demo", "../libdemo.a", "signature"),
            Err(NativeBuildArtifactCacheError::InvalidFileName(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn stale_lock_files_do_not_poison_the_cache() {
        let temp = TestDir::new("stale-lock");
        let cache = NativeBuildArtifactCache::new(temp.0.join("cache")).unwrap();
        let source = temp.0.join("libdemo.a");
        fs::write(&source, b"native-archive-v1").unwrap();
        let spec = NativeBuildArtifactSpec::new("static.demo", "libdemo.a", "flags=sm_89").unwrap();
        let entry = cache.entry_dir(&spec);
        fs::create_dir_all(&entry).unwrap();
        fs::write(entry.join("publish.lock"), b"pid=999999999\n").unwrap();

        let receipt = cache.publish(&spec, &source).unwrap();

        assert_eq!(receipt.artifact_sha256, sha256_file(&source).unwrap());
    }
}
