//! Worker-only bounded writes. Publication atomically creates a new name;
//! neither preflight checks nor a rename may overwrite an existing artifact.
use super::*;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};

const MAX_PRODUCER_BINARY_BYTES: u64 = 1 << 30;

#[cfg(test)]
mod producer_identity_tests;

/// One immutable origin per running executable. An exec starts a new process
/// and an empty store; explicit read(path) deliberately does not use this store.
static PROCESS_PRODUCER: ProducerIdentityStore = ProducerIdentityStore::new();

struct ProducerIdentityStore {
    result: std::sync::OnceLock<Result<ProducerIdentity, ProducerReadError>>,
}
impl ProducerIdentityStore {
    const fn new() -> Self {
        Self {
            result: std::sync::OnceLock::new(),
        }
    }

    fn current(
        &self,
        read: impl FnOnce() -> Result<ProducerIdentity, ProducerReadError>,
    ) -> Result<ProducerIdentity, ExportError> {
        self.result
            .get_or_init(read)
            .as_ref()
            .cloned()
            .map_err(ProducerReadError::export)
    }
}

/// Preserve the existing source/IO failure category on every read of a failed
/// immutable process origin, without trying the executable again.
enum ProducerReadError {
    Io(std::io::Error),
    Source(&'static str),
}
impl From<std::io::Error> for ProducerReadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
impl ProducerReadError {
    fn export(&self) -> ExportError {
        match self {
            Self::Io(error) => ExportError::Io(match error.raw_os_error() {
                Some(code) => std::io::Error::from_raw_os_error(code),
                None => std::io::Error::new(error.kind(), error.to_string()),
            }),
            Self::Source(reason) => ExportError::Source(reason),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct ProducerIdentity {
    pub executable_path: PathBuf,
    pub executable_sha256: String,
    pub executable_bytes: u64,
    pub package_version: &'static str,
    /// No source commit is inferred from a working directory or package version.
    pub source_revision: Option<String>,
}

impl ProducerIdentity {
    pub fn current() -> Result<Self, ExportError> {
        PROCESS_PRODUCER.current(|| Self::read_uncached(&std::env::current_exe()?))
    }

    pub(super) fn read(path: &Path) -> Result<Self, ExportError> {
        Self::read_uncached(path).map_err(|error| error.export())
    }

    fn read_uncached(path: &Path) -> Result<Self, ProducerReadError> {
        let path = path.canonicalize()?;
        let mut file = File::open(&path)?;
        let before = file.metadata()?;
        if !before.is_file() || before.len() > MAX_PRODUCER_BINARY_BYTES {
            return Err(ProducerReadError::Source(
                "producer executable exceeds the regular-file bound",
            ));
        }
        let mut hash = Sha256::new();
        let mut bytes = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            bytes = bytes
                .checked_add(n as u64)
                .filter(|bytes| *bytes <= MAX_PRODUCER_BINARY_BYTES)
                .ok_or(ProducerReadError::Source(
                    "producer executable grew beyond the byte bound",
                ))?;
            hash.update(&buffer[..n]);
        }
        let after = file.metadata()?;
        if bytes != before.len()
            || after.len() != before.len()
            || after.modified()? != before.modified()?
        {
            return Err(ProducerReadError::Source(
                "producer executable changed while hashing",
            ));
        }
        Ok(Self {
            executable_path: path,
            executable_sha256: format!("{:x}", hash.finalize()),
            executable_bytes: bytes,
            package_version: env!("CARGO_PKG_VERSION"),
            source_revision: None,
        })
    }
}

pub(super) fn destination(path: &Path) -> Result<PathBuf, ExportError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent.canonicalize()?;
    let name = path
        .file_name()
        .ok_or(ExportError::Source("export destination must name a file"))?;
    let path = parent.join(name);
    match std::fs::symlink_metadata(&path) {
        Ok(_) => Err(ExportError::Source("export destination already exists")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(error) => Err(error.into()),
    }
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct PublishedFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    #[serde(skip)]
    pub digest: [u8; 32],
}

pub(in crate::continuous_engine::inner::cost_observation) struct StagedFile {
    file: File,
    temp: PathBuf,
    destination: PathBuf,
    limit: u64,
    bytes: u64,
    hash: Sha256,
    preserve_unpublished: bool,
}

impl StagedFile {
    pub fn create(path: &Path, limit: u64) -> Result<Self, ExportError> {
        let destination = destination(path)?;
        let temp = destination
            .parent()
            .expect("resolved parent")
            .join(format!(".ferrum-cost-{}.tmp", uuid::Uuid::new_v4()));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        Ok(Self {
            file,
            temp,
            destination,
            limit,
            bytes: 0,
            hash: Sha256::new(),
            preserve_unpublished: false,
        })
    }

    /// Live service-window evidence must survive a failed writer or incomplete
    /// shutdown. Other exporters retain their ordinary temporary-file cleanup.
    pub fn preserve_unpublished(&mut self) {
        self.preserve_unpublished = true;
    }
    pub fn destination_path(&self) -> &Path {
        &self.destination
    }

    /// Inline rolling receipt for an original-record sink: no path/String clone.
    pub fn source_receipt(&self) -> (u64, [u8; 32]) {
        (self.bytes, self.hash.clone().finalize().into())
    }

    pub fn retained_path_bytes(&self) -> Option<usize> {
        self.temp
            .capacity()
            .checked_add(self.destination.capacity())
    }

    pub fn unpublished_receipt(&self) -> PublishedFile {
        let digest = self.hash.clone().finalize();
        PublishedFile {
            path: self.temp.clone(),
            bytes: self.bytes,
            sha256: format!("{digest:x}"),
            digest: digest.into(),
        }
    }

    pub fn json<T: Serialize>(&mut self, value: &T) -> Result<(), ExportError> {
        serde_json::to_writer(self, value)?;
        Ok(())
    }

    pub fn json_line<T: Serialize>(&mut self, value: &T) -> Result<(), ExportError> {
        self.json(value)?;
        self.write_all(b"\n")?;
        Ok(())
    }

    pub fn publish(mut self) -> Result<PublishedFile, ExportError> {
        self.flush()?;
        self.file.sync_all()?;
        // Same-directory hard-link creation is atomic and fails if a competing
        // writer created the destination after preflight. No rename overwrite.
        std::fs::hard_link(&self.temp, &self.destination)?;
        self.preserve_unpublished = false;
        let digest = self.hash.clone().finalize();
        Ok(PublishedFile {
            path: self.destination.clone(),
            bytes: self.bytes,
            sha256: format!("{digest:x}"),
            digest: digest.into(),
        })
    }
}

impl Write for StagedFile {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        if (buffer.len() as u64) > self.limit.saturating_sub(self.bytes) {
            return Err(std::io::Error::other(
                "cost export file byte limit exceeded",
            ));
        }
        let n = self.file.write(buffer)?;
        self.hash.update(&buffer[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}
impl Drop for StagedFile {
    fn drop(&mut self) {
        if !self.preserve_unpublished {
            let _ = std::fs::remove_file(&self.temp);
        }
    }
}
