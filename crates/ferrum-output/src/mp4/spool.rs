use super::*;
use ferrum_types::{ResourceChangeNotifier, ResourceChangeSubscription};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    task::Waker,
};

#[derive(Debug, Clone, Copy)]
pub struct Mp4SpoolLimits {
    pub maximum_bytes: u64,
    pub maximum_files: usize,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mp4SpoolUsage {
    /// Includes admitted maximums until finalization, and unreclaimed cleanup
    /// failures. This is not a Host allocation or an external RSS measurement.
    pub reserved_bytes: u64,
    pub reserved_files: usize,
    pub cleanup_failures: usize,
}
struct Inner {
    root: PathBuf,
    limits: Mp4SpoolLimits,
    usage: Mutex<Mp4SpoolUsage>,
    changed: ResourceChangeNotifier,
}
impl Drop for Inner {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.root);
    }
}
#[derive(Clone)]
pub struct Mp4Spool(Arc<Inner>);
impl Mp4Spool {
    /// Requires a new directory: existing/unowned files are never counted as
    /// free capacity or silently removed. This is a process-scoped store.
    pub fn create(root: impl AsRef<Path>, limits: Mp4SpoolLimits) -> Result<Self, Mp4Error> {
        if limits.maximum_bytes == 0 || limits.maximum_files < 3 {
            return Err(Mp4Error::invalid(
                "spool needs positive byte capacity and at least three staging files",
            ));
        }
        let builder = &mut fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(root.as_ref())?;
        let root = fs::canonicalize(root.as_ref())?;
        Ok(Self(Arc::new(Inner {
            root,
            limits,
            usage: Mutex::new(Mp4SpoolUsage::default()),
            changed: ResourceChangeNotifier::default(),
        })))
    }
    pub fn usage(&self) -> Mp4SpoolUsage {
        *self.0.usage.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub fn subscribe_capacity(&self, waker: Waker) -> ResourceChangeSubscription {
        self.0.changed.subscribe(waker)
    }
    /// Metadata-only probe: does not reserve credit or emit a refund wake.
    /// A subsequent grant still checks under the original quota mutex.
    pub(super) fn capacity_available(&self, bytes: u64, files: usize) -> Result<bool, Mp4Error> {
        if bytes > self.0.limits.maximum_bytes || files > self.0.limits.maximum_files {
            return Err(Mp4Error::new(
                Mp4ErrorKind::Capacity,
                "single encode exceeds immutable spool capacity",
            ));
        }
        let usage = self.0.usage.lock().unwrap_or_else(|e| e.into_inner());
        Ok(bytes <= self.0.limits.maximum_bytes - usage.reserved_bytes
            && files <= self.0.limits.maximum_files - usage.reserved_files)
    }
    pub(super) fn try_reserve(
        &self,
        bytes: u64,
        files: usize,
    ) -> Result<Option<SpoolEntry>, Mp4Error> {
        if bytes > self.0.limits.maximum_bytes || files > self.0.limits.maximum_files {
            return Err(Mp4Error::new(
                Mp4ErrorKind::Capacity,
                "single encode exceeds immutable spool capacity",
            ));
        }
        let mut usage = self.0.usage.lock().unwrap_or_else(|e| e.into_inner());
        if bytes > self.0.limits.maximum_bytes - usage.reserved_bytes
            || files > self.0.limits.maximum_files - usage.reserved_files
        {
            return Ok(None);
        }
        let directory = self.0.root.join(uuid::Uuid::new_v4().to_string());
        fs::create_dir(&directory)?;
        usage.reserved_bytes += bytes;
        usage.reserved_files += files;
        Ok(Some(SpoolEntry {
            store: self.clone(),
            directory,
            bytes,
            files,
        }))
    }
}
pub(super) struct SpoolEntry {
    store: Mp4Spool,
    directory: PathBuf,
    bytes: u64,
    files: usize,
}
impl SpoolEntry {
    pub fn path(&self, name: &str) -> PathBuf {
        self.directory.join(name)
    }
    pub fn publish(
        mut self,
        receipt: Mp4Receipt,
        session: OutputSession,
    ) -> Result<Mp4Artifact, Mp4Error> {
        let temporary = self.path("video.part");
        if fs::metadata(&temporary)?.len() != receipt.encoded_bytes || receipt.encoded_bytes == 0 {
            return Err(Mp4Error::new(
                Mp4ErrorKind::Codec,
                "encoded file length differs from bounded writer receipt",
            ));
        }
        let mut header = [0u8; 12];
        File::open(&temporary)?.read_exact(&mut header)?;
        if &header[4..8] != b"ftyp" {
            return Err(Mp4Error::new(
                Mp4ErrorKind::Codec,
                "encoder output has no MP4 file type box",
            ));
        }
        fs::remove_file(self.path("rgb.raw"))?;
        fs::remove_file(self.path("audio.f32le"))?;
        fs::rename(temporary, self.path("video.mp4"))?;
        {
            let mut usage = self.store.0.usage.lock().unwrap_or_else(|e| e.into_inner());
            usage.reserved_bytes -= self.bytes - receipt.encoded_bytes;
            usage.reserved_files -= self.files - 1;
            self.bytes = receipt.encoded_bytes;
            self.files = 1;
        }
        self.store.0.changed.notify();
        Ok(Mp4Artifact(Arc::new(ArtifactInner {
            entry: self,
            _session: session,
            receipt,
        })))
    }
}
impl Drop for SpoolEntry {
    fn drop(&mut self) {
        let removed = match fs::remove_dir_all(&self.directory) {
            Ok(()) => true,
            Err(e) if e.kind() == io::ErrorKind::NotFound => true,
            Err(_) => false,
        };
        {
            let mut usage = self.store.0.usage.lock().unwrap_or_else(|e| e.into_inner());
            if removed {
                usage.reserved_bytes -= self.bytes;
                usage.reserved_files -= self.files;
            } else {
                usage.cleanup_failures = usage.cleanup_failures.saturating_add(1);
            }
        }
        // Failed deletion remains charged; never advertise physically live
        // files as free capacity. The caller can inspect cleanup_failures.
        self.store.0.changed.notify();
    }
}
struct ArtifactInner {
    entry: SpoolEntry,
    _session: OutputSession,
    receipt: Mp4Receipt,
}
#[derive(Clone)]
pub struct Mp4Artifact(Arc<ArtifactInner>);
impl fmt::Debug for Mp4Artifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mp4Artifact")
            .field("receipt", &self.0.receipt)
            .finish_non_exhaustive()
    }
}
impl Mp4Artifact {
    pub fn receipt(&self) -> &Mp4Receipt {
        &self.0.receipt
    }
    pub fn open(&self) -> Result<Mp4ArtifactReader, Mp4Error> {
        Ok(Mp4ArtifactReader {
            file: File::open(self.0.entry.path("video.mp4"))?,
            _artifact: self.clone(),
        })
    }
}
/// Owns the original artifact lease through the last read; no bare path/File
/// escapes that could outlive its charge and deletion authority.
pub struct Mp4ArtifactReader {
    file: File,
    _artifact: Mp4Artifact,
}
impl Read for Mp4ArtifactReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.file.read(buffer)
    }
}
impl Seek for Mp4ArtifactReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.file.seek(position)
    }
}
