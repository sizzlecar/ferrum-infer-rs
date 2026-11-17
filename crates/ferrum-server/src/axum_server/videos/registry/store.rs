//! Versioned, exclusively owned job records and independently bounded MP4s.
//! The original encoder/output session is never retained by this store.
use super::{Snapshot, Status};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};
use uuid::Uuid;

const RECORD_LIMIT: u64 = 4096;
const OWNER: &str = "ferrum-video-jobs-v1\n";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    snapshot: Snapshot,
}

#[derive(Default)]
struct Usage {
    bytes: u64,
    artifacts: usize,
    records: usize,
    cleanup_error: Option<String>,
}
struct Inner {
    root: PathBuf,
    // Artifact readers also own this lock, including after registry shutdown.
    _lock: File,
    maximum_bytes: u64,
    maximum_records: usize,
    usage: Mutex<Usage>,
    #[cfg(test)]
    fail_after_publish: AtomicBool,
    #[cfg(test)]
    fail_record_write: AtomicBool,
}
#[derive(Clone)]
pub(super) struct Store(Arc<Inner>);

impl Store {
    pub fn open(
        root: &Path,
        maximum_bytes: u64,
        maximum_records: usize,
    ) -> Result<(Self, Vec<Snapshot>)> {
        let mut directory = fs::DirBuilder::new();
        directory.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            directory.mode(0o700);
        }
        directory.create(root)?;
        ensure!(
            fs::symlink_metadata(root)?.is_dir(),
            "video job store must be a real directory"
        );
        let owner = root.join(".owner");
        if !owner.exists() {
            for entry in fs::read_dir(root)? {
                ensure!(
                    entry?.file_name() == ".lock",
                    "unowned video job directory is not empty: {}",
                    root.display()
                );
            }
        }
        let lock_path = root.join(".lock");
        match fs::symlink_metadata(&lock_path) {
            Ok(metadata) => ensure!(metadata.is_file(), "invalid video store lock file"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;
        lock.try_lock()
            .context("video job directory is already in use")?;
        let marker = if owner.exists() {
            ensure!(
                fs::symlink_metadata(&owner)?.is_file(),
                "invalid video store owner marker"
            );
            read_bounded(&owner)?
        } else {
            Vec::new()
        };
        if marker != OWNER.as_bytes() {
            // An interrupted first marker is recoverable only in an otherwise
            // empty namespace, while this process holds its exclusive lock.
            ensure!(
                OWNER.as_bytes().starts_with(&marker),
                "unsupported video job store version"
            );
            for entry in fs::read_dir(root)? {
                let name = entry?.file_name();
                ensure!(
                    name == ".owner" || name == ".lock",
                    "incomplete store marker alongside existing data; data preserved"
                );
            }
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&owner)?;
            file.write_all(OWNER.as_bytes())?;
            file.sync_all()?;
            sync_directory(root)?;
        }
        let store = Self(Arc::new(Inner {
            root: fs::canonicalize(root)?,
            _lock: lock,
            maximum_bytes,
            maximum_records,
            usage: Mutex::new(Usage::default()),
            #[cfg(test)]
            fail_after_publish: AtomicBool::new(false),
            #[cfg(test)]
            fail_record_write: AtomicBool::new(false),
        }));
        let snapshots = store.recover()?;
        Ok((store, snapshots))
    }

    fn path(&self, id: Uuid, suffix: &str) -> PathBuf {
        self.0.root.join(format!("{id}.{suffix}"))
    }
    fn recover(&self) -> Result<Vec<Snapshot>> {
        let mut files: HashMap<Uuid, HashSet<String>> = HashMap::new();
        for file in fs::read_dir(&self.0.root)? {
            let file = file?;
            let name = file.file_name();
            let name = name.to_str().context("non-UTF8 file in video job store")?;
            ensure!(
                file.file_type()?.is_file(),
                "unknown non-file entry in video job store: {name}"
            );
            if matches!(name, ".owner" | ".lock") {
                continue;
            }
            let (id, suffix) = name
                .split_once('.')
                .context("unknown file in video job store")?;
            let id = Uuid::parse_str(id).context("unknown file in video job store")?;
            ensure!(
                matches!(suffix, "json" | "json.next" | "mp4" | "mp4.part"),
                "unknown video job file: {name}"
            );
            files.entry(id).or_default().insert(suffix.to_owned());
            ensure!(files.len() <= self.0.maximum_records, "video store exceeds configured record limit; restore previous retention limits before reducing them");
        }
        // Validate every record before removing any owned crash residue.
        let mut records = Vec::new();
        let mut uncommitted = Vec::new();
        let mut encoded_bytes = 0u64;
        for (id, suffixes) in &files {
            let suffix = if suffixes.contains("json") {
                "json"
            } else {
                "json.next"
            };
            ensure!(
                suffixes.contains(suffix),
                "unowned MP4 in video job store: {id}; file preserved"
            );
            let data = read_bounded(&self.path(*id, suffix))?;
            let record: Record = match serde_json::from_slice(&data) {
                Ok(record) => record,
                Err(_) if suffixes.len() == 1 && suffix == "json.next" => {
                    // A first record is never acknowledged before its rename.
                    // This reserved temporary name without any MP4/final
                    // record is an unaccepted, interrupted creation.
                    uncommitted.push(*id);
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            ensure!(
                record.version == 1 && record.snapshot.id == *id,
                "invalid video job record identity"
            );
            let snapshot = record.snapshot;
            ensure!(
                snapshot.error.as_ref().is_none_or(|e| e.len() <= 1024),
                "video job error metadata is too large"
            );
            if snapshot.status == Status::Completed && !snapshot.delete_requested {
                let bytes = snapshot
                    .encoded_bytes
                    .context("completed video has no byte receipt")?;
                validate_mp4(&self.path(*id, "mp4"), bytes)?;
                encoded_bytes = encoded_bytes
                    .checked_add(bytes)
                    .context("video store byte count overflow")?;
                ensure!(encoded_bytes <= self.0.maximum_bytes, "video store exceeds configured byte limit; restore previous retention limits before reducing them");
            }
            records.push(snapshot);
        }
        for id in uncommitted {
            remove_if_exists(&self.path(id, "json.next"))?;
        }
        self.0.usage.lock().unwrap().records = records.len();
        let mut recovered = Vec::new();
        for mut snapshot in records {
            let id = snapshot.id;
            // A complete .next can be the initial record at an interrupted
            // creation boundary. Its validated identity grants ownership.
            if !self.path(id, "json").exists() {
                fs::rename(self.path(id, "json.next"), self.path(id, "json"))?;
            }
            remove_if_exists(&self.path(id, "json.next"))?;
            remove_if_exists(&self.path(id, "mp4.part"))?;
            if snapshot.delete_requested {
                self.remove_record(id)?;
                continue;
            }
            if snapshot.status == Status::InProgress {
                remove_if_exists(&self.path(id, "mp4"))?;
                snapshot.status = Status::Failed;
                snapshot.finished_at = Some(chrono::Utc::now().timestamp());
                snapshot.error = Some("video generation interrupted by server restart".into());
                snapshot.encoded_bytes = None;
                self.write(&snapshot)?;
            } else if snapshot.status != Status::Completed {
                remove_if_exists(&self.path(id, "mp4"))?;
            }
            recovered.push(snapshot);
        }
        sync_directory(&self.0.root)?;
        Ok(recovered)
    }

    pub fn write(&self, snapshot: &Snapshot) -> Result<()> {
        #[cfg(test)]
        if self.0.fail_record_write.swap(false, Ordering::AcqRel) {
            bail!("injected video record write failure");
        }
        let bytes = serde_json::to_vec(&Record {
            version: 1,
            snapshot: snapshot.clone(),
        })?;
        ensure!(
            bytes.len() as u64 <= RECORD_LIMIT,
            "video job metadata exceeds record bound"
        );
        let new_record = !self.path(snapshot.id, "json").exists();
        if new_record {
            let usage = self.0.usage.lock().unwrap();
            ensure!(
                usage.records < self.0.maximum_records,
                "video job record capacity exhausted"
            );
        }
        let temporary = self.path(snapshot.id, "json.next");
        // A previous failed write belongs to this same known job. No unknown
        // file is opened with truncate or treated as disposable output.
        if temporary.exists() {
            ensure!(
                self.path(snapshot.id, "json").exists(),
                "uncommitted video job metadata requires recovery"
            );
            remove_if_exists(&temporary)?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(temporary, self.path(snapshot.id, "json"))?;
        if new_record {
            self.0.usage.lock().unwrap().records += 1;
        }
        sync_directory(&self.0.root)?;
        Ok(())
    }
    pub fn available(&self, bytes: u64) -> bool {
        let usage = self.0.usage.lock().unwrap();
        usage.cleanup_error.is_none() && bytes <= self.0.maximum_bytes.saturating_sub(usage.bytes)
    }
    pub fn error(&self) -> Option<String> {
        self.0.usage.lock().unwrap().cleanup_error.clone()
    }
    pub fn bytes(&self) -> u64 {
        self.0.usage.lock().unwrap().bytes
    }

    pub fn restore_artifact(&self, snapshot: &Snapshot) -> Result<Artifact> {
        let bytes = snapshot.encoded_bytes.context("missing MP4 byte receipt")?;
        self.charge(bytes)?;
        Ok(Artifact(Arc::new(ArtifactInner {
            store: self.clone(),
            id: snapshot.id,
            bytes,
            deleted: AtomicBool::new(false),
        })))
    }
    fn charge(&self, bytes: u64) -> Result<()> {
        let mut usage = self.0.usage.lock().unwrap();
        ensure!(
            usage.cleanup_error.is_none()
                && bytes <= self.0.maximum_bytes.saturating_sub(usage.bytes),
            "retained video byte capacity exhausted (active downloads can delay reclamation)"
        );
        usage.bytes += bytes;
        usage.artifacts += 1;
        Ok(())
    }
    pub fn reserve(&self, bytes: u64) -> Result<Reservation> {
        self.charge(bytes)?;
        Ok(Reservation {
            store: self.clone(),
            bytes,
            transferred: false,
        })
    }
    #[cfg(test)]
    pub fn copy(&self, id: Uuid, bytes: u64, source: impl Read) -> Result<Artifact> {
        self.copy_reserved(id, self.reserve(bytes)?, source)
    }
    pub fn copy_reserved(
        &self,
        id: Uuid,
        mut reservation: Reservation,
        mut source: impl Read,
    ) -> Result<Artifact> {
        ensure!(
            Arc::ptr_eq(&self.0, &reservation.store.0),
            "MP4 reservation belongs to another store"
        );
        let bytes = reservation.bytes;
        ensure!(
            self.path(id, "json").is_file(),
            "MP4 publication requires an owned job record"
        );
        ensure!(
            !self.path(id, "mp4").exists(),
            "video artifact already exists"
        );
        let mut created_partial = false;
        let mut published = false;
        let result = (|| {
            let temporary = self.path(id, "mp4.part");
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            created_partial = true;
            let mut buffer = [0u8; 8192];
            let mut count = 0u64;
            loop {
                let read = source.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                count = count
                    .checked_add(read as u64)
                    .context("MP4 byte count overflow")?;
                ensure!(count <= bytes, "MP4 exceeds its encoded byte receipt");
                output.write_all(&buffer[..read])?;
            }
            ensure!(
                count == bytes,
                "MP4 is shorter than its encoded byte receipt"
            );
            output.sync_all()?;
            validate_mp4(&temporary, bytes)?;
            ensure!(
                !self.path(id, "mp4").exists(),
                "video artifact already exists"
            );
            fs::rename(temporary, self.path(id, "mp4"))?;
            published = true;
            #[cfg(test)]
            if self.0.fail_after_publish.swap(false, Ordering::AcqRel) {
                bail!("injected directory sync failure after MP4 publication");
            }
            sync_directory(&self.0.root)?;
            reservation.transferred = true;
            Ok(Artifact(Arc::new(ArtifactInner {
                store: self.clone(),
                id,
                bytes,
                deleted: AtomicBool::new(false),
            })))
        })();
        if result.is_err() {
            let cleanup = (|| -> Result<()> {
                if created_partial {
                    remove_if_exists(&self.path(id, "mp4.part"))?;
                }
                if published {
                    remove_if_exists(&self.path(id, "mp4"))?;
                }
                Ok(())
            })();
            if cleanup.is_err() {
                reservation.transferred = true;
                self.0.usage.lock().unwrap().cleanup_error =
                    Some("video store could not remove a partial MP4".into());
            }
        }
        result
    }
    #[cfg(test)]
    pub fn fail_next_record_write(&self) {
        self.0.fail_record_write.store(true, Ordering::Release);
    }
    pub fn remove_record(&self, id: Uuid) -> Result<()> {
        remove_if_exists(&self.path(id, "mp4.part"))?;
        remove_if_exists(&self.path(id, "mp4"))?;
        remove_if_exists(&self.path(id, "json.next"))?;
        // Retain ownership metadata until media removal is itself durable.
        sync_directory(&self.0.root)?;
        remove_if_exists(&self.path(id, "json"))?;
        sync_directory(&self.0.root)?;
        self.0.usage.lock().unwrap().records -= 1;
        Ok(())
    }
}

pub(super) struct Reservation {
    store: Store,
    bytes: u64,
    transferred: bool,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.transferred {
            let mut usage = self.store.0.usage.lock().unwrap();
            usage.bytes -= self.bytes;
            usage.artifacts -= 1;
        }
    }
}

struct ArtifactInner {
    store: Store,
    id: Uuid,
    bytes: u64,
    deleted: AtomicBool,
}
impl Drop for ArtifactInner {
    fn drop(&mut self) {
        if !self.deleted.load(Ordering::Acquire) {
            return;
        }
        let result = self.store.remove_record(self.id);
        let mut usage = self.store.0.usage.lock().unwrap();
        if result.is_ok() {
            usage.bytes -= self.bytes;
            usage.artifacts -= 1;
        } else {
            usage.cleanup_error = Some(format!(
                "video artifact cleanup failed: {}",
                result.unwrap_err()
            ));
        }
    }
}
#[derive(Clone)]
pub(in crate::axum_server::videos) struct Artifact(Arc<ArtifactInner>);
impl Artifact {
    pub fn encoded_bytes(&self) -> u64 {
        self.0.bytes
    }
    pub fn delete(&self) {
        self.0.deleted.store(true, Ordering::Release);
    }
    pub fn open(&self) -> Result<Reader> {
        Ok(Reader {
            file: File::open(self.0.store.path(self.0.id, "mp4"))?,
            _artifact: self.clone(),
        })
    }
}
pub(in crate::axum_server::videos) struct Reader {
    file: File,
    _artifact: Artifact,
}
impl Read for Reader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buffer)
    }
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(RECORD_LIMIT + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= RECORD_LIMIT,
        "video store metadata exceeds bound"
    );
    Ok(bytes)
}
fn validate_mp4(path: &Path, bytes: u64) -> Result<()> {
    ensure!(
        bytes >= 12 && fs::metadata(path)?.len() == bytes,
        "stored MP4 length differs from receipt"
    );
    let mut header = [0u8; 12];
    File::open(path)?.read_exact(&mut header)?;
    ensure!(
        &header[4..8] == b"ftyp",
        "stored video has no MP4 file type box"
    );
    Ok(())
}
fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() => fs::remove_file(path)?,
        Ok(_) => bail!("refusing to remove unknown non-file {}", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}
fn sync_directory(_path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(_path)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests;
