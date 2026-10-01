//! Shared process/restart disk quota for optional automatic diagnostics.
//! Only worker/startup paths call this module. Unknown files fail closed.
use super::super::profile_export::StagedFile;
use super::*;
use ferrum_types::SloAutomaticCalibrationDiagnosticsV1;
use std::{
    fs::{self, File, OpenOptions, TryLockError},
    io::Write,
    path::Path,
};

const OWNED: &str = "ferrum-automatic-v1";
const MARKER: &[u8] = b"ferrum.automatic.diagnostics.v1\n";
const CONTROL_BYTES: u64 = 512;
const MAX_ENTRIES: usize = 65_536;

fn error(value: impl std::fmt::Display) -> FerrumError {
    FerrumError::config(format!("automatic diagnostic quota: {value}"))
}

#[derive(Clone)]
pub(in crate::continuous_engine::inner::cost_observation) struct Store {
    root: PathBuf,
    source_limit: u64,
    total_limit: u64,
    retained: usize,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    order: u64,
    maximum_bytes: u64,
    kind: Kind,
}
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum Kind {
    Reference,
    Service,
    DiscoveryFailure,
}

pub(in crate::continuous_engine::inner::cost_observation) struct Reservation {
    pub directory: PathBuf,
    // Releasing the lease makes this entry eligible for bounded retention.
    _lease: File,
}
struct Entry {
    path: PathBuf,
    order: u64,
    bytes: u64,
    // Hold acquired stale leases until pruning/scan completes.
    stale: Option<File>,
}

impl Store {
    pub(in crate::continuous_engine::inner::cost_observation) fn new(
        policy: &SloAutomaticCalibrationDiagnosticsV1,
    ) -> Option<Self> {
        match policy {
            SloAutomaticCalibrationDiagnosticsV1::MemoryOnly => None,
            SloAutomaticCalibrationDiagnosticsV1::Directory {
                directory,
                maximum_source_bytes,
                maximum_total_bytes,
                maximum_retained_generations,
            } => Some(Self {
                root: directory.join(OWNED),
                source_limit: maximum_source_bytes.get(),
                total_limit: maximum_total_bytes.get(),
                retained: maximum_retained_generations.get(),
            }),
        }
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn source_limit(&self) -> u64 {
        self.source_limit
    }

    fn locked(&self) -> Result<File, FerrumError> {
        fs::create_dir_all(self.root.parent().ok_or_else(|| error("missing parent"))?)
            .map_err(error)?;
        match fs::symlink_metadata(&self.root) {
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => return Err(error("owned root is not an ordinary directory")),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                match fs::create_dir(&self.root) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(e) => return Err(error(e)),
                }
            }
            Err(e) => return Err(error(e)),
        }
        let lock_path = self.root.join("lease");
        if fs::symlink_metadata(&lock_path).is_ok_and(|metadata| !metadata.file_type().is_file()) {
            return Err(error("owned quota lock is not a regular file"));
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)
            .map_err(error)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(error("quota is busy in another worker/process"))
            }
            Err(TryLockError::Error(e)) => return Err(error(e)),
        }
        let marker = self.root.join("owner");
        if !marker.exists() {
            // Claim only an empty root (apart from our lock), never an arbitrary
            // preexisting directory that happens to use the reserved name.
            if fs::read_dir(&self.root)
                .map_err(error)?
                .any(|entry| entry.map_or(true, |entry| entry.file_name() != "lease"))
            {
                return Err(error("unowned directory contains existing files"));
            }
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&marker)
                .map_err(error)?;
            file.write_all(MARKER).map_err(error)?;
            file.sync_all().map_err(error)?;
        }
        let marker_bytes = regular_bytes(&marker)?;
        if marker_bytes < MARKER.len() as u64 {
            let partial = fs::read(&marker).map_err(error)?;
            let empty_owned_root = fs::read_dir(&self.root).map_err(error)?.all(|entry| {
                entry
                    .is_ok_and(|entry| entry.file_name() == "lease" || entry.file_name() == "owner")
            });
            if MARKER.starts_with(&partial) && empty_owned_root {
                let mut file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&marker)
                    .map_err(error)?;
                file.write_all(MARKER).map_err(error)?;
                file.sync_all().map_err(error)?;
            }
        }
        if regular_bytes(&marker)? != MARKER.len() as u64
            || fs::read(&marker).map_err(error)? != MARKER
        {
            return Err(error("owned marker differs"));
        }
        Ok(lock)
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn reserve(
        &self,
        kind: Kind,
        maximum_payload: u64,
    ) -> Result<Reservation, FerrumError> {
        // Atomic no-overwrite publication can temporarily keep both names.
        // Count both, including crash leftovers; every partial remains charged.
        let promised = maximum_payload
            .checked_mul(2)
            .and_then(|n| n.checked_add(CONTROL_BYTES))
            .ok_or_else(|| error("reservation byte overflow"))?;
        if promised
            .checked_add(MARKER.len() as u64)
            .is_none_or(|n| n > self.total_limit)
        {
            return Err(error("one diagnostic reservation exceeds total byte quota"));
        }
        let _lock = self.locked()?;
        let mut entries = self.scan()?;
        let order = entries
            .iter()
            .map(|entry| entry.order)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| error("diagnostic order exhausted"))?;
        entries.sort_by(|a, b| (a.order, &a.path).cmp(&(b.order, &b.path)));
        let mut bytes = entries
            .iter()
            .try_fold(MARKER.len() as u64, |sum, entry| {
                sum.checked_add(entry.bytes)
            })
            .ok_or_else(|| error("retained byte overflow"))?;
        let mut count = entries.len();
        for entry in &entries {
            if count < self.retained
                && bytes
                    .checked_add(promised)
                    .is_some_and(|sum| sum <= self.total_limit)
            {
                break;
            }
            if entry.stale.is_some() {
                fs::remove_dir_all(&entry.path).map_err(error)?;
                bytes -= entry.bytes;
                count -= 1;
            }
        }
        if count >= self.retained
            || bytes
                .checked_add(promised)
                .is_none_or(|sum| sum > self.total_limit)
        {
            return Err(error(
                "active diagnostic leases exhaust retention/byte quota",
            ));
        }
        let directory = self.root.join(format!("entry-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).map_err(error)?;
        let created = (|| {
            let lease = OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(directory.join("lease"))
                .map_err(error)?;
            lease.lock().map_err(error)?;
            let manifest = serde_json::to_vec(&Manifest {
                version: 1,
                order,
                maximum_bytes: promised,
                kind,
            })
            .map_err(error)?;
            if manifest.len() as u64 > CONTROL_BYTES {
                return Err(error("manifest exceeds control quota"));
            }
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(directory.join("reservation.json"))
                .map_err(error)?;
            file.write_all(&manifest).map_err(error)?;
            file.sync_all().map_err(error)?;
            Ok(Reservation {
                directory: directory.clone(),
                _lease: lease,
            })
        })();
        if created.is_err() {
            let _ = fs::remove_dir_all(&directory);
        }
        created
    }

    fn scan(&self) -> Result<Vec<Entry>, FerrumError> {
        let mut entries = Vec::new();
        for item in fs::read_dir(&self.root).map_err(error)? {
            let item = item.map_err(error)?;
            let name = item.file_name();
            if name == "owner" || name == "lease" {
                let bytes = regular_bytes(&item.path())?;
                if name == "lease" && bytes != 0 {
                    return Err(error("quota lease contains unknown bytes"));
                }
                continue;
            }
            if entries.len() >= MAX_ENTRIES
                || !item.file_type().map_err(error)?.is_dir()
                || !name
                    .to_str()
                    .and_then(|name| name.strip_prefix("entry-"))
                    .is_some_and(valid_uuid)
            {
                return Err(error("unknown entry in owned directory"));
            }
            let mut actual = 0u64;
            let mut files = 0;
            for file in fs::read_dir(item.path()).map_err(error)? {
                let file = file.map_err(error)?;
                files += 1;
                if files > 12 || !file.file_name().to_str().is_some_and(known_file) {
                    return Err(error("unknown file in owned diagnostic entry"));
                }
                actual = actual
                    .checked_add(regular_bytes(&file.path())?)
                    .ok_or_else(|| error("entry byte overflow"))?;
            }
            let lease_path = item.path().join("lease");
            match fs::symlink_metadata(&lease_path) {
                Ok(_) if regular_bytes(&lease_path)? == 0 => {}
                Ok(_) => return Err(error("entry lease contains unknown bytes")),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(error(e)),
            }
            let lease = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(lease_path)
                .map_err(error)?;
            let stale = match lease.try_lock() {
                Ok(()) => Some(lease),
                Err(TryLockError::WouldBlock) => None,
                Err(TryLockError::Error(e)) => return Err(error(e)),
            };
            let manifest_path = item.path().join("reservation.json");
            let manifest = match fs::symlink_metadata(&manifest_path) {
                Ok(_) => {
                    if regular_bytes(&manifest_path)? > CONTROL_BYTES {
                        return Err(error("manifest is too large"));
                    }
                    match serde_json::from_slice::<Manifest>(
                        &fs::read(manifest_path).map_err(error)?,
                    ) {
                        Ok(manifest) => Some(manifest),
                        Err(e) if e.is_eof() => None,
                        Err(e) => return Err(error(e)),
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(error(e)),
            };
            let (order, charged) = match manifest {
                Some(manifest) => {
                    if manifest.version != 1
                        || manifest.maximum_bytes < actual
                        || manifest.maximum_bytes > 2 * 8 * 1024 * 1024 * 1024 + CONTROL_BYTES
                    {
                        return Err(error("entry exceeds original byte reservation"));
                    }
                    (
                        manifest.order,
                        if stale.is_some() {
                            actual
                        } else {
                            manifest.maximum_bytes
                        },
                    )
                }
                // A process can die between mkdir/lease and a complete bounded
                // manifest. Only known files plus a released lease qualify as
                // our incomplete reservation. Charge actual bytes and reclaim
                // it before complete historical entries when space is needed.
                None if stale.is_some() => (0, actual),
                None => return Err(error("active entry has incomplete reservation")),
            };
            entries.push(Entry {
                path: item.path(),
                order,
                bytes: charged,
                stale,
            });
        }
        Ok(entries)
    }

    pub(super) fn publish_reference(
        &self,
        source: &[u8],
        reference: &[u8],
    ) -> Result<(), FerrumError> {
        if source.len() as u64 > self.source_limit {
            return Err(error("reference source exceeds diagnostic source quota"));
        }
        let payload = (source.len() as u64)
            .checked_add(reference.len() as u64)
            .ok_or_else(|| error("reference pair overflow"))?;
        let reservation = self.reserve(Kind::Reference, payload)?;
        for (name, bytes) in [("source.jsonl", source), ("reference.json", reference)] {
            let mut file =
                StagedFile::create(&reservation.directory.join(name), bytes.len() as u64)
                    .map_err(error)?;
            file.preserve_unpublished();
            file.write_all(bytes).map_err(error)?;
            file.publish().map_err(error)?;
        }
        Ok(())
    }
}

fn regular_bytes(path: &Path) -> Result<u64, FerrumError> {
    let metadata = fs::symlink_metadata(path).map_err(error)?;
    if !metadata.file_type().is_file() {
        return Err(error("diagnostic entry contains non-regular file"));
    }
    Ok(metadata.len())
}
fn valid_uuid(value: &str) -> bool {
    uuid::Uuid::parse_str(value).is_ok_and(|id| id.to_string() == value)
}
fn known_file(value: &str) -> bool {
    matches!(
        value,
        "lease" | "reservation.json" | "source.jsonl" | "profile.json" | "reference.json"
    ) || [".ferrum-cost-", ".ferrum-structured-"]
        .iter()
        .any(|prefix| {
            value
                .strip_prefix(prefix)
                .and_then(|v| v.strip_suffix(".tmp"))
                .is_some_and(valid_uuid)
        })
}

#[cfg(test)]
mod tests;
