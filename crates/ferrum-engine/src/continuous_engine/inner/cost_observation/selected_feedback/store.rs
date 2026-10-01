//! Startup/worker-only bounded persistence. No inference lock is held here.
use super::state::{Binding, State};
use ferrum_types::{SloSelectedFeedbackSettingsV1, SloSelectedFeedbackStorageV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

pub(super) struct Store {
    files: Option<(PathBuf, PathBuf)>,
    maximum_bytes: usize,
    closed: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u32,
    state: State,
    state_sha256: [u8; 32],
}
fn state_hash(state: &State) -> io::Result<[u8; 32]> {
    serde_json::to_vec(state)
        .map(|bytes| Sha256::digest(bytes).into())
        .map_err(|_| invalid("feedback serialization"))
}
fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn sync_parent(path: &Path) -> io::Result<()> {
    File::open(
        path.parent()
            .ok_or_else(|| invalid("receipt parent absent"))?,
    )?
    .sync_all()
}

impl Store {
    /// Create a dirty receipt directly from the verified predecessor state.
    /// No empty State is ever a durable replacement for its margins/counters.
    pub(super) fn create_from_state(
        path: &Path,
        policy: &SloSelectedFeedbackSettingsV1,
        state: &State,
        capacity: usize,
    ) -> io::Result<Self> {
        if !state.validate(&state.binding, policy, capacity) {
            return Err(invalid("restart feedback state differs"));
        }
        let mut marker = path.as_os_str().to_owned();
        marker.push(".session");
        let marker = PathBuf::from(marker);
        let store = Self {
            files: Some((path.to_owned(), marker.clone())),
            maximum_bytes: policy.maximum_state_bytes.get(),
            closed: false,
        };
        // Bound the serialization before encode allocates its state clone or
        // output Vec. This path runs only at the already drained cold boundary.
        store.check_encoding_bound(state)?;
        let mut lock = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker)?;
        lock.write_all(b"selected-feedback-session-v1\n")?;
        lock.sync_all()?;
        sync_parent(&marker)?;
        let bytes = store.encode(state)?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        sync_parent(path)?;
        Ok(store)
    }

    fn check_encoding_bound(&self, state: &State) -> io::Result<()> {
        struct Count {
            bytes: usize,
            maximum: usize,
        }
        impl Write for Count {
            fn write(&mut self, value: &[u8]) -> io::Result<usize> {
                self.bytes = self
                    .bytes
                    .checked_add(value.len())
                    .filter(|n| *n <= self.maximum)
                    .ok_or_else(|| invalid("feedback receipt capacity"))?;
                Ok(value.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        #[derive(Serialize)]
        struct BorrowedReceipt<'a> {
            schema_version: u32,
            state: &'a State,
            state_sha256: [u8; 32],
        }
        // Maximum-length hash spelling is an allocation-free upper bound;
        // the real digest serialization can only be shorter.
        serde_json::to_writer(
            &mut Count {
                bytes: 0,
                maximum: self.maximum_bytes,
            },
            &BorrowedReceipt {
                schema_version: 1,
                state,
                state_sha256: [255; 32],
            },
        )
        .map_err(|_| invalid("feedback receipt capacity"))
    }

    pub(super) fn file_receipt(&self) -> io::Result<(PathBuf, u64, [u8; 32])> {
        let (path, _) = self
            .files
            .as_ref()
            .ok_or_else(|| invalid("restart receipt is not a file"))?;
        let file = File::open(path)?;
        if !file.metadata()?.is_file() || file.metadata()?.len() > self.maximum_bytes as u64 {
            return Err(invalid("restart receipt exceeds original byte bound"));
        }
        let mut hash = Sha256::new();
        let mut bytes = 0u64;
        let mut file = file.take(self.maximum_bytes as u64 + 1);
        let mut buffer = [0u8; 4096];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            bytes = bytes
                .checked_add(n as u64)
                .filter(|n| *n <= self.maximum_bytes as u64)
                .ok_or_else(|| invalid("restart receipt grew while reading"))?;
            hash.update(&buffer[..n]);
        }
        Ok((path.clone(), bytes, hash.finalize().into()))
    }

    pub fn open(
        options: &SloSelectedFeedbackStorageV1,
        policy: &SloSelectedFeedbackSettingsV1,
        binding: Binding,
        capacity: usize,
    ) -> io::Result<(Self, State)> {
        let Some(path) = options.path().map(Path::to_owned) else {
            let store = Self {
                files: None,
                maximum_bytes: policy.maximum_state_bytes.get(),
                closed: false,
            };
            let mut state = State::new(binding.clone());
            if !state.validate(&binding, policy, capacity) {
                return Err(invalid("feedback memory policy/shape binding mismatch"));
            }
            state.session = 1;
            store.encode(&state)?;
            return Ok((store, state));
        };
        let mut marker = path.as_os_str().to_owned();
        marker.push(".session");
        let marker = PathBuf::from(marker);
        // Atomic exclusivity across processes. A crash leaves this marker and
        // restart cannot mistake an incompletely persisted epoch for clean state.
        let mut lock = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker)?;
        lock.write_all(b"selected-feedback-session-v1\n")?;
        lock.sync_all()?;
        sync_parent(&marker)?;
        let mut store = Self {
            files: Some((path.clone(), marker)),
            maximum_bytes: policy.maximum_state_bytes.get(),
            closed: false,
        };
        let mut state = match options {
            SloSelectedFeedbackStorageV1::MemoryOnly => unreachable!("handled above"),
            SloSelectedFeedbackStorageV1::CreateNew { .. } => {
                let state = State::new(binding.clone());
                let bytes = store.encode(&state)?;
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)?;
                file.write_all(&bytes)?;
                file.sync_all()?;
                sync_parent(&path)?;
                state
            }
            SloSelectedFeedbackStorageV1::Resume { .. } => {
                let file = File::open(&path)?;
                if !file.metadata()?.is_file()
                    || file.metadata()?.len() > store.maximum_bytes as u64
                {
                    return Err(invalid("feedback receipt exceeds declared capacity"));
                }
                let mut bytes = Vec::new();
                file.take(store.maximum_bytes as u64 + 1)
                    .read_to_end(&mut bytes)?;
                if bytes.len() > store.maximum_bytes {
                    return Err(invalid("feedback receipt grew while loading"));
                }
                let receipt: Receipt = serde_json::from_slice(&bytes)
                    .map_err(|_| invalid("invalid feedback receipt"))?;
                if receipt.schema_version != 1
                    || receipt.state_sha256 != state_hash(&receipt.state)?
                {
                    return Err(invalid("feedback receipt checksum mismatch"));
                }
                receipt.state
            }
        };
        if !state.validate(&binding, policy, capacity) {
            return Err(invalid("feedback artifact/policy/shape binding mismatch"));
        }
        state.session = state
            .session
            .checked_add(1)
            .ok_or_else(|| invalid("feedback session exhausted"))?;
        store.persist(&state)?;
        Ok((store, state))
    }
    fn encode(&self, state: &State) -> io::Result<Vec<u8>> {
        // Corruption detection, not an authentication or rollback-proof store.
        let receipt = Receipt {
            schema_version: 1,
            state: state.clone(),
            state_sha256: state_hash(state)?,
        };
        let bytes = serde_json::to_vec(&receipt).map_err(|_| invalid("feedback serialization"))?;
        if bytes.len() > self.maximum_bytes {
            return Err(invalid("feedback receipt capacity"));
        }
        Ok(bytes)
    }
    pub fn persist(&mut self, state: &State) -> io::Result<()> {
        if self.closed {
            return Err(invalid("feedback store already closed"));
        }
        let bytes = self.encode(state)?;
        let Some((path, _)) = &self.files else {
            return Ok(());
        };
        let mut temporary = path.as_os_str().to_owned();
        temporary.push(".pending");
        let temporary = PathBuf::from(temporary);
        // Never overwrite a previous unfinished publication. That is evidence
        // of an interrupted/failed owner, not permission to reset its margin.
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)?;
        sync_parent(path)
    }
    pub fn finish(&mut self, state: &State) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.persist(state)?;
        if let Some((_, marker)) = &self.files {
            std::fs::remove_file(marker)?;
            sync_parent(marker)?;
        }
        self.closed = true;
        Ok(())
    }

    pub(super) fn retire_at(
        &mut self,
        state: &State,
        expected: &Path,
    ) -> io::Result<(PathBuf, u64, [u8; 32])> {
        if self.files.as_ref().is_none_or(|(path, _)| path != expected) {
            return Err(invalid("restart storage owner differs"));
        }
        self.finish(state)?;
        self.file_receipt()
    }
}
// Intentionally no Drop cleanup: an unacknowledged shutdown remains dirty.
