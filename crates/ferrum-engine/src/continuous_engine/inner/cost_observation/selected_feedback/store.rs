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
    path: PathBuf,
    marker: PathBuf,
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
    pub fn open(
        options: &SloSelectedFeedbackStorageV1,
        policy: &SloSelectedFeedbackSettingsV1,
        binding: Binding,
        capacity: usize,
    ) -> io::Result<(Self, State)> {
        let path = options.path().to_owned();
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
            path,
            marker,
            maximum_bytes: policy.maximum_state_bytes.get(),
            closed: false,
        };
        let mut state = match options {
            SloSelectedFeedbackStorageV1::CreateNew { .. } => {
                let state = State::new(binding.clone());
                let bytes = store.encode(&state)?;
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&store.path)?;
                file.write_all(&bytes)?;
                file.sync_all()?;
                sync_parent(&store.path)?;
                state
            }
            SloSelectedFeedbackStorageV1::Resume { .. } => {
                let file = File::open(&store.path)?;
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
        let mut temporary = self.path.as_os_str().to_owned();
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
        std::fs::rename(&temporary, &self.path)?;
        sync_parent(&self.path)
    }
    pub fn finish(&mut self, state: &State) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.persist(state)?;
        std::fs::remove_file(&self.marker)?;
        sync_parent(&self.marker)?;
        self.closed = true;
        Ok(())
    }
}
// Intentionally no Drop cleanup: an unacknowledged shutdown remains dirty.
