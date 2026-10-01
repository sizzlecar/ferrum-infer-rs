//! Original accepted source bytes, copied once without JSON reserialization.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_profile::StructuredSourceRecordSinkV1;

#[derive(Clone)]
pub(in crate::continuous_engine::inner) struct OriginalSourceJournal(Arc<Mutex<Journal>>);
struct Journal {
    session: CacheSession,
    index: usize,
    file: Option<codec::Writer>,
    compressed: bool,
    encoding: JournalEncoding,
    bytes: u64,
    hash: Sha256,
    checkpoint: Option<(u64, [u8; 32])>,
    failed: Option<CacheMiss>,
    sealed: Option<ArtifactReceipt>,
    discarded: bool,
}
impl OriginalSourceJournal {
    /// Fixed cache-only encoder reservation, paid inside the existing source
    /// quota before its original declaration is signed.
    pub const fn writer_retained_bytes() -> usize {
        codec::ENCODER_RETAINED
    }
    pub(super) fn adopt(
        session: CacheSession,
        path: &Path,
        stored: ArtifactReceipt,
        encoding: JournalEncoding,
        transaction: &ColdTransaction,
    ) -> Result<Self> {
        // The old manifest still names this source until the next commit.
        // Pruning or dropping the new session must never move/delete that
        // committed evidence. Both physical copies share the original quota.
        let destination = session.reserve_external(stored.bytes, transaction)?;
        destination.copy_verified_from(path, &stored, transaction)?;
        let sealed = destination.finish(transaction)?;
        let (bytes, raw_sha) = encoding.original(stored);
        Ok(Self(Arc::new(Mutex::new(Journal {
            session,
            index: sealed.index,
            file: None,
            compressed: !matches!(encoding, JournalEncoding::IdentityV1),
            encoding,
            bytes,
            hash: Sha256::new(),
            checkpoint: Some((bytes, raw_sha)),
            failed: None,
            sealed: Some(sealed),
            discarded: false,
        }))))
    }
    pub(super) fn new(session: CacheSession, compressed: bool) -> Result<Self> {
        // Fixed original sink state and path storage are covered by the
        // session's preauthorized bounded source count and retained allowance.
        let (index, path) = session.0.artifact()?;
        let file = match codec::Writer::new(session.clone(), index, &path, compressed) {
            Ok(file) => file,
            Err(reason) => {
                session.0.close_writer(index);
                return Err(reason);
            }
        };
        Ok(Self(Arc::new(Mutex::new(Journal {
            session,
            index,
            file: Some(file),
            compressed,
            encoding: JournalEncoding::IdentityV1,
            bytes: 0,
            hash: Sha256::new(),
            checkpoint: None,
            failed: None,
            sealed: None,
            discarded: false,
        }))))
    }
    pub fn checkpoint(&self, receipt: (u64, [u8; 32])) -> Result<()> {
        let mut inner = self.0.lock();
        inner.verify(receipt)?;
        inner.checkpoint = Some(receipt);
        Ok(())
    }
    pub fn seal(&self, receipt: (u64, [u8; 32])) -> Result<ArtifactReceipt> {
        let mut inner = self.0.lock();
        if let Some(sealed) = inner.sealed {
            return if inner.encoding.original(sealed) == receipt {
                Ok(sealed)
            } else {
                Err(CacheMiss::Corrupt)
            };
        }
        inner.verify(receipt)?;
        if inner.checkpoint.is_none() {
            return Err(CacheMiss::Corrupt);
        }
        let file = inner.file.take().ok_or(CacheMiss::Corrupt)?;
        let result = file.finish();
        inner.session.0.close_writer(inner.index);
        let sealed = match result {
            Ok(sealed) => sealed,
            Err(reason) => {
                inner.failed.get_or_insert(reason);
                return Err(reason);
            }
        };
        inner.encoding = if inner.compressed {
            JournalEncoding::GzipV1 {
                original_bytes: receipt.0,
                original_sha256: receipt.1,
            }
        } else {
            JournalEncoding::IdentityV1
        };
        inner.sealed = Some(sealed);
        Ok(sealed)
    }
    pub(super) fn encoding(&self) -> Result<JournalEncoding> {
        let inner = self.0.lock();
        if let Some(reason) = inner.failed {
            return Err(reason);
        }
        inner.sealed.ok_or(CacheMiss::Dirty)?;
        Ok(inner.encoding)
    }
    pub fn failure(&self) -> Option<CacheMiss> {
        self.0.lock().failed
    }
    /// A collector that failed before construction has no source receipt.
    /// Release its writer once without inventing a checkpoint or clean source.
    pub fn abandon(&self) {
        let mut inner = self.0.lock();
        inner.failed.get_or_insert(CacheMiss::Dirty);
        if inner.file.take().is_some() {
            inner.session.0.close_writer(inner.index);
        }
    }
    /// A failed/unqualified original capture closes its writer too. It never
    /// becomes a reusable checkpoint merely because its descriptor is closed.
    pub fn finish(&self, receipt: (u64, [u8; 32])) {
        if self.seal(receipt).is_err() {
            let mut inner = self.0.lock();
            inner.failed.get_or_insert(CacheMiss::Corrupt);
            if inner.file.take().is_some() {
                inner.session.0.close_writer(inner.index);
            }
        }
    }
    pub(super) fn sealed(&self) -> Result<ArtifactReceipt> {
        let inner = self.0.lock();
        if let Some(reason) = inner.failed {
            return Err(reason);
        }
        inner.sealed.ok_or(CacheMiss::Dirty)
    }
    pub(super) fn discard_if_unobserved(&self) -> Result<bool> {
        if Arc::strong_count(&self.0) != 1 {
            return Ok(false);
        }
        let mut inner = self.0.lock();
        if inner.file.is_some() || inner.discarded {
            return Ok(false);
        }
        let path = inner
            .session
            .0
            .generation_path
            .join(artifact_name(inner.index));
        let bytes = regular_bytes(&path)?;
        fs::remove_file(path).map_err(io)?;
        sync_directory(&inner.session.0.generation_path)?;
        let mut ledger = inner.session.0.ledger.lock();
        ledger.used = ledger.used.checked_sub(bytes).ok_or(CacheMiss::Corrupt)?;
        ledger.sources = ledger.sources.checked_sub(1).ok_or(CacheMiss::Corrupt)?;
        drop(ledger);
        inner.discarded = true;
        Ok(true)
    }
}
impl StructuredSourceRecordSinkV1 for OriginalSourceJournal {
    fn append_original(&mut self, bytes: &[u8], receipt: (u64, [u8; 32])) {
        let mut inner = self.0.lock();
        if let Err(reason) = inner.append(bytes, receipt) {
            inner.failed.get_or_insert(reason);
        }
    }
}
impl Journal {
    fn verify(&self, receipt: (u64, [u8; 32])) -> Result<()> {
        if let Some(reason) = self.failed {
            return Err(reason);
        }
        if receipt.0 != self.bytes || receipt.1 != <[u8; 32]>::from(self.hash.clone().finalize()) {
            return Err(CacheMiss::Corrupt);
        }
        Ok(())
    }
    fn append(&mut self, bytes: &[u8], receipt: (u64, [u8; 32])) -> Result<()> {
        if let Some(reason) = self.failed {
            return Err(reason);
        }
        if self.file.is_none() || self.sealed.is_some() {
            return Err(CacheMiss::Dirty);
        }
        let next = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or(CacheMiss::Capacity)?;
        if next > self.session.0.limits.maximum_source_bytes || next != receipt.0 {
            return Err(CacheMiss::Capacity);
        }
        let mut next_hash = self.hash.clone();
        next_hash.update(bytes);
        if <[u8; 32]>::from(next_hash.clone().finalize()) != receipt.1 {
            return Err(CacheMiss::Corrupt);
        }
        // Old commit + all current sources + profiles/feedback reservations.
        // The charge precedes the write; partial/error bytes remain charged.
        self.file.as_mut().ok_or(CacheMiss::Dirty)?.append(bytes)?;
        self.bytes = next;
        self.hash = next_hash;
        Ok(())
    }
}
impl Drop for Journal {
    fn drop(&mut self) {
        if self.file.take().is_some() {
            self.session.0.close_writer(self.index);
        }
        // Never clear session dirty state on abandonment.
    }
}
