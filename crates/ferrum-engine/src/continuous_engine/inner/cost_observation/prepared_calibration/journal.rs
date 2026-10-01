//! Optional exact source8 byte archive. Completed evidence still needs replay.
use super::super::{live_calibration::diagnostics, memory::ObservationBytePermit};
use super::*;
use crate::continuous_engine::inner::cost_observation::profile_export::{
    PublishedFile, StagedFile,
};
use ferrum_scheduler::implementations::continuous::cost_profile::StructuredSourceRecordSinkV1;
use parking_lot::Mutex;
use std::{
    io::Write,
    num::{NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy)]
pub(in crate::continuous_engine::inner) struct PreparedSourceJournalLimits {
    pub maximum_source_bytes: NonZeroU64,
    pub maximum_record_bytes: NonZeroUsize,
    pub maximum_checkpoints: NonZeroUsize,
    /// Separate cache reservation: paths, checkpoints and Arc/status storage.
    pub maximum_retained_bytes: NonZeroUsize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner) enum PreparedSourceJournalStage {
    Open,
    Header,
    Record,
    Checkpoint,
    Stop,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner) enum PreparedSourceJournalFailure {
    Open,
    EncodingOrCapacity,
    MemoryCapacity,
    Write,
    Collector,
    ReceiptMismatch,
    CheckpointCapacity,
    Publish,
    Incomplete,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner) struct PreparedSourceCheckpointReceipt {
    pub source_bytes: u64,
    pub source_sha256: [u8; 32],
    pub qualified_children: usize,
}
#[derive(Debug, PartialEq, Eq)]
pub(in crate::continuous_engine::inner) struct CompletedPreparedSourceJournal {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: [u8; 32],
    /// File completion does not imply a complete/qualified source population.
    pub population_complete: bool,
    /// Preserve the exact prefix used by each publication, never the last prefix.
    pub checkpoints: Vec<PreparedSourceCheckpointReceipt>,
}
#[derive(Debug, PartialEq, Eq)]
pub(in crate::continuous_engine::inner) enum PreparedSourceJournalStatus {
    Recording,
    Failed {
        stage: PreparedSourceJournalStage,
        reason: PreparedSourceJournalFailure,
    },
    Completed(CompletedPreparedSourceJournal),
}
#[derive(Clone)]
pub(in crate::continuous_engine::inner) struct PreparedSourceJournalObserver(Arc<Mutex<Journal>>);
impl PreparedSourceJournalObserver {
    /// Keep the original memory permit and disk lease with every borrowed
    /// status owner; paths/checkpoints are never copied into an uncharged Arc.
    pub fn status(&self) -> impl AsRef<PreparedSourceJournalStatus> + std::fmt::Debug + use<> {
        StatusRef {
            status: Arc::clone(&self.0.lock().status),
            _journal: Arc::clone(&self.0),
        }
    }
}
struct StatusRef {
    status: Arc<PreparedSourceJournalStatus>,
    _journal: Arc<Mutex<Journal>>,
}
impl AsRef<PreparedSourceJournalStatus> for StatusRef {
    fn as_ref(&self) -> &PreparedSourceJournalStatus {
        &self.status
    }
}
impl std::fmt::Debug for StatusRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.status, f)
    }
}
/// The caller reserves both complete disk and memory allowances in its cache
/// quota before construction. Numerical collection receives no extra quota.
#[derive(Clone)]
pub(in crate::continuous_engine::inner) struct PreparedSourceJournal(Arc<Mutex<Journal>>);
struct Journal {
    writer: Option<StagedFile>,
    limits: PreparedSourceJournalLimits,
    checkpoints: Vec<PreparedSourceCheckpointReceipt>,
    status: Arc<PreparedSourceJournalStatus>,
    records: u64,
    last_verified: (u64, [u8; 32]),
    _memory: Option<ObservationBytePermit>,
    _reservation: Option<diagnostics::Reservation>,
}
impl PreparedSourceJournal {
    #[cfg(test)]
    pub fn create(
        path: &Path,
        limits: PreparedSourceJournalLimits,
    ) -> std::result::Result<Self, PreparedSourceJournalFailure> {
        Self::create_inner(path, limits, None, None)
    }
    fn create_inner(
        path: &Path,
        limits: PreparedSourceJournalLimits,
        memory: Option<ObservationBytePermit>,
        reservation: Option<diagnostics::Reservation>,
    ) -> std::result::Result<Self, PreparedSourceJournalFailure> {
        // Reserve all checkpoint capacity before allocation. The retained
        // ceiling also covers a reader holding the former status during commit.
        let fixed = std::mem::size_of::<Mutex<Journal>>()
            .checked_add(3 * std::mem::size_of::<usize>())
            .and_then(|n| {
                n.checked_add(
                    3 * (std::mem::size_of::<PreparedSourceJournalStatus>()
                        + 2 * std::mem::size_of::<usize>()),
                )
            });
        let checkpoint_bytes = limits
            .maximum_checkpoints
            .get()
            .checked_mul(std::mem::size_of::<PreparedSourceCheckpointReceipt>());
        let preflight = fixed
            .zip(checkpoint_bytes)
            .and_then(|(a, b)| a.checked_add(b))
            .and_then(|n| path.as_os_str().len().checked_mul(4)?.checked_add(n));
        if preflight.is_none_or(|n| n > limits.maximum_retained_bytes.get()) {
            return Err(PreparedSourceJournalFailure::MemoryCapacity);
        }
        let mut inner = Journal {
            writer: None,
            limits,
            checkpoints: Vec::new(),
            status: Arc::new(PreparedSourceJournalStatus::Recording),
            records: 0,
            last_verified: (0, [0; 32]),
            _memory: memory,
            _reservation: reservation,
        };
        match StagedFile::create(path, limits.maximum_source_bytes.get()) {
            Ok(mut writer) => {
                // Keep the accepted prefix, including a possible partial final
                // write, under the same quota after failed/unclean collection.
                writer.preserve_unpublished();
                // The second path allowance covers publish's temporary owned
                // receipt; the file's display digest has exactly 64 ASCII bytes.
                let retained = fixed
                    .zip(checkpoint_bytes)
                    .and_then(|(a, b)| a.checked_add(b))
                    .and_then(|n| writer.retained_path_bytes()?.checked_mul(2)?.checked_add(n))
                    .and_then(|n| n.checked_add(64));
                if retained.is_none_or(|n| n > limits.maximum_retained_bytes.get())
                    || inner
                        .checkpoints
                        .try_reserve_exact(limits.maximum_checkpoints.get())
                        .is_err()
                {
                    return Err(PreparedSourceJournalFailure::MemoryCapacity);
                } else {
                    if inner.checkpoints.capacity() != limits.maximum_checkpoints.get() {
                        return Err(PreparedSourceJournalFailure::MemoryCapacity);
                    }
                    inner.writer = Some(writer);
                }
            }
            Err(_) => inner.fail(
                PreparedSourceJournalStage::Open,
                PreparedSourceJournalFailure::Open,
            ),
        }
        Ok(Self(Arc::new(Mutex::new(inner))))
    }
    pub fn observer(&self) -> PreparedSourceJournalObserver {
        PreparedSourceJournalObserver(Arc::clone(&self.0))
    }
    pub(super) fn collector_failed(&self) {
        self.0.lock().fail(
            PreparedSourceJournalStage::Record,
            PreparedSourceJournalFailure::Collector,
        );
    }
    pub(super) fn checkpoint(&self, receipt: PreparedSourceCheckpointReceipt) {
        let mut inner = self.0.lock();
        inner.verify((receipt.source_bytes, receipt.source_sha256));
        if inner.writer.is_none() {
            return;
        }
        if inner.checkpoints.len() == inner.limits.maximum_checkpoints.get() {
            inner.fail(
                PreparedSourceJournalStage::Checkpoint,
                PreparedSourceJournalFailure::CheckpointCapacity,
            );
            return;
        }
        inner.checkpoints.push(receipt);
        if let Some(writer) = &inner.writer {
            tracing::info!(target: "ferrum_engine::continuous_engine::inner::calibration::prepared_owner::startup",
                event = "automatic_source8_journal_checkpoint_v1", path = %writer.destination_path().display(),
                source_bytes = receipt.source_bytes, source_sha256 = %hex_digest(receipt.source_sha256),
                qualified_children = receipt.qualified_children,
                "Original source8 publication checkpoint prefix archived");
        }
    }
    pub(super) fn finish(&self, expected: (u64, [u8; 32]), population_complete: bool) {
        self.0.lock().finish(expected, population_complete);
    }
    pub(super) fn abandon(&self) {
        self.0.lock().fail(
            PreparedSourceJournalStage::Stop,
            PreparedSourceJournalFailure::Incomplete,
        );
    }
}
impl StructuredSourceRecordSinkV1 for PreparedSourceJournal {
    fn append_original(&mut self, bytes: &[u8], receipt: (u64, [u8; 32])) {
        self.0.lock().append(bytes, receipt);
    }
}
impl Journal {
    fn fail(&mut self, stage: PreparedSourceJournalStage, reason: PreparedSourceJournalFailure) {
        let receipt = self.writer.as_ref().map(StagedFile::unpublished_receipt);
        self.fail_with_receipt(stage, reason, receipt);
    }
    fn fail_with_receipt(
        &mut self,
        stage: PreparedSourceJournalStage,
        reason: PreparedSourceJournalFailure,
        receipt: Option<PublishedFile>,
    ) {
        if !matches!(*self.status, PreparedSourceJournalStatus::Recording) {
            return;
        }
        if let Some(receipt) = receipt {
            tracing::warn!(target: "ferrum_engine::continuous_engine::inner::calibration::prepared_owner::startup",
                event = "automatic_source8_journal_v1", state = "incomplete", ?stage, ?reason,
                path = %receipt.path.display(), file_bytes = receipt.bytes, file_sha256 = %receipt.sha256,
                verified_prefix_bytes = self.last_verified.0,
                verified_prefix_sha256 = %hex_digest(self.last_verified.1),
                checkpoints = self.checkpoints.len(), "Optional source8 archive is incomplete");
        } else {
            tracing::warn!(target: "ferrum_engine::continuous_engine::inner::calibration::prepared_owner::startup",
                event = "automatic_source8_journal_v1", state = "unavailable", ?stage, ?reason,
                "Optional source8 archive unavailable");
        }
        self.writer = None;
        self.checkpoints = Vec::new();
        self.status = Arc::new(PreparedSourceJournalStatus::Failed { stage, reason });
    }
    fn append(&mut self, bytes: &[u8], expected: (u64, [u8; 32])) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        let stage = if self.records == 0 {
            PreparedSourceJournalStage::Header
        } else {
            PreparedSourceJournalStage::Record
        };
        if bytes.is_empty()
            || bytes.last() != Some(&b'\n')
            || bytes.len() > self.limits.maximum_record_bytes.get()
            || bytes.len() > 8 * 1024 * 1024
            || writer.source_receipt().0.checked_add(bytes.len() as u64) != Some(expected.0)
            || expected.0 > self.limits.maximum_source_bytes.get()
        {
            self.fail(stage, PreparedSourceJournalFailure::EncodingOrCapacity);
            return;
        }
        if writer.write_all(bytes).is_err() {
            self.fail(stage, PreparedSourceJournalFailure::Write);
            return;
        }
        self.records += 1; // Every record consumes at least one bounded source byte.
        self.verify(expected);
    }
    fn verify(&mut self, expected: (u64, [u8; 32])) {
        let Some(writer) = &self.writer else {
            return;
        };
        if writer.source_receipt() != expected {
            self.fail(
                PreparedSourceJournalStage::Record,
                PreparedSourceJournalFailure::ReceiptMismatch,
            );
        } else {
            self.last_verified = expected;
        }
    }
    fn finish(&mut self, expected: (u64, [u8; 32]), population_complete: bool) {
        self.verify(expected);
        let Some(writer) = self.writer.take() else {
            return;
        };
        let unpublished = writer.unpublished_receipt();
        match writer.publish() {
            Ok(receipt) if (receipt.bytes, receipt.digest) == expected => {
                if receipt
                    .path
                    .parent()
                    .and_then(|p| std::fs::File::open(p).and_then(|f| f.sync_all()).ok())
                    .is_none()
                {
                    self.fail_with_receipt(
                        PreparedSourceJournalStage::Stop,
                        PreparedSourceJournalFailure::Publish,
                        Some(receipt),
                    );
                    return;
                }
                tracing::info!(target: "ferrum_engine::continuous_engine::inner::calibration::prepared_owner::startup",
                    event = "automatic_source8_journal_v1", state = "archived", population_complete,
                    path = %receipt.path.display(), source_bytes = receipt.bytes, source_sha256 = %receipt.sha256,
                    checkpoints = self.checkpoints.len(), "Original source8 archive closed");
                self.status = Arc::new(PreparedSourceJournalStatus::Completed(
                    CompletedPreparedSourceJournal {
                        path: receipt.path,
                        bytes: receipt.bytes,
                        sha256: receipt.digest,
                        population_complete,
                        checkpoints: std::mem::take(&mut self.checkpoints),
                    },
                ));
            }
            Ok(receipt) => self.fail_with_receipt(
                PreparedSourceJournalStage::Stop,
                PreparedSourceJournalFailure::ReceiptMismatch,
                Some(receipt),
            ),
            Err(_) => self.fail_with_receipt(
                PreparedSourceJournalStage::Stop,
                PreparedSourceJournalFailure::Publish,
                Some(unpublished),
            ),
        }
    }
}

fn hex_digest(bytes: [u8; 32]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(64);
    for byte in bytes {
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}

impl EngineCostRuntime {
    /// Cold source8 opening only. MemoryOnly returns before allocations or IO.
    pub(in crate::continuous_engine::inner) fn open_prepared_source_journal(
        &self,
        policy: &ferrum_types::SloAutomaticCalibrationDiagnosticsV1,
        original_source_limit: u64,
    ) -> Option<PreparedSourceJournal> {
        let ferrum_types::SloAutomaticCalibrationDiagnosticsV1::Directory { directory, .. } =
            policy
        else {
            return None;
        };
        let opened = (|| -> std::result::Result<PreparedSourceJournal, String> {
            // These are existing fixed-format names: owned root, UUID entry,
            // final filename, temporary UUID filename. Reserve path capacity
            // plus all live Journal/status/receipt/lease/control objects first.
            let cwd_bytes = if directory.is_absolute() {
                0
            } else {
                std::env::current_dir()
                    .map_err(|e| e.to_string())?
                    .as_os_str()
                    .len()
            };
            let path_bound = directory
                .as_os_str()
                .len()
                .checked_add(cwd_bytes)
                .and_then(|n| n.checked_add(256))
                .ok_or("journal path overflow")?;
            let retained = path_bound
                .checked_mul(16)
                .and_then(|n| n.checked_add(std::mem::size_of::<Mutex<Journal>>()))
                .and_then(|n| n.checked_add(3 * std::mem::size_of::<PreparedSourceJournalStatus>()))
                .and_then(|n| n.checked_add(std::mem::size_of::<diagnostics::Reservation>()))
                .and_then(|n| n.checked_add(std::mem::size_of::<PreparedSourceCheckpointReceipt>()))
                // Arc headers, permit, box, three digest displays and the
                // existing <=512-byte quota control manifest may coexist.
                .and_then(|n| n.checked_add(1024))
                .ok_or("journal memory overflow")?;
            let memory = self
                .sink
                .reserve_diagnostic_bytes(retained)
                .ok_or("shared observation byte allowance exhausted")?;
            let store = diagnostics::Store::new(policy).ok_or("diagnostic store absent")?;
            let maximum = store.source_limit().min(original_source_limit);
            let reservation = store
                .reserve(diagnostics::Kind::Service, maximum)
                .map_err(|e| e.to_string())?;
            // Use the existing Store-owned source filename. The original
            // header/protocol identifies source8; quota scanning stays strict.
            let path = reservation.directory.join("source.jsonl");
            let limits = PreparedSourceJournalLimits {
                maximum_source_bytes: NonZeroU64::new(maximum).ok_or("zero source allowance")?,
                maximum_record_bytes: NonZeroUsize::new(8 * 1024 * 1024).unwrap(),
                // The private automatic source series permits one activation
                // checkpoint per source. Explicit test journals may allow more.
                maximum_checkpoints: NonZeroUsize::MIN,
                maximum_retained_bytes: NonZeroUsize::new(retained).unwrap(),
            };
            PreparedSourceJournal::create_inner(&path, limits, Some(memory), Some(reservation))
                .map_err(|e| format!("{e:?}"))
        })();
        match opened {
            Ok(journal) => Some(journal),
            Err(reason) => {
                tracing::warn!(target: "ferrum_engine::continuous_engine::inner::calibration::prepared_owner::startup",
                    event = "automatic_source8_journal_v1", state = "unavailable", path = %directory.display(),
                    reason = %reason, "Optional source8 archive disabled; collection continues");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn limits() -> PreparedSourceJournalLimits {
        PreparedSourceJournalLimits {
            maximum_source_bytes: NonZeroU64::new(1024).unwrap(),
            maximum_record_bytes: NonZeroUsize::new(1024).unwrap(),
            maximum_checkpoints: NonZeroUsize::MIN,
            maximum_retained_bytes: NonZeroUsize::new(16 * 1024).unwrap(),
        }
    }
    #[test]
    fn source8_original_journal_status_holds_shared_memory_and_original_disk_lease() {
        let dir =
            std::env::temp_dir().join(format!("ferrum-source8-lease-{}", uuid::Uuid::new_v4()));
        let policy = ferrum_types::SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: dir.clone(),
            maximum_source_bytes: NonZeroU64::new(1024).unwrap(),
            maximum_total_bytes: NonZeroU64::new(8192).unwrap(),
            maximum_retained_generations: NonZeroUsize::MIN,
        };
        let store = diagnostics::Store::new(&policy).unwrap();
        let reservation = store.reserve(diagnostics::Kind::Service, 1024).unwrap();
        let path = reservation.directory.join("source.jsonl");
        let pool = super::super::super::memory::ObservationBytePool::new(16 * 1024);
        let permit = pool.reserve(16 * 1024).unwrap();
        let mut journal =
            PreparedSourceJournal::create_inner(&path, limits(), Some(permit), Some(reservation))
                .unwrap();
        let observer = journal.observer();
        let original = b"{}\n";
        let receipt = (original.len() as u64, Sha256::digest(original).into());
        journal.append_original(original, receipt);
        journal.finish(receipt, true);
        let status = observer.status();
        assert!(matches!(
            status.as_ref(),
            PreparedSourceJournalStatus::Completed(_)
        ));
        drop(journal);
        drop(observer);
        assert_eq!(pool.retained(), 16 * 1024);
        assert!(pool.reserve(1).is_none());
        assert!(store.reserve(diagnostics::Kind::Service, 1024).is_err());
        assert!(path.exists(), "live status retains the original disk lease");
        drop(status);
        assert_eq!(pool.retained(), 0);
        let next = store.reserve(diagnostics::Kind::Service, 1024).unwrap();
        assert!(
            !path.exists(),
            "released entry follows original Store retention"
        );
        drop(next);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn source8_original_journal_publish_collision_cannot_overwrite_or_complete() {
        let dir =
            std::env::temp_dir().join(format!("ferrum-source8-publish-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("source.jsonl");
        let mut journal = PreparedSourceJournal::create(&path, limits()).unwrap();
        let observer = journal.observer();
        let original = b"{\"kind\":\"test_original\"}\n";
        let receipt = (original.len() as u64, Sha256::digest(original).into());
        journal.append_original(original, receipt);
        std::fs::write(&path, b"another owner").unwrap();
        journal.finish(receipt, true);
        assert_eq!(
            observer.status().as_ref(),
            &PreparedSourceJournalStatus::Failed {
                stage: PreparedSourceJournalStage::Stop,
                reason: PreparedSourceJournalFailure::Publish,
            }
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"another owner");
        drop(journal);
        drop(observer);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn source8_original_journal_receipt_mismatch_and_memory_limit_never_complete() {
        let dir = std::env::temp_dir().join(format!(
            "ferrum-source8-incomplete-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir(&dir).unwrap();
        for memory_limit in [false, true] {
            let path = dir.join(format!("{memory_limit}.jsonl"));
            let mut bounded = limits();
            if memory_limit {
                bounded.maximum_retained_bytes = NonZeroUsize::MIN;
            }
            let created = PreparedSourceJournal::create(&path, bounded);
            if memory_limit {
                assert!(matches!(
                    created,
                    Err(PreparedSourceJournalFailure::MemoryCapacity)
                ));
                assert!(!path.exists());
                continue;
            }
            let mut journal = created.unwrap();
            let observer = journal.observer();
            if !memory_limit {
                journal.append_original(b"{}\n", (3, [7; 32]));
            }
            assert_eq!(
                observer.status().as_ref(),
                &PreparedSourceJournalStatus::Failed {
                    stage: if memory_limit {
                        PreparedSourceJournalStage::Open
                    } else {
                        PreparedSourceJournalStage::Record
                    },
                    reason: if memory_limit {
                        PreparedSourceJournalFailure::MemoryCapacity
                    } else {
                        PreparedSourceJournalFailure::ReceiptMismatch
                    },
                }
            );
            assert!(!path.exists());
            drop(journal);
            drop(observer);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
