//! Bounded, nonwaiting production delivery on the original ordered writer.
use super::{JournalCommand, JsonlJournal, JsonlJournalConfig, JsonlJournalError};
use serde::Serialize;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::TrySendError;
use std::sync::{Arc, MutexGuard, TryLockError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum JsonlJournalErrorKind {
    Full,
    GateBusy,
    Closed,
    Worker,
    Panic,
    Encoding,
    Config,
    Limit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum JsonlJournalWorkerState {
    Running,
    Closing,
    /// A failure is latched, but the worker may still own/drain queued work.
    Failed,
    Stopped,
}

/// A fixed-size observation, not an atomic transaction across all counters.
/// Record counters and payload credit cover `try_*` only. Legacy tool delivery
/// is disclosed separately and does not inherit the production payload limit.
/// `written` means handed to BufWriter; only `flushed` passed a successful flush.
/// `failed` counts accepted jobs lost before a successful buffered write; later
/// flush failures are reported by `failure_kind`, even when `failed` is zero.
/// Neither flush nor close erases dropped records or proves artifact completeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct JsonlJournalDeliveryStats {
    pub accepted: u64,
    pub dropped: u64,
    pub failed: u64,
    pub written: u64,
    pub flushed: u64,
    pub unbounded_accepted: u64,
    pub pending_payload_bytes: usize,
    pub peak_payload_bytes: usize,
    pub worker_state: JsonlJournalWorkerState,
    pub worker_finished: bool,
    pub flush_pending: bool,
    pub failure_kind: Option<JsonlJournalErrorKind>,
}

pub(super) struct DeliveryState {
    accepted: AtomicU64,
    dropped: AtomicU64,
    failed: AtomicU64,
    written: AtomicU64,
    flushed: AtomicU64,
    unbounded_accepted: AtomicU64,
    bytes: AtomicUsize,
    peak: AtomicUsize,
    finished: AtomicBool,
}

impl DeliveryState {
    pub(super) fn new() -> Self {
        Self {
            accepted: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            written: AtomicU64::new(0),
            flushed: AtomicU64::new(0),
            unbounded_accepted: AtomicU64::new(0),
            bytes: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            finished: AtomicBool::new(false),
        }
    }

    pub(super) fn record_unbounded(&self, count: usize) {
        self.unbounded_accepted
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    fn reject(&self, count: usize) {
        self.dropped.fetch_add(count as u64, Ordering::Relaxed);
    }

    pub(super) fn record_flushed(&self) {
        self.flushed
            .store(self.written.load(Ordering::Acquire), Ordering::Release);
    }

    pub(super) fn worker_finished(&self) {
        self.finished.store(true, Ordering::Release);
    }

    pub(super) fn snapshot(
        &self,
        closing: bool,
        failure_kind: Option<JsonlJournalErrorKind>,
    ) -> JsonlJournalDeliveryStats {
        let accepted = self.accepted.load(Ordering::Acquire);
        let failed = self.failed.load(Ordering::Acquire);
        let flushed = self.flushed.load(Ordering::Acquire);
        let worker_finished = self.finished.load(Ordering::Acquire);
        JsonlJournalDeliveryStats {
            accepted,
            dropped: self.dropped.load(Ordering::Acquire),
            failed,
            written: self.written.load(Ordering::Acquire),
            flushed,
            unbounded_accepted: self.unbounded_accepted.load(Ordering::Acquire),
            pending_payload_bytes: self.bytes.load(Ordering::Acquire),
            peak_payload_bytes: self.peak.load(Ordering::Acquire),
            worker_state: if worker_finished {
                JsonlJournalWorkerState::Stopped
            } else if failure_kind.is_some() {
                JsonlJournalWorkerState::Failed
            } else if closing {
                JsonlJournalWorkerState::Closing
            } else {
                JsonlJournalWorkerState::Running
            },
            worker_finished,
            flush_pending: accepted.saturating_sub(failed) > flushed,
            failure_kind,
        }
    }

    fn reserve(
        self: &Arc<Self>,
        bytes: usize,
        limit: usize,
    ) -> Result<PayloadCredit, JsonlJournalError> {
        let mut current = self.bytes.load(Ordering::Acquire);
        loop {
            let next = current
                .checked_add(bytes)
                .filter(|next| *next <= limit)
                .ok_or_else(|| {
                    error(JsonlJournalErrorKind::Full, "JSONL payload credit is full")
                })?;
            match self.bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::Relaxed);
                    return Ok(PayloadCredit {
                        state: self.clone(),
                        bytes,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

struct PayloadCredit {
    state: Arc<DeliveryState>,
    bytes: usize,
}
impl Drop for PayloadCredit {
    fn drop(&mut self) {
        self.state.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

fn error(kind: JsonlJournalErrorKind, message: impl Into<String>) -> JsonlJournalError {
    JsonlJournalError::with_kind(kind, message)
}

/// The count pass retains no encoded bytes. The worker uses the same checked
/// writer with storage enabled, so a stateful serializer cannot exceed its grant.
struct PayloadWriter {
    encoded: Option<Vec<u8>>,
    count: usize,
    record_start: usize,
    record_limit: usize,
    batch_limit: usize,
    exceeded: bool,
}
impl PayloadWriter {
    fn new(config: JsonlJournalConfig, storage: Option<usize>) -> Self {
        Self {
            encoded: storage.map(|bytes| Vec::with_capacity(bytes.min(4096))),
            count: 0,
            record_start: 0,
            record_limit: config.max_record_bytes,
            batch_limit: storage.unwrap_or(config.max_batch_bytes),
            exceeded: false,
        }
    }
    fn record<T: Serialize>(&mut self, event: &T) -> Result<(), JsonlJournalError> {
        self.record_start = self.count;
        if let Err(cause) = serde_json::to_writer(&mut *self, event) {
            return Err(if self.exceeded {
                self.limit_error()
            } else {
                error(
                    JsonlJournalErrorKind::Encoding,
                    format!("serialize JSONL event: {cause}"),
                )
            });
        }
        self.write_all(b"\n").map_err(|_| self.limit_error())
    }
    fn limit_error(&self) -> JsonlJournalError {
        if self.encoded.is_some() {
            error(
                JsonlJournalErrorKind::Encoding,
                "JSONL worker encoding differs from its measured payload bound",
            )
        } else {
            error(
                JsonlJournalErrorKind::Limit,
                "JSONL record or batch exceeds its configured payload limit",
            )
        }
    }
}
impl Write for PayloadWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self.count.checked_add(bytes.len());
        let Some(next) = next.filter(|next| {
            *next <= self.batch_limit && next.saturating_sub(self.record_start) <= self.record_limit
        }) else {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "JSONL payload limit",
            ));
        };
        if let Some(encoded) = self.encoded.as_mut() {
            encoded.extend_from_slice(bytes);
        }
        self.count = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Borrowed synchronous tool encoding obeys the same record/batch limits.
/// It does not claim bounded production queue credit or split a batch.
pub(super) fn encode_records<T: Serialize>(
    records: &[T],
    config: JsonlJournalConfig,
) -> Result<Vec<u8>, JsonlJournalError> {
    if records.len() > config.max_batch_records {
        return Err(error(
            JsonlJournalErrorKind::Limit,
            "JSONL batch has too many records",
        ));
    }
    let mut writer = PayloadWriter::new(config, Some(config.max_batch_bytes));
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for record in records {
            if let Err(cause) = writer.record(record) {
                return Err(if writer.exceeded {
                    error(
                        JsonlJournalErrorKind::Limit,
                        "JSONL record or batch exceeds its configured payload limit",
                    )
                } else {
                    cause
                });
            }
        }
        Ok(())
    }))
    .map_err(|_| {
        error(
            JsonlJournalErrorKind::Panic,
            "JSONL tool serializer panicked",
        )
    })??;
    Ok(writer.encoded.expect("synchronous encoding owns storage"))
}

type BoundedEncoder = Box<dyn FnOnce(&mut PayloadWriter) -> Result<(), JsonlJournalError> + Send>;
pub(super) struct BoundedJob {
    encoder: Option<BoundedEncoder>,
    credit: PayloadCredit,
    config: JsonlJournalConfig,
    records: usize,
    accepted: bool,
}
impl BoundedJob {
    fn new(
        encoder: BoundedEncoder,
        bytes: usize,
        records: usize,
        config: JsonlJournalConfig,
        state: &Arc<DeliveryState>,
    ) -> Result<Self, JsonlJournalError> {
        Ok(Self {
            encoder: Some(encoder),
            credit: state.reserve(bytes, config.payload_capacity_bytes)?,
            config,
            records,
            accepted: false,
        })
    }
    pub(super) fn encode(&mut self) -> Result<Vec<u8>, JsonlJournalError> {
        let mut writer = PayloadWriter::new(self.config, Some(self.credit.bytes));
        self.encoder.take().expect("one encoding per accepted job")(&mut writer)?;
        Ok(writer.encoded.expect("worker encoding owns storage"))
    }
    fn accept(&mut self) {
        self.credit
            .state
            .accepted
            .fetch_add(self.records as u64, Ordering::Release);
        self.accepted = true;
    }
    fn reject(&mut self) {
        self.credit
            .state
            .accepted
            .fetch_sub(self.records as u64, Ordering::AcqRel);
        self.credit.state.reject(self.records);
        self.accepted = false;
    }
    pub(super) fn finish(&mut self, written: bool) {
        if written {
            self.credit
                .state
                .written
                .fetch_add(self.records as u64, Ordering::Release);
        } else {
            self.credit
                .state
                .failed
                .fetch_add(self.records as u64, Ordering::Release);
        }
        self.accepted = false;
    }
}
impl Drop for BoundedJob {
    fn drop(&mut self) {
        if self.accepted {
            self.credit
                .state
                .failed
                .fetch_add(self.records as u64, Ordering::Release);
        }
        // encoder/payload is dropped before credit, including queue/worker unwind.
    }
}

impl<T: Serialize + Send + 'static> JsonlJournal<T> {
    /// Never waits on a journal lock, queue, byte budget or disk. The bounded
    /// serde measurement runs on the caller, so custom Serialize must not block.
    pub fn try_enqueue(&self, event: T) -> Result<(), JsonlJournalError> {
        let (_guard, bytes) = self.prepare_try(std::slice::from_ref(&event))?;
        let job = BoundedJob::new(
            Box::new(move |writer| writer.record(&event)),
            bytes,
            1,
            self.inner.config,
            &self.inner.delivery,
        )
        .inspect_err(|_| self.inner.delivery.reject(1))?;
        self.send_try(job)
    }

    /// One atomic delivery command: a failed batch is never partially encoded
    /// into the journal. An empty batch remains the existing no-op.
    pub fn try_enqueue_batch(&self, events: Vec<T>) -> Result<(), JsonlJournalError> {
        if events.is_empty() {
            return Ok(());
        }
        let (_guard, bytes) = self.prepare_try(&events)?;
        let records = events.len();
        let job = BoundedJob::new(
            Box::new(move |writer| {
                for event in &events {
                    writer.record(event)?;
                }
                Ok(())
            }),
            bytes,
            records,
            self.inner.config,
            &self.inner.delivery,
        )
        .inspect_err(|_| self.inner.delivery.reject(records))?;
        self.send_try(job)
    }

    fn prepare_try<'a>(
        &'a self,
        events: &[T],
    ) -> Result<(MutexGuard<'a, ()>, usize), JsonlJournalError> {
        let result = (|| {
            if self.inner.closed.load(Ordering::Acquire) {
                return Err(error(
                    JsonlJournalErrorKind::Closed,
                    "JSONL journal is closed",
                ));
            }
            let guard = match self.inner.send_gate.try_lock() {
                Ok(guard) => guard,
                Err(TryLockError::WouldBlock) => {
                    return Err(error(
                        JsonlJournalErrorKind::GateBusy,
                        "JSONL publication gate is busy",
                    ))
                }
                Err(TryLockError::Poisoned(_)) => {
                    return Err(self.inner.record_poison("publication gate"))
                }
            };
            if self.inner.closed.load(Ordering::Acquire) {
                return Err(error(
                    JsonlJournalErrorKind::Closed,
                    "JSONL journal is closed",
                ));
            }
            if let Some(error) = self.inner.failure.current() {
                return Err(error);
            }
            if events.len() > self.inner.config.max_batch_records {
                return Err(error(
                    JsonlJournalErrorKind::Limit,
                    "JSONL batch has too many records",
                ));
            }
            let mut measure = PayloadWriter::new(self.inner.config, None);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for event in events {
                    measure.record(event)?;
                }
                Ok::<_, JsonlJournalError>(())
            }))
            .map_err(|_| {
                error(
                    JsonlJournalErrorKind::Panic,
                    "JSONL measurement serializer panicked",
                )
            })??;
            Ok((guard, measure.count))
        })();
        if result.is_err() {
            self.inner.delivery.reject(events.len());
        }
        result
    }

    fn send_try(&self, mut job: BoundedJob) -> Result<(), JsonlJournalError> {
        job.accept();
        match self.inner.sender.try_send(JournalCommand::Bounded(job)) {
            Ok(()) => Ok(()),
            Err(cause) => {
                let (command, kind) = match cause {
                    TrySendError::Full(command) => (command, JsonlJournalErrorKind::Full),
                    TrySendError::Disconnected(command) => (command, JsonlJournalErrorKind::Worker),
                };
                if let JournalCommand::Bounded(mut job) = command {
                    job.reject();
                }
                if kind == JsonlJournalErrorKind::Full {
                    Err(error(kind, "JSONL delivery queue is full"))
                } else {
                    Err(self
                        .inner
                        .failure
                        .current()
                        .unwrap_or_else(|| error(kind, "JSONL writer stopped")))
                }
            }
        }
    }
}
