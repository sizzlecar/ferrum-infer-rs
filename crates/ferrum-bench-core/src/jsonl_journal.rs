use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::marker::PhantomData;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static JSONL_JOURNAL_REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<JsonlJournalInner>>>> =
    OnceLock::new();

/// Return one stable physical identity for a JSONL output path, including
/// paths whose final component does not exist yet.
pub fn normalize_jsonl_path(path: &Path) -> io::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut lexical = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => lexical.push(prefix.as_os_str()),
            Component::RootDir => lexical.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = lexical.pop();
            }
            Component::Normal(part) => lexical.push(part),
        }
    }
    if let Ok(canonical) = std::fs::canonicalize(&lexical) {
        return Ok(canonical);
    }

    let mut existing = lexical.as_path();
    let mut missing = Vec::new();
    while !existing.exists() {
        if let Some(name) = existing.file_name() {
            missing.push(name.to_os_string());
        }
        existing = existing.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("JSONL path {} has no existing ancestor", path.display()),
            )
        })?;
    }
    let mut canonical = std::fs::canonicalize(existing)?;
    for component in missing.into_iter().rev() {
        canonical.push(component);
    }
    Ok(canonical)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonlJournalOpenMode {
    Truncate,
    Append,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonlJournalConfig {
    pub queue_capacity: usize,
    pub buffer_capacity_bytes: usize,
    pub flush_interval: Duration,
}

impl Default for JsonlJournalConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 256,
            buffer_capacity_bytes: 1024 * 1024,
            flush_interval: Duration::from_millis(50),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonlJournalError {
    message: String,
}

impl JsonlJournalError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for JsonlJournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for JsonlJournalError {}

struct JournalFailure {
    failed: AtomicBool,
    error: Mutex<Option<JsonlJournalError>>,
}

impl JournalFailure {
    fn new() -> Self {
        Self {
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    fn record(&self, error: JsonlJournalError) {
        if self
            .failed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            *self.error.lock().unwrap() = Some(error);
        }
    }

    fn current(&self) -> Option<JsonlJournalError> {
        if !self.failed.load(Ordering::Acquire) {
            return None;
        }
        self.error.lock().unwrap().clone().or_else(|| {
            Some(JsonlJournalError::new(
                "JSONL journal failed without a recorded cause",
            ))
        })
    }
}

type EncodeJob = Box<dyn FnOnce(&mut Vec<u8>) -> Result<(), serde_json::Error> + Send + 'static>;

enum JournalCommand {
    Encode(EncodeJob),
    Encoded(Vec<u8>),
    Flush(SyncSender<Result<(), JsonlJournalError>>),
    Close(SyncSender<Result<(), JsonlJournalError>>),
}

struct JsonlJournalInner {
    path: PathBuf,
    sender: SyncSender<JournalCommand>,
    send_gate: Mutex<()>,
    worker: Mutex<Option<JoinHandle<()>>>,
    failure: Arc<JournalFailure>,
    closed: AtomicBool,
}

impl JsonlJournalInner {
    fn send(&self, command: JournalCommand) -> Result<(), JsonlJournalError> {
        let _send_guard = self.send_gate.lock().unwrap();
        if self.closed.load(Ordering::Acquire) {
            return Err(JsonlJournalError::new(format!(
                "JSONL journal {} is closed",
                self.path.display()
            )));
        }
        if let Some(error) = self.failure.current() {
            return Err(error);
        }
        self.sender.send(command).map_err(|_| {
            self.failure.current().unwrap_or_else(|| {
                JsonlJournalError::new(format!(
                    "JSONL journal worker for {} stopped",
                    self.path.display()
                ))
            })
        })
    }

    fn close_and_join(&self) -> Result<(), JsonlJournalError> {
        // Serialize close with all producers so every accepted command is
        // ordered before Close and every later producer is rejected.
        let _send_guard = self.send_gate.lock().unwrap();
        if !self.closed.swap(true, Ordering::AcqRel) {
            let (reply_tx, reply_rx) = sync_channel(1);
            if self.sender.send(JournalCommand::Close(reply_tx)).is_err() {
                self.failure.record(JsonlJournalError::new(format!(
                    "JSONL journal worker for {} stopped before close",
                    self.path.display()
                )));
            } else if reply_rx.recv().is_err() {
                self.failure.record(JsonlJournalError::new(format!(
                    "JSONL journal close barrier for {} was lost",
                    self.path.display()
                )));
            }
        }

        let worker = self.worker.lock().unwrap().take();
        if let Some(worker) = worker {
            if worker.join().is_err() {
                self.failure.record(JsonlJournalError::new(format!(
                    "JSONL journal worker for {} panicked",
                    self.path.display()
                )));
            }
        }
        self.failure.current().map_or(Ok(()), Err)
    }
}

impl Drop for JsonlJournalInner {
    fn drop(&mut self) {
        let _ = self.close_and_join();
    }
}

pub struct JsonlJournal<T>
where
    T: Serialize + Send + 'static,
{
    inner: Arc<JsonlJournalInner>,
    marker: PhantomData<fn(T)>,
}

impl<T> Clone for JsonlJournal<T>
where
    T: Serialize + Send + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            marker: PhantomData,
        }
    }
}

fn validate_config(config: JsonlJournalConfig) -> io::Result<()> {
    if config.queue_capacity == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "JSONL journal queue capacity must be greater than zero",
        ));
    }
    if config.buffer_capacity_bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "JSONL journal buffer capacity must be greater than zero",
        ));
    }
    if config.flush_interval.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "JSONL journal flush interval must be greater than zero",
        ));
    }
    Ok(())
}

fn open_shared_inner(
    path: impl Into<PathBuf>,
    mode: JsonlJournalOpenMode,
    config: JsonlJournalConfig,
) -> io::Result<Arc<JsonlJournalInner>> {
    validate_config(config)?;
    let path = normalize_jsonl_path(&path.into())?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut registry = JSONL_JOURNAL_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|error| {
            io::Error::other(format!("JSONL journal registry is poisoned: {error}"))
        })?;
    if let Some(existing) = registry.get(&path).and_then(Weak::upgrade) {
        if !existing.closed.load(Ordering::Acquire) {
            if let Some(error) = existing.failure.current() {
                return Err(io::Error::other(error));
            }
            // The first live owner controls the initial truncate/append state.
            // Every later role attaches to the same ordered writer.
            return Ok(existing);
        }
    }

    let mut options = OpenOptions::new();
    options.create(true).append(true);
    let file = options.open(&path)?;
    if mode == JsonlJournalOpenMode::Truncate {
        file.set_len(0)?;
    }
    let (sender, receiver) = sync_channel(config.queue_capacity);
    let failure = Arc::new(JournalFailure::new());
    let worker_failure = Arc::clone(&failure);
    let worker_path = path.clone();
    let worker = thread::Builder::new()
        .name("ferrum-jsonl".to_string())
        .spawn(move || {
            run_writer(
                file,
                receiver,
                worker_failure,
                &worker_path,
                config.buffer_capacity_bytes,
                config.flush_interval,
            )
        })?;
    let inner = Arc::new(JsonlJournalInner {
        path: path.clone(),
        sender,
        send_gate: Mutex::new(()),
        worker: Mutex::new(Some(worker)),
        failure,
        closed: AtomicBool::new(false),
    });
    registry.insert(path, Arc::downgrade(&inner));
    Ok(inner)
}

fn flush_inner(inner: &JsonlJournalInner) -> Result<(), JsonlJournalError> {
    let (reply_tx, reply_rx) = sync_channel(1);
    inner.send(JournalCommand::Flush(reply_tx))?;
    reply_rx.recv().map_err(|_| {
        inner.failure.current().unwrap_or_else(|| {
            JsonlJournalError::new(format!(
                "JSONL journal flush barrier for {} was lost",
                inner.path.display()
            ))
        })
    })?
}

impl<T> JsonlJournal<T>
where
    T: Serialize + Send + 'static,
{
    pub fn open(
        path: impl Into<PathBuf>,
        mode: JsonlJournalOpenMode,
        config: JsonlJournalConfig,
    ) -> io::Result<Self> {
        Ok(Self {
            inner: open_shared_inner(path, mode, config)?,
            marker: PhantomData,
        })
    }

    pub fn create(path: impl Into<PathBuf>) -> io::Result<Self> {
        Self::open(
            path,
            JsonlJournalOpenMode::Truncate,
            JsonlJournalConfig::default(),
        )
    }

    pub fn append(path: impl Into<PathBuf>) -> io::Result<Self> {
        Self::open(
            path,
            JsonlJournalOpenMode::Append,
            JsonlJournalConfig::default(),
        )
    }

    pub fn enqueue(&self, event: T) -> Result<(), JsonlJournalError> {
        self.inner
            .send(JournalCommand::Encode(Box::new(move |encoded| {
                serde_json::to_writer(&mut *encoded, &event)?;
                encoded.push(b'\n');
                Ok(())
            })))
    }

    pub fn enqueue_batch(&self, events: Vec<T>) -> Result<(), JsonlJournalError> {
        if events.is_empty() {
            return Ok(());
        }
        self.inner
            .send(JournalCommand::Encode(Box::new(move |encoded| {
                for event in &events {
                    serde_json::to_writer(&mut *encoded, event)?;
                    encoded.push(b'\n');
                }
                Ok(())
            })))
    }

    pub fn flush(&self) -> Result<(), JsonlJournalError> {
        flush_inner(&self.inner)
    }

    pub fn close(&self) -> Result<(), JsonlJournalError> {
        self.inner.close_and_join()
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }
}

pub fn write_jsonl_records<T: Serialize>(
    path: &Path,
    mode: JsonlJournalOpenMode,
    records: &[T],
) -> Result<(), JsonlJournalError> {
    if records.is_empty() {
        return Err(JsonlJournalError::new(format!(
            "JSONL record set for {} must not be empty",
            path.display()
        )));
    }
    let mut encoded = Vec::with_capacity(4096);
    for record in records {
        serde_json::to_writer(&mut encoded, record).map_err(|error| {
            JsonlJournalError::new(format!(
                "serialize JSONL record for {}: {error}",
                path.display()
            ))
        })?;
        encoded.push(b'\n');
    }

    let inner = open_shared_inner(path.to_path_buf(), mode, JsonlJournalConfig::default())
        .map_err(|error| {
            JsonlJournalError::new(format!("open JSONL journal {}: {error}", path.display()))
        })?;
    inner.send(JournalCommand::Encoded(encoded))?;
    flush_inner(&inner)
}

fn run_writer(
    file: File,
    receiver: Receiver<JournalCommand>,
    failure: Arc<JournalFailure>,
    path: &Path,
    buffer_capacity_bytes: usize,
    flush_interval: Duration,
) {
    let mut writer = BufWriter::with_capacity(buffer_capacity_bytes, file);
    let mut encoded = Vec::with_capacity(4096);
    let mut writable = true;
    let mut flush_deadline: Option<Instant> = None;

    loop {
        let command = match flush_deadline {
            Some(deadline) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    flush_if_healthy(&mut writer, &failure, path, &mut writable);
                    flush_deadline = None;
                    continue;
                }
                receiver.recv_timeout(remaining)
            }
            None => receiver.recv().map_err(|_| RecvTimeoutError::Disconnected),
        };
        match command {
            Ok(JournalCommand::Encode(job)) => {
                flush_deadline.get_or_insert_with(|| Instant::now() + flush_interval);
                encoded.clear();
                if let Err(error) = job(&mut encoded) {
                    writable = false;
                    failure.record(JsonlJournalError::new(format!(
                        "serialize JSONL event for {}: {error}",
                        path.display()
                    )));
                } else {
                    write_encoded_if_healthy(&mut writer, &encoded, &failure, path, &mut writable);
                }
            }
            Ok(JournalCommand::Encoded(encoded)) => {
                flush_deadline.get_or_insert_with(|| Instant::now() + flush_interval);
                write_encoded_if_healthy(&mut writer, &encoded, &failure, path, &mut writable);
            }
            Ok(JournalCommand::Flush(reply)) => {
                flush_if_healthy(&mut writer, &failure, path, &mut writable);
                flush_deadline = None;
                let _ = reply.send(failure.current().map_or(Ok(()), Err));
            }
            Ok(JournalCommand::Close(reply)) => {
                flush_if_healthy(&mut writer, &failure, path, &mut writable);
                let _ = reply.send(failure.current().map_or(Ok(()), Err));
                return;
            }
            Err(RecvTimeoutError::Timeout) => {
                flush_if_healthy(&mut writer, &failure, path, &mut writable);
                flush_deadline = None;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    flush_if_healthy(&mut writer, &failure, path, &mut writable);
}

fn write_encoded_if_healthy(
    writer: &mut BufWriter<File>,
    encoded: &[u8],
    failure: &JournalFailure,
    path: &Path,
    writable: &mut bool,
) {
    if !*writable {
        return;
    }
    if let Err(error) = writer.write_all(encoded) {
        *writable = false;
        failure.record(JsonlJournalError::new(format!(
            "write JSONL event to {}: {error}",
            path.display()
        )));
    }
}

fn flush_if_healthy(
    writer: &mut BufWriter<File>,
    failure: &JournalFailure,
    path: &Path,
    writable: &mut bool,
) {
    if !*writable {
        return;
    }
    if let Err(error) = writer.flush() {
        *writable = false;
        failure.record(JsonlJournalError::new(format!(
            "flush JSONL journal {}: {error}",
            path.display()
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::ser::SerializeStruct;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn temp_path(label: &str) -> PathBuf {
        let ordinal = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ferrum-jsonl-journal-{label}-{}-{nanos}-{ordinal}.jsonl",
            std::process::id()
        ))
    }

    fn read_values(path: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn flush_makes_ordered_events_and_batches_visible() {
        let path = temp_path("ordered");
        let journal = JsonlJournal::create(path.clone()).unwrap();
        journal.enqueue(serde_json::json!({"ordinal": 1})).unwrap();
        journal
            .enqueue_batch(vec![
                serde_json::json!({"ordinal": 2}),
                serde_json::json!({"ordinal": 3}),
            ])
            .unwrap();
        journal.flush().unwrap();

        let ordinals: Vec<_> = read_values(&path)
            .into_iter()
            .map(|value| value["ordinal"].as_u64().unwrap())
            .collect();
        assert_eq!(ordinals, vec![1, 2, 3]);
        drop(journal);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn truncate_journal_appends_after_an_external_startup_record() {
        let path = temp_path("external-startup-record");
        let journal = JsonlJournal::create(path.clone()).unwrap();
        let mut startup_writer = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(startup_writer, r#"{{"source":"startup"}}"#).unwrap();
        startup_writer.flush().unwrap();

        journal
            .enqueue(serde_json::json!({"source": "engine"}))
            .unwrap();
        journal.flush().unwrap();

        assert_eq!(
            read_values(&path),
            vec![
                serde_json::json!({"source": "startup"}),
                serde_json::json!({"source": "engine"}),
            ]
        );
        drop(journal);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn journal_and_direct_records_share_an_atomic_jsonl_boundary() {
        const RECORDS_PER_WRITER: u64 = 32;
        let path = temp_path("mixed-writers");
        let journal = JsonlJournal::create(path.clone()).unwrap();
        let producer = {
            let journal = journal.clone();
            std::thread::spawn(move || {
                for sequence in 0..RECORDS_PER_WRITER {
                    journal
                        .enqueue(serde_json::json!({
                            "source": "journal",
                            "sequence": sequence,
                        }))
                        .unwrap();
                }
            })
        };
        for sequence in 0..RECORDS_PER_WRITER {
            write_jsonl_records(
                &path,
                JsonlJournalOpenMode::Append,
                &[serde_json::json!({
                    "source": "direct",
                    "sequence": sequence,
                })],
            )
            .unwrap();
        }
        producer.join().unwrap();
        journal.close().unwrap();

        let rows = read_values(&path);
        assert_eq!(rows.len(), 2 * RECORDS_PER_WRITER as usize);
        let identities = rows
            .iter()
            .map(|row| {
                (
                    row["source"].as_str().unwrap().to_string(),
                    row["sequence"].as_u64().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(identities.len(), 2 * RECORDS_PER_WRITER as usize);
        let _ = std::fs::remove_file(path);
    }

    #[derive(Serialize)]
    struct TypedRecord {
        source: &'static str,
        sequence: u64,
    }

    #[test]
    fn path_aliases_and_distinct_record_types_share_one_ordered_writer() {
        let path = temp_path("typed-alias");
        let alias = path
            .parent()
            .unwrap()
            .join(".")
            .join(path.file_name().unwrap());
        let json_journal = JsonlJournal::<serde_json::Value>::create(path.clone()).unwrap();
        let typed_journal = JsonlJournal::<TypedRecord>::create(alias).unwrap();
        assert_eq!(json_journal.path(), typed_journal.path());

        json_journal
            .enqueue(serde_json::json!({"source": "json", "sequence": 1}))
            .unwrap();
        typed_journal
            .enqueue(TypedRecord {
                source: "typed",
                sequence: 2,
            })
            .unwrap();
        write_jsonl_records(
            &path,
            JsonlJournalOpenMode::Append,
            &[serde_json::json!({"source": "direct", "sequence": 3})],
        )
        .unwrap();
        typed_journal.flush().unwrap();

        assert_eq!(
            read_values(&path),
            vec![
                serde_json::json!({"source": "json", "sequence": 1}),
                serde_json::json!({"source": "typed", "sequence": 2}),
                serde_json::json!({"source": "direct", "sequence": 3}),
            ]
        );
        drop(typed_journal);
        drop(json_journal);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn last_handle_drop_closes_and_flushes() {
        let path = temp_path("drop-flush");
        {
            let journal = JsonlJournal::create(path.clone()).unwrap();
            journal
                .enqueue(serde_json::json!({"closed": true}))
                .unwrap();
        }
        assert_eq!(
            read_values(&path),
            vec![serde_json::json!({"closed": true})]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn explicit_close_drains_joins_and_rejects_new_events() {
        let path = temp_path("explicit-close");
        let journal = JsonlJournal::create(path.clone()).unwrap();
        journal
            .enqueue(serde_json::json!({"closed": true}))
            .unwrap();
        journal.close().unwrap();

        assert_eq!(
            read_values(&path),
            vec![serde_json::json!({"closed": true})]
        );
        assert!(journal
            .enqueue(serde_json::json!({"after_close": true}))
            .is_err());
        journal.close().unwrap();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn concurrent_producers_preserve_all_events_and_per_producer_order() {
        let path = temp_path("concurrent");
        let journal = JsonlJournal::create(path.clone()).unwrap();
        let mut workers = Vec::new();
        for producer in 0..4_u64 {
            let journal = journal.clone();
            workers.push(std::thread::spawn(move || {
                for sequence in 0..32_u64 {
                    journal
                        .enqueue(serde_json::json!({
                            "producer": producer,
                            "sequence": sequence,
                        }))
                        .unwrap();
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        journal.flush().unwrap();

        let rows = read_values(&path);
        assert_eq!(rows.len(), 128);
        let identities: BTreeSet<_> = rows
            .iter()
            .map(|row| {
                (
                    row["producer"].as_u64().unwrap(),
                    row["sequence"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(identities.len(), 128);
        let mut by_producer = BTreeMap::<u64, Vec<u64>>::new();
        for row in rows {
            by_producer
                .entry(row["producer"].as_u64().unwrap())
                .or_default()
                .push(row["sequence"].as_u64().unwrap());
        }
        for sequences in by_producer.values() {
            assert_eq!(sequences, &(0..32_u64).collect::<Vec<_>>());
        }
        drop(journal);
        let _ = std::fs::remove_file(path);
    }

    struct FailingEvent;

    impl Serialize for FailingEvent {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(serde::ser::Error::custom(
                "intentional serialization failure",
            ))
        }
    }

    #[test]
    fn serialization_failure_is_latched_and_returned_by_flush() {
        let path = temp_path("serialization-failure");
        let journal = JsonlJournal::create(path.clone()).unwrap();
        journal.enqueue(FailingEvent).unwrap();
        let error = journal.flush().unwrap_err();
        assert!(error
            .message()
            .contains("intentional serialization failure"));
        assert!(journal.enqueue(FailingEvent).is_err());
        drop(journal);
        assert!(std::fs::read(&path).unwrap().is_empty());
        let _ = std::fs::remove_file(path);
    }

    enum BatchEvent {
        Valid(u64),
        Failing,
    }

    impl Serialize for BatchEvent {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            match self {
                Self::Valid(ordinal) => ordinal.serialize(serializer),
                Self::Failing => Err(serde::ser::Error::custom(
                    "intentional batch serialization failure",
                )),
            }
        }
    }

    #[test]
    fn serialization_failure_never_writes_a_partial_batch() {
        let path = temp_path("atomic-batch");
        let journal = JsonlJournal::create(path.clone()).unwrap();
        journal
            .enqueue_batch(vec![
                BatchEvent::Valid(1),
                BatchEvent::Failing,
                BatchEvent::Valid(3),
            ])
            .unwrap();
        assert!(journal.flush().is_err());
        drop(journal);
        assert!(std::fs::read(&path).unwrap().is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[derive(Default)]
    struct BlockingGate {
        state: Mutex<(bool, bool)>,
        condition: Condvar,
    }

    impl BlockingGate {
        fn block_writer(&self) {
            let mut state = self.state.lock().unwrap();
            state.0 = true;
            self.condition.notify_all();
            while !state.1 {
                state = self.condition.wait(state).unwrap();
            }
        }

        fn wait_until_blocked(&self) {
            let mut state = self.state.lock().unwrap();
            while !state.0 {
                state = self.condition.wait(state).unwrap();
            }
        }

        fn release(&self) {
            let mut state = self.state.lock().unwrap();
            state.1 = true;
            self.condition.notify_all();
        }
    }

    struct BlockingEvent {
        ordinal: u64,
        gate: Option<Arc<BlockingGate>>,
    }

    impl Serialize for BlockingEvent {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            if let Some(gate) = &self.gate {
                gate.block_writer();
            }
            let mut state = serializer.serialize_struct("BlockingEvent", 1)?;
            state.serialize_field("ordinal", &self.ordinal)?;
            state.end()
        }
    }

    #[test]
    fn full_bounded_queue_applies_backpressure_without_dropping() {
        let path = temp_path("backpressure");
        let journal = JsonlJournal::open(
            path.clone(),
            JsonlJournalOpenMode::Truncate,
            JsonlJournalConfig {
                queue_capacity: 1,
                buffer_capacity_bytes: 4096,
                flush_interval: Duration::from_millis(10),
            },
        )
        .unwrap();
        let gate = Arc::new(BlockingGate::default());
        journal
            .enqueue(BlockingEvent {
                ordinal: 1,
                gate: Some(Arc::clone(&gate)),
            })
            .unwrap();
        gate.wait_until_blocked();
        journal
            .enqueue(BlockingEvent {
                ordinal: 2,
                gate: None,
            })
            .unwrap();

        let third = journal.clone();
        let (done_tx, done_rx) = sync_channel(1);
        let sender = std::thread::spawn(move || {
            let result = third.enqueue(BlockingEvent {
                ordinal: 3,
                gate: None,
            });
            done_tx.send(result).unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(50)).is_err());
        gate.release();
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        sender.join().unwrap();
        journal.flush().unwrap();

        let ordinals: Vec<_> = read_values(&path)
            .into_iter()
            .map(|value| value["ordinal"].as_u64().unwrap())
            .collect();
        assert_eq!(ordinals, vec![1, 2, 3]);
        drop(journal);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flush_interval_makes_events_visible_without_explicit_flush() {
        let path = temp_path("periodic-flush");
        let journal = JsonlJournal::open(
            path.clone(),
            JsonlJournalOpenMode::Truncate,
            JsonlJournalConfig {
                flush_interval: Duration::from_millis(10),
                ..JsonlJournalConfig::default()
            },
        )
        .unwrap();
        journal
            .enqueue(serde_json::json!({"visible": true}))
            .unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while std::fs::read_to_string(&path).unwrap().is_empty() {
            assert!(
                std::time::Instant::now() < deadline,
                "journal event did not become visible within the flush interval"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(
            read_values(&path),
            vec![serde_json::json!({"visible": true})]
        );
        drop(journal);
        let _ = std::fs::remove_file(path);
    }

    struct SlowEvent {
        ordinal: u64,
    }

    impl Serialize for SlowEvent {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            std::thread::sleep(Duration::from_millis(2));
            let mut state = serializer.serialize_struct("SlowEvent", 1)?;
            state.serialize_field("ordinal", &self.ordinal)?;
            state.end()
        }
    }

    #[test]
    fn continuous_event_stream_cannot_starve_periodic_flush() {
        let path = temp_path("continuous-periodic-flush");
        let journal = JsonlJournal::open(
            path.clone(),
            JsonlJournalOpenMode::Truncate,
            JsonlJournalConfig {
                queue_capacity: 128,
                buffer_capacity_bytes: 1024 * 1024,
                flush_interval: Duration::from_millis(25),
            },
        )
        .unwrap();
        for ordinal in 0..100 {
            journal.enqueue(SlowEvent { ordinal }).unwrap();
        }

        let deadline = Instant::now() + Duration::from_millis(100);
        while std::fs::read_to_string(&path).unwrap().is_empty() {
            assert!(
                Instant::now() < deadline,
                "continuous event stream starved the periodic flush"
            );
            std::thread::sleep(Duration::from_millis(2));
        }

        journal.close().unwrap();
        assert_eq!(read_values(&path).len(), 100);
        let _ = std::fs::remove_file(path);
    }
}
