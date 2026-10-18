//! Output ownership independent of model execution and HTTP transport.
//!
//! A completed request can still own unread output. Sessions and byte leases
//! follow that output, rather than being returned when model resources close.

use ferrum_types::{FerrumError, OutputConfig, Result, StreamChunk};
use futures::Stream;
use std::{
    fmt, io,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

/// A single terminal error can explain a full or undersized data pool without
/// waiting for it. This fixed allowance is per admitted session, never per event.
pub const TERMINAL_ERROR_BYTES: usize = 1024;

#[derive(Debug)]
struct BudgetInner {
    config: OutputConfig,
    sessions: Arc<Semaphore>,
    queued: Arc<Semaphore>,
    encoded: Arc<Semaphore>,
    control_bytes: AtomicUsize,
}

/// One instance is shared by an engine and all of its downstream consumers.
#[derive(Debug, Clone)]
pub struct OutputBudget(Arc<BudgetInner>);

impl OutputBudget {
    pub fn new(config: OutputConfig) -> Result<Self> {
        config.validate()?;
        config
            .max_sessions
            .checked_mul(TERMINAL_ERROR_BYTES)
            .ok_or_else(|| FerrumError::config("output error allowance exceeds address space"))?;
        for count in [
            config.max_sessions,
            config.engine_queue_items,
            config.queued_bytes_total,
            config.encoded_bytes_total,
        ] {
            if count > Semaphore::MAX_PERMITS {
                return Err(FerrumError::config(
                    "output limit exceeds this platform's semaphore capacity",
                ));
            }
        }
        Ok(Self(Arc::new(BudgetInner {
            sessions: Arc::new(Semaphore::new(config.max_sessions)),
            queued: Arc::new(Semaphore::new(config.queued_bytes_total)),
            encoded: Arc::new(Semaphore::new(config.encoded_bytes_total)),
            control_bytes: AtomicUsize::new(0),
            config,
        })))
    }

    /// Admission is bounded before model or scheduler resources are opened.
    pub fn try_open_session(&self) -> Result<OutputSession> {
        let permit = self.0.sessions.clone().try_acquire_owned().map_err(|_| {
            exhausted("output session limit reached; consume or drop an existing output")
        })?;
        Ok(OutputSession(Arc::new(SessionInner {
            budget: self.0.clone(),
            _permit: permit,
            error_claimed: AtomicBool::new(false),
        })))
    }

    pub fn config(&self) -> &OutputConfig {
        &self.0.config
    }

    pub fn usage(&self) -> OutputUsage {
        OutputUsage {
            sessions: self.0.config.max_sessions - self.0.sessions.available_permits(),
            queued_bytes: self.0.config.queued_bytes_total - self.0.queued.available_permits(),
            encoded_bytes: self.0.config.encoded_bytes_total - self.0.encoded.available_permits(),
            control_bytes: self.0.control_bytes.load(Ordering::Acquire),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputUsage {
    pub sessions: usize,
    pub queued_bytes: usize,
    pub encoded_bytes: usize,
    pub control_bytes: usize,
}

#[derive(Debug)]
struct SessionInner {
    budget: Arc<BudgetInner>,
    _permit: OwnedSemaphorePermit,
    error_claimed: AtomicBool,
}

/// Cloning transfers shared ownership of the same session, not a new admission.
#[derive(Debug, Clone)]
pub struct OutputSession(Arc<SessionInner>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferKind {
    Queued,
    Encoded,
    TerminalError,
}

/// A non-cloneable reservation. It retains its session until the content dies.
#[derive(Debug)]
pub struct OutputBytesLease {
    _session: OutputSession,
    _permit: Option<OwnedSemaphorePermit>,
    bytes: usize,
    kind: BufferKind,
}

impl OutputBytesLease {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for OutputBytesLease {
    fn drop(&mut self) {
        if self.kind == BufferKind::TerminalError {
            self._session
                .0
                .budget
                .control_bytes
                .fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

impl OutputSession {
    pub fn config(&self) -> &OutputConfig {
        &self.0.budget.config
    }

    fn pool(&self, kind: BufferKind, bytes: usize) -> Result<Arc<Semaphore>> {
        let (pool, limit) = match kind {
            BufferKind::Queued => (&self.0.budget.queued, self.config().queued_bytes_total),
            BufferKind::Encoded => (&self.0.budget.encoded, self.config().max_event_bytes),
            BufferKind::TerminalError => {
                return Err(FerrumError::internal(
                    "terminal errors require their one-shot allowance",
                ))
            }
        };
        if bytes > limit {
            return Err(exhausted(format!(
                "output {kind:?} content needs {bytes} bytes; limit is {limit}"
            )));
        }
        Ok(pool.clone())
    }

    pub fn try_reserve_queued(&self, bytes: usize) -> Result<Option<OutputBytesLease>> {
        let pool = self.pool(BufferKind::Queued, bytes)?;
        match pool.try_acquire_many_owned(bytes as u32) {
            Ok(permit) => Ok(Some(OutputBytesLease {
                _session: self.clone(),
                _permit: Some(permit),
                bytes,
                kind: BufferKind::Queued,
            })),
            Err(TryAcquireError::NoPermits) => Ok(None),
            Err(TryAcquireError::Closed) => Err(exhausted("output queue budget is closed")),
        }
    }

    async fn reserve(&self, kind: BufferKind, bytes: usize) -> Result<OutputBytesLease> {
        let permit = self
            .pool(kind, bytes)?
            .acquire_many_owned(bytes as u32)
            .await
            .map_err(|_| exhausted("output budget is closed"))?;
        Ok(OutputBytesLease {
            _session: self.clone(),
            _permit: Some(permit),
            bytes,
            kind,
        })
    }

    pub async fn reserve_queued(&self, bytes: usize) -> Result<OutputBytesLease> {
        self.reserve(BufferKind::Queued, bytes).await
    }

    pub async fn reserve_encoded(&self, bytes: usize) -> Result<OutputBytesLease> {
        self.reserve(BufferKind::Encoded, bytes).await
    }

    /// One bounded error per session, independent of normal data credit. It
    /// cannot be recycled into an unlimited sequence of control frames.
    pub fn try_reserve_error(&self, bytes: usize) -> Result<OutputBytesLease> {
        if bytes > TERMINAL_ERROR_BYTES {
            return Err(exhausted(
                "terminal output error exceeds its fixed allowance",
            ));
        }
        self.0
            .error_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| exhausted("terminal output error was already published"))?;
        self.0
            .budget
            .control_bytes
            .fetch_add(bytes, Ordering::AcqRel);
        Ok(OutputBytesLease {
            _session: self.clone(),
            _permit: None,
            bytes,
            kind: BufferKind::TerminalError,
        })
    }

    /// Suitable for a pull producer that owns at most its current input. Engine
    /// drivers use try_reserve_queued and park only the affected request instead.
    pub async fn wrap_chunk(&self, chunk: StreamChunk) -> Result<OutputChunk> {
        let bytes = OutputChunk::queued_cost(&chunk)?;
        let lease = self.reserve_queued(bytes).await?;
        Ok(OutputChunk {
            chunk,
            _lease: lease,
        })
    }
}

/// Payload and reservation cannot be separated or deeply cloned through this API.
#[derive(Debug)]
pub struct OutputChunk {
    chunk: StreamChunk,
    _lease: OutputBytesLease,
}

impl OutputChunk {
    /// Serialized content is a stable accounting unit across engine and client
    /// metadata. The counter does not allocate a second JSON representation.
    pub fn queued_cost(chunk: &StreamChunk) -> Result<usize> {
        let mut counter = ContentCounter(0);
        serde_json::to_writer(&mut counter, chunk).map_err(|error| {
            FerrumError::internal(format!("cannot count output chunk: {error}"))
        })?;
        Ok(counter.0)
    }

    pub fn new(chunk: StreamChunk, lease: OutputBytesLease) -> Result<Self> {
        if lease.kind != BufferKind::Queued || lease.bytes() < Self::queued_cost(&chunk)? {
            return Err(FerrumError::internal(
                "output chunk requires its complete queued-content reservation",
            ));
        }
        Ok(Self {
            chunk,
            _lease: lease,
        })
    }

    pub fn chunk(&self) -> &StreamChunk {
        &self.chunk
    }
}

struct ContentCounter(usize);

impl io::Write for ContentCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("output size overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The outer owner deliberately survives upstream EOF. Consumers can retain a
/// completed stream or a final HTTP frame independently of model completion.
pub struct InferenceStream {
    session: OutputSession,
    inner: Pin<Box<dyn Stream<Item = Result<OutputChunk>> + Send>>,
}

impl InferenceStream {
    pub fn new(
        session: OutputSession,
        inner: Pin<Box<dyn Stream<Item = Result<OutputChunk>> + Send>>,
    ) -> Self {
        Self { session, inner }
    }
    pub fn output_session(&self) -> &OutputSession {
        &self.session
    }
}

impl fmt::Debug for InferenceStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceStream").finish_non_exhaustive()
    }
}

impl Stream for InferenceStream {
    type Item = Result<OutputChunk>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

fn exhausted(message: impl Into<String>) -> FerrumError {
    FerrumError::ResourceExhausted {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests;
