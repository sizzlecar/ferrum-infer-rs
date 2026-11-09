//! Optional bounded observations of completed image work. The collector owns
//! one real Output backing from the caller's Host account. Small Arc/Mutex and
//! serializer control objects are outside that backing limit, as elsewhere in
//! the Host API; this is not a claim about total process RSS.
//!
//! Appending never changes inference success or retains model/runtime owners.
//! Export holds the collector lock, so the normal product path exports after
//! the original pipeline has retired. An unsealed export is only a snapshot.

use anyhow::{ensure, Context, Result};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBuffer};
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};
use uuid::Uuid;
mod events;
pub use events::*;

const SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageContinuousTraceLimits {
    /// Logical capacity for complete JSONL event records, excluding the small
    /// export footer. Host alignment may require more physical backing.
    pub maximum_bytes: usize,
}
impl ImageContinuousTraceLimits {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.maximum_bytes > 0 && self.maximum_bytes <= isize::MAX as usize,
            "continuous image trace capacity must be positive and addressable"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageContinuousTraceStatus {
    pub schema_version: u32,
    pub session_uuid: Uuid,
    pub recorded: u64,
    pub dropped: u64,
    pub sealed: bool,
    /// Sticky: at least one record was lost, could not be encoded, or arrived
    /// after seal. A truncated trace cannot establish complete execution history.
    pub truncated: bool,
    pub used_bytes: usize,
    pub maximum_bytes: usize,
    pub backing_bytes: usize,
}

struct Shared {
    session_uuid: Uuid,
    epoch: Instant,
    backing_bytes: usize,
    state: Mutex<State>,
}
struct State {
    buffer: ManagedHostBuffer,
    cursor: usize,
    recorded: u64,
    dropped: u64,
    sealed: bool,
    truncated: bool,
    last_monotonic_ns: u64,
}
impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.truncated = true;
                state
            }
        }
    }
    fn status(&self, state: &State) -> ImageContinuousTraceStatus {
        ImageContinuousTraceStatus {
            schema_version: SCHEMA_VERSION,
            session_uuid: self.session_uuid,
            recorded: state.recorded,
            dropped: state.dropped,
            sealed: state.sealed,
            truncated: state.truncated,
            used_bytes: state.cursor,
            maximum_bytes: state.buffer.len(),
            backing_bytes: self.backing_bytes,
        }
    }
}
impl State {
    fn lose_record(&mut self) {
        self.truncated = true;
        self.dropped = self.dropped.saturating_add(1);
    }
}

pub(crate) struct ImageContinuousTraceRecorder(Arc<Shared>);

/// Clones retain the same diagnostic backing, never a tensor, native owner,
/// runtime or additional account. The last recorder/handle releases its charge.
#[derive(Clone)]
pub struct ImageContinuousTraceHandle(Arc<Shared>);

#[derive(Serialize)]
struct Envelope<'a, T: ?Sized> {
    schema_version: u32,
    session_uuid: Uuid,
    sequence: u64,
    /// Serialized observation order, sampled under the collector lock.
    monotonic_ns: u64,
    /// Actual producer boundary in the same clock domain. Concurrent producer
    /// boundaries may arrive out of order; their timestamps are never clamped.
    event_ns: u64,
    event: &'a T,
}

impl ImageContinuousTraceRecorder {
    /// The original physical Output allocation remains held for the producer's
    /// entire lifetime, including after seal. This is not additional demand.
    pub(crate) fn backing_bytes(&self) -> usize {
        self.0.backing_bytes
    }

    pub(crate) fn new(
        host: &HostMemoryAccount,
        limits: ImageContinuousTraceLimits,
    ) -> Result<(Self, ImageContinuousTraceHandle)> {
        limits.validate()?;
        let backing_bytes = host.backing_bytes(limits.maximum_bytes)?;
        let buffer = host
            .allocate_zeroed(HostMemoryClass::Output, limits.maximum_bytes)
            .context("admit continuous image trace Output backing")?;
        let shared = Arc::new(Shared {
            session_uuid: Uuid::new_v4(),
            epoch: Instant::now(),
            backing_bytes,
            state: Mutex::new(State {
                buffer,
                cursor: 0,
                recorded: 0,
                dropped: 0,
                sealed: false,
                truncated: false,
                last_monotonic_ns: 0,
            }),
        });
        Ok((Self(shared.clone()), ImageContinuousTraceHandle(shared)))
    }

    /// Commit one complete record and newline, or retain the previous prefix.
    /// Once incomplete, later observations are counted but never overwrite or
    /// silently resume a history with missing events. Serializer panics are
    /// contained too; callers must still supply bounded typed event views.
    pub(crate) fn record<T: Serialize>(&self, event: &T) {
        let _ = self.0.record_at(Instant::now(), event);
    }

    pub(crate) fn handle(&self) -> ImageContinuousTraceHandle {
        ImageContinuousTraceHandle(self.0.clone())
    }

    /// Only the original pipeline's acknowledged shutdown authorizes the
    /// caller to seal. This flag itself does not prove native retirement.
    pub(crate) fn seal(&self) {
        self.0.lock().sealed = true;
    }
}

impl Shared {
    fn record_at<T: Serialize>(&self, at: Instant, event: &T) -> Option<u64> {
        let mut state = self.lock();
        let times = || {
            let observed = Instant::now().checked_duration_since(self.epoch)?;
            let boundary = at.checked_duration_since(self.epoch)?;
            Some((
                u64::try_from(observed.as_nanos()).ok()?,
                u64::try_from(boundary.as_nanos()).ok()?,
            ))
        };
        let Some((observed, boundary)) = times() else {
            state.lose_record();
            return None;
        };
        self.append(&mut state, event, observed, boundary)
    }

    fn append<T: Serialize>(
        &self,
        state: &mut State,
        event: &T,
        monotonic_ns: u64,
        event_ns: u64,
    ) -> Option<u64> {
        if state.sealed || state.truncated {
            state.lose_record();
            return None;
        }
        if monotonic_ns < state.last_monotonic_ns || event_ns > monotonic_ns {
            state.lose_record();
            return None;
        }
        let Some(sequence) = state.recorded.checked_add(1) else {
            state.lose_record();
            return None;
        };
        let start = state.cursor;
        let envelope = Envelope {
            schema_version: SCHEMA_VERSION,
            session_uuid: self.session_uuid,
            sequence,
            monotonic_ns,
            event_ns,
            event,
        };
        let (completed, written) = {
            let mut writer = BoundedWriter {
                bytes: &mut state.buffer.as_mut()[start..],
                written: 0,
            };
            let completed = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
                serde_json::to_writer(&mut writer, &envelope)?;
                writer.write_all(b"\n")?;
                Ok(())
            }))
            .is_ok_and(|result| result.is_ok());
            (completed, writer.written)
        };
        if completed {
            state.cursor = start + written;
            state.recorded = sequence;
            state.last_monotonic_ns = monotonic_ns;
            Some(sequence)
        } else {
            // Clear failed bytes as well as rolling back the logical cursor.
            // No partial JSON or uncommitted event is ever exported.
            state.buffer.as_mut()[start..start + written].fill(0);
            state.lose_record();
            None
        }
    }
}

impl ImageContinuousTraceHandle {
    /// Internal model-specific diagnostic views use the same charged buffer,
    /// atomic append and loss accounting. The public producer API stays typed.
    pub(crate) fn record_diagnostic<T: Serialize>(&self, event: &T) -> Option<u64> {
        self.0.record_at(Instant::now(), event)
    }

    pub fn session_uuid(&self) -> Uuid {
        self.0.session_uuid
    }

    /// Return a diagnostic span identity only after its complete start record
    /// committed. Failure never authorizes retrying or rejecting model work.
    pub fn record(&self, event: &ImageContinuousTraceEvent<'_>) -> Option<u64> {
        self.record_at(Instant::now(), event)
    }

    /// Record the original boundary (for example `Queued.submitted`). Times
    /// before this session or in the future make the trace incomplete rather
    /// than inventing a duration. No runtime/tensor owner is retained.
    pub fn record_at(&self, at: Instant, event: &ImageContinuousTraceEvent<'_>) -> Option<u64> {
        self.0.record_at(at, event)
    }

    pub fn status(&self) -> ImageContinuousTraceStatus {
        self.0.status(&self.0.lock())
    }

    /// Write the exact complete prefix followed by its snapshot footer. This
    /// neither seals the collector nor clones its managed buffer. A writer
    /// error leaves collection unchanged and permits a later export retry.
    pub fn write_jsonl(&self, writer: &mut impl Write) -> Result<ImageContinuousTraceStatus> {
        let state = self.0.lock();
        let status = self.0.status(&state);
        writer
            .write_all(&state.buffer.as_ref()[..state.cursor])
            .context("write continuous image trace prefix")?;
        serde_json::to_writer(
            &mut *writer,
            &Footer {
                schema_version: SCHEMA_VERSION,
                session_uuid: status.session_uuid,
                footer: FooterState {
                    sealed: status.sealed,
                    truncated: status.truncated,
                    recorded: status.recorded,
                    dropped: status.dropped,
                    used_bytes: status.used_bytes,
                    maximum_bytes: status.maximum_bytes,
                    backing_bytes: status.backing_bytes,
                },
            },
        )
        .context("write continuous image trace footer")?;
        writer.write_all(b"\n")?;
        Ok(status)
    }
}

#[derive(Serialize)]
struct Footer {
    schema_version: u32,
    session_uuid: Uuid,
    footer: FooterState,
}
#[derive(Serialize)]
struct FooterState {
    sealed: bool,
    truncated: bool,
    recorded: u64,
    dropped: u64,
    used_bytes: usize,
    maximum_bytes: usize,
    backing_bytes: usize,
}

struct BoundedWriter<'a> {
    bytes: &'a mut [u8],
    written: usize,
}
impl Write for BoundedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let remaining = &mut self.bytes[self.written..];
        if bytes.len() > remaining.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "continuous image trace is full",
            ));
        }
        remaining[..bytes.len()].copy_from_slice(bytes);
        self.written += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
