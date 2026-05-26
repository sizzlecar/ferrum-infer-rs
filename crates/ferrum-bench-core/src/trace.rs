//! Chrome Trace Event JSON emission — PLAYBOOK § Phase 1.5.
//!
//! Emits per-iteration spans in the de-facto-standard Chrome Trace Event
//! format (https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview),
//! consumable by Perfetto, Nsight Systems, and chrome://tracing without
//! conversion. This is the same format `vllm/profiler/layerwise_profile.py`
//! emits via `torch.profiler` — keeping ferrum's traces interchangeable
//! with vLLM's existing visualizer tooling (Phase 4).
//!
//! ## Format
//!
//! ```json
//! [
//!   {"name": "rms_norm", "cat": "norm", "ph": "X", "ts": 1234, "dur": 56, "pid": 0, "tid": 1},
//!   ...
//! ]
//! ```
//!
//! `ph: "X"` = a complete event (begin + end implied by `dur`). Other
//! phases (B/E for separate begin/end, M for metadata) are supported by
//! the format but not used here — complete events are simpler and match
//! how `BackendTimer` measures (one scope = one record_start/record_end pair).
//!
//! ## Wiring (Phase 1.2 — separate PR)
//!
//! Engine reads `FERRUM_TRACE_OUT=trace.json` at startup. A
//! `TraceWriter` is held in the engine; each migrated `FERRUM_*_PROF`
//! probe pushes a `TraceEvent` after `BackendTimer::elapsed_ms()`.
//! On engine drop, `TraceWriter::flush()` writes the array out.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Global TraceWriter — lazy-initialized from `FERRUM_TRACE_OUT` env
/// on first access. Returns a disabled writer when env is unset, so
/// callers can unconditionally `global_trace().push(...)` without a
/// per-call gate.
///
/// On program exit, `Drop` flushes buffered events to disk. For
/// processes that don't exit cleanly (e.g. killed by signal),
/// explicit `flush()` is required to avoid losing the buffer.
static GLOBAL_TRACE: OnceLock<TraceWriter> = OnceLock::new();

/// Get the global trace writer. Cheap (just an atomic load after the
/// first call) — safe to call from hot paths.
pub fn global_trace() -> &'static TraceWriter {
    GLOBAL_TRACE.get_or_init(TraceWriter::from_env)
}

/// Explicit flush of the global writer — useful before SIGINT / panic
/// hooks so the partial trace is on disk.
pub fn flush_global_trace() {
    if let Some(w) = GLOBAL_TRACE.get() {
        let _ = w.flush();
    }
}

/// One trace event ("complete" phase only — see module docs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: char, // always 'X'
    /// Microseconds since trace start.
    pub ts: u64,
    /// Duration in microseconds.
    pub dur: u64,
    pub pid: u32,
    pub tid: u32,
    /// Optional structured payload (e.g. `{ "tokens": 4, "dim": 4096 }`).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub args: serde_json::Map<String, serde_json::Value>,
}

impl TraceEvent {
    /// Construct a complete event from a name + category + elapsed ms.
    /// `start_ts_us` is the absolute timestamp at which this event began
    /// (microseconds since the writer's epoch).
    pub fn complete(
        name: impl Into<String>,
        cat: impl Into<String>,
        start_ts_us: u64,
        dur_ms: f64,
        tid: u32,
    ) -> Self {
        Self {
            name: name.into(),
            cat: cat.into(),
            ph: 'X',
            ts: start_ts_us,
            dur: (dur_ms * 1000.0).round() as u64,
            pid: 0,
            tid,
            args: serde_json::Map::new(),
        }
    }
}

/// Buffered, flush-on-drop trace writer.
///
/// Events accumulate in memory; `flush` (called on drop, or explicitly)
/// writes the buffered JSON array to disk. Disabled via the special
/// "no-op" constructor for builds where `FERRUM_TRACE_OUT` is unset —
/// `push` becomes a no-op so probe call-sites can call into the writer
/// unconditionally.
pub struct TraceWriter {
    inner: Mutex<TraceWriterInner>,
}

enum TraceWriterInner {
    Disabled,
    Buffering {
        out_path: PathBuf,
        events: Vec<TraceEvent>,
        epoch: std::time::Instant,
    },
}

impl TraceWriter {
    /// Construct from env var `FERRUM_TRACE_OUT`. If unset or empty,
    /// returns a disabled writer whose `push` is a no-op.
    pub fn from_env() -> Self {
        match std::env::var("FERRUM_TRACE_OUT") {
            Ok(p) if !p.is_empty() => Self::enabled(PathBuf::from(p)),
            _ => Self::disabled(),
        }
    }

    pub fn enabled(out_path: PathBuf) -> Self {
        Self {
            inner: Mutex::new(TraceWriterInner::Buffering {
                out_path,
                events: Vec::with_capacity(1024),
                epoch: std::time::Instant::now(),
            }),
        }
    }

    pub fn disabled() -> Self {
        Self {
            inner: Mutex::new(TraceWriterInner::Disabled),
        }
    }

    /// True if the writer is configured to emit. Probes can use this to
    /// skip the `BackendTimer` overhead entirely when tracing is off.
    pub fn is_enabled(&self) -> bool {
        matches!(
            *self.inner.lock().unwrap(),
            TraceWriterInner::Buffering { .. }
        )
    }

    /// Record a complete event with `name`, `cat`, elapsed milliseconds.
    /// `tid` should identify the layer / sub-op (0 for top-level engine).
    pub fn push(&self, name: impl Into<String>, cat: impl Into<String>, dur_ms: f64, tid: u32) {
        let mut inner = self.inner.lock().unwrap();
        if let TraceWriterInner::Buffering { events, epoch, .. } = &mut *inner {
            let now = std::time::Instant::now();
            let ts_us = now.duration_since(*epoch).as_micros() as u64;
            // The event "started" `dur_ms` before now — back-date the ts
            // so chrome://tracing renders the bar where it actually ran.
            let start_us = ts_us.saturating_sub((dur_ms * 1000.0) as u64);
            events.push(TraceEvent::complete(name, cat, start_us, dur_ms, tid));
        }
    }

    /// Same as `push`, but with structured args (e.g. tensor shapes).
    pub fn push_with_args(
        &self,
        name: impl Into<String>,
        cat: impl Into<String>,
        dur_ms: f64,
        tid: u32,
        args: serde_json::Map<String, serde_json::Value>,
    ) {
        let mut inner = self.inner.lock().unwrap();
        if let TraceWriterInner::Buffering { events, epoch, .. } = &mut *inner {
            let now = std::time::Instant::now();
            let ts_us = now.duration_since(*epoch).as_micros() as u64;
            let start_us = ts_us.saturating_sub((dur_ms * 1000.0) as u64);
            let mut e = TraceEvent::complete(name, cat, start_us, dur_ms, tid);
            e.args = args;
            events.push(e);
        }
    }

    /// Write the buffered events out as a JSON array. Subsequent `push`
    /// calls are buffered into a new file (caller responsibility — flushed
    /// writers reset their event buffer).
    pub fn flush(&self) -> std::io::Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if let TraceWriterInner::Buffering {
            out_path, events, ..
        } = &mut *inner
        {
            let json = serde_json::to_string(&events).expect("serialize trace");
            std::fs::write(out_path, json)?;
            events.clear();
        }
        Ok(())
    }
}

impl Drop for TraceWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_event_round_trip() {
        let e = TraceEvent::complete("rms_norm", "norm", 1_000_000, 0.123, 1);
        assert_eq!(e.ph, 'X');
        assert_eq!(e.dur, 123); // 0.123 ms = 123 us
        let j = serde_json::to_string(&e).unwrap();
        let back: TraceEvent = serde_json::from_str(&j).unwrap();
        assert_eq!(back.name, "rms_norm");
        assert_eq!(back.dur, 123);
    }

    #[test]
    fn disabled_writer_is_noop() {
        let w = TraceWriter::disabled();
        w.push("rms_norm", "norm", 1.0, 0);
        assert!(!w.is_enabled());
        w.flush().unwrap(); // no-op
    }

    #[test]
    fn enabled_writer_flushes_to_file() {
        let dir = tempdir();
        let path = dir.join("trace.json");
        let w = TraceWriter::enabled(path.clone());
        w.push("rms_norm", "norm", 1.0, 1);
        w.push("rope", "attn", 0.5, 1);
        w.flush().unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        let events: Vec<TraceEvent> = serde_json::from_str(&s).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].name, "rms_norm");
        assert_eq!(events[1].cat, "attn");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tempdir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ferrum-trace-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
