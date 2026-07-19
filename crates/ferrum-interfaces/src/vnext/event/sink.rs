use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::{
    has_active, ExecutionEvent, ExecutionEventCursor, ExecutionEventKind, RequestIdentity, RunId,
    TrustedExecutionEventContext,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionEventSinkError {
    message: String,
}

impl ExecutionEventSinkError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ExecutionEventSinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ExecutionEventSinkError {}

mod event_sink_seal {
    pub struct Seal;
}

/// Owned capability created only after the emitter has validated the event
/// against its transactional cursor. Ownership lets asynchronous sinks defer
/// materialization without cloning the event or extending producer lifetimes.
pub struct EventEmissionPermit {
    event: ExecutionEvent,
    _seal: event_sink_seal::Seal,
}

impl EventEmissionPermit {
    pub fn event(&self) -> &ExecutionEvent {
        &self.event
    }

    pub fn into_event(self) -> ExecutionEvent {
        self.event
    }
}

/// Owned capability created only after the emitter has validated an ordered
/// event batch against one transactional cursor.
pub struct EventBatchEmissionPermit {
    events: Vec<ExecutionEvent>,
    _seal: event_sink_seal::Seal,
}

impl EventBatchEmissionPermit {
    pub fn events(&self) -> &[ExecutionEvent] {
        &self.events
    }

    pub fn into_events(self) -> Vec<ExecutionEvent> {
        self.events
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionEventCapturePolicy {
    #[default]
    AllFrames,
    FirstFramePerRequest,
}

impl ExecutionEventCapturePolicy {
    pub const fn captures_frame(self, completed_frames: u64) -> bool {
        match self {
            Self::AllFrames => true,
            Self::FirstFramePerRequest => completed_frames == 0,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AllFrames => "all_frames",
            Self::FirstFramePerRequest => "first_frame_per_request",
        }
    }
}

pub trait ExecutionEventSink: Send + Sync {
    fn is_enabled(&self, kind: ExecutionEventKind) -> bool;

    fn device_timing_mode(&self) -> super::super::DeviceTimingMode {
        super::super::DeviceTimingMode::Off
    }

    fn capture_policy(&self) -> ExecutionEventCapturePolicy {
        ExecutionEventCapturePolicy::AllFrames
    }

    fn record(&self, permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError>;

    /// Records one cursor-ordered batch. Sinks with a buffered transport should
    /// override this boundary; the default preserves compatibility and order.
    fn record_batch(
        &self,
        permit: EventBatchEmissionPermit,
    ) -> Result<(), ExecutionEventSinkError> {
        for event in permit.into_events() {
            self.record(EventEmissionPermit {
                event,
                _seal: event_sink_seal::Seal,
            })?;
        }
        Ok(())
    }
}

enum ExecutionEventSinkHandle<'sink> {
    Borrowed(&'sink dyn ExecutionEventSink),
    Shared(Arc<dyn ExecutionEventSink>),
}

impl ExecutionEventSinkHandle<'_> {
    fn as_sink(&self) -> &dyn ExecutionEventSink {
        match self {
            Self::Borrowed(sink) => *sink,
            Self::Shared(sink) => sink.as_ref(),
        }
    }
}

pub struct ExecutionEventEmitter<'sink> {
    sink: ExecutionEventSinkHandle<'sink>,
    cursor: ExecutionEventCursor,
    sink_failed: bool,
}

impl<'sink> ExecutionEventEmitter<'sink> {
    pub fn new(
        sink: &'sink dyn ExecutionEventSink,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> Self {
        Self {
            sink: ExecutionEventSinkHandle::Borrowed(sink),
            cursor: ExecutionEventCursor::new(run_id, request_id),
            sink_failed: false,
        }
    }

    /// Creates a durable emitter that may be owned by a request/session.
    ///
    /// The borrowed constructor remains useful for bounded validation. Product
    /// runtimes use this form so event authority cannot outlive its sink.
    pub fn from_shared(
        sink: Arc<dyn ExecutionEventSink>,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> ExecutionEventEmitter<'static> {
        ExecutionEventEmitter {
            sink: ExecutionEventSinkHandle::Shared(sink),
            cursor: ExecutionEventCursor::new(run_id, request_id),
            sink_failed: false,
        }
    }

    fn validate_next(
        cursor: &mut ExecutionEventCursor,
        event: &ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        let requires_open_sequence = matches!(
            event.kind(),
            ExecutionEventKind::FrameStarted | ExecutionEventKind::NodeStarted
        );
        let requires_live_sequence = matches!(
            event.kind(),
            ExecutionEventKind::OperationSubmitted
                | ExecutionEventKind::NodeRetired
                | ExecutionEventKind::FrameCompleted
        ) || event.kind() == ExecutionEventKind::FailureObserved
            && has_active(event.identity().parts());
        if requires_open_sequence || requires_live_sequence {
            let active = context.active_binding().ok_or_else(|| {
                ExecutionEventSinkError::new(
                    "active execution emission lacks live sequence evidence",
                )
            })?;
            let live = if requires_open_sequence {
                active.ensure_open_for_emission()
            } else {
                active.ensure_live_for_emission()
            };
            live.map_err(|error| ExecutionEventSinkError::new(error.to_string()))?;
        }
        cursor
            .observe_against(event, context)
            .map_err(|error| ExecutionEventSinkError::new(error.to_string()))
    }

    pub fn emit(
        &mut self,
        event: ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        if self.sink_failed {
            return Err(ExecutionEventSinkError::new(
                "execution event emitter is sealed after a sink failure",
            ));
        }
        let mut next_cursor = self.cursor.clone();
        Self::validate_next(&mut next_cursor, &event, context)?;
        if self.sink.as_sink().is_enabled(event.kind()) {
            let permit = EventEmissionPermit {
                event,
                _seal: event_sink_seal::Seal,
            };
            if let Err(error) = self.sink.as_sink().record(permit) {
                self.sink_failed = true;
                return Err(error);
            }
        }
        self.cursor = next_cursor;
        Ok(())
    }

    pub fn emit_batch(
        &mut self,
        events: Vec<ExecutionEvent>,
        contexts: &[TrustedExecutionEventContext<'_>],
    ) -> Result<(), ExecutionEventSinkError> {
        if self.sink_failed {
            return Err(ExecutionEventSinkError::new(
                "execution event emitter is sealed after a sink failure",
            ));
        }
        if events.len() != contexts.len() {
            return Err(ExecutionEventSinkError::new(
                "execution event batch context count differs from event count",
            ));
        }
        if events.is_empty() {
            return Ok(());
        }

        let mut next_cursor = self.cursor.clone();
        for (event, context) in events.iter().zip(contexts) {
            Self::validate_next(&mut next_cursor, event, context)?;
        }

        let all_enabled = events
            .iter()
            .all(|event| self.sink.as_sink().is_enabled(event.kind()));
        if all_enabled {
            let permit = EventBatchEmissionPermit {
                events,
                _seal: event_sink_seal::Seal,
            };
            if let Err(error) = self.sink.as_sink().record_batch(permit) {
                self.sink_failed = true;
                return Err(error);
            }
        } else {
            for event in events {
                if !self.sink.as_sink().is_enabled(event.kind()) {
                    continue;
                }
                let permit = EventEmissionPermit {
                    event,
                    _seal: event_sink_seal::Seal,
                };
                if let Err(error) = self.sink.as_sink().record(permit) {
                    self.sink_failed = true;
                    return Err(error);
                }
            }
        }
        self.cursor = next_cursor;
        Ok(())
    }

    pub fn cursor(&self) -> &ExecutionEventCursor {
        &self.cursor
    }

    pub const fn sink_failed(&self) -> bool {
        self.sink_failed
    }
}

#[derive(Debug, Default)]
pub struct DisabledExecutionEventSink;

impl ExecutionEventSink for DisabledExecutionEventSink {
    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        false
    }

    fn record(&self, _permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }
}
