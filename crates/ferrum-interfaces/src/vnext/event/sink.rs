use std::error::Error;
use std::fmt;

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

/// Capability created only after the emitter has validated the event against
/// its transactional cursor. External callers cannot construct this value.
pub struct EventEmissionPermit<'event> {
    event: &'event ExecutionEvent,
    _seal: event_sink_seal::Seal,
}

impl<'event> EventEmissionPermit<'event> {
    pub fn event(&self) -> &'event ExecutionEvent {
        self.event
    }
}

pub trait ExecutionEventSink: Send + Sync {
    fn is_enabled(&self, kind: ExecutionEventKind) -> bool;

    fn record(
        &self,
        event: &ExecutionEvent,
        permit: EventEmissionPermit<'_>,
    ) -> Result<(), ExecutionEventSinkError>;
}

pub struct ExecutionEventEmitter<'sink> {
    sink: &'sink dyn ExecutionEventSink,
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
            sink,
            cursor: ExecutionEventCursor::new(run_id, request_id),
            sink_failed: false,
        }
    }

    pub fn emit(
        &mut self,
        event: &ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        if self.sink_failed {
            return Err(ExecutionEventSinkError::new(
                "execution event emitter is sealed after a sink failure",
            ));
        }
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
        let mut next_cursor = self.cursor.clone();
        next_cursor
            .observe_against(event, context)
            .map_err(|error| ExecutionEventSinkError::new(error.to_string()))?;
        if self.sink.is_enabled(event.kind()) {
            let permit = EventEmissionPermit {
                event,
                _seal: event_sink_seal::Seal,
            };
            if let Err(error) = self.sink.record(event, permit) {
                self.sink_failed = true;
                return Err(error);
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

    fn record(
        &self,
        _event: &ExecutionEvent,
        _permit: EventEmissionPermit<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }
}
