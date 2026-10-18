use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::ExecutorRequestOrigin;

use super::{
    has_active, BoundExecutionResourceMaintenance, ExecutionEvent, ExecutionEventCursor,
    ExecutionEventKind, RequestIdentity, RunId, TrustedExecutionEventContext,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionEventSinkError {
    kind: ExecutionEventSinkErrorKind,
    message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionEventSinkErrorKind {
    Contract,
    Transport,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionEventDeliveryPolicy {
    #[default]
    RequiredEvidence,
    OptionalTelemetry,
}

impl ExecutionEventSinkError {
    /// Unclassified existing errors remain contract failures. Only an explicit
    /// transport classification may be degraded by optional telemetry.
    pub fn new(message: impl Into<String>) -> Self {
        Self::contract(message)
    }

    pub fn contract(message: impl Into<String>) -> Self {
        Self {
            kind: ExecutionEventSinkErrorKind::Contract,
            message: message.into(),
        }
    }

    pub fn transport(message: impl Into<String>) -> Self {
        Self {
            kind: ExecutionEventSinkErrorKind::Transport,
            message: message.into(),
        }
    }

    pub const fn kind(&self) -> ExecutionEventSinkErrorKind {
        self.kind
    }

    pub const fn is_optional_transport(&self, policy: ExecutionEventDeliveryPolicy) -> bool {
        matches!(self.kind, ExecutionEventSinkErrorKind::Transport)
            && matches!(policy, ExecutionEventDeliveryPolicy::OptionalTelemetry)
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
    LifecycleOnly,
}

impl ExecutionEventCapturePolicy {
    pub const fn captures_frame(self, completed_frames: u64) -> bool {
        match self {
            Self::AllFrames => true,
            Self::FirstFramePerRequest | Self::LifecycleOnly => completed_frames == 0,
        }
    }

    pub const fn records_event(self, kind: ExecutionEventKind) -> bool {
        match self {
            Self::AllFrames | Self::FirstFramePerRequest => true,
            Self::LifecycleOnly => matches!(
                kind,
                ExecutionEventKind::RequestAccepted
                    | ExecutionEventKind::PlanBuilt
                    | ExecutionEventKind::FailureObserved
                    | ExecutionEventKind::SequenceCompleted
                    | ExecutionEventKind::SequenceAborted
                    | ExecutionEventKind::RequestCompleted
                    | ExecutionEventKind::RequestFailed
            ),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AllFrames => "all_frames",
            Self::FirstFramePerRequest => "first_frame_per_request",
            Self::LifecycleOnly => "lifecycle_only",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionEventSinkEnablement {
    None,
    All,
    #[default]
    PerKind,
}

pub trait ExecutionEventSink: Send + Sync {
    /// Delivery requirements are independent of event validation and capture.
    /// Optional transports must retain their own loss/failure status: accepting
    /// an execution event is not proof of a complete or flushed evidence file.
    fn delivery_policy(&self) -> ExecutionEventDeliveryPolicy {
        ExecutionEventDeliveryPolicy::RequiredEvidence
    }

    /// Resolves stable event enablement once when an emitter is constructed.
    ///
    /// Dynamically filtered sinks keep the conservative `PerKind` default.
    /// Permanently disabled and all-event sinks select `None` or `All` so the
    /// event hot path does not perform a virtual enablement query.
    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::PerKind
    }

    fn is_enabled(&self, kind: ExecutionEventKind) -> bool;

    fn device_timing_mode(&self) -> super::super::DeviceTimingMode {
        super::super::DeviceTimingMode::Off
    }

    fn capture_policy(&self) -> ExecutionEventCapturePolicy {
        ExecutionEventCapturePolicy::AllFrames
    }

    fn capture_policy_for_request(
        &self,
        _origin: ExecutorRequestOrigin,
    ) -> ExecutionEventCapturePolicy {
        self.capture_policy()
    }

    fn record_device_submission_attribution(
        &self,
        _attribution: &super::super::BoundDeviceSubmissionAttribution,
    ) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }

    fn record_physical_device_submission_timing(
        &self,
        _completion: &super::super::OperationCompletionReceipt,
    ) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }

    /// Resource maintenance is a plan/batch event and therefore does not run
    /// through a per-request execution cursor. Sinks opt in explicitly so a
    /// disabled profile path performs no event construction or serialization.
    fn records_execution_resource_maintenance(&self) -> bool {
        false
    }

    fn record_execution_resource_maintenance(
        &self,
        _maintenance: BoundExecutionResourceMaintenance,
    ) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }

    fn record(&self, permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError>;

    /// Records one cursor-ordered batch. Sinks with a buffered transport should
    /// override this boundary; the default preserves compatibility and order.
    fn record_batch(
        &self,
        permit: EventBatchEmissionPermit,
    ) -> Result<(), ExecutionEventSinkError> {
        let mut transport_failure = None;
        for event in permit.into_events() {
            if let Err(error) = self.record(EventEmissionPermit {
                event,
                _seal: event_sink_seal::Seal,
            }) {
                if error.is_optional_transport(self.delivery_policy()) {
                    transport_failure.get_or_insert(error);
                } else {
                    return Err(error);
                }
            }
        }
        transport_failure.map_or(Ok(()), Err)
    }
}

enum ExecutionEventSinkHandle<'sink, S>
where
    S: ExecutionEventSink + ?Sized,
{
    Borrowed(&'sink S),
    Shared(Arc<S>),
}

impl<S> ExecutionEventSinkHandle<'_, S>
where
    S: ExecutionEventSink + ?Sized,
{
    #[inline]
    fn as_sink(&self) -> &S {
        match self {
            Self::Borrowed(sink) => *sink,
            Self::Shared(sink) => sink.as_ref(),
        }
    }
}

pub struct ExecutionEventEmitter<'sink, S = dyn ExecutionEventSink>
where
    S: ExecutionEventSink + ?Sized,
{
    sink: ExecutionEventSinkHandle<'sink, S>,
    cursor: ExecutionEventCursor,
    capture_policy: ExecutionEventCapturePolicy,
    sink_enablement: ExecutionEventSinkEnablement,
    delivery_policy: ExecutionEventDeliveryPolicy,
    sink_failure: Option<ExecutionEventSinkError>,
    transport_failure: Option<ExecutionEventSinkError>,
}

impl<'sink, S> ExecutionEventEmitter<'sink, S>
where
    S: ExecutionEventSink + ?Sized,
{
    #[inline]
    pub fn new(sink: &'sink S, run_id: RunId, request_id: RequestIdentity) -> Self {
        let sink_enablement = sink.enablement();
        let capture_policy = if sink_enablement == ExecutionEventSinkEnablement::None {
            ExecutionEventCapturePolicy::AllFrames
        } else {
            sink.capture_policy()
        };
        Self {
            sink: ExecutionEventSinkHandle::Borrowed(sink),
            cursor: ExecutionEventCursor::new(run_id, request_id),
            capture_policy,
            sink_enablement,
            delivery_policy: sink.delivery_policy(),
            sink_failure: None,
            transport_failure: None,
        }
    }
}

impl ExecutionEventEmitter<'static, dyn ExecutionEventSink> {
    /// Creates a durable emitter that may be owned by a request/session.
    ///
    /// The borrowed constructor remains useful for bounded validation. Product
    /// runtimes use this form so event authority cannot outlive its sink.
    pub fn from_shared(
        sink: Arc<dyn ExecutionEventSink>,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> ExecutionEventEmitter<'static> {
        let sink_enablement = sink.enablement();
        let capture_policy = if sink_enablement == ExecutionEventSinkEnablement::None {
            ExecutionEventCapturePolicy::AllFrames
        } else {
            sink.capture_policy()
        };
        Self::from_shared_parts(sink, run_id, request_id, capture_policy, sink_enablement)
    }

    pub fn from_shared_with_capture_policy(
        sink: Arc<dyn ExecutionEventSink>,
        run_id: RunId,
        request_id: RequestIdentity,
        capture_policy: ExecutionEventCapturePolicy,
    ) -> ExecutionEventEmitter<'static> {
        let sink_enablement = sink.enablement();
        Self::from_shared_parts(sink, run_id, request_id, capture_policy, sink_enablement)
    }

    fn from_shared_parts(
        sink: Arc<dyn ExecutionEventSink>,
        run_id: RunId,
        request_id: RequestIdentity,
        capture_policy: ExecutionEventCapturePolicy,
        sink_enablement: ExecutionEventSinkEnablement,
    ) -> ExecutionEventEmitter<'static> {
        let delivery_policy = sink.delivery_policy();
        ExecutionEventEmitter {
            sink: ExecutionEventSinkHandle::Shared(sink),
            cursor: ExecutionEventCursor::new(run_id, request_id),
            capture_policy,
            sink_enablement,
            delivery_policy,
            sink_failure: None,
            transport_failure: None,
        }
    }
}

impl<'sink, S> ExecutionEventEmitter<'sink, S>
where
    S: ExecutionEventSink + ?Sized,
{
    fn accept_delivery(
        &mut self,
        result: Result<(), ExecutionEventSinkError>,
    ) -> Result<(), ExecutionEventSinkError> {
        let Err(error) = result else { return Ok(()) };
        if error.kind() == ExecutionEventSinkErrorKind::Transport {
            self.transport_failure.get_or_insert_with(|| error.clone());
        }
        if error.is_optional_transport(self.delivery_policy) {
            // Do not disable a shared/fanout sink. A full queue may recover,
            // and an independent destination may still accept later events.
            return Ok(());
        }
        self.sink_failure = Some(error.clone());
        Err(error)
    }

    #[inline]
    fn records_event(&self, event: &ExecutionEvent) -> bool {
        self.capture_policy.records_event(event.kind())
            && match self.sink_enablement {
                ExecutionEventSinkEnablement::None => false,
                ExecutionEventSinkEnablement::All => true,
                ExecutionEventSinkEnablement::PerKind => {
                    self.sink.as_sink().is_enabled(event.kind())
                }
            }
    }

    #[inline]
    fn validate_next(
        cursor: &mut ExecutionEventCursor,
        event: &ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        match event.kind() {
            ExecutionEventKind::FrameStarted | ExecutionEventKind::NodeStarted => context
                .active_binding()
                .ok_or_else(|| {
                    ExecutionEventSinkError::contract(
                        "active execution emission lacks live sequence evidence",
                    )
                })?
                .ensure_open_for_emission()
                .map_err(|error| ExecutionEventSinkError::contract(error.to_string()))?,
            ExecutionEventKind::OperationSubmitted
            | ExecutionEventKind::NodeRetired
            | ExecutionEventKind::FrameCompleted => context
                .active_binding()
                .ok_or_else(|| {
                    ExecutionEventSinkError::contract(
                        "active execution emission lacks live sequence evidence",
                    )
                })?
                .ensure_live_for_emission()
                .map_err(|error| ExecutionEventSinkError::contract(error.to_string()))?,
            ExecutionEventKind::FailureObserved if has_active(event.identity().parts()) => context
                .active_binding()
                .ok_or_else(|| {
                    ExecutionEventSinkError::contract(
                        "active execution emission lacks live sequence evidence",
                    )
                })?
                .ensure_live_for_emission()
                .map_err(|error| ExecutionEventSinkError::contract(error.to_string()))?,
            _ => {}
        }
        cursor
            .observe_candidate_in_place(event, context)
            .map_err(|error| ExecutionEventSinkError::contract(error.to_string()))
    }

    #[inline]
    pub fn emit(
        &mut self,
        event: ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        if self.sink_enablement == ExecutionEventSinkEnablement::None {
            let mut next_cursor = self.cursor.clone();
            Self::validate_next(&mut next_cursor, &event, context)?;
            self.cursor = next_cursor;
            return Ok(());
        }
        if let Some(error) = &self.sink_failure {
            return Err(error.clone());
        }
        let mut next_cursor = self.cursor.clone();
        Self::validate_next(&mut next_cursor, &event, context)?;
        if self.records_event(&event) {
            let permit = EventEmissionPermit {
                event,
                _seal: event_sink_seal::Seal,
            };
            let delivered = self.sink.as_sink().record(permit);
            self.accept_delivery(delivered)?;
        }
        self.cursor = next_cursor;
        Ok(())
    }

    pub fn emit_batch(
        &mut self,
        events: Vec<ExecutionEvent>,
        contexts: &[TrustedExecutionEventContext<'_>],
    ) -> Result<(), ExecutionEventSinkError> {
        if self.sink_enablement == ExecutionEventSinkEnablement::None {
            if events.len() != contexts.len() {
                return Err(ExecutionEventSinkError::contract(
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
            self.cursor = next_cursor;
            return Ok(());
        }
        if let Some(error) = &self.sink_failure {
            return Err(error.clone());
        }
        if events.len() != contexts.len() {
            return Err(ExecutionEventSinkError::contract(
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

        let all_enabled = events.iter().all(|event| self.records_event(event));
        if all_enabled {
            let permit = EventBatchEmissionPermit {
                events,
                _seal: event_sink_seal::Seal,
            };
            let delivered = self.sink.as_sink().record_batch(permit);
            self.accept_delivery(delivered)?;
        } else {
            for event in events {
                if !self.records_event(&event) {
                    continue;
                }
                let permit = EventEmissionPermit {
                    event,
                    _seal: event_sink_seal::Seal,
                };
                let delivered = self.sink.as_sink().record(permit);
                self.accept_delivery(delivered)?;
            }
        }
        self.cursor = next_cursor;
        Ok(())
    }

    pub fn cursor(&self) -> &ExecutionEventCursor {
        &self.cursor
    }

    pub const fn sink_failed(&self) -> bool {
        self.sink_failure.is_some()
    }

    /// First observed transport loss, including optional delivery failures.
    /// Cursor progress alone never certifies complete delivered evidence.
    pub fn transport_failure(&self) -> Option<&ExecutionEventSinkError> {
        self.transport_failure.as_ref()
    }
}

#[derive(Debug, Default)]
pub struct DisabledExecutionEventSink;

impl ExecutionEventSink for DisabledExecutionEventSink {
    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::None
    }

    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        false
    }

    fn record(&self, _permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }
}
