//! Observations of the existing queue owners. No scheduling state, second
//! clock, native owner or request content is retained here.
pub(crate) use super::observations::RequestOutcome;
use super::observations::{LifecycleEvent as Event, LifecyclePhase, PhaseOutcome};
use super::*;

pub(crate) fn outcome(kind: ServiceErrorKind) -> RequestOutcome {
    match kind {
        ServiceErrorKind::Cancelled => RequestOutcome::Cancelled,
        ServiceErrorKind::Shutdown => RequestOutcome::Shutdown,
        ServiceErrorKind::InvalidRequest | ServiceErrorKind::QueueFull => RequestOutcome::Rejected,
        ServiceErrorKind::Stopped
        | ServiceErrorKind::Generation
        | ServiceErrorKind::WorkerPanic => RequestOutcome::Failed,
    }
}

pub(crate) fn submission_rejected(
    handle: Option<&LifecycleObserverHandle>,
    correlation: RequestCorrelation,
    kind: ServiceErrorKind,
) {
    if let Some(handle) = handle {
        handle.record_at(
            Instant::now(),
            Event::SubmissionRejected {
                correlation,
                outcome: outcome(kind),
            },
        );
    }
}

pub(crate) struct RequestTrace {
    pub correlation: RequestCorrelation,
    handle: Option<LifecycleObserverHandle>,
}
impl RequestTrace {
    pub fn new(correlation: RequestCorrelation, handle: Option<LifecycleObserverHandle>) -> Self {
        Self {
            correlation,
            handle,
        }
    }
    #[cfg(test)]
    pub fn unobserved() -> Self {
        Self::new(RequestCorrelation::new(None), None)
    }
    pub fn queued(&self, at: Instant) {
        if let Some(handle) = &self.handle {
            handle.record_at(
                at,
                Event::Queued {
                    correlation: self.correlation,
                },
            );
        }
    }
    pub fn dequeued(&self) {
        if let Some(handle) = &self.handle {
            handle.record_at(
                Instant::now(),
                Event::Dequeued {
                    request_id: self.correlation.request_id,
                },
            );
        }
    }
    pub fn requeued(&self) {
        if let Some(handle) = &self.handle {
            handle.record_at(
                Instant::now(),
                Event::Requeued {
                    request_id: self.correlation.request_id,
                },
            );
        }
    }
    pub fn initialization_started(&self) {
        if let Some(handle) = &self.handle {
            handle.record_at(
                Instant::now(),
                Event::InitializationStarted {
                    request_id: self.correlation.request_id,
                },
            );
        }
    }
    pub fn phase_started(&self, phase: LifecyclePhase) -> Option<u64> {
        self.handle.as_ref().and_then(|handle| {
            handle.record_at(
                Instant::now(),
                Event::PhaseStarted {
                    request_id: self.correlation.request_id,
                    phase,
                },
            )
        })
    }
    pub fn phase_finished(&self, start: Option<u64>, outcome: PhaseOutcome) {
        if let (Some(handle), Some(start_sequence)) = (&self.handle, start) {
            handle.record_at(
                Instant::now(),
                Event::PhaseFinished {
                    start_sequence,
                    outcome,
                },
            );
        }
    }
    pub fn finished(&self, outcome: RequestOutcome) {
        if let Some(handle) = &self.handle {
            handle.record_at(
                Instant::now(),
                Event::Finished {
                    request_id: self.correlation.request_id,
                    outcome,
                },
            );
        }
    }
    pub fn error(&self, kind: ServiceErrorKind, message: &str) -> ServiceError {
        self.finished(outcome(kind));
        let mut error = service_error(kind, message);
        error.correlation = Some(self.correlation);
        error
    }
}
