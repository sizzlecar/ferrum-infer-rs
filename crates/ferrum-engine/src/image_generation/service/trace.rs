//! Observations of the existing queue owners. No scheduling state, second
//! clock, native owner or request content is retained here.
use super::*;
pub(super) use crate::image_generation::continuous_trace::ImageTraceRequestOutcome;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceEvent as Event, ImageTracePhase, ImageTracePhaseOutcome,
};

pub(super) fn outcome(kind: ImageServiceErrorKind) -> ImageTraceRequestOutcome {
    match kind {
        ImageServiceErrorKind::Cancelled => ImageTraceRequestOutcome::Cancelled,
        ImageServiceErrorKind::Shutdown => ImageTraceRequestOutcome::Shutdown,
        ImageServiceErrorKind::InvalidRequest | ImageServiceErrorKind::QueueFull => {
            ImageTraceRequestOutcome::Rejected
        }
        ImageServiceErrorKind::Stopped
        | ImageServiceErrorKind::Generation
        | ImageServiceErrorKind::WorkerPanic => ImageTraceRequestOutcome::Failed,
    }
}

pub(super) fn submission_rejected(
    handle: Option<&ImageContinuousTraceHandle>,
    correlation: ImageRequestCorrelation,
    kind: ImageServiceErrorKind,
) {
    if let Some(handle) = handle {
        handle.record(&Event::RequestSubmissionRejected {
            correlation,
            outcome: outcome(kind),
        });
    }
}

pub(super) struct RequestTrace {
    pub correlation: ImageRequestCorrelation,
    handle: Option<ImageContinuousTraceHandle>,
}
impl RequestTrace {
    pub fn new(
        correlation: ImageRequestCorrelation,
        handle: Option<ImageContinuousTraceHandle>,
    ) -> Self {
        Self {
            correlation,
            handle,
        }
    }
    #[cfg(test)]
    pub fn unobserved() -> Self {
        Self::new(ImageRequestCorrelation::new(None), None)
    }
    pub fn queued(&self, at: Instant) {
        if let Some(handle) = &self.handle {
            handle.record_at(
                at,
                &Event::RequestQueued {
                    correlation: self.correlation,
                },
            );
        }
    }
    pub fn dequeued(&self) {
        if let Some(handle) = &self.handle {
            handle.record(&Event::RequestDequeued {
                request_id: self.correlation.request_id,
            });
        }
    }
    pub fn requeued(&self) {
        if let Some(handle) = &self.handle {
            handle.record(&Event::RequestRequeued {
                request_id: self.correlation.request_id,
            });
        }
    }
    pub fn initialization_started(&self) {
        if let Some(handle) = &self.handle {
            handle.record(&Event::InitializationStarted {
                request_id: self.correlation.request_id,
            });
        }
    }
    pub fn phase_started(&self, phase: ImageTracePhase) -> Option<u64> {
        self.handle.as_ref().and_then(|handle| {
            handle.record(&Event::PhaseStarted {
                request_id: Some(self.correlation.request_id),
                request: None,
                phase,
            })
        })
    }
    pub fn phase_finished(&self, start: Option<u64>, outcome: ImageTracePhaseOutcome) {
        if let (Some(handle), Some(start_sequence)) = (&self.handle, start) {
            handle.record(&Event::PhaseFinished {
                start_sequence,
                outcome,
            });
        }
    }
    pub fn finished(&self, outcome: ImageTraceRequestOutcome) {
        if let Some(handle) = &self.handle {
            handle.record(&Event::RequestServiceFinished {
                request_id: self.correlation.request_id,
                outcome,
            });
        }
    }
    pub fn error(&self, kind: ImageServiceErrorKind, message: &str) -> ImageServiceError {
        self.finished(outcome(kind));
        let mut error = service_error(kind, message);
        error.correlation = Some(self.correlation);
        error
    }
}
