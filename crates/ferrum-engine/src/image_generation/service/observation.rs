//! Preserve the image JSONL schema while consuming the shared lifecycle facts.
//! The original charged collector and its loss accounting remain the authority.
use super::ImageContinuousTraceHandle;
use crate::{
    image_generation::continuous_trace::{
        ImageContinuousTraceEvent as Event, ImageTracePhase, ImageTracePhaseOutcome,
        ImageTraceRequestOutcome,
    },
    iterative_service::observations::{
        LifecycleEvent, LifecycleObserver, LifecyclePhase, PhaseOutcome, RequestOutcome,
    },
};
use std::time::Instant;

fn request_outcome(outcome: RequestOutcome) -> ImageTraceRequestOutcome {
    match outcome {
        RequestOutcome::Completed => ImageTraceRequestOutcome::Completed,
        RequestOutcome::Cancelled => ImageTraceRequestOutcome::Cancelled,
        RequestOutcome::Rejected => ImageTraceRequestOutcome::Rejected,
        RequestOutcome::Failed => ImageTraceRequestOutcome::Failed,
        RequestOutcome::Shutdown => ImageTraceRequestOutcome::Shutdown,
    }
}

impl LifecycleObserver for ImageContinuousTraceHandle {
    fn record_at(&self, at: Instant, event: LifecycleEvent) -> Option<u64> {
        let event = match event {
            LifecycleEvent::SubmissionRejected {
                correlation,
                outcome,
            } => Event::RequestSubmissionRejected {
                correlation,
                outcome: request_outcome(outcome),
            },
            LifecycleEvent::Queued { correlation } => Event::RequestQueued { correlation },
            LifecycleEvent::Dequeued { request_id } => Event::RequestDequeued { request_id },
            LifecycleEvent::Requeued { request_id } => Event::RequestRequeued { request_id },
            LifecycleEvent::InitializationStarted { request_id } => {
                Event::InitializationStarted { request_id }
            }
            LifecycleEvent::PhaseStarted { request_id, phase } => Event::PhaseStarted {
                request_id: Some(request_id),
                request: None,
                phase: match phase {
                    LifecyclePhase::MetadataPreparation => ImageTracePhase::MetadataPreparation,
                    LifecyclePhase::Initialization => ImageTracePhase::Initialization,
                    LifecyclePhase::InitializationAdvance => ImageTracePhase::InitializationAdvance,
                },
            },
            LifecycleEvent::PhaseFinished {
                start_sequence,
                outcome,
            } => Event::PhaseFinished {
                start_sequence,
                outcome: match outcome {
                    PhaseOutcome::Completed => ImageTracePhaseOutcome::Completed,
                    PhaseOutcome::Failed => ImageTracePhaseOutcome::Failed,
                    PhaseOutcome::Cancelled => ImageTracePhaseOutcome::Cancelled,
                },
            },
            LifecycleEvent::Finished {
                request_id,
                outcome,
            } => Event::RequestServiceFinished {
                request_id,
                outcome: request_outcome(outcome),
            },
        };
        ImageContinuousTraceHandle::record_at(self, at, &event)
    }
}
