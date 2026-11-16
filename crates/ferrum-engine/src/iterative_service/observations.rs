//! Immutable lifecycle observations from the original worker. The sink owns
//! its storage and loss accounting; observations cannot authorize execution.
use ferrum_types::RequestCorrelation;
use std::{sync::Arc, time::Instant};
use uuid::Uuid;

#[derive(Clone, Copy, Debug)]
pub enum RequestOutcome {
    Completed,
    Cancelled,
    Rejected,
    Failed,
    Shutdown,
}

#[derive(Clone, Copy, Debug)]
pub enum PhaseOutcome {
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug)]
pub enum LifecyclePhase {
    MetadataPreparation,
    Initialization,
    InitializationAdvance,
}

#[derive(Clone, Copy, Debug)]
pub enum LifecycleEvent {
    SubmissionRejected {
        correlation: RequestCorrelation,
        outcome: RequestOutcome,
    },
    Queued {
        correlation: RequestCorrelation,
    },
    Dequeued {
        request_id: Uuid,
    },
    Requeued {
        request_id: Uuid,
    },
    InitializationStarted {
        request_id: Uuid,
    },
    PhaseStarted {
        request_id: Uuid,
        phase: LifecyclePhase,
    },
    PhaseFinished {
        start_sequence: u64,
        outcome: PhaseOutcome,
    },
    Finished {
        request_id: Uuid,
        outcome: RequestOutcome,
    },
}

/// A bounded, observation-only consumer. Implementations preserve their own
/// dropped/incomplete evidence and must not retain execution owners, panic or
/// change inference outcomes. `Some` identifies a completely recorded event;
/// `None` never authorizes a retry or fabricates a successful span.
pub trait LifecycleObserver: Send + Sync + 'static {
    fn record_at(&self, at: Instant, event: LifecycleEvent) -> Option<u64>;
}

pub type LifecycleObserverHandle = Arc<dyn LifecycleObserver>;
