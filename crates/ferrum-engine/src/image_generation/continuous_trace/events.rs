//! Bounded metadata views of the original service and execution boundaries.
use ferrum_types::ImageRequestCorrelation;
use serde::Serialize;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTraceRequestOutcome {
    Completed,
    Cancelled,
    Rejected,
    Failed,
    Shutdown,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTracePhaseOutcome {
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTracePhase {
    MetadataPreparation,
    Initialization,
    InitializationAdvance,
    PromptExpansion,
    /// Upsampler image decode/preprocess, visual install/execute/readback,
    /// acknowledged close and causal-input bridge. Prior compilation and
    /// subsequent causal initialization/generation are excluded.
    PromptExpansionVision,
    Conditioning,
    /// Main conditioning visual preprocess, install/execute/readback, text
    /// input bridge and stage retirement. Prior compilation/outer resize,
    /// reference VAE and subsequent text encoder/connector are excluded.
    ConditioningVision,
    ReferenceEncoder,
    Refiner,
    Decoder,
    Noise,
    Vision,
    Projection,
    TextEncoder,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ImageTraceStepStartMember<'a> {
    pub request: &'a str,
    pub step_index: usize,
}

/// This records observations only. A start is not proof of native submission;
/// a failed span is not a completed phase or an acknowledged retirement.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageContinuousTraceEvent<'a> {
    RequestSubmissionRejected {
        correlation: ImageRequestCorrelation,
        outcome: ImageTraceRequestOutcome,
    },
    RequestQueued {
        correlation: ImageRequestCorrelation,
    },
    RequestDequeued {
        request_id: Uuid,
    },
    RequestRequeued {
        request_id: Uuid,
    },
    InitializationStarted {
        request_id: Uuid,
    },
    RequestLinked {
        request_id: Uuid,
        request: &'a str,
    },
    RequestServiceFinished {
        request_id: Uuid,
        outcome: ImageTraceRequestOutcome,
    },
    PhaseStarted {
        request_id: Option<Uuid>,
        request: Option<&'a str>,
        phase: ImageTracePhase,
    },
    PhaseFinished {
        start_sequence: u64,
        outcome: ImageTracePhaseOutcome,
    },
    PackedStepStarted {
        cohort_job: u64,
        plan_hash: &'a str,
        program_fingerprint: &'a str,
        members: &'a [ImageTraceStepStartMember<'a>],
    },
}
