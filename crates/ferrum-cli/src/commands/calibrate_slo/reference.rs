//! Optional singleton reference phases. A target-complete observer never
//! cancels its request: the driver must still drain the full requested output.
//! Discovery determines exact identities, then a persisted plan precedes fresh
//! trials. Neither cost predictions nor a mixed-wave wall split are samples.
use super::{inputs, manifest};
use ferrum_engine::continuous_engine::{
    CalibrationFrontier, CalibrationProfileArtifact, CalibrationReferenceArtifact,
    CalibrationReferenceCollector, CalibrationReferenceCurve, CalibrationReferenceDiscoverySample,
    CalibrationReferencePlan, CalibrationReferenceTrial, CalibrationRequestEvidence,
    CalibrationSession, CalibrationSubmissionState, CalibrationWaveReport, CalibrationWork,
};
use ferrum_interfaces::execution_cost::ActualRowWork;
use ferrum_scheduler::implementations::continuous::prefill_reference::{
    PiecewiseReferenceSpec, ReferenceEstimator, ReferenceProtocolV1,
};
use ferrum_types::{FerrumError, Result, SloPrefillReferenceLimits};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
};

mod config;
pub(in crate::commands::calibrate_slo) use config::prefill_count;
mod discovery;
mod files;
mod observer;
mod phases;
mod request_policy;
#[cfg(test)]
mod tests;

pub(super) use discovery::{DiscoverySet, FrozenReference, ReferenceReceipt};
pub(super) use observer::CohortObserver;
pub(super) use observer::{DiscoveryObserver, TrialObserver};
pub(super) use phases::collect;
pub(super) use request_policy::{InputIdentityLedger, ReferenceRequestPolicy};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReferenceConfig {
    /// Independent reference requests only; training/heldout always keep the source policy.
    #[serde(default)]
    pub request_policy: ReferenceRequestPolicy,
    pub revision: NonZeroU64,
    pub artifact_path: PathBuf,
    pub frozen_plan_path: PathBuf,
    /// Required even when empty. Empty declares no warmup, not an inferred
    /// warm condition. Warmup observations cannot become reference trials.
    pub warmup: Vec<manifest::Cohort>,
    /// Ordered singleton input indices. Actual equal token lengths are an
    /// explicit V1 incompatibility, never silently deduplicated.
    pub curve_prompt_indices: Vec<usize>,
    pub granule_tokens: NonZeroU32,
    /// Explicit V2 scoring domain and real absolute-prefix partition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub piecewise: Option<PiecewiseReferenceSpec>,
    pub repetitions: NonZeroUsize,
    pub decode_unit: DecodeUnit,
    #[serde(default)]
    pub limits: SloPrefillReferenceLimits,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DecodeUnit {
    pub prompt_index: usize,
    /// Select this actual decode's input generation, not the nth visible event.
    pub generated_before: NonZeroU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum DiscoveryTarget {
    Prefill { curve: usize },
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ObservationProgress {
    NotSubmitted,
    PreparingTarget,
    Recorded,
    TargetComplete,
    /// Request execution and output draining must continue unchanged.
    AfterTarget,
}

fn invalid(message: impl Into<String>) -> FerrumError {
    FerrumError::config(format!("calibration reference: {}", message.into()))
}
