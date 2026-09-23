//! Fixed scoring units built from a declared singleton calibration protocol.
//! This is not a service-time lower bound or a statistical SLO certificate.
//! Receipt fields must come from the calibration executor's actual commits;
//! strict parsing validates their consistency, not their physical authenticity.
use super::cost_model::{validate_timing, ExecutionFingerprint};
use super::cost_profile::{bounded_vec, v2::*, *};
use super::slo_planner::{
    linear_prefill_milestones, PrefillProgressView, PrefillReferenceWork, ReferenceWorkPoint,
};
use ferrum_interfaces::execution_cost::{
    project_host_cost_features, ActualRowWork, CostRowOutput, HostCostFeaturesV1,
};
use ferrum_types::{
    RequestId, SloPrefillReferenceLimits, PREFILL_REFERENCE_MAX_CURVES,
    PREFILL_REFERENCE_MAX_POINTS_PER_CURVE, PREFILL_REFERENCE_MAX_REPETITIONS,
};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
    sync::Arc,
};

mod wire;
pub use wire::*;
mod build;
pub use build::ReferenceCalibrationBuilder;
mod load;
pub use load::{load_prefill_reference, load_prefill_reference_bytes};
mod binding;
pub use binding::*;
mod piecewise;
pub use piecewise::{PiecewiseReferenceSpec, ReferenceCalibrationV2};

pub const PREFILL_REFERENCE_SCHEMA_V1: u32 = 1;
pub const PREFILL_REFERENCE_SCHEMA_V2: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceUnknown {
    LengthNotCalibrated,
    MissingEndpoint,
    NoLegalChunk,
    WrongIncarnation,
    InvalidProgress,
    FinalTokenNotCommitted,
    PointBudget,
    ArithmeticOverflow,
}

#[derive(Debug, thiserror::Error)]
pub enum ReferenceError {
    #[error("reference calibration IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("reference calibration JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("reference calibration limit: {0}")]
    Limit(&'static str),
    #[error("reference calibration incompatible identity or protocol")]
    Incompatible,
    #[error("unsupported reference calibration schema {0}")]
    Schema(u32),
    #[error("invalid reference calibration evidence: {0}")]
    Evidence(&'static str),
    #[error("reference calibration arithmetic overflow")]
    Overflow,
}

/// Full content identity, never a truncated hash masquerading as the revision.
/// Activation/replacement must compare this identity and drain prior bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceIdentity {
    pub revision: NonZeroU64,
    pub artifact_sha256: [u8; 32],
    pub protocol_sha256: [u8; 32],
}

#[derive(Debug)]
pub struct LoadedPrefillReference {
    identity: ReferenceIdentity,
    fingerprint: ExecutionFingerprint,
    protocol: ReferenceProtocolV1,
    tau_ref_ns: NonZeroU64,
    curves: BTreeMap<u32, Arc<PrefillReferenceWork>>,
    piecewise: Option<Arc<piecewise::PiecewiseDefinition>>,
    points: usize,
    samples: usize,
    generated_unix_ns: u64,
    source_path: Option<PathBuf>,
}
impl LoadedPrefillReference {
    pub fn identity(&self) -> ReferenceIdentity {
        self.identity
    }
    pub fn fingerprint(&self) -> &ExecutionFingerprint {
        &self.fingerprint
    }
    pub fn protocol(&self) -> &ReferenceProtocolV1 {
        &self.protocol
    }
    pub fn tau_ref_ns(&self) -> NonZeroU64 {
        self.tau_ref_ns
    }
    pub fn curve(&self, total: NonZeroU32) -> Result<Arc<PrefillReferenceWork>, ReferenceUnknown> {
        if let Some(definition) = &self.piecewise {
            return definition.bind(self.identity.revision.get(), total);
        }
        self.curves
            .get(&total.get())
            .cloned()
            .ok_or(ReferenceUnknown::LengthNotCalibrated)
    }
    pub fn supported_lengths(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.curves.keys().copied()
    }
    /// V2 coverage is a declared domain; supported_lengths lists measured anchors.
    pub fn piecewise_domain(&self) -> Option<(NonZeroU32, NonZeroU32)> {
        self.piecewise.as_ref().map(|value| {
            (
                value.spec.minimum_prompt_tokens,
                value.spec.maximum_prompt_tokens,
            )
        })
    }
    pub fn point_count(&self) -> usize {
        self.points
    }
    pub fn sample_count(&self) -> usize {
        self.samples
    }
    /// Original measurement artifact time; loading never refreshes it.
    pub fn generated_unix_ns(&self) -> u64 {
        self.generated_unix_ns
    }
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }
}

#[cfg(test)]
mod tests;
