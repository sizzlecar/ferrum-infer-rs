//! V2 numerical primitives. No serving model is qualified by these DTOs alone.
//! Actual input and planning uncertainty are separate; V1 behavior is unchanged.
use super::{CostBoundary, ExecutionFingerprint, WaveObservationOutcome};
use ferrum_interfaces::execution_cost::{
    CanonicalWaveCostShape, CoreReadbackRoute, HostContentForecastV2, HostPendingConstraintV2,
    StatisticalWaveEvidenceV1, StructuredHostRowV1, UnsettledStructuredWaveEvidenceV1,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub use super::structured::{
    StructuredSettingsV1 as StructuredSettingsV2, StructuredUnknown as StructuredUnknownV2,
};
use StructuredUnknownV2 as StructuredUnknown;
type Result<T> = std::result::Result<T, StructuredUnknownV2>;
pub const MODEL_REVISION_V2: &str = "structured_whole_wave_pending_envelope_v2";
mod types;
pub use types::*;
mod input;
pub use input::{StructuredInputV2, StructuredQueryV2};
pub mod population;
pub use population::StructuredOwnerFactsV2;
mod envelope;
mod fit;
mod support;
pub mod windows;
pub type StructuredNumericObservationV2 = StructuredObservationV2<StructuredInputV2>;
#[cfg(test)]
mod tests;
