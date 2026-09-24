//! Independent profile protocol for selected-algorithm whole-wave statistics.
//! The legacy 1--5 parser is not extended or relabeled. Only observations are
//! serialized; loading refits on fit records, then calibrates residual records.
use super::super::cost_model::statistical::model::*;
use super::*;
use ferrum_interfaces::execution_cost::{
    ActualRowWork, ActualWaveGraphState, ActualWaveKind, ActualWavePath, ActualWaveRowOrder,
    CanonicalWaveCostFeatures, CanonicalWaveCostShape, HostContentCostFeaturesV1,
    HostRowMultisetCostFeaturesV2, StatisticalWaveEvidenceV1, StatisticalWaveEvidenceWireV1,
};
mod wire;
pub use wire::*;
mod loader;
pub use loader::*;
#[cfg(test)]
mod tests;
pub const COST_PROFILE_SCHEMA_VERSION_V6: u32 = 6;

#[cfg(test)]
mod live_phase_tests;
