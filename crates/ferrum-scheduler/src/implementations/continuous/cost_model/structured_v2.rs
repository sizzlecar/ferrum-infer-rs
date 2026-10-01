//! V2 numerical primitives. No serving model is qualified by these DTOs alone.
//! Actual input and planning uncertainty are separate; V1 behavior is unchanged.
use super::{CostBoundary, ExecutionFingerprint, WaveObservationOutcome};
use ferrum_interfaces::execution_cost::{
    CanonicalWaveCostShape, CoreReadbackRoute, HostContentForecastV2, HostPendingConstraintV2,
    StatisticalWaveEvidenceV1, StructuredHostRowV1, UnsettledStructuredWaveEvidenceV1,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub use super::structured::StructuredUnknown as StructuredUnknownV2;
mod query_outcome;
use query_outcome::QueryResult;
pub use query_outcome::StructuredQueryFailureV2;
mod settings;
pub use settings::{StructuredLearnedDriftV2, StructuredSettingsV2};
use StructuredUnknownV2 as StructuredUnknown;
type Result<T> = std::result::Result<T, StructuredUnknownV2>;
pub const MODEL_REVISION_V2: &str = "structured_whole_wave_pending_envelope_v2_fit_floor_v1";
mod types;
pub use types::*;
mod algorithm_universe;
mod completion;
mod input;
mod numerical_family;
pub use algorithm_universe::{DeclaredAlgorithmUniverseBuilderV1, DeclaredAlgorithmUniverseV1};
mod repetition;
pub use input::{StructuredInputV2, StructuredQueryV2};
pub use numerical_family::{NumericalFamilyInputV1, NumericalFamilyKeyV1, NumericalFamilyV1};
pub mod population;
pub use population::StructuredOwnerFactsV2;
mod coverage;
mod demand;
pub use demand::StructuredQueryDemandV2;
mod envelope;
mod fit;
pub use fit::input_pivots::{
    input_geometry_pivot_scratch_bytes_v1, input_geometry_pivots_v1, StructuredInputGeometryWorkV1,
    StructuredInputPivotsV1,
};
mod nonnegative;
pub use nonnegative::{EnvelopeSettings, FitCertificate as NonNegativeFitCertificateV1};
mod phase;
mod support;
pub use coverage::{StructuredCoverageFactsV2, StructuredCoverageReportV2};
mod model;
pub use model::{
    CalibratedStructuredModelV2, FittedStructuredModelV2, NonNegativeEnvelopeContractV1,
    NonNegativePlanningEstimatorV1, OwnerAlgorithmUniversePolicyV1, OwnerBlockScheduleV1,
    OwnerInputReadinessDecisionV1, OwnerInputReadinessGapV1, OwnerInputReadinessV1,
    OwnerInputTargetV1, OwnerOpeningFrontierPolicyV1, OwnerPhaseSupportPolicyV1,
    OwnerPredictionValidityPolicyV1, QualifiedStructuredModelV2, StructuredFitSupportDiagnosticV1,
    StructuredFitSupportReasonV1, StructuredJointCellKeyV1, StructuredOwnerPhaseBoundaryV1,
    StructuredOwnerPhaseCloseV1, StructuredOwnerPhaseContractV1, StructuredPopulationPolicyV1,
    StructuredServiceDomainPolicyV1, StructuredServiceInputMembershipV1,
    StructuredServiceWindowCloseV2, StructuredServiceWindowContractV2,
    WorkAxisAndBranchChallengesV1,
};
pub mod prefixes;
pub mod windows;
pub type StructuredNumericObservationV2 = StructuredObservationV2<StructuredInputV2>;
#[cfg(test)]
mod tests;
