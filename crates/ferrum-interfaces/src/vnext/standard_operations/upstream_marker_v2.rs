//! Explicit upstream projection arithmetic inside unchanged strict operations.
//! The route sets are part of this numerical version, not backend heuristics.

use super::*;
use crate::vnext::{
    CompositeNumericalArithmetic, ProjectionArithmeticOverride, ProjectionBlockFormat,
    ProjectionRole, ProjectionShapeEligibility, QuantizedProjectionLeafContract,
    StrictNumericalOperation, StrictProjectionFallback, UpstreamPrefillRows,
    UpstreamProjectionArithmetic, UpstreamProjectionLayout, UpstreamProjectionPolicy,
    UpstreamProjectionRouteDeclaration, COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM,
};

pub const DENSE_SWIGLU_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_OPERATION_ID: &str =
    "operation.dense_swiglu.q4k-q5k-iq4xs-upstream-marker-v2";
pub const DENSE_SWIGLU_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.q4k-q5k-iq4xs-upstream-marker-v2";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_OPERATION_ID: &str =
    "operation.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_CAPABILITY_ID: &str =
    "capability.operation.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_OPERATION_ID: &str =
    "operation.causal_paged_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_CAPABILITY_ID: &str =
    "capability.operation.causal_paged_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2";

pub const DENSE_SWIGLU_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID: &str =
    "operation.dense_swiglu.q4k-q5k-iq4xs-upstream-marker-v2-prefill";
pub const DENSE_SWIGLU_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.q4k-q5k-iq4xs-upstream-marker-v2-prefill";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID: &str = "operation.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2-prefill";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID: &str = "capability.operation.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2-prefill";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID: &str = "operation.causal_paged_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2-prefill";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID: &str = "capability.operation.causal_paged_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2-prefill";

pub const DENSE_SWIGLU_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_OPERATION_ID: &str =
    "operation.dense_swiglu.q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1";
pub const DENSE_SWIGLU_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1";

pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_OPERATION_ID: &str = "operation.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_CAPABILITY_ID: &str = "capability.operation.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1";

pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_OPERATION_ID:
    &str = "operation.causal_paged_attention.f32-master.q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_CAPABILITY_ID: &str = "capability.operation.causal_paged_attention.f32-master.q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1";

pub const DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID: &str =
    "operation.dense_swiglu.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill";
pub const DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID: &str = "operation.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID: &str = "capability.operation.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID: &str = "operation.causal_paged_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID: &str = "capability.operation.causal_paged_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill";

pub const DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_OPERATION_ID: &str =
    "operation.dense_swiglu.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill";
pub const DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_OPERATION_ID: &str = "operation.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_CAPABILITY_ID: &str = "capability.operation.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_OPERATION_ID: &str = "operation.causal_paged_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_CAPABILITY_ID: &str = "capability.operation.causal_paged_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill";

pub const DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_OPERATION_ID: &str =
    "operation.dense_swiglu.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows";
pub const DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_OPERATION_ID: &str = "operation.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_CAPABILITY_ID: &str = "capability.operation.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_OPERATION_ID: &str = "operation.causal_paged_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_CAPABILITY_ID: &str = "capability.operation.causal_paged_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows";

pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_M8_GEOMETRY_V1_OPERATION_ID: &str = "operation.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-m8-geometry-v1";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_M8_GEOMETRY_V1_CAPABILITY_ID: &str = "capability.operation.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-m8-geometry-v1";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_M8_GEOMETRY_V1_OPERATION_ID: &str = "operation.causal_paged_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-m8-geometry-v1";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_M8_GEOMETRY_V1_CAPABILITY_ID: &str = "capability.operation.causal_paged_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-m8-geometry-v1";

#[path = "upstream_marker_v2/extra.rs"]
mod extra;
#[path = "upstream_marker_v2/geometry.rs"]
mod geometry;

/// A declaration for explicit Require. Exposing the standard contract is not
/// provider qualification; the native artifact, marker protocol and GPU gates
/// must be supplied independently. Auto must not infer eligibility from this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamMarkerV2Profile {
    SwiGlu,
    GatedDelta,
    Causal,
    SwiGluPrefill,
    SwiGluG32MmqPrefill,
    GatedDeltaPrefill,
    GatedDeltaG32MmqPrefill,
    CausalPrefill,
    CausalG32MmqPrefill,
    SwiGluExtraPrefill,
    GatedDeltaExtraPrefill,
    CausalExtraPrefill,
    SwiGluExtraLargePrefill,
    GatedDeltaExtraLargePrefill,
    CausalExtraLargePrefill,
    SwiGluExtraAllRows,
    GatedDeltaExtraAllRows,
    CausalExtraAllRows,
    GatedDeltaM8Geometry,
    CausalM8Geometry,
}

impl UpstreamMarkerV2Profile {
    /// Separate exact profile: fill undeclared extra local widths with MMQ.
    pub const fn extra_all_rows(self) -> bool {
        matches!(
            self,
            Self::SwiGluExtraAllRows
                | Self::GatedDeltaExtraAllRows
                | Self::GatedDeltaM8Geometry
                | Self::CausalExtraAllRows
                | Self::CausalM8Geometry
        )
    }

    /// Separate capability for extra-format MarkerV2 MMQ rows 33..2048.
    pub const fn extra_prefill(self) -> bool {
        matches!(
            self,
            Self::SwiGluExtraLargePrefill
                | Self::SwiGluExtraAllRows
                | Self::GatedDeltaExtraLargePrefill
                | Self::GatedDeltaExtraAllRows
                | Self::GatedDeltaM8Geometry
                | Self::CausalExtraLargePrefill
                | Self::CausalExtraAllRows
                | Self::CausalM8Geometry
        )
    }
    pub const fn extra(self) -> bool {
        self.extra_prefill()
            || matches!(
                self,
                Self::SwiGluExtraPrefill | Self::GatedDeltaExtraPrefill | Self::CausalExtraPrefill
            )
    }
    pub const fn hybrid(self) -> bool {
        matches!(
            self,
            Self::SwiGluG32MmqPrefill | Self::GatedDeltaG32MmqPrefill | Self::CausalG32MmqPrefill
        )
    }
    pub const fn uses_g32(self, rows: u32) -> bool {
        self.hybrid() && matches!(rows, 1..=32)
    }
    pub const fn prefill(self) -> bool {
        matches!(
            self,
            Self::SwiGluPrefill
                | Self::GatedDeltaPrefill
                | Self::CausalPrefill
                | Self::SwiGluExtraLargePrefill
                | Self::SwiGluExtraAllRows
                | Self::SwiGluExtraPrefill
                | Self::GatedDeltaExtraLargePrefill
                | Self::GatedDeltaExtraAllRows
                | Self::GatedDeltaM8Geometry
                | Self::GatedDeltaExtraPrefill
                | Self::CausalExtraLargePrefill
                | Self::CausalExtraAllRows
                | Self::CausalM8Geometry
                | Self::CausalExtraPrefill
        )
    }
    pub fn provider_id(self) -> String {
        self.operation_id()
            .replacen("operation.", "provider.cuda.", 1)
    }
    pub fn estimator_id(self) -> String {
        self.operation_id()
            .replacen("operation.", "resource-estimator.cuda.", 1)
    }

    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::SwiGlu => DENSE_SWIGLU_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_OPERATION_ID,
            Self::SwiGluPrefill => DENSE_SWIGLU_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID,
            Self::SwiGluExtraPrefill => DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID,
            Self::SwiGluExtraLargePrefill => DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_OPERATION_ID,
            Self::SwiGluExtraAllRows => DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_OPERATION_ID,
            Self::SwiGluG32MmqPrefill => DENSE_SWIGLU_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_OPERATION_ID,
            Self::GatedDelta => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_OPERATION_ID,
            Self::GatedDeltaPrefill => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID,
            Self::GatedDeltaExtraPrefill => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID,
            Self::GatedDeltaExtraLargePrefill => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_OPERATION_ID,
            Self::GatedDeltaExtraAllRows => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_OPERATION_ID,
            Self::GatedDeltaM8Geometry => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_M8_GEOMETRY_V1_OPERATION_ID,
            Self::GatedDeltaG32MmqPrefill => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_OPERATION_ID,
            Self::Causal => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_OPERATION_ID,
            Self::CausalPrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID,
            Self::CausalExtraPrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_OPERATION_ID,
            Self::CausalExtraLargePrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_OPERATION_ID,
            Self::CausalExtraAllRows => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_OPERATION_ID,
            Self::CausalM8Geometry => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_M8_GEOMETRY_V1_OPERATION_ID,
            Self::CausalG32MmqPrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::SwiGlu => DENSE_SWIGLU_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_CAPABILITY_ID,
            Self::SwiGluPrefill => DENSE_SWIGLU_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID,
            Self::SwiGluExtraPrefill => DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID,
            Self::SwiGluExtraLargePrefill => DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_CAPABILITY_ID,
            Self::SwiGluExtraAllRows => DENSE_SWIGLU_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_CAPABILITY_ID,
            Self::SwiGluG32MmqPrefill => DENSE_SWIGLU_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_CAPABILITY_ID,
            Self::GatedDelta => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_CAPABILITY_ID,
            Self::GatedDeltaPrefill => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID,
            Self::GatedDeltaExtraPrefill => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID,
            Self::GatedDeltaExtraLargePrefill => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_CAPABILITY_ID,
            Self::GatedDeltaExtraAllRows => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_CAPABILITY_ID,
            Self::GatedDeltaM8Geometry => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_M8_GEOMETRY_V1_CAPABILITY_ID,
            Self::GatedDeltaG32MmqPrefill => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_CAPABILITY_ID,
            Self::Causal => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_CAPABILITY_ID,
            Self::CausalPrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID,
            Self::CausalExtraPrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_CAPABILITY_ID,
            Self::CausalExtraLargePrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_CAPABILITY_ID,
            Self::CausalExtraAllRows => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_CAPABILITY_ID,
            Self::CausalM8Geometry => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_M8_GEOMETRY_V1_CAPABILITY_ID,
            Self::CausalG32MmqPrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_CAPABILITY_ID,
        }
    }
    pub const fn strict_operation_id(self) -> &'static str {
        match self {
            Self::SwiGlu
            | Self::SwiGluPrefill
            | Self::SwiGluG32MmqPrefill
            | Self::SwiGluExtraLargePrefill
            | Self::SwiGluExtraAllRows
            | Self::SwiGluExtraPrefill => DENSE_SWIGLU_OPERATION_ID,
            Self::GatedDelta
            | Self::GatedDeltaPrefill
            | Self::GatedDeltaG32MmqPrefill
            | Self::GatedDeltaExtraLargePrefill
            | Self::GatedDeltaExtraAllRows
            | Self::GatedDeltaM8Geometry
            | Self::GatedDeltaExtraPrefill => {
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID
            }
            Self::Causal
            | Self::CausalPrefill
            | Self::CausalG32MmqPrefill
            | Self::CausalExtraLargePrefill
            | Self::CausalExtraAllRows
            | Self::CausalM8Geometry
            | Self::CausalExtraPrefill => CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
        }
    }
    fn strict_contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::SwiGlu
            | Self::SwiGluPrefill
            | Self::SwiGluG32MmqPrefill
            | Self::SwiGluExtraLargePrefill
            | Self::SwiGluExtraAllRows
            | Self::SwiGluExtraPrefill => dense_swiglu_contract(),
            Self::GatedDelta
            | Self::GatedDeltaPrefill
            | Self::GatedDeltaG32MmqPrefill
            | Self::GatedDeltaExtraLargePrefill
            | Self::GatedDeltaExtraAllRows
            | Self::GatedDeltaM8Geometry
            | Self::GatedDeltaExtraPrefill => gated_delta_recurrent_attention_f32_master_contract(),
            Self::Causal
            | Self::CausalPrefill
            | Self::CausalG32MmqPrefill
            | Self::CausalExtraLargePrefill
            | Self::CausalExtraAllRows
            | Self::CausalM8Geometry
            | Self::CausalExtraPrefill => causal_paged_attention_f32_master_contract(),
        }
    }

    /// Construct once during family/provider preparation and retain the result.
    /// Encoding must consume prepared leaf indices/plans, not reconstruct and
    /// serialize this owned declaration on every projection or decode wave.
    pub fn arithmetic(self) -> CompositeNumericalArithmetic {
        if matches!(self, Self::GatedDeltaM8Geometry | Self::CausalM8Geometry) {
            return geometry::arithmetic(self);
        }
        if self.extra() {
            return extra::arithmetic(self);
        }
        let base = self
            .strict_contract()
            .expect("fixed standard strict contract");
        let roles: &[(ProjectionRole, u32)] = match self {
            Self::SwiGlu
            | Self::SwiGluPrefill
            | Self::SwiGluG32MmqPrefill
            | Self::SwiGluExtraLargePrefill
            | Self::SwiGluExtraAllRows
            | Self::SwiGluExtraPrefill => &[
                (ProjectionRole::SwiGluGateUp, 1),
                (ProjectionRole::SwiGluDown, 2),
            ],
            Self::GatedDelta
            | Self::GatedDeltaPrefill
            | Self::GatedDeltaG32MmqPrefill
            | Self::GatedDeltaExtraLargePrefill
            | Self::GatedDeltaExtraAllRows
            | Self::GatedDeltaM8Geometry
            | Self::GatedDeltaExtraPrefill => &[
                (ProjectionRole::GatedDeltaInput, 2),
                (ProjectionRole::GatedDeltaOutput, 7),
            ],
            Self::Causal
            | Self::CausalPrefill
            | Self::CausalG32MmqPrefill
            | Self::CausalExtraLargePrefill
            | Self::CausalExtraAllRows
            | Self::CausalM8Geometry
            | Self::CausalExtraPrefill => &[
                (ProjectionRole::CausalQuery, 2),
                (ProjectionRole::CausalKey, 3),
                (ProjectionRole::CausalValue, 4),
                (ProjectionRole::CausalOutput, 5),
            ],
        };
        CompositeNumericalArithmetic {
            schema_version: if self.hybrid() {
                crate::vnext::COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_G32_MMQ
            } else {
                COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM
            },
            strict_base: StrictNumericalOperation {
                operation_id: base.descriptor().id.clone(),
                version: base.descriptor().version,
            },
            projections: roles
                .iter()
                .map(
                    |&(role, weight_input_ordinal)| ProjectionArithmeticOverride {
                        role,
                        weight_input_ordinal,
                        activation_input_type: ElementType::F16,
                        activation_output_type: ElementType::F16,
                        fallback: StrictProjectionFallback::RetainBaseArithmetic {},
                        leaves: [
                            ProjectionBlockFormat::Q4K,
                            ProjectionBlockFormat::Q5K,
                            ProjectionBlockFormat::Iq4Xs,
                        ]
                        .into_iter()
                        .map(|format| {
                            if self.hybrid() {
                                return QuantizedProjectionLeafContract {
                                    format,
                                    shape: ProjectionShapeEligibility {
                                        minimum_input_features: 256,
                                        minimum_output_features: 1,
                                        input_features_multiple: 256,
                                    },
                                    arithmetic: crate::vnext::G32MmqPrefillPolicy::new(format)
                                        .expect("fixed original G32 format")
                                        .staged(),
                                };
                            }
                            let mmq = if format == ProjectionBlockFormat::Iq4Xs {
                                UpstreamProjectionArithmetic::MmqD4MarkerV2
                            } else {
                                UpstreamProjectionArithmetic::MmqDs4MarkerV2
                            };
                            // Explicit role/format arithmetic policy. M8 FFN
                            // gate/up and output projections use MMQ;
                            // attention keeps its independently declared MMVQ
                            // route. This is not a client-concurrency heuristic.
                            let mmq_eight = matches!(
                                role,
                                ProjectionRole::SwiGluGateUp | ProjectionRole::SwiGluDown
                            );
                            let mut mmvq_rows = BTreeSet::from([1, 4]);
                            // The dense MMQ ABI supports every positive M<=32.
                            // Keep the selected common-width routes below while
                            // giving partial batches an explicit native route.
                            // Larger prefill waves retain the strict base.
                            let mut mmq_rows = (1..=32)
                                .filter(|rows| !matches!(rows, 1 | 4 | 8))
                                .collect::<BTreeSet<_>>();
                            if mmq_eight {
                                mmq_rows.insert(8);
                            } else {
                                mmvq_rows.insert(8);
                            }
                            let mut routes = vec![
                                UpstreamProjectionRouteDeclaration {
                                    arithmetic: UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
                                    layout: UpstreamProjectionLayout::Columns,
                                    local_rows: mmvq_rows,
                                    prefill_rows: None,
                                },
                                UpstreamProjectionRouteDeclaration {
                                    arithmetic: mmq,
                                    layout: UpstreamProjectionLayout::Columns,
                                    local_rows: mmq_rows,
                                    prefill_rows: None,
                                },
                            ];
                            if self.prefill() {
                                routes.push(UpstreamProjectionRouteDeclaration {
                                    arithmetic: mmq,
                                    layout: UpstreamProjectionLayout::Columns,
                                    local_rows: BTreeSet::new(),
                                    prefill_rows: Some(UpstreamPrefillRows {
                                        first: 33,
                                        last: 2048,
                                    }),
                                });
                            }
                            QuantizedProjectionLeafContract {
                                format,
                                shape: ProjectionShapeEligibility {
                                    minimum_input_features: 256,
                                    minimum_output_features: 1,
                                    input_features_multiple: 256,
                                },
                                arithmetic: UpstreamProjectionPolicy {
                                    geometry_selection: None,
                                    format,
                                    fallback: StrictProjectionFallback::RetainBaseArithmetic {},
                                    routes,
                                }
                                .staged()
                                .expect("fixed upstream marker policy"),
                            }
                        })
                        .collect(),
                    },
                )
                .collect(),
        }
    }

    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        let mut contract = self.strict_contract()?;
        contract.descriptor.id = OperationId::new(self.operation_id())?;
        contract.descriptor.provider =
            provider_requirement(self.capability_id(), ContractVersion::new(1, 0))?;
        // These new operation IDs own two admitted validation flags per leaf
        // (MMQ and MMVQ coefficient semantics), even when a particular wave
        // takes the declared strict fallback. The strict base stays unchanged.
        contract.descriptor.resources.persistent = ResourcePresenceRequirement::Required;
        contract.descriptor.validate()?;
        Ok(contract)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::NumericalArithmeticStage;

    #[test]
    fn upstream_marker_v2_partial_rows_have_one_native_route_with_bounded_domain() {
        use UpstreamProjectionArithmetic::{MmqD4MarkerV2, MmqDs4MarkerV2, MmvqQ8_1MarkerV2};
        for profile in [
            UpstreamMarkerV2Profile::SwiGlu,
            UpstreamMarkerV2Profile::GatedDelta,
            UpstreamMarkerV2Profile::Causal,
        ] {
            let contract = profile.arithmetic();
            contract.validate().unwrap();
            for projection in &contract.projections {
                for leaf in &projection.leaves {
                    let [NumericalArithmeticStage::UpstreamProjection { policy }] =
                        leaf.arithmetic.stages.as_slice()
                    else {
                        panic!("upstream policy");
                    };
                    for rows in 1..=32 {
                        let routes: Vec<_> = policy
                            .routes
                            .iter()
                            .filter(|route| route.local_rows.contains(&rows))
                            .collect();
                        assert_eq!(routes.len(), 1, "one declared route for local M={rows}");
                        let mmvq = matches!(rows, 1 | 4)
                            || (rows == 8
                                && !matches!(
                                    projection.role,
                                    ProjectionRole::SwiGluGateUp | ProjectionRole::SwiGluDown
                                ));
                        let expected = if mmvq {
                            MmvqQ8_1MarkerV2
                        } else if leaf.format == ProjectionBlockFormat::Iq4Xs {
                            MmqD4MarkerV2
                        } else {
                            MmqDs4MarkerV2
                        };
                        assert_eq!(routes[0].arithmetic, expected);
                        assert_eq!(routes[0].layout, UpstreamProjectionLayout::Columns);
                    }
                    assert!(policy
                        .routes
                        .iter()
                        .all(|route| route.local_rows.iter().all(|rows| (1..=32).contains(rows))));
                }
            }
        }
    }

    #[test]
    fn upstream_marker_v2_contract_routes_m8_by_projection_role_and_format() {
        use ProjectionBlockFormat::{Iq4Xs, Q4K, Q5K};
        use ProjectionRole::*;
        use UpstreamProjectionArithmetic::{MmqD4MarkerV2, MmqDs4MarkerV2, MmvqQ8_1MarkerV2};
        for (profile, role, expected) in [
            (
                UpstreamMarkerV2Profile::SwiGlu,
                SwiGluGateUp,
                [MmqDs4MarkerV2, MmqDs4MarkerV2, MmqD4MarkerV2],
            ),
            (
                UpstreamMarkerV2Profile::SwiGlu,
                SwiGluDown,
                [MmqDs4MarkerV2, MmqDs4MarkerV2, MmqD4MarkerV2],
            ),
            (
                UpstreamMarkerV2Profile::GatedDelta,
                GatedDeltaInput,
                [MmvqQ8_1MarkerV2; 3],
            ),
            (
                UpstreamMarkerV2Profile::GatedDelta,
                GatedDeltaOutput,
                [MmvqQ8_1MarkerV2; 3],
            ),
            (
                UpstreamMarkerV2Profile::Causal,
                CausalQuery,
                [MmvqQ8_1MarkerV2; 3],
            ),
            (
                UpstreamMarkerV2Profile::Causal,
                CausalKey,
                [MmvqQ8_1MarkerV2; 3],
            ),
            (
                UpstreamMarkerV2Profile::Causal,
                CausalValue,
                [MmvqQ8_1MarkerV2; 3],
            ),
            (
                UpstreamMarkerV2Profile::Causal,
                CausalOutput,
                [MmvqQ8_1MarkerV2; 3],
            ),
        ] {
            let contract = profile.arithmetic();
            contract.validate().unwrap();
            let projection = contract
                .projections
                .iter()
                .find(|p| p.role == role)
                .unwrap();
            for (format, expected) in [Q4K, Q5K, Iq4Xs].into_iter().zip(expected) {
                let leaf = projection
                    .leaves
                    .iter()
                    .find(|l| l.format == format)
                    .unwrap();
                let [NumericalArithmeticStage::UpstreamProjection { policy }] =
                    leaf.arithmetic.stages.as_slice()
                else {
                    panic!("upstream policy");
                };
                let selected: Vec<_> = policy
                    .routes
                    .iter()
                    .filter(|r| r.local_rows.contains(&8))
                    .collect();
                assert_eq!(selected.len(), 1, "unambiguous role/format policy");
                assert_eq!(selected[0].arithmetic, expected);
                assert_eq!(selected[0].layout, UpstreamProjectionLayout::Columns);
                let mut ambiguous = policy.clone();
                ambiguous.routes.push(selected[0].clone());
                assert!(ambiguous.validate().is_err());
            }
        }
    }

    #[test]
    fn upstream_marker_v2_contract_adds_owned_flags_without_changing_strict_tensor_semantics() {
        for profile in [
            UpstreamMarkerV2Profile::SwiGlu,
            UpstreamMarkerV2Profile::GatedDelta,
            UpstreamMarkerV2Profile::Causal,
        ] {
            let base = profile.strict_contract().unwrap();
            let mut selected = profile.contract().unwrap().descriptor;
            assert_eq!(
                selected.resources.persistent,
                ResourcePresenceRequirement::Required
            );
            assert_eq!(
                base.descriptor.resources.persistent,
                ResourcePresenceRequirement::Forbidden
            );
            selected.id = base.descriptor.id.clone();
            selected.provider = base.descriptor.provider.clone();
            selected.resources.persistent = base.descriptor.resources.persistent;
            assert_eq!(selected, base.descriptor);
            assert_eq!(profile.arithmetic().strict_base.operation_id, selected.id);
        }
    }
}
