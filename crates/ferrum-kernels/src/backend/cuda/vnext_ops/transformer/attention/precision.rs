//! Hidden-state precision is an operation ABI, independent of matrix encoding.
use ferrum_interfaces::vnext::{
    gated_delta_recurrent_attention_contract, gated_delta_recurrent_attention_f32_master_contract,
    ElementType, StandardOperationContract, VNextError,
    GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
};

#[derive(Clone, Copy)]
pub(super) enum AttentionPrecision {
    F16,
    F32Master,
    F32MasterGgufF16Projections,
    F32MasterQ8Act,
    F32MasterUpstream,
    F32MasterUpstreamPrefill,
    F32MasterUpstreamExtraPrefill,
    F32MasterUpstreamExtraLargePrefill,
    F32MasterUpstreamExtraAllRows,
    F32MasterUpstreamM8Geometry,
    F32MasterUpstreamQ6F16,
    F32MasterG32MmqPrefill,
}

impl AttentionPrecision {
    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::F16 => GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            Self::F32Master => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
            Self::F32MasterGgufF16Projections => ferrum_interfaces::vnext::GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
            Self::F32MasterQ8Act => ferrum_interfaces::vnext::Q8ActAttentionProfile::GatedDelta.operation_id(),
            Self::F32MasterUpstream => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDelta.operation_id(),
            Self::F32MasterUpstreamPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaPrefill.operation_id(),
            Self::F32MasterUpstreamExtraPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraPrefill.operation_id(),
            Self::F32MasterUpstreamExtraLargePrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill.operation_id(),
            Self::F32MasterUpstreamExtraAllRows => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraAllRows.operation_id(),
            Self::F32MasterUpstreamM8Geometry => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaM8Geometry.operation_id(),
            Self::F32MasterUpstreamQ6F16 => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaQ6F16.operation_id(),
            Self::F32MasterG32MmqPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill.operation_id(),
        }
    }

    pub(super) fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::F16 => gated_delta_recurrent_attention_contract(),
            Self::F32Master => gated_delta_recurrent_attention_f32_master_contract(),
            Self::F32MasterGgufF16Projections => ferrum_interfaces::vnext::gated_delta_recurrent_attention_f32_master_gguf_f16_projections_contract(),
            Self::F32MasterQ8Act => ferrum_interfaces::vnext::Q8ActAttentionProfile::GatedDelta.contract(),
            Self::F32MasterUpstream => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDelta.contract(),
            Self::F32MasterUpstreamPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaPrefill.contract(),
            Self::F32MasterUpstreamExtraPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraPrefill.contract(),
            Self::F32MasterUpstreamExtraLargePrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill.contract(),
            Self::F32MasterUpstreamExtraAllRows => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraAllRows.contract(),
            Self::F32MasterUpstreamM8Geometry => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaM8Geometry.contract(),
            Self::F32MasterUpstreamQ6F16 => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaQ6F16.contract(),
            Self::F32MasterG32MmqPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill.contract(),
        }
    }

    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::F16 => GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
            Self::F32Master => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID,
            Self::F32MasterGgufF16Projections => ferrum_interfaces::vnext::GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID,
            Self::F32MasterQ8Act => ferrum_interfaces::vnext::Q8ActAttentionProfile::GatedDelta.capability_id(),
            Self::F32MasterUpstream => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDelta.capability_id(),
            Self::F32MasterUpstreamPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaPrefill.capability_id(),
            Self::F32MasterUpstreamExtraPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraPrefill.capability_id(),
            Self::F32MasterUpstreamExtraLargePrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill.capability_id(),
            Self::F32MasterUpstreamExtraAllRows => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraAllRows.capability_id(),
            Self::F32MasterUpstreamM8Geometry => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaM8Geometry.capability_id(),
            Self::F32MasterUpstreamQ6F16 => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaQ6F16.capability_id(),
            Self::F32MasterG32MmqPrefill => ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill.capability_id(),
        }
    }

    pub(super) fn provider(self) -> &'static str {
        match self {
            Self::F16 => "provider.cuda.gated_delta_recurrent_attention.f16",
            Self::F32Master => "provider.cuda.gated_delta_recurrent_attention.f32-master",
            Self::F32MasterGgufF16Projections => {
                "provider.cuda.gated_delta_recurrent_attention.f32-master.gguf-f16-projections"
            }
            Self::F32MasterUpstream => "provider.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2",
            Self::F32MasterUpstreamPrefill => "provider.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2-prefill",
            Self::F32MasterUpstreamExtraPrefill => "provider.cuda.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill",
            Self::F32MasterUpstreamExtraLargePrefill => "provider.cuda.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill",
            Self::F32MasterUpstreamExtraAllRows => "provider.cuda.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows",
            Self::F32MasterUpstreamM8Geometry => "provider.cuda.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-m8-geometry-v1",
            Self::F32MasterUpstreamQ6F16 => "provider.cuda.gated_delta_recurrent_attention.f32-master.upstream-q6-f16-marker-v1",
            Self::F32MasterG32MmqPrefill => "provider.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1",
            Self::F32MasterQ8Act => {
                "provider.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-q8act-g32"
            }
        }
    }

    pub(super) fn estimator(self) -> &'static str {
        match self {
            Self::F16 => "resource-estimator.cuda.gated_delta_recurrent_attention.f16",
            Self::F32Master => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master",
            Self::F32MasterGgufF16Projections => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.gguf-f16-projections",
            Self::F32MasterUpstream => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2",
            Self::F32MasterUpstreamPrefill => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-upstream-marker-v2-prefill",
            Self::F32MasterUpstreamExtraPrefill => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-prefill",
            Self::F32MasterUpstreamExtraLargePrefill => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-prefill",
            Self::F32MasterUpstreamExtraAllRows => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-extra-all-rows",
            Self::F32MasterUpstreamM8Geometry => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q3k-q4k-q5k-iq3s-iq4nl-iq4xs-upstream-marker-v2-m8-geometry-v1",
            Self::F32MasterUpstreamQ6F16 => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.upstream-q6-f16-marker-v1",
            Self::F32MasterG32MmqPrefill => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-g32-mmq-prefill-marker-v1",
            Self::F32MasterQ8Act => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-q8act-g32",
        }
    }

    pub(super) fn hidden(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32Master
            | Self::F32MasterGgufF16Projections
            | Self::F32MasterQ8Act
            | Self::F32MasterUpstream
            | Self::F32MasterUpstreamPrefill
            | Self::F32MasterUpstreamExtraPrefill
            | Self::F32MasterUpstreamExtraLargePrefill
            | Self::F32MasterUpstreamExtraAllRows
            | Self::F32MasterUpstreamM8Geometry
            | Self::F32MasterUpstreamQ6F16
            | Self::F32MasterG32MmqPrefill => ElementType::F32,
        }
    }

    pub(super) fn norm(self) -> &'static str {
        match self {
            Self::F16 => "rms_norm_f16",
            Self::F32Master
            | Self::F32MasterGgufF16Projections
            | Self::F32MasterQ8Act
            | Self::F32MasterUpstream
            | Self::F32MasterUpstreamPrefill
            | Self::F32MasterUpstreamExtraPrefill
            | Self::F32MasterUpstreamExtraLargePrefill
            | Self::F32MasterUpstreamExtraAllRows
            | Self::F32MasterUpstreamM8Geometry
            | Self::F32MasterUpstreamQ6F16
            | Self::F32MasterG32MmqPrefill => "vnext_rms_norm_f32_to_f16",
        }
    }

    pub(super) fn residual(self) -> &'static str {
        match self {
            Self::F16 => "residual_add_f16",
            Self::F32Master
            | Self::F32MasterGgufF16Projections
            | Self::F32MasterQ8Act
            | Self::F32MasterUpstream
            | Self::F32MasterUpstreamPrefill
            | Self::F32MasterUpstreamExtraPrefill
            | Self::F32MasterUpstreamExtraLargePrefill
            | Self::F32MasterUpstreamExtraAllRows
            | Self::F32MasterUpstreamM8Geometry
            | Self::F32MasterUpstreamQ6F16
            | Self::F32MasterG32MmqPrefill => "vnext_residual_add_f32_f16",
        }
    }

    pub(super) fn upstream_profile(self) -> ferrum_interfaces::vnext::UpstreamMarkerV2Profile {
        if matches!(self, Self::F32MasterUpstreamQ6F16) {
            return ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaQ6F16;
        }
        if matches!(self, Self::F32MasterUpstreamM8Geometry) {
            return ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaM8Geometry;
        }
        if matches!(self, Self::F32MasterUpstreamExtraAllRows) {
            return ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraAllRows;
        }
        if matches!(self, Self::F32MasterUpstreamExtraLargePrefill) {
            return ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill;
        }
        if matches!(self, Self::F32MasterUpstreamExtraPrefill) {
            return ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaExtraPrefill;
        }
        if matches!(self, Self::F32MasterG32MmqPrefill) {
            return ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill;
        }
        if matches!(self, Self::F32MasterUpstreamPrefill) {
            ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDeltaPrefill
        } else {
            ferrum_interfaces::vnext::UpstreamMarkerV2Profile::GatedDelta
        }
    }

    pub(super) fn upstream(self) -> bool {
        matches!(
            self,
            Self::F32MasterUpstream
                | Self::F32MasterUpstreamPrefill
                | Self::F32MasterUpstreamExtraPrefill
                | Self::F32MasterUpstreamExtraLargePrefill
                | Self::F32MasterUpstreamExtraAllRows
                | Self::F32MasterUpstreamM8Geometry
                | Self::F32MasterUpstreamQ6F16
                | Self::F32MasterG32MmqPrefill
        )
    }

    pub(super) fn q8act(self) -> bool {
        matches!(self, Self::F32MasterQ8Act)
    }
}
