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
}

impl AttentionPrecision {
    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::F16 => GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            Self::F32Master => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
            Self::F32MasterGgufF16Projections => ferrum_interfaces::vnext::GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
            Self::F32MasterQ8Act => ferrum_interfaces::vnext::Q8ActAttentionProfile::GatedDelta.operation_id(),
        }
    }

    pub(super) fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::F16 => gated_delta_recurrent_attention_contract(),
            Self::F32Master => gated_delta_recurrent_attention_f32_master_contract(),
            Self::F32MasterGgufF16Projections => ferrum_interfaces::vnext::gated_delta_recurrent_attention_f32_master_gguf_f16_projections_contract(),
            Self::F32MasterQ8Act => ferrum_interfaces::vnext::Q8ActAttentionProfile::GatedDelta.contract(),
        }
    }

    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::F16 => GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
            Self::F32Master => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID,
            Self::F32MasterGgufF16Projections => ferrum_interfaces::vnext::GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID,
            Self::F32MasterQ8Act => ferrum_interfaces::vnext::Q8ActAttentionProfile::GatedDelta.capability_id(),
        }
    }

    pub(super) fn provider(self) -> &'static str {
        match self {
            Self::F16 => "provider.cuda.gated_delta_recurrent_attention.f16",
            Self::F32Master => "provider.cuda.gated_delta_recurrent_attention.f32-master",
            Self::F32MasterGgufF16Projections => {
                "provider.cuda.gated_delta_recurrent_attention.f32-master.gguf-f16-projections"
            }
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
            Self::F32MasterQ8Act => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-q8act-g32",
        }
    }

    pub(super) fn hidden(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32Master | Self::F32MasterGgufF16Projections | Self::F32MasterQ8Act => {
                ElementType::F32
            }
        }
    }

    pub(super) fn norm(self) -> &'static str {
        match self {
            Self::F16 => "rms_norm_f16",
            Self::F32Master | Self::F32MasterGgufF16Projections | Self::F32MasterQ8Act => {
                "vnext_rms_norm_f32_to_f16"
            }
        }
    }

    pub(super) fn residual(self) -> &'static str {
        match self {
            Self::F16 => "residual_add_f16",
            Self::F32Master | Self::F32MasterGgufF16Projections | Self::F32MasterQ8Act => {
                "vnext_residual_add_f32_f16"
            }
        }
    }

    pub(super) fn q8act(self) -> bool {
        matches!(self, Self::F32MasterQ8Act)
    }
}
