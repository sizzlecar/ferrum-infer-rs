//! Hidden-state precision is an operation ABI, independent of matrix encoding.
use ferrum_interfaces::vnext::{
    gated_delta_recurrent_attention_contract, gated_delta_recurrent_attention_f32_master_contract,
    gated_delta_recurrent_attention_f32_master_q8_projections_contract, ElementType,
    StandardOperationContract, VNextError, GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
};

#[derive(Clone, Copy)]
pub(super) enum AttentionPrecision {
    F16,
    F32Master,
    F32MasterQ8Projections,
}

impl AttentionPrecision {
    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::F16 => GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            Self::F32Master => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
            Self::F32MasterQ8Projections => {
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_OPERATION_ID
            }
        }
    }

    pub(super) fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::F16 => gated_delta_recurrent_attention_contract(),
            Self::F32Master => gated_delta_recurrent_attention_f32_master_contract(),
            Self::F32MasterQ8Projections => {
                gated_delta_recurrent_attention_f32_master_q8_projections_contract()
            }
        }
    }

    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::F16 => GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
            Self::F32Master => GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_CAPABILITY_ID,
            Self::F32MasterQ8Projections => {
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_CAPABILITY_ID
            }
        }
    }

    pub(super) fn provider(self) -> &'static str {
        match self {
            Self::F16 => "provider.cuda.gated_delta_recurrent_attention.f16",
            Self::F32Master => "provider.cuda.gated_delta_recurrent_attention.f32-master",
            Self::F32MasterQ8Projections => {
                "provider.cuda.gated_delta_recurrent_attention.f32-master.q8-projections"
            }
        }
    }

    pub(super) fn estimator(self) -> &'static str {
        match self {
            Self::F16 => "resource-estimator.cuda.gated_delta_recurrent_attention.f16",
            Self::F32Master => "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master",
            Self::F32MasterQ8Projections => {
                "resource-estimator.cuda.gated_delta_recurrent_attention.f32-master.q8-projections"
            }
        }
    }

    pub(super) fn hidden(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32Master | Self::F32MasterQ8Projections => ElementType::F32,
        }
    }

    pub(super) fn norm(self) -> &'static str {
        match self {
            Self::F16 => "rms_norm_f16",
            Self::F32Master | Self::F32MasterQ8Projections => "vnext_rms_norm_f32_to_f16",
        }
    }

    pub(super) fn residual(self) -> &'static str {
        match self {
            Self::F16 => "residual_add_f16",
            Self::F32Master | Self::F32MasterQ8Projections => "vnext_residual_add_f32_f16",
        }
    }

    pub(super) fn quantizes_projections(self) -> bool {
        matches!(self, Self::F32MasterQ8Projections)
    }
}
