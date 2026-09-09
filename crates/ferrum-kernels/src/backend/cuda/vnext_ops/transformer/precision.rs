//! Declared activation types for dense normalization and residual providers.

use ferrum_interfaces::vnext::{
    residual_add_contract, residual_add_f32_f16_contract, rms_norm_contract, rms_norm_f32_contract,
    rms_norm_f32_to_f16_contract, ElementType, StandardOperationContract, VNextError,
    RESIDUAL_ADD_F16_CAPABILITY_ID, RESIDUAL_ADD_F32_F16_CAPABILITY_ID,
    RESIDUAL_ADD_F32_F16_OPERATION_ID, RESIDUAL_ADD_OPERATION_ID, RMS_NORM_F16_CAPABILITY_ID,
    RMS_NORM_F32_CAPABILITY_ID, RMS_NORM_F32_OPERATION_ID, RMS_NORM_F32_TO_F16_CAPABILITY_ID,
    RMS_NORM_F32_TO_F16_OPERATION_ID, RMS_NORM_OPERATION_ID,
};

#[derive(Clone, Copy)]
pub(super) enum RmsNormPrecision {
    F16,
    F32ToF16,
    F32,
}

impl RmsNormPrecision {
    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::F16 => RMS_NORM_OPERATION_ID,
            Self::F32ToF16 => RMS_NORM_F32_TO_F16_OPERATION_ID,
            Self::F32 => RMS_NORM_F32_OPERATION_ID,
        }
    }

    pub(super) fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::F16 => rms_norm_contract(),
            Self::F32ToF16 => rms_norm_f32_to_f16_contract(),
            Self::F32 => rms_norm_f32_contract(),
        }
    }

    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::F16 => RMS_NORM_F16_CAPABILITY_ID,
            Self::F32ToF16 => RMS_NORM_F32_TO_F16_CAPABILITY_ID,
            Self::F32 => RMS_NORM_F32_CAPABILITY_ID,
        }
    }

    pub(super) fn provider(self) -> &'static str {
        match self {
            Self::F16 => "provider.cuda.rms_norm.f16",
            Self::F32ToF16 => "provider.cuda.rms_norm.f32-to-f16",
            Self::F32 => "provider.cuda.rms_norm.f32",
        }
    }

    pub(super) fn estimator(self) -> &'static str {
        match self {
            Self::F16 => "resource-estimator.cuda.rms_norm.f16",
            Self::F32ToF16 => "resource-estimator.cuda.rms_norm.f32-to-f16",
            Self::F32 => "resource-estimator.cuda.rms_norm.f32",
        }
    }

    pub(super) fn kernel(self) -> &'static str {
        match self {
            Self::F16 => "rms_norm_f16",
            Self::F32ToF16 => "vnext_rms_norm_f32_to_f16",
            Self::F32 => "vnext_rms_norm_f32",
        }
    }

    pub(super) fn input(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32ToF16 | Self::F32 => ElementType::F32,
        }
    }

    pub(super) fn output(self) -> ElementType {
        match self {
            Self::F16 | Self::F32ToF16 => ElementType::F16,
            Self::F32 => ElementType::F32,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum ResidualPrecision {
    F16,
    F32F16,
}

impl ResidualPrecision {
    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::F16 => RESIDUAL_ADD_OPERATION_ID,
            Self::F32F16 => RESIDUAL_ADD_F32_F16_OPERATION_ID,
        }
    }

    pub(super) fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::F16 => residual_add_contract(),
            Self::F32F16 => residual_add_f32_f16_contract(),
        }
    }

    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::F16 => RESIDUAL_ADD_F16_CAPABILITY_ID,
            Self::F32F16 => RESIDUAL_ADD_F32_F16_CAPABILITY_ID,
        }
    }

    pub(super) fn provider(self) -> &'static str {
        match self {
            Self::F16 => "provider.cuda.residual_add.f16",
            Self::F32F16 => "provider.cuda.residual_add.f32-f16",
        }
    }

    pub(super) fn estimator(self) -> &'static str {
        match self {
            Self::F16 => "resource-estimator.cuda.residual_add.f16",
            Self::F32F16 => "resource-estimator.cuda.residual_add.f32-f16",
        }
    }

    pub(super) fn kernel(self) -> &'static str {
        match self {
            Self::F16 => "residual_add_f16",
            Self::F32F16 => "vnext_residual_add_f32_f16",
        }
    }

    pub(super) fn master(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32F16 => ElementType::F32,
        }
    }
}

#[cfg(test)]
mod tests;
