//! Precision of semantic logits and their private repetition-penalty view.

use super::*;
use ferrum_interfaces::vnext::{
    last_token_masked_argmax_f32_contract, StandardOperationContract,
    LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID, LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
};

#[derive(Clone, Copy)]
pub(super) enum ArgmaxPrecision {
    F16,
    F32,
}

impl ArgmaxPrecision {
    pub(super) fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::F16 => last_token_masked_argmax_contract(),
            Self::F32 => last_token_masked_argmax_f32_contract(),
        }
    }

    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
            Self::F32 => LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
        }
    }

    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID,
            Self::F32 => LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID,
        }
    }

    pub(super) fn provider(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_MASKED_ARGMAX_PROVIDER_ID,
            Self::F32 => "provider.cuda.last_token_masked_argmax.f32",
        }
    }

    pub(super) fn estimator(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_MASKED_ARGMAX_ESTIMATOR_ID,
            Self::F32 => "resource-estimator.cuda.last_token_masked_argmax.f32",
        }
    }

    pub(super) fn kernel(self) -> &'static str {
        match self {
            Self::F16 => MASKED_ARGMAX_PRESERVING_LOGITS_FUNCTION_NAME,
            Self::F32 => "last_token_masked_argmax_preserving_logits_f32",
        }
    }

    pub(super) fn element(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32 => ElementType::F32,
        }
    }
}

#[cfg(test)]
mod tests;
