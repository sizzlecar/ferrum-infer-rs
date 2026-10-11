//! Explicit local Q8 projection arithmetic inside strict attention operations.
use super::*;
use crate::vnext::{
    CompositeNumericalArithmetic, ProjectionArithmeticOverride, ProjectionRole,
    StrictNumericalOperation, COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_V2,
};

pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_Q8ACT_G32_OPERATION_ID: &str =
    "operation.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-q8act-g32";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_Q8ACT_G32_CAPABILITY_ID: &str =
    "capability.operation.gated_delta_recurrent_attention.f32-master.q4k-q5k-iq4xs-q8act-g32";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_Q8ACT_G32_OPERATION_ID: &str =
    "operation.causal_paged_attention.f32-master.q4k-q5k-iq4xs-q8act-g32";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_Q8ACT_G32_CAPABILITY_ID: &str =
    "capability.operation.causal_paged_attention.f32-master.q4k-q5k-iq4xs-q8act-g32";

/// These identities modify only declared projection leaves. State updates,
/// normalization, RoPE, gating, and all other work retain the strict base ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q8ActAttentionProfile {
    GatedDelta,
    Causal,
}

impl Q8ActAttentionProfile {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::GatedDelta => {
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_Q8ACT_G32_OPERATION_ID
            }
            Self::Causal => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_Q8ACT_G32_OPERATION_ID,
        }
    }

    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::GatedDelta => {
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_Q8ACT_G32_CAPABILITY_ID
            }
            Self::Causal => CAUSAL_PAGED_ATTENTION_F32_MASTER_Q4K_Q5K_IQ4XS_Q8ACT_G32_CAPABILITY_ID,
        }
    }

    fn strict_contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::GatedDelta => gated_delta_recurrent_attention_f32_master_contract(),
            Self::Causal => causal_paged_attention_f32_master_contract(),
        }
    }

    pub fn arithmetic(self) -> CompositeNumericalArithmetic {
        // Reuse the same format-specific five-stage declarations. The new
        // identities change eligible operation ports, never the old FFN policy.
        let mut arithmetic = dense_swiglu_q4k_q5k_iq4xs_q8act_g32_arithmetic();
        let projection = arithmetic.projections[0].clone();
        let base = self
            .strict_contract()
            .expect("fixed standard attention contract");
        arithmetic.strict_base = StrictNumericalOperation {
            operation_id: base.descriptor().id.clone(),
            version: base.descriptor().version,
        };
        let roles: &[(ProjectionRole, u32)] = match self {
            Self::GatedDelta => &[
                (ProjectionRole::GatedDeltaInput, 2),
                (ProjectionRole::GatedDeltaOutput, 7),
            ],
            Self::Causal => {
                arithmetic.schema_version = COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_V2;
                &[
                    (ProjectionRole::CausalQuery, 2),
                    (ProjectionRole::CausalKey, 3),
                    (ProjectionRole::CausalValue, 4),
                    (ProjectionRole::CausalOutput, 5),
                ]
            }
        };
        arithmetic.projections = roles
            .iter()
            .map(
                |&(role, weight_input_ordinal)| ProjectionArithmeticOverride {
                    role,
                    weight_input_ordinal,
                    ..projection.clone()
                },
            )
            .collect();
        arithmetic
    }

    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        let mut contract = self.strict_contract()?;
        contract.descriptor.id = OperationId::new(self.operation_id())?;
        contract.descriptor.provider =
            provider_requirement(self.capability_id(), ContractVersion::new(1, 0))?;
        contract.descriptor.validate()?;
        Ok(contract)
    }
}
