//! Explicit whole-M RN-F16 FFN execution. The two physical representations
//! restore the same once-rounded RN-even F16 source coefficients. For whole
//! invocation M=1..8 (including small prefill/tails), multiply via F16 fragment
//! MMA with F32 accumulation. For M>8 use the original dense RN-F16 GEMM.
//! Preserve gate/up F16 -> original F32 SiLU/multiply -> F16 -> down F16 and
//! separate residual boundaries. Different F32 reduction trees do not imply
//! bitwise equality or full-model quality. No default/Auto selection is granted.
use super::*;

pub const DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID: &str =
    "operation.dense_swiglu.gguf-rn-f16-fragment-m1to8";
pub const DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.gguf-rn-f16-fragment-m1to8";

/// Exact consumer authority only. Version, shape, all aliases, source encoding
/// and the separately verified approximate materializer remain mandatory.
/// The existing dense-only RN classifier deliberately does not recognize this.
pub fn gguf_rn_f16_fragment_role_v1(
    operation: &OperationId,
    input_ordinal: u32,
) -> Option<GgufF16ProjectionRoleV1> {
    if operation.as_str() != DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID {
        return None;
    }
    match input_ordinal {
        1 => Some(GgufF16ProjectionRoleV1::FfnGateUp),
        2 => Some(GgufF16ProjectionRoleV1::FfnDown),
        _ => None,
    }
}

pub fn dense_swiglu_gguf_rn_f16_fragment_m1to8_contract(
) -> Result<StandardOperationContract, VNextError> {
    let mut contract = dense_swiglu_gguf_f16_weights_contract()?;
    contract.descriptor.id =
        OperationId::new(DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID)?;
    contract.descriptor.version = ContractVersion::new(1, 0);
    contract.descriptor.provider = provider_requirement(
        DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    contract.descriptor.validate()?;
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rn_fragment_contract_keeps_tensor_resource_state_abi_and_exclusive_authority() {
        let original = dense_swiglu_gguf_f16_weights_contract().unwrap();
        let changed = dense_swiglu_gguf_rn_f16_fragment_m1to8_contract().unwrap();
        let mut expected = original.descriptor.clone();
        expected.id =
            OperationId::new(DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID).unwrap();
        expected.provider = provider_requirement(
            DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )
        .unwrap();
        assert_eq!(changed.descriptor, expected);
        for ordinal in 0..8 {
            assert_eq!(
                gguf_rn_f16_fragment_role_v1(&expected.id, ordinal).is_some(),
                matches!(ordinal, 1 | 2)
            );
            assert_eq!(gguf_f16_projection_role_v1(&expected.id, ordinal), None);
            assert_eq!(
                gguf_rn_f16_fragment_role_v1(&original.descriptor.id, ordinal),
                None
            );
        }
        assert_eq!(gguf_rn_f16_fragment_role_v1(&expected.id, u32::MAX), None);
    }
}
