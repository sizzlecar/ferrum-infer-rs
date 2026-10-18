use super::*;

pub const LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID: &str =
    "operation.last_token_dense_linear.f32.q6-d4-mmq-v1";
pub const LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID: &str =
    "capability.operation.last_token_dense_linear.f32.q6-d4-mmq-v1";

/// Same last-token tensor semantics and logical F16 weight port as the strict
/// F32 head. Q6 bytes remain the physical weight ABI. Resource sizes are the
/// provider's checked native plan plus four retained flag bytes per leaf.
pub fn last_token_dense_linear_q6_mmq_f32_contract() -> Result<StandardOperationContract, VNextError>
{
    let mut contract = last_token_dense_linear_contract_with_activation(
        LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID,
        ContractVersion::new(1, 0),
        LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID,
        ElementType::F32,
    )?;
    contract.descriptor.resources.scratch = ResourcePresenceRequirement::Required;
    contract.descriptor.resources.persistent = ResourcePresenceRequirement::Required;
    // The independent actual-pack/F64 oracle separates input quantization
    // from implementation error; no uniform relative-to-strict tolerance.
    contract.descriptor.oracle = OracleSpec::OperationDefined;
    contract.descriptor.validate()?;
    Ok(contract)
}
