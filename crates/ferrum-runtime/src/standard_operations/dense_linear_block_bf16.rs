//! Weight-only block quantization with explicit BF16 activation boundaries.
use super::*;

pub const DENSE_LINEAR_BLOCK_BF16_OPERATION_ID: &str = "operation.dense_linear_block.bf16";
pub const DENSE_LINEAR_BLOCK_BF16_CAPABILITY_ID: &str =
    "capability.operation.dense_linear_block.bf16";

/// Logical X[M,K], W[N,K], Y[M,N] use BF16 boundaries. W is stored in an
/// explicitly bound self-contained fixed-block ABI. Decode each block in F32,
/// round each decoded value once to BF16, then multiply/accumulate in F32 and
/// round the result to BF16. There is no bias or activation quantization.
/// Providers must publish their decode/reduction numerical identity; source
/// quantization error is independent of arithmetic error. This operation does
/// not promise equivalence to the unquantized checkpoint or another engine.
///
/// Any decoded matrix/tile is invocation scratch admitted by the core. No
/// persistent expanded weight or implicit native allocation is permitted.
pub fn dense_linear_block_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = dense_linear_contract_with_type(
        DENSE_LINEAR_BLOCK_BF16_OPERATION_ID,
        ContractVersion::new(1, 0),
        DENSE_LINEAR_BLOCK_BF16_CAPABILITY_ID,
        ElementType::Bf16,
        OracleSpec::OperationDefined,
    )?
    .descriptor;
    descriptor.resources.scratch = ResourcePresenceRequirement::Required;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_linear_has_an_independent_fixed_tensor_and_scratch_contract() {
        let block = dense_linear_block_bf16_contract().unwrap();
        let original = dense_linear_bf16_contract().unwrap();
        assert_ne!(block.descriptor().id, original.descriptor().id);
        assert_eq!(block.descriptor().inputs, original.descriptor().inputs);
        assert_eq!(block.descriptor().outputs, original.descriptor().outputs);
        assert_eq!(
            block.descriptor().resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            block.descriptor().resources.persistent,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(block.descriptor().oracle, OracleSpec::OperationDefined);
    }
}
