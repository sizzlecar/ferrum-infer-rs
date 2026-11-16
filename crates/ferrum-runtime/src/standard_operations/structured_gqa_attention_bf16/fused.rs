//! Vendor-fused grouped-query attention with compact K/V and explicit masks.
use super::*;
pub const STRUCTURED_GQA_ATTENTION_FUSED_BF16_OPERATION_ID: &str =
    "operation.structured_gqa_attention_fused.bf16";
pub const STRUCTURED_GQA_ATTENTION_FUSED_BF16_CAPABILITY_ID: &str =
    "capability.operation.structured_gqa_attention_fused.bf16";

/// Shares the exact compact Q/K/V shapes, grouped-head mapping, structured
/// visibility and explicit scale of `structured_gqa_attention_bf16`. For finite
/// visible data, implementations may use vendor matrix arithmetic, reduction
/// association and intermediate rounding, including BF16 probabilities. No
/// sequential FP32 or Torch bitwise equivalence is claimed. Masked keys cannot
/// affect outputs; empty visible sets produce positive zero. Provider support
/// may be narrower and must reject unsupported masks before execution.
///
/// Every native workspace is admitted invocation scratch. No hidden allocation,
/// dropout, paged state, or expanded K/V storage is permitted.
pub fn structured_gqa_attention_fused_bf16_contract(
) -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = structured_gqa_attention_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(STRUCTURED_GQA_ATTENTION_FUSED_BF16_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        STRUCTURED_GQA_ATTENTION_FUSED_BF16_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    descriptor.resources.scratch = ResourcePresenceRequirement::Optional;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
