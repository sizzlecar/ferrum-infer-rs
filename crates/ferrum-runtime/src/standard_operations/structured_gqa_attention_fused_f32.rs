//! Separate numerical contract for vendor-fused F32 grouped-query attention.
use super::*;

pub const STRUCTURED_GQA_ATTENTION_FUSED_F32_OPERATION_ID: &str =
    "operation.structured_gqa_attention_fused.f32";
pub const STRUCTURED_GQA_ATTENTION_FUSED_F32_CAPABILITY_ID: &str =
    "capability.operation.structured_gqa_attention_fused.f32";

/// Contiguous F32 Q/Y `[B*S,Hq*D]` and K/V `[B*S,Hkv*D]`, exact explicit
/// structured visibility, compact KV heads and explicit scale share
/// [`StructuredGqaAttentionF32Geometry`]. There is no BF16/F16 tensor boundary.
/// For finite visible inputs/logits the target is scaled dot-product softmax
/// attention over exactly the visible keys. Providers may use vendor fused
/// matrix arithmetic (including FastF32 tensor operations), different reduction
/// association and intermediate rounding; they must identify the actual method.
/// This contract does not inherit the strict pairwise/ascending-key operation's
/// bitwise, nonfinite-propagation or uniform error-bound guarantees. Qualification
/// is target- and workload-specific evidence, not a change to the old contract.
/// Empty visible sets still write positive zero without reading Q/K/V; masked
/// keys must not influence results. No dropout, bias or implicit scale is added.
///
/// Provider eligibility may be narrower (for example full all-valid MHA), with
/// the original structured kernel as an explicitly identified fallback. Any
/// packing or vendor workspace is declared, Core-admitted invocation scratch
/// retained until completion. Optional scratch permits zero workspace; binding
/// and persistent resources remain forbidden. Logical work uses shared geometry
/// rather than claiming a vendor hardware instruction count.
pub fn structured_gqa_attention_fused_f32_contract() -> Result<StandardOperationContract, VNextError>
{
    let mut descriptor = structured_gqa_attention_f32_contract()?.descriptor;
    descriptor.id = OperationId::new(STRUCTURED_GQA_ATTENTION_FUSED_F32_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        STRUCTURED_GQA_ATTENTION_FUSED_F32_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    descriptor.resources.scratch = ResourcePresenceRequirement::Optional;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[cfg(test)]
mod tests;
