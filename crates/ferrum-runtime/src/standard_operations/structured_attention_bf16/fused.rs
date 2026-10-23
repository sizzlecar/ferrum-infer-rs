//! A separate operation for vendor-fused structured attention.
//!
//! The original operation's descriptor and numerical identity remain unchanged.
//! This operation shares its exact visibility, tensor ABI and explicit scale,
//! while allowing provider-declared workspace and a different numerical method.

use super::*;

pub const STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID: &str =
    "operation.structured_attention_fused.bf16";
pub const STRUCTURED_ATTENTION_FUSED_BF16_CAPABILITY_ID: &str =
    "capability.operation.structured_attention_fused.bf16";

/// Q/K/V/Y are contiguous BF16 `[B*S,H*D]`, with the same immutable structured
/// visibility and F64-rational-to-F32 scale conversion as the original operation.
/// [`StructuredAttentionBf16Geometry`] validates both operations' identical port
/// and attribute geometry; provider shape eligibility can be narrower.
///
/// For finite visible logits the mathematical target is scaled dot-product
/// softmax attention over exactly the visible keys. A provider may use fused
/// matrix arithmetic, a different reduction association, and implementation-
/// defined intermediate rounding. This is not the original sequential FP32
/// online-softmax profile and makes no bitwise, uniform error-bound, or
/// nonfinite-propagation equivalence claim. The provider must identify its actual
/// numerical implementation; qualification is separate evidence. Empty visible
/// sets still produce positive zero without reading Q/K/V, and masked keys must
/// not affect the result. There is no dropout, paged state, or inferred scale.
///
/// All device packing, length arrays and vendor workspace must be declared by
/// the provider estimator as core-admitted invocation scratch and retained to
/// completion. Optional scratch permits a zero-workspace implementation; it is
/// not permission to allocate unreported workspace. Binding and persistent
/// storage remain forbidden. Logical work from the shared geometry is the same
/// reference-algorithm count, not a claim about vendor hardware instructions.
pub fn structured_attention_fused_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = structured_attention_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        STRUCTURED_ATTENTION_FUSED_BF16_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    descriptor.resources.scratch = ResourcePresenceRequirement::Optional;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[cfg(test)]
mod tests;
