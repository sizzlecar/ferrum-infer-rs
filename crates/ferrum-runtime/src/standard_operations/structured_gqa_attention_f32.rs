//! Independently typed F32 grouped-query attention, with no half conversion.
use super::*;
use crate::{ResolvedTensorSpec, SemanticValue};

pub const STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID: &str =
    "operation.structured_gqa_attention.f32";
pub const STRUCTURED_GQA_ATTENTION_F32_CAPABILITY_ID: &str =
    "capability.operation.structured_gqa_attention.f32";

/// F32 Q/Y[B*S,Hq*D], K/V[B*S,Hkv*D]. Compact KV heads, explicit scale,
/// causal/bidirectional segments and key-only padding retain the exact closed
/// structured mask semantics. Empty visible sets write positive zero without
/// reading Q/K/V; masked keys are never read. Visible nonfinite scores poison
/// their whole query/head. F32 pairwise dot, ascending-key online softmax and
/// value accumulation never cross a BF16/F16 probability or output boundary.
/// Providers declare reduction, exponential, division and contraction profiles;
/// this mathematical contract does not imply Torch binary parity.
/// Head widths in 2..=256 include nonpowers of two. The adjacent-pair reduction
/// carries an unmatched subtree unchanged; it does not insert padding zeros.
pub fn structured_gqa_attention_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = structured_gqa_attention_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        STRUCTURED_GQA_ATTENTION_F32_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    let q = vec![symbol("rows"), symbol("query_columns")];
    let kv = vec![symbol("rows"), symbol("key_value_columns")];
    descriptor.inputs = vec![
        contiguous_tensor(q.clone(), [ElementType::F32], TensorAccess::Read)?,
        contiguous_tensor(kv.clone(), [ElementType::F32], TensorAccess::Read)?,
        contiguous_tensor(kv, [ElementType::F32], TensorAccess::Read)?,
    ];
    descriptor.outputs = vec![contiguous_tensor(
        q,
        [ElementType::F32],
        TensorAccess::Write,
    )?];
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Shares scalar-independent mask/index/work bookkeeping with BF16, while
/// construction enforces F32 on all four original ports without copying them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredGqaAttentionF32Geometry(StructuredGqaAttentionBf16Geometry);
impl StructuredGqaAttentionF32Geometry {
    pub fn new(
        q: &ResolvedTensorSpec,
        k: &ResolvedTensorSpec,
        v: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        Ok(Self(
            StructuredGqaAttentionBf16Geometry::new_with_element_type(
                q,
                k,
                v,
                y,
                attributes,
                ElementType::F32,
            )?,
        ))
    }
    pub fn rows(&self) -> u64 {
        self.0.rows()
    }
    pub fn query_columns(&self) -> u64 {
        self.0.query_columns()
    }
    pub fn key_value_columns(&self) -> u64 {
        self.0.key_value_columns()
    }
    pub fn sequence_length(&self) -> u64 {
        self.0.sequence_length()
    }
    pub fn head_dim(&self) -> u64 {
        self.0.head_dim()
    }
    pub fn batch(&self) -> u64 {
        self.0.batch()
    }
    pub fn query_heads(&self) -> u64 {
        self.0.query_heads()
    }
    pub fn key_value_heads(&self) -> u64 {
        self.0.key_value_heads()
    }
    pub fn group_size(&self) -> u64 {
        self.0.group_size()
    }
    pub fn scale(&self) -> CanonicalRational {
        self.0.scale()
    }
    pub fn effective_scale(&self) -> f32 {
        self.0.effective_scale()
    }
    pub fn mask(&self) -> &StructuredAttentionBf16Mask {
        self.0.mask()
    }
    pub fn element_count(&self) -> u64 {
        self.0.element_count()
    }
    pub fn logical_arithmetic_operations(&self) -> u64 {
        self.0.logical_arithmetic_operations()
    }
    pub fn visible_pairs(&self) -> u64 {
        self.0.visible_pairs()
    }
    pub fn nonempty_rows(&self) -> u64 {
        self.0.nonempty_rows()
    }
}

#[cfg(test)]
mod tests;
