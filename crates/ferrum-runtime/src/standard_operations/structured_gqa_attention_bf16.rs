//! Nonpaged grouped-query BF16 attention without replicated K/V storage.
use super::structured_attention_bf16::{attribute, unsigned};
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID: &str =
    "operation.structured_gqa_attention.bf16";
pub const STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID: &str =
    "capability.operation.structured_gqa_attention.bf16";
/// Q/Y are BF16 [B*S,Hq*D], K/V are BF16 [B*S,Hkv*D], Hq is a
/// positive multiple of Hkv. Query head h reads KV head h/(Hq/Hkv); K/V
/// storage is never expanded. The immutable structured mask and explicit scale
/// have the same semantics as structured_attention.bf16, including key-only
/// padding and positive-zero empty rows without reading Q/K/V.
///
/// Scores, online softmax state and value accumulation remain FP32. No BF16
/// probability tensor is formed; only the final output crosses a BF16 boundary.
/// This is not bitwise equivalent to an eager attention path that rounds QK
/// matmul or softmax probabilities to BF16 before multiplying V. Providers
/// declare their FP32 dot reduction/exp order. Reference uses a balanced binary
/// dot tree and ascending-key online updates with separate FP32 products/sums.
/// A visible nonfinite score poisons that query/head; masked K/V are not read.
/// No inferred square-root scale, dropout, paged cache, or score matrix.
pub fn structured_gqa_attention_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let query_dims = vec![symbol("rows"), symbol("query_columns")];
    let kv_dims = vec![symbol("rows"), symbol("key_value_columns")];
    // Reuse the exact closed eight-attribute mask/scale schema.
    let attributes = super::structured_attention_bf16_contract()?
        .descriptor
        .attributes;
    let descriptor = OperationDescriptor {
        id: OperationId::new(STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(query_dims.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(kv_dims.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(kv_dims, [ElementType::Bf16], TensorAccess::Read)?,
        ],
        outputs: vec![contiguous_tensor(
            query_dims,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Geometry and checked logical work are descriptive; core alone admits buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredGqaAttentionBf16Geometry {
    rows: u64,
    query_columns: u64,
    key_value_columns: u64,
    sequence: u32,
    head_dim: u64,
    scale: CanonicalRational,
    mask: StructuredAttentionBf16Mask,
    elements: u64,
    operations: u64,
}
impl StructuredGqaAttentionBf16Geometry {
    pub fn new(
        q: &ResolvedTensorSpec,
        k: &ResolvedTensorSpec,
        v: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        structured_gqa_attention_bf16_contract()?
            .descriptor
            .validate_attributes(attributes)?;
        let [rows, columns] = q.dimensions() else {
            return Err(invalid("Q must be rank two"));
        };
        let [kv_rows, kv_columns] = k.dimensions() else {
            return Err(invalid("K must be rank two"));
        };
        if q != y
            || k != v
            || rows != kv_rows
            || k.element_type() != ElementType::Bf16
            || k.layout() != &ResolvedTensorLayout::Contiguous
            || q.element_type() != ElementType::Bf16
            || q.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid(
                "Q/Y and K/V must be respective identical contiguous BF16 rank-two tensors with equal rows",
            ));
        }
        let sequence = unsigned(attributes, "sequence_length")?;
        let head_dim = unsigned(attributes, "head_dim")?;
        if sequence == 0
            || sequence > u64::from(u32::MAX)
            || *rows == 0
            || *columns == 0
            || *kv_columns == 0
            || *rows % sequence != 0
            || head_dim < 2
            || head_dim > STRUCTURED_ATTENTION_BF16_MAX_HEAD_DIM
            || !head_dim.is_power_of_two()
            || *columns % head_dim != 0
            || *kv_columns % head_dim != 0
            || (*columns / head_dim) % (*kv_columns / head_dim) != 0
        {
            return Err(invalid("requires positive sequence, rows divisible by sequence, Q/KV columns divisible by power-of-two head_dim in 2..=256, and Hq a multiple of Hkv"));
        }
        let batch =
            usize::try_from(*rows / sequence).map_err(|_| invalid("batch exceeds usize"))?;
        if !(1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&batch) {
            return Err(invalid(
                "batch exceeds the representable mask metadata extent",
            ));
        }
        let Some(SemanticValue::Rational(scale)) = attribute(attributes, "scale") else {
            return Err(invalid("scale must be rational"));
        };
        let effective = (scale.numerator() as f64 / scale.denominator() as f64) as f32;
        if scale.numerator() <= 0 || !effective.is_finite() || effective <= 0.0 {
            return Err(invalid("effective FP32 scale must be finite and positive"));
        }
        q.minimum_storage_bytes()?;
        k.minimum_storage_bytes()?;
        let mask = StructuredAttentionBf16Mask::parse(sequence as u32, batch, attributes)?;
        let elements = rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        // First visible key: 2D dot+scale operations; subsequent keys: 5D+6
        // including two exp calls; final reciprocal and D multiplies: D+1.
        // Finite-path nominal work; comparisons/indexing/conversions excluded.
        // Nonfinite early exit can execute fewer operations. Not hardware FLOPs.
        let per_head = u128::from(5 * head_dim + 6) * u128::from(mask.visible_pairs())
            - u128::from(2 * head_dim + 5) * u128::from(mask.nonempty_rows());
        let operations = u64::try_from(
            per_head
                .checked_mul(u128::from(*columns / head_dim))
                .ok_or_else(|| invalid("logical work overflows u128"))?,
        )
        .map_err(|_| invalid("logical work overflows u64"))?;
        Ok(Self {
            rows: *rows,
            query_columns: *columns,
            key_value_columns: *kv_columns,
            sequence: sequence as u32,
            head_dim,
            scale: *scale,
            mask,
            elements,
            operations,
        })
    }
    pub const fn rows(&self) -> u64 {
        self.rows
    }
    pub const fn query_columns(&self) -> u64 {
        self.query_columns
    }
    pub const fn key_value_columns(&self) -> u64 {
        self.key_value_columns
    }
    pub const fn sequence_length(&self) -> u64 {
        self.sequence as u64
    }
    pub const fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn batch(&self) -> u64 {
        self.rows / u64::from(self.sequence)
    }
    pub fn query_heads(&self) -> u64 {
        self.query_columns / self.head_dim
    }
    pub fn key_value_heads(&self) -> u64 {
        self.key_value_columns / self.head_dim
    }
    pub fn group_size(&self) -> u64 {
        self.query_columns / self.key_value_columns
    }
    pub const fn scale(&self) -> CanonicalRational {
        self.scale
    }
    pub fn effective_scale(&self) -> f32 {
        (self.scale.numerator() as f64 / self.scale.denominator() as f64) as f32
    }
    pub fn mask(&self) -> &StructuredAttentionBf16Mask {
        &self.mask
    }
    pub const fn element_count(&self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(&self) -> u64 {
        self.operations
    }
    pub fn visible_pairs(&self) -> u64 {
        self.mask.visible_pairs()
    }
    pub fn nonempty_rows(&self) -> u64 {
        self.mask.nonempty_rows()
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 structured GQA attention: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
