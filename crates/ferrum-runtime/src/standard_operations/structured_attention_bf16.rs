//! Nonpaged fixed-tensor attention with a bounded, immutable structured mask.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
mod mask;
pub use mask::StructuredAttentionBf16Mask;

pub const STRUCTURED_ATTENTION_BF16_OPERATION_ID: &str = "operation.structured_attention.bf16";
pub const STRUCTURED_ATTENTION_BF16_CAPABILITY_ID: &str =
    "capability.operation.structured_attention.bf16";
pub const STRUCTURED_ATTENTION_BF16_MAX_BATCH: usize = 16;
pub const STRUCTURED_ATTENTION_BF16_MAX_SEGMENTS: usize = 32;
pub const STRUCTURED_ATTENTION_BF16_MAX_KEY_RUNS: usize = 32;
pub const STRUCTURED_ATTENTION_BF16_MAX_HEAD_DIM: u64 = 256;

/// Q/K/V/Y are BF16 [B*S,H*D]. Scale is explicit: canonical rational divided
/// in F64 then converted to F32. Visibility is valid-key AND (causal OR within
/// the query's bidirectional segment); padding does not suppress queries.
/// Empty visible sets produce positive zero without reading Q/K/V.
/// Providers declare their FP32 dot/exp/online-softmax order. No score matrix,
/// paged KV, dropout, frequency generation, or inferred square-root scale.
pub fn structured_attention_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let dims = vec![symbol("rows"), symbol("columns")];
    let mut attributes = BTreeMap::from([
        unsigned_attribute("sequence_length")?,
        unsigned_attribute("head_dim")?,
        positive_rational_attribute("scale")?,
    ]);
    for (name, max) in [
        ("segment_lengths", 32),
        ("segment_bidirectional", 32),
        ("key_run_counts", 16),
        ("key_run_lengths", 512),
        ("key_run_valid", 512),
    ] {
        attributes.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: 1,
                    maximum: max,
                },
            },
        );
    }
    let descriptor = OperationDescriptor {
        id: OperationId::new(STRUCTURED_ATTENTION_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(dims.clone(), [ElementType::Bf16], TensorAccess::Read)?; 3],
        outputs: vec![contiguous_tensor(
            dims,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(attributes)?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            STRUCTURED_ATTENTION_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Geometry and checked logical work are descriptive; core alone admits buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredAttentionBf16Geometry {
    rows: u64,
    columns: u64,
    sequence: u32,
    head_dim: u64,
    scale: CanonicalRational,
    mask: StructuredAttentionBf16Mask,
    elements: u64,
    operations: u64,
}
impl StructuredAttentionBf16Geometry {
    pub fn new(
        q: &ResolvedTensorSpec,
        k: &ResolvedTensorSpec,
        v: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        structured_attention_bf16_contract()?
            .descriptor
            .validate_attributes(attributes)?;
        let [rows, columns] = q.dimensions() else {
            return Err(invalid("Q must be rank two"));
        };
        if q != k
            || q != v
            || q != y
            || q.element_type() != ElementType::Bf16
            || q.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid(
                "Q/K/V/Y must be identical contiguous BF16 [rows,columns]",
            ));
        }
        let sequence = unsigned(attributes, "sequence_length")?;
        let head_dim = unsigned(attributes, "head_dim")?;
        if sequence == 0
            || sequence > u64::from(u32::MAX)
            || *rows == 0
            || *columns == 0
            || *rows % sequence != 0
            || head_dim < 2
            || head_dim > STRUCTURED_ATTENTION_BF16_MAX_HEAD_DIM
            || !head_dim.is_power_of_two()
            || *columns % head_dim != 0
        {
            return Err(invalid("requires positive sequence, rows divisible by sequence, and columns divisible by power-of-two head_dim in 2..=256"));
        }
        let batch =
            usize::try_from(*rows / sequence).map_err(|_| invalid("batch exceeds usize"))?;
        if !(1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&batch) {
            return Err(invalid("batch must be in 1..=16"));
        }
        let Some(SemanticValue::Rational(scale)) = attribute(attributes, "scale") else {
            return Err(invalid("scale must be rational"));
        };
        let effective = (scale.numerator() as f64 / scale.denominator() as f64) as f32;
        if scale.numerator() <= 0 || !effective.is_finite() || effective <= 0.0 {
            return Err(invalid("effective FP32 scale must be finite and positive"));
        }
        q.minimum_storage_bytes()?;
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
            columns: *columns,
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
    pub const fn columns(&self) -> u64 {
        self.columns
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
    pub fn heads(&self) -> u64 {
        self.columns / self.head_dim
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
pub(super) fn attribute<'a>(
    attrs: &'a BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Option<&'a SemanticValue> {
    attrs
        .iter()
        .find(|(id, _)| id.as_str() == name)
        .map(|(_, value)| value)
}
pub(super) fn unsigned(
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<u64, VNextError> {
    match attribute(attrs, name) {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(invalid(format!("{name} must be unsigned"))),
    }
}
fn integers<'a>(
    attrs: &'a BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<&'a [i64], VNextError> {
    match attribute(attrs, name) {
        Some(SemanticValue::Integers(value)) => Ok(value),
        _ => Err(invalid(format!("{name} must be an integer list"))),
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 structured attention: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
