//! Noncausal cross-attention with independently shaped queries and key/value rows.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const CROSS_ATTENTION_BF16_OPERATION_ID: &str = "operation.cross_attention.bf16";
pub const CROSS_ATTENTION_BF16_CAPABILITY_ID: &str = "capability.operation.cross_attention.bf16";
pub const CROSS_ATTENTION_FUSED_BF16_OPERATION_ID: &str = "operation.cross_attention_fused.bf16";
pub const CROSS_ATTENTION_FUSED_BF16_CAPABILITY_ID: &str =
    "capability.operation.cross_attention_fused.bf16";

/// Q/Y are BF16 `[B*Q,H*D]`; K/V are BF16 `[B*KV,H*D]`. All queries see
/// exactly their batch's valid keys, in ascending key order. Padding suppresses
/// keys only. Empty sets produce +0 without reading Q/K/V. No causal mask,
/// dropout, paged state, position inference or probability BF16 boundary.
/// Providers identify their dot/exp/reduction arithmetic independently.
pub fn cross_attention_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let q = vec![symbol("query_rows"), symbol("columns")];
    let kv = vec![symbol("key_value_rows"), symbol("columns")];
    let mut attributes = BTreeMap::from([
        unsigned_attribute("batch")?,
        unsigned_attribute("head_dim")?,
        positive_rational_attribute("scale")?,
    ]);
    for (name, maximum) in [
        (
            "key_run_counts",
            crate::STRUCTURED_ATTENTION_BF16_MAX_BATCH as u32,
        ),
        ("key_run_lengths", u32::MAX),
        ("key_run_valid", u32::MAX),
    ] {
        attributes.insert(
            AttributeId::new(name)?,
            AttributeSpec {
                value_kind: AttributeValueKind::Integers,
                required: true,
                constraint: AttributeConstraint::IntegerListLength {
                    minimum: 1,
                    maximum,
                },
            },
        );
    }
    let descriptor = OperationDescriptor {
        id: OperationId::new(CROSS_ATTENTION_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(q.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(kv.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(kv, [ElementType::Bf16], TensorAccess::Read)?,
        ],
        outputs: vec![contiguous_tensor(
            q,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(attributes)?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            CROSS_ATTENTION_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Same visibility/shape as [`cross_attention_bf16_contract`], with a separate
/// vendor-fused numerical identity. Finite visible inputs target mathematical
/// scaled softmax attention; intermediate precision/association are provider
/// defined. No uniform error bound, nonfinite propagation, or bitwise agreement
/// with sequential FP32 or Torch is implied. All packing, sequence-length
/// arrays and vendor workspace are admitted invocation scratch, not hidden
/// allocations. Binding/persistent storage remain forbidden.
pub fn cross_attention_fused_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = cross_attention_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(CROSS_ATTENTION_FUSED_BF16_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        CROSS_ATTENTION_FUSED_BF16_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    descriptor.resources.scratch = ResourcePresenceRequirement::Optional;
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossAttentionBf16Geometry {
    batch: u64,
    query_length: u64,
    key_value_length: u64,
    columns: u64,
    head_dim: u64,
    scale: CanonicalRational,
    query_bytes: u64,
    key_value_bytes: u64,
    ends: Vec<Vec<u32>>,
    valid: Vec<Vec<bool>>,
    valid_keys: Vec<u32>,
    operations: u64,
}
impl CrossAttentionBf16Geometry {
    pub fn new(
        q: &ResolvedTensorSpec,
        k: &ResolvedTensorSpec,
        v: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        cross_attention_bf16_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        let [qr, c] = q.dimensions() else {
            return Err(invalid("Q must be rank two"));
        };
        let [kr, kc] = k.dimensions() else {
            return Err(invalid("K must be rank two"));
        };
        if q != y
            || k != v
            || c != kc
            || [q, k, v, y].iter().any(|t| {
                t.element_type() != ElementType::Bf16
                    || t.layout() != &ResolvedTensorLayout::Contiguous
            })
        {
            return Err(invalid(
                "Q/Y and K/V must match within each pair, with common BF16 contiguous columns",
            ));
        }
        let batch = unsigned(attrs, "batch")?;
        let d = unsigned(attrs, "head_dim")?;
        if !(1..=crate::STRUCTURED_ATTENTION_BF16_MAX_BATCH as u64).contains(&batch)
            || *qr == 0
            || *kr == 0
            || *c == 0
            || qr % batch != 0
            || kr % batch != 0
            || !(2..=256).contains(&d)
            || !(d.is_power_of_two() || d == 72)
            || c % d != 0
        {
            return Err(invalid("positive rows divisible by a representable positive batch and columns divisible by head_dim (power2 2..256 or 72) required"));
        }
        let query_length = qr / batch;
        let key_value_length = kr / batch;
        if query_length > u32::MAX as u64 || key_value_length > u32::MAX as u64 {
            return Err(invalid("per-batch sequence lengths exceed u32"));
        }
        let scale = match attribute(attrs, "scale") {
            Some(SemanticValue::Rational(s)) => *s,
            _ => return Err(invalid("scale must be rational")),
        };
        let effective = (scale.numerator() as f64 / scale.denominator() as f64) as f32;
        if scale.numerator() <= 0 || !effective.is_finite() || effective <= 0.0 {
            return Err(invalid("effective FP32 scale must be finite and positive"));
        }
        let query_bytes = q.minimum_storage_bytes()?;
        let key_value_bytes = k.minimum_storage_bytes()?;
        if query_bytes > isize::MAX as u64 || key_value_bytes > isize::MAX as u64 {
            return Err(invalid("tensor bytes exceed address space"));
        }
        let counts = integers(attrs, "key_run_counts")?;
        let lengths = integers(attrs, "key_run_lengths")?;
        let flags = integers(attrs, "key_run_valid")?;
        if counts.len() != batch as usize || lengths.len() != flags.len() {
            return Err(invalid(
                "key runs must specify every batch and equal lengths/flags",
            ));
        }
        let mut offset = 0usize;
        let mut ends = Vec::with_capacity(batch as usize);
        let mut valid = Vec::with_capacity(batch as usize);
        let mut valid_keys = Vec::with_capacity(batch as usize);
        let mut operations = 0u128;
        for &count in counts {
            let count = usize::try_from(count)
                .ok()
                .filter(|n| (1..=32).contains(n))
                .ok_or_else(|| invalid("run count must be 1..32"))?;
            let next = offset
                .checked_add(count)
                .filter(|&n| n <= lengths.len())
                .ok_or_else(|| invalid("key run count exceeds lists"))?;
            let mut row_ends = Vec::with_capacity(count);
            let mut row_valid = Vec::with_capacity(count);
            let mut end = 0u32;
            let mut included = 0u32;
            for (&length, &flag) in lengths[offset..next].iter().zip(&flags[offset..next]) {
                let length = u32::try_from(length)
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| invalid("key lengths must be positive u32"))?;
                let flag = match flag {
                    0 => false,
                    1 => true,
                    _ => return Err(invalid("key flags must be 0/1")),
                };
                if row_valid.last() == Some(&flag) {
                    return Err(invalid("adjacent key runs must alternate"));
                }
                end = end
                    .checked_add(length)
                    .filter(|&n| u64::from(n) <= key_value_length)
                    .ok_or_else(|| invalid("key runs exceed KV sequence"))?;
                if flag {
                    included = included
                        .checked_add(length)
                        .ok_or_else(|| invalid("valid key count overflows"))?;
                }
                row_ends.push(end);
                row_valid.push(flag);
            }
            if u64::from(end) != key_value_length {
                return Err(invalid("key runs must cover entire KV sequence"));
            }
            if included != 0 {
                // Reference online nominal scalar work per query/head. Not GPU FLOPs.
                operations += u128::from(query_length)
                    * u128::from(*c / d)
                    * (u128::from(5 * d + 6) * u128::from(included) - u128::from(2 * d + 5));
            }
            ends.push(row_ends);
            valid.push(row_valid);
            valid_keys.push(included);
            offset = next;
        }
        if offset != lengths.len() {
            return Err(invalid("unused key runs"));
        }
        Ok(Self {
            batch,
            query_length,
            key_value_length,
            columns: *c,
            head_dim: d,
            scale,
            query_bytes,
            key_value_bytes,
            ends,
            valid,
            valid_keys,
            operations: u64::try_from(operations)
                .map_err(|_| invalid("logical work exceeds u64"))?,
        })
    }
    pub fn batch(&self) -> u64 {
        self.batch
    }
    pub fn query_length(&self) -> u64 {
        self.query_length
    }
    pub fn key_value_length(&self) -> u64 {
        self.key_value_length
    }
    pub fn columns(&self) -> u64 {
        self.columns
    }
    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn heads(&self) -> u64 {
        self.columns / self.head_dim
    }
    pub fn scale(&self) -> CanonicalRational {
        self.scale
    }
    pub fn effective_scale(&self) -> f32 {
        (self.scale.numerator() as f64 / self.scale.denominator() as f64) as f32
    }
    pub fn query_bytes(&self) -> u64 {
        self.query_bytes
    }
    pub fn key_value_bytes(&self) -> u64 {
        self.key_value_bytes
    }
    pub fn element_count(&self) -> u64 {
        self.query_bytes / 2
    }
    pub fn logical_arithmetic_operations(&self) -> u64 {
        self.operations
    }
    pub fn key_run_ends(&self, batch: usize) -> Option<&[u32]> {
        self.ends.get(batch).map(Vec::as_slice)
    }
    pub fn key_run_valid(&self, batch: usize) -> Option<&[bool]> {
        self.valid.get(batch).map(Vec::as_slice)
    }
    pub fn valid_keys(&self, batch: usize) -> Option<u32> {
        self.valid_keys.get(batch).copied()
    }
    pub fn key_is_valid(&self, batch: usize, key: u32) -> Option<bool> {
        if u64::from(key) >= self.key_value_length {
            return None;
        }
        let ends = self.ends.get(batch)?;
        self.valid
            .get(batch)?
            .get(ends.partition_point(|&e| e <= key))
            .copied()
    }
}
fn attribute<'a>(
    attrs: &'a BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Option<&'a SemanticValue> {
    attrs
        .iter()
        .find(|(k, _)| k.as_str() == name)
        .map(|(_, v)| v)
}
fn unsigned(attrs: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> Result<u64, VNextError> {
    match attribute(attrs, name) {
        Some(SemanticValue::Unsigned(v)) => Ok(*v),
        _ => Err(invalid(format!("{name} must be unsigned"))),
    }
}
fn integers<'a>(
    attrs: &'a BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<&'a [i64], VNextError> {
    match attribute(attrs, name) {
        Some(SemanticValue::Integers(v)) => Ok(v),
        _ => Err(invalid(format!("{name} must be integers"))),
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 cross attention: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
