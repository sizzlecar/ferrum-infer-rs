//! Single-head full spatial attention directly over a fused NCHW QKV port.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
pub const SPATIAL_ATTENTION_BF16_OPERATION_ID: &str = "operation.spatial_attention.bf16";
pub const SPATIAL_ATTENTION_BF16_CAPABILITY_ID: &str =
    "capability.operation.spatial_attention.bf16";
pub const SPATIAL_ATTENTION_BF16_MAX_CHANNELS: u64 = 2048;

/// QKV[N,3*C,H,W] has contiguous Q, K, V channel blocks; Y[N,C,H,W].
/// Exactly one head of width C attends all H*W keys independently per sample.
/// No causal mask, dropout, implicit scale, or intermediate BF16 probability.
/// Scale is a positive rational evaluated by F64 division then F32 conversion.
/// Providers specify their FP32 reduction/exp/softmax profile; this does not
/// assert PyTorch SDPA bitwise equivalence. Output BF16, NoAlias.
/// A nonfinite score makes its entire query output NaN. Nonfinite V propagates
/// component-wise through the declared arithmetic. All keys are visible.
pub fn spatial_attention_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(SPATIAL_ATTENTION_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            vec![
                symbol("batch"),
                symbol("qkv_channels"),
                symbol("height"),
                symbol("width"),
            ],
            [ElementType::Bf16],
            TensorAccess::Read,
        )?],
        outputs: vec![contiguous_tensor(
            vec![
                symbol("batch"),
                symbol("channels"),
                symbol("height"),
                symbol("width"),
            ],
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([positive_rational_attribute("scale")?]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            SPATIAL_ATTENTION_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialAttentionBf16Geometry {
    shape: [u64; 4],
    spatial: u64,
    scale: CanonicalRational,
    elements: u64,
    operations: u64,
}
impl SpatialAttentionBf16Geometry {
    pub fn new(
        qkv: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        spatial_attention_bf16_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        let x = checked_nchw(qkv)?;
        let y = checked_nchw(output)?;
        if y[1] > SPATIAL_ATTENTION_BF16_MAX_CHANNELS
            || x != [
                y[0],
                y[1].checked_mul(3)
                    .ok_or_else(|| invalid("3*C overflows"))?,
                y[2],
                y[3],
            ]
        {
            return Err(invalid(
                "requires exact packed QKV[N,3*C,H,W], Y[N,C,H,W], C in 1..=2048",
            ));
        }
        let Some(SemanticValue::Rational(scale)) = attrs.get(&AttributeId::new("scale")?) else {
            return Err(invalid("scale must be rational"));
        };
        let effective = (scale.numerator() as f64 / scale.denominator() as f64) as f32;
        if scale.numerator() <= 0 || !effective.is_finite() || effective <= 0.0 {
            return Err(invalid("effective scale must be finite positive F32"));
        }
        let spatial = y[2]
            .checked_mul(y[3])
            .ok_or_else(|| invalid("spatial extent overflows"))?;
        let elements = y[0]
            .checked_mul(y[1])
            .and_then(|n| n.checked_mul(spatial))
            .ok_or_else(|| invalid("elements overflow"))?;
        // Nominal finite-path scalar work, including padded dot additions.
        // For a query: S*(C + P) dot+scale, (S-1)*(3C+6)
        // online update, then 1+C normalization. P=next_power_of_two(C).
        let c = u128::from(y[1]);
        let p = u128::from(y[1].next_power_of_two());
        let s = u128::from(spatial);
        let per_query = s * (c + p) + (s - 1) * (3 * c + 6) + (c + 1);
        let operations = per_query
            .checked_mul(s)
            .and_then(|n| n.checked_mul(u128::from(y[0])))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| invalid("logical work overflows"))?;
        Ok(Self {
            shape: y,
            spatial,
            scale: *scale,
            elements,
            operations,
        })
    }
    pub const fn batch(self) -> u64 {
        self.shape[0]
    }
    pub const fn channels(self) -> u64 {
        self.shape[1]
    }
    pub const fn height(self) -> u64 {
        self.shape[2]
    }
    pub const fn width(self) -> u64 {
        self.shape[3]
    }
    pub const fn spatial(self) -> u64 {
        self.spatial
    }
    pub const fn scale(self) -> CanonicalRational {
        self.scale
    }
    pub fn effective_scale(self) -> f32 {
        (self.scale.numerator() as f64 / self.scale.denominator() as f64) as f32
    }
    pub const fn element_count(self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.operations
    }
}
pub(super) fn checked_nchw(t: &ResolvedTensorSpec) -> Result<[u64; 4], VNextError> {
    let dims: [u64; 4] = t
        .dimensions()
        .try_into()
        .map_err(|_| invalid("requires rank-four NCHW"))?;
    if dims.contains(&0)
        || t.element_type() != ElementType::Bf16
        || t.layout() != &ResolvedTensorLayout::Contiguous
    {
        return Err(invalid("requires positive contiguous BF16 NCHW"));
    }
    let bytes = t.minimum_storage_bytes()?;
    if bytes > isize::MAX as u64 {
        return Err(invalid("tensor exceeds address space"));
    }
    Ok(dims)
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 spatial attention: {}", reason.into()),
    }
}
