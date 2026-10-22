//! Channel L2 normalization with independent BF16 storage boundaries.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};

pub const CHANNEL_L2_NORM_BF16_OPERATION_ID: &str = "operation.channel_l2_norm.bf16";
pub const CHANNEL_L2_NORM_BF16_CAPABILITY_ID: &str = "capability.operation.channel_l2_norm.bf16";
pub const CHANNEL_L2_NORM_BF16_MAX_CHANNELS: u64 = 1 << 24;

/// X/Y are contiguous BF16 NCHW; gamma is the actual contiguous BF16
/// [C,1,1,1] operand. A rank-three gamma is a different contract, not an
/// implicit reshape. For each N/H/W position: L=sqrt(sum_c x[c]^2) in FP32,
/// d=max(L,epsilon), t=B(RN32(x/d)), u=B(RN32(t*scale)),
/// v=B(RN32(u*gamma)), y=B(RN32(v+positive_zero)). The last add clears -0.
/// Epsilon and scale are explicit positive rationals converted by F64 division
/// then F32. No mean reduction, additive epsilon, inferred sqrt(C), or fusion
/// across the BF16 boundaries. NaN in L propagates through clamp and division.
pub fn channel_l2_norm_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let dims = vec![
        symbol("batch"),
        symbol("channels"),
        symbol("height"),
        symbol("width"),
    ];
    let descriptor = OperationDescriptor {
        id: OperationId::new(CHANNEL_L2_NORM_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(dims.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![
                    symbol("channels"),
                    DimensionConstraint::Exact(1),
                    DimensionConstraint::Exact(1),
                    DimensionConstraint::Exact(1),
                ],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            dims,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            positive_epsilon_attribute("epsilon")?,
            (
                AttributeId::new("scale")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Rational,
                    required: true,
                    constraint: AttributeConstraint::None,
                },
            ),
        ]))?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            CHANNEL_L2_NORM_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelL2NormBf16Geometry {
    shape: [u64; 4],
    epsilon: CanonicalRational,
    scale: CanonicalRational,
    elements: u64,
    operations: u64,
}
impl ChannelL2NormBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        channel_l2_norm_bf16_contract()?
            .descriptor
            .validate_attributes(attrs)?;
        let shape: [u64; 4] = input
            .dimensions()
            .try_into()
            .map_err(|_| invalid("requires NCHW rank four"))?;
        if input != output
            || shape.contains(&0)
            || shape[1] > CHANNEL_L2_NORM_BF16_MAX_CHANNELS
            || gamma.dimensions() != [shape[1], 1, 1, 1]
        {
            return Err(invalid("requires identical NCHW input/output, channels in 1..=2^24, and actual gamma[C,1,1,1]"));
        }
        for t in [input, gamma, output] {
            if t.element_type() != ElementType::Bf16
                || t.layout() != &ResolvedTensorLayout::Contiguous
            {
                return Err(invalid("requires contiguous BF16 operands"));
            }
            t.minimum_storage_bytes()?;
        }
        let epsilon = rational(attrs, "epsilon")?;
        let scale = rational(attrs, "scale")?;
        for v in [epsilon, scale] {
            let f = effective(v);
            if v.numerator() <= 0 || !f.is_finite() || f <= 0.0 {
                return Err(invalid("effective FP32 scalar must be finite and positive"));
            }
        }
        let elements = shape
            .into_iter()
            .try_fold(1_u64, |n, d| n.checked_mul(d))
            .ok_or_else(|| invalid("element count overflows"))?;
        // C squares + C-1 adds + sqrt + C*(division,scale,gamma,+0).
        // Clamp comparisons and BF16 conversions are not arithmetic operations.
        let operations = elements
            .checked_mul(6)
            .ok_or_else(|| invalid("logical work overflows"))?;
        Ok(Self {
            shape,
            epsilon,
            scale,
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
    pub const fn epsilon(self) -> CanonicalRational {
        self.epsilon
    }
    pub const fn scale(self) -> CanonicalRational {
        self.scale
    }
    pub fn effective_epsilon(self) -> f32 {
        effective(self.epsilon)
    }
    pub fn effective_scale(self) -> f32 {
        effective(self.scale)
    }
    pub const fn element_count(self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.operations
    }
}
fn effective(v: CanonicalRational) -> f32 {
    (v.numerator() as f64 / v.denominator() as f64) as f32
}
fn rational(
    a: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<CanonicalRational, VNextError> {
    match a.iter().find(|(k, _)| k.as_str() == name).map(|(_, v)| v) {
        Some(SemanticValue::Rational(r)) => Ok(*r),
        _ => Err(invalid("missing rational scalar")),
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 channel L2 norm: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
