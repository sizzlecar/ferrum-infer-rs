//! Whole-row BF16 RMS normalization with a zero-centered learned scale.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
pub const ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID: &str = "operation.zero_centered_rms_norm.bf16";
pub const ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID: &str =
    "capability.operation.zero_centered_rms_norm.bf16";
pub const ZERO_CENTERED_RMS_NORM_BF16_MAX_WIDTH: u64 = 1 << 24;

/// Input and output are BF16 [rows,columns]; the original zero-centered
/// weight is BF16 [columns] and is shared across rows. The width is inferred
/// from actual ports, not an independent attribute. Epsilon is rational.
///
/// Reduction and inverse-root follow the provider's FP32 numerical profile.
/// With R denoting RN32 and B BF16 RNE, n=R(x*inverse), g=R(weight+1),
/// y=B(R(n*g)). There is NO intermediate B on n or g, no bias and no head
/// subdivision. Preflight checks structure; nonfinite values propagate IEEE.
pub fn zero_centered_rms_norm_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("columns")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(dimensions.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("columns")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            dimensions,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([positive_epsilon_attribute("epsilon")?]))?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
/// Descriptive shape proof only, with no allocation/execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroCenteredRmsNormBf16Geometry {
    rows: u64,
    columns: u64,
    epsilon: CanonicalRational,
}
impl ZeroCenteredRmsNormBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        weight: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let g = Bf16SameShapeGeometry::new(input, output)?;
        let width = g.width();
        if width > ZERO_CENTERED_RMS_NORM_BF16_MAX_WIDTH
            || weight.dimensions() != [width]
            || weight.element_type() != ElementType::Bf16
            || weight.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid(
                "requires contiguous BF16 weight[columns], whole-row width in 1..=2^24",
            ));
        }
        let numerator = i128::from(epsilon.numerator());
        let denominator = i128::from(epsilon.denominator());
        if numerator * 1_000_000_000_000 < denominator || numerator > denominator {
            return Err(invalid("epsilon must be rational in [1e-12,1]"));
        }
        input.minimum_storage_bytes()?;
        weight.minimum_storage_bytes()?;
        Ok(Self {
            rows: g.rows(),
            columns: width,
            epsilon,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn epsilon(self) -> CanonicalRational {
        self.epsilon
    }
    pub fn effective_epsilon(self) -> f32 {
        (self.epsilon.numerator() as f64 / self.epsilon.denominator() as f64) as f32
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 zero-centered RMSNorm: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
