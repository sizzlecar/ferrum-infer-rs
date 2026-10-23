//! Last-axis BF16 LayerNorm without affine weights. This contract is distinct
//! from RMSNorm and from a provider's particular floating-point reduction tree.

use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const LAYER_NORM_BF16_OPERATION_ID: &str = "operation.layer_norm.bf16";
pub const LAYER_NORM_BF16_CAPABILITY_ID: &str = "capability.operation.layer_norm.bf16";
/// Every supported width is exactly representable as an FP32 divisor.
pub const LAYER_NORM_BF16_MAX_HIDDEN_SIZE: u64 = 1 << 24;

/// Population LayerNorm over each complete row, with no gamma or beta.
///
/// The mathematical parameter is the effective FP32 epsilon obtained by
/// dividing the canonical rational in F64 and then converting to F32. Inputs
/// and outputs have BF16 storage; providers declare and validate their FP32
/// arithmetic and final BF16 round-to-nearest-even behavior. Version 1.0 uses
/// an operation-defined numerical oracle, not RMSNorm's relative tolerance.
pub fn layer_norm_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("hidden_size")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(LAYER_NORM_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            dimensions.clone(),
            [ElementType::Bf16],
            TensorAccess::Read,
        )?],
        outputs: vec![contiguous_tensor(
            dimensions,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            (
                AttributeId::new("hidden_size")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Unsigned,
                    required: true,
                    constraint: AttributeConstraint::UnsignedRange {
                        minimum: 1,
                        maximum: LAYER_NORM_BF16_MAX_HIDDEN_SIZE,
                    },
                },
            ),
            positive_epsilon_attribute("epsilon")?,
        ]))?,
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(LAYER_NORM_BF16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Checked semantic geometry, never an allocation, invocation, or submission
/// authority. Providers use it during estimation and encoding to relate the
/// `hidden_size` attribute to actual tensor dimensions; matching symbol and
/// attribute names alone do not establish that relationship in core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerNormBf16Geometry {
    rows: u64,
    hidden_size: u64,
    epsilon: CanonicalRational,
}

impl LayerNormBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        hidden_size: u64,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let [rows, width] = input.dimensions() else {
            return Err(invalid_geometry("input must have rank two"));
        };
        if input != output
            || input.element_type() != ElementType::Bf16
            || input.layout() != &ResolvedTensorLayout::Contiguous
            || *rows == 0
            || *width != hidden_size
            || !(1..=LAYER_NORM_BF16_MAX_HIDDEN_SIZE).contains(&hidden_size)
        {
            return Err(invalid_geometry(
                "input/output must be identical contiguous BF16 [rows, hidden_size] tensors with a supported width",
            ));
        }
        // Exact integer comparisons, identical to the descriptor's rational
        // interval. Both products fit i128 for CanonicalRational's i64/u64 ABI.
        let numerator = i128::from(epsilon.numerator());
        let denominator = i128::from(epsilon.denominator());
        if numerator * 1_000_000_000_000 < denominator || numerator > denominator {
            return Err(invalid_geometry("epsilon must be in [1e-12, 1]"));
        }
        Ok(Self {
            rows: *rows,
            hidden_size,
            epsilon,
        })
    }

    pub const fn rows(self) -> u64 {
        self.rows
    }

    pub const fn hidden_size(self) -> u64 {
        self.hidden_size
    }

    pub const fn epsilon(self) -> CanonicalRational {
        self.epsilon
    }

    pub fn effective_epsilon(self) -> f32 {
        (self.epsilon.numerator() as f64 / self.epsilon.denominator() as f64) as f32
    }
}

fn invalid_geometry(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 LayerNorm geometry: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
