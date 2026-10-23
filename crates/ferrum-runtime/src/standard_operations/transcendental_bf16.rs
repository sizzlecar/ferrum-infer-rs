//! Same-shape BF16 activations with explicit FP32 and BF16 boundaries.
//!
//! Transcendental approximation belongs to a declared, qualified provider
//! numerical profile. OperationDefined does not assert correct rounding of
//! exp/tanh or bitwise agreement between providers.

use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const SILU_BF16_OPERATION_ID: &str = "operation.silu.bf16";
pub const SILU_BF16_CAPABILITY_ID: &str = "capability.operation.silu.bf16";
pub const SILU_MUL_BF16_OPERATION_ID: &str = "operation.silu_mul.bf16";
pub const SILU_MUL_BF16_CAPABILITY_ID: &str = "capability.operation.silu_mul.bf16";
pub const TANH_BF16_OPERATION_ID: &str = "operation.tanh.bf16";
pub const TANH_BF16_CAPABILITY_ID: &str = "capability.operation.tanh.bf16";

/// Unary SiLU: `e = exp_profile(-x)`, `d = RN32(1 + e)`,
/// `y = B(RN32(x / d))`, where B is BF16 round-to-nearest-even.
///
/// The exponential returns FP32 under the provider's numerical profile.
/// The explicit FP32 overflow and rounding boundaries preclude substituting
/// an ideal-real or stable-sigmoid expression. The unary operation needs only
/// its activation input; it does not require an all-one multiplier tensor.
pub fn silu_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    transcendental_bf16_contract(SILU_BF16_OPERATION_ID, SILU_BF16_CAPABILITY_ID, 1)
}

/// SiLU followed by multiplication: `y = B(RN32(decode(silu(g)) * u))`.
///
/// Input ordinals are gate `g`, then update `u`, with exactly the same shape.
/// `silu(g)` includes the unary contract's BF16 rounding before the FP32
/// product. Fusion must preserve that boundary. No broadcasting is supported.
pub fn silu_mul_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    transcendental_bf16_contract(SILU_MUL_BF16_OPERATION_ID, SILU_MUL_BF16_CAPABILITY_ID, 2)
}

/// Unary tanh: `y = B(tanh_profile(x))`.
///
/// The provider's declared numerical profile determines the FP32 tanh
/// approximation; B is BF16 round-to-nearest-even. This is a separate
/// capability from SiLU and does not apply a residual or multiplier.
pub fn tanh_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    transcendental_bf16_contract(TANH_BF16_OPERATION_ID, TANH_BF16_CAPABILITY_ID, 1)
}

fn transcendental_bf16_contract(
    operation_id: &str,
    capability_id: &str,
    input_count: usize,
) -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("width")];
    let input = contiguous_tensor(dimensions.clone(), [ElementType::Bf16], TensorAccess::Read)?;
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation_id)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![input; input_count],
        outputs: vec![contiguous_tensor(
            dimensions,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(capability_id, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Checked semantic geometry, not an allocation or invocation authority.
///
/// Providers use this during resource estimation and encoding. A binary
/// same-shape provider must also require the second input's complete tensor
/// specification to equal the first input's. Physical aliases and storage
/// ownership remain subject to the operation and runtime binding contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16SameShapeGeometry {
    rows: u64,
    width: u64,
}

impl Bf16SameShapeGeometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let [rows, width] = input.dimensions() else {
            return Err(invalid_geometry("input must have rank two"));
        };
        if input != output
            || input.element_type() != ElementType::Bf16
            || input.layout() != &ResolvedTensorLayout::Contiguous
            || *rows == 0
            || *width == 0
        {
            return Err(invalid_geometry(
                "input/output must be identical contiguous BF16 [rows, width] tensors",
            ));
        }
        // ResolvedTensorSpec already validates the checked byte extent. These
        // elementwise operations do not convert the width into a FP32 divisor
        // and therefore need no LayerNorm-style 2^24 width restriction.
        Ok(Self {
            rows: *rows,
            width: *width,
        })
    }

    pub const fn rows(self) -> u64 {
        self.rows
    }

    pub const fn width(self) -> u64 {
        self.width
    }
}

fn invalid_geometry(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 same-shape geometry: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
