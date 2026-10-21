//! Independent head-wise BF16 RMS normalization with a shared BF16 gamma.

use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const HEAD_RMS_NORM_BF16_OPERATION_ID: &str = "operation.head_rms_norm.bf16";
pub const HEAD_RMS_NORM_BF16_CAPABILITY_ID: &str = "capability.operation.head_rms_norm.bf16";
/// The divisor is exactly representable in FP32 throughout this domain.
pub const HEAD_RMS_NORM_BF16_MAX_HEAD_DIM: u64 = 1 << 24;

/// For every contiguous D-element head, compute RMS normalization in FP32,
/// round its result to BF16, multiply by gamma in FP32, then round to BF16.
/// Gamma has D elements and is shared across heads and rows. There is no
/// centering, bias, gamma offset, implicit view, or token geometry.
///
/// Epsilon is the canonical rational divided in F64 and converted to F32.
/// Providers declare their reduction and inverse-square-root arithmetic;
/// OperationDefined does not make different reduction orders bitwise equal.
pub fn head_rms_norm_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("columns")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(HEAD_RMS_NORM_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(dimensions.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("head_dim")],
                [ElementType::Bf16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            dimensions,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            (
                AttributeId::new("head_dim")?,
                AttributeSpec {
                    value_kind: AttributeValueKind::Unsigned,
                    required: true,
                    constraint: AttributeConstraint::UnsignedRange {
                        minimum: 1,
                        maximum: HEAD_RMS_NORM_BF16_MAX_HEAD_DIM,
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
        provider: provider_requirement(
            HEAD_RMS_NORM_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Semantic shape proof only; this does not grant storage or execution rights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadRmsNormBf16Geometry {
    rows: u64,
    columns: u64,
    head_dim: u64,
    epsilon: CanonicalRational,
}

impl HeadRmsNormBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        head_dim: u64,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let [rows, columns] = input.dimensions() else {
            return Err(invalid_geometry("input must have rank two"));
        };
        if input != output
            || input.element_type() != ElementType::Bf16
            || input.layout() != &ResolvedTensorLayout::Contiguous
            || *rows == 0
            || !(1..=HEAD_RMS_NORM_BF16_MAX_HEAD_DIM).contains(&head_dim)
            || *columns == 0
            || *columns % head_dim != 0
            || gamma.dimensions() != [head_dim]
            || gamma.element_type() != ElementType::Bf16
            || gamma.layout() != &ResolvedTensorLayout::Contiguous
        {
            return Err(invalid_geometry(
                "requires identical contiguous BF16 [rows, columns] input/output, contiguous BF16 gamma[head_dim], and columns divisible by head_dim in 1..=2^24",
            ));
        }
        let numerator = i128::from(epsilon.numerator());
        let denominator = i128::from(epsilon.denominator());
        if numerator * 1_000_000_000_000 < denominator || numerator > denominator {
            return Err(invalid_geometry("epsilon must be in [1e-12, 1]"));
        }
        // ResolvedTensorSpec also validates extents; keep the contract's
        // complete storage requirement explicit at this semantic boundary.
        input.minimum_storage_bytes()?;
        gamma.minimum_storage_bytes()?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            head_dim,
            epsilon,
        })
    }

    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn head_dim(self) -> u64 {
        self.head_dim
    }
    pub const fn heads(self) -> u64 {
        self.columns / self.head_dim
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
        reason: format!("BF16 head RMSNorm geometry: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
