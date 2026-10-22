//! Adjacent-pair BF16 rotary multiplication by explicit FP32 frequencies.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const ADJACENT_ROPE_BF16_OPERATION_ID: &str = "operation.adjacent_rope.bf16";
pub const ADJACENT_ROPE_BF16_CAPABILITY_ID: &str = "capability.operation.adjacent_rope.bf16";

/// For adjacent BF16 values (a,b) and supplied FP32 frequency (c,s), use
/// real=B(FMA_RN(a,c,RN32((-b)*s))) and imag=B(FMA_RN(b,c,RN32(a*s))).
/// B is BF16 RNE. The explicit fused operation is part of this version's
/// rounding contract; independent multiplication and addition are different.
/// Frequencies are shared across heads and batches, indexed by row % sequence.
/// No frequency synthesis, unit-magnitude constraint, or position convention
/// is inferred by this operation.
pub fn adjacent_rope_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("columns")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(ADJACENT_ROPE_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(dimensions.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(
                vec![symbol("sequence"), symbol("pairs"), exact(2)],
                [ElementType::F32],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            dimensions,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([unsigned_attribute("head_dim")?]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            ADJACENT_ROPE_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Checked semantic dimensions, not allocation or submission authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdjacentRopeBf16Geometry {
    rows: u64,
    columns: u64,
    sequence: u64,
    head_dim: u64,
}
impl AdjacentRopeBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        frequencies: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        head_dim: u64,
    ) -> Result<Self, VNextError> {
        let [rows, columns] = input.dimensions() else {
            return Err(invalid("input must have rank two"));
        };
        let [sequence, pairs, components] = frequencies.dimensions() else {
            return Err(invalid("frequencies must have rank three"));
        };
        if input != output
            || input.element_type() != ElementType::Bf16
            || input.layout() != &ResolvedTensorLayout::Contiguous
            || frequencies.element_type() != ElementType::F32
            || frequencies.layout() != &ResolvedTensorLayout::Contiguous
            || *rows == 0
            || *columns == 0
            || *sequence == 0
            || head_dim == 0
            || head_dim % 2 != 0
            || *rows % *sequence != 0
            || *columns % head_dim != 0
            || *pairs != head_dim / 2
            || *components != 2
        {
            return Err(invalid("requires identical contiguous BF16 x/y[rows,columns], contiguous F32 frequencies[sequence,head_dim/2,2], positive even head_dim, rows divisible by sequence and columns divisible by head_dim"));
        }
        input.minimum_storage_bytes()?;
        frequencies.minimum_storage_bytes()?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            sequence: *sequence,
            head_dim,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
    pub const fn batch(self) -> u64 {
        self.rows / self.sequence
    }
    pub const fn heads(self) -> u64 {
        self.columns / self.head_dim
    }
    pub const fn head_dim(self) -> u64 {
        self.head_dim
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 adjacent RoPE geometry: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
