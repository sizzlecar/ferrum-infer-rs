//! Half-split BF16 rotary multiplication with explicit BF16 frequencies.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const HALF_SPLIT_ROPE_BF16_OPERATION_ID: &str = "operation.half_split_rope.bf16";
pub const HALF_SPLIT_ROPE_BF16_CAPABILITY_ID: &str = "capability.operation.half_split_rope.bf16";

/// For each head, rotate_half(x) is (-second_half, first_half), and
/// `y = B(RN32(B(RN32(x*cos)) + B(RN32(rotate_half(x)*sin))))`.
/// B denotes BF16 RNE. Both products cross a BF16 boundary before addition;
/// neither FP32-only products nor FMA implement this contract.
///
/// x/y are [rows,columns], cos/sin are [rows,head_dim], all contiguous BF16.
/// Frequencies are explicit for every row and shared across heads only. They
/// need not repeat between halves or have unit magnitude. This operation does
/// not generate positions or trigonometric values, or infer a batch layout.
/// IEEE propagation applies, without a NaN payload guarantee.
pub fn half_split_rope_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let matrix = vec![symbol("rows"), symbol("columns")];
    let frequency = vec![symbol("rows"), symbol("head_dim")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(HALF_SPLIT_ROPE_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(matrix.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(frequency.clone(), [ElementType::Bf16], TensorAccess::Read)?,
            contiguous_tensor(frequency, [ElementType::Bf16], TensorAccess::Read)?,
        ],
        outputs: vec![contiguous_tensor(
            matrix,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([unsigned_attribute("head_dim")?]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            HALF_SPLIT_ROPE_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Checked semantic dimensions, never an allocation or submission authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfSplitRopeBf16Geometry {
    rows: u64,
    columns: u64,
    head_dim: u64,
}
impl HalfSplitRopeBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        cosine: &ResolvedTensorSpec,
        sine: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        head_dim: u64,
    ) -> Result<Self, VNextError> {
        let [rows, columns] = input.dimensions() else {
            return Err(invalid("input must have rank two"));
        };
        if input != output
            || cosine != sine
            || input.element_type() != ElementType::Bf16
            || input.layout() != &ResolvedTensorLayout::Contiguous
            || cosine.element_type() != ElementType::Bf16
            || cosine.layout() != &ResolvedTensorLayout::Contiguous
            || *rows == 0
            || *columns == 0
            || head_dim == 0
            || head_dim % 2 != 0
            || *columns % head_dim != 0
            || cosine.dimensions() != [*rows, head_dim]
        {
            return Err(invalid("requires identical contiguous BF16 x/y[rows,columns], cos/sin[rows,head_dim], positive even head_dim dividing columns"));
        }
        input.minimum_storage_bytes()?;
        cosine.minimum_storage_bytes()?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            head_dim,
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
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("BF16 half-split RoPE geometry: {}", reason.into()),
    }
}
#[cfg(test)]
mod tests;
