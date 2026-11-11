//! Leading-channel half-split rotation. The original full-head operation
//! remains a separate contract with its original attributes and dimensions.

use super::*;

pub const PARTIAL_HALF_SPLIT_ROPE_BF16_OPERATION_ID: &str =
    "operation.partial_half_split_rope.bf16";
pub const PARTIAL_HALF_SPLIT_ROPE_BF16_CAPABILITY_ID: &str =
    "capability.operation.partial_half_split_rope.bf16";

/// x/y are contiguous BF16 `[rows,columns]`, partitioned into `head_dim`
/// channels per head. Only the leading even `rotary_dim` channels rotate;
/// rotate_half splits that leading block, not the whole head, in half.
/// The remaining input bits are copied without arithmetic or conversions.
///
/// Explicit cos/sin are BF16 `[rows,rotary_dim]`, shared over heads only.
/// Each product rounds to BF16 before the final BF16 sum, exactly as in the
/// full-head contract. IEEE propagation applies to rotated channels; tail
/// copies preserve every bit, including nonfinite payloads and signed zeros.
/// There is no frequency generation, padding, batch or request inference.
pub fn partial_half_split_rope_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let matrix = vec![symbol("rows"), symbol("columns")];
    let frequency = vec![symbol("rows"), symbol("rotary_dim")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(PARTIAL_HALF_SPLIT_ROPE_BF16_OPERATION_ID)?,
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
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("head_dim")?,
            unsigned_attribute("rotary_dim")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            PARTIAL_HALF_SPLIT_ROPE_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// Shape proof only; it grants no allocation, ownership or execution rights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialHalfSplitRopeBf16Geometry {
    rows: u64,
    columns: u64,
    head_dim: u64,
    rotary_dim: u64,
}

impl PartialHalfSplitRopeBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        cosine: &ResolvedTensorSpec,
        sine: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
        head_dim: u64,
        rotary_dim: u64,
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
            || rotary_dim == 0
            || rotary_dim % 2 != 0
            || rotary_dim > head_dim
            || *columns % head_dim != 0
            || cosine.dimensions() != [*rows, rotary_dim]
        {
            return Err(invalid("requires contiguous BF16 x/y[rows,columns], cos/sin[rows,rotary_dim], positive head_dim dividing columns, and positive even rotary_dim <= head_dim"));
        }
        input.minimum_storage_bytes()?;
        cosine.minimum_storage_bytes()?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            head_dim,
            rotary_dim,
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
    pub const fn rotary_dim(self) -> u64 {
        self.rotary_dim
    }
    pub const fn heads(self) -> u64 {
        self.columns / self.head_dim
    }
}

#[cfg(test)]
mod tests;
