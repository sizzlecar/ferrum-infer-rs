//! Full-F32 half-split RoPE with explicit per-row frequencies.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
pub const HALF_SPLIT_ROPE_F32_OPERATION_ID: &str = "operation.half_split_rope.f32";
pub const HALF_SPLIT_ROPE_F32_CAPABILITY_ID: &str = "capability.operation.half_split_rope.f32";
/// F32 x/y[rows,columns], cos/sin[rows,head_dim]. For each head, rotate_half(x)
/// is (-second_half,first_half). y=RN32(RN32(x*cos)+RN32(rotate_half(x)*sin)),
/// with no FMA, BF16/F16 product/store or inferred positions/trigonometry.
/// Frequencies may differ between halves and rows and are shared across heads
/// only. IEEE propagation applies, without a NaN payload preservation promise.
pub fn half_split_rope_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut descriptor = half_split_rope_bf16_contract()?.descriptor;
    descriptor.id = OperationId::new(HALF_SPLIT_ROPE_F32_OPERATION_ID)?;
    descriptor.provider = provider_requirement(
        HALF_SPLIT_ROPE_F32_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    let matrix = vec![symbol("rows"), symbol("columns")];
    let frequency = vec![symbol("rows"), symbol("head_dim")];
    descriptor.inputs = vec![
        contiguous_tensor(matrix.clone(), [ElementType::F32], TensorAccess::Read)?,
        contiguous_tensor(frequency.clone(), [ElementType::F32], TensorAccess::Read)?,
        contiguous_tensor(frequency, [ElementType::F32], TensorAccess::Read)?,
    ];
    descriptor.outputs = vec![contiguous_tensor(
        matrix,
        [ElementType::F32],
        TensorAccess::Write,
    )?];
    descriptor.resources = ResourceRequirements {
        minimum_value_alignment_bytes: 16,
        ..no_auxiliary_resources()
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalfSplitRopeF32Geometry {
    rows: u64,
    columns: u64,
    head_dim: u64,
    elements: u64,
    arithmetic: u64,
}
impl HalfSplitRopeF32Geometry {
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
            || input.element_type() != ElementType::F32
            || cosine.element_type() != ElementType::F32
            || [input, cosine]
                .iter()
                .any(|t| t.layout() != &ResolvedTensorLayout::Contiguous)
            || *rows == 0
            || *columns == 0
            || head_dim == 0
            || !head_dim.is_multiple_of(2)
            || !columns.is_multiple_of(head_dim)
            || cosine.dimensions() != [*rows, head_dim]
        {
            return Err(invalid("requires contiguous F32 x/y[rows,columns], cos/sin[rows,head_dim] with positive even head_dim dividing columns"));
        }
        for t in [input, cosine, sine, output] {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let elements = rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        let arithmetic = (elements / 2)
            .checked_mul(7)
            .ok_or_else(|| invalid("logical arithmetic overflows"))?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            head_dim,
            elements,
            arithmetic,
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
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 half-split RoPE: {reason}"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn rope_frequency_shape_and_precision_are_explicit() {
        let x = t(&[3, 24], ElementType::F32);
        let c = t(&[3, 6], ElementType::F32);
        assert_eq!(
            HalfSplitRopeF32Geometry::new(&x, &c, &c, &x, 6)
                .unwrap()
                .elements(),
            72
        );
        assert!(
            HalfSplitRopeF32Geometry::new(&x, &t(&[3, 6], ElementType::Bf16), &c, &x, 6).is_err()
        );
        assert!(
            HalfSplitRopeF32Geometry::new(&x, &t(&[1, 6], ElementType::F32), &c, &x, 6).is_err()
        );
        assert!(HalfSplitRopeF32Geometry::new(&x, &c, &c, &x, 3).is_err());
        assert_eq!(
            half_split_rope_f32_contract()
                .unwrap()
                .descriptor()
                .resources
                .scratch,
            ResourcePresenceRequirement::Forbidden
        );
    }
}
