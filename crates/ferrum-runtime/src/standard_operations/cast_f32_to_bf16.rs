//! Explicit full-tensor IEEE F32 -> BF16 conversion.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const CAST_F32_TO_BF16_OPERATION_ID: &str = "operation.cast.f32_to_bf16";
pub const CAST_F32_TO_BF16_CAPABILITY_ID: &str = "capability.operation.cast.f32_to_bf16";

/// Contiguous F32 X[rows,width] -> BF16 Y[rows,width], round to nearest,
/// ties to even. Signed zero and infinity retain their sign; finite overflow
/// produces signed infinity; BF16 subnormals use gradual underflow. No FTZ,
/// clamping, stochastic rounding or intermediate F16 conversion is permitted.
/// NaNs remain NaN with their sign; signaling NaNs are quieted. NaN payload
/// preservation is not contractual. Output must not alias the read input.
pub fn cast_f32_to_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let shape = || {
        vec![
            DimensionConstraint::Symbol("rows".into()),
            DimensionConstraint::Symbol("width".into()),
        ]
    };
    let descriptor = OperationDescriptor {
        id: OperationId::new(CAST_F32_TO_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            shape(),
            [ElementType::F32],
            TensorAccess::Read,
        )?],
        outputs: vec![contiguous_tensor(
            shape(),
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::new())?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(CAST_F32_TO_BF16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastF32ToBf16Geometry {
    rows: u64,
    width: u64,
    elements: u64,
}
impl CastF32ToBf16Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        let [rows, width] = input.dimensions() else {
            return Err(invalid("requires rank-two input"));
        };
        if *rows == 0
            || *width == 0
            || output.dimensions() != input.dimensions()
            || input.element_type() != ElementType::F32
            || output.element_type() != ElementType::Bf16
            || [input, output]
                .iter()
                .any(|t| t.layout() != &ResolvedTensorLayout::Contiguous)
        {
            return Err(invalid(
                "requires positive contiguous F32 [rows,width] -> BF16 with identical shape",
            ));
        }
        for tensor in [input, output] {
            if tensor.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let elements = rows
            .checked_mul(*width)
            .ok_or_else(|| invalid("element count overflows"))?;
        Ok(Self {
            rows: *rows,
            width: *width,
            elements,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn width(self) -> u64 {
        self.width
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 to BF16 cast: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn cast_requires_same_shape_exact_dtypes_and_addressable_capacity() {
        let x = tensor(&[2, 10], ElementType::F32);
        let y = tensor(&[2, 10], ElementType::Bf16);
        let g = CastF32ToBf16Geometry::new(&x, &y).unwrap();
        assert_eq!((g.rows(), g.width(), g.elements()), (2, 10, 20));
        for bad in [
            tensor(&[1, 20], ElementType::Bf16),
            tensor(&[2, 10], ElementType::F16),
            tensor(&[2, 10], ElementType::F32),
        ] {
            assert!(CastF32ToBf16Geometry::new(&x, &bad).is_err());
        }
        assert!(CastF32ToBf16Geometry::new(&y, &y).is_err());
        assert!(CastF32ToBf16Geometry::new(
            &tensor(&[20], ElementType::F32),
            &tensor(&[20], ElementType::Bf16)
        )
        .is_err());
        let huge = (isize::MAX as u64) / 4 + 1;
        assert!(CastF32ToBf16Geometry::new(
            &tensor(&[1, huge], ElementType::F32),
            &tensor(&[1, huge], ElementType::Bf16)
        )
        .is_err());
        let contract = cast_f32_to_bf16_contract().unwrap();
        assert_eq!(
            contract.descriptor().resources.scratch,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            contract.descriptor().resources.binding,
            ResourcePresenceRequirement::Forbidden
        );
    }
}
