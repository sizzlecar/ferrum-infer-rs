//! Full-F32 pointwise boundaries. Provider profiles qualify exp separately;
//! neither operation includes a BF16 conversion, rotation or quantization.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const PLAIN_RESIDUAL_ADD_F32_OPERATION_ID: &str = "operation.plain_residual_add.f32";
pub const PLAIN_RESIDUAL_ADD_F32_CAPABILITY_ID: &str =
    "capability.operation.plain_residual_add.f32";
pub const SILU_MUL_F32_OPERATION_ID: &str = "operation.silu_mul.f32";
pub const SILU_MUL_F32_CAPABILITY_ID: &str = "capability.operation.silu_mul.f32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32PointwiseOp {
    ResidualAdd,
    SiluMul,
}
impl F32PointwiseOp {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::ResidualAdd => PLAIN_RESIDUAL_ADD_F32_OPERATION_ID,
            Self::SiluMul => SILU_MUL_F32_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::ResidualAdd => PLAIN_RESIDUAL_ADD_F32_CAPABILITY_ID,
            Self::SiluMul => SILU_MUL_F32_CAPABILITY_ID,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        contract(self.operation_id(), self.capability_id())
    }
}

/// RN32(x + update), with identical F32 matrix shapes and no broadcast.
pub fn plain_residual_add_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(
        PLAIN_RESIDUAL_ADD_F32_OPERATION_ID,
        PLAIN_RESIDUAL_ADD_F32_CAPABILITY_ID,
    )
}
/// e=exp_profile(-gate); d=RN32(1+e); s=RN32(gate/d);
/// output=RN32(s*up). Each intermediate remains F32, including SiLU before
/// multiplication. This explicit expression retains IEEE overflow behavior;
/// a stable sigmoid rewrite or a fused downstream quantizer is a different op.
pub fn silu_mul_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(SILU_MUL_F32_OPERATION_ID, SILU_MUL_F32_CAPABILITY_ID)
}
fn contract(operation: &str, capability: &str) -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("width")];
    let input = contiguous_tensor(dimensions.clone(), [ElementType::F32], TensorAccess::Read)?;
    let descriptor = OperationDescriptor {
        id: OperationId::new(operation)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![input.clone(), input],
        outputs: vec![contiguous_tensor(
            dimensions,
            [ElementType::F32],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(capability, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32SameShapeGeometry {
    elements: u64,
}
impl F32SameShapeGeometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        update: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        if input.dimensions().len() != 2
            || input.dimensions().contains(&0)
            || input.element_type() != ElementType::F32
            || input.layout() != &ResolvedTensorLayout::Contiguous
            || input != update
            || input != output
        {
            return Err(invalid(
                "requires three identical nonempty contiguous F32 matrices",
            ));
        }
        let bytes = input.minimum_storage_bytes()?;
        if bytes > isize::MAX as u64 {
            return Err(invalid("byte extent exceeds Host address range"));
        }
        Ok(Self {
            elements: bytes / 4,
        })
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 same-shape pointwise: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn all_three_shapes_and_dtypes_are_checked_without_implicit_broadcast() {
        let input = spec(&[3, 7], ElementType::F32);
        assert_eq!(
            F32SameShapeGeometry::new(&input, &input, &input)
                .unwrap()
                .elements(),
            21
        );
        for wrong in [
            spec(&[1, 7], ElementType::F32),
            spec(&[3, 7], ElementType::Bf16),
            spec(&[21], ElementType::F32),
        ] {
            assert!(F32SameShapeGeometry::new(&input, &wrong, &input).is_err());
            assert!(F32SameShapeGeometry::new(&input, &input, &wrong).is_err());
        }
    }
}
