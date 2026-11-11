//! Distinct full-F32 GELU contracts. The selected provider qualifies the
//! transcendental implementation; no stage rounds through BF16 or F16.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const GELU_TANH_F32_OPERATION_ID: &str = "operation.gelu_tanh.f32";
pub const GELU_TANH_F32_CAPABILITY_ID: &str = "capability.operation.gelu_tanh.f32";
pub const GELU_ERF_F32_OPERATION_ID: &str = "operation.gelu_erf.f32";
pub const GELU_ERF_F32_CAPABILITY_ID: &str = "capability.operation.gelu_erf.f32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeluF32Variant {
    Tanh,
    Erf,
}
impl GeluF32Variant {
    pub const fn operation_id(self) -> &'static str {
        match self {
            Self::Tanh => GELU_TANH_F32_OPERATION_ID,
            Self::Erf => GELU_ERF_F32_OPERATION_ID,
        }
    }
    pub const fn capability_id(self) -> &'static str {
        match self {
            Self::Tanh => GELU_TANH_F32_CAPABILITY_ID,
            Self::Erf => GELU_ERF_F32_CAPABILITY_ID,
        }
    }
    /// Scalar arithmetic plus one transcendental evaluation, not hardware FLOPs.
    pub const fn logical_operations_per_element(self) -> u64 {
        match self {
            Self::Tanh => 9,
            Self::Erf => 5,
        }
    }
    pub fn contract(self) -> Result<StandardOperationContract, VNextError> {
        contract(self)
    }
}

/// Let R=RN32, beta=F32 bits 0x3f4c422a and kappa=0x3d372713.
/// square=R(x*x), cube=R(square*x), inner=R(beta*R(x+R(kappa*cube))),
/// t=tanh_profile(inner), y=R(R(0.5*x)*R(1+t)). All stages remain F32.
/// There is no FMA contraction, saturation shortcut or sign-dependent rewrite.
pub fn gelu_tanh_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(GeluF32Variant::Tanh)
}
/// Let R=RN32 and inv_sqrt_two=F32 bits 0x3f3504f3.
/// inner=R(x*inv_sqrt_two), t=erf_profile(inner),
/// y=R(R(0.5*x)*R(1+t)). This preserves finite-negative cancellation to -0,
/// negative-infinity NaN and gradual underflow. No stable-tail rewrite.
pub fn gelu_erf_f32_contract() -> Result<StandardOperationContract, VNextError> {
    contract(GeluF32Variant::Erf)
}
fn contract(variant: GeluF32Variant) -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("width")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(variant.operation_id())?,
        version: ContractVersion::new(1, 0),
        inputs: vec![contiguous_tensor(
            dimensions.clone(),
            [ElementType::F32],
            TensorAccess::Read,
        )?],
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
        provider: provider_requirement(variant.capability_id(), ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeluF32Geometry {
    elements: u64,
}
impl GeluF32Geometry {
    pub fn new(
        input: &ResolvedTensorSpec,
        output: &ResolvedTensorSpec,
    ) -> Result<Self, VNextError> {
        if input.dimensions().len() != 2
            || input.dimensions().contains(&0)
            || input.element_type() != ElementType::F32
            || input.layout() != &ResolvedTensorLayout::Contiguous
            || input != output
        {
            return Err(invalid(
                "requires identical nonempty contiguous F32 matrices",
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
    pub fn logical_operations(self, variant: GeluF32Variant) -> Result<u64, VNextError> {
        self.elements
            .checked_mul(variant.logical_operations_per_element())
            .ok_or_else(|| invalid("logical operation count overflows"))
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 GELU: {reason}"),
    }
}

#[cfg(test)]
mod tests;
