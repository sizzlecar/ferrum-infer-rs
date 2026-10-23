//! Unary BF16 GELU with tanh approximation and a single BF16 boundary.
use super::*;
pub const GELU_TANH_BF16_OPERATION_ID: &str = "operation.gelu_tanh.bf16";
pub const GELU_TANH_BF16_CAPABILITY_ID: &str = "capability.operation.gelu_tanh.bf16";
/// Let R=RN32, B=BF16 RNE, beta=FP32 bits 0x3f4c422a and
/// kappa=FP32 bits 0x3d372713. Compute xx=R(x*x), cube=R(xx*x),
/// inner=R(beta*R(x+R(kappa*cube))), t=tanh_profile(inner),
/// y=B(R(R(0.5*x)*R(1+t))). No FMA contraction or intermediate BF16.
///
/// The provider declares its FP32 tanh implementation and qualification scope.
/// Tanh sees general FP32 inputs, so BF16-input tanh qualification alone does
/// not qualify this operation. IEEE overflow/NaN/signed-zero propagation is
/// retained; this is not an ideal-real GELU or stable-saturation rewrite.
pub fn gelu_tanh_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("width")];
    let descriptor = OperationDescriptor {
        id: OperationId::new(GELU_TANH_BF16_OPERATION_ID)?,
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
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            scratch: ResourcePresenceRequirement::Optional,
            ..no_auxiliary_resources()
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(GELU_TANH_BF16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
#[cfg(test)]
mod tests;
