//! Same-shape BF16 residual addition, without a synthetic gate input.
use super::*;

pub const PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID: &str = "operation.plain_residual_add.bf16";
pub const PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID: &str =
    "capability.operation.plain_residual_add.bf16";

/// `y = B(RN32(x + update))`, where B is BF16 round-to-nearest-even.
/// Both read operands and the output have identical contiguous [rows,width]
/// BF16 specifications. No broadcasting, gating, or in-place output is implied.
/// IEEE NaN/infinity/signed-zero propagation applies; NaN payload bits are not
/// specified. Providers validate structure, not values produced by prior nodes.
pub fn plain_residual_add_bf16_contract() -> Result<StandardOperationContract, VNextError> {
    let dimensions = vec![symbol("rows"), symbol("width")];
    let input = contiguous_tensor(dimensions.clone(), [ElementType::Bf16], TensorAccess::Read)?;
    let descriptor = OperationDescriptor {
        id: OperationId::new(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![input; 2],
        outputs: vec![contiguous_tensor(
            dimensions,
            [ElementType::Bf16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::empty(),
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(
            PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID,
            ContractVersion::new(1, 0),
        )?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[cfg(test)]
mod tests;
