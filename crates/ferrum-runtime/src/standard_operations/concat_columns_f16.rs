use super::*;

pub const CONCAT_COLUMNS_F16_OPERATION_ID: &str = "operation.concat_columns.f16";
pub const CONCAT_COLUMNS_F16_CAPABILITY_ID: &str = "capability.operation.concat_columns.f16";

/// Row-wise bit-preserving concatenation, left columns followed by right.
/// Providers must validate `out_features == left_features + right_features`
/// with checked arithmetic before accessing any input or output. This is a
/// transport operation: it performs no floating-point conversion or arithmetic.
pub fn concat_columns_f16_contract() -> Result<StandardOperationContract, VNextError> {
    let version = ContractVersion::new(1, 0);
    let tensor = |width, access| {
        contiguous_tensor(
            vec![symbol("tokens"), symbol(width)],
            [ElementType::F16],
            access,
        )
    };
    let descriptor = OperationDescriptor {
        id: OperationId::new(CONCAT_COLUMNS_F16_OPERATION_ID)?,
        version,
        inputs: vec![
            tensor("left_features", TensorAccess::Read)?,
            tensor("right_features", TensorAccess::Read)?,
        ],
        outputs: vec![tensor("out_features", TensorAccess::Write)?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("left_features")?,
            unsigned_attribute("right_features")?,
            unsigned_attribute("out_features")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(CONCAT_COLUMNS_F16_CAPABILITY_ID, version)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
