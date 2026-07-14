use std::collections::{BTreeMap, BTreeSet};

use super::{
    AliasPolicy, AttributeConstraint, AttributeId, AttributeSchema, AttributeSpec,
    AttributeValueKind, CapabilityId, ContractVersion, DimensionConstraint, ElementType,
    LayoutConstraint, OperationContract, OperationDescriptor, OperationId, OracleSpec,
    ProfilePhase, ProviderRequirement, ResourcePresenceRequirement, ResourceRequirements,
    TensorAccess, TensorContract, VNextError,
};

pub const TOKEN_EMBEDDING_OPERATION_ID: &str = "operation.token_embedding";
pub const TOKEN_EMBEDDING_F16_CAPABILITY_ID: &str = "capability.operation.token_embedding.f16";

/// One checked-in standard operation contract. Construction stays private so
/// production registries cannot mutate a descriptor after a provider binds its
/// fingerprint.
pub struct StandardOperationContract {
    descriptor: OperationDescriptor,
}

impl OperationContract for StandardOperationContract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.descriptor
    }

    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        if inputs != self.descriptor.inputs || outputs != self.descriptor.outputs {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "operation `{}` signature differs from its standard contract",
                    self.descriptor.id
                ),
            });
        }
        Ok(())
    }
}

pub fn token_embedding_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(TOKEN_EMBEDDING_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![DimensionConstraint::Symbol("tokens".to_owned())],
                [ElementType::U32],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    DimensionConstraint::Symbol("vocab_size".to_owned()),
                    DimensionConstraint::Symbol("hidden_size".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![
                DimensionConstraint::Symbol("tokens".to_owned()),
                DimensionConstraint::Symbol("hidden_size".to_owned()),
            ],
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("vocab_size")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: OracleSpec::Exact,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::from([CapabilityId::new(
                TOKEN_EMBEDDING_F16_CAPABILITY_ID,
            )?]),
        },
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

fn contiguous_tensor(
    dimensions: Vec<DimensionConstraint>,
    element_types: impl IntoIterator<Item = ElementType>,
    access: TensorAccess,
) -> Result<TensorContract, VNextError> {
    TensorContract::new(
        dimensions,
        element_types.into_iter().collect(),
        vec![LayoutConstraint::Contiguous],
        access,
        AliasPolicy::NoAlias,
    )
}

fn unsigned_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Unsigned,
            required: true,
            constraint: AttributeConstraint::UnsignedRange {
                minimum: 1,
                maximum: u32::MAX as u64,
            },
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_embedding_contract_is_backend_and_model_neutral() {
        let contract = token_embedding_contract().unwrap();
        let descriptor = contract.descriptor();
        assert_eq!(descriptor.id.as_str(), TOKEN_EMBEDDING_OPERATION_ID);
        assert!(!serde_json::to_string(descriptor)
            .unwrap()
            .to_ascii_lowercase()
            .contains("qwen"));
        assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
    }
}
