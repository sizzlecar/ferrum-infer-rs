use std::collections::{BTreeMap, BTreeSet};

use super::{
    AliasPolicy, AttributeConstraint, AttributeId, AttributeSchema, AttributeSpec,
    AttributeValueKind, CanonicalRational, CapabilityId, ContractVersion, DimensionConstraint,
    ElementType, LayoutConstraint, OperationContract, OperationDescriptor, OperationId, OracleSpec,
    ProfilePhase, ProviderRequirement, ResourcePresenceRequirement, ResourceRequirements,
    TensorAccess, TensorContract, VNextError,
};

pub const TOKEN_EMBEDDING_OPERATION_ID: &str = "operation.token_embedding";
pub const TOKEN_EMBEDDING_F16_CAPABILITY_ID: &str = "capability.operation.token_embedding.f16";
pub const RMS_NORM_OPERATION_ID: &str = "operation.rms_norm";
pub const RMS_NORM_F16_CAPABILITY_ID: &str = "capability.operation.rms_norm.f16";
pub const DENSE_LINEAR_OPERATION_ID: &str = "operation.dense_linear";
pub const DENSE_LINEAR_F16_CAPABILITY_ID: &str = "capability.operation.dense_linear.f16";
pub const DENSE_SWIGLU_OPERATION_ID: &str = "operation.dense_swiglu";
pub const DENSE_SWIGLU_F16_CAPABILITY_ID: &str = "capability.operation.dense_swiglu.f16";
pub const RESIDUAL_ADD_OPERATION_ID: &str = "operation.residual_add";
pub const RESIDUAL_ADD_F16_CAPABILITY_ID: &str = "capability.operation.residual_add.f16";

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

pub fn rms_norm_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(RMS_NORM_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![DimensionConstraint::Symbol("hidden_size".to_owned())],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            positive_epsilon_attribute("epsilon")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(RMS_NORM_F16_CAPABILITY_ID)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn dense_linear_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(DENSE_LINEAR_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                vec![
                    DimensionConstraint::Symbol("rows".to_owned()),
                    DimensionConstraint::Symbol("in_features".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    DimensionConstraint::Symbol("out_features".to_owned()),
                    DimensionConstraint::Symbol("in_features".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            vec![
                DimensionConstraint::Symbol("rows".to_owned()),
                DimensionConstraint::Symbol("out_features".to_owned()),
            ],
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("in_features")?,
            unsigned_attribute("out_features")?,
        ]))?,
        resources: no_auxiliary_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(DENSE_LINEAR_F16_CAPABILITY_ID)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn dense_swiglu_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(DENSE_SWIGLU_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                intermediate_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                intermediate_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                hidden_intermediate_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("intermediate_size")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(DENSE_SWIGLU_F16_CAPABILITY_ID)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

pub fn residual_add_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(RESIDUAL_ADD_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor_with_alias(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
            AliasPolicy::MayAlias { tensor_index: 0 },
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([unsigned_attribute("hidden_size")?]))?,
        resources: no_auxiliary_resources(),
        oracle: f16_reference_tolerance()?,
        provider: provider_requirement(RESIDUAL_ADD_F16_CAPABILITY_ID)?,
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
    contiguous_tensor_with_alias(dimensions, element_types, access, AliasPolicy::NoAlias)
}

fn contiguous_tensor_with_alias(
    dimensions: Vec<DimensionConstraint>,
    element_types: impl IntoIterator<Item = ElementType>,
    access: TensorAccess,
    alias: AliasPolicy,
) -> Result<TensorContract, VNextError> {
    TensorContract::new(
        dimensions,
        element_types.into_iter().collect(),
        vec![LayoutConstraint::Contiguous],
        access,
        alias,
    )
}

fn token_hidden_dimensions() -> Vec<DimensionConstraint> {
    vec![
        DimensionConstraint::Symbol("tokens".to_owned()),
        DimensionConstraint::Symbol("hidden_size".to_owned()),
    ]
}

fn intermediate_hidden_dimensions() -> Vec<DimensionConstraint> {
    vec![
        DimensionConstraint::Symbol("intermediate_size".to_owned()),
        DimensionConstraint::Symbol("hidden_size".to_owned()),
    ]
}

fn hidden_intermediate_dimensions() -> Vec<DimensionConstraint> {
    vec![
        DimensionConstraint::Symbol("hidden_size".to_owned()),
        DimensionConstraint::Symbol("intermediate_size".to_owned()),
    ]
}

fn no_auxiliary_resources() -> ResourceRequirements {
    ResourceRequirements {
        minimum_value_alignment_bytes: 16,
        scratch: ResourcePresenceRequirement::Forbidden,
        persistent: ResourcePresenceRequirement::Forbidden,
    }
}

fn provider_requirement(capability: &str) -> Result<ProviderRequirement, VNextError> {
    Ok(ProviderRequirement {
        minimum_version: ContractVersion::new(1, 0),
        required_capabilities: BTreeSet::from([CapabilityId::new(capability)?]),
    })
}

fn f16_reference_tolerance() -> Result<OracleSpec, VNextError> {
    Ok(OracleSpec::RelativeTolerance {
        tolerance: CanonicalRational::new(1, 1_000)?,
    })
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

fn positive_epsilon_attribute(name: &str) -> Result<(AttributeId, AttributeSpec), VNextError> {
    Ok((
        AttributeId::new(name)?,
        AttributeSpec {
            value_kind: AttributeValueKind::Rational,
            required: true,
            constraint: AttributeConstraint::RationalRange {
                minimum: CanonicalRational::new(1, 1_000_000_000_000)?,
                maximum: CanonicalRational::new(1, 1)?,
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

    #[test]
    fn transformer_primitives_have_explicit_math_and_resource_boundaries() {
        let contracts = [
            rms_norm_contract().unwrap(),
            dense_linear_contract().unwrap(),
            dense_swiglu_contract().unwrap(),
            residual_add_contract().unwrap(),
        ];
        for contract in &contracts {
            let descriptor = contract.descriptor();
            assert!(!serde_json::to_string(descriptor)
                .unwrap()
                .to_ascii_lowercase()
                .contains("qwen"));
            assert_eq!(descriptor.fingerprint().unwrap().len(), 64);
            contract
                .validate_signature(&descriptor.inputs, &descriptor.outputs)
                .unwrap();
        }
        assert_eq!(
            contracts[2].descriptor().resources.scratch,
            ResourcePresenceRequirement::Required
        );
        assert_eq!(
            contracts[3].descriptor().outputs[0].alias(),
            &AliasPolicy::MayAlias { tensor_index: 0 }
        );
    }
}
