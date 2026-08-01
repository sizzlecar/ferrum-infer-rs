use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::super::{
    CanonicalRational, CapabilityId, ContractVersion, OperationId, SemanticValue, VNextError,
};
use super::foundation::invalid_operation;
use super::{
    AliasPolicy, AttributeId, AttributeSchema, DimensionConstraint, ElementType, LayoutConstraint,
    ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueBinding, ResolvedValueRole,
    ResolvedValueStorage, StrideConstraint, TensorAccess, TensorContract,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePresenceRequirement {
    Forbidden,
    Optional,
    Required,
}

impl ResourcePresenceRequirement {
    pub const fn accepts(self, present: bool) -> bool {
        matches!(
            (self, present),
            (Self::Forbidden, false) | (Self::Optional, _) | (Self::Required, true)
        )
    }
}

/// Shape-independent resource contract. Concrete byte counts, scopes, and
/// alignment are produced by the selected provider's versioned estimator and
/// bound into the immutable execution plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceRequirements {
    pub minimum_value_alignment_bytes: u64,
    pub scratch: ResourcePresenceRequirement,
    /// Small request-shaped control workspace whose contents are written in
    /// the wave binding preamble and consumed by reusable compute.
    pub binding: ResourcePresenceRequirement,
    pub persistent: ResourcePresenceRequirement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleSpec {
    Exact,
    AbsoluteTolerance {
        tolerance: CanonicalRational,
    },
    RelativeTolerance {
        tolerance: CanonicalRational,
    },
    ReferenceOperation {
        operation_id: OperationId,
        version: ContractVersion,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfilePhase {
    Load,
    Prepare,
    /// Backend operation shared by prefill and decode. The exact request phase
    /// is derived from the bound work shape rather than changing operation
    /// identity or selecting another provider in the hot path.
    Forward,
    Prefill,
    Decode,
    Transfer,
    Synchronize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderRequirement {
    pub minimum_version: ContractVersion,
    pub required_capabilities: BTreeSet<CapabilityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationDescriptor {
    pub id: OperationId,
    pub version: ContractVersion,
    pub inputs: Vec<TensorContract>,
    pub outputs: Vec<TensorContract>,
    pub attributes: AttributeSchema,
    pub resources: ResourceRequirements,
    pub oracle: OracleSpec,
    pub provider: ProviderRequirement,
    pub profile_phase: ProfilePhase,
}

impl OperationDescriptor {
    pub fn validate(&self) -> Result<(), VNextError> {
        if self.version.major == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("operation `{}` has an unstable zero major version", self.id),
            });
        }
        if self.outputs.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("operation `{}` has no outputs", self.id),
            });
        }
        for (index, input) in self.inputs.iter().enumerate() {
            input.validate(&format!("operation.{}.inputs[{index}]", self.id))?;
            if !matches!(input.access(), TensorAccess::Read | TensorAccess::ReadWrite)
                || !matches!(input.alias(), AliasPolicy::NoAlias)
            {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "operation `{}` input {index} has invalid access or alias semantics",
                        self.id
                    ),
                });
            }
        }
        for (index, output) in self.outputs.iter().enumerate() {
            output.validate(&format!("operation.{}.outputs[{index}]", self.id))?;
            if !matches!(
                output.access(),
                TensorAccess::Write | TensorAccess::ReadWrite
            ) {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!("operation `{}` output {index} is not writable", self.id),
                });
            }
            if let AliasPolicy::MayAlias { tensor_index }
            | AliasPolicy::MustAlias { tensor_index } = output.alias()
            {
                if *tensor_index as usize >= self.inputs.len() {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("operation `{}` output {index} aliases no input", self.id),
                    });
                }
            }
        }
        if self.resources.minimum_value_alignment_bytes == 0
            || !self
                .resources
                .minimum_value_alignment_bytes
                .is_power_of_two()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("operation `{}` has invalid resource requirements", self.id),
            });
        }
        match self.oracle {
            OracleSpec::AbsoluteTolerance { tolerance }
            | OracleSpec::RelativeTolerance { tolerance }
                if tolerance.numerator() < 0 =>
            {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!("operation `{}` has a negative oracle tolerance", self.id),
                });
            }
            OracleSpec::AbsoluteTolerance { .. } | OracleSpec::RelativeTolerance { .. }
                if self
                    .outputs
                    .iter()
                    .any(|output| output.element_types().contains(&ElementType::Bool)) =>
            {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "operation `{}` applies numeric oracle tolerance to a possible boolean output",
                        self.id
                    ),
                });
            }
            _ => {}
        }
        if self.provider.minimum_version.major == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("operation `{}` has a zero provider major version", self.id),
            });
        }
        if self.provider.minimum_version.major != self.version.major {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "operation `{}` version {} and provider minimum version {} have incompatible major versions",
                    self.id, self.version, self.provider.minimum_version
                ),
            });
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize operation descriptor",
            message: error.to_string(),
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn validate_attributes(
        &self,
        values: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<(), VNextError> {
        self.attributes
            .validate_values(values, &format!("operation.{}.attributes", self.id))
    }

    pub fn validate_resolved_bindings(
        &self,
        bindings: &[ResolvedValueBinding],
    ) -> Result<(), VNextError> {
        self.validate()?;
        if bindings.len() != self.inputs.len() + self.outputs.len() {
            return Err(invalid_operation(format!(
                "operation `{}` expects {} value bindings, received {}",
                self.id,
                self.inputs.len() + self.outputs.len(),
                bindings.len()
            )));
        }

        let mut dimensions = BTreeMap::<String, u64>::new();
        let mut strides = BTreeMap::<String, u64>::new();
        let mut positions = BTreeSet::new();
        for (index, binding) in bindings.iter().enumerate() {
            let expected_position = if index < self.inputs.len() {
                (ResolvedValueRole::Input, index as u32)
            } else {
                (
                    ResolvedValueRole::Output,
                    (index - self.inputs.len()) as u32,
                )
            };
            if (binding.role(), binding.ordinal()) != expected_position {
                return Err(invalid_operation(format!(
                    "operation `{}` bindings are not in canonical input/output ordinal order",
                    self.id
                )));
            }
            if !positions.insert((binding.role(), binding.ordinal())) {
                return Err(invalid_operation(format!(
                    "operation `{}` contains duplicate ordinal bindings",
                    self.id
                )));
            }
            if let Some(previous) = bindings[..index]
                .iter()
                .find(|previous| previous.value_id() == binding.value_id())
            {
                let repeated_readonly_input = previous.role() == ResolvedValueRole::Input
                    && binding.role() == ResolvedValueRole::Input
                    && previous.access() == TensorAccess::Read
                    && binding.access() == TensorAccess::Read
                    && previous.tensor() == binding.tensor()
                    && previous.storage() == binding.storage()
                    && previous.usage() == binding.usage();
                if !repeated_readonly_input {
                    return Err(invalid_operation(format!(
                        "operation `{}` repeats a value outside identical read-only input slots",
                        self.id
                    )));
                }
            }
            let contract = match binding.role() {
                ResolvedValueRole::Input => self.inputs.get(binding.ordinal() as usize),
                ResolvedValueRole::Output => self.outputs.get(binding.ordinal() as usize),
            }
            .ok_or_else(|| {
                invalid_operation(format!(
                    "operation `{}` binding ordinal is out of range",
                    self.id
                ))
            })?;
            if binding.access() != contract.access() || binding.alias() != contract.alias() {
                return Err(invalid_operation(format!(
                    "operation `{}` binding access or alias differs from its contract",
                    self.id
                )));
            }
            Self::validate_resolved_tensor(
                &self.id,
                contract,
                binding.tensor(),
                &mut dimensions,
                &mut strides,
            )?;
        }
        let inputs = &bindings[..self.inputs.len()];
        let outputs = &bindings[self.inputs.len()..];
        for (index, input) in inputs.iter().enumerate() {
            for previous in &inputs[..index] {
                if storage_overlaps(input.storage(), previous.storage())
                    && (input.value_id() != previous.value_id()
                        || input.access() != TensorAccess::Read
                        || previous.access() != TensorAccess::Read)
                {
                    return Err(invalid_operation(format!(
                        "operation `{}` shares input storage between different or writable values",
                        self.id
                    )));
                }
            }
        }
        for (index, output) in outputs.iter().enumerate() {
            let aliased_inputs = inputs
                .iter()
                .enumerate()
                .filter(|(_, input)| storage_overlaps(output.storage(), input.storage()))
                .map(|(ordinal, _)| ordinal as u32)
                .collect::<Vec<_>>();
            match output.alias() {
                AliasPolicy::NoAlias if !aliased_inputs.is_empty() => {
                    return Err(invalid_operation(format!(
                        "operation `{}` output {index} aliases despite a no-alias contract",
                        self.id
                    )));
                }
                AliasPolicy::MayAlias { tensor_index } => {
                    if aliased_inputs.iter().any(|ordinal| ordinal != tensor_index)
                        || (aliased_inputs.contains(tensor_index)
                            && output.storage() != inputs[*tensor_index as usize].storage())
                    {
                        return Err(invalid_operation(format!(
                            "operation `{}` output {index} partially aliases or aliases the wrong input",
                            self.id
                        )));
                    }
                }
                AliasPolicy::MustAlias { tensor_index }
                    if aliased_inputs != [*tensor_index]
                        || output.storage() != inputs[*tensor_index as usize].storage() =>
                {
                    return Err(invalid_operation(format!(
                        "operation `{}` output {index} does not exactly alias its declared input",
                        self.id
                    )));
                }
                _ => {}
            }
            if outputs[..index]
                .iter()
                .any(|previous| storage_overlaps(output.storage(), previous.storage()))
            {
                return Err(invalid_operation(format!(
                    "operation `{}` output resources overlap",
                    self.id
                )));
            }
        }
        Ok(())
    }

    fn validate_resolved_tensor(
        operation_id: &OperationId,
        contract: &TensorContract,
        tensor: &ResolvedTensorSpec,
        dimensions: &mut BTreeMap<String, u64>,
        strides: &mut BTreeMap<String, u64>,
    ) -> Result<(), VNextError> {
        if tensor.dimensions().len() != contract.dimensions().len()
            || !contract.element_types().contains(&tensor.element_type())
        {
            return Err(invalid_operation(format!(
                "operation `{operation_id}` resolved tensor rank or element type is incompatible"
            )));
        }
        for (constraint, extent) in contract.dimensions().iter().zip(tensor.dimensions()) {
            let compatible = match constraint {
                DimensionConstraint::Exact(expected) => expected == extent,
                DimensionConstraint::Range { minimum, maximum } => {
                    minimum <= extent && extent <= maximum
                }
                DimensionConstraint::Symbol(symbol) => match dimensions.get(symbol) {
                    Some(expected) => expected == extent,
                    None => {
                        dimensions.insert(symbol.clone(), *extent);
                        true
                    }
                },
            };
            if !compatible {
                return Err(invalid_operation(format!(
                    "operation `{operation_id}` resolved tensor violates a dimension constraint"
                )));
            }
        }

        let mut matched_strides = None;
        let layout_matches =
            contract
                .layouts()
                .iter()
                .any(|layout| match (layout, tensor.layout()) {
                    (LayoutConstraint::Contiguous, ResolvedTensorLayout::Contiguous) => true,
                    (
                        LayoutConstraint::Blocked {
                            block: expected_block,
                            axis_order: expected_axis_order,
                        },
                        ResolvedTensorLayout::Blocked {
                            block: actual_block,
                            axis_order: actual_axis_order,
                            ..
                        },
                    ) => expected_block == actual_block && expected_axis_order == actual_axis_order,
                    (
                        LayoutConstraint::Strided {
                            strides: constraints,
                        },
                        ResolvedTensorLayout::Strided { byte_strides },
                    ) if constraints.len() == byte_strides.len() => {
                        let mut candidate = strides.clone();
                        let matches =
                            constraints
                                .iter()
                                .zip(byte_strides)
                                .all(|(constraint, actual)| match constraint {
                                    StrideConstraint::ExactBytes(expected) => expected == actual,
                                    StrideConstraint::Symbol(symbol) => match candidate.get(symbol)
                                    {
                                        Some(expected) => expected == actual,
                                        None => {
                                            candidate.insert(symbol.clone(), *actual);
                                            true
                                        }
                                    },
                                });
                        if matches {
                            matched_strides = Some(candidate);
                        }
                        matches
                    }
                    _ => false,
                });
        if !layout_matches {
            return Err(invalid_operation(format!(
                "operation `{operation_id}` resolved tensor layout is incompatible"
            )));
        }
        if let Some(candidate) = matched_strides {
            *strides = candidate;
        }
        Ok(())
    }
}

fn storage_overlaps(left: &ResolvedValueStorage, right: &ResolvedValueStorage) -> bool {
    left.components().iter().any(|left| {
        right.components().iter().any(|right| {
            left.resource_id() == right.resource_id()
                && left.offset_bytes() < right.offset_bytes().saturating_add(right.length_bytes())
                && right.offset_bytes() < left.offset_bytes().saturating_add(left.length_bytes())
        })
    })
}

/// Object-safe semantic operation contract used while building a plan.
pub trait OperationContract: Send + Sync {
    fn descriptor(&self) -> &OperationDescriptor;

    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError>;
}
