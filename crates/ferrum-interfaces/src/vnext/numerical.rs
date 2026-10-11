//! Numerical ABI declarations are independent of source containers and devices.

use std::collections::{BTreeMap, BTreeSet};

pub use ferrum_types::{KvStorageFormat, NumericalExecutionPolicy, NumericalProfileId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    ContractVersion, ElementType, ModelFamilyId, ModelProgram, OperationContract, OperationId,
    ProgramValueId, ResolvedTensorSpec, StateSpec, VNextError,
};

mod kv_storage;
pub use kv_storage::KvStateStorage;
mod arithmetic;
mod g32_mmq;
mod q6_mmq_f32;
pub use arithmetic::*;
pub use g32_mmq::*;
pub use q6_mmq_f32::*;
mod composite;
pub use composite::*;
mod prepared;
pub use prepared::*;
mod upstream;
pub use upstream::*;

/// The referenced operation version defines its internal ordering and rounding
/// points. Arithmetic types are explicit here; copying/indexing operations have
/// no multiplication or accumulation. A fused operation's internal conversions
/// remain part of its versioned operation contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalOperationContract {
    pub operation_id: OperationId,
    pub version: ContractVersion,
    pub multiplication_type: Option<ElementType>,
    pub accumulation_type: Option<ElementType>,
    /// Ordered mixed arithmetic, when a single floating-point pair cannot
    /// describe the operation. Legacy declarations omit this field entirely.
    /// With stages present both legacy arithmetic fields must be `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub staged_arithmetic: Option<StagedNumericalArithmetic>,
    /// Strict fused semantics with explicitly scoped projection overrides.
    /// Mutually exclusive with stages and the legacy arithmetic pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composite_arithmetic: Option<CompositeNumericalArithmetic>,
}

/// A complete numerical choice declared by a family before program generation.
/// Public decoding alone does not confer trust: the family registration must
/// reproduce this profile from its validated typed model definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalExecutionProfile {
    pub id: NumericalProfileId,
    pub version: ContractVersion,
    pub family_id: ModelFamilyId,
    /// The main activation is explicit so product metadata need not guess a
    /// dtype from weights, a container, or a profile's human-readable name.
    pub primary_activation: ProgramValueId,
    /// Every operation output has a declared storage/rounding boundary. This
    /// includes residuals, intermediate activations, logits and token outputs.
    pub boundaries: BTreeMap<ProgramValueId, ElementType>,
    /// State shapes, initialization and capacity are shared with the final
    /// program, permitting startup sizing without constructing a dummy program.
    pub states: Vec<StateSpec>,
    /// Typed association of every causal KV payload and its optional scales.
    /// Empty means the family declares no applicable KV state.
    pub kv_storage: Vec<KvStateStorage>,
    pub operations: Vec<NumericalOperationContract>,
}

fn invalid(family: &ModelFamilyId, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: family.to_string(),
        field: "numerical_profile".to_owned(),
        reason: reason.into(),
    }
}

fn floating(element_type: ElementType) -> bool {
    matches!(
        element_type,
        ElementType::F16 | ElementType::Bf16 | ElementType::F32
    )
}

impl NumericalExecutionProfile {
    pub fn validate(&self) -> Result<(), VNextError> {
        if self.version.major == 0 || self.operations.is_empty() {
            return Err(invalid(
                &self.family_id,
                "profile version and operation contracts must be explicit",
            ));
        }
        if !self
            .boundaries
            .get(&self.primary_activation)
            .is_some_and(|dtype| floating(*dtype))
        {
            return Err(invalid(
                &self.family_id,
                "primary activation must name a floating-point boundary",
            ));
        }
        let mut operation_ids = BTreeSet::new();
        for operation in &self.operations {
            if operation.version.major == 0 || !operation_ids.insert(&operation.operation_id) {
                return Err(invalid(
                    &self.family_id,
                    "operation contracts need unique identities and valid versions",
                ));
            }
            if let Some(composite) = &operation.composite_arithmetic {
                if operation.staged_arithmetic.is_some()
                    || operation.multiplication_type.is_some()
                    || operation.accumulation_type.is_some()
                {
                    return Err(invalid(&self.family_id, "composite arithmetic cannot also declare whole-operation stages or an arithmetic pair"));
                }
                composite
                    .validate()
                    .map_err(|reason| invalid(&self.family_id, reason))?;
                if operation.operation_id == composite.strict_base.operation_id
                    && operation.version.major <= composite.strict_base.version.major
                {
                    return Err(invalid(&self.family_id, "projection overrides need a distinct operation identity or a breaking major version from the strict base"));
                }
            } else if let Some(staged) = &operation.staged_arithmetic {
                if operation.multiplication_type.is_some() || operation.accumulation_type.is_some()
                {
                    return Err(invalid(
                        &self.family_id,
                        "staged arithmetic cannot also declare a single-stage arithmetic pair",
                    ));
                }
                staged
                    .validate()
                    .map_err(|reason| invalid(&self.family_id, reason))?;
            } else if operation
                .multiplication_type
                .is_some_and(|dtype| !floating(dtype))
                || operation
                    .accumulation_type
                    .is_some_and(|dtype| !floating(dtype))
            {
                return Err(invalid(&self.family_id, "single-stage operation arithmetic must remain floating-point; integer arithmetic requires explicit stages"));
            }
        }
        let mut state_ids = BTreeSet::new();
        let mut state_values = BTreeSet::new();
        for state in &self.states {
            if !state_ids.insert(&state.id) || !state_values.insert(&state.value_id) {
                return Err(invalid(
                    &self.family_id,
                    "state identities and values must be unique",
                ));
            }
            state.tensor.validate("numerical_profile.state")?;
            state.capacity_demand.validate(state.tensor.byte_len()?)?;
        }
        self.kv_storage_format()?;
        Ok(())
    }

    pub(crate) fn normalize(&mut self) {
        self.states.sort_by(|left, right| left.id.cmp(&right.id));
        self.kv_storage
            .sort_by(|left, right| left.payload_state().cmp(right.payload_state()));
        self.operations
            .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    }

    pub fn activation_type(&self) -> Result<ElementType, VNextError> {
        self.validate()?;
        Ok(self.boundaries[&self.primary_activation])
    }

    pub fn kv_storage_format(&self) -> Result<Option<KvStorageFormat>, VNextError> {
        kv_storage::validate_kv_storage(&self.family_id, &self.kv_storage, &self.states)
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.normalize();
        let bytes = serde_json::to_vec(&canonical).map_err(|error| VNextError::Serialization {
            context: "serialize numerical execution profile",
            message: error.to_string(),
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Preparation validates the declared graph contract; compiler inference
    /// subsequently proves every boundary's actual dtype against real operations.
    pub fn validate_program(&self, program: &ModelProgram) -> Result<(), VNextError> {
        self.validate()?;
        if program.family_id() != &self.family_id {
            return Err(invalid(
                &self.family_id,
                "program belongs to another family",
            ));
        }
        let contracts: BTreeMap<_, _> = self
            .operations
            .iter()
            .map(|operation| (&operation.operation_id, operation))
            .collect();
        let mut used_operations = BTreeSet::new();
        let mut outputs = BTreeSet::new();
        for node in program.blocks().iter().flat_map(|block| &block.nodes) {
            let contract = contracts.get(&node.operation_id);
            if contract.map(|operation| operation.version) != Some(node.required_version) {
                return Err(invalid(
                    &self.family_id,
                    format!(
                        "node {} uses an undeclared numerical operation/version",
                        node.id
                    ),
                ));
            }
            if let Some(output_type) = contract
                .and_then(|operation| operation.staged_arithmetic.as_ref())
                .and_then(StagedNumericalArithmetic::output_type)
            {
                if node
                    .outputs
                    .iter()
                    .any(|output| self.boundaries.get(output) != Some(&output_type))
                {
                    return Err(invalid(
                        &self.family_id,
                        format!(
                            "node {} output boundary differs from its staged output rounding",
                            node.id
                        ),
                    ));
                }
            }
            if let Some(composite) =
                contract.and_then(|operation| operation.composite_arithmetic.as_ref())
            {
                let base = composite
                    .base_contract()
                    .map_err(|reason| invalid(&self.family_id, reason))?;
                let descriptor = base.descriptor();
                descriptor
                    .attributes
                    .validate_values(&node.attributes, "composite.strict_base")
                    .map_err(|error| invalid(&self.family_id, error.to_string()))?;
                if node.inputs.len() != descriptor.inputs.len()
                    || node.outputs.len() != descriptor.outputs.len()
                {
                    return Err(invalid(
                        &self.family_id,
                        format!("node {} arity differs from its strict base", node.id),
                    ));
                }
                // The fused result follows the base signature, NOT a local
                // projection's F16 store (GDN's external result is F32).
                for (output, port) in node.outputs.iter().zip(&descriptor.outputs) {
                    if !self
                        .boundaries
                        .get(output)
                        .is_some_and(|dtype| port.element_types().contains(dtype))
                    {
                        return Err(invalid(
                            &self.family_id,
                            format!(
                                "node {} output boundary differs from its strict base",
                                node.id
                            ),
                        ));
                    }
                }
                // Earlier node outputs have known numerical boundaries here.
                // External inputs/weight encodings still require the compiler's
                // full typed signature and real provider qualification.
                for (input, port) in node.inputs.iter().zip(&descriptor.inputs) {
                    if self
                        .boundaries
                        .get(input)
                        .is_some_and(|dtype| !port.element_types().contains(dtype))
                    {
                        return Err(invalid(
                            &self.family_id,
                            format!(
                                "node {} input boundary differs from its strict base",
                                node.id
                            ),
                        ));
                    }
                }
            }
            used_operations.insert(&node.operation_id);
            outputs.extend(&node.outputs);
        }
        if used_operations.len() != contracts.len() || outputs != self.boundaries.keys().collect() {
            return Err(invalid(
                &self.family_id,
                "profile must describe exactly the program's operations and output boundaries",
            ));
        }
        let states: BTreeMap<_, _> = self.states.iter().map(|state| (&state.id, state)).collect();
        let actual: BTreeMap<_, _> = program
            .states()
            .iter()
            .map(|state| (&state.id, state))
            .collect();
        if states != actual {
            return Err(invalid(
                &self.family_id,
                "program state ABI differs from the numerical profile",
            ));
        }
        kv_storage::validate_program_kv_storage(self, program)?;
        Ok(())
    }

    pub fn validate_inferred_boundaries(
        &self,
        values: &BTreeMap<ProgramValueId, ResolvedTensorSpec>,
    ) -> Result<(), VNextError> {
        for (value, dtype) in &self.boundaries {
            if values.get(value).map(ResolvedTensorSpec::element_type) != Some(*dtype) {
                return Err(invalid(
                    &self.family_id,
                    format!(
                        "inferred dtype for {value} differs from the selected numerical profile"
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// The preference order is a versioned family declaration, not registry order.
/// Profiles absent from `auto_preference` remain available only to `Require`;
/// this lets new numerical combinations be qualified before changing defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FamilyNumericalProfiles {
    version: ContractVersion,
    profiles: Vec<NumericalExecutionProfile>,
    auto_preference: Vec<NumericalProfileId>,
}

impl FamilyNumericalProfiles {
    pub fn new(
        family_id: &ModelFamilyId,
        version: ContractVersion,
        mut profiles: Vec<NumericalExecutionProfile>,
        auto_preference: Vec<NumericalProfileId>,
    ) -> Result<Self, VNextError> {
        if version.major == 0 || profiles.is_empty() {
            return Err(invalid(
                family_id,
                "family must declare versioned numerical profiles",
            ));
        }
        let mut ids = BTreeSet::new();
        for profile in &mut profiles {
            profile.validate()?;
            profile.normalize();
            if &profile.family_id != family_id || !ids.insert(profile.id.clone()) {
                return Err(invalid(
                    family_id,
                    "numerical profiles must belong to this family and have unique identities",
                ));
            }
        }
        let mut automatic = BTreeSet::new();
        if auto_preference
            .iter()
            .any(|id| !ids.contains(id) || !automatic.insert(id))
        {
            return Err(invalid(
                family_id,
                "Auto preferences must name unique declared profiles",
            ));
        }
        profiles.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(Self {
            version,
            profiles,
            auto_preference,
        })
    }

    pub fn version(&self) -> ContractVersion {
        self.version
    }
    pub fn profiles(&self) -> &[NumericalExecutionProfile] {
        &self.profiles
    }
    pub fn auto_preference(&self) -> &[NumericalProfileId] {
        &self.auto_preference
    }

    pub fn resolve(
        &self,
        id: &NumericalProfileId,
    ) -> Result<&NumericalExecutionProfile, VNextError> {
        self.profiles
            .iter()
            .find(|profile| &profile.id == id)
            .ok_or_else(|| {
                invalid(
                    &self.profiles[0].family_id,
                    format!("numerical profile {id} is not declared by this family"),
                )
            })
    }

    pub fn candidates(
        &self,
        policy: &NumericalExecutionPolicy,
        kv_storage: KvStorageFormat,
    ) -> Result<Vec<&NumericalExecutionProfile>, VNextError> {
        let ids = match policy {
            NumericalExecutionPolicy::Auto => self.auto_preference.iter().collect::<Vec<_>>(),
            NumericalExecutionPolicy::Require(id) => vec![id],
        };
        if ids.is_empty() {
            return Err(invalid(
                &self.profiles[0].family_id,
                "no numerical profile is qualified for Auto",
            ));
        }
        let mut candidates = Vec::new();
        for id in ids {
            let profile = self.resolve(id)?;
            // F16 is the existing default, including programs with no causal
            // KV state. An explicit INT8 request must actually change KV storage.
            let matches = profile
                .kv_storage_format()?
                .map_or(kv_storage == KvStorageFormat::F16, |actual| {
                    actual == kv_storage
                });
            if matches {
                candidates.push(profile);
            } else if matches!(policy, NumericalExecutionPolicy::Require(_)) {
                return Err(invalid(&profile.family_id, format!(
                    "numerical profile {} conflicts with requested KV storage {kv_storage}; select a compatible profile and kv_dtype", profile.id
                )));
            }
        }
        if candidates.is_empty() {
            return Err(invalid(&self.profiles[0].family_id, format!(
                "no qualified numerical profile supports KV storage {kv_storage}; it may be unsupported or not applicable to this family; use fp16"
            )));
        }
        Ok(candidates)
    }
}
