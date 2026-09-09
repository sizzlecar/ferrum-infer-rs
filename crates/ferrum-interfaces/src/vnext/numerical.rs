//! Numerical ABI declarations are independent of source containers and devices.

use std::collections::{BTreeMap, BTreeSet};

pub use ferrum_types::{NumericalExecutionPolicy, NumericalProfileId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    ContractVersion, ElementType, ModelFamilyId, ModelProgram, OperationId, ProgramValueId,
    ResolvedTensorSpec, StateSpec, VNextError,
};

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
            if operation.version.major == 0
                || !operation_ids.insert(&operation.operation_id)
                || operation
                    .multiplication_type
                    .is_some_and(|dtype| !floating(dtype))
                || operation
                    .accumulation_type
                    .is_some_and(|dtype| !floating(dtype))
            {
                return Err(invalid(&self.family_id, "operation contracts need unique identities, valid versions and floating-point arithmetic types"));
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
        Ok(())
    }

    pub(crate) fn normalize(&mut self) {
        self.states.sort_by(|left, right| left.id.cmp(&right.id));
        self.operations
            .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    }

    pub fn activation_type(&self) -> Result<ElementType, VNextError> {
        self.validate()?;
        Ok(self.boundaries[&self.primary_activation])
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
            .map(|operation| (&operation.operation_id, operation.version))
            .collect();
        let mut used_operations = BTreeSet::new();
        let mut outputs = BTreeSet::new();
        for node in program.blocks().iter().flat_map(|block| &block.nodes) {
            if contracts.get(&node.operation_id) != Some(&node.required_version) {
                return Err(invalid(
                    &self.family_id,
                    format!(
                        "node {} uses an undeclared numerical operation/version",
                        node.id
                    ),
                ));
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
        ids.into_iter().map(|id| self.resolve(id)).collect()
    }
}
