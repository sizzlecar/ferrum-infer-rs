//! External evidence for the numerical choice made during static composition.

use serde::{Deserialize, Serialize};

use super::{
    canonical_runtime_policy_fingerprint, CapabilityCatalog, ContractVersion, ExecutionPlan,
    ModelFamilyDefinition, NumericalExecutionPolicy, NumericalProfileId, PreparedModelFamily,
    ResolvedRuntimePolicy, VNextError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalProfileRejectionStage {
    FamilyPreparation,
    WeightMaterializer,
    ProgramCompilation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalProfileRejection {
    pub profile_id: NumericalProfileId,
    pub stage: NumericalProfileRejectionStage,
    pub reason: String,
}

/// Created by the composition owner from the retained static plan. It cannot be
/// deserialized as trusted evidence. Revalidation requires the independently
/// supplied selection, including the original request and rejected candidates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct NumericalProfileResolution {
    parts: NumericalProfileResolutionWire,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NumericalProfileResolutionWire {
    requested: NumericalExecutionPolicy,
    selected_profile: NumericalProfileId,
    selected_version: ContractVersion,
    profile_fingerprint: String,
    qualification_version: ContractVersion,
    definition_fingerprint: String,
    prepared_family_fingerprint: String,
    capability_catalog_fingerprint: String,
    runtime_policy_fingerprint: String,
    execution_plan_hash: String,
    rejected: Vec<NumericalProfileRejection>,
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("numerical resolution: {}", reason.into()),
    }
}

impl NumericalProfileResolution {
    /// The caller supplies failures observed while trying the declared prefix
    /// of candidates with the same registries. Successful compilation remains
    /// bound to this exact family, catalog and runtime policy. This constructor
    /// does not qualify new numerical profiles for Auto.
    #[allow(clippy::too_many_arguments)]
    pub fn from_static_plan(
        requested: NumericalExecutionPolicy,
        definition: &ModelFamilyDefinition,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        runtime: &ResolvedRuntimePolicy,
        plan: &ExecutionPlan,
        rejected: Vec<NumericalProfileRejection>,
    ) -> Result<Self, VNextError> {
        let profiles = definition.numerical_profiles();
        let selected = family.numerical_profile();
        if definition.family_id() != family.family_id()
            || definition.canonical_config() != family.canonical_config()
            || profiles.resolve(&selected.id)? != selected
        {
            return Err(invalid("selected family differs from its typed definition"));
        }
        let candidates = profiles.candidates(&requested)?;
        let selected_position = candidates
            .iter()
            .position(|profile| profile.id == selected.id)
            .ok_or_else(|| invalid("selected profile is not a candidate for the request"))?;
        if rejected.len() != selected_position
            || rejected
                .iter()
                .zip(&candidates)
                .any(|(rejection, profile)| {
                    rejection.profile_id != profile.id || rejection.reason.trim().is_empty()
                })
        {
            return Err(invalid(
                "rejections must explain exactly the declared candidates before the selected profile",
            ));
        }
        let prepared_family_fingerprint = family.fingerprint()?;
        let capability_catalog_fingerprint = capabilities.fingerprint()?;
        let runtime_policy_fingerprint = canonical_runtime_policy_fingerprint(runtime)?;
        let payload = plan.payload();
        if payload.prepared_family_fingerprint() != prepared_family_fingerprint
            || payload.program_fingerprint() != family.program().fingerprint()?
            || payload.capability_catalog_fingerprint() != capability_catalog_fingerprint
            || payload.policy_fingerprint() != runtime_policy_fingerprint
        {
            return Err(invalid(
                "static plan belongs to another family, catalog or runtime policy",
            ));
        }
        Ok(Self {
            parts: NumericalProfileResolutionWire {
                requested,
                selected_profile: selected.id.clone(),
                selected_version: selected.version,
                profile_fingerprint: selected.fingerprint()?,
                qualification_version: profiles.version(),
                definition_fingerprint: definition.fingerprint()?,
                prepared_family_fingerprint,
                capability_catalog_fingerprint,
                runtime_policy_fingerprint,
                execution_plan_hash: plan.plan_hash().as_str().to_owned(),
                rejected,
            },
        })
    }

    pub fn requested(&self) -> &NumericalExecutionPolicy {
        &self.parts.requested
    }

    pub fn selected_profile(&self) -> &NumericalProfileId {
        &self.parts.selected_profile
    }

    pub fn rejected(&self) -> &[NumericalProfileRejection] {
        &self.parts.rejected
    }

    pub(crate) fn matches_wire(&self, wire: &NumericalProfileResolutionWire) -> bool {
        &self.parts == wire
    }

    pub(crate) fn validate_for(
        &self,
        definition: &ModelFamilyDefinition,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        runtime: &ResolvedRuntimePolicy,
        plan: &ExecutionPlan,
    ) -> Result<(), VNextError> {
        let rebuilt = Self::from_static_plan(
            self.parts.requested.clone(),
            definition,
            family,
            capabilities,
            runtime,
            plan,
            self.parts.rejected.clone(),
        )?;
        if &rebuilt != self {
            return Err(invalid(
                "selection evidence differs from typed reconstruction",
            ));
        }
        Ok(())
    }
}
