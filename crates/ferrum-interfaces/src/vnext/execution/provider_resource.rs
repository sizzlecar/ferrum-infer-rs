use super::{
    canonical_fingerprint, invalid_plan, is_canonical_sha256, ContractVersion,
    OperationProviderDescriptor, OperationResourceEstimate, ProviderId,
    ProviderWorkspaceRequirement, ProviderWorkspaceScope, Serialize, VNextError,
};

/// Trusted output from the selected provider's shape/attribute-specific
/// resource estimator. The core binds it to the exact estimator input and
/// selected provider before the values can enter an executable plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderResourcePlan {
    pub(super) provider_id: ProviderId,
    pub(super) estimator_id: String,
    pub(super) estimator_version: ContractVersion,
    pub(super) estimator_implementation_fingerprint: String,
    pub(super) estimator_input_fingerprint: String,
    pub(super) estimate_fingerprint: String,
    pub(super) value_alignment_bytes: u64,
    pub(super) scratch: Option<ProviderWorkspaceRequirement>,
    pub(super) persistent: Option<ProviderWorkspaceRequirement>,
}

impl ProviderResourcePlan {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_provider_output(
        descriptor: &OperationProviderDescriptor,
        estimator_input_fingerprint: &str,
        estimate: OperationResourceEstimate,
    ) -> Result<Self, VNextError> {
        if estimate.estimator_id() != descriptor.resource_estimator_id()
            || estimate.estimator_version() != descriptor.resource_estimator_version()
            || estimate.estimator_implementation_fingerprint()
                != descriptor.resource_estimator_implementation_fingerprint()
            || estimate.claimed_input_fingerprint() != estimator_input_fingerprint
        {
            return Err(invalid_plan(
                "provider raw resource estimate identity or input claim differs from the selected registered implementation",
            ));
        }
        let mut plan = Self {
            provider_id: descriptor.provider_id().clone(),
            estimator_id: descriptor.resource_estimator_id().to_owned(),
            estimator_version: descriptor.resource_estimator_version(),
            estimator_implementation_fingerprint: descriptor
                .resource_estimator_implementation_fingerprint()
                .to_owned(),
            estimator_input_fingerprint: estimator_input_fingerprint.to_owned(),
            estimate_fingerprint: String::new(),
            value_alignment_bytes: estimate.value_alignment_bytes(),
            scratch: estimate.scratch().cloned(),
            persistent: estimate.persistent().cloned(),
        };
        plan.validate_fields()?;
        plan.estimate_fingerprint = plan.compute_estimate_fingerprint()?;
        plan.validate_shape()?;
        Ok(plan)
    }

    pub(super) fn validate_fields(&self) -> Result<(), VNextError> {
        if self.estimator_id.is_empty()
            || self.estimator_id.len() > 160
            || !self.estimator_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
            })
            || self.estimator_version.major == 0
            || !is_canonical_sha256(&self.estimator_implementation_fingerprint)
            || !is_canonical_sha256(&self.estimator_input_fingerprint)
            || self.value_alignment_bytes == 0
            || !self.value_alignment_bytes.is_power_of_two()
            || self.scratch.as_ref().is_some_and(|workspace| {
                workspace.scope != ProviderWorkspaceScope::Invocation
                    || ProviderWorkspaceRequirement::from_formula(
                        workspace.size_formula.clone(),
                        workspace.alignment_bytes,
                        workspace.scope,
                        workspace.storage.clone(),
                    )
                    .is_err()
            })
            || self.persistent.as_ref().is_some_and(|workspace| {
                workspace.scope == ProviderWorkspaceScope::Invocation
                    || ProviderWorkspaceRequirement::from_formula(
                        workspace.size_formula.clone(),
                        workspace.alignment_bytes,
                        workspace.scope,
                        workspace.storage.clone(),
                    )
                    .is_err()
            })
        {
            return Err(invalid_plan(
                "provider resource estimate identity, alignment, or scope is invalid",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_shape(&self) -> Result<(), VNextError> {
        self.validate_fields()?;
        if !is_canonical_sha256(&self.estimate_fingerprint)
            || self.estimate_fingerprint != self.compute_estimate_fingerprint()?
        {
            return Err(invalid_plan(
                "provider resource estimate fingerprint does not match its typed fields",
            ));
        }
        Ok(())
    }

    pub(super) fn compute_estimate_fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(
            &ProviderEstimateFingerprintMaterial {
                provider_id: &self.provider_id,
                estimator_id: &self.estimator_id,
                estimator_version: self.estimator_version,
                estimator_implementation_fingerprint: &self.estimator_implementation_fingerprint,
                estimator_input_fingerprint: &self.estimator_input_fingerprint,
                value_alignment_bytes: self.value_alignment_bytes,
                scratch: &self.scratch,
                persistent: &self.persistent,
            },
            "fingerprint provider resource estimate",
        )
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn estimator_id(&self) -> &str {
        &self.estimator_id
    }

    pub const fn estimator_version(&self) -> ContractVersion {
        self.estimator_version
    }

    pub fn estimator_implementation_fingerprint(&self) -> &str {
        &self.estimator_implementation_fingerprint
    }

    pub fn estimator_input_fingerprint(&self) -> &str {
        &self.estimator_input_fingerprint
    }

    pub fn estimate_fingerprint(&self) -> &str {
        &self.estimate_fingerprint
    }

    pub const fn value_alignment_bytes(&self) -> u64 {
        self.value_alignment_bytes
    }

    pub fn scratch(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.scratch.as_ref()
    }

    pub fn persistent(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.persistent.as_ref()
    }
}

#[derive(Serialize)]
pub(super) struct ProviderEstimateFingerprintMaterial<'a> {
    pub(super) provider_id: &'a ProviderId,
    pub(super) estimator_id: &'a str,
    pub(super) estimator_version: ContractVersion,
    pub(super) estimator_implementation_fingerprint: &'a str,
    pub(super) estimator_input_fingerprint: &'a str,
    pub(super) value_alignment_bytes: u64,
    pub(super) scratch: &'a Option<ProviderWorkspaceRequirement>,
    pub(super) persistent: &'a Option<ProviderWorkspaceRequirement>,
}
