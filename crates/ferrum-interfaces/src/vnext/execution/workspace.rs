use super::{
    align_up, invalid_plan, Deserialize, Deserializer, DynamicResourceDemand, DynamicResourceShape,
    DynamicStorageRequirement, ProviderWorkspaceSizeFormula, ResourceWorkShape, Serialize,
    VNextError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderWorkspaceScope {
    Plan,
    Request,
    Sequence,
    Step,
    Invocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderWorkspaceRequirement {
    pub(super) size_formula: ProviderWorkspaceSizeFormula,
    pub(super) alignment_bytes: u64,
    pub(super) scope: ProviderWorkspaceScope,
    pub(super) storage: DynamicStorageRequirement,
}

impl ProviderWorkspaceRequirement {
    /// Convenience constructor for a fixed-size workspace. Shape-dependent
    /// providers must use [`Self::from_formula`].
    pub fn new(
        fixed_bytes: u64,
        alignment_bytes: u64,
        scope: ProviderWorkspaceScope,
        storage: DynamicStorageRequirement,
    ) -> Result<Self, VNextError> {
        Self::from_formula(
            ProviderWorkspaceSizeFormula::fixed(fixed_bytes)?,
            alignment_bytes,
            scope,
            storage,
        )
    }

    pub fn from_formula(
        size_formula: ProviderWorkspaceSizeFormula,
        alignment_bytes: u64,
        scope: ProviderWorkspaceScope,
        storage: DynamicStorageRequirement,
    ) -> Result<Self, VNextError> {
        size_formula.validate()?;
        if alignment_bytes == 0
            || !alignment_bytes.is_power_of_two()
            || (scope == ProviderWorkspaceScope::Plan && !size_formula.is_fixed())
            || (scope == ProviderWorkspaceScope::Sequence
                && !size_formula.is_valid_for_sequence_scope())
        {
            return Err(invalid_plan(
                "provider workspace has invalid formula, alignment, or scope",
            ));
        }
        let requirement = Self {
            size_formula,
            alignment_bytes,
            scope,
            storage,
        };
        requirement.minimum_bytes()?;
        requirement.theoretical_maximum_bytes()?;
        Ok(requirement)
    }

    pub fn size_formula(&self) -> &ProviderWorkspaceSizeFormula {
        &self.size_formula
    }

    pub fn evaluate_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(work.immediate_shape())
    }

    pub fn evaluate_fit_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(work.fit_shape())
    }

    pub(crate) fn evaluate_shape_bytes(
        &self,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        align_up(
            self.size_formula.evaluate_shape_bytes(shape)?,
            self.alignment_bytes,
        )
    }

    pub fn minimum_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(self.size_formula.minimum_shape())
    }

    pub fn theoretical_maximum_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(self.size_formula.theoretical_maximum_shape())
    }

    pub fn fixed_bytes(&self) -> Option<u64> {
        match &self.size_formula {
            DynamicResourceDemand::Fixed { bytes } => Some(*bytes),
            _ => None,
        }
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    pub const fn scope(&self) -> ProviderWorkspaceScope {
        self.scope
    }

    pub fn storage(&self) -> &DynamicStorageRequirement {
        &self.storage
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProviderWorkspaceRequirementWire {
    pub(super) size_formula: ProviderWorkspaceSizeFormula,
    pub(super) alignment_bytes: u64,
    pub(super) scope: ProviderWorkspaceScope,
    pub(super) storage: DynamicStorageRequirement,
}

impl<'de> Deserialize<'de> for ProviderWorkspaceRequirement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderWorkspaceRequirementWire::deserialize(deserializer)?;
        Self::from_formula(
            wire.size_formula,
            wire.alignment_bytes,
            wire.scope,
            wire.storage,
        )
        .map_err(serde::de::Error::custom)
    }
}
