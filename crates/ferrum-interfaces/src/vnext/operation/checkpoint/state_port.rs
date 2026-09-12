use serde::{Deserialize, Serialize};

use crate::vnext::{DynamicStorageProfile, ResolvedValueRole};

/// Exact logical byte ABI promised by a provider for one state port. Both
/// forms require a contiguous semantic tensor and one physical component;
/// the selected storage profile may translate its logical range into pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderCheckpointStateLayout {
    /// The component contains the complete tensor value at the boundary.
    ContiguousBoundaryValue,
    /// Ordered token-major values with no per-token padding. The declared
    /// stride is the entire resolved semantic tensor's byte size. Only [0,N)
    /// is valid, regardless of how much backing capacity was reserved.
    TokenMajorPrefix,
}

/// A declaration is specific to an operation port and physical storage ABI.
/// Its ordinal is validated against the selected operation; it is not a
/// semantic state name or a user-supplied resource byte range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCheckpointStatePort {
    role: ResolvedValueRole,
    ordinal: u32,
    storage_profile: DynamicStorageProfile,
    layout: ProviderCheckpointStateLayout,
}

impl ProviderCheckpointStatePort {
    pub const fn new(
        role: ResolvedValueRole,
        ordinal: u32,
        storage_profile: DynamicStorageProfile,
        layout: ProviderCheckpointStateLayout,
    ) -> Self {
        Self {
            role,
            ordinal,
            storage_profile,
            layout,
        }
    }

    pub const fn role(&self) -> ResolvedValueRole {
        self.role
    }
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }
    pub const fn storage_profile(&self) -> DynamicStorageProfile {
        self.storage_profile
    }
    pub const fn layout(&self) -> ProviderCheckpointStateLayout {
        self.layout
    }

    pub(super) fn key(&self) -> (ResolvedValueRole, u32, DynamicStorageProfile) {
        (self.role, self.ordinal, self.storage_profile)
    }
}
