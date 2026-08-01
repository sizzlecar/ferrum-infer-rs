use serde::{Deserialize, Deserializer, Serialize};

use super::super::VNextError;
use super::foundation::invalid_operation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    Bool,
    U8,
    U32,
    I8,
    I32,
    F16,
    Bf16,
    F32,
}

impl ElementType {
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicStorageAllocator {
    LinearArena,
    FixedBlockArena { block_bytes: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicStorageView {
    Contiguous,
    PagedRegions { block_bytes: u64 },
}

/// Backend-neutral physical addressability offered by a runtime and accepted
/// by an operation provider. This is independent from capacity formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct DynamicStorageProfile {
    allocator: DynamicStorageAllocator,
    view: DynamicStorageView,
}

impl DynamicStorageProfile {
    pub fn new(
        allocator: DynamicStorageAllocator,
        view: DynamicStorageView,
    ) -> Result<Self, VNextError> {
        let valid = match (allocator, view) {
            (DynamicStorageAllocator::LinearArena, DynamicStorageView::Contiguous) => true,
            (
                DynamicStorageAllocator::FixedBlockArena { block_bytes },
                DynamicStorageView::Contiguous,
            ) => block_bytes.is_power_of_two(),
            (
                DynamicStorageAllocator::FixedBlockArena {
                    block_bytes: allocator_block,
                },
                DynamicStorageView::PagedRegions {
                    block_bytes: view_block,
                },
            ) => allocator_block.is_power_of_two() && allocator_block == view_block,
            (DynamicStorageAllocator::LinearArena, DynamicStorageView::PagedRegions { .. }) => {
                false
            }
        };
        if !valid {
            return Err(invalid_operation(
                "dynamic storage allocator/view profile is incompatible or invalid",
            ));
        }
        Ok(Self { allocator, view })
    }

    pub const fn allocator(self) -> DynamicStorageAllocator {
        self.allocator
    }

    pub const fn view(self) -> DynamicStorageView {
        self.view
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicStorageProfileWire {
    allocator: DynamicStorageAllocator,
    view: DynamicStorageView,
}

impl<'de> Deserialize<'de> for DynamicStorageProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicStorageProfileWire::deserialize(deserializer)?;
        Self::new(wire.allocator, wire.view).map_err(serde::de::Error::custom)
    }
}

/// Canonical non-empty set of profiles accepted by a provider binding or one
/// provider-owned workspace. The planner intersects this with runtime offers
/// and the ordered runtime-policy allowlist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicStorageRequirement {
    accepted_profiles: Vec<DynamicStorageProfile>,
}

impl DynamicStorageRequirement {
    pub fn new(mut accepted_profiles: Vec<DynamicStorageProfile>) -> Result<Self, VNextError> {
        accepted_profiles.sort_unstable();
        accepted_profiles.dedup();
        if accepted_profiles.is_empty() {
            return Err(invalid_operation(
                "dynamic storage requirement has no accepted profile",
            ));
        }
        Ok(Self { accepted_profiles })
    }

    pub fn contiguous() -> Self {
        Self {
            accepted_profiles: vec![DynamicStorageProfile {
                allocator: DynamicStorageAllocator::LinearArena,
                view: DynamicStorageView::Contiguous,
            }],
        }
    }

    pub fn accepted_profiles(&self) -> &[DynamicStorageProfile] {
        &self.accepted_profiles
    }

    pub fn accepts(&self, profile: DynamicStorageProfile) -> bool {
        self.accepted_profiles.binary_search(&profile).is_ok()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicStorageRequirementWire {
    accepted_profiles: Vec<DynamicStorageProfile>,
}

impl<'de> Deserialize<'de> for DynamicStorageRequirement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicStorageRequirementWire::deserialize(deserializer)?;
        let original = wire.accepted_profiles.clone();
        let requirement = Self::new(wire.accepted_profiles).map_err(serde::de::Error::custom)?;
        if requirement.accepted_profiles != original {
            return Err(serde::de::Error::custom(
                "dynamic storage requirement profiles are not canonical and unique",
            ));
        }
        Ok(requirement)
    }
}
