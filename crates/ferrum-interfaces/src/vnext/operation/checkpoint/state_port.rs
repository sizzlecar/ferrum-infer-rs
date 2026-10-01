use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;

use crate::vnext::{
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView, ElementType,
    ResolvedTensorSpec, ResolvedValueRole, VNextError,
};

/// Exact logical byte ABI promised by a provider for one state port. All
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
    /// A paged [key/value, heads, head dimension] F16 prefix. When a complete
    /// token block fits in, and evenly divides, each page and the head dimension
    /// is divisible by the key pack, keys use [head, dimension/pack, token,
    /// pack] and values use [head, dimension, token]. Otherwise the provider
    /// uses token-major pages. The compiler resolves that same geometry rule.
    /// Only initialized prefix slots participate, including a partial block.
    PagedKeyValueBlockPrefix {
        tokens_per_block: NonZeroU64,
        key_pack_elements: NonZeroU64,
    },
}

impl ProviderCheckpointStateLayout {
    pub(crate) fn resolve_prefix_mapping(
        self,
        tensor: &ResolvedTensorSpec,
        storage: DynamicStorageProfile,
    ) -> Result<Self, VNextError> {
        let Self::PagedKeyValueBlockPrefix {
            tokens_per_block,
            key_pack_elements,
        } = self
        else {
            return Ok(self);
        };
        let [2, heads, dimension] = tensor.dimensions() else {
            return Err(super::invalid_operation(
                "paged key/value checkpoint requires [2, heads, dimension] state",
            ));
        };
        if *heads == 0 || *dimension == 0 || tensor.element_type() != ElementType::F16 {
            return Err(super::invalid_operation(
                "paged key/value checkpoint requires positive F16 state geometry",
            ));
        }
        let (
            DynamicStorageAllocator::FixedBlockArena { block_bytes: page },
            DynamicStorageView::PagedRegions {
                block_bytes: view_page,
            },
        ) = (storage.allocator(), storage.view())
        else {
            return Err(super::invalid_operation(
                "paged key/value checkpoint requires fixed paged backing",
            ));
        };
        if page == 0 || page != view_page {
            return Err(super::invalid_operation(
                "paged key/value checkpoint has inconsistent page geometry",
            ));
        }
        let block_bytes = tensor
            .minimum_storage_bytes()?
            .checked_mul(tokens_per_block.get())
            .ok_or_else(|| super::invalid_operation("checkpoint token block overflows u64"))?;
        if dimension.is_multiple_of(key_pack_elements.get())
            && block_bytes <= page
            && page.is_multiple_of(block_bytes)
        {
            Ok(self)
        } else {
            Ok(Self::TokenMajorPrefix)
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::ResolvedTensorLayout;

    #[test]
    fn paged_key_value_mapping_resolves_the_real_pack_and_page_eligibility() {
        let layout = ProviderCheckpointStateLayout::PagedKeyValueBlockPrefix {
            tokens_per_block: NonZeroU64::new(16).unwrap(),
            key_pack_elements: NonZeroU64::new(8).unwrap(),
        };
        let storage = DynamicStorageProfile::new(
            DynamicStorageAllocator::FixedBlockArena { block_bytes: 65536 },
            DynamicStorageView::PagedRegions { block_bytes: 65536 },
        )
        .unwrap();
        for (heads, dimension, blocked) in [
            (1, 8, true),
            (4, 256, true),
            (1, 12, false),
            (3, 256, false),
            (17, 256, false),
        ] {
            let tensor = ResolvedTensorSpec::new(
                vec![2, heads, dimension],
                ElementType::F16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap();
            assert_eq!(
                layout.resolve_prefix_mapping(&tensor, storage).unwrap(),
                if blocked {
                    layout
                } else {
                    ProviderCheckpointStateLayout::TokenMajorPrefix
                }
            );
        }
        let float = ResolvedTensorSpec::new(
            vec![2, 1, 8],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        assert!(layout.resolve_prefix_mapping(&float, storage).is_err());
        let wrong = ResolvedTensorSpec::new(
            vec![1, 8],
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        assert!(layout.resolve_prefix_mapping(&wrong, storage).is_err());
    }
}
