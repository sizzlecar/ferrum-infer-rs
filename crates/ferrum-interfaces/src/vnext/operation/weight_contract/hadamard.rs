use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

use super::{PhysicalWeightComponentBinding, ResolvedWeightBinding, WeightComponentRole};
use crate::vnext::{ResolvedStorageComponent, VNextError};

/// Transpose the two outer feature axes before applying input signs.
/// Dimensions are fastest-axis-first: `[inner, first, second]` becomes
/// `[inner, second, first]`. The complete last tensor axis is permuted;
/// token and batch coordinates are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedFeatureTranspose {
    pub inner_extent: u64,
    pub first_outer_extent: u64,
    pub second_outer_extent: u64,
}

impl GroupedFeatureTranspose {
    pub fn width(&self) -> Option<u64> {
        if self.inner_extent == 0 || self.first_outer_extent == 0 || self.second_outer_extent == 0 {
            return None;
        }
        self.inner_extent
            .checked_mul(self.first_outer_extent)?
            .checked_mul(self.second_outer_extent)
    }
}

/// Signs span the full last-axis width, not one repeated Hadamard block.
/// Explicit signs bind immutable, exact-contiguous `TransformSigns` F32
/// components whose source has validated every value as either -1 or +1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HadamardSigns {
    Identity,
    Explicit(PhysicalWeightComponentBinding),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum HadamardApplication {
    /// Consume `H(signs * permutation(input))` before the packed matrix.
    BeforeMatmul {
        input_permutation: Option<GroupedFeatureTranspose>,
    },
    /// Restore a looked-up latent row with `signs * H(row)`.
    /// A projection-input permutation has no meaning in this direction.
    AfterEmbeddingLookup,
}

/// Normalized, blockwise Sylvester Walsh-Hadamard on the complete last axis.
///
/// For block width B, H[i,j] = (-1)^popcount(i & j) / sqrt(B).
/// Providers perform butterflies, normalization and intermediate storage in
/// F32; no F16 narrowing is allowed between this transform and its matrix
/// projection. Embedding output is converted only after H and output signs.
/// The logical operation's output dtype remains its declared dtype.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HadamardTransformSpec {
    pub block_size: NonZeroU32,
    pub signs: HadamardSigns,
    pub application: HadamardApplication,
}

impl HadamardTransformSpec {
    pub fn validate(&self, width: u64) -> Result<(), VNextError> {
        let block = self.block_size.get();
        if !block.is_power_of_two() || width == 0 || !width.is_multiple_of(u64::from(block)) {
            return Err(invalid(
                "block size must be a power of two dividing the nonzero last axis",
            ));
        }
        if let HadamardApplication::BeforeMatmul {
            input_permutation: Some(permutation),
        } = &self.application
        {
            if permutation.width() != Some(width) {
                return Err(invalid(
                    "input permutation shape differs from the full last axis",
                ));
            }
        }
        Ok(())
    }
}

fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("invalid Hadamard transform: {reason}"),
    }
}

/// Only an entire immutable auxiliary component can alias another weight.
/// Callers separately require read-only weight bindings. Logical validation
/// proves that this role is bound by Hadamard, with exact F32 vector storage.
pub(crate) fn same_shared_transform_sign_component(
    left: &ResolvedWeightBinding,
    left_storage: &ResolvedStorageComponent,
    right: &ResolvedWeightBinding,
    right_storage: &ResolvedStorageComponent,
) -> bool {
    if left_storage != right_storage {
        return false;
    }
    left.components().iter().any(|component| {
        Some(component.component_id()) == left_storage.component_id()
            && component.role() == WeightComponentRole::TransformSigns
            && right.components().iter().any(|other| other == component)
    })
}
