//! Pure product-range planning. No execution resources or statistical features.

use super::PrefillChunk;
use ferrum_runtime::CheckpointBoundaryConstraint;
use ferrum_types::{FerrumError, Result};

/// A stable rule for one executor instance. Unknown is not an implicit promise
/// that the candidate range is an exact prediction of completed work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillShapePolicy {
    Unknown,
    Identity,
    PromptTail(CheckpointBoundaryConstraint),
}

/// The projected product range; not a capacity or execution permit. Callers
/// must revalidate the original request/frontier after any intervening await.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct PrefillShapePlan {
    chunk: PrefillChunk,
    exact: bool,
}

impl PrefillShapePlan {
    pub const fn chunk(self) -> PrefillChunk {
        self.chunk
    }

    pub const fn is_exact(self) -> bool {
        self.exact
    }
}

impl PrefillShapePolicy {
    /// Project a validated candidate using its current start and context.
    /// Original prompts may stop at the last legal reusable boundary. Replay
    /// keeps its full candidate, and a candidate before the boundary is intact.
    /// Re-projecting the result is idempotent; no independent late rewrite is
    /// needed or permitted for an already-priced exact range.
    pub fn project(self, chunk: PrefillChunk, original_prompt: bool) -> Result<PrefillShapePlan> {
        let exact = self != Self::Unknown;
        let Self::PromptTail(boundaries) = self else {
            return Ok(PrefillShapePlan { chunk, exact });
        };
        if !original_prompt {
            return Ok(PrefillShapePlan { chunk, exact });
        }
        let processed = u64::try_from(chunk.tokens_processed())
            .map_err(|_| FerrumError::backend("prefill start exceeds boundary domain"))?;
        let total = u64::try_from(chunk.total_prompt_tokens())
            .map_err(|_| FerrumError::backend("prefill context exceeds boundary domain"))?;
        let Some(boundary) = boundaries.prompt_tail_boundary(processed, total) else {
            return Ok(PrefillShapePlan { chunk, exact });
        };
        let boundary = usize::try_from(boundary)
            .map_err(|_| FerrumError::backend("prefill boundary exceeds token domain"))?;
        if boundary >= chunk.end() {
            return Ok(PrefillShapePlan { chunk, exact });
        }
        let tokens = boundary
            .checked_sub(chunk.tokens_processed())
            .ok_or_else(|| {
                FerrumError::backend("prefill boundary precedes the original frontier")
            })?;
        Ok(PrefillShapePlan {
            chunk: PrefillChunk::new(
                chunk.tokens_processed(),
                tokens,
                chunk.total_prompt_tokens(),
            )?,
            exact,
        })
    }
}

#[cfg(test)]
mod tests;
