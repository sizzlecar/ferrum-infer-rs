//! Full-depth, independent temporal clip. Packing/rotary preparation owns no
//! execution authority; runtime outputs retain the caller's original owners.
use super::*;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod family;
mod program;
mod source;
mod transport;
pub use family::FastH3VideoClipFamily;
pub use source::FastH3VideoClipSource;
pub use transport::FastH3VideoRawClip;

#[derive(Clone, Copy, Debug)]
pub struct FastH3VideoClipLimits {
    pub blocks: FastH3VideoDecoderBlocksLimits,
    pub maximum_plan_nodes: usize,
    pub maximum_activation_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3VideoClipConfig {
    architecture: FastH3VideoVaeConfig,
    // [batch,time,height,width], in latent coordinates. Time is not batch.
    grid: [u64; 4],
    blocks: FastH3VideoDecoderBlocksConfig,
    outer: Vec<Outer>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Outer {
    original: WeightComponentSpec,
    component: WeightComponentSpec,
}
impl FastH3VideoClipConfig {
    pub fn architecture(&self) -> &FastH3VideoVaeConfig {
        &self.architecture
    }
    pub fn latent_grid(&self) -> [u64; 4] {
        self.grid
    }
    pub fn patch_rows_per_clip(&self) -> Result<u64, VNextError> {
        product(&self.grid[1..])
    }
    pub fn patch_rows(&self) -> Result<u64, VNextError> {
        product(&self.grid)
    }
    pub fn sequence_rows(&self) -> Result<u64, VNextError> {
        self.patch_rows_per_clip()?
            .checked_add(self.architecture.register_tokens() as u64 + 1)
            .ok_or_else(|| invalid("sequence extent overflow"))
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        product(&[self.grid[0], self.sequence_rows()?])
    }
    pub fn compile_options(
        &self,
        limits: FastH3VideoClipLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(program::compose(self, limits)?.options)
    }
    fn validate(&self, limits: FastH3VideoClipLimits) -> Result<(), VNextError> {
        self.architecture.validate().map_err(invalid)?;
        self.blocks.validate_full_architecture(&self.architecture)?;
        if self.grid.contains(&0)
            || self.grid[0] > limits.blocks.maximum_rows
            || self.rows()? > limits.blocks.maximum_rows
            || limits.maximum_activation_bytes == 0
            || limits.maximum_plan_nodes == 0
            || self.blocks.sequence_lengths().len() as u64 != self.grid[0]
            || self
                .blocks
                .sequence_lengths()
                .iter()
                .any(|r| *r != self.sequence_rows().unwrap_or(0))
        {
            return Err(invalid("invalid clip geometry or capacity"));
        }
        let entries = source::entries(&self.architecture)?;
        if self.outer.len() != entries.len() {
            return Err(invalid("incomplete outer inventory"));
        }
        for (o, e) in self.outer.iter().zip(entries) {
            if o.original.external_names != [e.name.to_owned()]
                || o.original.dimensions != e.original
                || o.original.encoding
                    != (WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    })
                || o.original.role != WeightComponentRole::Values
                || !o.original.required
                || source::component(&o.original, e.shape, e.dtype)? != o.component
            {
                return Err(invalid(
                    "outer weight changes original slot, conversion, or shape",
                ));
            }
        }
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.blocks.maximum_metadata_bytes {
            return Err(invalid("clip metadata exceeds capacity"));
        }
        Ok(())
    }
}
fn product(values: &[u64]) -> Result<u64, VNextError> {
    values.iter().try_fold(1u64, |n, v| {
        n.checked_mul(*v)
            .filter(|n| *n <= i64::MAX as u64)
            .ok_or_else(|| invalid("clip extent overflow"))
    })
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 video clip: {e}"),
    }
}
fn value(n: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.fast_h3.video_clip.{n}"))
}
fn tensor(rows: u64, width: u64, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![rows, width],
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
