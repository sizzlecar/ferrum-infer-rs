//! Same-graph FastH3 attention inputs: immutable tile maps, tile-first Q/K/V,
//! original packed-order gate, single-round QK/RoPE, and unsmoothed tile means.
//! No sparse-attention decision or backend execution is implied by this family.
use super::{vsa::*, *};
use crate::tensor_program_fragment::{self as fragment, Fragment, StoredComposition};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod family;
mod program;
mod source;
pub use family::FastH3AttentionInputFamily;
pub use source::FastH3AttentionInputSource;

#[derive(Clone, Copy, Debug)]
pub struct FastH3AttentionInputLimits {
    pub projection: FastH3AttentionProjectionLimits,
    pub qk: FastH3QkNormRopeLimits,
    pub transport: FastH3VsaRowTransportLimits,
    pub pooling: FastH3VsaPoolLimits,
    pub maximum_plan_nodes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3AttentionInputLimits {
    fn validate(self) -> Result<(), VNextError> {
        FastH3AttentionProjectionFamily::new(self.projection)?;
        FastH3QkNormRopeFamily::new(self.qk)?;
        FastH3VsaRowTransportFamily::new(self.transport)?;
        FastH3VsaPoolFamily::new(self.pooling)?;
        if self.maximum_plan_nodes == 0
            || self.maximum_stored_bytes == 0
            || self.maximum_metadata_bytes == 0
        {
            return Err(invalid("composition capacities must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3AttentionInputConfig {
    tile_hidden: FastH3VsaRowTransportConfig,
    tile_rotary: FastH3VsaRowTransportConfig,
    qkv: FastH3AttentionProjectionConfig,
    gate: FastH3AttentionProjectionConfig,
    qk: FastH3QkNormRopeConfig,
    pooling: FastH3VsaPoolConfig,
}
impl FastH3AttentionInputConfig {
    pub fn block(&self) -> usize {
        self.qkv.block()
    }
    pub fn batches(&self) -> u64 {
        self.tile_hidden.batches()
    }
    pub fn packed_rows(&self) -> u64 {
        self.tile_hidden.source_rows()
    }
    pub fn tiled_rows(&self) -> u64 {
        self.tile_hidden.output_rows()
    }
    pub fn hidden_size(&self) -> u64 {
        self.qkv.hidden()
    }
    pub fn output_width(&self) -> Result<u64, VNextError> {
        self.qkv.output_width()
    }
    pub fn rotary_dim(&self) -> u64 {
        self.qk.rotary_dim()
    }
    pub(in crate::fast_h3) fn same_qk_numerical_geometry(&self, other: &Self) -> bool {
        self.qk.numerical_geometry() == other.qk.numerical_geometry()
    }
    pub fn tile_count(&self) -> u64 {
        self.pooling.tiles()
    }
    pub fn compile_options(
        &self,
        limits: FastH3AttentionInputLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(program::compose(self, limits)?.options)
    }
    fn validate(&self, limits: FastH3AttentionInputLimits) -> Result<(), VNextError> {
        limits.validate()?;
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("configuration exceeds metadata capacity"));
        }
        let rows = self
            .batches()
            .checked_mul(self.tiled_rows())
            .ok_or_else(|| invalid("tiled batch row product overflows"))?;
        let packed = self
            .batches()
            .checked_mul(self.packed_rows())
            .ok_or_else(|| invalid("packed batch row product overflows"))?;
        if self.tile_hidden.kind() != FastH3VsaRowTransport::Tile
            || self.tile_hidden.columns() != self.hidden_size()
            || self
                .tile_hidden
                .with_columns(self.rotary_dim(), limits.transport)?
                != self.tile_rotary
            || self.qkv.group() != FastH3AttentionProjectionGroup::Qkv
            || self.gate.group() != FastH3AttentionProjectionGroup::Gate
            || self.qkv.rows() != rows
            || self.qk.rows() != rows
            || self.gate.rows() != packed
            || self.qkv.block() != self.gate.block()
            || self.qkv.block() != self.qk.block()
            || self.qkv.hidden() != self.gate.hidden()
            || self.qkv.heads() != self.gate.heads()
            || self.qkv.heads() != self.qk.heads()
            || self.qkv.head_dim() != self.gate.head_dim()
            || self.qkv.head_dim() != self.qk.head_dim()
            || self.pooling.batches() != self.batches()
            || self.pooling.heads() != self.qkv.heads()
            || self.pooling.head_dim() != self.qkv.head_dim()
            || self.pooling.input_shape()?
                != [self.batches(), self.tiled_rows(), self.output_width()?]
        {
            return Err(invalid(
                "projection, tile-map, rotary and pooling geometry disagree",
            ));
        }
        Ok(())
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 attention input: {e}"),
    }
}
