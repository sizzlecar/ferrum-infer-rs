//! One actual FastH3 DiT block, with the original compact time table shared
//! across both norms and final gated residual. No scheduler or executor here.
use super::{vsa::*, *};
use crate::tensor_program_fragment::{self as fragment, Fragment, StoredComposition};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct FastH3DitBlockLimits {
    pub modulation: FastH3ModulationLimits,
    pub attention_input: FastH3AttentionInputLimits,
    pub attention_core: FastH3AttentionCoreLimits,
    pub ffn: FastH3FfnLimits,
    pub maximum_plan_nodes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3DitBlockLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.maximum_plan_nodes == 0
            || self.maximum_stored_bytes == 0
            || self.maximum_metadata_bytes == 0
        {
            return Err(invalid("positive composition capacities required"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3DitBlockConfig {
    norm1: FastH3ModulationConfig,
    attention_input: FastH3AttentionInputConfig,
    attention_core: FastH3AttentionCoreConfig,
    norm2: FastH3ModulationConfig,
    ffn: FastH3FfnConfig,
}
impl FastH3DitBlockConfig {
    pub fn with_ffn_chunk_rows(mut self, rows: Option<std::num::NonZeroU64>) -> Self {
        self.ffn = self.ffn.with_chunk_rows(rows);
        self
    }
    pub(super) fn composition(
        &self,
        limits: FastH3DitBlockLimits,
    ) -> Result<StoredComposition, VNextError> {
        program::compose(self, limits)
    }
    pub fn block(&self) -> usize {
        self.attention_input.block()
    }
    pub fn batches(&self) -> u64 {
        self.attention_input.batches()
    }
    pub fn packed_rows(&self) -> u64 {
        self.attention_input.packed_rows()
    }
    pub fn hidden_size(&self) -> u64 {
        self.attention_input.hidden_size()
    }
    pub fn rotary_dim(&self) -> u64 {
        self.attention_input.rotary_dim()
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.batches()
            .checked_mul(self.packed_rows())
            .ok_or_else(|| invalid("packed batch rows overflow"))
    }
    /// The live Time table adapter verifies the block, width and exact time
    /// bits against this same config before admitting one compact input copy.
    pub fn modulation_config(&self) -> &FastH3ModulationConfig {
        &self.norm1
    }
    /// Scalars not captured by physical weight declarations. Batch extents and
    /// per-request role routing are checked separately by the batch adapter.
    pub(in crate::fast_h3) fn same_numerical_geometry(&self, other: &Self) -> bool {
        self.norm1.normalization_epsilon() == other.norm1.normalization_epsilon()
            && self.norm2.normalization_epsilon() == other.norm2.normalization_epsilon()
            && self
                .attention_input
                .same_qk_numerical_geometry(&other.attention_input)
    }
    pub fn region_digest(&self) -> &str {
        self.attention_core.region_digest()
    }
    pub fn profile_id(&self) -> &'static str {
        if self.attention_core.profile_id() == FastH3AttentionCoreFamily::VENDOR_FUSED_PROFILE_ID {
            FastH3DitBlockFamily::VENDOR_FUSED_PROFILE_ID
        } else {
            FastH3DitBlockFamily::PROFILE_ID
        }
    }
    pub fn compile_options(
        &self,
        l: FastH3DitBlockLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(program::compose(self, l)?.options)
    }
    fn validate(&self, l: FastH3DitBlockLimits) -> Result<(), VNextError> {
        l.validate()?;
        let rows = self.rows()?;
        if self.norm1.stage() != FastH3ModulationStage::Attention
            || self.norm2.stage() != FastH3ModulationStage::FeedForward
            || [
                self.norm1.block(),
                self.norm2.block(),
                self.attention_core.block(),
                self.ffn.block(),
            ]
            .iter()
            .any(|&b| b != self.block())
            || [self.norm1.rows(), self.norm2.rows(), self.ffn.rows()]
                .iter()
                .any(|&r| r != rows)
            || [self.norm1.width(), self.norm2.width(), self.ffn.hidden()]
                .iter()
                .any(|&h| h != self.hidden_size())
            || !self.norm1.time_values().iter().map(|t| t.to_bits()).eq(self
                .norm2
                .time_values()
                .iter()
                .map(|t| t.to_bits()))
            || !self
                .norm1
                .parameter_row_runs()
                .eq(self.norm2.parameter_row_runs())
            || !self.norm1.same_routing(&self.norm2)
            || self.attention_core.batches() != self.batches()
            || self.attention_core.packed_rows() != self.packed_rows()
            || self.attention_core.padded_rows() != self.attention_input.tiled_rows()
            || self.attention_core.columns()? != self.attention_input.output_width()?
            || serde_json::to_vec(self).map_err(invalid)?.len() > l.maximum_metadata_bytes
        {
            return Err(invalid(
                "block, time routing, stage, batch geometry or capacity differs",
            ));
        }
        Ok(())
    }
    fn final_attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let (lengths, rows) = self
            .norm2
            .parameter_row_runs()
            .map(|(n, r)| {
                Ok((
                    i64::try_from(n).map_err(invalid)?,
                    i64::try_from(r).map_err(invalid)?,
                ))
            })
            .collect::<Result<Vec<_>, VNextError>>()?
            .into_iter()
            .unzip();
        Ok(BTreeMap::from([
            (id("run_lengths")?, SemanticValue::Integers(lengths)),
            (id("parameter_rows")?, SemanticValue::Integers(rows)),
            (
                id("gate_column")?,
                SemanticValue::Unsigned(
                    self.hidden_size()
                        .checked_mul(5)
                        .ok_or_else(|| invalid("gate column overflow"))?,
                ),
            ),
        ]))
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 DiT block: {e}"),
    }
}
mod family;
mod program;
mod source;
pub use family::FastH3DitBlockFamily;
pub use source::FastH3DitBlockSource;
