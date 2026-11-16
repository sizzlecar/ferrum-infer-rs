//! The complete checkpoint block stack as one retained tensor component.
//! Geometry and text tags are fixed for this admitted cohort. Evaluation is
//! carried only by compact live role-table inputs, never by stored weights.
use super::{vsa::*, *};
use crate::tensor_program_fragment::{
    self as fragment, Fragment, StoredComposition, StoredCompositionIdentity,
    StoredCompositionLimits,
};
use ferrum_kernels::native_ops::sage_vsa_fp4::CheckedVsaRegions;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct FastH3DitBackboneLimits {
    pub block: FastH3DitBlockLimits,
    pub maximum_layers: usize,
    pub maximum_plan_nodes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3DitBackboneLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.maximum_layers == 0
            || self.maximum_plan_nodes == 0
            || self.maximum_stored_bytes == 0
            || self.maximum_metadata_bytes == 0
        {
            return Err(invalid("positive stack capacities required"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3DitBackboneConfig {
    blocks: Vec<FastH3DitBlockConfig>,
}
impl FastH3DitBackboneConfig {
    pub fn with_ffn_chunk_rows(mut self, rows: Option<std::num::NonZeroU64>) -> Self {
        self.blocks = self
            .blocks
            .into_iter()
            .map(|b| b.with_ffn_chunk_rows(rows))
            .collect();
        self
    }
    pub(super) fn composition(
        &self,
        limits: FastH3DitBackboneLimits,
    ) -> Result<StoredComposition, VNextError> {
        compose(self, limits)
    }
    pub fn layers(&self) -> usize {
        self.blocks.len()
    }
    pub fn blocks(&self) -> &[FastH3DitBlockConfig] {
        &self.blocks
    }
    pub fn batches(&self) -> u64 {
        self.blocks[0].batches()
    }
    pub fn packed_rows(&self) -> u64 {
        self.blocks[0].packed_rows()
    }
    pub fn hidden_size(&self) -> u64 {
        self.blocks[0].hidden_size()
    }
    pub fn rotary_dim(&self) -> u64 {
        self.blocks[0].rotary_dim()
    }
    pub fn region_digest(&self) -> &str {
        self.blocks[0].region_digest()
    }
    pub fn profile_id(&self) -> &'static str {
        if self.blocks[0].profile_id() == FastH3DitBlockFamily::VENDOR_FUSED_PROFILE_ID {
            FastH3DitBackboneFamily::VENDOR_FUSED_PROFILE_ID
        } else {
            FastH3DitBackboneFamily::PROFILE_ID
        }
    }
    pub fn table_input_id(&self, block: usize) -> Result<ProgramValueId, VNextError> {
        if block >= self.layers() {
            return Err(invalid("block outside stack"));
        }
        id(&format!("value.fast_h3.dit_backbone.layer{block}.table"))
    }
    pub fn compile_options(
        &self,
        limits: FastH3DitBackboneLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(compose(self, limits)?.options)
    }
    /// Select one live compact input per layer; failed admission drops earlier
    /// inputs automatically. The retained Time tables keep their original class.
    pub fn managed_role_tables(
        &self,
        tables: &[FastH3ConditioningTable],
        schedule: &FastH3ConditioningSchedule,
        evaluations: &[usize],
        host: &HostMemoryAccount,
    ) -> Result<Vec<ManagedHostTensor>, VNextError> {
        if tables.len() != self.layers() {
            return Err(invalid("live table count differs from full stack"));
        }
        self.blocks
            .iter()
            .zip(tables)
            .enumerate()
            .map(|(i, (block, table))| {
                let original = block.modulation_config().managed_role_table(
                    table,
                    schedule,
                    evaluations,
                    host,
                )?;
                ManagedHostTensor::new(
                    self.table_input_id(i)?,
                    original.tensor().clone(),
                    original.bytes().clone(),
                )
            })
            .collect()
    }
    fn validate(&self, l: FastH3DitBackboneLimits) -> Result<(), VNextError> {
        l.validate()?;
        let first = self
            .blocks
            .first()
            .ok_or_else(|| invalid("empty block stack"))?;
        if self.blocks.len() > l.maximum_layers
            || serde_json::to_vec(self).map_err(invalid)?.len() > l.maximum_metadata_bytes
        {
            return Err(invalid("stack layer or metadata capacity exceeded"));
        }
        for (i, block) in self.blocks.iter().enumerate() {
            if block.block() != i
                || block.batches() != first.batches()
                || block.packed_rows() != first.packed_rows()
                || block.hidden_size() != first.hidden_size()
                || block.rotary_dim() != first.rotary_dim()
                || block.region_digest() != first.region_digest()
                || block.profile_id() != first.profile_id()
                || block.modulation_config().conditioning_request_count() != Some(first.batches())
                || !block
                    .modulation_config()
                    .same_routing(first.modulation_config())
                || !block
                    .modulation_config()
                    .time_values()
                    .iter()
                    .map(|t| t.to_bits())
                    .eq(first
                        .modulation_config()
                        .time_values()
                        .iter()
                        .map(|t| t.to_bits()))
            {
                return Err(invalid(
                    "full stack block order, role routing or shared geometry differs",
                ));
            }
        }
        Ok(())
    }
}
fn compose(
    c: &FastH3DitBackboneConfig,
    l: FastH3DitBackboneLimits,
) -> Result<StoredComposition, VNextError> {
    c.validate(l)?;
    let mut fragments = Vec::with_capacity(c.layers());
    let mut inputs = vec![
        id(FastH3DitBackboneFamily::HIDDEN_ID)?,
        id(FastH3DitBackboneFamily::COSINE_ID)?,
        id(FastH3DitBackboneFamily::SINE_ID)?,
    ];
    let mut hidden = id(FastH3DitBackboneFamily::HIDDEN_ID)?;
    for (i, block) in c.blocks.iter().enumerate() {
        let scope = format!("layer{i}");
        // The final block retains its output ID, with earlier blocks explicitly
        // namespaced. No copy/identity kernel is inserted at the boundary.
        let instance = (i + 1 != c.layers()).then_some(scope.as_str());
        let table = c.table_input_id(i)?;
        inputs.push(table.clone());
        fragments.push(Fragment::composition(
            block.composition(l.block)?,
            BTreeMap::from([
                (id(FastH3DitBlockFamily::HIDDEN_ID)?, hidden),
                (
                    id(FastH3DitBlockFamily::COSINE_ID)?,
                    id(FastH3DitBackboneFamily::COSINE_ID)?,
                ),
                (
                    id(FastH3DitBlockFamily::SINE_ID)?,
                    id(FastH3DitBackboneFamily::SINE_ID)?,
                ),
                (id(FastH3DitBlockFamily::TABLE_ID)?, table),
            ]),
            instance,
        )?);
        hidden = fragment::scoped_value(&id(FastH3DitBlockFamily::OUTPUT_ID)?, instance)?;
    }
    StoredComposition::join(
        StoredCompositionIdentity {
            family: id(FastH3DitBackboneFamily::FAMILY_ID)?,
            profile: NumericalProfileId::new(c.profile_id()).map_err(invalid)?,
            format: id("weight-format.safetensors.dense")?,
            layout: id("weight-layout.fast_h3.dit_backbone.v1")?,
            block: "block.fast_h3.dit_backbone".into(),
        },
        fragments,
        inputs,
        vec![hidden.clone()],
        hidden,
        StoredCompositionLimits {
            maximum_nodes: l.maximum_plan_nodes,
            maximum_stored_bytes: l.maximum_stored_bytes,
            maximum_metadata_bytes: l.maximum_metadata_bytes,
        },
    )
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 DiT backbone: {e}"),
    }
}
mod family;
mod source;
pub use family::FastH3DitBackboneFamily;
pub use source::FastH3DitBackboneSource;
