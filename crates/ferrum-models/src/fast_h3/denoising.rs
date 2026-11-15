//! Reusable hot path: current F32 media -> original projections -> packed
//! refined text/media -> all checkpoint DiT blocks. No solver, executor or
//! resource authority is introduced by this composition.
use super::{vsa::*, *};
use crate::tensor_program_fragment::{
    Fragment, StoredComposition, StoredCompositionIdentity, StoredCompositionLimits,
};
use ferrum_kernels::native_ops::sage_vsa_fp4::CheckedVsaRegions;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
#[derive(Clone, Copy, Debug)]
pub struct FastH3DenoisingLimits {
    pub input: FastH3InputLimits,
    pub backbone: FastH3DitBackboneLimits,
    pub maximum_plan_nodes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3DenoisingConfig {
    media: FastH3InputProjectionConfig,
    packing: FastH3InputPackingConfig,
    backbone: FastH3DitBackboneConfig,
    #[serde(default)]
    schedule: crate::TensorProgramSchedule,
}
impl FastH3DenoisingConfig {
    pub fn with_schedule(mut self, schedule: crate::TensorProgramSchedule) -> Self {
        self.schedule = schedule;
        self
    }
    pub fn with_ffn_chunk_rows(mut self, rows: Option<std::num::NonZeroU64>) -> Self {
        self.backbone = self.backbone.with_ffn_chunk_rows(rows);
        self
    }
    pub fn backbone(&self) -> &FastH3DitBackboneConfig {
        &self.backbone
    }
    pub fn media(&self) -> &FastH3InputProjectionConfig {
        &self.media
    }
    pub fn compile_options(
        &self,
        l: FastH3DenoisingLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(compose(self, l)?.options)
    }
    pub fn profile_id(&self) -> &'static str {
        if self.backbone.profile_id() == FastH3DitBackboneFamily::VENDOR_FUSED_PROFILE_ID {
            FastH3DenoisingFamily::VENDOR_FUSED_PROFILE_ID
        } else {
            FastH3DenoisingFamily::PROFILE_ID
        }
    }
}
fn compose(
    c: &FastH3DenoisingConfig,
    l: FastH3DenoisingLimits,
) -> Result<StoredComposition, VNextError> {
    if c.backbone.layers() == 0
        || c.backbone.batches() != c.packing.batches()
        || c.media.phase() != FastH3InputProjectionPhase::Media
        || c.packing.sequence_rows()? != c.backbone.packed_rows()
        || c.packing.hidden_size() != c.backbone.hidden_size()
        || c.media.hidden_size() != c.backbone.hidden_size()
        || !c
            .media
            .rows()
            .iter()
            .zip(c.packing.request_rows())
            .all(|(&total, rows)| rows.checked_mul(c.backbone.batches()) == Some(total))
    {
        return Err(invalid(
            "denoising phase, request count or packed geometry differs",
        ));
    }
    let media = Fragment::family(
        &FastH3InputProjectionFamily::new(l.input)?,
        &c.media,
        c.media.phase().profile_id(),
        c.media.compile_options()?.tensor_specs(),
        BTreeMap::from([
            (
                id(FastH3InputProjectionFamily::VIDEO_INPUT_ID)?,
                id(FastH3DenoisingFamily::VIDEO_INPUT_ID)?,
            ),
            (
                id(FastH3InputProjectionFamily::AUDIO_INPUT_ID)?,
                id(FastH3DenoisingFamily::AUDIO_INPUT_ID)?,
            ),
        ]),
    )?;
    let packing = Fragment::family(
        &FastH3InputPackingFamily::new(l.input)?,
        &c.packing,
        FastH3InputPackingFamily::PROFILE_ID,
        c.packing.compile_options()?.tensor_specs(),
        BTreeMap::from([
            (
                id(FastH3InputPackingFamily::VIDEO_ID)?,
                id(FastH3InputProjectionFamily::VIDEO_OUTPUT_ID)?,
            ),
            (
                id(FastH3InputPackingFamily::AUDIO_ID)?,
                id(FastH3InputProjectionFamily::AUDIO_OUTPUT_ID)?,
            ),
            (
                id(FastH3InputPackingFamily::REFINED_TEXT_ID)?,
                id(FastH3DenoisingFamily::TEXT_INPUT_ID)?,
            ),
        ]),
    )?;
    let mut mapping = BTreeMap::from([
        (
            id(FastH3DitBackboneFamily::HIDDEN_ID)?,
            id(FastH3InputPackingFamily::OUTPUT_ID)?,
        ),
        (
            id(FastH3DitBackboneFamily::COSINE_ID)?,
            id(FastH3DenoisingFamily::COSINE_ID)?,
        ),
        (
            id(FastH3DitBackboneFamily::SINE_ID)?,
            id(FastH3DenoisingFamily::SINE_ID)?,
        ),
    ]);
    let mut inputs = [
        FastH3DenoisingFamily::VIDEO_INPUT_ID,
        FastH3DenoisingFamily::AUDIO_INPUT_ID,
        FastH3DenoisingFamily::TEXT_INPUT_ID,
        FastH3DenoisingFamily::COSINE_ID,
        FastH3DenoisingFamily::SINE_ID,
    ]
    .into_iter()
    .map(id)
    .collect::<Result<Vec<_>, _>>()?;
    for i in 0..c.backbone.layers() {
        let table = c.backbone.table_input_id(i)?;
        mapping.insert(table.clone(), table.clone());
        inputs.push(table);
    }
    let backbone = Fragment::composition(c.backbone.composition(l.backbone)?, mapping, None)?;
    StoredComposition::join(
        StoredCompositionIdentity {
            family: id(FastH3DenoisingFamily::FAMILY_ID)?,
            profile: NumericalProfileId::new(c.profile_id()).map_err(invalid)?,
            format: id("weight-format.safetensors.dense")?,
            layout: id("weight-layout.fast_h3.denoising.v1")?,
            block: "block.fast_h3.denoising".into(),
        },
        vec![media, packing, backbone],
        inputs,
        vec![id(FastH3DenoisingFamily::OUTPUT_ID)?],
        id(FastH3DenoisingFamily::OUTPUT_ID)?,
        StoredCompositionLimits {
            maximum_nodes: l.maximum_plan_nodes,
            maximum_stored_bytes: l.maximum_stored_bytes,
            maximum_metadata_bytes: l.maximum_metadata_bytes,
        },
    )?
    .with_schedule(c.schedule)
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 denoising: {e}"),
    }
}
mod family;
mod request;
mod source;
pub use family::FastH3DenoisingFamily;
pub use request::{FastH3DenoisingBatch, FastH3DenoisingBatchPlan, FastH3DenoisingRequest};
pub use source::{FastH3DenoisingSource, OwnedFastH3DenoisingSource};
