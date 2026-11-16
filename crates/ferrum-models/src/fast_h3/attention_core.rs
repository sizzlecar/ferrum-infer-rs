//! The actual active-compression VSA attention core, composed in one Core plan.
//! Inputs are the seven existing AttentionInput outputs. Original unsmoothed
//! pooled Q/K/V feed one shared score tensor, selection and compression.
use super::vsa::*;
use super::*;
use crate::tensor_program_fragment::{self as fragment, Fragment, StoredComposition};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct FastH3AttentionCoreLimits {
    pub prefix: FastH3PrefixAttentionLimits,
    pub quantize: FastH3VsaQkvQuantizeLimits,
    pub pooled: FastH3PooledAttentionLimits,
    pub pooling: FastH3VsaPoolLimits,
    pub transport: FastH3VsaRowTransportLimits,
    pub head_pack: FastH3HeadAxisPackLimits,
    pub projection: FastH3AttentionOutputLimits,
    pub maximum_region_bytes: u64,
    pub maximum_plan_nodes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3AttentionCoreLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.maximum_region_bytes == 0
            || self.maximum_plan_nodes == 0
            || self.maximum_stored_bytes == 0
            || self.maximum_metadata_bytes == 0
        {
            return Err(invalid("positive composition capacities required"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3AttentionCoreConfig {
    prefix: FastH3PrefixAttentionConfig,
    quantize: FastH3VsaQkvQuantizeConfig,
    pooled: FastH3PooledAttentionConfig,
    video_query: FastH3VsaRowTransportConfig,
    fine_pack: FastH3HeadAxisPackConfig,
    pooled_pack: FastH3HeadAxisPackConfig,
    untile: FastH3VsaRowTransportConfig,
    expand: FastH3VsaRowTransportConfig,
    projection: FastH3AttentionOutputConfig,
    region_digest: String,
}
impl FastH3AttentionCoreConfig {
    pub fn batches(&self) -> u64 {
        self.quantize.batches()
    }
    pub fn heads(&self) -> u64 {
        self.quantize.heads()
    }
    pub fn columns(&self) -> Result<u64, VNextError> {
        self.projection.attention_width()
    }
    pub fn packed_rows(&self) -> u64 {
        self.untile.output_rows()
    }
    pub fn padded_rows(&self) -> u64 {
        self.quantize.key_rows()
    }
    pub fn video_rows(&self) -> u64 {
        self.quantize.query_rows()
    }
    pub fn block(&self) -> usize {
        self.projection.block()
    }
    pub fn region_digest(&self) -> &str {
        &self.region_digest
    }
    pub fn profile_id(&self) -> &'static str {
        if self.prefix.profile_id() == FastH3PrefixAttentionFamily::VENDOR_FUSED_PROFILE_ID {
            FastH3AttentionCoreFamily::VENDOR_FUSED_PROFILE_ID
        } else {
            FastH3AttentionCoreFamily::PROFILE_ID
        }
    }
    pub fn compile_options(
        &self,
        l: FastH3AttentionCoreLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(program::compose(self, l)?.options)
    }
    fn validate(&self, l: FastH3AttentionCoreLimits) -> Result<(), VNextError> {
        l.validate()?;
        let columns = self.columns()?;
        self.prefix.compile_options(l.prefix)?;
        self.quantize.compile_options(l.quantize)?;
        self.pooled.compile_options(l.pooled)?;
        self.video_query.compile_options(l.transport)?;
        self.untile.compile_options(l.transport)?;
        self.expand.compile_options(l.transport)?;
        self.fine_pack.compile_options(l.head_pack)?;
        self.pooled_pack.compile_options(l.head_pack)?;
        if self.prefix.head_dim() != 128
            || self.prefix.batches() != self.batches()
            || self.prefix.padded_rows() != self.padded_rows()
            || self.prefix.valid_key_rows() != self.packed_rows()
            || self.prefix.columns() != columns
            || self.prefix.prefix_rows().checked_add(self.video_rows()) != Some(self.padded_rows())
            || self.pooled.pooled_shape()
                != vec![self.batches(), self.heads(), self.padded_rows() / 128, 128]
            || self.quantize.input_shapes()?[0] != vec![self.batches(), self.video_rows(), columns]
            || self.video_query.kind() != FastH3VsaRowTransport::VideoQueryRows
            || self.video_query.batches() != self.batches()
            || self.video_query.source_rows() != self.padded_rows()
            || self.video_query.output_rows() != self.video_rows()
            || self.video_query.columns() != columns
            || self.fine_pack.input_shape()
                != vec![self.batches(), self.heads(), self.video_rows(), 128]
            || self.fine_pack.kind() != HeadAxisPackKind::Bf16
            || self.pooled_pack.input_shape() != self.pooled.pooled_shape()
            || self.pooled_pack.kind() != HeadAxisPackKind::F32ToBf16
            || self.untile.kind() != FastH3VsaRowTransport::Untile
            || self.untile.source_rows() != self.padded_rows()
            || self.untile.batches() != self.batches()
            || self.untile.columns() != columns
            || self.expand.kind() != FastH3VsaRowTransport::ExpandTileValues
            || self.expand.source_rows() != self.padded_rows() / 128
            || self.expand.output_rows() != self.packed_rows()
            || self.expand.batches() != self.batches()
            || self.expand.columns() != columns
            || self.batches().checked_mul(self.packed_rows()) != Some(self.projection.rows())
            || self.region_digest.len() != 64
            || !self
                .region_digest
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            || serde_json::to_vec(self).map_err(invalid)?.len() > l.maximum_metadata_bytes
        {
            return Err(invalid(
                "attention fragment geometry, map kind, numerical source or capacity differs",
            ));
        }
        Ok(())
    }
    fn attention_attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        Ok(BTreeMap::from([
            (
                id("pattern_sha256")?,
                SemanticValue::Text(self.region_digest.clone()),
            ),
            (
                id("scale")?,
                SemanticValue::Rational(CanonicalRational::new(11_863_283, 134_217_728)?),
            ),
        ]))
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 attention core: {e}"),
    }
}
mod family;
mod program;
mod regions;
mod source;
pub use family::FastH3AttentionCoreFamily;
pub use source::FastH3AttentionCoreSource;
#[cfg(test)]
mod tests;
