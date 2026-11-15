//! Fixed FP4 VSA's dense FP8 prefix. Input Q/K/V are already tiled and Q/K
//! normalized/rotated; pooling of unsmoothed K is a separate predecessor.
//! One composed Core graph centers every padded K row, rounds Q/K per head,
//! gathers valid K/V in ascending tile order, then performs dense attention.
use super::*;
use crate::fast_h3::FastH3TransformerConfig;
use crate::tensor_program_fragment::{self as fragment, Fragment, StoredComposition};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Explicitly distinct dense-attention arithmetic, without implicit fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3PrefixAttentionArithmetic {
    ReferenceSequential,
    VendorFused,
}
impl FastH3PrefixAttentionArithmetic {
    fn operation_id(self) -> &'static str {
        match self {
            Self::ReferenceSequential => CROSS_ATTENTION_BF16_OPERATION_ID,
            Self::VendorFused => CROSS_ATTENTION_FUSED_BF16_OPERATION_ID,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FastH3PrefixAttentionLimits {
    pub center: FastH3KeyCenterLimits,
    pub transport: FastH3VsaRowTransportLimits,
    pub maximum_plan_nodes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3PrefixAttentionLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.maximum_plan_nodes == 0
            || self.maximum_stored_bytes == 0
            || self.maximum_metadata_bytes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3PrefixAttentionConfig {
    center: FastH3KeyCenterConfig,
    valid: FastH3VsaRowTransportConfig,
    query: FastH3VsaRowTransportConfig,
    prefix_rows: u64,
    head_dim: u64,
    arithmetic: FastH3PrefixAttentionArithmetic,
}
impl FastH3PrefixAttentionConfig {
    pub fn with_attention_arithmetic(
        &self,
        arithmetic: FastH3PrefixAttentionArithmetic,
        limits: FastH3PrefixAttentionLimits,
    ) -> Result<Self, VNextError> {
        let mut c = self.clone();
        c.arithmetic = arithmetic;
        c.validate(limits)?;
        Ok(c)
    }
    pub fn profile_id(&self) -> &'static str {
        match self.arithmetic {
            FastH3PrefixAttentionArithmetic::ReferenceSequential => {
                FastH3PrefixAttentionFamily::PROFILE_ID
            }
            FastH3PrefixAttentionArithmetic::VendorFused => {
                FastH3PrefixAttentionFamily::VENDOR_FUSED_PROFILE_ID
            }
        }
    }
    pub fn input_shape(&self) -> Vec<u64> {
        self.center.input_shape()
    }
    pub fn output_shape(&self) -> Vec<u64> {
        vec![self.batches(), self.prefix_rows, self.columns()]
    }
    pub fn prefix_rows(&self) -> u64 {
        self.prefix_rows
    }
    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn valid_key_rows(&self) -> u64 {
        self.valid.output_rows()
    }
    pub fn padded_rows(&self) -> u64 {
        self.valid.source_rows()
    }
    pub fn batches(&self) -> u64 {
        self.valid.batches()
    }
    pub fn columns(&self) -> u64 {
        self.valid.columns()
    }
    pub fn compile_options(
        &self,
        l: FastH3PrefixAttentionLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(program::compose(self, l)?.options)
    }
    fn validate(&self, l: FastH3PrefixAttentionLimits) -> Result<(), VNextError> {
        l.validate()?;
        self.center.compile_options(l.center)?;
        self.valid.compile_options(l.transport)?;
        self.query.compile_options(l.transport)?;
        if self.batches() > STRUCTURED_ATTENTION_BF16_MAX_BATCH as u64
            || self.valid.kind() != FastH3VsaRowTransport::ValidTileRows
            || self.query.kind() != FastH3VsaRowTransport::PrefixQueryRows
            || self.query.batches() != self.batches()
            || self.query.columns() != self.columns()
            || self.query.source_rows() != self.padded_rows()
            || self.query.output_rows() != self.prefix_rows
            || self.input_shape() != vec![self.batches(), self.padded_rows(), self.columns()]
            || self.prefix_rows == 0
            || self.prefix_rows >= self.padded_rows()
            || self.prefix_rows % FAST_H3_VSA_TILE_ROWS as u64 != 0
            || self.head_dim == 0
            || self.columns() % self.head_dim != 0
            || serde_json::to_vec(self).map_err(invalid)?.len() > l.maximum_metadata_bytes
        {
            return Err(invalid("invalid prefix geometry, order or capacity"));
        }
        let x = resolved(vec![self.batches(), self.prefix_rows, self.columns()])?;
        HeadFp8RoundtripBf16Geometry::new(
            &x,
            &x,
            &HeadFp8RoundtripBf16Geometry::attributes(self.head_dim)?,
        )?;
        let q = resolved(vec![self.batches() * self.prefix_rows, self.columns()])?;
        let kv = resolved(vec![self.batches() * self.valid_key_rows(), self.columns()])?;
        CrossAttentionBf16Geometry::new(&q, &kv, &kv, &q, &self.attention_attributes()?)?;
        Ok(())
    }
    fn attention_attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        let bits = scale.to_bits();
        let shift = 150u32
            .checked_sub((bits >> 23) & 255)
            .filter(|&n| n < 64)
            .ok_or_else(|| invalid("scale exponent is not representable"))?;
        let scale =
            CanonicalRational::new(i64::from((bits & 0x7fffff) | (1 << 23)), 1u64 << shift)?;
        Ok(BTreeMap::from([
            (id("batch")?, SemanticValue::Unsigned(self.batches())),
            (id("head_dim")?, SemanticValue::Unsigned(self.head_dim)),
            (id("scale")?, SemanticValue::Rational(scale)),
            (
                id("key_run_counts")?,
                SemanticValue::Integers(vec![1; self.batches() as usize]),
            ),
            (
                id("key_run_lengths")?,
                SemanticValue::Integers(vec![
                    i64::try_from(self.valid_key_rows())
                        .map_err(invalid)?;
                    self.batches() as usize
                ]),
            ),
            (
                id("key_run_valid")?,
                SemanticValue::Integers(vec![1; self.batches() as usize]),
            ),
        ]))
    }
}
pub struct FastH3PrefixAttentionSource {
    valid: FastH3VsaRowTransportSource,
    query: FastH3VsaRowTransportSource,
}
impl FastH3PrefixAttentionSource {
    pub fn prepare(
        layout: &FastH3VsaLayout,
        batches: u64,
        transformer: &FastH3TransformerConfig,
        host: &HostMemoryAccount,
        limits: FastH3PrefixAttentionLimits,
    ) -> Result<(FastH3PrefixAttentionConfig, Self), VNextError> {
        limits.validate()?;
        let center =
            FastH3KeyCenterConfig::from_layout(layout, batches, transformer, limits.center)?;
        let prefix_rows = u64::try_from(layout.prefix_tiles())
            .map_err(invalid)?
            .checked_mul(FAST_H3_VSA_TILE_ROWS as u64)
            .ok_or_else(|| invalid("prefix rows overflow"))?;
        if prefix_rows == 0 || prefix_rows >= layout.padded_rows() as u64 {
            return Err(invalid(
                "prefix path requires both a nonempty prefix and video query tail",
            ));
        }
        let (valid, source) = FastH3VsaRowTransportSource::prepare(
            layout,
            FastH3VsaRowTransport::ValidTileRows,
            batches,
            center.input_shape()[2],
            host,
            limits.transport,
        )?;
        let (query, query_source) = FastH3VsaRowTransportSource::prepare(
            layout,
            FastH3VsaRowTransport::PrefixQueryRows,
            batches,
            center.input_shape()[2],
            host,
            limits.transport,
        )?;
        let c = FastH3PrefixAttentionConfig {
            center,
            valid,
            query,
            prefix_rows,
            head_dim: transformer.attention_head_dim() as u64,
            arithmetic: FastH3PrefixAttentionArithmetic::ReferenceSequential,
        };
        program::compose(&c, limits)?;
        Ok((
            c,
            Self {
                valid: source,
                query: query_source,
            },
        ))
    }
}
impl WeightComponentSource for FastH3PrefixAttentionSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if c.id == self.query.component_spec().id {
            self.query.component(c)
        } else {
            self.valid.component(c)
        }
    }
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn resolved(dimensions: Vec<u64>) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(
        dimensions,
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 prefix attention: {e}"),
    }
}
mod family;
mod program;
pub use family::FastH3PrefixAttentionFamily;
#[cfg(test)]
mod tests;
