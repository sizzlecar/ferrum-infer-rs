//! Complete original decoder blocks over F32 residual rows. Sequence lengths
//! isolate spatial/temporal clips; no block attends across another clip.
use super::{FastH3DecoderAlphaDerivation, FastH3VideoDecoderPackage};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod family;
mod program;
mod source;
pub use family::FastH3VideoDecoderBlocksFamily;
pub use source::FastH3VideoDecoderBlocksSource;

#[derive(Clone, Copy, Debug)]
pub struct FastH3VideoDecoderBlocksLimits {
    pub maximum_rows: u64,
    pub maximum_blocks: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_metadata_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3VideoDecoderBlocksConfig {
    sequences: Vec<u64>,
    hidden: u64,
    ffn: u64,
    head_dim: u64,
    rotary_dim: u64,
    epsilon: CanonicalRational,
    declared_layers: usize,
    blocks: Vec<Block>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Block {
    index: usize,
    norms_and_scales: [WeightComponentSpec; 4],
    projections: [Projection; 6],
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Projection {
    packed: WeightComponentSpec,
    scales: WeightComponentSpec,
    bias: WeightComponentSpec,
    alpha: FastH3DecoderAlphaDerivation,
}
const PROJECTIONS: [&str; 6] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
];
const NORMS: [&str; 4] = ["norm1.weight", "norm2.weight", "scale1", "scale2"];

impl FastH3VideoDecoderBlocksConfig {
    pub(super) fn validate_full_architecture(
        &self,
        a: &super::FastH3VideoVaeConfig,
    ) -> Result<(), VNextError> {
        if !self.covers_full_decoder()
            || self.declared_layers != a.decoder_layers()
            || self.hidden != a.hidden_size().map_err(invalid)? as u64
            || self.ffn != a.ffn_size().map_err(invalid)? as u64
            || self.head_dim != a.attention_head_dim() as u64
            || self.rotary_dim != a.rotary_dimension() as u64
            || (self.epsilon.numerator() as f64 / self.epsilon.denominator() as f64) as f32
                != a.norm_epsilon() as f32
        {
            return Err(invalid(
                "full decoder blocks differ from original architecture",
            ));
        }
        Ok(())
    }
    pub fn sequence_lengths(&self) -> &[u64] {
        &self.sequences
    }
    pub fn selected_blocks(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.blocks.iter().map(|b| b.index)
    }
    pub fn covers_full_decoder(&self) -> bool {
        self.blocks.len() == self.declared_layers
            && self.blocks.first().is_some_and(|b| b.index == 0)
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn rotary_dim(&self) -> u64 {
        self.rotary_dim
    }
    pub fn output_id(&self) -> Result<ProgramValueId, VNextError> {
        value(&format!(
            "block{}.residual2",
            self.blocks
                .last()
                .ok_or_else(|| invalid("empty block selection"))?
                .index
        ))
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.sequences.iter().try_fold(0u64, |n, r| {
            n.checked_add(*r)
                .ok_or_else(|| invalid("row count overflow"))
        })
    }
    pub fn compile_options(
        &self,
        limits: FastH3VideoDecoderBlocksLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate(limits)?;
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
    fn shape(&self, index: usize) -> (u64, u64) {
        match index {
            4 => (2 * self.ffn, self.hidden),
            5 => (self.hidden, self.ffn),
            _ => (self.hidden, self.hidden),
        }
    }
    fn validate(&self, limits: FastH3VideoDecoderBlocksLimits) -> Result<(), VNextError> {
        let epsilon = self.epsilon.numerator() as f64 / self.epsilon.denominator() as f64;
        if limits.maximum_rows == 0
            || limits.maximum_blocks == 0
            || limits.maximum_stored_bytes == 0
            || limits.maximum_metadata_bytes == 0
            || self.sequences.is_empty()
            || self
                .sequences
                .iter()
                .any(|r| *r == 0 || *r > i64::MAX as u64)
            || self.rows()? > limits.maximum_rows
            || self.rows()? > i64::MAX as u64
            || self.hidden == 0
            || self.ffn == 0
            || self.ffn.checked_mul(2).is_none()
            || self.head_dim != 64
            || self.hidden % self.head_dim != 0
            || self.rotary_dim == 0
            || self.rotary_dim > self.head_dim
            || self.rotary_dim % 6 != 0
            || self.blocks.is_empty()
            || self.blocks.len() > limits.maximum_blocks
            || self.blocks.len() > self.declared_layers
            || !(1e-12..=1.0).contains(&epsilon)
        {
            return Err(invalid("invalid decoder block geometry or capacity"));
        }
        for (ordinal, b) in self.blocks.iter().enumerate() {
            if b.index >= self.declared_layers
                || (ordinal > 0 && self.blocks[ordinal - 1].index.checked_add(1) != Some(b.index))
            {
                return Err(invalid("decoder blocks must be a contiguous ordered range"));
            }
            for (suffix, c) in NORMS.iter().zip(&b.norms_and_scales) {
                check_original(
                    c,
                    &format!("decoder.transformer_blocks.{}.{suffix}", b.index),
                    &[self.hidden],
                    ElementType::F32,
                    WeightComponentRole::Values,
                    false,
                )?;
            }
            for (i, p) in b.projections.iter().enumerate() {
                let prefix = format!("decoder.transformer_blocks.{}.{}", b.index, PROJECTIONS[i]);
                let (n, k) = self.shape(i);
                let g = Nvfp4MatrixLayout::new(n, k)?;
                check_original(
                    &p.packed,
                    &format!("{prefix}.packed_weight"),
                    &g.packed_shape(),
                    ElementType::U8,
                    WeightComponentRole::PackedValues,
                    true,
                )?;
                check_original(
                    &p.scales,
                    &format!("{prefix}.weight_inv_scale"),
                    &g.scale_shape(),
                    ElementType::U8,
                    WeightComponentRole::Scales,
                    false,
                )?;
                check_original(
                    &p.bias,
                    &format!("{prefix}.bias"),
                    &[n],
                    ElementType::Bf16,
                    WeightComponentRole::Values,
                    false,
                )?;
                check_original(
                    p.alpha.source_global_scale(),
                    &format!("{prefix}.weight_global_sf"),
                    &[],
                    ElementType::F32,
                    WeightComponentRole::Metadata,
                    false,
                )?;
                p.alpha.output_component()?;
            }
        }
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("decoder block metadata exceeds capacity"));
        }
        let schema = self.schema()?;
        let bytes = schema.components.iter().try_fold(0u64, |n, c| {
            n.checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("decoder stored bytes overflow"))
        })?;
        if bytes > limits.maximum_stored_bytes {
            return Err(invalid("decoder block stored bytes exceed capacity"));
        }
        for t in program::build(self)?.1.values() {
            if t.byte_len()? > isize::MAX as u64 {
                return Err(invalid("decoder activation exceeds address space"));
            }
        }
        Ok(())
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        let mut tensors = Vec::new();
        let mut components = Vec::new();
        let mut push =
            |name: &str, c: WeightComponentSpec, physical: bool| -> Result<(), VNextError> {
                tensors.push(WeightTensorSpec {
                    id: weight(name)?,
                    dimensions: c.dimensions.clone(),
                    logical_element_type: c.physical_element_type(),
                    required: true,
                    physical_layout: if physical {
                        PhysicalWeightLayout::PhysicalComponent(PhysicalWeightComponentView::new(
                            c.id.clone(),
                        ))
                    } else {
                        PhysicalWeightLayout::Dense {
                            component_id: c.id.clone(),
                        }
                    },
                });
                components.push(c);
                Ok(())
            };
        push("unit_sf", source::unit_component()?, false)?;
        push("unit_gamma", source::gamma_component(self.head_dim)?, false)?;
        for b in &self.blocks {
            for (suffix, c) in NORMS.iter().zip(&b.norms_and_scales) {
                push(&format!("block{}.{}", b.index, suffix), c.clone(), false)?;
            }
            for (i, p) in b.projections.iter().enumerate() {
                let prefix = format!("block{}.{}", b.index, PROJECTIONS[i]);
                for (suffix, c, physical) in [
                    ("packed", p.packed.clone(), true),
                    ("scales", p.scales.clone(), true),
                    ("bias", p.bias.clone(), false),
                    ("alpha", p.alpha.output_component()?, false),
                ] {
                    push(&format!("{prefix}.{suffix}"), c, physical)?;
                }
            }
        }
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout_id: WeightLayoutId::new("weight-layout.fast-h3-video-decoder-original")?,
            version: ContractVersion::new(1, 0),
            tensors,
            components,
        };
        schema.validate(&ModelFamilyId::new(
            FastH3VideoDecoderBlocksFamily::FAMILY_ID,
        )?)?;
        Ok(schema)
    }
}
fn check_original(
    c: &WeightComponentSpec,
    name: &str,
    shape: &[u64],
    dtype: ElementType,
    role: WeightComponentRole,
    packed: bool,
) -> Result<(), VNextError> {
    let encoding = if packed {
        WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4())
    } else {
        WeightEncoding::Dense {
            element_type: dtype,
        }
    };
    if !c.required
        || c.id.as_str() != format!("fast_h3.video_vae.{name}")
        || c.external_names != [name]
        || c.dimensions != shape
        || c.role != role
        || c.encoding != encoding
    {
        return Err(invalid(format!(
            "original decoder parameter differs: {name}"
        )));
    }
    Ok(())
}
fn tensor(shape: Vec<u64>, dtype: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: shape,
        element_type: dtype,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn weight(name: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.fast_h3.video_decoder.{name}"))
}
fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.fast_h3.video_decoder.{name}"))
}
fn wvalue(name: &str) -> Result<ProgramValueId, VNextError> {
    value(&format!("weights.{name}"))
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 video decoder blocks: {e}"),
    }
}
