//! Dense, timestep-free FastH3 text refinement after context projection.
//!
//! This profile preserves eager BF16 normalization, SiLU/product and residual
//! boundaries. Attention is noncausal and isolated per declared request. The
//! selected provider owns reduction/transcendental arithmetic; this is not a
//! claim of bitwise equivalence to a particular Torch/FlashAttention binary.
use super::{FastH3TransformerPackage, FastH3WeightStorage};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod family;
mod program;
pub use family::FastH3TextRefinerFamily;

#[derive(Clone, Copy, Debug)]
pub struct FastH3TextRefinerLimits {
    pub maximum_rows: u64,
    pub maximum_requests: usize,
    pub maximum_layers: usize,
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
}
impl FastH3TextRefinerLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.maximum_rows == 0
            || self.maximum_requests == 0
            || self.maximum_layers == 0
            || self.maximum_metadata_bytes == 0
            || self.maximum_stored_bytes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3TextRefinerConfig {
    requests: Vec<u64>,
    hidden: u64,
    heads: u64,
    head_dim: u64,
    ffn: u64,
    layers: usize,
    norm_epsilon: CanonicalRational,
    qk_epsilon: CanonicalRational,
    final_epsilon: CanonicalRational,
    components: Vec<WeightComponentSpec>,
}
impl FastH3TextRefinerConfig {
    pub fn from_package(
        p: &FastH3TransformerPackage,
        request_lengths: Vec<u64>,
        limits: FastH3TextRefinerLimits,
    ) -> Result<Self, VNextError> {
        let t = p.config();
        if t.refiner_layers() > limits.maximum_layers {
            return Err(invalid("refiner layers exceed capacity"));
        }
        let mut c = Self {
            requests: request_lengths,
            hidden: t.hidden_size() as u64,
            heads: t.attention_heads() as u64,
            head_dim: t.attention_head_dim() as u64,
            ffn: t.ffn_dim() as u64,
            layers: t.refiner_layers(),
            norm_epsilon: rational(t.norm_eps())?,
            qk_epsilon: rational(t.qk_norm_eps())?,
            final_epsilon: rational(t.final_norm_eps())?,
            components: Vec::new(),
        };
        c.check_geometry()?;
        for (name, _) in c.entries()? {
            let w = p
                .weight(&name)
                .ok_or_else(|| invalid(format!("missing {name}")))?;
            let FastH3WeightStorage::Dense(component) = &w.storage else {
                return Err(invalid("refiner must retain dense parameters"));
            };
            if w.compute_element_type != ElementType::Bf16 {
                return Err(invalid("refiner compute dtype must be BF16"));
            }
            c.components.push(component.clone());
        }
        c.validate(limits)?;
        Ok(c)
    }
    pub fn request_lengths(&self) -> &[u64] {
        &self.requests
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.requests.iter().try_fold(0u64, |n, &r| {
            n.checked_add(r)
                .ok_or_else(|| invalid("row extent overflows"))
        })
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn layers(&self) -> usize {
        self.layers
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
    fn inner(&self) -> Result<u64, VNextError> {
        self.heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("attention width overflows"))
    }
    fn check_geometry(&self) -> Result<(), VNextError> {
        if self.requests.is_empty()
            || self.requests.len() > 32
            || self.requests.iter().any(|&r| r == 0 || r > i32::MAX as u64)
            || self.rows()? > i32::MAX as u64
            || self.layers == 0
            || self.hidden == 0
            || self.hidden > HEAD_RMS_NORM_BF16_MAX_HEAD_DIM
            || self.heads == 0
            || self.head_dim < 2
            || self.head_dim > 256
            || self.head_dim % 2 != 0
            || self.inner()? > i32::MAX as u64
            || self.ffn == 0
            || self.ffn > i32::MAX as u64 / 2
        {
            return Err(invalid("invalid refiner geometry or request partition"));
        }
        for (width, eps) in [
            (self.hidden, self.norm_epsilon),
            (self.head_dim, self.qk_epsilon),
            (self.hidden, self.final_epsilon),
        ] {
            let x = ResolvedTensorSpec::new(
                vec![1, width],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            let g = ResolvedTensorSpec::new(
                vec![width],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            HeadRmsNormBf16Geometry::new(&x, &g, &x, width, eps)?;
        }
        Ok(())
    }
    fn entries(&self) -> Result<Vec<(String, Vec<u64>)>, VNextError> {
        let count = self
            .layers
            .checked_mul(10)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("parameter inventory overflows"))?;
        // A malformed wire config must not allocate an inventory larger than
        // the already supplied component metadata. Construction starts empty.
        if !self.components.is_empty() && count != self.components.len() {
            return Err(invalid("incomplete refiner parameter inventory"));
        }
        let mut entries = Vec::new();
        let inner = self.inner()?;
        for layer in 0..self.layers {
            let prefix = format!("token_refiner.refiner_blocks.{layer}");
            for (suffix, shape) in [
                ("norm1.weight", vec![self.hidden]),
                ("norm2.weight", vec![self.hidden]),
                ("attn.norm_q.weight", vec![self.head_dim]),
                ("attn.norm_k.weight", vec![self.head_dim]),
                ("attn.to_q.weight", vec![inner, self.hidden]),
                ("attn.to_k.weight", vec![inner, self.hidden]),
                ("attn.to_v.weight", vec![inner, self.hidden]),
                ("attn.to_out.weight", vec![self.hidden, inner]),
                ("ff.fc_in.weight", vec![2 * self.ffn, self.hidden]),
                ("ff.fc_out.weight", vec![self.hidden, self.ffn]),
            ] {
                entries.push((format!("{prefix}.{suffix}"), shape));
            }
        }
        entries.push(("token_refiner.final_norm.weight".into(), vec![self.hidden]));
        Ok(entries)
    }
    fn check_inventory(&self) -> Result<(), VNextError> {
        self.check_geometry()?;
        if self.components.is_empty() {
            return Err(invalid("missing refiner sources"));
        }
        for (c, (_, shape)) in self.components.iter().zip(self.entries()?) {
            if c.dimensions != shape
                || c.role != WeightComponentRole::Values
                || !c.required
                || c.external_names.len() != 1
                || c.encoding
                    != (WeightEncoding::Dense {
                        element_type: ElementType::Bf16,
                    })
            {
                return Err(invalid(
                    "refiner source shape, role or BF16 encoding differs",
                ));
            }
        }
        Ok(())
    }
    fn validate(&self, limits: FastH3TextRefinerLimits) -> Result<(), VNextError> {
        limits.validate()?;
        if self.layers > limits.maximum_layers
            || self.requests.len() > limits.maximum_requests
            || self.rows()? > limits.maximum_rows
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("refiner geometry/metadata capacity exceeded"));
        }
        self.check_inventory()?;
        let bytes = self.components.iter().try_fold(0u64, |n, c| {
            n.checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("source bytes overflow"))
        })?;
        if bytes > limits.maximum_stored_bytes {
            return Err(invalid("refiner source capacity exceeded"));
        }
        self.schema()?
            .validate(&id(FastH3TextRefinerFamily::FAMILY_ID)?)?;
        for s in program::build(self)?.1.values() {
            if s.byte_len()? > isize::MAX as u64 {
                return Err(invalid("refiner tensor exceeds address space"));
            }
        }
        Ok(())
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        self.check_inventory()?;
        Ok(WeightSchema {
            format_id: id("weight-format.safetensors.dense")?,
            layout_id: id("weight-layout.fast_h3.text_refiner.dense_bf16.v1")?,
            version: ContractVersion::new(1, 0),
            components: self.components.clone(),
            tensors: self
                .components
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    Ok(WeightTensorSpec {
                        id: weight_id(i)?,
                        dimensions: c.dimensions.clone(),
                        logical_element_type: ElementType::Bf16,
                        physical_layout: PhysicalWeightLayout::Dense {
                            component_id: c.id.clone(),
                        },
                        required: true,
                    })
                })
                .collect::<Result<_, VNextError>>()?,
        })
    }
}

/// Original archive components and their retained Host owners, without a
/// precision conversion, alternate quantization mapping or private allocator.
pub struct FastH3TextRefinerSource<'a> {
    package: &'a FastH3TransformerPackage,
    components: Vec<WeightComponentSpec>,
}
impl<'a> FastH3TextRefinerSource<'a> {
    pub fn new(
        package: &'a FastH3TransformerPackage,
        c: &FastH3TextRefinerConfig,
        limits: FastH3TextRefinerLimits,
    ) -> Result<Self, VNextError> {
        c.validate(limits)?;
        if FastH3TextRefinerConfig::from_package(package, c.requests.clone(), limits)? != *c {
            return Err(invalid("refiner source differs from checkpoint"));
        }
        Ok(Self {
            package,
            components: c.components.clone(),
        })
    }
}
impl WeightComponentSource for FastH3TextRefinerSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if !self.components.contains(c) {
            return Err(invalid("undeclared refiner source component"));
        }
        self.package.dense_archive().component(c)
    }
}
fn rational(x: f64) -> Result<CanonicalRational, VNextError> {
    CanonicalRational::from_decimal_str(&x.to_string())
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn weight_id(i: usize) -> Result<WeightId, VNextError> {
    id(&format!("weight.fast_h3.text_refiner.{i}"))
}
fn weight_value(i: usize) -> Result<ProgramValueId, VNextError> {
    id(&format!("value.fast_h3.text_refiner.weight{i}"))
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 text refiner: {e}"),
    }
}
