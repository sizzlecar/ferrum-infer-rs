use super::{invalid, Qwen3VLVisionFamily as Family};
use ferrum_runtime::*;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchitectureData {
    depth: u64,
    hidden_size: u64,
    intermediate_size: u64,
    heads: u64,
    channels: u64,
    patch_size: u64,
    temporal_patch_size: u64,
    merge_size: u64,
    position_embeddings: u64,
    output_size: u64,
    deepstack_layers: [u64; 3],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Qwen3VLVisionArchitecture(ArchitectureData);
impl<'de> Deserialize<'de> for Qwen3VLVisionArchitecture {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Self::checked(ArchitectureData::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}
impl Qwen3VLVisionArchitecture {
    fn checked(d: ArchitectureData) -> Result<Self, VNextError> {
        if [
            d.depth,
            d.hidden_size,
            d.intermediate_size,
            d.heads,
            d.channels,
            d.patch_size,
            d.temporal_patch_size,
            d.merge_size,
            d.position_embeddings,
            d.output_size,
        ]
        .contains(&0)
            || d.hidden_size % d.heads != 0
            || (d.hidden_size / d.heads) % 4 != 0
            || d.hidden_size / d.heads > 256
            || d.deepstack_layers.windows(2).any(|v| v[0] >= v[1])
            || d.deepstack_layers[2] >= d.depth
            || d.depth.checked_mul(12).is_none_or(|n| n > 65536)
        {
            return Err(invalid("architecture", "requires positive checked dimensions, axial head_dim divisible by four and three increasing DeepStack layers"));
        }
        let side = d.position_embeddings.isqrt();
        if side.checked_mul(side) != Some(d.position_embeddings) {
            return Err(invalid(
                "position_embeddings",
                "learned position grid must be square",
            ));
        }
        let a = Self(d);
        let qkv = a
            .hidden_size()
            .checked_mul(3)
            .ok_or_else(|| invalid("qkv", "projection width overflows"))?;
        for dims in [
            vec![qkv, a.hidden_size()],
            vec![a.hidden_size(), a.patch_columns()?],
            vec![a.merger_width()?, a.merger_width()?],
            vec![a.output_size(), a.merger_width()?],
            vec![a.intermediate_size(), a.hidden_size()],
            vec![a.position_embeddings(), a.hidden_size()],
        ] {
            ResolvedTensorSpec::new(dims, ElementType::Bf16, ResolvedTensorLayout::Contiguous)?;
        }
        Ok(a)
    }
    pub fn from_hf_value(raw: &Value) -> Result<Self, VNextError> {
        if raw.get("model_type").and_then(Value::as_str) != Some("qwen3_vl")
            || raw.get("architectures")
                != Some(&serde_json::json!(["Qwen3VLForConditionalGeneration"]))
        {
            return Err(invalid(
                "architecture",
                "requires Qwen3VL conditional-generation metadata",
            ));
        }
        let v = raw
            .get("vision_config")
            .ok_or_else(|| invalid("vision_config", "missing vision configuration"))?;
        if v.get("model_type").and_then(Value::as_str) != Some("qwen3_vl")
            || v.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || v.get("hidden_act").and_then(Value::as_str) != Some("gelu_pytorch_tanh")
        {
            return Err(invalid(
                "vision_config",
                "requires BF16 Qwen3VL vision with tanh-GELU blocks",
            ));
        }
        // These are the defaults of the pinned Qwen3VL architecture, not backend knobs.
        if v.get("rope_parameters").is_some()
            || v.get("rope_theta").is_some()
            || v.get("layer_norm_eps").is_some()
        {
            return Err(invalid(
                "vision_config",
                "custom rotary or normalization parameters require a separate declared profile",
            ));
        }
        let u = |k: &str| {
            v.get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| invalid(k, "missing unsigned architecture value"))
        };
        Self::checked(ArchitectureData {
            depth: u("depth")?,
            hidden_size: u("hidden_size")?,
            intermediate_size: u("intermediate_size")?,
            heads: u("num_heads")?,
            channels: u("in_channels")?,
            patch_size: u("patch_size")?,
            temporal_patch_size: u("temporal_patch_size")?,
            merge_size: u("spatial_merge_size")?,
            position_embeddings: u("num_position_embeddings")?,
            output_size: u("out_hidden_size")?,
            deepstack_layers: serde_json::from_value(
                v.get("deepstack_visual_indexes")
                    .cloned()
                    .ok_or_else(|| invalid("deepstack_visual_indexes", "missing three stages"))?,
            )
            .map_err(|e| invalid("deepstack_visual_indexes", e.to_string()))?,
        })
    }
    pub const fn depth(&self) -> u64 {
        self.0.depth
    }
    pub const fn hidden_size(&self) -> u64 {
        self.0.hidden_size
    }
    pub const fn intermediate_size(&self) -> u64 {
        self.0.intermediate_size
    }
    pub const fn heads(&self) -> u64 {
        self.0.heads
    }
    pub const fn head_dim(&self) -> u64 {
        self.0.hidden_size / self.0.heads
    }
    pub const fn channels(&self) -> u64 {
        self.0.channels
    }
    pub const fn patch_size(&self) -> u64 {
        self.0.patch_size
    }
    pub const fn temporal_patch_size(&self) -> u64 {
        self.0.temporal_patch_size
    }
    pub const fn merge_size(&self) -> u64 {
        self.0.merge_size
    }
    pub const fn position_embeddings(&self) -> u64 {
        self.0.position_embeddings
    }
    pub const fn output_size(&self) -> u64 {
        self.0.output_size
    }
    pub const fn deepstack_layers(&self) -> [u64; 3] {
        self.0.deepstack_layers
    }
    pub fn patch_columns(&self) -> Result<u64, VNextError> {
        [
            self.channels(),
            self.temporal_patch_size(),
            self.patch_size(),
            self.patch_size(),
        ]
        .into_iter()
        .try_fold(1_u64, |n, x| {
            n.checked_mul(x)
                .ok_or_else(|| invalid("patch", "patch width overflows"))
        })
    }
    pub fn merger_width(&self) -> Result<u64, VNextError> {
        self.hidden_size()
            .checked_mul(self.merge_size())
            .and_then(|x| x.checked_mul(self.merge_size()))
            .ok_or_else(|| invalid("merge", "merged width overflows"))
    }
    pub fn epsilon(&self) -> CanonicalRational {
        CanonicalRational::new(1, 1_000_000).expect("fixed vision epsilon")
    }
}

/// One static image grid. Video/frame semantics are deliberately not inferred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3VLVisionGrid {
    temporal: u64,
    height: u64,
    width: u64,
}
impl Qwen3VLVisionGrid {
    pub fn new(temporal: u64, height: u64, width: u64) -> Result<Self, VNextError> {
        let g = Self {
            temporal,
            height,
            width,
        };
        if temporal != 1 || height == 0 || width == 0 {
            return Err(invalid(
                "grid",
                "first image profile requires temporal=1 and positive spatial dimensions",
            ));
        }
        g.rows()?;
        Ok(g)
    }
    pub const fn temporal(&self) -> u64 {
        self.temporal
    }
    pub const fn height(&self) -> u64 {
        self.height
    }
    pub const fn width(&self) -> u64 {
        self.width
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.temporal
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(self.width))
            .ok_or_else(|| invalid("grid", "patch count overflows"))
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen3VLVisionConfig {
    architecture: Qwen3VLVisionArchitecture,
    grids: Vec<Qwen3VLVisionGrid>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    architecture: Qwen3VLVisionArchitecture,
    grids: Vec<Qwen3VLVisionGrid>,
}
impl<'de> Deserialize<'de> for Qwen3VLVisionConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(w.architecture, w.grids).map_err(serde::de::Error::custom)
    }
}
impl Qwen3VLVisionConfig {
    pub fn new(
        architecture: Qwen3VLVisionArchitecture,
        grids: Vec<Qwen3VLVisionGrid>,
    ) -> Result<Self, VNextError> {
        let c = Self {
            architecture,
            grids,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn from_hf_json(bytes: &[u8], grids: Vec<Qwen3VLVisionGrid>) -> Result<Self, VNextError> {
        if bytes.is_empty() || bytes.len() > 4 * 1024 * 1024 {
            return Err(invalid(
                "config",
                "HF metadata must be nonempty and at most 4 MiB",
            ));
        }
        let v = serde_json::from_slice(bytes).map_err(|e| invalid("config", e.to_string()))?;
        Self::new(Qwen3VLVisionArchitecture::from_hf_value(&v)?, grids)
    }
    pub fn architecture(&self) -> &Qwen3VLVisionArchitecture {
        &self.architecture
    }
    pub fn grids(&self) -> &[Qwen3VLVisionGrid] {
        &self.grids
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.grids.iter().try_fold(0_u64, |n, g| {
            n.checked_add(g.rows()?)
                .ok_or_else(|| invalid("grids", "total patches overflow"))
        })
    }
    pub fn merged_rows(&self) -> Result<u64, VNextError> {
        Ok(self.rows()? / self.architecture.merge_size() / self.architecture.merge_size())
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        let a = &self.architecture;
        [
            (Family::PATCHES_ID, a.patch_columns()?, ElementType::Bf16),
            (Family::POSITIONS_ID, a.hidden_size(), ElementType::Bf16),
            (Family::COSINE_ID, a.head_dim(), ElementType::F32),
            (Family::SINE_ID, a.head_dim(), ElementType::F32),
        ]
        .into_iter()
        .map(|(id, width, element_type)| {
            Ok((
                ProgramValueId::new(id)?,
                ProgramTensorSpec {
                    dimensions: vec![self.rows()?, width],
                    element_type,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            ))
        })
        .collect()
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        let (_, specs) = super::program::build(self, &ModelFamilyId::new(Family::FAMILY_ID)?)?;
        ProgramPlanCompileOptions::new(specs)
    }
    pub fn expected_weights(&self) -> Result<Vec<super::Qwen3VLVisionExpectedWeight>, VNextError> {
        super::weights::expected(self)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if self.grids.is_empty() || self.grids.len() > 32 {
            return Err(invalid("grids", "requires 1..=32 image grids"));
        }
        for g in &self.grids {
            Qwen3VLVisionGrid::new(g.temporal, g.height, g.width)?;
            if g.height % self.architecture.merge_size() != 0
                || g.width % self.architecture.merge_size() != 0
                || g.rows()? > u64::from(u32::MAX)
            {
                return Err(invalid(
                    "grids",
                    "grid must be merge-aligned with a bounded attention segment",
                ));
            }
        }
        for spec in self.input_tensor_specs()?.values() {
            ResolvedTensorSpec::new(
                spec.dimensions.clone(),
                spec.element_type,
                spec.layout.clone(),
            )?;
        }
        Ok(())
    }
}
