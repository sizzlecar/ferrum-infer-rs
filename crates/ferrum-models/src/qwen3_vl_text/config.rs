use super::{invalid, Qwen3VLTextEncoderFamily as Family};
use ferrum_runtime::*;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// A run is explicitly valid or padding; the adapter chooses its position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3VLTextKeyRun {
    length: u32,
    valid: bool,
}
impl Qwen3VLTextKeyRun {
    pub const fn new(length: u32, valid: bool) -> Self {
        Self { length, valid }
    }
    pub const fn length(&self) -> u32 {
        self.length
    }
    pub const fn is_valid(&self) -> bool {
        self.valid
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen3VLTextLayout {
    sequence_length: u32,
    key_runs: Vec<Vec<Qwen3VLTextKeyRun>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayoutWire {
    sequence_length: u32,
    key_runs: Vec<Vec<Qwen3VLTextKeyRun>>,
}
impl<'de> Deserialize<'de> for Qwen3VLTextLayout {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = LayoutWire::deserialize(d)?;
        Self::new(w.sequence_length, w.key_runs).map_err(serde::de::Error::custom)
    }
}
impl Qwen3VLTextLayout {
    pub fn new(
        sequence_length: u32,
        key_runs: Vec<Vec<Qwen3VLTextKeyRun>>,
    ) -> Result<Self, VNextError> {
        if sequence_length == 0
            || key_runs.is_empty()
            || key_runs.len() > STRUCTURED_ATTENTION_BF16_MAX_BATCH
        {
            return Err(invalid(
                "layout",
                "sequence must be positive and batch within the structured-attention bound",
            ));
        }
        for runs in &key_runs {
            if runs.is_empty()
                || runs.len() > STRUCTURED_ATTENTION_BF16_MAX_KEY_RUNS
                || runs.iter().any(|r| r.length == 0)
                || runs.windows(2).any(|w| w[0].valid == w[1].valid)
                || runs.iter().try_fold(0_u32, |n, r| n.checked_add(r.length))
                    != Some(sequence_length)
            {
                return Err(invalid("layout","every sample needs bounded positive alternating key runs covering the full sequence"));
            }
        }
        Ok(Self {
            sequence_length,
            key_runs,
        })
    }
    pub const fn sequence_length(&self) -> u32 {
        self.sequence_length
    }
    pub fn batch(&self) -> usize {
        self.key_runs.len()
    }
    pub fn rows(&self) -> u64 {
        u64::from(self.sequence_length) * self.key_runs.len() as u64
    }
    pub fn key_runs(&self) -> &[Vec<Qwen3VLTextKeyRun>] {
        &self.key_runs
    }
    pub(super) fn attributes(
        &self,
        d: u64,
        scale: CanonicalRational,
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let mut a = BTreeMap::from([
            (
                AttributeId::new("sequence_length")?,
                SemanticValue::Unsigned(u64::from(self.sequence_length)),
            ),
            (AttributeId::new("head_dim")?, SemanticValue::Unsigned(d)),
            (AttributeId::new("scale")?, SemanticValue::Rational(scale)),
        ]);
        for (name, value) in [
            ("segment_lengths", vec![i64::from(self.sequence_length)]),
            ("segment_bidirectional", vec![0]),
            (
                "key_run_counts",
                self.key_runs.iter().map(|r| r.len() as i64).collect(),
            ),
            (
                "key_run_lengths",
                self.key_runs
                    .iter()
                    .flatten()
                    .map(|r| i64::from(r.length))
                    .collect(),
            ),
            (
                "key_run_valid",
                self.key_runs
                    .iter()
                    .flatten()
                    .map(|r| i64::from(r.valid))
                    .collect(),
            ),
        ] {
            a.insert(AttributeId::new(name)?, SemanticValue::Integers(value));
        }
        Ok(a)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchitectureData {
    layers: u64,
    hidden_size: u64,
    intermediate_size: u64,
    query_heads: u64,
    key_value_heads: u64,
    head_dim: u64,
    vocabulary_size: u64,
    maximum_positions: u64,
    epsilon: CanonicalRational,
    rope_theta: u64,
    mrope_sections: [u64; 3],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Qwen3VLTextArchitecture(ArchitectureData);
impl<'de> Deserialize<'de> for Qwen3VLTextArchitecture {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Self::checked(ArchitectureData::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}
impl Qwen3VLTextArchitecture {
    fn checked(d: ArchitectureData) -> Result<Self, VNextError> {
        if [
            d.layers,
            d.hidden_size,
            d.intermediate_size,
            d.query_heads,
            d.key_value_heads,
            d.head_dim,
            d.vocabulary_size,
            d.maximum_positions,
            d.rope_theta,
        ]
        .contains(&0)
            || d.query_heads % d.key_value_heads != 0
            || !(2..=256).contains(&d.head_dim)
            || !d.head_dim.is_power_of_two()
            || d.mrope_sections
                .iter()
                .try_fold(0_u64, |n, x| n.checked_add(*x))
                != Some(d.head_dim / 2)
        {
            return Err(invalid("architecture","requires positive dimensions, divisible GQA heads, supported even power-of-two head_dim and mRoPE sections summing to head_dim/2"));
        }
        // Metadata inventory bound: not a device-memory or execution quota.
        if d.layers.checked_mul(11).is_none_or(|n| n > 65536) {
            return Err(invalid(
                "layers",
                "required weight inventory exceeds 65536 tensors",
            ));
        }
        let c = Self(d);
        let q = c.query_columns()?;
        let kv = c.key_value_columns()?;
        for shape in [
            vec![c.hidden_size(), c.intermediate_size()],
            vec![q, c.hidden_size()],
            vec![kv, c.hidden_size()],
            vec![c.vocabulary_size(), c.hidden_size()],
        ] {
            ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous)?;
        }
        let norm = ResolvedTensorSpec::new(
            vec![1, c.hidden_size()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let gamma = ResolvedTensorSpec::new(
            vec![c.hidden_size()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        HeadRmsNormBf16Geometry::new(&norm, &gamma, &norm, c.hidden_size(), c.epsilon())?;
        Ok(c)
    }
    pub fn from_hf_value(raw: &Value) -> Result<Self, VNextError> {
        if raw.get("model_type").and_then(Value::as_str) != Some("qwen3_vl")
            || raw.get("architectures")
                != Some(&serde_json::json!(["Qwen3VLForConditionalGeneration"]))
        {
            return Err(invalid(
                "architecture",
                "requires the Qwen3VL conditional-generation container",
            ));
        }
        let t = raw
            .get("text_config")
            .ok_or_else(|| invalid("text_config", "missing text subconfiguration"))?;
        if t.get("model_type").and_then(Value::as_str) != Some("qwen3_vl_text")
            || t.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || t.get("hidden_act").and_then(Value::as_str) != Some("silu")
            || t.get("attention_bias").and_then(Value::as_bool) != Some(false)
            || t.get("attention_dropout").and_then(Value::as_f64) != Some(0.0)
            || t.get("use_sliding_window").and_then(Value::as_bool) == Some(true)
            || t.get("sliding_window").is_some_and(|v| !v.is_null())
            || t.get("layer_types").is_some_and(|v| {
                v.as_array().is_none_or(|a| {
                    Some(a.len() as u64) != t.get("num_hidden_layers").and_then(Value::as_u64)
                        || a.iter().any(|s| s.as_str() != Some("full_attention"))
                })
            })
        {
            return Err(invalid("text_config","supports BF16 full causal attention, no projection bias/dropout, and SiLU MLP only"));
        }
        let u = |name: &str| {
            t.get(name)
                .and_then(Value::as_u64)
                .ok_or_else(|| invalid(name, "missing unsigned dimension"))
        };
        let rope = t
            .get("rope_scaling")
            .ok_or_else(|| invalid("rope_scaling", "missing explicit mRoPE metadata"))?;
        if rope.get("rope_type").and_then(Value::as_str) != Some("default")
            || rope.get("mrope_interleaved").and_then(Value::as_bool) != Some(true)
        {
            return Err(invalid(
                "rope_scaling",
                "supports explicit default interleaved mRoPE metadata only",
            ));
        }
        let sections = serde_json::from_value(
            rope.get("mrope_section")
                .cloned()
                .ok_or_else(|| invalid("mrope_section", "missing sections"))?,
        )
        .map_err(|e| invalid("mrope_section", e.to_string()))?;
        let Some(Value::Number(epsilon)) = t.get("rms_norm_eps") else {
            return Err(invalid("rms_norm_eps", "missing numeric epsilon"));
        };
        Self::checked(ArchitectureData {
            layers: u("num_hidden_layers")?,
            hidden_size: u("hidden_size")?,
            intermediate_size: u("intermediate_size")?,
            query_heads: u("num_attention_heads")?,
            key_value_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            vocabulary_size: u("vocab_size")?,
            maximum_positions: u("max_position_embeddings")?,
            epsilon: CanonicalRational::from_decimal_str(&epsilon.to_string())?,
            rope_theta: u("rope_theta")?,
            mrope_sections: sections,
        })
    }
    pub const fn layers(&self) -> u64 {
        self.0.layers
    }
    pub const fn hidden_size(&self) -> u64 {
        self.0.hidden_size
    }
    pub const fn intermediate_size(&self) -> u64 {
        self.0.intermediate_size
    }
    pub const fn query_heads(&self) -> u64 {
        self.0.query_heads
    }
    pub const fn key_value_heads(&self) -> u64 {
        self.0.key_value_heads
    }
    pub const fn head_dim(&self) -> u64 {
        self.0.head_dim
    }
    pub const fn vocabulary_size(&self) -> u64 {
        self.0.vocabulary_size
    }
    pub const fn maximum_positions(&self) -> u64 {
        self.0.maximum_positions
    }
    pub const fn epsilon(&self) -> CanonicalRational {
        self.0.epsilon
    }
    pub const fn rope_theta(&self) -> u64 {
        self.0.rope_theta
    }
    pub const fn mrope_sections(&self) -> [u64; 3] {
        self.0.mrope_sections
    }
    pub fn query_columns(&self) -> Result<u64, VNextError> {
        self.0
            .query_heads
            .checked_mul(self.0.head_dim)
            .ok_or_else(|| invalid("query_heads", "Q width overflows"))
    }
    pub fn key_value_columns(&self) -> Result<u64, VNextError> {
        self.0
            .key_value_heads
            .checked_mul(self.0.head_dim)
            .ok_or_else(|| invalid("key_value_heads", "KV width overflows"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen3VLTextConfig {
    architecture: Qwen3VLTextArchitecture,
    layout: Qwen3VLTextLayout,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigWire {
    architecture: Qwen3VLTextArchitecture,
    layout: Qwen3VLTextLayout,
}
impl<'de> Deserialize<'de> for Qwen3VLTextConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = ConfigWire::deserialize(d)?;
        Self::new(w.architecture, w.layout).map_err(serde::de::Error::custom)
    }
}
impl Qwen3VLTextConfig {
    pub fn new(
        architecture: Qwen3VLTextArchitecture,
        layout: Qwen3VLTextLayout,
    ) -> Result<Self, VNextError> {
        let c = Self {
            architecture,
            layout,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn from_hf_json(bytes: &[u8], layout: Qwen3VLTextLayout) -> Result<Self, VNextError> {
        if bytes.is_empty() || bytes.len() > 4 * 1024 * 1024 {
            return Err(invalid(
                "config",
                "HF metadata must be nonempty and at most 4 MiB",
            ));
        }
        let raw = serde_json::from_slice(bytes).map_err(|e| invalid("config", e.to_string()))?;
        Self::new(Qwen3VLTextArchitecture::from_hf_value(&raw)?, layout)
    }
    pub fn architecture(&self) -> &Qwen3VLTextArchitecture {
        &self.architecture
    }
    pub fn layout(&self) -> &Qwen3VLTextLayout {
        &self.layout
    }
    pub fn scale(&self) -> Result<CanonicalRational, VNextError> {
        let effective = (1.0 / (self.architecture.head_dim() as f64).sqrt()) as f32;
        let bits = effective.to_bits();
        let exponent = ((bits >> 23) & 255) as i32 - 150;
        CanonicalRational::new(
            i64::from((bits & 0x7fffff) | (1 << 23)),
            1_u64
                .checked_shl(
                    u32::try_from(-exponent).map_err(|_| invalid("scale", "invalid exponent"))?,
                )
                .ok_or_else(|| invalid("scale", "denominator overflows"))?,
        )
    }
    pub fn attention_attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        self.layout
            .attributes(self.architecture.head_dim(), self.scale()?)
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        [
            (Family::EMBEDDINGS_ID, self.architecture.hidden_size()),
            (Family::COSINE_ID, self.architecture.head_dim()),
            (Family::SINE_ID, self.architecture.head_dim()),
        ]
        .into_iter()
        .map(|(id, c)| {
            Ok((
                ProgramValueId::new(id)?,
                ProgramTensorSpec {
                    dimensions: vec![self.layout.rows(), c],
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            ))
        })
        .collect()
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(self.input_tensor_specs()?)
    }
    pub fn expected_weights(&self) -> Result<Vec<super::Qwen3VLTextExpectedWeight>, VNextError> {
        super::weights::expected(self)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if u64::from(self.layout.sequence_length()) > self.architecture.maximum_positions() {
            return Err(invalid(
                "layout",
                "sequence exceeds declared position capacity",
            ));
        }
        let t = |c| {
            ResolvedTensorSpec::new(
                vec![self.layout.rows(), c],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
        };
        let q = t(self.architecture.query_columns()?)?;
        let kv = t(self.architecture.key_value_columns()?)?;
        StructuredGqaAttentionBf16Geometry::new(&q, &kv, &kv, &q, &self.attention_attributes()?)?;
        t(self.architecture.hidden_size())?;
        t(self.architecture.intermediate_size())?;
        Ok(())
    }
}
