use super::{invalid, QwenImage21Block0AttentionFamily, QwenImage21Config};
use ferrum_runtime::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenImage21AttentionSegment {
    length: u32,
    bidirectional: bool,
}
impl QwenImage21AttentionSegment {
    pub const fn new(length: u32, bidirectional: bool) -> Self {
        Self {
            length,
            bidirectional,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenImage21AttentionKeyRun {
    length: u32,
    valid: bool,
}
impl QwenImage21AttentionKeyRun {
    pub const fn new(length: u32, valid: bool) -> Self {
        Self { length, valid }
    }
}

/// Immutable block-causal layout. Separate adjacent image segments are retained
/// even when both are bidirectional; their boundaries affect visibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21AttentionLayout {
    sequence_length: u32,
    segments: Vec<QwenImage21AttentionSegment>,
    key_runs: Vec<Vec<QwenImage21AttentionKeyRun>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayoutWire {
    sequence_length: u32,
    segments: Vec<QwenImage21AttentionSegment>,
    key_runs: Vec<Vec<QwenImage21AttentionKeyRun>>,
}
impl<'de> Deserialize<'de> for QwenImage21AttentionLayout {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = LayoutWire::deserialize(d)?;
        Self::new(wire.sequence_length, wire.segments, wire.key_runs)
            .map_err(serde::de::Error::custom)
    }
}
impl QwenImage21AttentionLayout {
    pub fn new(
        sequence_length: u32,
        segments: Vec<QwenImage21AttentionSegment>,
        key_runs: Vec<Vec<QwenImage21AttentionKeyRun>>,
    ) -> Result<Self, VNextError> {
        if sequence_length == 0
            || segments.is_empty()
            || segments.len() > STRUCTURED_ATTENTION_BF16_MAX_SEGMENTS
            || key_runs.is_empty()
            || key_runs.len() > STRUCTURED_ATTENTION_BF16_MAX_BATCH
            || key_runs
                .iter()
                .any(|runs| runs.is_empty() || runs.len() > STRUCTURED_ATTENTION_BF16_MAX_KEY_RUNS)
        {
            return Err(invalid("layout","attention sequence/segments/key runs must be nonempty and within the declared operation bounds"));
        }
        let result = Self {
            sequence_length,
            segments,
            key_runs,
        };
        // Delegate the visibility geometry and canonical key-run validation to
        // the real contract, using a small descriptive width (no allocation).
        let spec = ResolvedTensorSpec::new(
            vec![result.rows(), 2],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        StructuredAttentionBf16Geometry::new(
            &spec,
            &spec,
            &spec,
            &spec,
            &result.attributes(2, CanonicalRational::new(1, 1)?)?,
        )?;
        Ok(result)
    }
    pub fn uniform(sequence_length: u32, bidirectional: bool) -> Result<Self, VNextError> {
        Self::new(
            sequence_length,
            vec![QwenImage21AttentionSegment::new(
                sequence_length,
                bidirectional,
            )],
            vec![vec![QwenImage21AttentionKeyRun::new(sequence_length, true)]],
        )
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
    pub(super) fn attributes(
        &self,
        head_dim: u64,
        scale: CanonicalRational,
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let mut attrs = BTreeMap::from([
            (
                AttributeId::new("sequence_length")?,
                SemanticValue::Unsigned(u64::from(self.sequence_length)),
            ),
            (
                AttributeId::new("head_dim")?,
                SemanticValue::Unsigned(head_dim),
            ),
            (AttributeId::new("scale")?, SemanticValue::Rational(scale)),
        ]);
        for (name, values) in [
            (
                "segment_lengths",
                self.segments.iter().map(|s| i64::from(s.length)).collect(),
            ),
            (
                "segment_bidirectional",
                self.segments
                    .iter()
                    .map(|s| if s.bidirectional { 1 } else { 0 })
                    .collect(),
            ),
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
                    .map(|r| if r.valid { 1 } else { 0 })
                    .collect(),
            ),
        ] {
            attrs.insert(AttributeId::new(name)?, SemanticValue::Integers(values));
        }
        Ok(attrs)
    }
}

/// Canonical family configuration binds input layout and exact effective scale
/// to the checked transformer dimensions. It is not an upstream HF config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21AttentionConfig {
    transformer: QwenImage21Config,
    layout: QwenImage21AttentionLayout,
    scale: CanonicalRational,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigWire {
    transformer: QwenImage21Config,
    layout: QwenImage21AttentionLayout,
    scale: CanonicalRational,
}
impl<'de> Deserialize<'de> for QwenImage21AttentionConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = ConfigWire::deserialize(d)?;
        let config = Self {
            transformer: wire.transformer,
            layout: wire.layout,
            scale: wire.scale,
        };
        config.validate().map_err(serde::de::Error::custom)?;
        Ok(config)
    }
}
impl QwenImage21AttentionConfig {
    pub fn new(
        transformer: QwenImage21Config,
        layout: QwenImage21AttentionLayout,
    ) -> Result<Self, VNextError> {
        let scale = QwenImage21Block0AttentionFamily::scale_for_head_dim(transformer.head_dim())?;
        let config = Self {
            transformer,
            layout,
            scale,
        };
        config.validate()?;
        Ok(config)
    }
    pub fn transformer(&self) -> &QwenImage21Config {
        &self.transformer
    }
    pub fn layout(&self) -> &QwenImage21AttentionLayout {
        &self.layout
    }
    pub const fn scale(&self) -> CanonicalRational {
        self.scale
    }
    pub fn effective_scale(&self) -> f32 {
        (self.scale.numerator() as f64 / self.scale.denominator() as f64) as f32
    }
    pub fn attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        self.layout
            .attributes(self.transformer.head_dim(), self.scale)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if self.scale
            != QwenImage21Block0AttentionFamily::scale_for_head_dim(self.transformer.head_dim())?
        {
            return Err(invalid(
                "scale",
                "serialized scale differs from the model's exact effective FP32 scale",
            ));
        }
        let spec = ResolvedTensorSpec::new(
            vec![self.layout.rows(), self.transformer.hidden_size()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        StructuredAttentionBf16Geometry::new(&spec, &spec, &spec, &spec, &self.attributes()?)?;
        Ok(())
    }
}
