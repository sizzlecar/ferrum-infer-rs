use super::*;
use serde::{Deserialize, Deserializer, Serialize};

#[path = "edit.rs"]
mod edit;
pub use edit::QwenImage21ReferenceImage;

/// Homogeneous single-frame geometry. `text_tokens` is the cropped VLM length,
/// including visual slots when a reference image is configured. All samples
/// share slot positions and grids; right-padding validity may differ.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21T2iLayout {
    text_tokens: u32,
    latent_height: u32,
    latent_width: u32,
    valid_text_lengths: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reference_image: Option<QwenImage21ReferenceImage>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayoutWire {
    text_tokens: u32,
    latent_height: u32,
    latent_width: u32,
    valid_text_lengths: Vec<u32>,
    #[serde(default)]
    reference_image: Option<QwenImage21ReferenceImage>,
}
impl<'de> Deserialize<'de> for QwenImage21T2iLayout {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = LayoutWire::deserialize(d)?;
        let mut layout = Self::new(
            wire.text_tokens,
            wire.latent_height,
            wire.latent_width,
            wire.valid_text_lengths,
        )
        .map_err(serde::de::Error::custom)?;
        if let Some(reference) = wire.reference_image {
            layout = layout
                .with_reference(reference)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(layout)
    }
}
impl QwenImage21T2iLayout {
    pub fn new(
        text_tokens: u32,
        latent_height: u32,
        latent_width: u32,
        valid_text_lengths: Vec<u32>,
    ) -> Result<Self, VNextError> {
        // Upstream positive frequency indices are 0..8192 and negative spatial
        // indices are -1024..0. Pixel alignment is 32, hence even latent axes.
        if text_tokens == 0
            || text_tokens >= 8192
            || latent_height == 0
            || latent_width == 0
            || latent_height > 2048
            || latent_width > 2048
            || latent_height % 2 != 0
            || latent_width % 2 != 0
            || valid_text_lengths.is_empty()
            || valid_text_lengths.len() > STRUCTURED_ATTENTION_BF16_MAX_BATCH
            || valid_text_lengths
                .iter()
                .any(|length| *length > text_tokens)
        {
            return Err(invalid("layout","requires nonempty text within rotary index domain, positive even latent axes <=2048, and 1..16 valid prompt lengths <= padded text length"));
        }
        let layout = Self {
            text_tokens,
            latent_height,
            latent_width,
            valid_text_lengths,
            reference_image: None,
        };
        layout
            .target_tokens()
            .checked_add(u64::from(text_tokens))
            .filter(|v| *v <= u64::from(u32::MAX))
            .ok_or_else(|| invalid("layout", "joint sequence overflows u32"))?;
        layout.attention_layout()?;
        Ok(layout)
    }
    pub fn batch(&self) -> u64 {
        self.valid_text_lengths.len() as u64
    }
    pub const fn text_tokens(&self) -> u32 {
        self.text_tokens
    }
    pub const fn latent_height(&self) -> u32 {
        self.latent_height
    }
    pub const fn latent_width(&self) -> u32 {
        self.latent_width
    }
    pub fn valid_text_lengths(&self) -> &[u32] {
        &self.valid_text_lengths
    }
    pub fn target_tokens(&self) -> u64 {
        u64::from(self.latent_height) * u64::from(self.latent_width)
    }
    pub fn sequence_length(&self) -> u64 {
        u64::from(self.text_tokens)
            + self.target_tokens()
            + self
                .reference_image
                .as_ref()
                .map_or(0, |r| r.latent_tokens() - u64::from(r.image_slots()))
    }
    pub fn rows(&self) -> u64 {
        self.batch() * self.sequence_length()
    }
    /// Axis order matches adjacent complex pairs: frame, height, width. Padding
    /// still occupies text positions; target frame uses padded T for all samples.
    pub fn rotary_position(&self, row: u64) -> Option<[i32; 3]> {
        if self.reference_image.is_some() {
            return self.edit_rotary_position(row);
        }
        if row >= self.sequence_length() {
            return None;
        }
        if row < u64::from(self.text_tokens) {
            return Some([row as i32; 3]);
        }
        let image = row - u64::from(self.text_tokens);
        Some([
            self.text_tokens as i32,
            (image / u64::from(self.latent_width)) as i32
                - (self.latent_height - self.latent_height / 2) as i32,
            (image % u64::from(self.latent_width)) as i32
                - (self.latent_width - self.latent_width / 2) as i32,
        ])
    }
    pub fn attention_layout(&self) -> Result<QwenImage21AttentionLayout, VNextError> {
        if self.reference_image.is_some() {
            return self.edit_attention_layout();
        }
        let image = u32::try_from(self.target_tokens())
            .map_err(|_| invalid("layout", "target length exceeds u32"))?;
        let runs = self
            .valid_text_lengths
            .iter()
            .map(|valid| {
                let mut runs = Vec::new();
                if *valid == self.text_tokens {
                    runs.push(QwenImage21AttentionKeyRun::new(
                        self.text_tokens + image,
                        true,
                    ));
                } else {
                    if *valid > 0 {
                        runs.push(QwenImage21AttentionKeyRun::new(*valid, true));
                    }
                    runs.push(QwenImage21AttentionKeyRun::new(
                        self.text_tokens - *valid,
                        false,
                    ));
                    runs.push(QwenImage21AttentionKeyRun::new(image, true));
                }
                runs
            })
            .collect();
        QwenImage21AttentionLayout::new(
            self.text_tokens + image,
            vec![
                QwenImage21AttentionSegment::new(self.text_tokens, false),
                QwenImage21AttentionSegment::new(image, true),
            ],
            runs,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21TransformerConfig {
    transformer: QwenImage21Config,
    layout: QwenImage21T2iLayout,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigWire {
    transformer: QwenImage21Config,
    layout: QwenImage21T2iLayout,
}
impl<'de> Deserialize<'de> for QwenImage21TransformerConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = ConfigWire::deserialize(d)?;
        Self::new(wire.transformer, wire.layout).map_err(serde::de::Error::custom)
    }
}
impl QwenImage21TransformerConfig {
    pub fn new(
        transformer: QwenImage21Config,
        layout: QwenImage21T2iLayout,
    ) -> Result<Self, VNextError> {
        let config = Self {
            transformer,
            layout,
        };
        config.validate()?;
        Ok(config)
    }
    pub fn transformer(&self) -> &QwenImage21Config {
        &self.transformer
    }
    pub fn layout(&self) -> &QwenImage21T2iLayout {
        &self.layout
    }
    pub fn time_feature_rows(&self) -> u64 {
        self.layout.batch()
            + if self.transformer.causal_condition() {
                1
            } else {
                0
            }
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        type F = QwenImage21TransformerFamily;
        let l = &self.layout;
        let t = &self.transformer;
        let specs = [
            (
                F::LATENTS_ID,
                vec![l.batch() * l.target_tokens(), t.in_channels()],
                ElementType::Bf16,
            ),
            (
                F::TEXT_ID,
                vec![l.batch() * u64::from(l.text_tokens), t.context_dim()],
                ElementType::Bf16,
            ),
            (
                F::TIME_FEATURES_ID,
                vec![self.time_feature_rows(), 256],
                ElementType::Bf16,
            ),
            (
                F::FREQUENCIES_ID,
                vec![l.sequence_length(), t.head_dim() / 2, 2],
                ElementType::F32,
            ),
        ];
        let mut specs = specs
            .into_iter()
            .map(|(id, dimensions, element_type)| {
                let spec = ProgramTensorSpec {
                    dimensions,
                    element_type,
                    layout: ResolvedTensorLayout::Contiguous,
                };
                spec.validate(id)?;
                Ok((ProgramValueId::new(id)?, spec))
            })
            .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
        if let Some(reference) = l.reference_image() {
            let reference_spec = ProgramTensorSpec {
                dimensions: vec![l.batch() * reference.latent_tokens(), t.in_channels()],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            };
            reference_spec.validate(F::REFERENCE_LATENTS_ID)?;
            specs.insert(
                ProgramValueId::new(F::REFERENCE_LATENTS_ID)?,
                reference_spec,
            );
        }
        Ok(specs)
    }
    /// Compile every layer as one fixed-tensor program. Transport output
    /// dimensions depend on typed run-map attributes, so they are supplied
    /// explicitly and revalidated by the generic transport providers.
    /// No intermediate value is retained as a completion diagnostic.
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        let mut options = ProgramPlanCompileOptions::new(self.input_tensor_specs()?)?;
        let l = &self.layout;
        let h = self.transformer.hidden_size();
        if let Some(reference) = l.reference_image() {
            for (name, rows, columns) in [
                (
                    program::COMBINED_LATENTS,
                    l.batch() * (reference.latent_tokens() + l.target_tokens()),
                    self.transformer.in_channels(),
                ),
                (
                    program::UNORDERED_JOINT,
                    l.batch()
                        * (u64::from(l.text_tokens())
                            + reference.latent_tokens()
                            + l.target_tokens()),
                    h,
                ),
            ] {
                options.insert_tensor_spec(
                    ProgramValueId::new(name)?,
                    ProgramTensorSpec {
                        dimensions: vec![rows, columns],
                        element_type: ElementType::Bf16,
                        layout: ResolvedTensorLayout::Contiguous,
                    },
                )?;
            }
        }
        for (name, rows) in std::iter::once((program::JOINT, l.rows()))
            .chain(
                program::CONDITIONING
                    .into_iter()
                    .map(|name| (name, l.rows())),
            )
            .chain([
                (program::TARGET_HIDDEN, l.batch() * l.target_tokens()),
                (program::FINAL_SCALE, l.batch() * l.target_tokens()),
            ])
        {
            options.insert_tensor_spec(
                ProgramValueId::new(name)?,
                ProgramTensorSpec {
                    dimensions: vec![rows, h],
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            )?;
        }
        Ok(options)
    }
    pub(super) fn block_config(&self) -> Result<QwenImage21Block0Config, VNextError> {
        QwenImage21Block0Config::new(
            QwenImage21AttentionConfig::new(
                self.transformer.clone(),
                self.layout.attention_layout()?,
            )?,
            QwenImage21Block0Broadcast::Elementwise,
        )
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        self.block_config()?;
        let inputs = self.input_tensor_specs()?;
        let text = &inputs[&ProgramValueId::new(QwenImage21TransformerFamily::TEXT_ID)?];
        let text = ResolvedTensorSpec::new(
            text.dimensions.clone(),
            text.element_type,
            text.layout.clone(),
        )?;
        let gamma = ResolvedTensorSpec::new(
            vec![self.transformer.context_dim()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        ZeroCenteredRmsNormBf16Geometry::new(&text, &gamma, &text, self.transformer.epsilon())?;
        ExpectedWeights::from_config(&self.transformer, StructureLimits::default())
            .map_err(|e| invalid("weights", e.to_string()))?;
        Ok(())
    }
}
