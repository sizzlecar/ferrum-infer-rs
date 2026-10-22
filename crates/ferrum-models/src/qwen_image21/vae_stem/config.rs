use super::*;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21VaeStemConfig {
    latent_channels: u64,
    hidden_channels: u64,
    batch: u64,
    height: u64,
    width: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    latent_channels: u64,
    hidden_channels: u64,
    batch: u64,
    height: u64,
    width: u64,
}
impl<'de> Deserialize<'de> for QwenImage21VaeStemConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(
            w.latent_channels,
            w.hidden_channels,
            w.batch,
            w.height,
            w.width,
        )
        .map_err(serde::de::Error::custom)
    }
}
impl QwenImage21VaeStemConfig {
    pub fn new(
        latent_channels: u64,
        hidden_channels: u64,
        batch: u64,
        height: u64,
        width: u64,
    ) -> Result<Self, VNextError> {
        let c = Self {
            latent_channels,
            hidden_channels,
            batch,
            height,
            width,
        };
        c.validate()?;
        Ok(c)
    }
    /// Adapts only the architecture fields used by this prefix. The input has
    /// already undergone the separate channel mean/std denormalization step.
    pub fn from_hf_json(
        bytes: &[u8],
        batch: u64,
        height: u64,
        width: u64,
    ) -> Result<Self, VNextError> {
        if bytes.is_empty() || bytes.len() > 4 * 1024 * 1024 {
            return Err(invalid(
                "config",
                "HF VAE config must be nonempty and <=4 MiB",
            ));
        }
        let raw: Value =
            serde_json::from_slice(bytes).map_err(|e| invalid("config", e.to_string()))?;
        if raw["_class_name"] != "AutoencoderKLQwenImage21"
            || !raw["patch_size"].is_null()
            || raw["dropout"].as_f64() != Some(0.0)
        {
            return Err(invalid(
                "config",
                "supports this unpatched VAE prefix with zero dropout only",
            ));
        }
        let z = raw["z_dim"]
            .as_u64()
            .ok_or_else(|| invalid("z_dim", "missing unsigned channel count"))?;
        let base = raw
            .get("decoder_base_dim")
            .filter(|x| !x.is_null())
            .unwrap_or(&raw["base_dim"])
            .as_u64()
            .ok_or_else(|| invalid("decoder_base_dim", "missing unsigned decoder base width"))?;
        let multipliers = raw["dim_mult"]
            .as_array()
            .ok_or_else(|| invalid("dim_mult", "missing dimension multipliers"))?;
        if multipliers.is_empty()
            || multipliers
                .iter()
                .any(|v| v.as_u64().is_none_or(|n| n == 0))
        {
            return Err(invalid(
                "dim_mult",
                "requires nonempty positive unsigned multipliers",
            ));
        }
        let hidden = base
            .checked_mul(multipliers.last().unwrap().as_u64().unwrap())
            .ok_or_else(|| invalid("decoder_base_dim", "channel product overflows"))?;
        Self::new(z, hidden, batch, height, width)
    }
    pub const fn latent_channels(&self) -> u64 {
        self.latent_channels
    }
    pub const fn hidden_channels(&self) -> u64 {
        self.hidden_channels
    }
    pub const fn batch(&self) -> u64 {
        self.batch
    }
    pub const fn height(&self) -> u64 {
        self.height
    }
    pub const fn width(&self) -> u64 {
        self.width
    }
    pub fn tensor(&self, channels: u64) -> Result<ResolvedTensorSpec, VNextError> {
        ResolvedTensorSpec::new(
            vec![self.batch, channels, self.height, self.width],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([(
            ProgramValueId::new(QwenImage21VaeStemFamily::INPUT_ID)?,
            spec(&self.tensor(self.latent_channels)?),
        )]))
    }
    pub fn output_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        QwenImage21VaeStemFamily::OUTPUT_IDS
            .into_iter()
            .enumerate()
            .map(|(i, id)| {
                Ok((
                    ProgramValueId::new(id)?,
                    spec(&self.tensor(if i == 0 {
                        self.latent_channels
                    } else {
                        self.hidden_channels
                    })?),
                ))
            })
            .collect()
    }
    /// Conv output height/width derive from attributes, so insert their exact
    /// specifications explicitly. Providers independently recheck geometry.
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        let mut out = ProgramPlanCompileOptions::new(self.input_tensor_specs()?)?;
        for (id, spec) in self.output_tensor_specs()? {
            out.insert_tensor_spec(id, spec)?;
        }
        Ok(out)
    }
    pub fn expected_weights(&self) -> Result<Vec<QwenImage21VaeStemWeight>, VNextError> {
        weights::expected(self)
    }
    pub fn norm_attributes(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        // Python computes sqrt(C) in binary64; the op's scalar parameter is its
        // effective FP32 value, represented exactly as a canonical rational.
        let scale = (self.hidden_channels as f64).sqrt() as f32;
        let bits = scale.to_bits();
        let shift = 150_i32 - ((bits >> 23) & 255) as i32;
        if !(0..64).contains(&shift) || !scale.is_finite() || scale <= 0.0 {
            return Err(invalid(
                "norm_scale",
                "cannot represent the positive FP32 scalar",
            ));
        }
        let scale =
            CanonicalRational::new(i64::from((bits & 0x7fffff) | (1 << 23)), 1_u64 << shift)?;
        Ok(BTreeMap::from([
            (
                AttributeId::new("epsilon")?,
                SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000)?),
            ),
            (AttributeId::new("scale")?, SemanticValue::Rational(scale)),
        ]))
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if [
            self.latent_channels,
            self.hidden_channels,
            self.batch,
            self.height,
            self.width,
        ]
        .contains(&0)
            || self.hidden_channels > CHANNEL_L2_NORM_BF16_MAX_CHANNELS
        {
            return Err(invalid(
                "geometry",
                "requires positive single-frame NCHW dimensions and supported channel count",
            ));
        }
        let latent = self.tensor(self.latent_channels)?;
        let hidden = self.tensor(self.hidden_channels)?;
        let gamma = ResolvedTensorSpec::new(
            vec![self.hidden_channels, 1, 1, 1],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        ChannelL2NormBf16Geometry::new(&hidden, &gamma, &hidden, &self.norm_attributes()?)?;
        for (input, output, k) in [
            (&latent, &latent, 1),
            (&latent, &hidden, 3),
            (&hidden, &hidden, 3),
        ] {
            let weight = ResolvedTensorSpec::new(
                vec![output.dimensions()[1], input.dimensions()[1], k, k],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            let bias = ResolvedTensorSpec::new(
                vec![output.dimensions()[1]],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            Conv2dBf16Geometry::new(
                input,
                &weight,
                &bias,
                output,
                &super::graph::conv_attributes(k)?,
            )?;
        }
        Ok(())
    }
}
fn spec(t: &ResolvedTensorSpec) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: t.dimensions().to_vec(),
        element_type: t.element_type(),
        layout: t.layout().clone(),
    }
}
