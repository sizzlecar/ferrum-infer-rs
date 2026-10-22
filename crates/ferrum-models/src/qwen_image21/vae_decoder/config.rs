use super::*;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21VaeDecoderConfig {
    latent_channels: u64,
    base_channels: u64,
    dim_mult: Vec<u64>,
    num_res_blocks: u64,
    temporal_upsample: Vec<bool>,
    batch: u64,
    height: u64,
    width: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    latent_channels: u64,
    base_channels: u64,
    dim_mult: Vec<u64>,
    num_res_blocks: u64,
    temporal_upsample: Vec<bool>,
    batch: u64,
    height: u64,
    width: u64,
}
impl<'de> Deserialize<'de> for QwenImage21VaeDecoderConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(
            w.latent_channels,
            w.base_channels,
            w.dim_mult,
            w.num_res_blocks,
            w.temporal_upsample,
            w.batch,
            w.height,
            w.width,
        )
        .map_err(serde::de::Error::custom)
    }
}
impl QwenImage21VaeDecoderConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        latent_channels: u64,
        base_channels: u64,
        dim_mult: Vec<u64>,
        num_res_blocks: u64,
        temporal_upsample: Vec<bool>,
        batch: u64,
        height: u64,
        width: u64,
    ) -> Result<Self, VNextError> {
        let c = Self {
            latent_channels,
            base_channels,
            dim_mult,
            num_res_blocks,
            temporal_upsample,
            batch,
            height,
            width,
        };
        c.validate()?;
        Ok(c)
    }
    /// Architecture-only adaptation. Mean/std are intentionally not consumed:
    /// the single input is explicitly an already denormalized BF16 latent.
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
            || raw["is_residual"] != true
            || !raw["patch_size"].is_null()
            || raw["dropout"].as_f64() != Some(0.0)
            || raw["out_channels"].as_u64() != Some(4)
            || raw["attn_scales"].as_array().is_none_or(|a| !a.is_empty())
        {
            return Err(invalid(
                "config",
                "requires unpatched residual RGBA decoder, zero dropout, empty attention scales",
            ));
        }
        let unsigned = |name: &str| {
            raw[name]
                .as_u64()
                .ok_or_else(|| invalid(name, "missing unsigned architecture field"))
        };
        let base = raw
            .get("decoder_base_dim")
            .filter(|v| !v.is_null())
            .unwrap_or(&raw["base_dim"])
            .as_u64()
            .ok_or_else(|| invalid("decoder_base_dim", "missing unsigned base width"))?;
        let mult = raw["dim_mult"]
            .as_array()
            .ok_or_else(|| invalid("dim_mult", "missing multipliers"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| invalid("dim_mult", "requires unsigned multipliers"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let temporal = raw["temperal_downsample"]
            .as_array()
            .ok_or_else(|| invalid("temperal_downsample", "missing temporal flags"))?
            .iter()
            .rev()
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| invalid("temperal_downsample", "requires booleans"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            unsigned("z_dim")?,
            base,
            mult,
            unsigned("num_res_blocks")?,
            temporal,
            batch,
            height,
            width,
        )
    }
    pub const fn latent_channels(&self) -> u64 {
        self.latent_channels
    }
    pub const fn base_channels(&self) -> u64 {
        self.base_channels
    }
    pub fn dim_mult(&self) -> &[u64] {
        &self.dim_mult
    }
    pub const fn num_res_blocks(&self) -> u64 {
        self.num_res_blocks
    }
    pub fn temporal_upsample(&self) -> &[bool] {
        &self.temporal_upsample
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
    pub fn decoder_channels(&self) -> Result<Vec<u64>, VNextError> {
        let first = *self
            .dim_mult
            .last()
            .ok_or_else(|| invalid("dim_mult", "empty multipliers"))?;
        std::iter::once(first)
            .chain(self.dim_mult.iter().rev().copied())
            .map(|m| {
                m.checked_mul(self.base_channels)
                    .ok_or_else(|| invalid("channels", "width overflows"))
            })
            .collect()
    }
    pub fn input_tensor(&self) -> Result<ResolvedTensorSpec, VNextError> {
        tensor(&[self.batch, self.latent_channels, self.height, self.width])
    }
    pub fn output_tensor(&self) -> Result<ResolvedTensorSpec, VNextError> {
        let factor = 1_u64
            .checked_shl((self.dim_mult.len() - 1) as u32)
            .ok_or_else(|| invalid("geometry", "scale overflows"))?;
        tensor(&[
            self.batch,
            4,
            self.height
                .checked_mul(factor)
                .ok_or_else(|| invalid("height", "output overflows"))?,
            self.width
                .checked_mul(factor)
                .ok_or_else(|| invalid("width", "output overflows"))?,
        ])
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([(
            ProgramValueId::new(QwenImage21VaeDecoderFamily::INPUT_ID)?,
            spec(self.input_tensor()?.dimensions()),
        )]))
    }
    pub fn expected_weights(&self) -> Result<Vec<QwenImage21VaeDecoderWeight>, VNextError> {
        Ok(graph::build(self)?.weights)
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        let mut options = ProgramPlanCompileOptions::new(self.input_tensor_specs()?)?;
        for (id, spec) in graph::build(self)?.specs {
            options.insert_tensor_spec(id, spec)?;
        }
        Ok(options)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if [
            self.latent_channels,
            self.base_channels,
            self.batch,
            self.height,
            self.width,
        ]
        .contains(&0)
            || !(1..=8).contains(&self.dim_mult.len())
            || self.dim_mult.contains(&0)
            || self.num_res_blocks > 16
            || self.temporal_upsample.len() + 1 != self.dim_mult.len()
        {
            return Err(invalid("geometry","requires positive extents, 1..8 up blocks, 0..16 residual depth and one temporal flag per upsample"));
        }
        let channels = self.decoder_channels()?;
        if channels
            .iter()
            .any(|&c| c > SPATIAL_ATTENTION_BF16_MAX_CHANNELS)
        {
            return Err(invalid(
                "channels",
                "channel width exceeds the declared spatial-attention domain",
            ));
        }
        for (i, pair) in channels
            .windows(2)
            .enumerate()
            .take(self.temporal_upsample.len())
        {
            let factor = if self.temporal_upsample[i] { 8 } else { 4 };
            if pair[0] % pair[1] != 0 || factor % (pair[0] / pair[1]) != 0 {
                return Err(invalid(
                    "shortcut",
                    "DupUp first-frame mapping is not representable by the declared channel gather",
                ));
            }
        }
        self.input_tensor()?.minimum_storage_bytes()?;
        self.output_tensor()?.minimum_storage_bytes()?;
        Ok(())
    }
}
