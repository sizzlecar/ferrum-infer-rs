use super::*;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21VaeEncoderConfig {
    latent_channels: u64,
    base_channels: u64,
    dim_mult: Vec<u64>,
    num_res_blocks: u64,
    temporal_downsample: Vec<bool>,
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
    temporal_downsample: Vec<bool>,
    batch: u64,
    height: u64,
    width: u64,
}
impl<'de> Deserialize<'de> for QwenImage21VaeEncoderConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(
            w.latent_channels,
            w.base_channels,
            w.dim_mult,
            w.num_res_blocks,
            w.temporal_downsample,
            w.batch,
            w.height,
            w.width,
        )
        .map_err(serde::de::Error::custom)
    }
}
impl QwenImage21VaeEncoderConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        latent_channels: u64,
        base_channels: u64,
        dim_mult: Vec<u64>,
        num_res_blocks: u64,
        temporal_downsample: Vec<bool>,
        batch: u64,
        height: u64,
        width: u64,
    ) -> Result<Self, VNextError> {
        let config = Self {
            latent_channels,
            base_channels,
            dim_mult,
            num_res_blocks,
            temporal_downsample,
            batch,
            height,
            width,
        };
        config.validate()?;
        Ok(config)
    }
    /// H/W are normalized pixel dimensions, not latent dimensions. The
    /// encoder base comes from base_dim, never decoder_base_dim.
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
            || raw["in_channels"].as_u64() != Some(4)
            || raw["attn_scales"].as_array().is_none_or(|v| !v.is_empty())
        {
            return Err(invalid(
                "config",
                "requires unpatched residual RGBA encoder, zero dropout and empty attention scales",
            ));
        }
        let unsigned = |name: &str| {
            raw[name]
                .as_u64()
                .ok_or_else(|| invalid(name, "missing unsigned architecture field"))
        };
        let multipliers = raw["dim_mult"]
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
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| invalid("temperal_downsample", "requires booleans"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            unsigned("z_dim")?,
            unsigned("base_dim")?,
            multipliers,
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
    pub fn temporal_downsample(&self) -> &[bool] {
        &self.temporal_downsample
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
    pub fn spatial_scale(&self) -> u64 {
        1_u64 << (self.dim_mult.len() - 1)
    }
    pub fn encoder_channels(&self) -> Result<Vec<u64>, VNextError> {
        std::iter::once(1)
            .chain(self.dim_mult.iter().copied())
            .map(|m| {
                m.checked_mul(self.base_channels)
                    .ok_or_else(|| invalid("channels", "width overflows"))
            })
            .collect()
    }
    pub fn input_tensor(&self) -> Result<ResolvedTensorSpec, VNextError> {
        tensor(&[self.batch, 4, self.height, self.width])
    }
    pub fn output_tensor(&self) -> Result<ResolvedTensorSpec, VNextError> {
        tensor(&[
            self.batch,
            self.latent_channels,
            self.height / self.spatial_scale(),
            self.width / self.spatial_scale(),
        ])
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([(
            ProgramValueId::new(QwenImage21VaeEncoderFamily::INPUT_ID)?,
            spec(self.input_tensor()?.dimensions()),
        )]))
    }
    pub fn expected_weights(&self) -> Result<Vec<QwenImage21VaeEncoderWeight>, VNextError> {
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
            || !(1..=16).contains(&self.num_res_blocks)
            || self.temporal_downsample.len() + 1 != self.dim_mult.len()
        {
            return Err(invalid("geometry", "requires positive extents, 1..8 blocks, 1..16 residual depth and one temporal flag per downsample"));
        }
        let scale = self.spatial_scale();
        if self.height % scale != 0 || self.width % scale != 0 {
            return Err(invalid(
                "geometry",
                "pixel axes must be exactly divisible by the encoder spatial scale",
            ));
        }
        let channels = self.encoder_channels()?;
        if channels
            .iter()
            .any(|&c| c > SPATIAL_ATTENTION_BF16_MAX_CHANNELS)
        {
            return Err(invalid(
                "channels",
                "channel width exceeds spatial attention domain",
            ));
        }
        for (i, pair) in channels.windows(2).enumerate() {
            let temporal = if self.temporal_downsample.get(i) == Some(&true) {
                2
            } else {
                1
            };
            let spatial = if i < self.temporal_downsample.len() {
                4
            } else {
                1
            };
            if pair[0]
                .checked_mul(temporal * spatial)
                .is_none_or(|n| n % pair[1] != 0)
            {
                return Err(invalid(
                    "shortcut",
                    "rearranged channels must divide exactly into output averaging groups",
                ));
            }
        }
        self.latent_channels
            .checked_mul(2)
            .ok_or_else(|| invalid("latent_channels", "posterior channels overflow"))?;
        self.input_tensor()?.minimum_storage_bytes()?;
        self.output_tensor()?.minimum_storage_bytes()?;
        Ok(())
    }
}
