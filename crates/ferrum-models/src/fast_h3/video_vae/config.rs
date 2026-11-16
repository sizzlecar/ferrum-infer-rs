use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// Original video configuration. This is deliberately not a P-H3 image config.
/// Metadata does not grant Host/device capacity or select an attention backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3VideoVaeConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    in_channels: usize,
    out_channels: usize,
    latent_channels: usize,
    block_out_channels: Vec<usize>,
    layers_per_block: usize,
    spatial_downsample_factors: Vec<usize>,
    temporal_downsample_factors: Vec<usize>,
    norm_num_groups: usize,
    norm_eps: f64,
    spatial_padding_mode: String,
    decoder_num_layers: usize,
    decoder_num_attention_heads: usize,
    decoder_attention_head_dim: usize,
    decoder_num_register_tokens: usize,
    decoder_ffn_mult: usize,
    decoder_rope_theta: f64,
    decoder_rope_dim_ratio: f64,
    decoder_norm_eps: f64,
    clip_length: usize,
    token_drop: usize,
    latents_mean: Vec<f64>,
    latents_std: Vec<f64>,
}

impl FastH3VideoVaeConfig {
    pub fn parse(bytes: &[u8], maximum_metadata_bytes: usize) -> Result<Self> {
        ensure!(
            !bytes.is_empty() && bytes.len() <= maximum_metadata_bytes,
            "video VAE config exceeds metadata capacity"
        );
        let config: Self = serde_json::from_slice(bytes)?;
        config.validate()?;
        Ok(config)
    }

    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.class_name == "AutoencoderKLMiniMaxH3",
            "not an original H3 video VAE"
        );
        ensure!(
            self.in_channels == 3 && self.out_channels == 3,
            "video VAE requires RGB channels"
        );
        ensure!(
            self.latent_channels > 0
                && self.layers_per_block > 0
                && self.decoder_num_layers > 0
                && self.decoder_num_attention_heads > 0
                && self.decoder_attention_head_dim > 0
                && self.decoder_ffn_mult > 0
                && self.decoder_num_register_tokens > 0
                && self.clip_length > 0
                && self.norm_num_groups > 0,
            "video VAE dimensions must be positive"
        );
        let stages = self.block_out_channels.len();
        ensure!(
            stages > 0
                && self.spatial_downsample_factors.len() == stages
                && self.temporal_downsample_factors.len() == stages,
            "video VAE encoder stage metadata differs"
        );
        ensure!(
            self.block_out_channels
                .iter()
                .all(|c| *c > 0 && c % self.norm_num_groups == 0),
            "video VAE encoder groups must divide every channel width"
        );
        ensure!(
            self.spatial_padding_mode == "reflect",
            "unsupported spatial padding profile"
        );
        for eps in [self.norm_eps, self.decoder_norm_eps] {
            ensure!(
                eps.is_finite() && eps > 0. && (eps as f32).is_finite() && (eps as f32) > 0.,
                "VAE normalization epsilon must be positive finite F32"
            );
        }
        ensure!(
            self.decoder_rope_theta.is_finite()
                && self.decoder_rope_theta > 0.
                && self.decoder_rope_dim_ratio.is_finite()
                && self.decoder_rope_dim_ratio > 0.
                && self.decoder_rope_dim_ratio <= 1.,
            "invalid VAE rotary parameters"
        );
        let rotary = self.decoder_attention_head_dim as f64 * self.decoder_rope_dim_ratio;
        ensure!(
            rotary.fract() == 0. && rotary > 0. && (rotary as usize) % 6 == 0,
            "VAE three-axis rotary width must be an exact positive multiple of six"
        );
        ensure!(
            self.latents_mean.len() == self.latent_channels
                && self.latents_std.len() == self.latent_channels,
            "VAE latent statistics must cover every channel"
        );
        ensure!(
            self.latents_mean
                .iter()
                .all(|x| x.is_finite() && (*x as f32).is_finite())
                && self.latents_std.iter().all(|x| x.is_finite()
                    && *x > 0.
                    && (*x as f32).is_finite()
                    && (*x as f32) > 0.),
            "VAE latent statistics require finite F32 values and positive scales"
        );
        self.hidden_size()?;
        self.ffn_size()?;
        self.output_patch_width()?;
        let ratio = self.temporal_compression_ratio()?;
        let tokens = self
            .clip_length
            .checked_add(ratio - 1)
            .context("VAE clip geometry overflow")?
            / ratio;
        ensure!(
            self.token_drop < tokens,
            "VAE token drop must be less than a clip"
        );
        self.decoder_num_register_tokens
            .checked_add(1)
            .context("VAE suffix overflow")?;
        Ok(())
    }

    pub fn latent_channels(&self) -> usize {
        self.latent_channels
    }
    /// Shared causal CNN metadata; the caller declares literal T=1 or the
    /// complete NCTHW graph. This contains no decoder approximation.
    pub(super) fn causal_encoder_architecture(
        &self,
    ) -> Result<crate::p_h3_image_vae::PH3ImageVaeEncoderArchitecture> {
        self.validate()?;
        Ok(crate::p_h3_image_vae::PH3ImageVaeEncoderArchitecture::new(
            self.in_channels.try_into()?,
            self.latent_channels.try_into()?,
            self.block_out_channels
                .iter()
                .copied()
                .map(u64::try_from)
                .collect::<Result<_, _>>()?,
            self.layers_per_block.try_into()?,
            self.spatial_downsample_factors
                .iter()
                .copied()
                .map(u64::try_from)
                .collect::<Result<_, _>>()?,
            self.temporal_downsample_factors
                .iter()
                .copied()
                .map(u64::try_from)
                .collect::<Result<_, _>>()?,
            self.norm_num_groups.try_into()?,
            ferrum_runtime::CanonicalRational::from_decimal_str(&self.norm_eps.to_string())?,
        )?)
    }
    pub fn decoder_layers(&self) -> usize {
        self.decoder_num_layers
    }
    pub fn attention_heads(&self) -> usize {
        self.decoder_num_attention_heads
    }
    pub fn attention_head_dim(&self) -> usize {
        self.decoder_attention_head_dim
    }
    pub fn register_tokens(&self) -> usize {
        self.decoder_num_register_tokens
    }
    pub fn clip_length(&self) -> usize {
        self.clip_length
    }
    pub fn token_drop(&self) -> usize {
        self.token_drop
    }
    pub fn latent_mean(&self) -> &[f64] {
        &self.latents_mean
    }
    pub fn latent_std(&self) -> &[f64] {
        &self.latents_std
    }
    pub fn norm_epsilon(&self) -> f64 {
        self.decoder_norm_eps
    }
    pub fn rotary_theta(&self) -> f64 {
        self.decoder_rope_theta
    }
    pub fn rotary_dimension(&self) -> usize {
        (self.decoder_attention_head_dim as f64 * self.decoder_rope_dim_ratio) as usize
    }
    pub fn hidden_size(&self) -> Result<usize> {
        self.decoder_num_attention_heads
            .checked_mul(self.decoder_attention_head_dim)
            .context("video VAE hidden width overflow")
    }
    pub fn ffn_size(&self) -> Result<usize> {
        self.hidden_size()?
            .checked_mul(self.decoder_ffn_mult)
            .context("video VAE FFN width overflow")
    }
    pub fn spatial_compression_ratio(&self) -> Result<usize> {
        product(&self.spatial_downsample_factors)
    }
    pub fn temporal_compression_ratio(&self) -> Result<usize> {
        product(&self.temporal_downsample_factors)
    }
    pub fn output_patch_width(&self) -> Result<usize> {
        self.out_channels
            .checked_mul(self.temporal_compression_ratio()?)
            .and_then(|x| x.checked_mul(self.spatial_compression_ratio().ok()?))
            .and_then(|x| x.checked_mul(self.spatial_compression_ratio().ok()?))
            .context("video VAE output patch width overflow")
    }
}

fn product(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |n, f| {
        ensure!(*f > 0, "VAE compression factors must be positive");
        n.checked_mul(*f).context("VAE compression ratio overflow")
    })
}
