use super::{invalid, PH3ImageVaeDecoderWeight, PH3_IMAGE_VAE_CLASS};
use ferrum_runtime::{
    CanonicalRational, ElementType, ResolvedTensorLayout, ResolvedTensorSpec, VNextError,
};
use serde::{Deserialize, Deserializer, Serialize};

/// ViT architecture only. Inputs to a future decoder are already denormalized;
/// `latents_mean` and `latents_std` are not folded into this weight mapping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PH3ImageVaeDecoderConfig {
    latent_channels: u64,
    out_channels: u64,
    patch_size: u64,
    layers: u64,
    heads: u64,
    head_dim: u64,
    register_tokens: u64,
    ffn_multiplier: u64,
    rotary_dim: u64,
    rope_theta: CanonicalRational,
    norm_epsilon: CanonicalRational,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    latent_channels: u64,
    out_channels: u64,
    patch_size: u64,
    layers: u64,
    heads: u64,
    head_dim: u64,
    register_tokens: u64,
    ffn_multiplier: u64,
    rotary_dim: u64,
    rope_theta: CanonicalRational,
    norm_epsilon: CanonicalRational,
}

#[derive(Deserialize)]
struct HfWire {
    #[serde(rename = "_class_name")]
    class_name: String,
    latent_channels: u64,
    out_channels: u64,
    spatial_downsample_factors: Vec<u64>,
    decoder_num_layers: u64,
    decoder_num_attention_heads: u64,
    decoder_attention_head_dim: u64,
    decoder_num_register_tokens: u64,
    decoder_ffn_mult: u64,
    decoder_rope_dim_ratio: f64,
    decoder_rope_theta: serde_json::Number,
    decoder_norm_eps: serde_json::Number,
}

impl<'de> Deserialize<'de> for PH3ImageVaeDecoderConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(deserializer)?;
        let config = Self {
            latent_channels: w.latent_channels,
            out_channels: w.out_channels,
            patch_size: w.patch_size,
            layers: w.layers,
            heads: w.heads,
            head_dim: w.head_dim,
            register_tokens: w.register_tokens,
            ffn_multiplier: w.ffn_multiplier,
            rotary_dim: w.rotary_dim,
            rope_theta: w.rope_theta,
            norm_epsilon: w.norm_epsilon,
        };
        config.validate().map_err(serde::de::Error::custom)?;
        Ok(config)
    }
}

impl PH3ImageVaeDecoderConfig {
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, VNextError> {
        if bytes.is_empty() || bytes.len() > 4 * 1024 * 1024 {
            return Err(invalid(
                "config",
                "requires nonempty VAE metadata of at most 4 MiB",
            ));
        }
        let raw: HfWire =
            serde_json::from_slice(bytes).map_err(|e| invalid("config", e.to_string()))?;
        if raw.class_name != PH3_IMAGE_VAE_CLASS {
            return Err(invalid(
                "_class_name",
                "requires the declared P-H3 image VAE class",
            ));
        }
        let head_dim = raw.decoder_attention_head_dim;
        let ratio = raw.decoder_rope_dim_ratio;
        let rotary = head_dim as f64 * ratio;
        if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 || rotary.fract() != 0.0 {
            return Err(invalid(
                "decoder_rope_dim_ratio",
                "must select an integral number of head channels",
            ));
        }
        let factors = &raw.spatial_downsample_factors;
        if factors.is_empty() {
            return Err(invalid(
                "spatial_downsample_factors",
                "requires a nonempty list of positive spatial factors",
            ));
        }
        let patch_size = factors.iter().try_fold(1u64, |product, &value| {
            (value > 0)
                .then(|| product.checked_mul(value))
                .flatten()
                .ok_or_else(|| {
                    invalid(
                        "spatial_downsample_factors",
                        "invalid factor or compression product overflow",
                    )
                })
        })?;
        let config = Self {
            latent_channels: raw.latent_channels,
            out_channels: raw.out_channels,
            patch_size,
            layers: raw.decoder_num_layers,
            heads: raw.decoder_num_attention_heads,
            head_dim,
            register_tokens: raw.decoder_num_register_tokens,
            ffn_multiplier: raw.decoder_ffn_mult,
            rotary_dim: rotary as u64,
            rope_theta: CanonicalRational::from_decimal_str(&raw.decoder_rope_theta.to_string())?,
            norm_epsilon: CanonicalRational::from_decimal_str(&raw.decoder_norm_eps.to_string())?,
        };
        config.validate()?;
        Ok(config)
    }
    pub const fn latent_channels(&self) -> u64 {
        self.latent_channels
    }
    pub const fn out_channels(&self) -> u64 {
        self.out_channels
    }
    pub const fn patch_size(&self) -> u64 {
        self.patch_size
    }
    pub const fn layers(&self) -> u64 {
        self.layers
    }
    pub const fn heads(&self) -> u64 {
        self.heads
    }
    pub const fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub const fn register_tokens(&self) -> u64 {
        self.register_tokens
    }
    pub const fn ffn_multiplier(&self) -> u64 {
        self.ffn_multiplier
    }
    pub const fn rotary_dim(&self) -> u64 {
        self.rotary_dim
    }
    pub const fn rope_theta(&self) -> CanonicalRational {
        self.rope_theta
    }
    pub const fn norm_epsilon(&self) -> CanonicalRational {
        self.norm_epsilon
    }
    pub fn hidden_size(&self) -> Result<u64, VNextError> {
        self.heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("heads", "decoder width overflows"))
    }
    pub fn intermediate_size(&self) -> Result<u64, VNextError> {
        self.hidden_size()?
            .checked_mul(self.ffn_multiplier)
            .ok_or_else(|| invalid("ffn_multiplier", "decoder intermediate width overflows"))
    }
    pub fn patch_output_channels(&self) -> Result<u64, VNextError> {
        self.out_channels
            .checked_mul(self.patch_size)
            .and_then(|x| x.checked_mul(self.patch_size))
            .ok_or_else(|| invalid("patch_size", "output patch width overflows"))
    }
    /// The suffix consists of learned register tokens and one zero CLS token.
    pub fn sequence_length(
        &self,
        latent_height: u64,
        latent_width: u64,
    ) -> Result<u64, VNextError> {
        if latent_height == 0 || latent_width == 0 {
            return Err(invalid(
                "latent_shape",
                "latent spatial dimensions must be positive",
            ));
        }
        latent_height
            .checked_mul(latent_width)
            .and_then(|x| x.checked_add(self.register_tokens))
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| invalid("latent_shape", "ViT sequence length overflows"))
    }
    pub fn expected_comfy_weights(&self) -> Result<Vec<PH3ImageVaeDecoderWeight>, VNextError> {
        super::weights::expected(self)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if [
            self.latent_channels,
            self.out_channels,
            self.patch_size,
            self.layers,
            self.heads,
            self.head_dim,
            self.register_tokens,
            self.ffn_multiplier,
        ]
        .contains(&0)
            || self.rotary_dim == 0
            || self.rotary_dim > self.head_dim
            || self.rotary_dim % 6 != 0
            || self.rope_theta.numerator() <= 0
            || self.norm_epsilon.numerator() <= 0
            || self.norm_epsilon.numerator() as u64 > self.norm_epsilon.denominator()
        {
            return Err(invalid("architecture", "requires positive dimensions, positive epsilon <=1 and a three-axis even rotary prefix within the head"));
        }
        // Metadata-only inventory bound, independent of GPU capacity.
        if self
            .layers
            .checked_mul(20)
            .and_then(|x| x.checked_add(9))
            .is_none_or(|x| x > 65536)
        {
            return Err(invalid(
                "layers",
                "decoder source tensor inventory exceeds 65536",
            ));
        }
        let h = self.hidden_size()?;
        let m = self.intermediate_size()?;
        for shape in [
            vec![1, self.register_tokens, h],
            vec![h, self.latent_channels],
            vec![
                h.checked_mul(3)
                    .ok_or_else(|| invalid("heads", "QKV width overflows"))?,
                h,
            ],
            vec![
                m.checked_mul(2)
                    .ok_or_else(|| invalid("ffn_multiplier", "packed gate/up width overflows"))?,
                h,
            ],
            vec![h, m],
            vec![self.patch_output_channels()?, h],
        ] {
            ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous)?;
        }
        Ok(())
    }
}
