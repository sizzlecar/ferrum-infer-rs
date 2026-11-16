use super::*;
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct FastH3AudioVaeConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    pub(super) encoder_dim: u64,
    pub(super) encoder_rates: Vec<u64>,
    pub(super) num_attention_heads: u64,
    pub(super) latent_dim: u64,
    pub(super) latent_channels: u64,
    pub(super) decoder_dim: u64,
    pub(super) decoder_rates: Vec<u64>,
    pub(super) decoder_kernel_sizes: Vec<u64>,
    pub(super) resblock_kernel_sizes: Vec<u64>,
    pub(super) resblock_dilation_sizes: Vec<Vec<u64>>,
    pub(super) sampling_rate: u32,
    pub(super) latents_mean: Vec<f32>,
    pub(super) latents_std: Vec<f32>,
}
#[derive(Clone, Copy, Debug)]
pub struct FastH3AudioComponentLimits {
    pub maximum_metadata_bytes: usize,
    pub maximum_source_bytes: u64,
    pub maximum_stored_bytes: u64,
    pub maximum_plan_nodes: usize,
    pub maximum_activation_bytes: u64,
}
/// Decoder callers retain their original public name and limits.
pub type FastH3AudioDecoderLimits = FastH3AudioComponentLimits;
impl FastH3AudioVaeConfig {
    pub fn parse(bytes: &[u8], maximum_metadata_bytes: usize) -> Result<Self> {
        ensure!(
            maximum_metadata_bytes > 0 && bytes.len() <= maximum_metadata_bytes,
            "audio metadata capacity exceeded"
        );
        let c: Self = serde_json::from_slice(bytes)?;
        c.validate()?;
        Ok(c)
    }
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.class_name == "AutoencoderKLMiniMaxH3Audio",
            "wrong original audio VAE class"
        );
        ensure!(
            self.latent_dim > 0
                && self.latent_channels > 0
                && self.decoder_dim > 0
                && self.encoder_dim > 0
                && self.num_attention_heads > 0
                && self.latent_dim % self.num_attention_heads == 0
                && self.latent_dim % self.latent_channels == 0
                && self.sampling_rate > 0,
            "audio dimensions must be positive"
        );
        ensure!(
            !self.decoder_rates.is_empty()
                && self.decoder_rates.len() < 64
                && self.decoder_rates.len() == self.decoder_kernel_sizes.len(),
            "invalid audio upsample stages"
        );
        let divisor = 1u64
            .checked_shl(self.decoder_rates.len() as u32)
            .context("audio channel divisor overflow")?;
        ensure!(
            self.decoder_dim % divisor == 0,
            "decoder channels must halve exactly per stage"
        );
        for (&s, &k) in self.decoder_rates.iter().zip(&self.decoder_kernel_sizes) {
            ensure!(
                s > 0 && k >= s && (k - s) % 2 == 0,
                "audio transposed convolution does not preserve exact rate"
            );
        }
        ensure!(
            !self.resblock_kernel_sizes.is_empty()
                && self.resblock_kernel_sizes.len() == self.resblock_dilation_sizes.len(),
            "audio residual branch counts differ"
        );
        for (&k, d) in self
            .resblock_kernel_sizes
            .iter()
            .zip(&self.resblock_dilation_sizes)
        {
            ensure!(
                k > 0 && k % 2 == 1 && !d.is_empty() && d.iter().all(|&n| n > 0),
                "invalid audio residual kernel/dilations"
            );
        }
        ensure!(
            self.latents_mean.len() as u64 == self.latent_channels
                && self.latents_std.len() as u64 == self.latent_channels
                && self.latents_mean.iter().all(|x| x.is_finite())
                && self.latents_std.iter().all(|x| x.is_finite() && *x > 0.),
            "audio latent normalization differs"
        );
        ensure!(
            !self.encoder_rates.is_empty()
                && self.encoder_rates.len() < 64
                && self
                    .encoder_rates
                    .iter()
                    .all(|&rate| rate > 0 && rate <= i32::MAX as u64 / 2),
            "invalid original audio encoder strides"
        );
        self.encoder_dim
            .checked_mul(1u64 << self.encoder_rates.len())
            .context("audio encoder channel extent overflow")?;
        let encoder_hop = self
            .encoder_rates
            .iter()
            .try_fold(1u64, |n, &rate| n.checked_mul(rate))
            .context("audio encoder hop overflow")?;
        ensure!(
            encoder_hop == self.hop_length()?,
            "audio encoder and decoder rates disagree"
        );
        Ok(())
    }
    pub fn latent_channels(&self) -> u64 {
        self.latent_channels
    }
    pub fn encoder_feature_channels(&self) -> u64 {
        self.latent_dim
    }
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }
    pub fn hop_length(&self) -> Result<u64> {
        self.decoder_rates
            .iter()
            .try_fold(1u64, |n, &r| n.checked_mul(r))
            .context("audio hop overflow")
    }
    pub fn latent_mean(&self) -> &[f32] {
        &self.latents_mean
    }
    pub fn latent_std(&self) -> &[f32] {
        &self.latents_std
    }
    /// Right-zero padding is part of the original encoder, before any strided
    /// convolution. Each latent row represents exactly one complete hop.
    pub fn encoder_padded_samples(&self, samples: u64) -> Result<u64> {
        ensure!(samples > 0, "audio reference must contain samples");
        let hop = self.hop_length()?;
        samples
            .div_ceil(hop)
            .checked_mul(hop)
            .context("audio right padding overflow")
    }
}
impl DecoderConfig {
    pub(super) fn validate(&self, limits: FastH3AudioDecoderLimits) -> Result<(), VNextError> {
        self.architecture.validate().map_err(invalid)?;
        if limits.maximum_metadata_bytes == 0
            || limits.maximum_plan_nodes == 0
            || limits.maximum_source_bytes == 0
            || limits.maximum_stored_bytes == 0
            || limits.maximum_activation_bytes == 0
            || self.latent_samples == 0
        {
            return Err(invalid("positive audio geometry/capacities required"));
        }
        self.batch()?;
        validate_inventory(
            &self.weights,
            source::entries(&self.architecture).map_err(invalid)?,
            limits,
        )?;
        self.schema()?.validate(&id(family::Family::FAMILY_ID)?)?;
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("audio metadata capacity exceeded"));
        }
        Ok(())
    }
}
pub(super) fn validate_inventory(
    weights: &[AudioWeight],
    entries: Vec<source::Entry>,
    limits: FastH3AudioComponentLimits,
) -> Result<(), VNextError> {
    if limits.maximum_metadata_bytes == 0
        || limits.maximum_plan_nodes == 0
        || limits.maximum_source_bytes == 0
        || limits.maximum_stored_bytes == 0
        || limits.maximum_activation_bytes == 0
    {
        return Err(invalid("positive audio capacities required"));
    }
    if weights.len() != entries.len() {
        return Err(invalid("audio weight inventory count differs"));
    }
    for (w, e) in weights.iter().zip(entries) {
        if w.name != e.name
            || w.component.dimensions != e.physical
            || w.component.external_names != e.names()
            || w.component.role != WeightComponentRole::Values
            || !w.component.required
            || w.component.encoding
                != (WeightEncoding::Dense {
                    element_type: ElementType::F32,
                })
        {
            return Err(invalid("audio canonical checkpoint slot differs"));
        }
    }
    let bytes = weights.iter().try_fold(0u64, |n, w| {
        n.checked_add(w.component.physical_bytes()?)
            .ok_or_else(|| invalid("audio stored extent overflow"))
    })?;
    if bytes > limits.maximum_stored_bytes {
        return Err(invalid("audio stored capacity exceeded"));
    }
    Ok(())
}
