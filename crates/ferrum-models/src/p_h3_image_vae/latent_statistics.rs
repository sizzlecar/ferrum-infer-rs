//! Borrow original F32 normalization parameters from the VAE checkpoint.
use super::PH3ImageVaeDecoderConfig;
use anyhow::{ensure, Result};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;

pub(crate) struct PH3ImageVaeLatentStatistics<'a> {
    source: &'a ComfyCanonicalWeightSource,
    mean: &'a [u8],
    std: &'a [u8],
}
impl<'a> PH3ImageVaeLatentStatistics<'a> {
    pub const PROFILE_ID: &'static str =
        "p_h3_image.vae.latent_denormalize.f32.separate_mul_add.v1";

    pub fn new(
        config: &PH3ImageVaeDecoderConfig,
        source: &'a ComfyCanonicalWeightSource,
    ) -> Result<Self> {
        config.validate()?;
        let read = |name: &str, positive: bool| -> Result<&'a [u8]> {
            let tensor = source.archive().tensor(name)?;
            ensure!(
                tensor.dtype() == safetensors::Dtype::F32
                    && tensor.shape() == [config.latent_channels()],
                "VAE {name} requires its original F32[channel] tensor"
            );
            ensure!(
                tensor.bytes().chunks_exact(4).all(|bytes| {
                    let value = f32::from_le_bytes(bytes.try_into().unwrap());
                    value.is_finite() && (!positive || value > 0.0)
                }),
                "VAE {name} contains invalid channel statistics"
            );
            Ok(tensor.bytes())
        };
        Ok(Self {
            source,
            mean: read("latents_mean", false)?,
            std: read("latents_std", true)?,
        })
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
    pub fn channels(&self) -> usize {
        self.mean.len() / 4
    }
    pub(super) fn normalize(&self, channel: usize, sample: f32) -> Result<f32> {
        ensure!(
            channel < self.channels() && sample.is_finite(),
            "VAE posterior has invalid channel or nonfinite sample"
        );
        let read = |bytes: &[u8]| {
            f32::from_le_bytes(bytes[channel * 4..channel * 4 + 4].try_into().unwrap())
        };
        let centered = sample - read(self.mean);
        let result = centered / read(self.std);
        ensure!(result.is_finite(), "VAE latent normalization overflowed");
        Ok(result)
    }
    pub(super) fn denormalize(&self, channel: usize, sample: f32) -> Result<f32> {
        ensure!(
            channel < self.channels() && sample.is_finite(),
            "VAE normalized latent has invalid channel or nonfinite sample"
        );
        let read = |bytes: &[u8]| {
            f32::from_le_bytes(bytes[channel * 4..channel * 4 + 4].try_into().unwrap())
        };
        // The pinned wrapper executes two separate tensor operations. Do not
        // contract these or round either operand through BF16.
        let scaled = sample * read(self.std);
        let result = scaled + read(self.mean);
        ensure!(result.is_finite(), "VAE latent denormalization overflowed");
        Ok(result)
    }
}

#[cfg(test)]
pub(super) mod tests;
