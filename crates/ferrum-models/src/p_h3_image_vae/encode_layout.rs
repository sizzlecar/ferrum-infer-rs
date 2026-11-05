//! Explicit single-frame pixel and posterior domains. Sampling consumes caller
//! supplied F32 noise; this is not a claim of Torch CPU generator equivalence.
use super::latent_statistics::PH3ImageVaeLatentStatistics;
use anyhow::{ensure, Context, Result};
use ferrum_runtime::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageVaePixelDomain {
    /// The wrapper maps [0,1] to [-1,1] before the encoder normalizes it again.
    UnitRgb,
    SignedRgb,
}
pub(crate) const PIXEL_PROFILE_ID: &str =
    "p_h3_image.vae.encoder_pixels.f32.separate_wrapper_and_imagenet.v1";
pub(crate) const POSTERIOR_PROFILE_ID: &str =
    "p_h3_image.vae.posterior.provided_f32_noise.libm-0.2.15.exp_f32.fma.v1";

/// Preserve the wrapper's separate F32 operations, including the [0,1] to
/// [-1,1] round trip. No implicit resizing, clipping or BF16 promotion occurs.
pub(crate) fn normalized_pixels(
    host: &HostMemoryAccount,
    input: &ManagedHostTensor,
    domain: PH3ImageVaePixelDomain,
    output_id: ProgramValueId,
) -> Result<ManagedHostTensor> {
    let [batch, channels, height, width] = dimensions(input.tensor())?;
    ensure!(
        channels == 3 && host.owns(input.bytes()),
        "encoder requires original Host-owned F32 RGB"
    );
    let pixels = height
        .checked_mul(width)
        .context("pixel extent overflows")?;
    let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, input.bytes().len())?;
    for b in 0..batch {
        for c in 0..3 {
            for p in 0..pixels {
                let offset = ((b * 3 + c) * pixels + p) * 4;
                let value = read(input.bytes().bytes(), offset);
                let normalized = normalized_pixel(value, c, domain)?;
                bytes.as_mut()[offset..offset + 4].copy_from_slice(&normalized.to_le_bytes());
            }
        }
    }
    Ok(ManagedHostTensor::new(
        output_id,
        input.tensor().clone(),
        bytes.freeze(),
    )?)
}

/// Shared scalar boundary for direct tile extraction. Keep every original
/// F32 operation, especially the UnitRgb wrapper's lossy signed round trip.
/// This performs no allocation or implicit clipping/resizing.
pub(super) fn normalized_pixel(
    value: f32,
    channel: usize,
    domain: PH3ImageVaePixelDomain,
) -> Result<f32> {
    let (minimum, maximum) = match domain {
        PH3ImageVaePixelDomain::UnitRgb => (0.0, 1.0),
        PH3ImageVaePixelDomain::SignedRgb => (-1.0, 1.0),
    };
    ensure!(
        channel < 3 && value.is_finite() && (minimum..=maximum).contains(&value),
        "pixel or channel is outside its declared RGB domain"
    );
    let signed = match domain {
        PH3ImageVaePixelDomain::UnitRgb => {
            let scaled = value * 2.0;
            scaled - 1.0
        }
        PH3ImageVaePixelDomain::SignedRgb => value,
    };
    let shifted = signed + 1.0;
    let unit = shifted * 0.5;
    let centered = unit - [0.485_f32, 0.456, 0.406][channel];
    Ok(centered / [0.229_f32, 0.224, 0.225][channel])
}

/// Output moments are NCHW with [mean channels, log-variance channels] within
/// each request. Complete all per-image tile blending before calling this.
/// Noise must already have the caller's declared per-request RNG identity.
pub(crate) fn posterior_from_noise(
    host: &HostMemoryAccount,
    moments_spec: &ResolvedTensorSpec,
    moments: &ManagedHostBytes,
    noise: &ManagedHostTensor,
    statistics: &PH3ImageVaeLatentStatistics<'_>,
    output_id: ProgramValueId,
) -> Result<ManagedHostTensor> {
    let [batch, channels, height, width] = dimensions(noise.tensor())?;
    ensure!(
        channels == statistics.channels(),
        "posterior statistics differ from noise channels"
    );
    let moment_channels = channels
        .checked_mul(2)
        .context("moment channel extent overflows")?;
    ensure!(
        dimensions(moments_spec)? == [batch, moment_channels, height, width]
            && u64::try_from(moments.len())? == moments_spec.minimum_storage_bytes()?
            && moments.class() == HostMemoryClass::Output
            && host.owns(moments)
            && host.owns(noise.bytes()),
        "posterior requires exact original F32 moments and noise on the same Host parent"
    );
    let positions = height
        .checked_mul(width)
        .context("posterior extent overflows")?;
    let mut output = host.allocate_zeroed(HostMemoryClass::InputPreprocess, noise.bytes().len())?;
    for b in 0..batch {
        for c in 0..channels {
            for p in 0..positions {
                let offset = ((b * channels + c) * positions + p) * 4;
                let mean = read(
                    moments.bytes(),
                    ((b * moment_channels + c) * positions + p) * 4,
                );
                let logvar = read(
                    moments.bytes(),
                    ((b * moment_channels + channels + c) * positions + p) * 4,
                );
                let noise = read(noise.bytes().bytes(), offset);
                ensure!(
                    mean.is_finite() && logvar.is_finite() && noise.is_finite(),
                    "nonfinite posterior moment or noise"
                );
                let exponent = 0.5_f32 * logvar.clamp(-30.0, 20.0);
                let scale = libm::expf(exponent);
                let sample = scale.mul_add(noise, mean);
                let normalized = statistics.normalize(c, sample)?;
                output.as_mut()[offset..offset + 4].copy_from_slice(&normalized.to_le_bytes());
            }
        }
    }
    Ok(ManagedHostTensor::new(
        output_id,
        noise.tensor().clone(),
        output.freeze(),
    )?)
}
fn dimensions(spec: &ResolvedTensorSpec) -> Result<[usize; 4]> {
    let [b, c, h, w] = spec.dimensions() else {
        anyhow::bail!("VAE image boundary requires NCHW");
    };
    ensure!(
        spec.element_type() == ElementType::F32
            && spec.layout() == &ResolvedTensorLayout::Contiguous
            && [b, c, h, w].into_iter().all(|&n| n > 0)
            && spec.minimum_storage_bytes()? <= isize::MAX as u64,
        "VAE image boundary requires bounded nonempty contiguous F32"
    );
    Ok([
        usize::try_from(*b)?,
        usize::try_from(*c)?,
        usize::try_from(*h)?,
        usize::try_from(*w)?,
    ])
}
fn read(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[cfg(test)]
mod tests;
