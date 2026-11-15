//! T1 posterior sampling after the original causal CNN. Each batch item is an
//! independent reference: fresh CPU seed 42, F32 sample, F16 round trip, then
//! the checkpoint's F32 channel normalization. This explicit scalar-libm RNG
//! profile is not a claim of bitwise equality with Torch's AVX2 normal transform.
use super::FastH3VideoVaeConfig;
use crate::normal_noise::cpu_mt19937::{fill_normal_noise, SCRATCH_BYTES};
use anyhow::{ensure, Context, Result};
use ferrum_runtime::*;
use half::f16;

pub const FAST_H3_KEYFRAME_POSTERIOR_PROFILE: &str =
    "fasth3.keyframe.cpu_mt19937_seed42.scalar_f32_libm.separate_mul_add.f16_roundtrip.normalize_f32.v1";
pub const FAST_H3_VIDEO_POSTERIOR_PROFILE: &str =
    "fasth3.video.cpu_mt19937_seed42.scalar_f32_libm.full_ncthw.separate_mul_add.f16_roundtrip.normalize_f32.v1";

/// Sample normalized F32 NC1HW latents from F32 N(2C)HW posterior moments.
/// Both RNG scratch and output use the caller's existing Host account. No
/// request-global RNG state is retained, and batched references match B=1 calls.
pub fn sample_fast_h3_keyframe_posterior(
    config: &FastH3VideoVaeConfig,
    tensor: &ResolvedTensorSpec,
    moments: &ManagedHostBytes,
    output_id: ProgramValueId,
    host: &HostMemoryAccount,
) -> Result<ManagedHostTensor> {
    let [batch, twice_channels, height, width]: [u64; 4] = tensor
        .dimensions()
        .try_into()
        .map_err(|_| anyhow::anyhow!("keyframe posterior requires N(2C)HW moments"))?;
    let volume = ResolvedTensorSpec::new(
        vec![batch, twice_channels, 1, height, width],
        tensor.element_type(),
        tensor.layout().clone(),
    )?;
    sample_fast_h3_video_posterior(config, &volume, moments, output_id, host)
}

/// Sample the complete reference posterior after clip concatenation, suffix
/// removal and spatial stitching. RNG is reset per reference, never per clip.
pub fn sample_fast_h3_video_posterior(
    config: &FastH3VideoVaeConfig,
    tensor: &ResolvedTensorSpec,
    moments: &ManagedHostBytes,
    output_id: ProgramValueId,
    host: &HostMemoryAccount,
) -> Result<ManagedHostTensor> {
    config.validate()?;
    let [batch, twice_channels, frames, height, width]: [u64; 5] =
        tensor
            .dimensions()
            .try_into()
            .map_err(|_| anyhow::anyhow!("visual posterior requires N(2C)THW moments"))?;
    let channels = u64::try_from(config.latent_channels())?;
    ensure!(
        tensor.element_type() == ElementType::F32
            && tensor.layout() == &ResolvedTensorLayout::Contiguous
            && batch > 0
            && frames > 0
            && height > 0
            && width > 0
            && channels.checked_mul(2) == Some(twice_channels),
        "visual posterior shape, dtype or channel count differs"
    );
    let spatial = usize::try_from(
        height
            .checked_mul(width)
            .and_then(|v| v.checked_mul(frames))
            .context("posterior spatial overflow")?,
    )?;
    let per_reference = spatial
        .checked_mul(config.latent_channels())
        .context("posterior channel overflow")?;
    let per_reference_bytes = per_reference
        .checked_mul(4)
        .context("posterior byte overflow")?;
    ensure!(
        per_reference >= 16,
        "scalar CPU posterior requires at least 16 samples per reference"
    );
    let spec = ResolvedTensorSpec::new(
        vec![batch, channels, frames, height, width],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )?;
    let bytes = usize::try_from(spec.minimum_storage_bytes()?)?;
    ensure!(
        moments.len() == bytes.checked_mul(2).context("moment byte overflow")?
            && moments.bytes().chunks_exact(4).all(|b| read(b).is_finite()),
        "posterior moments require exact finite F32 storage"
    );
    let mut grants = host
        .reserve_many(&[
            (HostMemoryClass::InputPreprocess, bytes),
            (HostMemoryClass::InputPreprocess, SCRATCH_BYTES),
        ])?
        .into_iter();
    let mut output = grants.next().unwrap().allocate_zeroed()?;
    let mut scratch = grants.next().unwrap().allocate_zeroed()?;
    for (n, reference) in output
        .as_mut()
        .chunks_exact_mut(per_reference_bytes)
        .enumerate()
    {
        fill_normal_noise(42, reference, scratch.as_mut())?;
        let start = n * per_reference_bytes * 2;
        sample_and_normalize(
            config,
            spatial,
            &moments.bytes()[start..start + per_reference_bytes * 2],
            reference,
        )?;
    }
    Ok(ManagedHostTensor::new(output_id, spec, output.freeze())?)
}

fn sample_and_normalize(
    config: &FastH3VideoVaeConfig,
    spatial: usize,
    moments: &[u8],
    noise_output: &mut [u8],
) -> Result<()> {
    let sample_bytes = noise_output.len();
    for (i, output) in noise_output.chunks_exact_mut(4).enumerate() {
        let mean = read(&moments[i * 4..i * 4 + 4]);
        let logvar = read(&moments[sample_bytes + i * 4..sample_bytes + i * 4 + 4]);
        let deviation = libm::expf(0.5f32 * logvar.clamp(-30., 20.));
        let scaled_noise = deviation * read(output);
        let sample = mean + scaled_noise;
        let rounded = f16::from_f32(sample).to_f32();
        let channel = i / spatial;
        let centered = rounded - config.latent_mean()[channel] as f32;
        let normalized = centered / config.latent_std()[channel] as f32;
        ensure!(
            normalized.is_finite(),
            "keyframe posterior sample or normalization overflowed"
        );
        output.copy_from_slice(&normalized.to_le_bytes());
    }
    Ok(())
}
fn read(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().unwrap())
}

#[cfg(test)]
mod tests;
