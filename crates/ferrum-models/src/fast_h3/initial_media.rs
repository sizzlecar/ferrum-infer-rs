//! Ordered Ref2VA references and target preparation under the existing Host
//! authority. Encoders supply normalized latents; their independent posterior
//! sampling finishes before the request's visual-condition/video/audio RNG.
use super::{FastH3PackedLayout, FastH3Reference, FastH3SegmentKind, FastH3VideoGeometry};
use crate::normal_noise::cpu_mt19937::{fill_noise_sequence, SCRATCH_BYTES};
use anyhow::{ensure, Context, Result};
use ferrum_diffusion::SpatiotemporalLatentShape;
use ferrum_runtime::*;

pub const FAST_H3_INITIAL_MEDIA_PROFILE: &str =
    "fasth3.ref2va.cpu_mt19937.scalar_f32_libm.condition_then_video_then_audio.patch_c_major.condition_0p999_separate_mul_add.v1";

/// Borrowed normalized encoder outputs in original reference order. Video
/// latents are NCtHW; audio latents are [stereo channel, latent channel, time].
/// This describes data, not an encoder completion or a new allocation owner.
#[derive(Clone, Copy)]
pub enum FastH3ReferenceLatents<'a> {
    Image(&'a ManagedHostTensor),
    Video {
        video: &'a ManagedHostTensor,
        audio: Option<&'a ManagedHostTensor>,
    },
    Audio(&'a ManagedHostTensor),
}

fn validate_storage(value: &ManagedHostTensor, host: &HostMemoryAccount) -> Result<()> {
    ensure!(
        value.tensor().element_type() == ElementType::F32
            && value.tensor().layout() == &ResolvedTensorLayout::Contiguous
            && host.owns(value.bytes())
            && value.bytes().len() as u64 == value.tensor().minimum_storage_bytes()?
            && value
                .bytes()
                .bytes()
                .chunks_exact(4)
                .all(|b| read(b).is_finite()),
        "reference requires finite exact contiguous F32 storage in this Host account"
    );
    Ok(())
}

fn visual_shape(
    value: &ManagedHostTensor,
    host: &HostMemoryAccount,
    image: bool,
) -> Result<SpatiotemporalLatentShape> {
    validate_storage(value, host)?;
    let dims: [u64; 5] = value.tensor().dimensions().try_into()?;
    ensure!(
        dims[..2] == [1, 24] && (!image || dims[2] == 1),
        "visual reference requires [1,24,T,H,W], with T=1 for an image"
    );
    Ok(SpatiotemporalLatentShape::new(
        [
            1,
            24,
            dims[2].try_into()?,
            dims[3].try_into()?,
            dims[4].try_into()?,
        ],
        [1, 2, 2],
    )?)
}

fn audio_samples(value: &ManagedHostTensor, host: &HostMemoryAccount) -> Result<usize> {
    validate_storage(value, host)?;
    let dims: [u64; 3] = value.tensor().dimensions().try_into()?;
    ensure!(
        dims[..2] == [2, 32] && dims[2] > 0,
        "audio reference requires normalized F32 [2,32,T]"
    );
    Ok(dims[2].try_into()?)
}

/// One explicit image reference and its request's unit target noise. Returns
/// `[video, audio]` in the existing packed stream contract. Only the continuation
/// applies initial target sigma; references receive their original .999 clean
/// conditioning augmentation here and remain frozen afterwards.
pub fn prepare_fast_h3_keyframe_media(
    seed: u64,
    target: &FastH3VideoGeometry,
    layout: &FastH3PackedLayout,
    normalized_reference: &ManagedHostTensor,
    output_ids: [ProgramValueId; 2],
    host: &HostMemoryAccount,
) -> Result<[ManagedHostTensor; 2]> {
    prepare_fast_h3_reference_media(
        seed,
        target,
        layout,
        &[FastH3ReferenceLatents::Image(normalized_reference)],
        output_ids,
        host,
    )
}

/// Preserve each visual reference's independently shaped RNG draw, then draw
/// target video and target audio. Reference audio is copied without noise
/// augmentation. All layout, owner and byte checks precede admitted buffers.
pub fn prepare_fast_h3_reference_media(
    seed: u64,
    target: &FastH3VideoGeometry,
    layout: &FastH3PackedLayout,
    references: &[FastH3ReferenceLatents<'_>],
    output_ids: [ProgramValueId; 2],
    host: &HostMemoryAccount,
) -> Result<[ManagedHostTensor; 2]> {
    ensure!(
        !references.is_empty() && output_ids[0] != output_ids[1],
        "references must be nonempty and output IDs distinct"
    );
    let mut geometry = Vec::with_capacity(references.len());
    let mut visual = Vec::new();
    let mut audio_references = Vec::new();
    for reference in references {
        geometry.push(match *reference {
            FastH3ReferenceLatents::Image(value) => {
                let shape = visual_shape(value, host, true)?;
                visual.push((shape, value));
                let [_, _, _, latent_height, latent_width] = shape.ncthw_dims();
                FastH3Reference::Image {
                    latent_height,
                    latent_width,
                }
            }
            FastH3ReferenceLatents::Audio(value) => {
                let count = audio_samples(value, host)?;
                audio_references.push((count, value));
                FastH3Reference::Audio {
                    latents_per_channel: count,
                }
            }
            FastH3ReferenceLatents::Video { video, audio } => {
                let shape = visual_shape(video, host, false)?;
                visual.push((shape, video));
                let [_, _, latent_frames, latent_height, latent_width] = shape.ncthw_dims();
                let audio_latents_per_channel = audio
                    .map(|value| {
                        let count = audio_samples(value, host)?;
                        audio_references.push((count, value));
                        Ok::<_, anyhow::Error>(count)
                    })
                    .transpose()?;
                FastH3Reference::Video {
                    latent_frames,
                    latent_height,
                    latent_width,
                    audio_latents_per_channel,
                }
            }
        });
    }
    ensure!(!visual.is_empty(), "Ref2VA requires a visual reference");
    let text = layout
        .segments()
        .first()
        .filter(|s| s.kind == FastH3SegmentKind::Text)
        .ok_or_else(|| anyhow::anyhow!("Ref2VA requires the declared text prefix"))?;
    let expected = FastH3PackedLayout::ref2va(
        text.packed_rows.len(),
        &geometry,
        target,
        layout.sequence_rows(),
    )?;
    ensure!(
        layout.segments() == expected.segments()
            && layout.target_video_rows() == expected.target_video_rows()
            && layout.target_audio_rows() == expected.target_audio_rows(),
        "ordered reference geometry and complete packed layout differ"
    );
    let video_shape = target.latent_shape();
    let video_spec = ResolvedTensorSpec::new(
        vec![layout.target_video_rows().end as u64, 96],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )?;
    let audio_spec = ResolvedTensorSpec::new(
        vec![layout.target_audio_rows().end as u64, 32],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )?;
    let mut claims = visual
        .iter()
        .map(|(shape, _)| Ok((HostMemoryClass::InputPreprocess, shape.byte_len::<f32>()?)))
        .collect::<Result<Vec<_>>>()?;
    claims.extend([
        (
            HostMemoryClass::InputPreprocess,
            video_shape.byte_len::<f32>()?,
        ),
        (
            HostMemoryClass::InputPreprocess,
            usize::try_from(audio_spec.minimum_storage_bytes()?)?,
        ),
        (
            HostMemoryClass::InputPreprocess,
            usize::try_from(video_spec.minimum_storage_bytes()?)?,
        ),
        (HostMemoryClass::InputPreprocess, SCRATCH_BYTES),
    ]);
    let mut grants = host.reserve_many(&claims)?.into_iter();
    let mut reference_noise = grants
        .by_ref()
        .take(visual.len())
        .map(|grant| grant.allocate_zeroed())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut video_noise = grants.next().unwrap().allocate_zeroed()?;
    let mut audio = grants.next().unwrap().allocate_zeroed()?;
    let mut video = grants.next().unwrap().allocate_zeroed()?;
    let mut scratch = grants.next().unwrap().allocate_zeroed()?;
    let audio_prefix_bytes = layout
        .target_audio_rows()
        .start
        .checked_mul(32 * 4)
        .context("audio prefix byte overflow")?;
    {
        let mut draws = reference_noise
            .iter_mut()
            .map(|value| value.as_mut())
            .collect::<Vec<_>>();
        draws.push(video_noise.as_mut());
        draws.push(&mut audio.as_mut()[audio_prefix_bytes..]);
        fill_noise_sequence(seed, &mut draws, scratch.as_mut())?;
    }
    let clean = 0.999_f32;
    let noise = 1.0_f32 - clean;
    let mut offset = 0usize;
    for ((shape, value), reference_noise) in visual.iter().zip(&reference_noise) {
        shape.for_each_packed_index(|packed, dense| {
            let a = clean * read(&value.bytes().bytes()[dense * 4..dense * 4 + 4]);
            let b = noise * read(&reference_noise.as_ref()[dense * 4..dense * 4 + 4]);
            video.as_mut()[offset + packed * 4..offset + packed * 4 + 4]
                .copy_from_slice(&(a + b).to_le_bytes());
        });
        offset = offset
            .checked_add(shape.byte_len::<f32>()?)
            .context("visual reference offset overflow")?;
    }
    video_shape.for_each_packed_index(|packed, dense| {
        video.as_mut()[offset + packed * 4..offset + packed * 4 + 4]
            .copy_from_slice(&video_noise.as_ref()[dense * 4..dense * 4 + 4]);
    });
    let mut audio_row = 0usize;
    for (samples, value) in audio_references {
        // Original encoder emits channel-major temporal latents. Packed audio
        // keeps the stereo channels separate, with 32 features per time row.
        for stereo in 0..2 {
            for t in 0..samples {
                for channel in 0..32 {
                    let source = ((stereo * 32 + channel) * samples + t) * 4;
                    let destination = ((audio_row + stereo * samples + t) * 32 + channel) * 4;
                    audio.as_mut()[destination..destination + 4]
                        .copy_from_slice(&value.bytes().bytes()[source..source + 4]);
                }
            }
        }
        audio_row = audio_row
            .checked_add(samples.checked_mul(2).context("stereo row overflow")?)
            .context("audio reference offset overflow")?;
    }
    ensure!(
        video
            .as_ref()
            .chunks_exact(4)
            .chain(audio.as_ref().chunks_exact(4))
            .all(|b| read(b).is_finite()),
        "initial media overflowed F32"
    );
    let [video_id, audio_id] = output_ids;
    Ok([
        ManagedHostTensor::new(video_id, video_spec, video.freeze())?,
        ManagedHostTensor::new(audio_id, audio_spec, audio.freeze())?,
    ])
}

fn read(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().unwrap())
}

#[cfg(test)]
mod tests;
