//! Ordered geometry and simultaneous Host payloads, before request admission.
//! PNG headers are only scheduling hints; execution still uses the strict
//! decoder and checks the resulting dimensions against these original plans.
use super::*;
use anyhow::{ensure, Context};
use ferrum_models::{
    fast_h3::{FastH3ReferenceImagePlan, FAST_H3_REFERENCE_IMAGE_SHORT_EDGE},
    qwen3_vl_vision::Qwen3VLVisionImagePlan,
    qwen_image21::image_input::png_dimensions_hint,
};

pub(crate) struct ImageConditioningPlan {
    pub reference: FastH3ReferenceImagePlan,
    pub vision: Qwen3VLVisionImagePlan,
}
pub(crate) struct ConditioningPlan {
    pub references: Vec<ReferenceConditioningPlan>,
    pub presentation: FastH3MediaPresentation,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub weight_bytes: usize,
}
pub(crate) enum ReferenceConditioningPlan {
    Image(ImageConditioningPlan),
    Audio(AudioConditioningPlan),
    Video(super::video::VideoConditioningPlan),
}
pub(crate) struct AudioConditioningPlan {
    pub wave: WaveAudioPlan,
    pub maximum_samples: usize,
}

pub(super) fn add(a: usize, b: usize) -> anyhow::Result<usize> {
    a.checked_add(b)
        .context("conditioning Host backing overflow")
}
pub(super) fn payload(
    host: &HostMemoryAccount,
    dimensions: &[u64],
    element: ElementType,
) -> anyhow::Result<usize> {
    let spec = ResolvedTensorSpec::new(
        dimensions.to_vec(),
        element,
        ResolvedTensorLayout::Contiguous,
    )?;
    Ok(host.backing_bytes(usize::try_from(spec.minimum_storage_bytes()?)?)?)
}

pub(super) fn plan(
    assets: &FastH3Assets,
    video_input: &ferrum_models::video_input::VideoInputCodec,
    limits: ConditioningLimits,
    request: &VideoGenerationRequest,
    host: &HostMemoryAccount,
    input_floor: usize,
    output_floor: usize,
    weight_floor: usize,
) -> anyhow::Result<ConditioningPlan> {
    assets.validate_request(request)?;
    let mut references = Vec::with_capacity(request.references.len());
    let mut retained_latents = 0usize;
    let mut retained_visual = 0usize;
    let mut input_bytes = input_floor;
    let mut output_bytes = output_floor;
    let mut weight_bytes = weight_floor;
    let compression = u64::try_from(assets.video_vae.spatial_compression_ratio()?)?;
    for reference in &request.references {
        let encoded = match reference {
            VideoReference::Image(encoded) => encoded,
            VideoReference::Audio(encoded) => {
                let wave = WaveAudioPlan::parse(encoded.clone(), limits.audio)?;
                ensure!(
                    wave.sampling_rate() == assets.audio_vae.sampling_rate(),
                    "audio references currently require WAVE at the model's {} Hz sample rate",
                    assets.audio_vae.sampling_rate()
                );
                let maximum_samples = u64::from(request.parameters.num_frames)
                    .checked_mul(u64::from(request.parameters.frame_rate.denominator))
                    .and_then(|n| n.checked_mul(u64::from(wave.sampling_rate())))
                    .context("audio reference trim duration overflow")?
                    .checked_div(u64::from(request.parameters.frame_rate.numerator))
                    .context("positive reference frame rate required")?;
                let maximum_samples = usize::try_from(maximum_samples)?;
                wave.validate_samples(maximum_samples)?;
                let samples = wave.stereo_spec(maximum_samples)?.dimensions()[2];
                let padded = assets.audio_vae.encoder_padded_samples(samples)?;
                let time = padded / assets.audio_vae.hop_length()?;
                let waveform = wave.stereo_backing_bytes(maximum_samples, host)?;
                let padded_waveform = payload(host, &[2, 1, 1, padded], ElementType::F32)?;
                let hidden = payload(
                    host,
                    &[2, assets.audio_vae.encoder_feature_channels(), 1, time],
                    ElementType::F32,
                )?;
                let mode = payload(
                    host,
                    &[2, assets.audio_vae.latent_channels(), time],
                    ElementType::F32,
                )?;
                // Source+right pad, then completed CNN+transposed projection
                // input, then completed mean+normalized mode. The two CNN
                // layouts use separate classes during their overlap.
                input_bytes = input_bytes.max(add(
                    retained_latents,
                    add(waveform, padded_waveform)?.max(hidden).max(mode),
                )?);
                output_bytes = output_bytes.max(add(retained_visual, hidden.max(mode))?);
                retained_latents = add(retained_latents, mode)?;
                weight_bytes = weight_bytes
                    .max(PreparedFastH3AudioEncoderTrunk::host_weight_backing_bytes(
                        &assets.audio_vae,
                        host,
                    )?)
                    .max(
                        PreparedFastH3AudioEncoderProjection::host_weight_backing_bytes(
                            &assets.audio_vae,
                            host,
                        )?,
                    );
                references.push(ReferenceConditioningPlan::Audio(AudioConditioningPlan {
                    wave,
                    maximum_samples,
                }));
                continue;
            }
            VideoReference::Video(encoded) => {
                let video = super::video::VideoConditioningPlan::new(
                    assets,
                    video_input,
                    encoded.clone(),
                    request,
                    limits,
                    host,
                )?;
                input_bytes = input_bytes.max(add(retained_latents, video.input_bytes)?);
                output_bytes = output_bytes.max(add(retained_visual, video.output_bytes)?);
                weight_bytes = weight_bytes.max(video.weight_bytes);
                retained_latents = add(retained_latents, video.latent_bytes)?;
                retained_visual = add(retained_visual, video.visual_bytes)?;
                references.push(ReferenceConditioningPlan::Video(video));
                continue;
            }
        };
        let (width, height) = png_dimensions_hint(encoded.bytes(), limits.image)
            .context("reference requires a bounded static RGB(A)8 PNG")?;
        let reference = FastH3ReferenceImagePlan::new(
            width,
            height,
            FAST_H3_REFERENCE_IMAGE_SHORT_EDGE,
            limits.image,
        )?;
        let (w, h) = reference.output_dimensions();
        ensure!(
            u64::from(w) % compression == 0 && u64::from(h) % compression == 0,
            "reference canvas differs from VAE spatial compression"
        );
        let vision = assets
            .processor
            .prepare_image(assets.vision_architecture.clone(), w, h)?;
        // The existing per-image working floor coexists with all preceding
        // normalized references. Completed vision outputs have their own class.
        input_bytes = input_bytes.max(add(input_floor, retained_latents)?);
        output_bytes = output_bytes.max(add(output_floor, retained_visual)?);
        let decoded = payload(
            host,
            &[u64::from(height), u64::from(width), 4],
            ElementType::U8,
        )?;
        input_bytes = input_bytes.max(add(
            add(decoded, reference.input_peak_bytes(host)?)?,
            retained_latents,
        )?);
        let latent = payload(
            host,
            &[
                1,
                assets.video_vae.latent_channels() as u64,
                1,
                u64::from(h) / compression,
                u64::from(w) / compression,
            ],
            ElementType::F32,
        )?;
        retained_latents = add(retained_latents, latent)?;
        let image = payload(host, &[u64::from(h), u64::from(w), 4], ElementType::U8)?;
        let normalized = payload(host, &[1, 3, u64::from(h), u64::from(w)], ElementType::F32)?;
        input_bytes = input_bytes.max(add(add(image, normalized)?, retained_latents)?);
        let mut vision_input = vision.input_backing_bytes(host)?;
        for (id, spec) in vision.config().input_tensor_specs()? {
            if id.as_str() != Qwen3VLVisionFamily::PATCHES_ID {
                vision_input = add(
                    vision_input,
                    payload(host, &spec.dimensions, spec.element_type)?,
                )?;
            }
        }
        input_bytes = input_bytes.max(add(add(image, vision_input)?, retained_latents)?);
        let feature = payload(
            host,
            &[
                vision.config().merged_rows()?,
                vision.config().architecture().output_size(),
            ],
            ElementType::Bf16,
        )?;
        retained_visual = add(
            retained_visual,
            feature.checked_mul(4).context("vision output overflow")?,
        )?;
        references.push(ReferenceConditioningPlan::Image(ImageConditioningPlan {
            reference,
            vision,
        }));
    }
    let configurations = references
        .iter()
        .map(|reference| match reference {
            ReferenceConditioningPlan::Image(image) => {
                FastH3PresentationReference::Image(image.vision.config())
            }
            ReferenceConditioningPlan::Audio(_) => FastH3PresentationReference::Audio,
            ReferenceConditioningPlan::Video(video) => FastH3PresentationReference::Video {
                vision: video.vision.config(),
                block_timestamps: video.reference.block_timestamps(),
                has_audio: video.maximum_audio_samples.is_some(),
            },
        })
        .collect::<Vec<_>>();
    let presentation = FastH3MediaPresentation::ordered_references(
        &assets.tokenizer,
        &request.parameters.prompt,
        &configurations,
        limits.presentation,
    )?;
    let text = presentation.host_footprint(&assets.text_architecture, host)?;
    ensure!(
        text.vision_output_bytes == retained_visual,
        "presentation and vision output extents differ"
    );
    input_bytes = input_bytes.max(add(text.input_bytes, retained_latents)?);
    output_bytes = output_bytes.max(add(retained_visual, text.text_output_bytes)?);
    let refined = payload(
        host,
        &[
            presentation.token_ids().len() as u64,
            assets.transformer.config().hidden_size() as u64,
        ],
        ElementType::Bf16,
    )?;
    output_bytes = output_bytes.max(add(
        text.text_output_bytes,
        refined.checked_mul(2).context("refiner output overflow")?,
    )?);
    Ok(ConditioningPlan {
        references,
        presentation,
        input_bytes,
        output_bytes,
        weight_bytes,
    })
}
