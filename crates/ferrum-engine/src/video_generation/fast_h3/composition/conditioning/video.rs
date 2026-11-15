use super::plan::{add, payload};
use super::*;
use anyhow::{ensure, Context};
use ferrum_models::{
    fast_h3::{
        video_vae::{
            FastH3VideoEncoderGeometry, FastH3VideoEncoderProfile, FastH3VideoTemporalPolicy,
            PreparedFastH3TiledVideoEncoder,
        },
        FastH3DecodedVideoGeometry, FastH3ReferenceVideoCanvas, FastH3ReferenceVideoPlan,
    },
    qwen3_vl_vision::Qwen3VLVisionVideoPlan,
    video_input::{VideoContainerPlan, VideoInputCodec},
};

pub(super) struct VideoConditioningPlan {
    pub container: VideoContainerPlan,
    pub reference: FastH3ReferenceVideoPlan,
    pub encoder: Arc<PreparedFastH3TiledVideoEncoder>,
    pub vision: Qwen3VLVisionVideoPlan,
    pub maximum_audio_samples: Option<usize>,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub weight_bytes: usize,
    pub latent_bytes: usize,
    pub visual_bytes: usize,
}
impl VideoConditioningPlan {
    pub(super) fn new(
        assets: &FastH3Assets,
        codec: &VideoInputCodec,
        encoded: ManagedHostBytes,
        request: &VideoGenerationRequest,
        limits: ConditioningLimits,
        host: &HostMemoryAccount,
    ) -> anyhow::Result<Self> {
        let container = codec.inspect(encoded, &AtomicBool::new(false))?;
        let g = container.geometry();
        // Decode only the released model's 15-second source prefix. Qwen sees
        // the full 24-FPS selection; the VAE's clip trim is a separate policy.
        let last = (15. * f64::from(g.frame_rate_numerator) / f64::from(g.frame_rate_denominator))
            .ceil() as usize
            + 1;
        let reference = FastH3ReferenceVideoPlan::new(
            FastH3DecodedVideoGeometry {
                width: g.width,
                height: g.height,
                frames: g.frames.min(last),
                frame_rate_numerator: g.frame_rate_numerator,
                frame_rate_denominator: g.frame_rate_denominator,
            },
            request.parameters.num_frames as usize,
            FastH3ReferenceVideoCanvas::default(),
            limits.reference_video,
        )?;
        let (width, height) = reference.output_dimensions();
        let frames = reference.output_spec().dimensions()[2];
        let encoder = Arc::new(PreparedFastH3TiledVideoEncoder::prepare(
            &assets.video_vae,
            assets.video_encoder.clone(),
            FastH3VideoEncoderGeometry {
                batch: 1,
                frames,
                height: height.into(),
                width: width.into(),
            },
            FastH3VideoTemporalPolicy::ReferenceClipTrim,
            FastH3ImageSpatialPolicy::FastVideoTiled {
                maximum_tiles: limits.maximum_image_tiles,
            },
            limits.video_encoder,
            FastH3VideoEncoderProfile::Cuda,
        )?);
        let vision = assets.video_processor.prepare_video(
            assets.vision_architecture.clone(),
            width,
            height,
            frames as usize,
            reference.qwen_frame_indices(),
        )?;
        let pixels = host.backing_bytes(
            reference
                .output_spec()
                .minimum_storage_bytes()?
                .try_into()?,
        )?;
        let tile = host.backing_bytes(
            encoder
                .tile_encoder()
                .input_spec()
                .minimum_storage_bytes()?
                .try_into()?,
        )?;
        let mut latent_dims = encoder.output_spec().dimensions().to_vec();
        latent_dims[1] /= 2;
        let mut latent_bytes = payload(host, &latent_dims, ElementType::F32)?;
        let mut input_bytes = reference
            .input_peak_bytes(host)?
            .max(add(pixels, tile)?)
            .max(add(pixels, latent_bytes)?);
        let mut vision_input = vision.input_backing_bytes(host)?;
        for (id, spec) in vision.config().input_tensor_specs()? {
            if id.as_str() != Qwen3VLVisionFamily::PATCHES_ID {
                vision_input = add(
                    vision_input,
                    payload(host, &spec.dimensions, spec.element_type)?,
                )?;
            }
        }
        input_bytes = input_bytes.max(add(add(pixels, latent_bytes)?, vision_input)?);
        let feature = payload(
            host,
            &[
                vision.config().merged_rows()?,
                vision.config().architecture().output_size(),
            ],
            ElementType::Bf16,
        )?;
        let visual_bytes = feature
            .checked_mul(4)
            .context("video vision output overflow")?;
        let mut output_bytes = encoder.output_peak_bytes(host)?.max(visual_bytes);
        let mut weight_bytes = 0;
        let maximum_audio_samples = container
            .audio()
            .map(|audio| -> anyhow::Result<_> {
                ensure!(
                    (1..=2).contains(&audio.channels),
                    "reference soundtrack must be mono or stereo"
                );
                let samples = u64::from(request.parameters.num_frames)
                    .checked_mul(u64::from(request.parameters.frame_rate.denominator))
                    .and_then(|n| n.checked_mul(u64::from(assets.audio_vae.sampling_rate())))
                    .context("video soundtrack trim overflow")?
                    / u64::from(request.parameters.frame_rate.numerator);
                let padded = assets.audio_vae.encoder_padded_samples(samples)?;
                let time = padded / assets.audio_vae.hop_length()?;
                let waveform = payload(host, &[2, 1, samples], ElementType::F32)?;
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
                // Visual pixels have been released before soundtrack processing.
                input_bytes = input_bytes.max(add(
                    latent_bytes,
                    add(waveform, padded_waveform)?.max(hidden).max(mode),
                )?);
                output_bytes = output_bytes.max(add(visual_bytes, hidden.max(mode))?);
                latent_bytes = add(latent_bytes, mode)?;
                weight_bytes = PreparedFastH3AudioEncoderTrunk::host_weight_backing_bytes(
                    &assets.audio_vae,
                    host,
                )?
                .max(
                    PreparedFastH3AudioEncoderProjection::host_weight_backing_bytes(
                        &assets.audio_vae,
                        host,
                    )?,
                );
                usize::try_from(samples).map_err(Into::into)
            })
            .transpose()?;
        Ok(Self {
            container,
            reference,
            encoder,
            vision,
            maximum_audio_samples,
            input_bytes,
            output_bytes,
            weight_bytes,
            latent_bytes,
            visual_bytes,
        })
    }
}

impl PreparedConditioner {
    pub(super) fn execute_reference_video(
        &mut self,
        plan: &VideoConditioningPlan,
        host: &HostMemoryAccount,
        actor: &mut TensorSerialActor<Device>,
        cancellation: &AtomicBool,
        initialization: &mut impl FnMut(
            &str,
        ) -> std::result::Result<
            ComponentInitializationConfig,
            VNextError,
        >,
        original_request: &mut Option<RequestIdentity>,
    ) -> Result<Option<(ConditionedReference, [TensorExecutionOutput; 4])>> {
        if cancellation.load(Ordering::Acquire) {
            return Ok(None);
        }
        let frames = plan
            .reference
            .source_frame_indices()
            .last()
            .copied()
            .expect("nonempty planned frames")
            + 1;
        let pixels = plan
            .container
            .with_frames(frames, host, cancellation, |reader| {
                plan.reference
                    .prepare_with_frames(host, |index| reader.read(index))
            });
        let pixels = match pixels {
            Ok(pixels) => pixels,
            Err(_) if cancellation.load(Ordering::Acquire) => return Ok(None),
            Err(error) => return Err(prep("reference video decode", error)),
        };
        let encoder = plan.encoder.tile_encoder();
        let fingerprint = encoder
            .family()
            .fingerprint()
            .map_err(|e| prep("reference encoder family", e))?;
        let reuse = self
            .last_video
            .as_ref()
            .map(|c| c.family().fingerprint())
            .transpose()
            .map_err(|e| prep("reference encoder family", e))?
            .as_ref()
            == Some(&fingerprint);
        if !reuse {
            self.last_video =
                Some(self.compile(encoder.family().clone(), encoder.compile_options())?);
        }
        let init = initialization("reference_video")
            .map_err(|e| prep("reference encoder initialization", e))?;
        let request_id = init.request_id.clone();
        if original_request
            .as_ref()
            .is_some_and(|id| id != &request_id)
        {
            return Err(prep(
                "reference identity",
                "conditioning changed request identity",
            ));
        }
        *original_request = Some(request_id.clone());
        let Some(video) = encode_video_reference(
            &self.assets.video_vae,
            plan.encoder.clone(),
            self.last_video.as_ref().unwrap().clone(),
            self.cuda.runtime.clone(),
            host,
            actor,
            init,
            pixels.clone(),
            cancellation,
        )?
        else {
            return Ok(None);
        };
        let patches = plan
            .vision
            .managed_patches(host, &pixels)
            .map_err(|e| prep("video vision patches", e))?;
        drop(pixels);
        if cancellation.load(Ordering::Acquire) {
            return Ok(None);
        }
        let init = checked_init(initialization, "vision", &request_id)?;
        let visual = self.vision_forward(plan.vision.config(), patches, host, actor, init)?;
        if cancellation.load(Ordering::Acquire) {
            return Ok(None);
        }
        let audio = if let Some(samples) = plan.maximum_audio_samples {
            let waveform = plan.container.prepare_audio(
                self.assets.audio_vae.sampling_rate(),
                samples,
                host,
                cancellation,
            );
            let waveform = match waveform {
                Ok(waveform) => waveform,
                Err(_) if cancellation.load(Ordering::Acquire) => return Ok(None),
                Err(error) => return Err(prep("video soundtrack decode", error)),
            };
            let compilation = ConditioningCompilation {
                registry: &self.registry,
                catalog: &self.catalog,
                policy: &self.cuda.policy,
                materializers: &self.cuda.materializers,
            };
            Some(encode_audio_reference(
                &self.assets.audio_vae,
                self.assets.audio_archive.clone(),
                waveform,
                self.assets.audio_vae.sampling_rate(),
                &compilation,
                self.cuda.runtime.clone(),
                host,
                actor,
                self.limits.audio_encoder,
                |name| {
                    let init = initialization(name)?;
                    if init.request_id != request_id {
                        return Err(VNextError::InvalidExecutionPlan {
                            reason: "conditioning changed request identity".into(),
                        });
                    }
                    Ok(init)
                },
            )?)
        } else {
            None
        };
        if cancellation.load(Ordering::Acquire) {
            return Ok(None);
        }
        Ok(Some((ConditionedReference::Video { video, audio }, visual)))
    }
}
