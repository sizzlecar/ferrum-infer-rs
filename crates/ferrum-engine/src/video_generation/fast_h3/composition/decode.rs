//! Final target-only decode on the service's existing runtime and serial lane.
use crate::video_generation::{
    fast_h3::{assets::FastH3Assets, cuda::FastH3CudaComposition, decode::*},
    RawVideoOutput,
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::{
        ComponentInitializationConfig, RetainedComponentCloseError,
        RetainedComponentInitializationError,
    },
    TensorSerialActor,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use ferrum_models::fast_h3::{
    audio_vae::{FastH3AudioDecoderLimits, PreparedFastH3AudioDecoder},
    video_vae::{FastH3VideoClipLimits, FastH3VideoSpatialPolicy, FastH3VideoTemporalLimits},
    FastH3DenoisingRequest, FastH3PackedLayout, FastH3VideoGeometry,
};
use ferrum_models::spatial_tiles::{SpatialTileBatch, SpatialTileGroups};
use ferrum_output::mp4::VideoAudioSpec;
use ferrum_runtime::*;
use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

type Device = CudaDeviceRuntime;
type Phase = Arc<FastH3CudaComposition>;
type VideoPlan = FastH3VideoDecodePlan<Device>;
type AudioPlan = FastH3AudioDecodePlan<Device>;
type VideoSession = FastH3VideoDecodeSession<Device, Phase>;
type VideoPhase = FastH3VideoDecodePhase<Device, Phase>;
type AudioSession = FastH3AudioDecodeSession<Device, Phase>;

#[derive(Clone, Copy, Debug)]
pub struct FastH3DecodingLimits {
    pub video: FastH3VideoClipLimits,
    pub audio: FastH3AudioDecoderLimits,
    pub temporal: FastH3VideoTemporalLimits,
    pub maximum_spatial_tiles: usize,
    pub tile_batch: SpatialTileBatch,
    pub postprocessing: FastH3VideoPostprocessing,
}

/// Bind the grouping preference to this request's validated tile inventory.
pub(super) fn spatial_groups(
    assets: &FastH3Assets,
    geometry: &FastH3VideoGeometry,
    limits: FastH3DecodingLimits,
) -> Result<SpatialTileGroups> {
    Ok(spatial_plan(assets, geometry, limits)?.tile_groups(limits.tile_batch))
}

fn spatial_plan(
    assets: &FastH3Assets,
    geometry: &FastH3VideoGeometry,
    limits: FastH3DecodingLimits,
) -> Result<ferrum_models::fast_h3::video_vae::FastH3VideoSpatialPlan> {
    ferrum_models::fast_h3::video_vae::FastH3VideoSpatialPlan::new(
        assets.video_decoder.config(),
        geometry.latent_shape().ncthw_dims(),
        FastH3VideoSpatialPolicy::FastVideoTiled {
            maximum_tiles: limits.maximum_spatial_tiles,
        },
        limits.temporal.maximum_buffer_bytes,
    )
}

/// Admission bound for the existing decoder's live Output backings, not a
/// measured peak. Model plans supply the actual crop, tile and overlap sizes;
/// each independent allocation includes its own Host alignment. Conditioning
/// and joined denoising allocations have separate lifetimes and admission.
pub(super) fn output_host_bytes(
    assets: &FastH3Assets,
    geometry: &FastH3VideoGeometry,
    host: &HostMemoryAccount,
    limits: FastH3DecodingLimits,
) -> Result<usize> {
    use ferrum_models::fast_h3::video_vae::FastH3VideoDecodePlan as TemporalPlan;
    let shape = geometry.latent_shape().ncthw_dims();
    let architecture = assets.video_decoder.config();
    let scale = architecture.spatial_compression_ratio()?;
    let height = shape[3]
        .checked_mul(scale)
        .context("decoded height overflow")?;
    let width = shape[4]
        .checked_mul(scale)
        .context("decoded width overflow")?;
    let temporal = TemporalPlan::new(
        architecture,
        shape[2],
        limits.temporal.maximum_output_frames,
    )?;
    ensure!(
        shape[0] == 1 && temporal.output_frames() == geometry.aligned_frames(),
        "decoder Output admission requires one complete request"
    );
    let spatial = spatial_plan(assets, geometry, limits)?;
    let frame = height
        .checked_mul(width)
        .and_then(|n| n.checked_mul(3))
        .context("decoded frame extent overflow")?;
    let backing = |frames: usize, element_bytes: usize| -> Result<usize> {
        if frames == 0 {
            return Ok(0);
        }
        let bytes = frame
            .checked_mul(frames)
            .and_then(|n| n.checked_mul(element_bytes))
            .context("decoded Output extent overflow")?;
        Ok(host.backing_bytes(bytes)?)
    };
    let sum = |values: &[usize]| -> Result<usize> {
        values
            .iter()
            .try_fold(0usize, |n, bytes| n.checked_add(*bytes))
            .context("decoder Output admission overflow")
    };
    let rgb = backing(temporal.output_frames(), 1)?;
    let mut maximum = rgb;
    let mut prior_overlap = 0;
    for ordinal in 0..temporal.chunk_count() {
        let chunk = temporal.chunk(ordinal)?;
        maximum = maximum.max(sum(&[
            rgb,
            prior_overlap,
            spatial.grouped_output_backing_bytes(chunk.raw_frames, limits.tile_batch, host)?,
        ])?);
        let raw = backing(chunk.raw_frames, 2)?;
        let next_overlap = backing(chunk.next_overlap_frames, 2)?;
        let published = backing(chunk.publish_frames, 2)?;
        // Prepared successor retains both old and new overlap; commit drops
        // the old overlap before the separate RGB conversion is allocated.
        maximum = maximum.max(sum(&[rgb, raw, prior_overlap, next_overlap, published])?);
        maximum = maximum.max(sum(&[
            rgb,
            raw,
            next_overlap,
            published,
            backing(chunk.publish_frames, 1)?,
        ])?);
        prior_overlap = next_overlap;
    }
    let tail = temporal.final_overlap_frames();
    maximum = maximum.max(sum(&[rgb, prior_overlap, backing(tail, 2)?])?);
    maximum = maximum.max(sum(&[rgb, backing(tail, 2)?, backing(tail, 1)?])?);
    // Video is closed before audio begins. The mono-batch readback and its
    // interleaved stereo copy coexist with the completed RGB artifact.
    let pcm = geometry
        .audio_latents_per_channel()
        .checked_mul(usize::try_from(assets.audio_vae.hop_length()?)?)
        .and_then(|n| n.checked_mul(2 * std::mem::size_of::<f32>()))
        .context("audio Output extent overflow")?;
    let pcm = host.backing_bytes(pcm)?;
    // Optional overlap readbacks are admitted at decode, never held during
    // denoising or charged to every pending request's base reservation.
    Ok(maximum.max(sum(&[rgb, pcm, pcm])?))
}

/// Initialization identities are supplied by the existing worker lifecycle;
/// they cannot be inferred from a native pointer or a global counter.
pub struct DecodingInvocation {
    pub video_initialization: ComponentInitializationConfig,
    pub audio_initialization: ComponentInitializationConfig,
    pub run: RunId,
}

pub struct PreparedDecoding {
    composition: Arc<FastH3CudaComposition>,
    video: Arc<VideoPlan>,
    audio: Arc<AudioPlan>,
    audio_target: ResolvedTensorSpec,
    pub spec: VideoAudioSpec,
}
impl PreparedDecoding {
    /// Both target copies survive while video and audio execute sequentially.
    /// Include the larger phase's repacking, tile input and positional tables.
    pub(super) fn additional_input_host_bytes(&self) -> Result<usize> {
        let host = &self.composition.host;
        let video = host.backing_bytes(usize::try_from(
            self.video.input_spec().minimum_storage_bytes()?,
        )?)?;
        let audio =
            host.backing_bytes(usize::try_from(self.audio_target.minimum_storage_bytes()?)?)?;
        let phase = self
            .video
            .additional_input_host_bytes(host)?
            .max(input_backing_bytes(
                self.audio.compiled().family().program(),
                host,
            )?);
        video
            .checked_add(audio)
            .and_then(|n| n.checked_add(phase))
            .context("decoder input phase Host bound overflow")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        assets: Arc<FastH3Assets>,
        composition: Arc<FastH3CudaComposition>,
        geometry: &FastH3VideoGeometry,
        limits: FastH3DecodingLimits,
        host: &HostMemoryAccount,
        registry: &OperationRuntimeRegistry<Device>,
        catalog: &CapabilityCatalog,
    ) -> Result<Self> {
        ensure!(
            host.same_parent(&composition.host),
            "decoder preparation has a foreign Host parent"
        );
        let video = Arc::new(VideoPlan::compile(
            assets.video_decoder.clone(),
            FastH3VideoDecodeGeometry {
                latent_shape: geometry.latent_shape().ncthw_dims(),
                spatial: FastH3VideoSpatialPolicy::FastVideoTiled {
                    maximum_tiles: limits.maximum_spatial_tiles,
                },
                temporal: limits.temporal,
                tile_batch: limits.tile_batch,
                postprocessing: limits.postprocessing,
            },
            limits.video,
            host,
            registry,
            catalog,
            &composition.materializers,
            &composition.policy,
        )?);
        let source = Arc::new(PreparedFastH3AudioDecoder::prepare(
            &assets.audio_vae,
            assets.audio_archive.clone(),
            1,
            u64::try_from(geometry.audio_latents_per_channel())?,
            host,
            limits.audio,
        )?);
        let audio = Arc::new(AudioPlan::compile(
            source,
            registry,
            catalog,
            &composition.materializers,
            &composition.policy,
        )?);
        let [batch, frames, height, width, channels] = video.output_shape();
        let output = audio.source().output_spec().dimensions();
        ensure!(
            batch == 1
                && channels == 3
                && frames == geometry.aligned_frames()
                && output.len() == 4
                && output[..3] == [2, 1, 1],
            "decoder output geometry is not one complete video/stereo request"
        );
        let spec = VideoAudioSpec {
            width: u32::try_from(width)?,
            height: u32::try_from(height)?,
            frames: frames.try_into()?,
            fps_numerator: 24,
            fps_denominator: 1,
            audio_sample_rate: audio.source().architecture().sampling_rate(),
            audio_sample_frames: output[3],
        };
        spec.rgb_bytes()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        spec.pcm_bytes()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let input = audio.source().input_spec().dimensions();
        let audio_target = ResolvedTensorSpec::new(
            vec![
                input[0]
                    .checked_mul(input[3])
                    .context("stereo target rows overflow")?,
                input[1],
            ],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        Ok(Self {
            composition,
            video,
            audio,
            audio_target,
            spec,
        })
    }

    /// A read-only coexistence forecast on the original account, after idle
    /// denoiser temporaries have actually retired. Video and audio execute
    /// sequentially, so only their larger cold runnable peak is additional.
    /// This is not a permit; initialization and execution still admit normally.
    pub(super) fn fits_with_current_residency(&self) -> Result<bool> {
        let mut peak = 0;
        for component in [self.video.compiled(), self.audio.compiled()] {
            let memory = component.executable().execution_plan().payload().memory();
            peak = peak.max(
                memory
                    .static_bytes()
                    .checked_add(memory.minimum_runnable_request_bytes())
                    .context("auxiliary runnable peak overflows")?,
            );
        }
        match self.composition.capacity.check_available_bytes(peak) {
            Ok(()) => {}
            Err(VNextError::DeviceCapacityUnavailable(_)) => return Ok(false),
            Err(error) => return Err(error.into()),
        }
        let Some(memory) = self.composition.runtime.memory_availability()? else {
            // Optional retention declines when the physical observation is not
            // available, preserving the original conservative lifecycle.
            return Ok(false);
        };
        let reserve = memory
            .total_bytes
            .saturating_sub(self.composition.device_bytes);
        let fits = peak <= memory.available_bytes().saturating_sub(reserve);
        tracing::debug!(
            additional_peak_bytes = peak,
            available_bytes = memory.available_bytes(),
            reserve_bytes = reserve,
            retained = fits,
            "FastH3 decoder residency forecast"
        );
        Ok(fits)
    }

    /// Native failures retain the session and its original composition owner.
    /// The enclosing worker must retain its borrowed actor when this fails.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        request: &FastH3DenoisingRequest,
        layout: &FastH3PackedLayout,
        host: &HostMemoryAccount,
        actor: &mut TensorSerialActor<Device>,
        cancellation: &AtomicBool,
        invocation: DecodingInvocation,
    ) -> Result<Option<RawVideoOutput>, DecodingFailure> {
        let cancelled = || cancellation.load(Ordering::Acquire);
        if !host.same_parent(&self.composition.host)
            || invocation.video_initialization.request_id != *request.request()
            || invocation.audio_initialization.request_id != *request.request()
            || invocation.video_initialization.transaction_id
                == invocation.audio_initialization.transaction_id
            || invocation.video_initialization.run_id == invocation.audio_initialization.run_id
        {
            return Err(DecodingFailure::Validation(anyhow::anyhow!(
                "decoder Host or initialization identities differ"
            )));
        }
        if cancelled() {
            return Ok(None);
        }
        let [video_target, audio_target] = managed_decoder_targets(
            request,
            layout,
            host,
            self.video.input_spec(),
            &self.audio_target,
        )
        .map_err(DecodingFailure::Validation)?;
        let readbacks = self
            .video
            .try_admit_postprocessing(&self.composition.host)
            .map_err(DecodingFailure::Validation)?;
        let mut video = self
            .video
            .clone()
            .initialize(
                self.composition.runtime.clone(),
                host.clone(),
                invocation.video_initialization,
                self.composition.clone(),
                readbacks,
            )
            .map_err(DecodingFailure::VideoInitialization)?;
        let decoded = video.decode(
            actor,
            invocation.run.clone(),
            request.request().clone(),
            &video_target,
            cancelled,
        );
        let decoded = match decoded {
            Ok(output) => Some(output),
            Err(FastH3DecodeError::Cancelled) => None,
            Err(error) => {
                return Err(DecodingFailure::VideoExecution {
                    error,
                    session: video,
                })
            }
        };
        video.close().map_err(DecodingFailure::VideoClose)?;
        let Some(decoded) = decoded else {
            return Ok(None);
        };
        if cancelled() {
            return Ok(None);
        }
        let rgb = decoded.into_bytes();
        let mut audio = self
            .audio
            .clone()
            .initialize(
                self.composition.runtime.clone(),
                host.clone(),
                invocation.audio_initialization,
                self.composition.clone(),
            )
            .map_err(DecodingFailure::AudioInitialization)?;
        let waveform = audio.decode(
            actor,
            invocation.run,
            request.request().clone(),
            &audio_target,
            cancelled,
        );
        let waveform = match waveform {
            Ok(output) => Some(output),
            Err(FastH3DecodeError::Cancelled) => None,
            Err(error) => {
                return Err(DecodingFailure::AudioExecution {
                    error,
                    session: audio,
                })
            }
        };
        audio.close().map_err(DecodingFailure::AudioClose)?;
        let Some(waveform) = waveform else {
            return Ok(None);
        };
        if cancelled() {
            return Ok(None);
        }
        ensure_output(&waveform, self.spec).map_err(DecodingFailure::Validation)?;
        RawVideoOutput::new(self.spec, rgb, waveform.into_bytes(), host)
            .map(Some)
            .map_err(|error| DecodingFailure::Validation(anyhow::anyhow!(error.to_string())))
    }
}
fn ensure_output(
    waveform: &ferrum_models::fast_h3::audio_vae::FastH3AudioWaveform,
    spec: VideoAudioSpec,
) -> Result<()> {
    ensure!(
        waveform.requests() == 1
            && waveform.samples_per_channel() == spec.audio_sample_frames
            && waveform.sampling_rate() == spec.audio_sample_rate,
        "actual waveform differs from decoder output plan"
    );
    Ok(())
}

/// The original error and native owner remain together. Diagnostic formatting
/// borrows that owner and catches external formatter panics without replacing
/// or dropping the original failure.
#[must_use]
pub enum DecodingFailure {
    Validation(anyhow::Error),
    VideoInitialization(RetainedComponentInitializationError<Device, VideoPhase>),
    VideoExecution {
        error: FastH3DecodeError<Device>,
        session: VideoSession,
    },
    VideoClose(RetainedComponentCloseError<Device, VideoPhase>),
    AudioInitialization(RetainedComponentInitializationError<Device, (Arc<AudioPlan>, Phase)>),
    AudioExecution {
        error: FastH3DecodeError<Device>,
        session: AudioSession,
    },
    AudioClose(RetainedComponentCloseError<Device, (Arc<AudioPlan>, Phase)>),
}
impl fmt::Display for DecodingFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Validation(_) => "decoder validation failed",
            Self::VideoInitialization(_) => "video initialization failed with retained owner",
            Self::VideoExecution { .. } => "video execution failed with retained session",
            Self::VideoClose(_) => "video close failed with retained owner",
            Self::AudioInitialization(_) => "audio initialization failed with retained owner",
            Self::AudioExecution { .. } => "audio execution failed with retained session",
            Self::AudioClose(_) => "audio close failed with retained owner",
        };
        f.write_str(label)?;
        let detail = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match self {
            Self::Validation(error) => error.to_string(),
            Self::VideoInitialization(error) => error.to_string(),
            Self::VideoExecution { error, .. } => error.to_string(),
            Self::VideoClose(error) => error.to_string(),
            Self::AudioInitialization(error) => error.to_string(),
            Self::AudioExecution { error, .. } => error.to_string(),
            Self::AudioClose(error) => error.to_string(),
        }));
        match detail {
            Ok(detail) => write!(f, ": {detail}"),
            Err(_) => f.write_str("; error formatter panicked, original owner retained"),
        }
    }
}
impl fmt::Debug for DecodingFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl std::error::Error for DecodingFailure {}
