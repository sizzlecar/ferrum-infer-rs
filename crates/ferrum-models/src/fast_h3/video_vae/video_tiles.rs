//! Bounded reference-video encoding: original temporal clips and spatial tile
//! seams, with one existing component session and the caller's Host account.
use super::{
    FastH3ImageSpatialPolicy, FastH3VideoEncodePlan, FastH3VideoEncoderGeometry,
    FastH3VideoEncoderLimits, FastH3VideoEncoderProfile, FastH3VideoVaeConfig,
    PreparedFastH3VideoEncoder,
};
use crate::p_h3_image_vae::tiling::{
    MomentBlendArithmetic, PH3ImageVaeMomentCanvas, PH3ImageVaeSpatialPolicy, PH3ImageVaeTilePlan,
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::TensorComponentSession, TensorExecutionError, TensorExecutionOutput,
    TensorSerialActor,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use std::sync::Arc;

pub struct PreparedFastH3TiledVideoEncoder {
    encoder: PreparedFastH3VideoEncoder,
    spatial: PH3ImageVaeTilePlan,
    temporal: FastH3VideoEncodePlan,
    input: ResolvedTensorSpec,
    output: ResolvedTensorSpec,
    fingerprint: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastH3VideoTemporalPolicy {
    AllFrames,
    /// Preserve full source pixels for Qwen, but encode only complete reference
    /// clip boundaries before the VAE's final repeat-last padding.
    ReferenceClipTrim,
}
impl PreparedFastH3TiledVideoEncoder {
    pub fn prepare(
        architecture: &FastH3VideoVaeConfig,
        source: Arc<SafetensorsArchive>,
        geometry: FastH3VideoEncoderGeometry,
        temporal: FastH3VideoTemporalPolicy,
        spatial: FastH3ImageSpatialPolicy,
        limits: FastH3VideoEncoderLimits,
        profile: FastH3VideoEncoderProfile,
    ) -> Result<Self> {
        let scale = architecture.spatial_compression_ratio()?;
        let spatial = match spatial {
            FastH3ImageSpatialPolicy::FastVideoTiled { maximum_tiles } => {
                PH3ImageVaeSpatialPolicy::Tiled {
                    tile_pixels: 256,
                    minimum_overlap_pixels: 64,
                    max_tiles: maximum_tiles,
                }
            }
            FastH3ImageSpatialPolicy::WholeImageDiagnostic => PH3ImageVaeSpatialPolicy::WholeImage,
        };
        let spatial = PH3ImageVaeTilePlan::new(
            geometry.width.try_into()?,
            geometry.height.try_into()?,
            scale,
            spatial,
        )?;
        let frames = geometry.frames.try_into()?;
        let frames = match temporal {
            FastH3VideoTemporalPolicy::AllFrames => frames,
            FastH3VideoTemporalPolicy::ReferenceClipTrim => {
                FastH3VideoEncodePlan::reference_frames(architecture, frames)?
            }
        };
        let temporal = FastH3VideoEncodePlan::new(architecture, frames, usize::MAX)?;
        let input = spec(
            vec![
                geometry.batch,
                3,
                geometry.frames,
                geometry.height,
                geometry.width,
            ],
            ElementType::U8,
        )?;
        let channels = u64::try_from(architecture.latent_channels())?
            .checked_mul(2)
            .context("posterior channels overflow")?;
        let output = spec(
            vec![
                geometry.batch,
                channels,
                temporal.output_moments().try_into()?,
                geometry.height / scale as u64,
                geometry.width / scale as u64,
            ],
            ElementType::F32,
        )?;
        ensure!(
            input.minimum_storage_bytes()? <= limits.maximum_activation_bytes
                && output.minimum_storage_bytes()? <= limits.maximum_activation_bytes,
            "video pixels or full posterior exceed activation capacity"
        );
        let tile = spatial.tile(0)?;
        let encoder = PreparedFastH3VideoEncoder::prepare(
            architecture,
            source,
            FastH3VideoEncoderGeometry {
                batch: geometry.batch,
                frames: temporal.clip_frames().try_into()?,
                height: tile.height.try_into()?,
                width: tile.width.try_into()?,
            },
            limits,
            profile,
        )?;
        ensure!(
            encoder.output_spec().dimensions()
                == [
                    geometry.batch,
                    channels,
                    temporal.clip_moments() as u64,
                    (tile.height / scale) as u64,
                    (tile.width / scale) as u64
                ],
            "original encoder output differs from spatial/temporal tile metadata"
        );
        let fingerprint = encoder.family().fingerprint()?;
        Ok(Self {
            encoder,
            spatial,
            temporal,
            input,
            output,
            fingerprint,
        })
    }
    pub fn tile_encoder(&self) -> &PreparedFastH3VideoEncoder {
        &self.encoder
    }
    pub fn input_spec(&self) -> &ResolvedTensorSpec {
        &self.input
    }
    pub fn output_spec(&self) -> &ResolvedTensorSpec {
        &self.output
    }
    /// Canvas/readback/retained raw neighbours are simultaneous Output owners.
    /// Input pixels, the normalized tile and runtime workspace are separate.
    pub fn output_peak_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        let [batch, channels, _, _, _]: [u64; 5] = self.output.dimensions().try_into()?;
        let planes = channels
            .checked_mul(self.temporal.clip_moments().try_into()?)
            .context("clip planes overflow")?;
        host.backing_bytes(self.output.minimum_storage_bytes()?.try_into()?)?
            .checked_add(PH3ImageVaeMomentCanvas::output_peak(
                host,
                &self.spatial,
                batch.try_into()?,
                planes.try_into()?,
            )?)
            .context("video moment output peak overflows")
    }
    pub fn begin(
        &self,
        pixels: &ManagedHostTensor,
        request: RequestIdentity,
        host: &HostMemoryAccount,
    ) -> Result<FastH3VideoMomentRequest> {
        ensure!(
            pixels.tensor() == &self.input
                && pixels.bytes().class() == HostMemoryClass::InputPreprocess
                && host.owns(pixels.bytes()),
            "video reference requires exact U8 NCTHW pixels on the original Host account"
        );
        let output = host.allocate_zeroed(
            HostMemoryClass::Output,
            self.output.minimum_storage_bytes()?.try_into()?,
        )?;
        Ok(FastH3VideoMomentRequest {
            spatial: self.spatial.clone(),
            temporal: self.temporal.clone(),
            pixels: pixels.clone(),
            request,
            host: host.clone(),
            fingerprint: self.fingerprint.clone(),
            tile_output: self.encoder.output_spec().clone(),
            output_spec: self.output.clone(),
            output,
            canvas: None,
            pending: None,
            next_clip: 0,
            executed_tiles: 0,
        })
    }
}

pub struct FastH3VideoMomentRequest {
    spatial: PH3ImageVaeTilePlan,
    temporal: FastH3VideoEncodePlan,
    pixels: ManagedHostTensor,
    request: RequestIdentity,
    host: HostMemoryAccount,
    fingerprint: String,
    tile_output: ResolvedTensorSpec,
    output_spec: ResolvedTensorSpec,
    output: ManagedHostBuffer,
    canvas: Option<PH3ImageVaeMomentCanvas>,
    pending: Option<Vec<TensorExecutionOutput>>,
    next_clip: usize,
    executed_tiles: usize,
}
#[must_use = "execution failure may retain native retirement owners"]
pub enum FastH3VideoMomentError<R: DeviceRuntime> {
    Preparation(anyhow::Error),
    Execution(TensorExecutionError<R>),
}
impl<R: DeviceRuntime> std::fmt::Display for FastH3VideoMomentError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preparation(e) => write!(f, "video tile preparation: {e}"),
            Self::Execution(e) => write!(f, "video tile execution: {e}"),
        }
    }
}
impl<R: DeviceRuntime> std::fmt::Debug for FastH3VideoMomentError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
impl FastH3VideoMomentRequest {
    pub fn completed_clips(&self) -> usize {
        self.next_clip
    }
    pub fn executed_tiles(&self) -> usize {
        self.executed_tiles
    }
    pub fn advance<R: DeviceRuntime>(
        &mut self,
        session: &mut TensorComponentSession<R>,
        run: RunId,
    ) -> Result<bool, FastH3VideoMomentError<R>> {
        self.advance_inner(session, run, None)
    }
    pub fn advance_on_actor<R: DeviceRuntime>(
        &mut self,
        session: &mut TensorComponentSession<R>,
        run: RunId,
        actor: &mut TensorSerialActor<R>,
    ) -> Result<bool, FastH3VideoMomentError<R>> {
        self.advance_inner(session, run, Some(actor))
    }
    fn advance_inner<R: DeviceRuntime>(
        &mut self,
        session: &mut TensorComponentSession<R>,
        run: RunId,
        actor: Option<&mut TensorSerialActor<R>>,
    ) -> Result<bool, FastH3VideoMomentError<R>> {
        let check = || -> Result<()> {
            ensure!(
                self.next_clip < self.temporal.chunk_count(),
                "video moments already complete"
            );
            ensure!(
                session.host_account().same_parent(&self.host)
                    && session
                        .executable()
                        .execution_plan()
                        .payload()
                        .prepared_family_fingerprint()
                        == self.fingerprint,
                "video tile session differs from original component or Host parent"
            );
            Ok(())
        };
        check().map_err(FastH3VideoMomentError::Preparation)?;
        if self.canvas.is_none() {
            let d = self.tile_output.dimensions();
            self.canvas = Some(
                PH3ImageVaeMomentCanvas::with_arithmetic(
                    self.spatial.clone(),
                    self.host.clone(),
                    d[0] as usize,
                    (d[1] * d[2]) as usize,
                    MomentBlendArithmetic::ScalarReciprocalMultiply,
                )
                .map_err(FastH3VideoMomentError::Preparation)?,
            );
        }
        let tile = self
            .canvas
            .as_ref()
            .unwrap()
            .next_tile()
            .map_err(FastH3VideoMomentError::Preparation)?;
        if self.pending.is_none() {
            let input = self
                .tile_input(tile.index)
                .map_err(FastH3VideoMomentError::Preparation)?;
            let outputs = match actor {
                Some(actor) => {
                    session.execute_on_actor(actor, run, self.request.clone(), vec![input])
                }
                None => session.execute(run, self.request.clone(), vec![input]),
            };
            self.pending = Some(outputs.map_err(FastH3VideoMomentError::Execution)?);
            self.executed_tiles += 1;
        }
        let check_output = || -> Result<ResolvedTensorSpec> {
            let outputs = self.pending.as_ref().unwrap();
            ensure!(
                outputs.len() == 1
                    && outputs[0].value_id().as_str() == PreparedFastH3VideoEncoder::OUTPUT_ID
                    && outputs[0].tensor() == &self.tile_output,
                "video tile output differs from original encoder port/shape"
            );
            let d = self.tile_output.dimensions();
            // Explicit Host-only contiguous view: C,T are adjacent physical
            // axes. Spatial blending treats every (channel,time) plane alike.
            spec(vec![d[0], d[1] * d[2], d[3], d[4]], ElementType::F32)
        };
        let view = check_output().map_err(FastH3VideoMomentError::Preparation)?;
        let canvas = self.canvas.as_mut().unwrap();
        canvas
            .push(tile.index, &view, self.pending.as_ref().unwrap()[0].bytes())
            .map_err(FastH3VideoMomentError::Preparation)?;
        self.pending = None;
        if canvas.completed_tiles() == self.spatial.tile_count() {
            self.commit_clip()
                .map_err(FastH3VideoMomentError::Preparation)?;
        }
        Ok(self.next_clip == self.temporal.chunk_count())
    }
    fn tile_input(&self, index: usize) -> Result<ManagedHostTensor> {
        let tile = self.spatial.tile(index)?;
        let [batch, _, frames, height, width]: [u64; 5] =
            self.pixels.tensor().dimensions().try_into()?;
        let shape = spec(
            vec![
                batch,
                3,
                self.temporal.clip_frames() as u64,
                tile.height as u64,
                tile.width as u64,
            ],
            ElementType::F32,
        )?;
        let mut bytes = self.host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            shape.minimum_storage_bytes()?.try_into()?,
        )?;
        for bc in 0..batch as usize * 3 {
            for t in 0..self.temporal.clip_frames() {
                let frame = self.temporal.source_frame(self.next_clip, t)?;
                for y in 0..tile.height {
                    let source = ((bc * frames as usize + frame) * height as usize + tile.y + y)
                        * width as usize
                        + tile.x;
                    let dest =
                        ((bc * self.temporal.clip_frames() + t) * tile.height + y) * tile.width * 4;
                    for (word, &pixel) in bytes.as_mut()[dest..dest + tile.width * 4]
                        .chunks_exact_mut(4)
                        .zip(&self.pixels.bytes().bytes()[source..source + tile.width])
                    {
                        word.copy_from_slice(
                            &super::super::normalize_reference_rgb(pixel, bc % 3).to_le_bytes(),
                        );
                    }
                }
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(PreparedFastH3VideoEncoder::INPUT_ID)?,
            shape,
            bytes.freeze(),
        )?)
    }
    fn commit_clip(&mut self) -> Result<()> {
        let chunk = self.temporal.chunk(self.next_clip)?;
        let d = self.output_spec.dimensions();
        let plane = usize::try_from(d[3] * d[4] * 4)?;
        let clip_channel = plane * self.temporal.clip_moments();
        let output_channel = plane * self.temporal.output_moments();
        let publish = plane * chunk.publish_moments;
        // No fallible allocation occurs after consuming the finished canvas.
        // Failed tile blending retains the genuine readback above for retry.
        let (_, bytes) = self
            .canvas
            .take()
            .context("video clip canvas missing")?
            .finish()?;
        for bc in 0..usize::try_from(d[0] * d[1])? {
            let from = bc * clip_channel;
            let to = bc * output_channel + chunk.output_start * plane;
            self.output.as_mut()[to..to + publish]
                .copy_from_slice(&bytes.bytes()[from..from + publish]);
        }
        self.next_clip += 1;
        Ok(())
    }
    pub fn finish(self) -> Result<FastH3VideoMoments> {
        ensure!(
            self.next_clip == self.temporal.chunk_count()
                && self.pending.is_none()
                && self.canvas.is_none(),
            "cannot publish partial video moments"
        );
        Ok(FastH3VideoMoments {
            tensor: self.output_spec,
            bytes: self.output.freeze(),
        })
    }
}
pub struct FastH3VideoMoments {
    tensor: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
}
impl FastH3VideoMoments {
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
}
fn spec(dimensions: Vec<u64>, dtype: ElementType) -> Result<ResolvedTensorSpec> {
    Ok(ResolvedTensorSpec::new(
        dimensions,
        dtype,
        ResolvedTensorLayout::Contiguous,
    )?)
}
#[cfg(test)]
mod tests;
