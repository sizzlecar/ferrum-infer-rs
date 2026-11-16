//! Original FastVideo keyframe tile traversal, sharing the causal CNN, checked
//! spatial split and raw-neighbour moment canvas with the existing image VAE.
use super::{
    FastH3ImageEncoderLimits, FastH3ImageEncoderProfile, FastH3VideoVaeConfig,
    PreparedFastH3ImageEncoder,
};
use crate::p_h3_image_vae::tiling::{
    MomentBlendArithmetic, PH3ImageVaeMomentCanvas, PH3ImageVaeSpatialPolicy, PH3ImageVaeTilePlan,
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::TensorComponentSession, TensorExecutionError, TensorExecutionOutput,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use std::sync::Arc;

pub const FAST_H3_IMAGE_TILE_PROFILE: &str = "fasth3.keyframe.tiles256_overlap64.row_major.raw_neighbours.vertical_then_horizontal.f32.scalar_reciprocal.v1";
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FastH3ImageSpatialPolicy {
    /// Original runtime defaults: 256-pixel tiles, at least 64-pixel overlap.
    FastVideoTiled { maximum_tiles: usize },
    /// Explicit diagnostic arithmetic: whole-image GroupNorm differs from tiles.
    WholeImageDiagnostic,
}
impl Default for FastH3ImageSpatialPolicy {
    fn default() -> Self {
        Self::FastVideoTiled {
            maximum_tiles: 1024,
        }
    }
}
pub struct PreparedFastH3TiledImageEncoder {
    encoder: PreparedFastH3ImageEncoder,
    plan: PH3ImageVaeTilePlan,
    input: ResolvedTensorSpec,
    output: ResolvedTensorSpec,
    fingerprint: String,
}
impl PreparedFastH3TiledImageEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        architecture: &FastH3VideoVaeConfig,
        source: Arc<SafetensorsArchive>,
        batch: u64,
        height: u64,
        width: u64,
        policy: FastH3ImageSpatialPolicy,
        limits: FastH3ImageEncoderLimits,
        profile: FastH3ImageEncoderProfile,
        host: &HostMemoryAccount,
    ) -> Result<Self> {
        let scale = architecture.spatial_compression_ratio()?;
        let spatial = match policy {
            FastH3ImageSpatialPolicy::FastVideoTiled { maximum_tiles } => {
                PH3ImageVaeSpatialPolicy::Tiled {
                    tile_pixels: 256,
                    minimum_overlap_pixels: 64,
                    max_tiles: maximum_tiles,
                }
            }
            FastH3ImageSpatialPolicy::WholeImageDiagnostic => PH3ImageVaeSpatialPolicy::WholeImage,
        };
        let plan = PH3ImageVaeTilePlan::new(
            usize::try_from(width)?,
            usize::try_from(height)?,
            scale,
            spatial,
        )?;
        let input = spec(vec![batch, 3, height, width])?;
        let output = spec(vec![
            batch,
            u64::try_from(architecture.latent_channels())?
                .checked_mul(2)
                .context("moment channel overflow")?,
            height / scale as u64,
            width / scale as u64,
        ])?;
        ensure!(
            input.minimum_storage_bytes()? <= limits.maximum_input_bytes
                && output.minimum_storage_bytes()? <= limits.maximum_output_bytes,
            "full keyframe/canvas exceeds capacity"
        );
        let tile = plan.tile(0)?;
        let encoder = PreparedFastH3ImageEncoder::prepare(
            architecture,
            source,
            batch,
            tile.height as u64,
            tile.width as u64,
            limits,
            profile,
            host,
        )?;
        let fingerprint = encoder.family().fingerprint()?;
        Ok(Self {
            encoder,
            plan,
            input,
            output,
            fingerprint,
        })
    }
    pub fn tile_encoder(&self) -> &PreparedFastH3ImageEncoder {
        &self.encoder
    }
    pub fn family(&self) -> &PreparedModelFamily {
        self.encoder.family()
    }
    pub fn compile_options(&self) -> &ProgramPlanCompileOptions {
        self.encoder.compile_options()
    }
    pub fn weight_source(&self, host: HostMemoryAccount) -> super::FastH3ImageEncoderSource<'_> {
        self.encoder.weight_source(host)
    }
    pub fn input_spec(&self) -> &ResolvedTensorSpec {
        &self.input
    }
    pub fn output_spec(&self) -> &ResolvedTensorSpec {
        &self.output
    }
    pub fn tile_count(&self) -> usize {
        self.plan.tile_count()
    }
    pub fn begin(
        &self,
        pixels: &ManagedHostTensor,
        request: RequestIdentity,
        host: &HostMemoryAccount,
    ) -> Result<FastH3ImageMomentRequest> {
        ensure!(
            pixels.tensor() == &self.input
                && pixels.bytes().class() == HostMemoryClass::InputPreprocess
                && host.owns(pixels.bytes()),
            "keyframe pixels require the exact full F32 input and original Host parent"
        );
        ensure!(
            pixels
                .bytes()
                .bytes()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "nonfinite keyframe pixels"
        );
        let canvas = PH3ImageVaeMomentCanvas::with_arithmetic(
            self.plan.clone(),
            host.clone(),
            usize::try_from(self.output.dimensions()[0])?,
            usize::try_from(self.output.dimensions()[1])?,
            MomentBlendArithmetic::ScalarReciprocalMultiply,
        )?;
        Ok(FastH3ImageMomentRequest {
            plan: self.plan.clone(),
            pixels: pixels.clone(),
            host: host.clone(),
            request,
            fingerprint: self.fingerprint.clone(),
            tile_output: self.encoder.output_spec().clone(),
            canvas: Some(canvas),
            pending: None,
            executed_tiles: 0,
        })
    }
}

/// Request-owned progress retains the genuine readback before Host blending.
/// An allocation rejection can retry that commit without executing a tile again.
pub struct FastH3ImageMomentRequest {
    plan: PH3ImageVaeTilePlan,
    pixels: ManagedHostTensor,
    host: HostMemoryAccount,
    request: RequestIdentity,
    fingerprint: String,
    tile_output: ResolvedTensorSpec,
    canvas: Option<PH3ImageVaeMomentCanvas>,
    pending: Option<Vec<TensorExecutionOutput>>,
    executed_tiles: usize,
}
#[must_use = "execution failures may retain native retirement owners"]
pub enum FastH3ImageMomentError<R: DeviceRuntime> {
    Preparation(anyhow::Error),
    Execution(TensorExecutionError<R>),
}
impl<R: DeviceRuntime> std::fmt::Display for FastH3ImageMomentError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preparation(e) => write!(f, "keyframe tile preparation: {e}"),
            Self::Execution(e) => write!(f, "keyframe tile execution: {e}"),
        }
    }
}
impl<R: DeviceRuntime> std::fmt::Debug for FastH3ImageMomentError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
impl FastH3ImageMomentRequest {
    pub fn completed_tiles(&self) -> usize {
        self.canvas.as_ref().map_or(
            self.plan.tile_count(),
            PH3ImageVaeMomentCanvas::completed_tiles,
        )
    }
    pub fn executed_tiles(&self) -> usize {
        self.executed_tiles
    }
    pub fn advance<R: DeviceRuntime>(
        &mut self,
        session: &mut TensorComponentSession<R>,
        run: RunId,
    ) -> std::result::Result<bool, FastH3ImageMomentError<R>> {
        self.advance_inner(session, run, None)
    }
    /// Use the product worker's existing serial lane for every spatial tile.
    pub fn advance_on_actor<R: DeviceRuntime>(
        &mut self,
        session: &mut TensorComponentSession<R>,
        run: RunId,
        actor: &mut ferrum_execution::TensorSerialActor<R>,
    ) -> std::result::Result<bool, FastH3ImageMomentError<R>> {
        self.advance_inner(session, run, Some(actor))
    }
    fn advance_inner<R: DeviceRuntime>(
        &mut self,
        session: &mut TensorComponentSession<R>,
        run: RunId,
        actor: Option<&mut ferrum_execution::TensorSerialActor<R>>,
    ) -> std::result::Result<bool, FastH3ImageMomentError<R>> {
        let check = || -> Result<()> {
            ensure!(
                session.host_account().same_parent(&self.host)
                    && session
                        .executable()
                        .execution_plan()
                        .payload()
                        .prepared_family_fingerprint()
                        == self.fingerprint,
                "tile session differs from prepared original CNN or Host parent"
            );
            Ok(())
        };
        check().map_err(FastH3ImageMomentError::Preparation)?;
        let tile = self
            .canvas
            .as_ref()
            .context("keyframe moments already published")
            .and_then(PH3ImageVaeMomentCanvas::next_tile)
            .map_err(FastH3ImageMomentError::Preparation)?;
        if self.pending.is_none() {
            let input = self
                .tile_input(tile.index)
                .map_err(FastH3ImageMomentError::Preparation)?;
            let outputs = match actor {
                Some(actor) => {
                    session.execute_on_actor(actor, run, self.request.clone(), vec![input])
                }
                None => session.execute(run, self.request.clone(), vec![input]),
            };
            self.pending = Some(outputs.map_err(FastH3ImageMomentError::Execution)?);
            self.executed_tiles += 1;
        }
        let outputs = self
            .pending
            .as_ref()
            .expect("real completed tile outputs retained");
        let check = || -> Result<()> {
            ensure!(
                outputs.len() == 1
                    && outputs[0].value_id().as_str() == PreparedFastH3ImageEncoder::OUTPUT_ID
                    && outputs[0].tensor() == &self.tile_output,
                "tile output differs from original CNN port/shape"
            );
            Ok(())
        };
        check().map_err(FastH3ImageMomentError::Preparation)?;
        let canvas = self.canvas.as_mut().expect("active canvas");
        canvas
            .push(tile.index, outputs[0].tensor(), outputs[0].bytes())
            .map_err(FastH3ImageMomentError::Preparation)?;
        self.pending = None;
        Ok(canvas.completed_tiles() == self.plan.tile_count())
    }
    fn tile_input(&self, index: usize) -> Result<ManagedHostTensor> {
        let tile = self.plan.tile(index)?;
        let d = self.pixels.tensor().dimensions();
        let (batch, height, width) = (
            usize::try_from(d[0])?,
            usize::try_from(d[2])?,
            usize::try_from(d[3])?,
        );
        let tensor = spec(vec![d[0], 3, tile.height as u64, tile.width as u64])?;
        let mut bytes = self.host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(tensor.minimum_storage_bytes()?)?,
        )?;
        for bc in 0..batch * 3 {
            for y in 0..tile.height {
                let source = ((bc * height + tile.y + y) * width + tile.x) * 4;
                let dest = (bc * tile.height + y) * tile.width * 4;
                bytes.as_mut()[dest..dest + tile.width * 4]
                    .copy_from_slice(&self.pixels.bytes().bytes()[source..source + tile.width * 4]);
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(PreparedFastH3ImageEncoder::INPUT_ID)?,
            tensor,
            bytes.freeze(),
        )?)
    }
    pub fn finish(mut self) -> Result<FastH3ImageMoments> {
        ensure!(
            self.pending.is_none() && self.completed_tiles() == self.plan.tile_count(),
            "cannot publish partial keyframe moments"
        );
        let (tensor, bytes) = self
            .canvas
            .take()
            .context("keyframe canvas missing")?
            .finish()?;
        Ok(FastH3ImageMoments { tensor, bytes })
    }
}
/// Stitched Output ownership, deliberately not a manufactured device completion.
pub struct FastH3ImageMoments {
    tensor: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
}
impl FastH3ImageMoments {
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
}
fn spec(d: Vec<u64>) -> Result<ResolvedTensorSpec> {
    Ok(ResolvedTensorSpec::new(
        d,
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )?)
}

#[cfg(test)]
mod tests;
