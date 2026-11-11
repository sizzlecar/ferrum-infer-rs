//! Request-owned tiled editing encoder. Reuses initialized complete Stored
//! weights and the caller's serial actor; no runtime, RNG, cache or resource
//! ledger is created here. Pixel/noise payloads remain immutable borrows.
use super::{
    encode_layout::{normalized_pixel, posterior_from_noise, PH3ImageVaePixelDomain},
    encoder::{PH3ImageVaeEncoderCheckpoint, PH3ImageVaeEncoderFamily, PH3ImageVaeEncoderLimits},
    latent_statistics::PH3ImageVaeLatentStatistics,
    tiling::{PH3ImageVaeMomentCanvas, PH3ImageVaeTile, PH3ImageVaeTilePlan},
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::TensorComponentSession, TensorExecutionError, TensorExecutionOutput,
    TensorSerialActor,
};
use ferrum_runtime::*;

#[must_use = "execution failures may retain actual retirement/resource owners"]
pub(crate) enum PH3ImageVaeEncodeError<R: DeviceRuntime> {
    Preparation(anyhow::Error),
    Execution(TensorExecutionError<R>),
}
impl<R: DeviceRuntime> std::fmt::Debug for PH3ImageVaeEncodeError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preparation(e) => f.debug_tuple("Preparation").field(e).finish(),
            Self::Execution(e) => f.debug_tuple("Execution").field(e).finish(),
        }
    }
}

/// Each device run encodes one spatial tile, preserving all request rows in
/// that tile. Retained raw upper/left neighbours belong to MomentCanvas; its
/// encoding rule deliberately differs from recursive decoder blending.
pub(crate) struct PH3ImageVaeTileEncoder<'a, R: DeviceRuntime> {
    session: &'a mut TensorComponentSession<R>,
    plan: PH3ImageVaeTilePlan,
    pixels: &'a ManagedHostTensor,
    pixel_domain: PH3ImageVaePixelDomain,
    noise: &'a ManagedHostTensor,
    statistics: PH3ImageVaeLatentStatistics<'a>,
    request: RequestIdentity,
    input_host: HostMemoryAccount,
    output_host: HostMemoryAccount,
    tile_output: ResolvedTensorSpec,
    canvas: Option<PH3ImageVaeMomentCanvas>,
    // Preserve all actual readback owners before validating ports, so even a
    // contract mismatch cannot turn retry into another device execution.
    pending: Option<Vec<TensorExecutionOutput>>,
    executed_tiles: usize,
    moments: Option<(ResolvedTensorSpec, ManagedHostBytes)>,
    finished: bool,
}
impl<'a, R: DeviceRuntime> PH3ImageVaeTileEncoder<'a, R> {
    /// Static installation remains the caller's existing initialization
    /// transaction. `checkpoint` must describe that initialized session; the
    /// posterior statistics are borrowed from exactly the same archive.
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        session: &'a mut TensorComponentSession<R>,
        checkpoint: &PH3ImageVaeEncoderCheckpoint<'_>,
        limits: PH3ImageVaeEncoderLimits,
        plan: PH3ImageVaeTilePlan,
        pixels: &'a ManagedHostTensor,
        pixel_domain: PH3ImageVaePixelDomain,
        noise: &'a ManagedHostTensor,
        statistics: PH3ImageVaeLatentStatistics<'a>,
        request: RequestIdentity,
        input_host: HostMemoryAccount,
        output_host: HostMemoryAccount,
    ) -> Result<Self> {
        ensure!(
            session.host_account().same_parent(&input_host)
                && input_host.same_parent(&output_host)
                && input_host.owns(pixels.bytes())
                && input_host.owns(noise.bytes())
                && pixels.bytes().class() == HostMemoryClass::InputPreprocess
                && noise.bytes().class() == HostMemoryClass::InputPreprocess,
            "encoder pixels, noise and outputs require the existing Host parent"
        );
        let config = checkpoint.config();
        let architecture = config.architecture();
        ensure!(
            std::ptr::eq(statistics.source().archive(), checkpoint.source())
                && statistics.channels() == usize::try_from(architecture.latent_channels())?,
            "posterior must borrow original channel statistics from the encoder archive"
        );
        ensure!(
            architecture.in_channels() == 3,
            "encoder requires RGB input"
        );
        let scale =
            architecture
                .spatial_downsample_factors()
                .iter()
                .try_fold(1usize, |p, &v| {
                    p.checked_mul(usize::try_from(v)?)
                        .context("encoder spatial scale overflows")
                })?;
        ensure!(
            plan.spatial_scale() == scale,
            "tile scale differs from encoder architecture"
        );
        let tile = plan.tile(0)?;
        ensure!(
            tile.height == usize::try_from(config.height())?
                && tile.width == usize::try_from(config.width())?,
            "encoder session shape differs from the declared tile"
        );
        let batch = usize::try_from(config.batch())?;
        let [height, width] = plan.dimensions();
        ensure!(
            pixels.tensor() == &f32_spec(&[batch, 3, height, width])?,
            "original RGB shape differs from planned canvas"
        );
        ensure!(
            noise.tensor()
                == &f32_spec(&[batch, statistics.channels(), height / scale, width / scale])?,
            "provided posterior F32 noise differs from the complete moment geometry"
        );
        let output = config.output_tensor_spec()?;
        let tile_output =
            ResolvedTensorSpec::new(output.dimensions, output.element_type, output.layout)?;
        ensure!(
            tile_output
                == f32_spec(&[
                    batch,
                    statistics.channels() * 2,
                    tile.height / scale,
                    tile.width / scale
                ])?,
            "literal encoder output is not the declared compressed tile geometry"
        );
        let family = PH3ImageVaeEncoderFamily::new(limits)?;
        let profiles = family.numerical_profiles(config)?;
        let profile = NumericalProfileId::new(PH3ImageVaeEncoderFamily::REFERENCE_PROFILE_ID)
            .map_err(anyhow::Error::msg)?;
        let program = family.semantic_program(config, profiles.resolve(&profile)?)?;
        let compiled = session.executable().execution_plan().payload();
        ensure!(
            compiled.family_id() == family.family_id()
                && compiled.program_fingerprint() == program.fingerprint()?,
            "session differs from the complete Stored encoder program"
        );
        let canvas = PH3ImageVaeMomentCanvas::new(
            plan.clone(),
            output_host.clone(),
            batch,
            statistics.channels() * 2,
        )?;
        Ok(Self {
            session,
            plan,
            pixels,
            pixel_domain,
            noise,
            statistics,
            request,
            input_host,
            output_host,
            tile_output,
            canvas: Some(canvas),
            pending: None,
            executed_tiles: 0,
            moments: None,
            finished: false,
        })
    }

    /// Crop and normalize directly into one managed Input tile. Every value
    /// uses the same separate wrapper/Imagenet F32 operations as whole-image
    /// normalization. There is no normalized full-image intermediate.
    fn tile_input(&self, tile: PH3ImageVaeTile) -> Result<ManagedHostTensor> {
        let [_, _, height, width] = self.pixels.tensor().dimensions() else {
            unreachable!("bind validated RGB NCHW");
        };
        let batch = usize::try_from(self.pixels.tensor().dimensions()[0])?;
        let (height, width) = (usize::try_from(*height)?, usize::try_from(*width)?);
        let tensor = f32_spec(&[batch, 3, tile.height, tile.width])?;
        let mut output = self.input_host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(tensor.minimum_storage_bytes()?)?,
        )?;
        for b in 0..batch {
            for c in 0..3 {
                for y in 0..tile.height {
                    for x in 0..tile.width {
                        let source = (((b * 3 + c) * height + tile.y + y) * width + tile.x + x) * 4;
                        let destination = (((b * 3 + c) * tile.height + y) * tile.width + x) * 4;
                        let value = f32::from_le_bytes(
                            self.pixels.bytes().bytes()[source..source + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let value = normalized_pixel(value, c, self.pixel_domain)?;
                        output.as_mut()[destination..destination + 4]
                            .copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(PH3ImageVaeEncoderFamily::INPUT_ID)?,
            tensor,
            output.freeze(),
        )?)
    }

    pub fn advance(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
    ) -> std::result::Result<bool, PH3ImageVaeEncodeError<R>> {
        let tile = self
            .canvas
            .as_ref()
            .context("encoder tile phase is complete")
            .and_then(PH3ImageVaeMomentCanvas::next_tile)
            .map_err(PH3ImageVaeEncodeError::Preparation)?;
        if self.pending.is_none() {
            let input = self
                .tile_input(tile)
                .map_err(PH3ImageVaeEncodeError::Preparation)?;
            self.pending = Some(
                self.session
                    .execute_on_actor(actor, run, self.request.clone(), vec![input])
                    .map_err(PH3ImageVaeEncodeError::Execution)?,
            );
            self.executed_tiles += 1;
        }
        let outputs = self
            .pending
            .as_ref()
            .expect("completed encoder output retained");
        let validate = || -> Result<()> {
            ensure!(
                outputs.len() == 1,
                "encoder must return one complete moment tensor"
            );
            let output = &outputs[0];
            ensure!(
                output.value_id().as_str() == PH3ImageVaeEncoderFamily::OUTPUT_ID
                    && output.tensor() == &self.tile_output
                    && self.output_host.owns(output.bytes()),
                "encoder moments differ from declared port, geometry or Host parent"
            );
            Ok(())
        };
        validate().map_err(PH3ImageVaeEncodeError::Preparation)?;
        let output = &outputs[0];
        let canvas = self.canvas.as_mut().expect("active tile phase");
        canvas
            .push(tile.index, output.tensor(), output.bytes())
            .map_err(PH3ImageVaeEncodeError::Preparation)?;
        self.pending = None;
        Ok(canvas.completed_tiles() == self.plan.tile_count())
    }
    pub fn completed_tiles(&self) -> usize {
        self.canvas.as_ref().map_or(
            self.plan.tile_count(),
            PH3ImageVaeMomentCanvas::completed_tiles,
        )
    }

    /// Successful readbacks, including a retained tile whose Host stitch has
    /// not committed. Retrying that tile does not execute or count it twice.
    pub fn executed_tiles(&self) -> usize {
        self.executed_tiles
    }

    pub fn take_execution_profile(
        &mut self,
    ) -> Option<ferrum_execution::tensor::TensorExecutionProfile> {
        self.session.take_execution_profile()
    }

    /// Freeze complete moments once; a real posterior Input admission failure
    /// preserves this owner for retry. Noise is explicit and immutable, so no
    /// repeated device encoding or implicit RNG advance occurs on retry.
    pub fn finish(&mut self, output_id: ProgramValueId) -> Result<ManagedHostTensor> {
        ensure!(
            !self.finished,
            "encoder posterior has already been published"
        );
        ensure!(
            self.completed_tiles() == self.plan.tile_count() && self.pending.is_none(),
            "cannot sample incomplete encoder moments"
        );
        if self.moments.is_none() {
            let canvas = self.canvas.take().context("completed moments were lost")?;
            self.moments = Some(canvas.finish()?);
        }
        let (spec, moments) = self.moments.as_ref().unwrap();
        let output = posterior_from_noise(
            &self.input_host,
            spec,
            moments,
            self.noise,
            &self.statistics,
            output_id,
        )?;
        self.moments = None;
        self.finished = true;
        Ok(output)
    }
}
fn f32_spec(dimensions: &[usize]) -> Result<ResolvedTensorSpec> {
    Ok(ResolvedTensorSpec::new(
        dimensions
            .iter()
            .map(|&n| u64::try_from(n))
            .collect::<Result<Vec<_>, _>>()?,
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )?)
}

#[cfg(test)]
mod tests;
