//! Reuse an initialized Stored decoder across spatial tiles. The caller owns
//! initialization, device admission and the readback Host view. This adapter
//! retains only request inputs and pending tile owners; it creates no runtime,
//! account, weight cache or independent resource ledger.
use super::{
    decoder::{PH3ImageVaeDecoderCheckpoint, PH3ImageVaeDecoderFamily, PH3ImageVaeDecoderLimits},
    image_layout::PH3ImageVaeImageLayout,
    latent_statistics::PH3ImageVaeLatentStatistics,
    tiling::{PH3ImageVaeCanvas, PH3ImageVaeCanvasDomain, PH3ImageVaeTilePlan},
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::TensorComponentSession, TensorExecutionError, TensorExecutionOutput,
    TensorSerialActor,
};
use ferrum_runtime::*;

#[must_use = "execution failures may retain retirement and resource ownership"]
pub(crate) enum PH3ImageVaeDecodeError<R: DeviceRuntime> {
    Preparation(anyhow::Error),
    Execution(TensorExecutionError<R>),
}
impl<R: DeviceRuntime> std::fmt::Debug for PH3ImageVaeDecodeError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preparation(e) => f.debug_tuple("Preparation").field(e).finish(),
            Self::Execution(e) => f.debug_tuple("Execution").field(e).finish(),
        }
    }
}
enum Pending {
    Patches(Vec<TensorExecutionOutput>),
    Pixels(ResolvedTensorSpec, ManagedHostBytes),
}

/// One spatial tile per decoder execution, retaining the request batch within
/// each tile. All tiles in a declared plan have the same physical shape, so no
/// recompilation or weight reload occurs between them. Larger tile microbatches
/// require a separately admitted session, not a hidden free-memory heuristic.
pub(crate) struct PH3ImageVaeTileDecoder<'a, R: DeviceRuntime> {
    session: &'a mut TensorComponentSession<R>,
    plan: PH3ImageVaeTilePlan,
    layout: PH3ImageVaeImageLayout,
    statistics: PH3ImageVaeLatentStatistics<'a>,
    normalized: &'a ManagedHostTensor,
    request: RequestIdentity,
    input_host: HostMemoryAccount,
    output_host: HostMemoryAccount,
    cosine: ManagedHostTensor,
    sine: ManagedHostTensor,
    canvas: PH3ImageVaeCanvas,
    pending: Option<Pending>,
    executed_tiles: usize,
}
impl<'a, R: DeviceRuntime> PH3ImageVaeTileDecoder<'a, R> {
    /// `checkpoint` is the source used to initialize `session`. Session program
    /// identity and all shapes are checked here; static source installation is
    /// the caller's existing initialization transaction responsibility.
    pub fn bind(
        session: &'a mut TensorComponentSession<R>,
        checkpoint: &PH3ImageVaeDecoderCheckpoint<'a>,
        limits: PH3ImageVaeDecoderLimits,
        plan: PH3ImageVaeTilePlan,
        normalized: &'a ManagedHostTensor,
        request: RequestIdentity,
        input_host: HostMemoryAccount,
        output_host: HostMemoryAccount,
    ) -> Result<Self> {
        ensure!(
            session.host_account().same_parent(&input_host)
                && input_host.same_parent(&output_host)
                && input_host.owns(normalized.bytes())
                && normalized.bytes().class() == HostMemoryClass::InputPreprocess,
            "VAE decoder inputs and outputs must use the existing Host parent"
        );
        let config = checkpoint.config();
        let architecture = config.architecture();
        ensure!(
            architecture.out_channels() == 3,
            "image decoder requires RGB output"
        );
        let scale = usize::try_from(architecture.patch_size())?;
        ensure!(
            plan.spatial_scale() == scale,
            "tile plan compression differs from decoder architecture"
        );
        let tile = plan.tile(0)?;
        ensure!(
            tile.width
                == usize::try_from(config.latent_width())?
                    .checked_mul(scale)
                    .context("tile width overflows")?
                && tile.height
                    == usize::try_from(config.latent_height())?
                        .checked_mul(scale)
                        .context("tile height overflows")?,
            "decoder session shape differs from the declared tile"
        );
        let [height, width] = plan.dimensions();
        let whole = PH3ImageVaeImageLayout::new(
            architecture.clone(),
            usize::try_from(config.batch())?,
            height / scale,
            width / scale,
        )?;
        ensure!(
            normalized.tensor() == &whole.latent_spec()?,
            "normalized sampler shape differs from the canvas"
        );
        let family = PH3ImageVaeDecoderFamily::new(limits)?;
        let profiles = family.numerical_profiles(config)?;
        let profile = NumericalProfileId::new(PH3ImageVaeDecoderFamily::CUDA_PROFILE_ID)
            .map_err(anyhow::Error::msg)?;
        let program = family.semantic_program(config, profiles.resolve(&profile)?)?;
        let compiled = session.executable().execution_plan().payload();
        ensure!(
            compiled.family_id() == family.family_id()
                && compiled.program_fingerprint() == program.fingerprint()?,
            "session differs from the declared complete Stored VAE decoder"
        );
        let layout = PH3ImageVaeImageLayout::new(
            architecture.clone(),
            usize::try_from(config.batch())?,
            usize::try_from(config.latent_height())?,
            usize::try_from(config.latent_width())?,
        )?;
        let statistics = PH3ImageVaeLatentStatistics::new(architecture, checkpoint.source())?;
        let (cosine, sine) = layout.managed_rotary(
            &input_host,
            ProgramValueId::new(PH3ImageVaeDecoderFamily::COSINE_ID)?,
            ProgramValueId::new(PH3ImageVaeDecoderFamily::SINE_ID)?,
        )?;
        let canvas = PH3ImageVaeCanvas::new(
            plan.clone(),
            output_host.clone(),
            usize::try_from(config.batch())?,
            3,
        )?;
        Ok(Self {
            session,
            plan,
            layout,
            statistics,
            normalized,
            request,
            input_host,
            output_host,
            cosine,
            sine,
            canvas,
            pending: None,
            executed_tiles: 0,
        })
    }

    /// A completed GPU output survives Host postprocessing admission failure.
    /// Retrying then consumes no additional GPU run and cannot change an already
    /// committed seam. The immutable sampler borrow fixes all tiles to one state.
    pub fn advance(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
    ) -> std::result::Result<bool, PH3ImageVaeDecodeError<R>> {
        let tile = self
            .canvas
            .next_tile()
            .map_err(PH3ImageVaeDecodeError::Preparation)?;
        if self.pending.is_none() {
            let prepare = || -> Result<_> {
                let rows = self.plan.managed_tile_rows(
                    &self.input_host,
                    self.normalized,
                    &self.statistics,
                    tile.index,
                    1,
                    ProgramValueId::new(PH3ImageVaeDecoderFamily::INPUT_ID)?,
                )?;
                // Same immutable, already charged local-tile coordinates for
                // every tile. Cloning retains the owners without a payload copy.
                Ok(vec![rows, self.cosine.clone(), self.sine.clone()])
            };
            let inputs = prepare().map_err(PH3ImageVaeDecodeError::Preparation)?;
            let outputs = self
                .session
                .execute_on_actor(actor, run, self.request.clone(), inputs)
                .map_err(PH3ImageVaeDecodeError::Execution)?;
            self.pending = Some(Pending::Patches(outputs));
            self.executed_tiles += 1;
        }
        if let Some(Pending::Patches(outputs)) = &self.pending {
            // Retain every completed owner before validating ports. A model
            // contract error must never cause retry to execute the GPU again.
            let validate = || -> Result<()> {
                ensure!(
                    outputs.len() == 1,
                    "decoder must return exactly one patch tensor"
                );
                ensure!(
                    outputs[0].value_id().as_str() == PH3ImageVaeDecoderFamily::OUTPUT_ID
                        && outputs[0].tensor() == &self.layout.patch_output_spec()?
                        && self.output_host.owns(outputs[0].bytes()),
                    "decoder patch output or Host owner differs"
                );
                Ok(())
            };
            validate().map_err(PH3ImageVaeDecodeError::Preparation)?;
            let output = &outputs[0];
            let (tensor, bytes) = self
                .layout
                .unpatchify(&self.output_host, output.tensor(), output.bytes())
                .map_err(PH3ImageVaeDecodeError::Preparation)?;
            // Readback storage retires before the canvas allocates stitch work.
            self.pending = Some(Pending::Pixels(tensor, bytes));
        }
        let Some(Pending::Pixels(tensor, bytes)) = &self.pending else {
            unreachable!("completed decoding has pixel storage")
        };
        self.canvas
            .push(tile.index, tensor, bytes)
            .map_err(PH3ImageVaeDecodeError::Preparation)?;
        self.pending = None;
        Ok(self.canvas.completed_tiles() == self.plan.tile_count())
    }
    pub fn completed_tiles(&self) -> usize {
        self.canvas.completed_tiles()
    }
    pub fn executed_tiles(&self) -> usize {
        self.executed_tiles
    }
    pub fn take_execution_profile(
        &mut self,
    ) -> Option<ferrum_execution::tensor::TensorExecutionProfile> {
        self.session.take_execution_profile()
    }
    pub fn finish(self) -> Result<(ResolvedTensorSpec, ManagedHostBytes)> {
        self.canvas.finish(PH3ImageVaeCanvasDomain::UnitRgb)
    }
}
#[cfg(all(test, feature = "cuda"))]
mod tests;
