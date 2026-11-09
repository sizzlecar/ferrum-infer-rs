//! Reuse the caller's Stored decoder across spatial tiles. The common state
//! retains the request canvas/pending owners; session lifetime and admission
//! remain with the caller, including when an actual tail changes batch shape.
use super::{
    decoder::{PH3ImageVaeDecoderCheckpoint, PH3ImageVaeDecoderFamily, PH3ImageVaeDecoderLimits},
    image_layout::PH3ImageVaeImageLayout,
    latent_statistics::PH3ImageVaeLatentStatistics,
    tiling::{
        PH3ImageVaeCanvas, PH3ImageVaeCanvasDomain, PH3ImageVaeTileBatch, PH3ImageVaeTileGroup,
        PH3ImageVaeTilePlan,
    },
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::TensorComponentSession, TensorExecutionError, TensorExecutionOutput,
    TensorSerialActor,
};
use ferrum_runtime::*;
mod state;
#[cfg(all(test, feature = "cuda"))]
use state::Pending;
pub(crate) use state::{DecodeShape, PH3ImageVaeDecodeState};

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

/// Capacity-one compatibility wrapper around the same grouped state machine.
/// It borrows one session because this route has no physical shape transition.
pub(crate) struct PH3ImageVaeTileDecoder<'a, R: DeviceRuntime> {
    session: &'a mut TensorComponentSession<R>,
    state: PH3ImageVaeDecodeState<'a>,
}
impl<'a, R: DeviceRuntime> PH3ImageVaeTileDecoder<'a, R> {
    #[allow(clippy::too_many_arguments)]
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
        let config = checkpoint.config();
        let family = PH3ImageVaeDecoderFamily::new(limits)?;
        let profiles = family.numerical_profiles(config)?;
        let profile = NumericalProfileId::new(PH3ImageVaeDecoderFamily::CUDA_PROFILE_ID)
            .map_err(anyhow::Error::msg)?;
        let program = family.semantic_program(config, profiles.resolve(&profile)?)?;
        let shape = DecodeShape::new(
            config,
            1,
            family.family_id().clone(),
            program.fingerprint()?,
        )?;
        shape.validate_session(session, &input_host)?;
        let state = PH3ImageVaeDecodeState::new(
            plan,
            config.batch(),
            PH3ImageVaeTileBatch::default(),
            vec![shape],
            PH3ImageVaeLatentStatistics::new(config.architecture(), checkpoint.source())?,
            normalized,
            request,
            input_host,
            output_host,
        )?;
        Ok(Self { session, state })
    }
    pub fn advance(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
    ) -> std::result::Result<bool, PH3ImageVaeDecodeError<R>> {
        self.state.advance_with_session(self.session, actor, run)
    }
    pub fn completed_tiles(&self) -> usize {
        self.state.completed_tiles()
    }
    pub fn executed_tiles(&self) -> usize {
        self.state.executed_tiles()
    }
    pub fn take_execution_profile(
        &mut self,
    ) -> Option<ferrum_execution::tensor::TensorExecutionProfile> {
        self.session.take_execution_profile()
    }
    pub fn finish(self) -> Result<(ResolvedTensorSpec, ManagedHostBytes)> {
        self.state.finish()
    }
}
#[cfg(all(test, feature = "cuda"))]
mod tests;
