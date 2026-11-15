//! Original video/audio decoders on the caller's runtime and serial actor.
//! Compilation and source ownership are reusable; request-local spatial and
//! temporal state never becomes a scheduler, context, or allocation ledger.
use std::{fmt, sync::Arc};

use anyhow::{ensure, Context};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        RetainedComponentCloseError, RetainedComponentInitializationError,
        RetainedTensorComponentSession, TensorComponentPolicy,
    },
    TensorExecutionError, TensorSerialActor,
};
use ferrum_runtime::*;

mod admission;
mod audio;
mod targets;
mod video;
pub(crate) use admission::{
    input_backing_bytes, release_conditioning_input_credit, select_host_tile_group,
};
pub use audio::{FastH3AudioDecodePlan, FastH3AudioDecodeSession};
pub(crate) use targets::managed_decoder_targets;
pub use video::{
    FastH3DecodedVideo, FastH3VideoDecodeGeometry, FastH3VideoDecodePhase, FastH3VideoDecodePlan,
    FastH3VideoDecodeSession, FastH3VideoReadbackAdmission,
};

/// CPU work may overlap the serial GPU lane after decode-phase admission of
/// separate readback credit. Optional credit never reduces denoising capacity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3VideoPostprocessing {
    #[default]
    Inline,
    OverlapIfCapacityAllows,
}

/// The original typed execution failure remains owning. On error the caller
/// still owns the session and actor and must retire them through their normal
/// close paths; formatting an error never substitutes a retirement receipt.
pub enum FastH3DecodeError<R: DeviceRuntime> {
    Validation(anyhow::Error),
    Execution(TensorExecutionError<R>),
    Cancelled,
}
impl<R: DeviceRuntime> From<anyhow::Error> for FastH3DecodeError<R> {
    fn from(error: anyhow::Error) -> Self {
        Self::Validation(error)
    }
}
impl<R: DeviceRuntime> From<VNextError> for FastH3DecodeError<R> {
    fn from(error: VNextError) -> Self {
        Self::Validation(error.into())
    }
}
impl<R: DeviceRuntime> fmt::Display for FastH3DecodeError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => write!(f, "FastH3 decode: {error}"),
            Self::Execution(error) => write!(f, "FastH3 decode execution: {error}"),
            Self::Cancelled => f.write_str("FastH3 decode cancelled between submissions"),
        }
    }
}
impl<R: DeviceRuntime> fmt::Debug for FastH3DecodeError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for FastH3DecodeError<R> {}

fn derived_run(run: &RunId, phase: &str, chunk: usize, tile: usize) -> anyhow::Result<RunId> {
    Ok(RunId::new(format!(
        "{}.decode.{phase}.{chunk}.{tile}",
        run.as_str()
    ))?)
}
