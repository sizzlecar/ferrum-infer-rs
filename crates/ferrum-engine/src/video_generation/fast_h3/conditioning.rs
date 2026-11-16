//! Original image conditioning on the caller's runtime, Host account and lane.
//! Checkpoint preparation is separate from execution; no stage opens a device,
//! launches an example process or exchanges tensors through files.
use ferrum_execution::{
    component::{CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutionOutput, TensorSerialActor,
};
use ferrum_runtime::*;
use std::{any::Any, fmt, sync::Arc};

mod audio;
mod dit;
pub use audio::encode_audio_reference;
mod keyframe;
mod reuse;
mod text;
mod video;
mod vision;
pub use dit::{precompute_tables, refine_text, ConditioningCompilation, ConditioningTables};
pub use keyframe::{encode_keyframe, prepare_reference_image};
#[cfg(feature = "cuda-fast-h3")]
pub(crate) use reuse::ConditioningKey;
pub use reuse::FastH3ConditioningReuse;
pub use text::{encode_text, prepare_text, PreparedTextConditioning};
pub use video::encode_video_reference;
pub use vision::{encode_vision, prepare_vision, PreparedVisionConditioning};

/// Native failures retain the original error/session/caller phase. The worker
/// must stop and retain this error with its ordinary retirement authority.
#[must_use]
pub struct ConditioningFailure {
    pub stage: &'static str,
    diagnostic: String,
    owner: Option<Box<dyn Any + Send>>,
}
impl ConditioningFailure {
    pub(crate) fn preparation(stage: &'static str, error: impl fmt::Display) -> Self {
        Self {
            stage,
            diagnostic: error.to_string(),
            owner: None,
        }
    }
    fn retain<E: fmt::Display + Send + 'static, T: Send + 'static>(
        stage: &'static str,
        error: E,
        owner: T,
    ) -> Self {
        let owner = Box::new((error, owner));
        let diagnostic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.0.to_string()))
                .unwrap_or_else(|_| {
                    "conditioning error formatting failed; original owners retained".into()
                });
        Self {
            stage,
            diagnostic,
            owner: Some(owner),
        }
    }
    pub fn has_retirement_owner(&self) -> bool {
        self.owner.is_some()
    }
}
impl fmt::Display for ConditioningFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.stage, self.diagnostic)
    }
}
impl fmt::Debug for ConditioningFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl std::error::Error for ConditioningFailure {}

/// A single real component execution, fenced and explicitly closed before
/// another model is installed. The original phase is retained before upload.
#[allow(clippy::too_many_arguments)]
fn forward<R: DeviceRuntime, P: Send + 'static>(
    compiled: Arc<CompiledTensorComponent<R>>,
    runtime: Arc<R>,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<R>,
    initialization: ComponentInitializationConfig,
    run: RunId,
    request: RequestIdentity,
    source: &dyn WeightComponentSource,
    phase: P,
    inputs: Vec<ManagedHostTensor>,
) -> Result<Vec<TensorExecutionOutput>, ConditioningFailure> {
    let phase = (compiled.clone(), runtime.clone(), phase, inputs.clone());
    let mut session = compiled
        .initialize_retained_with_pool_policy(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(source),
            initialization,
            phase,
            ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
        )
        .map_err(|e| ConditioningFailure::retain("conditioning initialization", e, ()))?;
    let outputs = match session
        .session_mut()
        .execute_on_actor(actor, run, request, inputs)
    {
        Ok(outputs) => outputs,
        Err(e) => {
            return Err(ConditioningFailure::retain(
                "conditioning forward",
                e,
                session,
            ))
        }
    };
    let (returned, _) = session
        .close()
        .map_err(|e| ConditioningFailure::retain("conditioning close", e, ()))?;
    drop(returned);
    Ok(outputs)
}
