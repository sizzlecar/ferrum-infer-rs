//! The controller's one Host/dataflow implementation chooses only the existing
//! session execution lane. It never creates a runtime or a second actor.
use super::*;
use ferrum_execution::TensorSerialActor;
pub(super) enum TextExecution<'a, R: DeviceRuntime> {
    Direct,
    Actor(&'a mut TensorSerialActor<R>),
}
impl<R: DeviceRuntime> TextExecution<'_, R> {
    pub fn execute(
        &mut self,
        session: &mut TensorComponentSession<R>,
        run: RunId,
        request: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
    ) -> std::result::Result<Vec<ferrum_execution::TensorExecutionOutput>, TensorExecutionError<R>>
    {
        match self {
            Self::Direct => session.execute(run, request, inputs),
            Self::Actor(actor) => session.execute_on_actor(actor, run, request, inputs),
        }
    }
}
impl<R: DeviceRuntime> PH3ImageTextConditioning<'_, R> {
    /// Both real submissions borrow the engine's same serial actor. The
    /// caller fences an older denoising batch before this conditioning stage.
    pub fn execute_on_actor(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        prompt: &PH3ImageTextPromptBatch,
        run: PH3ImageTextConditioningRun,
    ) -> std::result::Result<PH3ImageTextConditioningOutput, PH3ImageTextConditioningError<R>> {
        self.execute_with(prompt, run, TextExecution::Actor(actor))
    }
}
#[cfg(test)]
mod tests;
