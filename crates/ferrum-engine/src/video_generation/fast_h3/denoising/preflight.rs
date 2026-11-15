use super::*;
use ferrum_execution::TensorDynamicPreparationOutcome;

impl<R: DeviceRuntime> DenoisingStage<R> {
    /// A membership proposal performs real zero-submission Core admission.
    /// Candidate refusal releases its partial backing, preserving every model
    /// continuation; the old batch can reacquire its original pools normally.
    pub fn try_prepare_batch_on_actor(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        host: &HostMemoryAccount,
        run: RunId,
        request: RequestIdentity,
        requests: Vec<&mut FastH3DenoisingRequest>,
    ) -> Result<bool, DenoisingStageError<R>> {
        if !self.is_resident() || !host.same_parent(&self.host) {
            return Err(
                invalid("dynamic preparation requires a resident original Host domain").into(),
            );
        }
        let count = requests.len();
        let prepared = self
            .batches
            .get(&count)
            .ok_or_else(|| invalid("unsupported candidate batch"))?;
        let inputs = prepared.plan.bind(requests)?.inputs_in(host)?;
        let previous = self.active_batch;
        self.select_active_batch(count)?;
        self.failed = true;
        let outcome = self
            .sessions
            .get_mut(&count)
            .unwrap()
            .try_prepare_dynamic_backing_on_actor(actor, run, request, inputs)
            .map_err(DenoisingStageError::Execute)?;
        match outcome {
            TensorDynamicPreparationOutcome::Prepared(receipt) => {
                tracing::debug!(
                    batch = count,
                    ?receipt,
                    "FastH3 candidate acquired original dynamic backing without submission"
                );
                self.failed = false;
                Ok(true)
            }
            TensorDynamicPreparationOutcome::Capacity { refusal, rollback } => {
                tracing::debug!(batch=count,%refusal,?rollback,"FastH3 candidate capacity refused with original rollback");
                self.active_batch = previous;
                self.failed = false;
                Ok(false)
            }
        }
    }
}
