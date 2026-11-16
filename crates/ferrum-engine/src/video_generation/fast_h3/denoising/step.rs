use super::*;
use ferrum_execution::component::{
    RetainedComponentCloseError, RetainedComponentInitializationError,
    RetainedTensorComponentSession,
};

pub type PredictionPhase<R> = (Arc<CompiledTensorComponent<R>>, Vec<ManagedHostTensor>);
pub enum PredictionFailure<R: DeviceRuntime> {
    Device(Box<DevicePredictionFailure<R>>),
    Initialize(RetainedComponentInitializationError<R, PredictionPhase<R>>),
    Execute {
        error: TensorExecutionError<R>,
        session: RetainedTensorComponentSession<R, PredictionPhase<R>>,
    },
    Close(RetainedComponentCloseError<R, PredictionPhase<R>>),
}
impl<R: DeviceRuntime> fmt::Display for PredictionFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(error) => write!(f, "device handoff: {error}"),
            Self::Initialize(error) => write!(f, "initialization: {error}"),
            Self::Execute { error, .. } => write!(f, "execution: {error}"),
            Self::Close(error) => write!(f, "close: {error}"),
        }
    }
}
impl<R: DeviceRuntime> fmt::Debug for PredictionFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for PredictionFailure<R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Device(error) => Some(error.as_ref()),
            Self::Initialize(error) => Some(error),
            Self::Execute { error, .. } => Some(error),
            Self::Close(error) => Some(error),
        }
    }
}

/// Each slot supplies the exact source and identifiers for its own evaluation.
/// This is a transient call argument, not a persistent slot attached to a request.
pub struct PredictionInvocation {
    pub initialization: ComponentInitializationConfig,
    pub run_id: RunId,
    pub request_id: RequestIdentity,
}

impl<R: DeviceRuntime> DenoisingStage<R> {
    /// One completed joint forward followed by each member's PDD prediction.
    /// Solver state is published only after all outputs and successors validate.
    pub fn advance(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        batch_host: &HostMemoryAccount,
        run_id: RunId,
        request_id: RequestIdentity,
        requests: Vec<&mut FastH3DenoisingRequest>,
        invocations: Vec<PredictionInvocation>,
    ) -> Result<Vec<bool>, DenoisingStageError<R>> {
        if !self.is_resident()
            || !self.host.same_parent(batch_host)
            || requests.len() != invocations.len()
            || requests.iter().any(|r| {
                r.evaluation().checked_add(r.remaining_evaluations())
                    != Some(self.predictions.len())
            })
        {
            return Err(invalid(
                "step needs resident plans, exact member predictions and the original Host parent",
            )
            .into());
        }
        let count = requests.len();
        self.select_active_batch(count)?;
        let plan = self
            .batches
            .get(&count)
            .ok_or_else(|| invalid("batch cardinality was not compiled"))?
            .plan
            .clone();
        let batch = plan.bind(requests)?;
        let mut complete = Vec::with_capacity(count);
        for (slot, invocation) in invocations.iter().enumerate() {
            let (member, evaluation) = batch.member(slot)?;
            if member != &invocation.request_id || evaluation >= self.predictions.len() {
                return Err(
                    invalid("prediction identity/evaluation differs from actual member").into(),
                );
            }
            complete.push(evaluation + 1 == self.predictions.len());
        }
        let inputs = batch.inputs_in(batch_host)?;
        if let Some(heads) = self.try_prepare_device_predictions(
            actor,
            batch_host,
            &run_id,
            &request_id,
            &batch,
            &invocations,
            &inputs,
        )? {
            self.failed = true;
            let predictions = self
                .execute_device_predictions(actor, batch_host, run_id, request_id, inputs, heads)?;
            batch
                .advance(&predictions)
                .map_err(DenoisingStageError::Solver)?;
            self.failed = false;
            return Ok(complete);
        }
        let session = self.sessions.get_mut(&count).unwrap();
        session
            .set_output_host_account(batch_host.clone())
            .map_err(DenoisingStageError::Execute)?;
        self.failed = true;
        let mut outputs = session
            .execute_on_actor(actor, run_id, request_id, inputs)
            .map_err(DenoisingStageError::Execute)?;
        if outputs.len() != 1 || outputs[0].value_id().as_str() != FastH3DenoisingFamily::OUTPUT_ID
        {
            return Err(invalid("denoiser returned an unexpected output partition").into());
        }
        let hidden = outputs.pop().unwrap();
        // execute_on_actor returned only after readback and retirement. The
        // Host hidden tensor owns no Device pool storage. Prediction admission
        // must therefore see the phase peak, not the sum of two idle/active
        // workspaces; retain the original installation, not its temporaries.
        match self.workspace {
            DenoisingWorkspacePolicy::ActivePhaseOnly => {
                match session.try_release_idle_dynamic_backing()? {
                    IdleDynamicReleaseOutcome::Released(receipt) => {
                        tracing::debug!(
                            batch = count,
                            ?receipt,
                            "released completed FastH3 forward backing before prediction"
                        );
                    }
                    IdleDynamicReleaseOutcome::Busy(owner) => {
                        return Err(DenoisingStageError::IdleBusy(owner));
                    }
                }
            }
        }
        let mut predictions = Vec::with_capacity(count);
        for (slot, invocation) in invocations.into_iter().enumerate() {
            let (_, evaluation) = batch.member(slot)?;
            let head = &self.predictions[evaluation];
            let inputs = batch.final_output_inputs(slot, &head.config, &hidden)?;
            let phase = (head.compiled.clone(), inputs.clone());
            let mut head_session = head
                .compiled
                .initialize_retained_with_pool_policy(
                    self.runtime.clone(),
                    batch_host.clone(),
                    ComponentWeightSource::Stored(self.sources[evaluation + 1].as_ref()),
                    invocation.initialization,
                    phase,
                    ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
                )
                .map_err(|e| DenoisingStageError::Prediction(PredictionFailure::Initialize(e)))?;
            let output = match head_session.session_mut().execute_on_actor(
                actor,
                invocation.run_id,
                invocation.request_id,
                inputs,
            ) {
                Ok(output) => output,
                Err(error) => {
                    return Err(DenoisingStageError::Prediction(
                        PredictionFailure::Execute {
                            error,
                            session: head_session,
                        },
                    ))
                }
            };
            let (_phase, _receipt) = head_session
                .close()
                .map_err(|e| DenoisingStageError::Prediction(PredictionFailure::Close(e)))?;
            predictions.push(output);
        }
        // This consumes the exact borrowed membership, including request order.
        // A failed later prepare rolls back all earlier prepared successors.
        batch
            .advance(&predictions)
            .map_err(DenoisingStageError::Solver)?;
        self.failed = false;
        Ok(complete)
    }
}
