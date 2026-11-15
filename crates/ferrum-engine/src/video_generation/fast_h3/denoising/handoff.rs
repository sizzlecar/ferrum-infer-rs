//! Optional on-device prediction handoff. Consumer Request state is admitted
//! before the producer, and survives retirement of the producer's Step pools.
use super::*;
use ferrum_execution::{
    component::{
        RetainedComponentCloseError, RetainedComponentInitializationError,
        RetainedTensorComponentSession,
    },
    TensorDynamicPreparationOutcome, TensorRequestCloseFailure, TensorRequestInitialization,
    TensorRequestSession,
};

struct DevicePrediction<R: DeviceRuntime> {
    component: Option<RetainedTensorComponentSession<R, PredictionPhase<R>>>,
    request: Option<TensorRequestSession<R>>,
    inputs: Vec<ManagedHostTensor>,
    mapping: TensorRequestStateMapping,
}
pub(super) struct DevicePredictions<R: DeviceRuntime> {
    heads: Vec<DevicePrediction<R>>,
}
enum Cause<R: DeviceRuntime> {
    Initialize(RetainedComponentInitializationError<R, PredictionPhase<R>>),
    Execute(TensorExecutionError<R>),
    RequestClose(TensorRequestCloseFailure<R>),
    ComponentClose(RetainedComponentCloseError<R, PredictionPhase<R>>),
    Runtime(VNextError),
    Busy(IdleDynamicReleaseBusy),
}

/// Failed/nonterminal transfers retain every original consumer and its state,
/// alongside the producer failure owned by the ordinary execution protocol.
pub struct DevicePredictionFailure<R: DeviceRuntime> {
    cause: Cause<R>,
    _owners: DevicePredictions<R>,
}
impl<R: DeviceRuntime> fmt::Display for DevicePredictionFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Initialize(e) => write!(f, "initialization: {e}"),
            Cause::Execute(e) => write!(f, "execution: {e}"),
            Cause::RequestClose(e) => write!(f, "request close: {e}"),
            Cause::ComponentClose(e) => write!(f, "component close: {e}"),
            Cause::Runtime(e) => write!(f, "runtime: {e}"),
            Cause::Busy(e) => write!(f, "producer backing still owned: {e:?}"),
        }
    }
}
impl<R: DeviceRuntime> fmt::Debug for DevicePredictionFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for DevicePredictionFailure<R> {}

impl<R: DeviceRuntime> DevicePredictions<R> {
    fn failure(self, cause: Cause<R>) -> DenoisingStageError<R> {
        DenoisingStageError::Prediction(PredictionFailure::Device(Box::new(
            DevicePredictionFailure {
                cause,
                _owners: self,
            },
        )))
    }
    fn close_one(&mut self, index: usize) -> Result<(), Cause<R>> {
        let head = &mut self.heads[index];
        if let Some(request) = head.request.take() {
            request.close().map_err(Cause::RequestClose)?;
        }
        if let Some(component) = head.component.take() {
            component.close().map_err(Cause::ComponentClose)?;
        }
        Ok(())
    }
    fn close(mut self) -> Result<(), DenoisingStageError<R>> {
        for index in 0..self.heads.len() {
            if let Err(cause) = self.close_one(index) {
                return Err(self.failure(cause));
            }
        }
        Ok(())
    }
    fn predict(
        mut self,
        actor: &mut TensorSerialActor<R>,
    ) -> Result<Vec<Vec<TensorExecutionOutput>>, DenoisingStageError<R>> {
        let mut outputs = Vec::with_capacity(self.heads.len());
        for index in 0..self.heads.len() {
            let head = &mut self.heads[index];
            let output = head
                .request
                .as_mut()
                .expect("admitted prediction state")
                .execute_on_actor(actor, head.inputs.clone());
            match output {
                Ok(output) => outputs.push(output),
                Err(error) => return Err(self.failure(Cause::Execute(error))),
            }
            if let Err(cause) = self.close_one(index) {
                return Err(self.failure(cause));
            }
        }
        Ok(outputs)
    }
}

impl<R: DeviceRuntime> DenoisingStage<R> {
    pub(super) fn try_prepare_device_predictions(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        host: &HostMemoryAccount,
        run: &RunId,
        request: &RequestIdentity,
        batch: &FastH3DenoisingBatch<'_>,
        invocations: &[PredictionInvocation],
        inputs: &[ManagedHostTensor],
    ) -> Result<Option<DevicePredictions<R>>, DenoisingStageError<R>> {
        let Some((capacity, device_bytes)) = &self.handoff_capacity else {
            return Ok(None);
        };
        let count = invocations.len();
        let mut prepared = Vec::with_capacity(count);
        let mut retained = 0u64;
        let mut prediction_transient = 0u64;
        for slot in 0..count {
            let (_, evaluation) = batch.member(slot)?;
            let Some((config, component)) = &self.predictions[evaluation].device else {
                return Ok(None);
            };
            let (head_inputs, mapping) = batch.final_output_state_inputs(slot, config)?;
            let memory = component.executable().execution_plan().payload().memory();
            let state = memory
                .minimum_request_bytes()
                .checked_add(memory.minimum_sequence_bytes())
                .ok_or_else(|| invalid("prediction state peak overflow"))?;
            retained = retained
                .checked_add(memory.static_bytes())
                .and_then(|n| n.checked_add(state))
                .ok_or_else(|| invalid("prediction retained peak overflow"))?;
            prediction_transient = prediction_transient.max(
                memory
                    .minimum_runnable_request_bytes()
                    .checked_sub(state)
                    .ok_or_else(|| invalid("prediction state exceeds its runnable peak"))?,
            );
            prepared.push((component.clone(), head_inputs, mapping, evaluation));
        }
        let producer = self
            .sessions
            .get_mut(&count)
            .expect("selected resident batch");
        match producer.try_release_idle_dynamic_backing()? {
            IdleDynamicReleaseOutcome::Released(_) => {}
            IdleDynamicReleaseOutcome::Busy(owner) => {
                return Err(DenoisingStageError::IdleBusy(owner))
            }
        }
        let peak = retained
            .checked_add(
                prediction_transient.max(producer.minimum_runnable_dynamic_bytes_with_residency()?),
            )
            .ok_or_else(|| invalid("prediction coexistence peak overflow"))?;
        match capacity.check_available_bytes(peak) {
            Ok(()) => {}
            Err(VNextError::DeviceCapacityUnavailable(_)) => return Ok(None),
            Err(error) => return Err(error.into()),
        }
        let Some(memory) = self
            .runtime
            .memory_availability()
            .map_err(DenoisingStageError::DeviceMemory)?
        else {
            return Ok(None);
        };
        if peak
            > memory
                .available_bytes()
                .saturating_sub(memory.total_bytes.saturating_sub(*device_bytes))
        {
            return Ok(None);
        }
        // The forecast grants nothing. Actual consumer admission, followed by
        // actual zero-submission producer admission, proves coexistence. A
        // capacity refusal must close all consumers before falling back.
        let mut heads = DevicePredictions {
            heads: Vec::with_capacity(count),
        };
        for ((compiled, head_inputs, mapping, evaluation), invocation) in
            prepared.into_iter().zip(invocations)
        {
            let phase = (compiled.clone(), head_inputs.clone());
            let component = match compiled.initialize_retained_with_pool_policy(
                self.runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(self.sources[evaluation + 1].as_ref()),
                invocation.initialization.clone(),
                phase,
                ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
            ) {
                Ok(component) => component,
                Err(error) => {
                    if matches!(
                        error.error(),
                        ComponentInitializationError::Contract(
                            VNextError::DeviceCapacityUnavailable(_)
                        )
                    ) {
                        match error.try_into_preflight() {
                            Ok((refusal, _phase)) => {
                                tracing::debug!(%refusal, "prediction state capacity declined before native initialization");
                                heads.close()?;
                                return Ok(None);
                            }
                            Err(error) => return Err(heads.failure(Cause::Initialize(error))),
                        }
                    }
                    return Err(heads.failure(Cause::Initialize(error)));
                }
            };
            heads.heads.push(DevicePrediction {
                component: Some(component),
                request: None,
                inputs: head_inputs,
                mapping,
            });
            let head = heads.heads.last_mut().unwrap();
            let state = head.component.as_ref().unwrap().session().begin_request(
                invocation.run_id.clone(),
                invocation.request_id.clone(),
                head.inputs.clone(),
            );
            match state {
                Ok(state) => head.request = Some(state),
                Err(refusal) if refusal.is_capacity_refusal() => {
                    tracing::debug!(%refusal, "prediction state admission declined; closing consumers before Host handoff");
                    heads.close()?;
                    return Ok(None);
                }
                Err(error) => return Err(heads.failure(Cause::Execute(error))),
            }
        }
        match producer.try_prepare_dynamic_backing_on_actor(
            actor,
            run.clone(),
            request.clone(),
            inputs.to_vec(),
        ) {
            Ok(TensorDynamicPreparationOutcome::Prepared(_)) => {
                tracing::debug!(
                    batch = count,
                    additional_peak_bytes = peak,
                    "FastH3 prediction device handoff admitted"
                );
                Ok(Some(heads))
            }
            Ok(TensorDynamicPreparationOutcome::Capacity { refusal, .. }) => {
                tracing::debug!(%refusal, "prediction producer coexistence declined; restoring Host handoff");
                heads.close()?;
                Ok(None)
            }
            Err(error) => Err(heads.failure(Cause::Execute(error))),
        }
    }

    pub(super) fn execute_device_predictions(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        host: &HostMemoryAccount,
        run: RunId,
        request: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
        mut heads: DevicePredictions<R>,
    ) -> Result<Vec<Vec<TensorExecutionOutput>>, DenoisingStageError<R>> {
        let producer = self
            .sessions
            .get_mut(&heads.heads.len())
            .expect("selected resident batch");
        if let Err(error) = producer.set_output_host_account(host.clone()) {
            return Err(heads.failure(Cause::Execute(error)));
        }
        let mut destinations = heads
            .heads
            .iter_mut()
            .map(|head| TensorRequestInitialization {
                request: head.request.as_mut().expect("admitted prediction state"),
                mappings: std::slice::from_ref(&head.mapping),
            })
            .collect::<Vec<_>>();
        let output = producer.execute_initializing_requests_on_actor(
            actor,
            run,
            request,
            inputs,
            &mut destinations,
        );
        drop(destinations);
        match output {
            Ok(outputs) if outputs.is_empty() => {}
            Ok(_) => {
                return Err(heads.failure(Cause::Runtime(invalid(
                    "device handoff returned an undeclared Host output",
                ))))
            }
            Err(error) => return Err(heads.failure(Cause::Execute(error))),
        }
        // Only separately admitted Request state survives this release. The
        // consumer has no alias into the producer's reusable activation pools.
        match producer.try_release_idle_dynamic_backing() {
            Ok(IdleDynamicReleaseOutcome::Released(_)) => {}
            Ok(IdleDynamicReleaseOutcome::Busy(owner)) => {
                return Err(heads.failure(Cause::Busy(owner)))
            }
            Err(error) => return Err(heads.failure(Cause::Runtime(error))),
        }
        heads.predict(actor)
    }
}
