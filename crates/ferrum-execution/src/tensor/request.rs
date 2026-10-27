//! Owned fixed-request lifecycle. Only explicit state initialization permits
//! execution; each step has fresh Host input evidence and ordinary retirement.
use super::*;

#[must_use = "fixed request state must be explicitly closed after its final step"]
pub struct TensorRequestSession<R: DeviceRuntime> {
    executor: PreparedTensorExecutor<R>,
    inputs: TensorRequestInputBinding<R>,
    target: TensorRequestStateTarget<R>,
    session: Option<Arc<SequenceSession<R>>>,
    run_id: RunId,
    request_id: RequestIdentity,
    failed: bool,
}

impl<R: DeviceRuntime> fmt::Debug for TensorRequestSession<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorRequestSession")
            .field("request_id", &self.request_id)
            .field("failed", &self.failed)
            .field(
                "initialization_in_flight",
                &self.target.initialization_in_flight(),
            )
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> PreparedTensorExecutor<R> {
    pub fn begin_request(
        &self,
        run_id: RunId,
        request_id: RequestIdentity,
        initial_inputs: Vec<ManagedHostTensor>,
    ) -> Result<TensorRequestSession<R>, TensorExecutionError<R>> {
        self.begin_request_with_control(run_id, request_id, initial_inputs, None)
    }

    pub fn begin_request_with_control(
        &self,
        run_id: RunId,
        request_id: RequestIdentity,
        initial_inputs: Vec<ManagedHostTensor>,
        control: Option<&TensorAdmissionControl>,
    ) -> Result<TensorRequestSession<R>, TensorExecutionError<R>> {
        let executor = Self::new(
            self.executable.clone(),
            self.providers.clone(),
            self.resources.clone(),
            self.host_account().clone(),
            self.config,
        )?;
        let inputs = executor.binding.request_input_binding()?;
        let work = inputs.admission_work(initial_inputs)?;
        let sequence =
            executor.admit_sequence(&work, run_id.clone(), request_id.clone(), control)?;
        let session = sequence.open_session()?;
        let target =
            TensorRequestStateTarget::new(executor.executable.execution_plan(), session.clone())?;
        Ok(TensorRequestSession {
            executor,
            inputs,
            target,
            session: Some(session),
            run_id,
            request_id,
            failed: false,
        })
    }

    pub fn execute_initializing_request(
        &mut self,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
        destination: &mut TensorRequestSession<R>,
        mappings: &[TensorRequestStateMapping],
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        self.execute_initializing_request_with_lane(
            run_id,
            request_id,
            inputs,
            destination,
            mappings,
            None,
            None,
        )
    }

    pub fn execute_initializing_request_on_actor(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
        destination: &mut TensorRequestSession<R>,
        mappings: &[TensorRequestStateMapping],
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        let control = actor.admission_control().cloned();
        actor.execute(|lane| {
            self.execute_initializing_request_with_lane(
                run_id,
                request_id,
                inputs,
                destination,
                mappings,
                Some(lane),
                control.as_ref(),
            )
        })
    }

    fn execute_initializing_request_with_lane(
        &mut self,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
        destination: &mut TensorRequestSession<R>,
        mappings: &[TensorRequestStateMapping],
        lane: Option<&Arc<ExecutionLane<R>>>,
        control: Option<&TensorAdmissionControl>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        if destination.failed || destination.session.is_none() {
            return Err(invalid("cannot initialize a failed or closed tensor request").into());
        }
        let work = self.binding.bind_inputs(inputs)?;
        let result = self.execute_bound(
            run_id,
            request_id,
            work,
            None,
            Some((&destination.target, mappings)),
            lane,
            control,
        );
        if result
            .as_ref()
            .is_err_and(|error| !error.is_cancelled_before_submission())
        {
            destination.failed = true;
        }
        result
    }
}

impl<R: DeviceRuntime> TensorRequestSession<R> {
    pub fn execute(
        &mut self,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        self.execute_with_lane(inputs, None, None)
    }

    pub fn execute_on_actor(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        let control = actor.admission_control().cloned();
        actor.execute(|lane| self.execute_with_lane(inputs, Some(lane), control.as_ref()))
    }

    fn execute_with_lane(
        &mut self,
        inputs: Vec<ManagedHostTensor>,
        lane: Option<&Arc<ExecutionLane<R>>>,
        control: Option<&TensorAdmissionControl>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        if self.failed {
            return Err(invalid("failed fixed request cannot execute again").into());
        }
        self.target.ensure_ready()?;
        let work = self.inputs.bind_inputs(inputs)?;
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| invalid("fixed request is closed"))?
            .clone();
        let result = self.executor.execute_bound(
            self.run_id.clone(),
            self.request_id.clone(),
            work,
            Some(session),
            None,
            lane,
            control,
        );
        if result
            .as_ref()
            .is_err_and(|error| !error.is_cancelled_before_submission())
        {
            self.failed = true;
        }
        result
    }

    pub fn completion_reaper(&self) -> &Arc<CompletionReaper<R>> {
        self.executor.completion_reaper()
    }

    pub fn close(mut self) -> Result<(), TensorRequestCloseFailure<R>> {
        let result = if self.target.initialization_in_flight() {
            Err(invalid(
                "fixed request initialization still owns a producer fence",
            ))
        } else if let Some(session) = &self.session {
            if self.failed || self.target.ensure_ready().is_err() {
                session.try_abort_if_quiescent().map(|_| ())
            } else {
                session.try_complete().map(|_| ())
            }
        } else {
            Ok(())
        };
        match result {
            Ok(()) => {
                self.session.take();
                Ok(())
            }
            Err(error) => Err(TensorRequestCloseFailure {
                error,
                request: self,
            }),
        }
    }
}

impl<R: DeviceRuntime> Drop for TensorRequestSession<R> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            // Existing session/reaper ownership retains unknown device work.
            // A failed abort never grants an initialized-state success seal.
            let _ = session.request_cancel();
            let _ = session.try_abort_if_quiescent();
        }
    }
}

#[must_use = "failed close retains the exact request for recovery and retry"]
pub struct TensorRequestCloseFailure<R: DeviceRuntime> {
    error: VNextError,
    request: TensorRequestSession<R>,
}

impl<R: DeviceRuntime> TensorRequestCloseFailure<R> {
    pub fn error(&self) -> &VNextError {
        &self.error
    }
    pub fn request(&self) -> &TensorRequestSession<R> {
        &self.request
    }
    pub fn into_parts(self) -> (VNextError, TensorRequestSession<R>) {
        (self.error, self.request)
    }
}
impl<R: DeviceRuntime> fmt::Debug for TensorRequestCloseFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorRequestCloseFailure")
            .field("error", &self.error)
            .field("request", &self.request)
            .finish()
    }
}
impl<R: DeviceRuntime> fmt::Display for TensorRequestCloseFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fixed request close failed: {}", self.error)
    }
}
impl<R: DeviceRuntime> std::error::Error for TensorRequestCloseFailure<R> {}
