//! Exact-shape tensor execution over the shared runtime authority.
//!
//! This synchronous first slice has one job per call, owned Host inputs and
//! outputs, and ordinary submission. It is not a token-generation driver or a
//! diffusion scheduler. Capacity deferrals return without waiting unless the
//! serial scheduler explicitly supplies a cancellable admission control.

use std::{fmt, sync::Arc, time::Instant};

use ferrum_runtime::*;

mod admission;
mod admission_control;
mod request;
mod serial_actor;
mod timing;
mod wave_identity;
pub use admission_control::{TensorAdmissionCancellation, TensorAdmissionControl};
pub use ferrum_runtime::TensorRequestStateMapping;
pub use request::{TensorRequestCloseFailure, TensorRequestSession};
pub use serial_actor::{TensorSerialActor, TensorSerialActorStreamRequirements};
pub use timing::{TensorExecutionTiming, TensorTimingNode};
#[cfg(test)]
mod output_slices;

/// Fixed-tensor maintenance bounds and explicit device diagnostics. Logical
/// capacity waiting remains controlled by the serial actor's admission policy.
#[derive(Debug, Clone, Copy)]
pub struct TensorExecutorConfig {
    pub maximum_backing_maintenance_passes: usize,
    /// Explicit backend observations; Off preserves the ordinary submission path.
    pub device_timing: DeviceTimingMode,
}

impl TensorExecutorConfig {
    /// A serial actor retains its existing lane between blocking executions.
    pub const fn serial_actor_stream_requirements(self) -> TensorSerialActorStreamRequirements {
        TensorSerialActorStreamRequirements::new()
    }

    /// One lane spans submission, readback and retirement of a blocking call.
    /// Tensor batch dimensions do not create one lane per sample.
    pub const fn execution_stream_requirement(self) -> BlockingStreamRequirement {
        BlockingStreamRequirement::new(1)
    }

    /// Fixed request admission holds state backing, not a lane. Initialization
    /// copies run on the producing execution's lane; later steps each use the
    /// ordinary blocking execution requirement above.
    pub const fn request_state_stream_requirement(self) -> BlockingStreamRequirement {
        BlockingStreamRequirement::new(0)
    }
}

impl Default for TensorExecutorConfig {
    fn default() -> Self {
        Self {
            maximum_backing_maintenance_passes: 8,
            device_timing: DeviceTimingMode::Off,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorAdmissionStage {
    Request,
    Step,
    Wave,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorAdmissionFailure {
    Deferred,
    Rejected,
    MaintenanceLimit,
    RequestState,
}

/// Indeterminate outcomes retain both their exact handle and reaper. Callers
/// may use the existing runtime recovery APIs; dropping an error never grants
/// permission to reuse memory that has not been proved quiescent.
#[derive(thiserror::Error)]
pub enum TensorExecutionError<R: DeviceRuntime> {
    #[error("tensor execution cancelled before submission after transient retirement")]
    CancelledBeforeSubmission(TensorAdmissionCancellation),
    #[error(transparent)]
    Runtime(#[from] VNextError),
    #[error(transparent)]
    Host(#[from] HostMemoryError),
    #[error("tensor lane creation failed: {0}")]
    Lane(ExecutionLaneCreationError<R::Error>),
    #[error("tensor {stage:?} admission {kind:?}: {reason}")]
    Admission {
        stage: TensorAdmissionStage,
        kind: TensorAdmissionFailure,
        reason: String,
    },
    #[error("tensor dispatch failed: {source}")]
    Dispatch {
        source: SubmissionWaveDispatchError<R>,
        retirement: TensorExecutionRetirement<R>,
        reaper: Arc<CompletionReaper<R>>,
    },
    #[error("tensor completion observation failed: {source}")]
    Completion {
        source: VNextError,
        completion: CompletionHandle<R>,
        retirement: TensorExecutionRetirement<R>,
        reaper: Arc<CompletionReaper<R>>,
    },
    #[error("tensor completion is not terminal: {observation:?}")]
    NonTerminal {
        observation: CompletionObservation,
        completion: CompletionHandle<R>,
        retirement: TensorExecutionRetirement<R>,
        reaper: Arc<CompletionReaper<R>>,
    },
    #[error("tensor operation failed after quiescence: {receipt:?}")]
    ExecutionFailed {
        receipt: OperationCompletionReceipt,
        retirement: TensorExecutionRetirement<R>,
    },
    #[error("tensor output readback failed after quiescent compute: {source}")]
    Readback {
        source: VNextError,
        receipt: OperationCompletionReceipt,
        retirement: TensorExecutionRetirement<R>,
    },
    #[error("tensor retirement failed: {source}")]
    Retirement {
        source: VNextError,
        retirement: TensorExecutionRetirement<R>,
    },
}

impl<R: DeviceRuntime> fmt::Debug for TensorExecutionError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<R: DeviceRuntime> TensorExecutionError<R> {
    pub fn is_cancelled_before_submission(&self) -> bool {
        matches!(self, Self::CancelledBeforeSubmission(_))
    }
}

/// Compiled output (or an explicitly selected batch item) with an immutable,
/// accounted Host owner.
#[derive(Debug, Clone)]
pub struct TensorExecutionOutput {
    value_id: ProgramValueId,
    tensor: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
    timing: Option<Arc<TensorExecutionTiming>>,
}

impl TensorExecutionOutput {
    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
    /// Shared once per execution, not a separate measurement per output.
    pub fn device_timing(&self) -> Option<&TensorExecutionTiming> {
        self.timing.as_deref()
    }

    /// Select one item of a contiguous leading batch axis, retaining that axis
    /// with extent one. Every slice keeps the entire physical backing charged
    /// until its last owner drops; this neither copies nor reclassifies bytes.
    pub fn batch_item(&self, index: usize) -> Result<Self, VNextError> {
        if self.tensor.layout() != &ResolvedTensorLayout::Contiguous {
            return Err(invalid("batch slicing requires a contiguous output"));
        }
        let mut dimensions = self.tensor.dimensions().to_vec();
        let batch = dimensions
            .first_mut()
            .ok_or_else(|| invalid("output has no batch axis"))?;
        let count = usize::try_from(*batch).map_err(|_| invalid("batch exceeds address space"))?;
        if index >= count || self.bytes.len() % count != 0 {
            return Err(invalid(
                "batch item is out of bounds or has an invalid byte extent",
            ));
        }
        *batch = 1;
        let stride = self.bytes.len() / count;
        let start = index
            .checked_mul(stride)
            .ok_or_else(|| invalid("batch offset overflow"))?;
        let tensor = ResolvedTensorSpec::new(
            dimensions,
            self.tensor.element_type(),
            ResolvedTensorLayout::Contiguous,
        )?;
        let bytes = self
            .bytes
            .slice(start..start + stride)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            value_id: self.value_id.clone(),
            tensor,
            bytes,
            timing: self.timing.clone(),
        })
    }
}

pub struct PreparedTensorExecutor<R: DeviceRuntime> {
    executable: ExecutablePlan,
    providers: BoundOperationProviderSet<R>,
    resources: Arc<PlanRuntimeResources<R>>,
    binding: TensorExecutionBinding<R>,
    outputs: Vec<(CompiledTensorPort, CompletionReadbackRequest)>,
    reaper: Arc<CompletionReaper<R>>,
    config: TensorExecutorConfig,
    wave_identity: wave_identity::TensorWaveIdentityCache<R>,
}

impl<R: DeviceRuntime> PreparedTensorExecutor<R> {
    pub fn new(
        executable: ExecutablePlan,
        providers: BoundOperationProviderSet<R>,
        resources: Arc<PlanRuntimeResources<R>>,
        host: HostMemoryAccount,
        config: TensorExecutorConfig,
    ) -> Result<Self, TensorExecutionError<R>> {
        let plan = executable.execution_plan();
        let binding = resources.bind_tensor_execution(plan, host)?;
        if !binding.supports_managed_io() {
            return Err(invalid("backend does not support managed tensor transfers").into());
        }
        if plan.payload().nodes().is_empty()
            || providers.len() != plan.payload().nodes().len()
            || binding.ports().outputs().is_empty()
        {
            return Err(
                invalid("tensor executor needs complete providers and output ports").into(),
            );
        }
        let outputs = binding
            .ports()
            .outputs()
            .iter()
            .map(|port| {
                let node = plan
                    .payload()
                    .nodes()
                    .iter()
                    .find(|node| node.id() == port.node_id())
                    .ok_or_else(|| invalid("compiled tensor output node is missing"))?;
                let value = node
                    .values()
                    .iter()
                    .find(|value| {
                        value.value_id() == port.value_id()
                            && value.role() == ResolvedValueRole::Output
                            && value.ordinal() == port.ordinal()
                    })
                    .ok_or_else(|| invalid("compiled tensor output binding is missing"))?;
                let [component] = value.storage().components() else {
                    return Err(invalid(
                        "tensor output requires one contiguous physical component",
                    ));
                };
                if !plan
                    .payload()
                    .terminal_output_resources()
                    .contains(component.resource_id())
                    || value.tensor() != port.tensor()
                    || component.length_bytes() != port.tensor().minimum_storage_bytes()?
                {
                    return Err(invalid(
                        "tensor output must retain its exact storage through completion",
                    ));
                }
                let layout = HostTransferLayout::new(
                    port.tensor().element_type(),
                    port.tensor().minimum_storage_bytes()?
                        / port.tensor().element_type().size_bytes(),
                )?;
                Ok((
                    port.clone(),
                    CompletionReadbackRequest::new(
                        port.node_id().clone(),
                        0,
                        component.resource_id().clone(),
                        component.offset_bytes(),
                        layout,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        Ok(Self {
            executable,
            providers,
            resources,
            binding,
            outputs,
            reaper: CompletionReaper::new(),
            config,
            wave_identity: wave_identity::TensorWaveIdentityCache::default(),
        })
    }

    pub fn host_account(&self) -> &HostMemoryAccount {
        self.binding.host_account()
    }

    /// The same runtime reaper remains available for pending/indeterminate work.
    pub fn completion_reaper(&self) -> &Arc<CompletionReaper<R>> {
        &self.reaper
    }

    pub fn execute(
        &mut self,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        let work = self.binding.bind_inputs(inputs)?;
        self.execute_bound(run_id, request_id, work, None, None, None, None)
    }

    /// Execute through a scheduler-owned serial lane. Admission and completion
    /// retain the same exact plan, request, step and resource authorities.
    pub fn execute_on_actor(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        let control = actor.admission_control().cloned();
        actor.execute(|lane| {
            let work = self.binding.bind_inputs(inputs)?;
            self.execute_bound(
                run_id,
                request_id,
                work,
                None,
                None,
                Some(lane),
                control.as_ref(),
            )
        })
    }

    fn execute_bound(
        &mut self,
        run_id: RunId,
        request_id: RequestIdentity,
        work: ResourceWorkShape,
        request_session: Option<Arc<SequenceSession<R>>>,
        initialize: Option<(&TensorRequestStateTarget<R>, &[TensorRequestStateMapping])>,
        execution_lane: Option<&Arc<ExecutionLane<R>>>,
        control: Option<&TensorAdmissionControl>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        // Host wall observations only: native timing, synchronization and the
        // admission/retirement sequence remain unchanged. The span also binds
        // nested backend construction observations to this exact execution.
        let trace_enabled = tracing::enabled!(tracing::Level::DEBUG);
        let trace = tracing::debug_span!(
            "tensor_execution",
            run_id = %run_id,
            request_id = %request_id,
        );
        let _entered = trace.enter();
        let total_started = trace_enabled.then(Instant::now);
        let timing_context = self
            .config
            .device_timing
            .completion_enabled()
            .then(|| (run_id.clone(), request_id.clone()));
        let persistent_request = request_session.is_some();
        let outputs = self
            .outputs
            .iter()
            .filter(|(port, _)| {
                !initialize.is_some_and(|(_, mappings)| {
                    mappings.iter().any(|m| &m.source_output == port.value_id())
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        if outputs.is_empty() {
            return Err(
                invalid("tensor execution requires at least one Host completion output").into(),
            );
        }
        // Reserve every output atomically, before device admission or transfer
        // slots. Real buffers remain charged through returned clones/slices.
        let demand = outputs
            .iter()
            .map(|(_, request)| {
                let bytes = usize::try_from(request.output_layout().byte_len()?)
                    .map_err(|_| invalid("tensor output exceeds Host address space"))?;
                Ok((HostMemoryClass::Output, bytes))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let claims = self.host_account().reserve_many(&demand)?;
        let readbacks = outputs
            .iter()
            .zip(claims)
            .map(|((_, request), claim)| Ok((request.clone(), claim.allocate_zeroed()?)))
            .collect::<Result<Vec<_>, HostMemoryError>>()?;
        let readback_prepare_us = host_wall_us(total_started);

        let sequence_started = trace_enabled.then(Instant::now);
        let session = match request_session {
            Some(session) => session,
            None => self
                .admit_sequence(&work, run_id, request_id, control)?
                .open_session()?,
        };
        let mut scope = TensorExecutionRetirement {
            step: None,
            session: Some(session),
        };
        let batch = ExecutionBatchParticipants::new(vec![scope.session()?.clone()])?;
        let sequence_admission_us = host_wall_us(sequence_started);
        let initial_admission_us = host_wall_us(total_started);
        let lane_started = trace_enabled.then(Instant::now);
        let lane = match execution_lane {
            Some(lane) => Arc::clone(lane),
            None => self
                .resources
                .create_execution_lane()
                .map_err(TensorExecutionError::Lane)?,
        };
        let lane_create_us = host_wall_us(lane_started);
        let admission_started = trace_enabled.then(Instant::now);
        // Accumulate Step/Wave work separately from their capacity waits on
        // every retry. Initial sequence waiting stays in sequence_admission_us.
        let mut step_admission_us = 0_u64;
        let mut wave_prepare_us = 0_u64;
        let mut admission_wait_us = 0_u64;
        let mut step_maintenance = 0;
        let mut wave_maintenance = 0;
        let mut wave = loop {
            if control.is_some_and(TensorAdmissionControl::is_cancelled) {
                return Err(scope.cancel_unsubmitted(persistent_request));
            }
            let step_started = trace_enabled.then(Instant::now);
            let step = self.admit_step(&batch, &lane, &work, &mut step_maintenance)?;
            step_admission_us = step_admission_us.saturating_add(host_wall_us(step_started));
            match step {
                admission::AdmissionProgress::Ready(step) => scope.step = Some(step),
                admission::AdmissionProgress::Deferred(condition) => {
                    let wait_started = trace_enabled.then(Instant::now);
                    let changed =
                        self.wait_for_capacity(TensorAdmissionStage::Step, condition, control)?;
                    admission_wait_us =
                        admission_wait_us.saturating_add(host_wall_us(wait_started));
                    if !changed {
                        return Err(scope.cancel_unsubmitted(persistent_request));
                    }
                    continue;
                }
            }
            let wave_started = trace_enabled.then(Instant::now);
            let prepared = self.prepare_wave(
                scope.step.as_ref().expect("admitted step"),
                &mut wave_maintenance,
            )?;
            wave_prepare_us = wave_prepare_us.saturating_add(host_wall_us(wave_started));
            match prepared {
                admission::AdmissionProgress::Ready(wave) => break wave,
                admission::AdmissionProgress::Deferred(condition) => {
                    // No wave/participant flight exists. Release the exact
                    // unsubmitted step before subscribing; no scheduler clone
                    // may keep the frame or its backing live while parked.
                    let condition = match scope.rollback_for_capacity_wait(&condition) {
                        Ok(condition) => condition,
                        Err(source) => {
                            return Err(TensorExecutionError::Retirement {
                                source,
                                retirement: scope,
                            })
                        }
                    };
                    let wait_started = trace_enabled.then(Instant::now);
                    let changed =
                        self.wait_for_capacity(TensorAdmissionStage::Wave, condition, control)?;
                    admission_wait_us =
                        admission_wait_us.saturating_add(host_wall_us(wait_started));
                    if !changed {
                        return Err(scope.cancel_unsubmitted(persistent_request));
                    }
                }
            }
        };
        if let Some((target, mappings)) = initialize {
            let initialize_started = trace_enabled.then(Instant::now);
            wave.initialize_tensor_request_state(
                self.executable.execution_plan(),
                target,
                mappings,
            )?;
            wave_prepare_us = wave_prepare_us.saturating_add(host_wall_us(initialize_started));
        }
        let identity_started = trace_enabled.then(Instant::now);
        let active = TrustedActiveSequenceBinding::from_session(scope.session()?)?;
        // Admission above still proves the lane's exact runtime and physical
        // owners. Only the immutable node/provider identity is reused here;
        // each bind checks this wave's live request, work and frame authority.
        let topology = self.wave_identity.get_or_compile(&self.executable, &lane)?;
        let identity = OperationDispatch::bind_compiled_submission_wave_identity(
            topology,
            std::iter::once(&active),
            &wave,
            &lane,
        )?;
        let identity_bind_us = host_wall_us(identity_started);
        let uploads_started = trace_enabled.then(Instant::now);
        let uploads = self.binding.input_uploads(&work)?;
        let uploads_prepare_us = host_wall_us(uploads_started);
        let admission_wave_us =
            initial_admission_us.saturating_add(host_wall_us(admission_started));
        let encode_started = trace_enabled.then(Instant::now);
        let dispatch = OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
            self.providers.providers(),
            &self.executable,
            &identity,
            std::iter::once(&active),
            self.config.device_timing,
            &uploads,
            SubmissionExecutionPolicy::adaptive(),
            &timing::NoHostDispatchTiming,
            wave,
            &lane,
            &self.reaper,
        );
        let (handle, submission_observation) = match dispatch {
            Ok(handle) => handle.into_parts(),
            Err(source) => {
                return Err(TensorExecutionError::Dispatch {
                    source,
                    retirement: scope,
                    reaper: self.reaper.clone(),
                })
            }
        };
        let encode_submit_us = host_wall_us(encode_started);
        let wait_started = trace_enabled.then(Instant::now);
        let observed = match handle.wait_with_managed_readbacks(readbacks) {
            Ok(observed) => observed,
            Err(source) => {
                return Err(TensorExecutionError::Completion {
                    source,
                    completion: handle,
                    retirement: scope,
                    reaper: self.reaper.clone(),
                })
            }
        };
        let (observation, output) = observed.into_parts();
        let receipt = match observation {
            CompletionObservation::Terminal(receipt) => receipt,
            observation => {
                return Err(TensorExecutionError::NonTerminal {
                    observation,
                    completion: handle,
                    retirement: scope,
                    reaper: self.reaper.clone(),
                })
            }
        };
        if !matches!(
            receipt.disposition(),
            OperationCompletionDisposition::Succeeded
        ) {
            return Err(TensorExecutionError::ExecutionFailed {
                receipt,
                retirement: scope,
            });
        }
        let bytes = match output
            .unwrap_or_else(|| Err(invalid("terminal tensor completion omitted readback")))
        {
            Ok(bytes) => bytes,
            Err(source) => {
                return Err(TensorExecutionError::Readback {
                    source,
                    receipt,
                    retirement: scope,
                })
            }
        };
        if bytes.len() != outputs.len() {
            return Err(TensorExecutionError::Readback {
                source: invalid("tensor output count differs from compiled ports"),
                receipt,
                retirement: scope,
            });
        }
        let wait_readback_us = host_wall_us(wait_started);
        let timing = timing_context.map(|(run, request)| {
            Arc::new(TensorExecutionTiming::from_completion(
                run,
                request,
                &receipt,
                submission_observation,
            ))
        });
        let drop_started = trace_enabled.then(Instant::now);
        drop((handle, active, identity, uploads, batch, lane));
        // This drops execution owners, not a promise that all native owners
        // disappear here: the step may retain them until normal retirement.
        let drop_lane_us = host_wall_us(drop_started);
        let retire_started = trace_enabled.then(Instant::now);
        if let Err(source) = scope.finish_with_request(persistent_request) {
            return Err(TensorExecutionError::Retirement {
                source,
                retirement: scope,
            });
        }
        let retire_us = host_wall_us(retire_started);
        if let Some(timing) = &timing {
            timing.log();
        }
        if trace_enabled {
            tracing::debug!(
                admission_wave_us,
                readback_prepare_us,
                sequence_admission_us,
                step_admission_us,
                wave_prepare_us,
                admission_wait_us,
                identity_bind_us,
                uploads_prepare_us,
                lane_create_us,
                encode_submit_us,
                wait_readback_us,
                drop_lane_us,
                retire_us,
                total_host_us = host_wall_us(total_started),
                "tensor execution host timing through retirement"
            );
        }
        outputs
            .iter()
            .zip(bytes)
            .map(|((port, _), bytes)| {
                if bytes.len() as u64 != port.tensor().minimum_storage_bytes()?
                    || bytes.class() != HostMemoryClass::Output
                    || !self.host_account().owns(&bytes)
                {
                    return Err(
                        invalid("readback owner differs from admitted tensor output").into(),
                    );
                }
                Ok(TensorExecutionOutput {
                    value_id: port.value_id().clone(),
                    tensor: port.tensor().clone(),
                    bytes,
                    timing: timing.clone(),
                })
            })
            .collect()
    }
}

fn host_wall_us(started: Option<Instant>) -> u64 {
    started.map_or(0, |started| {
        u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
    })
}

/// The exact step/session ownership of a failed or nonterminal job. After
/// recovering its completion through the runtime, call try_abort to retire it.
/// This supports safe cleanup, not resuming a failed job's output production.
pub struct TensorExecutionRetirement<R: DeviceRuntime> {
    step: Option<Arc<StepResourceLease<R>>>,
    session: Option<Arc<SequenceSession<R>>>,
}

impl<R: DeviceRuntime> TensorExecutionRetirement<R> {
    fn rollback_for_capacity_wait(
        &mut self,
        condition: &CapacityWaitCondition,
    ) -> Result<CapacityWaitCondition, VNextError> {
        let step = self
            .step
            .take()
            .ok_or_else(|| invalid("capacity rollback has no step"))?;
        match step.try_rollback_unsubmitted_for_capacity_wait(condition) {
            Ok((_, condition)) => Ok(condition),
            Err(failure) => {
                let source = invalid(format!(
                    "tensor capacity rollback failed: {}",
                    failure.error()
                ));
                self.step = Some(failure.into_step());
                Err(source)
            }
        }
    }

    fn rollback_unsubmitted(&mut self) -> Result<(), VNextError> {
        if let Some(step) = self.step.take() {
            if let Err(failure) = step.try_rollback_unsubmitted() {
                let source = invalid(format!(
                    "tensor unsubmitted rollback failed: {}",
                    failure.error()
                ));
                self.step = Some(failure.into_step());
                return Err(source);
            }
        }
        Ok(())
    }

    fn cancel_unsubmitted(mut self, keep_request: bool) -> TensorExecutionError<R> {
        let result = self.rollback_unsubmitted().and_then(|()| {
            if !keep_request {
                self.session()?.try_abort_if_quiescent()?;
            }
            self.session.take();
            Ok(())
        });
        match result {
            Ok(()) => TensorExecutionError::CancelledBeforeSubmission(
                TensorAdmissionCancellation::retired(),
            ),
            Err(source) => TensorExecutionError::Retirement {
                source,
                retirement: self,
            },
        }
    }

    /// Failure retains the exact step and session so recovery can be retried.
    /// Success releases this owner's session and input evidence, even when the
    /// caller keeps the retirement value for diagnostics. Other owners, if
    /// any, continue to retain their own resource claims.
    /// No device wait or drain is performed by this method.
    pub fn try_abort(&mut self) -> Result<(), VNextError> {
        let Some(session) = self.session.as_ref() else {
            return Ok(());
        };
        if self.step.is_none() && session.try_abort_if_quiescent().is_ok() {
            self.session.take();
            return Ok(());
        }
        // A poisoned session rejects another cancellation request but can
        // still be terminally aborted after its flights become quiescent.
        // The guarded step/session transitions below decide whether cleanup
        // is legal; cancellation failure alone is not a quiescence failure.
        let _ = session.request_cancel();
        if let Some(step) = self.step.take() {
            if let Err(failure) = step.try_abort() {
                let error = invalid(format!(
                    "tensor abort cannot retire step: {}",
                    failure.error()
                ));
                self.step = Some(failure.into_step());
                return Err(error);
            }
        }
        session.try_abort()?;
        self.session.take();
        Ok(())
    }

    fn session(&self) -> Result<&Arc<SequenceSession<R>>, VNextError> {
        self.session
            .as_ref()
            .ok_or_else(|| invalid("tensor execution session has already retired"))
    }

    fn finish_with_request(&mut self, keep_request: bool) -> Result<(), VNextError> {
        let step = self
            .step
            .take()
            .ok_or_else(|| invalid("tensor execution has no step"))?;
        if let Err(failure) = step.try_retire_normal() {
            let error = invalid(format!(
                "tensor step retirement failed: {}",
                failure.error()
            ));
            self.step = Some(failure.into_step());
            return Err(error);
        }
        if !keep_request {
            self.session()?.try_complete()?;
        }
        self.session.take();
        Ok(())
    }
}

impl<R: DeviceRuntime> fmt::Debug for TensorExecutionRetirement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorExecutionRetirement")
            .field("retains_step", &self.step.is_some())
            .field("finished", &self.session.is_none())
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> Drop for TensorExecutionRetirement<R> {
    fn drop(&mut self) {
        // Runtime-owned references/fences retain any non-quiescent memory.
        // Dropping this owner can fail-close a job, never fabricate completion.
        let _ = self.try_abort();
    }
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
