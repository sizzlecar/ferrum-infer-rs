//! Exact-shape tensor execution over the shared runtime authority.
//!
//! This synchronous first slice has one job per call, owned Host inputs and
//! outputs, and ordinary submission. It is not a token-generation driver or a
//! diffusion scheduler. Capacity deferrals return to the caller without waiting.

use std::{fmt, sync::Arc};

use ferrum_runtime::*;

mod admission;
mod request;
pub use ferrum_runtime::TensorRequestStateMapping;
pub use request::{TensorRequestCloseFailure, TensorRequestSession};
#[cfg(test)]
mod output_slices;

/// Bounds physical-backing maintenance, not logical capacity waiting.
#[derive(Debug, Clone, Copy)]
pub struct TensorExecutorConfig {
    pub maximum_backing_maintenance_passes: usize,
}

impl Default for TensorExecutorConfig {
    fn default() -> Self {
        Self {
            maximum_backing_maintenance_passes: 8,
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

/// Compiled output (or an explicitly selected batch item) with an immutable,
/// accounted Host owner.
#[derive(Debug, Clone)]
pub struct TensorExecutionOutput {
    value_id: ProgramValueId,
    tensor: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
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
        self.execute_bound(run_id, request_id, work, None, None)
    }

    fn execute_bound(
        &mut self,
        run_id: RunId,
        request_id: RequestIdentity,
        work: ResourceWorkShape,
        request_session: Option<Arc<SequenceSession<R>>>,
        initialize: Option<(&TensorRequestStateTarget<R>, &[TensorRequestStateMapping])>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
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

        let session = match request_session {
            Some(session) => session,
            None => self
                .admit_sequence(&work, run_id, request_id)?
                .open_session()?,
        };
        let mut scope = TensorExecutionRetirement {
            step: None,
            session: Some(session),
        };
        let batch = ExecutionBatchParticipants::new(vec![scope.session()?.clone()])?;
        let lane = self
            .resources
            .create_execution_lane()
            .map_err(TensorExecutionError::Lane)?;
        let step = self.admit_step(&batch, &lane, &work)?;
        scope.step = Some(step.clone());
        let mut wave = self.prepare_wave(&step)?;
        if let Some((target, mappings)) = initialize {
            wave.initialize_tensor_request_state(
                self.executable.execution_plan(),
                target,
                mappings,
            )?;
        }
        drop(step);
        let active = TrustedActiveSequenceBinding::from_session(scope.session()?)?;
        let identity = OperationDispatch::bind_submission_wave_identity(
            &self.executable,
            std::iter::once(&active),
            &wave,
            &lane,
        )?;
        let uploads = self.binding.input_uploads(&work)?;
        let dispatch = OperationDispatch::encode_and_submit_wave_with_inputs(
            self.providers.providers(),
            &self.executable,
            &identity,
            std::iter::once(&active),
            DeviceTimingMode::Off,
            &uploads,
            wave,
            &lane,
            &self.reaper,
        );
        let handle = match dispatch {
            Ok(handle) => handle,
            Err(source) => {
                return Err(TensorExecutionError::Dispatch {
                    source,
                    retirement: scope,
                    reaper: self.reaper.clone(),
                })
            }
        };
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
        drop((handle, active, identity, uploads, batch, lane));
        if let Err(source) = scope.finish_with_request(persistent_request) {
            return Err(TensorExecutionError::Retirement {
                source,
                retirement: scope,
            });
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
                })
            })
            .collect()
    }
}

/// The exact step/session ownership of a failed or nonterminal job. After
/// recovering its completion through the runtime, call try_abort to retire it.
/// This supports safe cleanup, not resuming a failed job's output production.
pub struct TensorExecutionRetirement<R: DeviceRuntime> {
    step: Option<Arc<StepResourceLease<R>>>,
    session: Option<Arc<SequenceSession<R>>>,
}

impl<R: DeviceRuntime> TensorExecutionRetirement<R> {
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
