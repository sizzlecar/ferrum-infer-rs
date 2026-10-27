//! Reusable fixed-tensor component lifecycle over the existing runtime owners.
//!
//! This module neither constructs a backend nor creates a memory account. The
//! composition root supplies one runtime and Host account for all components.
//! A session initializes its static resources once and can execute repeatedly.
//! It is not a diffusion scheduler, a model loader, or a token executor.

use std::{collections::BTreeSet, fmt, sync::Arc};

use ferrum_runtime::*;
use serde::{Deserialize, Serialize};

use crate::{
    PreparedTensorExecutor, TensorExecutionError, TensorExecutionOutput, TensorExecutorConfig,
};

/// Single-participant, ordinary fixed-tensor execution. Capacity is a policy
/// ceiling consumed by the existing device account, not a new allocation ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorComponentPolicy {
    capacity_bytes: u64,
    storage_profiles: Vec<DynamicStorageProfile>,
}

impl<'de> Deserialize<'de> for TensorComponentPolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            capacity_bytes: u64,
            storage_profiles: Vec<DynamicStorageProfile>,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.capacity_bytes, wire.storage_profiles).map_err(serde::de::Error::custom)
    }
}

impl TensorComponentPolicy {
    pub fn new(
        capacity_bytes: u64,
        storage_profiles: Vec<DynamicStorageProfile>,
    ) -> Result<Self, VNextError> {
        let result = Self {
            capacity_bytes,
            storage_profiles,
        };
        result.validate()?;
        Ok(result)
    }
}

impl RuntimePolicy for TensorComponentPolicy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }
    fn memory_capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }
    fn memory_reserve_bytes(&self) -> u64 {
        0
    }
    fn maximum_active_sequences(&self) -> u32 {
        1
    }
    fn work_limits(&self) -> RuntimeWorkLimits {
        RuntimeWorkLimits::FixedTensor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn execution_determinism_requirement(&self) -> ExecutionDeterminismRequirement {
        ExecutionDeterminismRequirement::BitwiseSameRuntime
    }
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.storage_profiles
    }
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy> {
        None
    }
    fn validate(&self) -> Result<(), VNextError> {
        if self.capacity_bytes == 0
            || self.storage_profiles.is_empty()
            || self.storage_profiles.iter().collect::<BTreeSet<_>>().len()
                != self.storage_profiles.len()
        {
            return Err(invalid(
                "component policy needs positive capacity and distinct storage profiles",
            ));
        }
        Ok(())
    }
}

/// Absence of weights is explicit. It does not imply absence of plan-static
/// persistent buffers; those still take the normal zero-initialization path.
#[derive(Clone, Copy)]
pub enum ComponentWeightSource<'a> {
    None,
    Stored(&'a dyn WeightComponentSource),
}

impl ComponentWeightSource<'_> {
    fn validate(self, weights: &ModelWeights) -> Result<(), VNextError> {
        match (self, weights) {
            (Self::None, ModelWeights::None) | (Self::Stored(_), ModelWeights::Stored(_)) => Ok(()),
            _ => Err(invalid(
                "component source must match the prepared None/Stored weight kind",
            )),
        }
    }
}

/// Explicit cold-initialization identity and finite work budgets. The static
/// policy also applies to weightless components that have static workspaces.
#[derive(Debug, Clone)]
pub struct ComponentInitializationConfig {
    pub request_id: RequestIdentity,
    pub run_id: RunId,
    pub transaction_id: TransactionId,
    pub static_policy: StaticInitializationPolicy,
    pub executor: TensorExecutorConfig,
}

/// A receipt for completed initialization, separate from eventual plan close.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentInitializationReceipt {
    NoStatic,
    Initialized {
        receipt: StaticInitializationReceipt,
    },
    SharedWeights {
        evidence: SharedComponentWeightsEvidence,
    },
}

/// Compilation and provider binding happen before physical provisioning. The
/// registry remains a caller-owned capability selection; no model/backend
/// names are interpreted here. Repeated initialization reuses compilation and
/// provider authority; each session independently admits all runtime resources.
pub struct CompiledTensorComponent<R: DeviceRuntime> {
    family: PreparedModelFamily,
    executable: ExecutablePlan,
    providers: BoundOperationProviderSet<R>,
}

impl<R: DeviceRuntime> CompiledTensorComponent<R> {
    pub fn compile(
        family: PreparedModelFamily,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        policy: &TensorComponentPolicy,
        materializers: &WeightMaterializerRegistry,
        options: &ProgramPlanCompileOptions,
    ) -> Result<Self, VNextError> {
        if !matches!(family.metadata(), ModelSemanticMetadata::TensorComponent) {
            return Err(invalid(
                "component sessions require TensorComponent semantics",
            ));
        }
        let mut options = options.clone();
        // Completion retention extends storage lifetime; only the family's
        // declared outputs become public ports.
        for output in family.program().outputs() {
            options.retain_completion_value(output.clone());
        }
        let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
            &family,
            catalog,
            policy,
            &registry.planning(),
            materializers,
            &options,
        )?;
        let (executable, _, _, _) = compilation.into_parts();
        let providers = registry.bind_plan(&executable)?;
        Ok(Self {
            family,
            executable,
            providers,
        })
    }

    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }
    pub fn executable(&self) -> &ExecutablePlan {
        &self.executable
    }

    /// Stored weights use managed TransferStaging bytes from `host`. Source
    /// mmap/materializer backing remains owned by its source, and is not a
    /// claim that this staging account bounds total process RSS.
    pub fn initialize(
        &self,
        runtime: Arc<R>,
        host: HostMemoryAccount,
        source: ComponentWeightSource<'_>,
        config: ComponentInitializationConfig,
    ) -> Result<TensorComponentSession<R>, ComponentInitializationError<R>> {
        let (resources, receipt, _) = initialize_resources(
            runtime,
            &self.family,
            self.executable.execution_plan(),
            source,
            &host,
            &config,
            false,
        )?;
        self.create_session(resources, receipt, host, config.executor)
    }

    /// Cold initialize once, then explicitly retain this exact immutable
    /// installation for compatible plans. No path/content cache is consulted.
    pub fn initialize_shared_weights(
        &self,
        runtime: Arc<R>,
        host: HostMemoryAccount,
        source: ComponentWeightSource<'_>,
        config: ComponentInitializationConfig,
    ) -> Result<
        (TensorComponentSession<R>, SharedComponentWeights<R>),
        ComponentInitializationError<R>,
    > {
        SharedComponentWeights::<R>::validate_plan(self.executable.execution_plan())?;
        let (resources, receipt, weights) = initialize_resources(
            runtime,
            &self.family,
            self.executable.execution_plan(),
            source,
            &host,
            &config,
            true,
        )?;
        let weights = weights.expect("shared initialization returns its sealed owner");
        self.create_session(resources, receipt, host, config.executor)
            .map(|session| (session, weights))
    }

    /// Attach an existing installation; deliberately takes no checkpoint source.
    /// The new plan has independent dynamic pools and uses the same device ledger.
    pub fn initialize_with_shared_weights(
        &self,
        runtime: Arc<R>,
        host: HostMemoryAccount,
        weights: &SharedComponentWeights<R>,
        config: ComponentInitializationConfig,
    ) -> Result<TensorComponentSession<R>, ComponentInitializationError<R>> {
        let resources =
            weights.bind_plan(self.executable.execution_plan(), runtime, config.request_id)?;
        let receipt = ComponentInitializationReceipt::SharedWeights {
            evidence: weights.evidence().clone(),
        };
        // As for NoStatic plans, backing starts empty and grows through the
        // executor's existing bounded admission/maintenance path. This avoids
        // eager native allocations while merely attaching a weight snapshot.
        self.create_session(resources, receipt, host, config.executor)
    }

    fn create_session(
        &self,
        resources: Arc<PlanRuntimeResources<R>>,
        receipt: ComponentInitializationReceipt,
        host: HostMemoryAccount,
        config: TensorExecutorConfig,
    ) -> Result<TensorComponentSession<R>, ComponentInitializationError<R>> {
        match PreparedTensorExecutor::new(
            self.executable.clone(),
            self.providers.clone(),
            resources.clone(),
            host,
            config,
        ) {
            Ok(executor) => Ok(TensorComponentSession {
                executor: Some(executor),
                resources: Some(resources),
                executable: self.executable.clone(),
                receipt,
            }),
            Err(source) => Err(ComponentInitializationError::Executor {
                source,
                resources,
                receipt,
            }),
        }
    }
}

/// Errors keep their original transaction, fence/recovery handle, or root.
/// Callers can match a variant to use the runtime's existing recovery APIs.
/// Dropping an error invokes those owners' safe abandonment behavior; it is
/// never evidence of successful cleanup.
#[must_use = "initialization errors may retain resource or recovery ownership"]
pub enum ComponentInitializationError<R: DeviceRuntime> {
    Contract(VNextError),
    Reserve(ResourcePrepareTransitionError<RuntimeResourceDriver<R>, TransactionNew>),
    Commit(ResourceCommitTransitionError<RuntimeResourceDriver<R>>),
    Pool {
        source: VNextError,
        pool_id: DynamicBackingPoolId,
        transaction: ResourceTransaction<RuntimeResourceDriver<R>, TransactionCommitted>,
    },
    Static(StaticInitializationFailure<RuntimeResourceDriver<R>>),
    Handoff {
        source: StaticInitializationFailure<RuntimeResourceDriver<R>>,
        receipt: ComponentInitializationReceipt,
    },
    Executor {
        source: TensorExecutionError<R>,
        resources: Arc<PlanRuntimeResources<R>>,
        receipt: ComponentInitializationReceipt,
    },
}

impl<R: DeviceRuntime> From<VNextError> for ComponentInitializationError<R> {
    fn from(error: VNextError) -> Self {
        Self::Contract(error)
    }
}
impl<R: DeviceRuntime> fmt::Display for ComponentInitializationError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(e) => write!(f, "component preparation: {e}"),
            Self::Reserve(e) => write!(f, "component reserve: {:?}", e.failure()),
            Self::Commit(ResourceCommitTransitionError::Recoverable(e)) => {
                write!(f, "component commit: {:?}", e.failure())
            }
            Self::Commit(ResourceCommitTransitionError::Poisoned(e)) => {
                write!(f, "component commit indeterminate: {:?}", e.failure())
            }
            Self::Pool { source, .. } => write!(f, "component pool initialization: {source}"),
            Self::Static(e) => write!(f, "component static initialization: {e}"),
            Self::Handoff { source, .. } => {
                write!(f, "component initialized runtime handoff: {source}")
            }
            Self::Executor { source, .. } => write!(f, "component executor construction: {source}"),
        }
    }
}
impl<R: DeviceRuntime> fmt::Debug for ComponentInitializationError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for ComponentInitializationError<R> {}

fn initialize_resources<R: DeviceRuntime>(
    runtime: Arc<R>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    source: ComponentWeightSource<'_>,
    host: &HostMemoryAccount,
    config: &ComponentInitializationConfig,
    share_weights: bool,
) -> Result<
    (
        Arc<PlanRuntimeResources<R>>,
        ComponentInitializationReceipt,
        Option<SharedComponentWeights<R>>,
    ),
    ComponentInitializationError<R>,
> {
    source.validate(family.weights())?;
    let driver = RuntimeResourceDriver::new(runtime.clone())?;
    let provisioning = plan
        .provision_static(runtime, config.request_id.clone())?
        .into_provisioning();
    let permit = match provisioning {
        StaticProvisioning::NoStatic(provisioned) => {
            return Ok((
                provisioned.into_plan_runtime(),
                ComponentInitializationReceipt::NoStatic,
                None,
            ));
        }
        StaticProvisioning::Required(permit) => permit,
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        config.run_id.clone(),
        config.transaction_id.clone(),
    );
    let reserved = ResourceTransaction::begin(driver, identity, permit)?
        .reserve()
        .map_err(ComponentInitializationError::Reserve)?;
    let committed = reserved
        .commit()
        .map_err(ComponentInitializationError::Commit)?;
    let pools = committed
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool in pools {
        if let Err(source) = committed.maintenance_controller().initialize_pool(&pool) {
            return Err(ComponentInitializationError::Pool {
                source,
                pool_id: pool,
                transaction: committed,
            });
        }
    }
    let initialized = match source {
        ComponentWeightSource::None => {
            committed.initialize_static_without_weights(family, plan, config.static_policy)
        }
        ComponentWeightSource::Stored(source) => {
            committed.initialize_static_managed(family, plan, source, host, config.static_policy)
        }
    }
    .map_err(ComponentInitializationError::Static)?;
    let static_receipt = initialized.receipt();
    tracing::debug!(
        target: "ferrum_execution::component",
        run_id = %config.run_id,
        request_id = %config.request_id,
        initialized_resource_count = static_receipt.initialized_resource_count(),
        uploaded_component_count = static_receipt.uploaded_component_count(),
        uploaded_bytes = static_receipt.uploaded_bytes(),
        upload_command_count = static_receipt.upload_command_count(),
        transform_command_count = static_receipt.transform_command_count(),
        submission_batch_count = static_receipt.submission_batch_count(),
        total_duration_us = static_receipt.total_duration_us(),
        setup_duration_us = static_receipt.setup_duration_us(),
        source_materialization_duration_us = static_receipt.source_materialization_duration_us(),
        device_encode_duration_us = static_receipt.device_encode_duration_us(),
        device_import_duration_us = static_receipt.device_import_duration_us(),
        submission_wait_duration_us = static_receipt.submission_wait_duration_us(),
        import_seal_duration_us = static_receipt.import_seal_duration_us(),
        "tensor component static initialization completed"
    );
    let receipt = ComponentInitializationReceipt::Initialized {
        receipt: static_receipt.clone(),
    };
    if share_weights {
        match initialized.into_shared_plan_runtime(plan) {
            Ok((resources, weights)) => Ok((resources, receipt, Some(weights))),
            Err(source) => Err(ComponentInitializationError::Handoff { source, receipt }),
        }
    } else {
        match initialized.into_plan_runtime() {
            Ok(resources) => Ok((resources, receipt, None)),
            Err(source) => Err(ComponentInitializationError::Handoff { source, receipt }),
        }
    }
}

/// One active component. Input bytes must already belong to its injected Host
/// account; the executor validates provenance, class, exact ports and shape.
/// Static resources are reused across executions and released on close.
pub struct TensorComponentSession<R: DeviceRuntime> {
    // Drop the executor before the root, including its binding and reaper.
    executor: Option<PreparedTensorExecutor<R>>,
    resources: Option<Arc<PlanRuntimeResources<R>>>,
    executable: ExecutablePlan,
    receipt: ComponentInitializationReceipt,
}

impl<R: DeviceRuntime> TensorComponentSession<R> {
    pub fn begin_request(
        &self,
        run_id: RunId,
        request_id: RequestIdentity,
        initial_inputs: Vec<ManagedHostTensor>,
    ) -> Result<crate::TensorRequestSession<R>, TensorExecutionError<R>> {
        self.executor
            .as_ref()
            .expect("live session owns executor")
            .begin_request(run_id, request_id, initial_inputs)
    }

    pub fn begin_request_with_control(
        &self,
        run_id: RunId,
        request_id: RequestIdentity,
        initial_inputs: Vec<ManagedHostTensor>,
        control: Option<&crate::TensorAdmissionControl>,
    ) -> Result<crate::TensorRequestSession<R>, TensorExecutionError<R>> {
        self.executor
            .as_ref()
            .expect("live session owns executor")
            .begin_request_with_control(run_id, request_id, initial_inputs, control)
    }

    pub fn execute_initializing_request(
        &mut self,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
        destination: &mut crate::TensorRequestSession<R>,
        mappings: &[TensorRequestStateMapping],
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        self.executor
            .as_mut()
            .expect("live session owns executor")
            .execute_initializing_request(run_id, request_id, inputs, destination, mappings)
    }

    pub fn execute_initializing_request_on_actor(
        &mut self,
        actor: &mut crate::TensorSerialActor<R>,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
        destination: &mut crate::TensorRequestSession<R>,
        mappings: &[TensorRequestStateMapping],
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        self.executor
            .as_mut()
            .expect("live session owns executor")
            .execute_initializing_request_on_actor(
                actor,
                run_id,
                request_id,
                inputs,
                destination,
                mappings,
            )
    }
    pub fn executable(&self) -> &ExecutablePlan {
        &self.executable
    }
    pub fn initialization_receipt(&self) -> &ComponentInitializationReceipt {
        &self.receipt
    }
    pub fn host_account(&self) -> &HostMemoryAccount {
        self.executor
            .as_ref()
            .expect("live session owns executor")
            .host_account()
    }
    pub fn completion_reaper(&self) -> &Arc<CompletionReaper<R>> {
        self.executor
            .as_ref()
            .expect("live session owns executor")
            .completion_reaper()
    }
    pub fn execute(
        &mut self,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        self.executor
            .as_mut()
            .expect("live session owns executor")
            .execute(run_id, request_id, inputs)
    }

    pub fn execute_on_actor(
        &mut self,
        actor: &mut crate::TensorSerialActor<R>,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>> {
        self.executor
            .as_mut()
            .expect("live session owns executor")
            .execute_on_actor(actor, run_id, request_id, inputs)
    }

    /// Return idle activation/workspace backing while retaining initialized
    /// static weights and the exact compiled plan. Busy preserves all owners;
    /// successful release does not close the session. Subsequent execution
    /// reacquires dynamic capacity through ordinary runtime admission.
    pub fn try_release_idle_dynamic_backing(
        &mut self,
    ) -> Result<IdleDynamicReleaseOutcome, VNextError> {
        self.resources
            .as_ref()
            .expect("live session owns resource root")
            .try_release_idle_dynamic_backing()
    }

    /// Pool-aware planning evidence for this session's exact compiled plan.
    /// Includes the runnable minima of cold pools alongside actual retained
    /// pools; it does not reserve capacity or grant another execution.
    pub fn minimum_runnable_dynamic_bytes_with_residency(&self) -> Result<u64, VNextError> {
        self.resources
            .as_ref()
            .expect("live session owns resource root")
            .minimum_runnable_dynamic_bytes_with_residency()
    }

    /// Keep fully idle dynamic backing within the explicit byte bound, with
    /// its original device capacity grants. Zero or an exceeded bound releases
    /// all idle backing. Busy preserves the session and every resource owner.
    pub fn try_retain_idle_dynamic_backing(
        &mut self,
        maximum_bytes: u64,
    ) -> Result<IdleDynamicBackingOutcome, VNextError> {
        self.resources
            .as_ref()
            .expect("live session owns resource root")
            .try_retain_idle_dynamic_backing(maximum_bytes)
    }

    /// Retain largest fully idle chunks within the cap, with their original
    /// capacity grants. The receipt also describes any released chunks.
    pub fn try_retain_largest_idle_dynamic_chunks(
        &mut self,
        maximum_bytes: u64,
    ) -> Result<IdleDynamicBackingOutcome, VNextError> {
        self.resources
            .as_ref()
            .expect("live session owns resource root")
            .try_retain_largest_idle_dynamic_chunks(maximum_bytes)
    }

    /// Stops further admission and returns the actual runtime close outcome.
    /// `Referenced` preserves the root for a retry after outstanding execution
    /// errors/completion handles retire. Host outputs may outlive this receipt.
    pub fn close(mut self) -> Result<PlanRuntimeCloseOutcome<R>, PlanRuntimeCloseFailure<R>> {
        drop(self.executor.take());
        PlanRuntimeResources::close(
            self.resources
                .take()
                .expect("live session owns resource root"),
        )
    }
}

impl<R: DeviceRuntime> Drop for TensorComponentSession<R> {
    fn drop(&mut self) {
        drop(self.executor.take());
        if let Some(resources) = self.resources.take() {
            // No recovery loop or invented success receipt. Unknown-fence
            // owners and close failures retain/abandon through runtime rules.
            drop(PlanRuntimeResources::close(resources));
        }
    }
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
