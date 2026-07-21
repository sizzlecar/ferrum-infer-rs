use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};

use super::super::{
    DeviceCommandBatch, DeviceTerminal, HostTransferLayout, PreparedModelFamily,
    WeightComponentPayload, WeightComponentSource, WeightComponentSpec, WeightId,
};
use super::{
    defer_device_cleanup, deferred_device_cleanup_status, maintain_deferred_device_cleanups,
    new_deferred_device_cleanup_domain, BufferUsage, DeferredDeviceCleanupDisposition,
    DeferredDeviceCleanupDomainId, DeferredDeviceCleanupMaintenanceReceipt,
    DeferredDeviceCleanupStatus, DeferredDeviceCleanupTask, DeviceRuntime, ElementType,
    ExecutionPlan, FailureDomain, FailureEnvelope, PlanRuntimeHandoffError, PlanRuntimeResources,
    ResourceId, ResourceTransaction, ResourceTransactionDriver, TransactionCommitted, VNextError,
    MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS,
};

static STATIC_INITIALIZATION_CLEANUP_DOMAIN: OnceLock<DeferredDeviceCleanupDomainId> =
    OnceLock::new();

fn static_initialization_cleanup_domain() -> DeferredDeviceCleanupDomainId {
    *STATIC_INITIALIZATION_CLEANUP_DOMAIN.get_or_init(new_deferred_device_cleanup_domain)
}

/// Process-reachable status for initialization owners whose submission state
/// was indeterminate and whose explicit failure owner was dropped.
pub fn static_initialization_cleanup_status() -> DeferredDeviceCleanupStatus {
    deferred_device_cleanup_status(static_initialization_cleanup_domain())
}

/// Runs bounded recovery for abandoned static-initialization owners. This may
/// block in backend synchronization and belongs on a recovery thread.
pub fn maintain_static_initialization_cleanups(
    maximum_tasks: usize,
) -> Result<DeferredDeviceCleanupMaintenanceReceipt, VNextError> {
    if maximum_tasks == 0 || maximum_tasks > MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS {
        return Err(VNextError::InvalidExecutionPlan {
            reason: format!(
                "static initialization cleanup maintenance size must be in 1..={MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS}"
            ),
        });
    }
    Ok(maintain_deferred_device_cleanups(
        static_initialization_cleanup_domain(),
        maximum_tasks,
    ))
}

/// Explicit host-staging budget for cold plan initialization. The composition
/// root supplies this policy through typed configuration; it is not inferred
/// from a model name, GPU name, or environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticInitializationPolicy {
    maximum_staging_bytes: u64,
    maximum_commands_per_batch: usize,
}

impl StaticInitializationPolicy {
    pub fn new(
        maximum_staging_bytes: u64,
        maximum_commands_per_batch: usize,
    ) -> Result<Self, VNextError> {
        if maximum_staging_bytes == 0 || maximum_commands_per_batch == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "static initialization requires non-zero staging and command budgets"
                    .to_owned(),
            });
        }
        Ok(Self {
            maximum_staging_bytes,
            maximum_commands_per_batch,
        })
    }

    pub const fn maximum_staging_bytes(self) -> u64 {
        self.maximum_staging_bytes
    }

    pub const fn maximum_commands_per_batch(self) -> usize {
        self.maximum_commands_per_batch
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct StaticInitializationReceipt {
    initialized_resource_count: usize,
    uploaded_component_count: usize,
    uploaded_bytes: u64,
    imported_component_count: usize,
    imported_bytes: u64,
    upload_command_count: usize,
    submission_batch_count: usize,
    source_files: BTreeSet<String>,
}

impl StaticInitializationReceipt {
    pub const fn initialized_resource_count(&self) -> usize {
        self.initialized_resource_count
    }

    pub const fn uploaded_component_count(&self) -> usize {
        self.uploaded_component_count
    }

    pub const fn uploaded_bytes(&self) -> u64 {
        self.uploaded_bytes
    }

    pub const fn imported_component_count(&self) -> usize {
        self.imported_component_count
    }

    pub const fn imported_bytes(&self) -> u64 {
        self.imported_bytes
    }

    pub const fn upload_command_count(&self) -> usize {
        self.upload_command_count
    }

    pub const fn submission_batch_count(&self) -> usize {
        self.submission_batch_count
    }

    pub fn source_files(&self) -> &BTreeSet<String> {
        &self.source_files
    }
}

/// Typestate owner proving every plan-static allocation was initialized and
/// every selected weight component reached either a quiescent successful
/// upload fence or one sealed all-or-nothing import transaction.
#[must_use = "initialized static resources must be handed to the plan runtime"]
pub struct InitializedResourceTransaction<D>
where
    D: ResourceTransactionDriver,
{
    transaction: ResourceTransaction<D, TransactionCommitted>,
    receipt: StaticInitializationReceipt,
}

impl<D> InitializedResourceTransaction<D>
where
    D: ResourceTransactionDriver,
{
    pub fn receipt(&self) -> &StaticInitializationReceipt {
        &self.receipt
    }

    pub fn into_plan_runtime(
        self,
    ) -> Result<Arc<PlanRuntimeResources<D::Runtime>>, PlanRuntimeHandoffError<D>>
    where
        D: 'static,
    {
        self.transaction.into_plan_runtime()
    }
}

struct StaticInitializationRecovery<R>
where
    R: DeviceRuntime,
{
    stream: R::Stream,
    fence: Option<R::Fence>,
}

/// Failure owner for initialization. A quiescent failure can return the
/// committed transaction immediately. An indeterminate failure first requires
/// explicit stream recovery; dropping it intentionally retains all device and
/// capacity ownership rather than risking premature reuse.
#[must_use = "static initialization failure retains transaction and possibly in-flight ownership"]
pub struct StaticInitializationFailure<D>
where
    D: ResourceTransactionDriver + 'static,
{
    transaction: Option<ResourceTransaction<D, TransactionCommitted>>,
    failure: FailureEnvelope,
    recovery: Option<StaticInitializationRecovery<D::Runtime>>,
}

impl<D> fmt::Debug for StaticInitializationFailure<D>
where
    D: ResourceTransactionDriver + 'static,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StaticInitializationFailure")
            .field("failure", &self.failure)
            .field("indeterminate", &self.recovery.is_some())
            .finish_non_exhaustive()
    }
}

impl<D> fmt::Display for StaticInitializationFailure<D>
where
    D: ResourceTransactionDriver + 'static,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "static initialization failed: {}",
            self.failure.message()
        )
    }
}

impl<D> std::error::Error for StaticInitializationFailure<D> where
    D: ResourceTransactionDriver + 'static
{
}

impl<D> StaticInitializationFailure<D>
where
    D: ResourceTransactionDriver + 'static,
{
    fn new(
        transaction: ResourceTransaction<D, TransactionCommitted>,
        step: InitializationStepFailure<D::Runtime>,
    ) -> Self {
        match step {
            InitializationStepFailure::Quiescent(failure) => Self {
                transaction: Some(transaction),
                failure,
                recovery: None,
            },
            InitializationStepFailure::Indeterminate { failure, recovery } => Self {
                transaction: Some(transaction),
                failure,
                recovery: Some(recovery),
            },
        }
    }

    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub const fn is_indeterminate(&self) -> bool {
        self.recovery.is_some()
    }

    pub fn into_transaction(
        mut self,
    ) -> Result<ResourceTransaction<D, TransactionCommitted>, Self> {
        if self.recovery.is_some() {
            return Err(self);
        }
        Ok(self
            .transaction
            .take()
            .expect("static initialization failure owns its transaction"))
    }

    /// Blocks until the failed initialization stream is quiescent. Success
    /// returns the committed transaction for a complete retry from byte zero.
    pub fn recover(mut self) -> Result<ResourceTransaction<D, TransactionCommitted>, Self> {
        let Some(mut recovery) = self.recovery.take() else {
            return Ok(self
                .transaction
                .take()
                .expect("static initialization failure owns its transaction"));
        };
        let runtime = Arc::clone(
            self.transaction
                .as_ref()
                .expect("static initialization failure owns its transaction")
                .lease()
                .runtime(),
        );
        let synchronized = catch_unwind(AssertUnwindSafe(|| {
            runtime.synchronize(&mut recovery.stream)
        }));
        match synchronized {
            Ok(Ok(())) => {
                drop(recovery.fence.take());
                Ok(self
                    .transaction
                    .take()
                    .expect("static initialization failure owns its transaction"))
            }
            Ok(Err(error)) => {
                self.failure = device_failure(&runtime, &error, "static_recovery");
                self.recovery = Some(recovery);
                Err(self)
            }
            Err(payload) => {
                self.failure = portable_failure(
                    FailureDomain::Device,
                    "static_recovery_panic",
                    panic_message(payload),
                    false,
                );
                self.recovery = Some(recovery);
                Err(self)
            }
        }
    }
}

impl<D> Drop for StaticInitializationFailure<D>
where
    D: ResourceTransactionDriver + 'static,
{
    fn drop(&mut self) {
        if let Some(recovery) = self.recovery.take() {
            let transaction = self
                .transaction
                .take()
                .expect("indeterminate static initialization owns its transaction");
            defer_device_cleanup(
                static_initialization_cleanup_domain(),
                DeferredStaticInitializationCleanup {
                    transaction: Some(transaction),
                    recovery: Some(recovery),
                },
            );
        }
    }
}

struct DeferredStaticInitializationCleanup<D>
where
    D: ResourceTransactionDriver + 'static,
{
    transaction: Option<ResourceTransaction<D, TransactionCommitted>>,
    recovery: Option<StaticInitializationRecovery<D::Runtime>>,
}

impl<D> DeferredDeviceCleanupTask for DeferredStaticInitializationCleanup<D>
where
    D: ResourceTransactionDriver + 'static,
{
    fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
        let transaction = self
            .transaction
            .as_ref()
            .expect("deferred static initialization owns its transaction");
        let recovery = self
            .recovery
            .as_mut()
            .expect("deferred static initialization owns its recovery stream");
        let runtime = Arc::clone(transaction.lease().runtime());
        let synchronized = catch_unwind(AssertUnwindSafe(|| {
            runtime.synchronize(&mut recovery.stream)
        }));
        if !matches!(synchronized, Ok(Ok(()))) {
            return DeferredDeviceCleanupDisposition::Retryable;
        }
        let mut recovery = self
            .recovery
            .take()
            .expect("successful recovery retains its stream and fence");
        drop(recovery.fence.take());
        drop(recovery);
        drop(
            self.transaction
                .take()
                .expect("successful recovery retains its transaction"),
        );
        DeferredDeviceCleanupDisposition::Completed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WeightPlacement {
    component_id: WeightId,
    resource_id: ResourceId,
    offset_bytes: u64,
    length_bytes: u64,
    element_type: ElementType,
}

enum InitializationStepFailure<R>
where
    R: DeviceRuntime,
{
    Quiescent(FailureEnvelope),
    Indeterminate {
        failure: FailureEnvelope,
        recovery: StaticInitializationRecovery<R>,
    },
}

impl<D> ResourceTransaction<D, TransactionCommitted>
where
    D: ResourceTransactionDriver + 'static,
{
    pub fn initialize_static(
        self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        source: &dyn WeightComponentSource,
        policy: StaticInitializationPolicy,
    ) -> Result<InitializedResourceTransaction<D>, StaticInitializationFailure<D>> {
        match initialize_static_inner(&self, family, plan, source, policy) {
            Ok(receipt) => Ok(InitializedResourceTransaction {
                transaction: self,
                receipt,
            }),
            Err(step) => Err(StaticInitializationFailure::new(self, step)),
        }
    }
}

fn initialize_static_inner<D>(
    transaction: &ResourceTransaction<D, TransactionCommitted>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    source: &dyn WeightComponentSource,
    policy: StaticInitializationPolicy,
) -> Result<StaticInitializationReceipt, InitializationStepFailure<D::Runtime>>
where
    D: ResourceTransactionDriver,
{
    preflight_transaction(transaction, family, plan).map_err(contract_failure)?;
    let placements = weight_placements(family, plan).map_err(contract_failure)?;
    let runtime = Arc::clone(transaction.lease().runtime());
    let mut weight_import = if placements.is_empty() {
        None
    } else {
        match runtime.begin_static_weight_import() {
            None => None,
            Some(Ok(import)) => Some(import),
            Some(Err(error)) => {
                return Err(InitializationStepFailure::Quiescent(device_failure(
                    &runtime,
                    &error,
                    "static_weight_import_begin",
                )))
            }
        }
    };
    let created_stream = runtime.create_stream().map_err(|error| {
        InitializationStepFailure::Quiescent(device_failure(
            &runtime,
            &error,
            "static_stream_create",
        ))
    })?;
    let mut stream = Some(created_stream);
    let mut pending = Vec::<<D::Runtime as DeviceRuntime>::Command>::new();
    let mut pending_staging_bytes = 0_u64;
    let mut submission_batch_count = 0_usize;
    let mut upload_command_count = 0_usize;
    let mut uploaded_component_count = 0_usize;
    let mut uploaded_bytes = 0_u64;
    let mut imported_component_count = 0_usize;
    let mut imported_bytes = 0_u64;
    let mut source_files = BTreeSet::new();

    for allocation in plan.payload().memory().static_allocations() {
        if allocation.usage() == BufferUsage::Weights && weight_import.is_some() {
            continue;
        }
        let command = with_static_buffer(transaction, allocation.resource_id(), |buffer| {
            runtime.encode_zero(buffer, 0, allocation.size_bytes())
        })
        .map_err(|error| runtime_or_contract_failure(&runtime, error, "static_zero_encode"))?;
        pending.push(command);
        if pending.len() == policy.maximum_commands_per_batch() {
            submit_pending(
                &runtime,
                &mut stream,
                &mut pending,
                &mut pending_staging_bytes,
            )?;
            submission_batch_count += 1;
        }
    }

    for component in &family.weight_schema().components {
        let Some(placement) = placements.get(&component.id) else {
            continue;
        };
        // Materialize one component at a time. Format adapters may own a
        // converted payload as large as an embedding table; retaining every
        // converted component until the final submit would duplicate the
        // entire model in host memory.
        let upload = prepare_upload(source, component, placement).map_err(contract_failure)?;
        source_files.extend(upload.source_files().iter().cloned());
        if let Some(import) = weight_import.as_mut() {
            with_static_buffer(transaction, &placement.resource_id, |buffer| {
                import.import_component(&upload, buffer, placement.offset_bytes)
            })
            .map_err(|error| {
                runtime_or_contract_failure(&runtime, error, "static_weight_component_import")
            })?;
            imported_component_count += 1;
            imported_bytes = imported_bytes
                .checked_add(placement.length_bytes)
                .ok_or_else(|| {
                    contract_failure(VNextError::InvalidExecutionPlan {
                        reason: "static initialization imported bytes overflow u64".to_owned(),
                    })
                })?;
            continue;
        }
        let element_bytes = upload.element_type().size_bytes();
        let maximum_chunk_bytes =
            policy.maximum_staging_bytes() - policy.maximum_staging_bytes() % element_bytes;
        if maximum_chunk_bytes == 0 {
            return Err(contract_failure(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "static staging budget cannot hold one {:?} element",
                    upload.element_type()
                ),
            }));
        }
        let bytes = upload.bytes();
        let mut source_offset = 0_usize;
        while source_offset < bytes.len() {
            let remaining = bytes.len() - source_offset;
            let chunk_bytes =
                remaining.min(usize::try_from(maximum_chunk_bytes).map_err(|_| {
                    contract_failure(VNextError::InvalidExecutionPlan {
                        reason: "static staging budget exceeds host address space".to_owned(),
                    })
                })?);
            let chunk_bytes = chunk_bytes - chunk_bytes % element_bytes as usize;
            if chunk_bytes == 0 {
                return Err(contract_failure(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "component `{}` has a partial trailing element",
                        placement.component_id
                    ),
                }));
            }
            let chunk_bytes_u64 = chunk_bytes as u64;
            if !pending.is_empty()
                && (pending.len() == policy.maximum_commands_per_batch()
                    || pending_staging_bytes
                        .checked_add(chunk_bytes_u64)
                        .is_none_or(|bytes| bytes > policy.maximum_staging_bytes()))
            {
                submit_pending(
                    &runtime,
                    &mut stream,
                    &mut pending,
                    &mut pending_staging_bytes,
                )?;
                submission_batch_count += 1;
            }
            let source_end = source_offset + chunk_bytes;
            let destination_offset = placement
                .offset_bytes
                .checked_add(source_offset as u64)
                .ok_or_else(|| {
                    contract_failure(VNextError::InvalidExecutionPlan {
                        reason: "static upload destination offset overflows".to_owned(),
                    })
                })?;
            let layout =
                HostTransferLayout::new(upload.element_type(), chunk_bytes_u64 / element_bytes)
                    .map_err(contract_failure)?;
            let command = with_static_buffer(transaction, &placement.resource_id, |buffer| {
                runtime.encode_upload(
                    &bytes[source_offset..source_end],
                    layout,
                    buffer,
                    destination_offset,
                )
            })
            .map_err(|error| {
                runtime_or_contract_failure(&runtime, error, "static_upload_encode")
            })?;
            pending.push(command);
            pending_staging_bytes += chunk_bytes_u64;
            upload_command_count += 1;
            source_offset = source_end;
        }
        uploaded_component_count += 1;
        uploaded_bytes = uploaded_bytes
            .checked_add(placement.length_bytes)
            .ok_or_else(|| {
                contract_failure(VNextError::InvalidExecutionPlan {
                    reason: "static initialization uploaded bytes overflow u64".to_owned(),
                })
            })?;
    }

    if !pending.is_empty() {
        submit_pending(
            &runtime,
            &mut stream,
            &mut pending,
            &mut pending_staging_bytes,
        )?;
        submission_batch_count += 1;
    }
    if let Some(import) = weight_import {
        import.seal().map_err(|error| {
            InitializationStepFailure::Quiescent(device_failure(
                &runtime,
                &error,
                "static_weight_import_seal",
            ))
        })?;
    }
    Ok(StaticInitializationReceipt {
        initialized_resource_count: plan.payload().memory().static_allocations().len(),
        uploaded_component_count,
        uploaded_bytes,
        imported_component_count,
        imported_bytes,
        upload_command_count,
        submission_batch_count,
        source_files,
    })
}

fn submit_pending<R>(
    runtime: &Arc<R>,
    stream: &mut Option<R::Stream>,
    pending: &mut Vec<R::Command>,
    pending_staging_bytes: &mut u64,
) -> Result<(), InitializationStepFailure<R>>
where
    R: DeviceRuntime,
{
    debug_assert!(!pending.is_empty());
    let commands = std::mem::take(pending);
    *pending_staging_bytes = 0;
    let mut batch = DeviceCommandBatch::with_capacity(commands.len());
    for command in commands {
        batch.push_initialization(command);
    }
    let submitted = catch_unwind(AssertUnwindSafe(|| {
        runtime.submit(
            stream
                .as_mut()
                .expect("static initialization owns its stream"),
            batch,
        )
    }));
    let fence = match submitted {
        Ok(Ok(fence)) => fence,
        Ok(Err(not_submitted)) => {
            return Err(InitializationStepFailure::Quiescent(device_failure(
                runtime,
                not_submitted.error(),
                "static_submit_not_submitted",
            )))
        }
        Err(payload) => {
            return Err(InitializationStepFailure::Indeterminate {
                failure: portable_failure(
                    FailureDomain::Device,
                    "static_submit_indeterminate",
                    panic_message(payload),
                    false,
                ),
                recovery: StaticInitializationRecovery {
                    stream: stream
                        .take()
                        .expect("static initialization owns its stream"),
                    fence: None,
                },
            })
        }
    };
    let waited = catch_unwind(AssertUnwindSafe(|| runtime.wait_fence(&fence)));
    match waited {
        Ok(Ok(receipt)) => match receipt.into_parts().0 {
            DeviceTerminal::Succeeded => Ok(()),
            DeviceTerminal::FailedButQuiescent(error) => Err(InitializationStepFailure::Quiescent(
                device_failure(runtime, &error, "static_fence_failed"),
            )),
        },
        Ok(Err(indeterminate)) => Err(InitializationStepFailure::Indeterminate {
            failure: device_failure(runtime, indeterminate.error(), "static_fence_indeterminate"),
            recovery: StaticInitializationRecovery {
                stream: stream
                    .take()
                    .expect("static initialization owns its stream"),
                fence: Some(fence),
            },
        }),
        Err(payload) => Err(InitializationStepFailure::Indeterminate {
            failure: portable_failure(
                FailureDomain::Device,
                "static_fence_wait_panic",
                panic_message(payload),
                false,
            ),
            recovery: StaticInitializationRecovery {
                stream: stream
                    .take()
                    .expect("static initialization owns its stream"),
                fence: Some(fence),
            },
        }),
    }
}

fn preflight_transaction<D>(
    transaction: &ResourceTransaction<D, TransactionCommitted>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
) -> Result<(), VNextError>
where
    D: ResourceTransactionDriver,
{
    let payload = plan.payload();
    let admission = transaction.admission();
    if payload.family_id() != family.family_id()
        || payload.prepared_family_fingerprint() != family.fingerprint()?
        || admission.plan_id() != payload.plan_id()
        || admission.plan_hash() != plan.plan_hash()
        || admission.device_id() != payload.device_id()
        || admission.device_runtime_implementation_fingerprint()
            != payload.device_runtime_implementation_fingerprint()
        || transaction.lease().plan_static_entries().count()
            != payload.memory().static_allocations().len()
    {
        return Err(VNextError::InvalidExecutionPlan {
            reason: "static initialization family, plan, admission, runtime, or lease differs"
                .to_owned(),
        });
    }
    Ok(())
}

fn weight_placements(
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
) -> Result<BTreeMap<WeightId, WeightPlacement>, VNextError> {
    let schema = family
        .weight_schema()
        .components
        .iter()
        .map(|component| (&component.id, component))
        .collect::<BTreeMap<_, _>>();
    let allocations = plan
        .payload()
        .memory()
        .static_allocations()
        .iter()
        .map(|allocation| (allocation.resource_id(), allocation))
        .collect::<BTreeMap<_, _>>();
    let mut placements = BTreeMap::new();
    for node in plan.payload().nodes() {
        for binding in node
            .values()
            .iter()
            .filter(|binding| binding.usage() == BufferUsage::Weights)
        {
            for resolved in binding.storage().components() {
                let component_id =
                    resolved
                        .component_id()
                        .ok_or_else(|| VNextError::InvalidExecutionPlan {
                            reason: format!(
                                "weight resource `{}` lacks a physical component identity",
                                resolved.resource_id()
                            ),
                        })?;
                let component =
                    schema
                        .get(component_id)
                        .ok_or_else(|| VNextError::InvalidExecutionPlan {
                            reason: format!("plan binds unknown weight component `{component_id}`"),
                        })?;
                let placement = WeightPlacement {
                    component_id: component_id.clone(),
                    resource_id: resolved.resource_id().clone(),
                    offset_bytes: resolved.offset_bytes(),
                    length_bytes: resolved.length_bytes(),
                    element_type: resolved.element_type(),
                };
                if placement.length_bytes != component.physical_bytes()?
                    || placement.element_type != component.physical_element_type()
                {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "weight component `{component_id}` placement differs from its physical schema"
                        ),
                    });
                }
                match placements.get(component_id) {
                    Some(existing) if existing != &placement => {
                        return Err(VNextError::InvalidExecutionPlan {
                            reason: format!(
                                "weight component `{component_id}` has inconsistent placements"
                            ),
                        })
                    }
                    Some(_) => {}
                    None => {
                        placements.insert(component_id.clone(), placement);
                    }
                }
            }
        }
    }
    for component in family.weight_schema().components.iter() {
        if component.required && !placements.contains_key(&component.id) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "required weight component `{}` has no plan placement",
                    component.id
                ),
            });
        }
    }
    let mut ranges = BTreeMap::<ResourceId, Vec<(u64, u64, WeightId)>>::new();
    for placement in placements.values() {
        let allocation = allocations.get(&placement.resource_id).ok_or_else(|| {
            VNextError::InvalidExecutionPlan {
                reason: format!(
                    "weight component `{}` references a non-static resource",
                    placement.component_id
                ),
            }
        })?;
        let end = placement
            .offset_bytes
            .checked_add(placement.length_bytes)
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: "weight placement range overflows u64".to_owned(),
            })?;
        if allocation.usage() != BufferUsage::Weights
            || allocation.element_type() != placement.element_type
            || end > allocation.size_bytes()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "weight component `{}` placement exceeds or differs from its allocation",
                    placement.component_id
                ),
            });
        }
        ranges
            .entry(placement.resource_id.clone())
            .or_default()
            .push((placement.offset_bytes, end, placement.component_id.clone()));
    }
    for (resource_id, ranges) in &mut ranges {
        ranges.sort();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("weight placements overlap in resource `{resource_id}`"),
            });
        }
    }
    if allocations
        .values()
        .filter(|allocation| allocation.usage() == BufferUsage::Weights)
        .any(|allocation| !ranges.contains_key(allocation.resource_id()))
    {
        return Err(VNextError::InvalidExecutionPlan {
            reason: "a static weight allocation has no schema component placement".to_owned(),
        });
    }
    Ok(placements)
}

fn prepare_upload<'source>(
    source: &'source dyn WeightComponentSource,
    component: &WeightComponentSpec,
    placement: &WeightPlacement,
) -> Result<WeightComponentPayload<'source>, VNextError> {
    let payload = source.component(component)?;
    if payload.component_id() != &placement.component_id
        || payload.element_type() != placement.element_type
        || payload.bytes().len() as u64 != placement.length_bytes
    {
        return Err(VNextError::InvalidExecutionPlan {
            reason: format!(
                "weight source payload for `{}` differs from its selected placement",
                placement.component_id
            ),
        });
    }
    Ok(payload)
}

enum StaticBufferAccessError<E> {
    Contract(VNextError),
    Runtime(E),
}

fn with_static_buffer<D, T>(
    transaction: &ResourceTransaction<D, TransactionCommitted>,
    resource_id: &ResourceId,
    action: impl FnOnce(&D::Buffer) -> Result<T, <D::Runtime as DeviceRuntime>::Error>,
) -> Result<T, StaticBufferAccessError<<D::Runtime as DeviceRuntime>::Error>>
where
    D: ResourceTransactionDriver,
{
    let lease = transaction.lease();
    let entry = lease
        .plan_static_entries()
        .find(|entry| entry.resource_id() == resource_id)
        .ok_or_else(|| {
            StaticBufferAccessError::Contract(VNextError::InvalidExecutionPlan {
                reason: format!("static lease lacks resource `{resource_id}`"),
            })
        })?;
    let view = lease
        .view(resource_id, entry.generation())
        .map_err(StaticBufferAccessError::Contract)?;
    action(view.buffer()).map_err(StaticBufferAccessError::Runtime)
}

fn runtime_or_contract_failure<R>(
    runtime: &Arc<R>,
    error: StaticBufferAccessError<R::Error>,
    code: &'static str,
) -> InitializationStepFailure<R>
where
    R: DeviceRuntime,
{
    InitializationStepFailure::Quiescent(match error {
        StaticBufferAccessError::Contract(error) => resource_failure(code, error),
        StaticBufferAccessError::Runtime(error) => device_failure(runtime, &error, code),
    })
}

fn contract_failure<R>(error: VNextError) -> InitializationStepFailure<R>
where
    R: DeviceRuntime,
{
    InitializationStepFailure::Quiescent(resource_failure("static_contract", error))
}

fn resource_failure(code: &'static str, error: impl fmt::Display) -> FailureEnvelope {
    portable_failure(FailureDomain::Resource, code, error, false)
}

fn device_failure<R>(
    runtime: &Arc<R>,
    error: &R::Error,
    fallback_code: &'static str,
) -> FailureEnvelope
where
    R: DeviceRuntime,
{
    match catch_unwind(AssertUnwindSafe(|| runtime.describe_error(error))) {
        Ok(Ok(report)) => portable_failure(
            FailureDomain::Device,
            report.code(),
            report.message(),
            report.retryable(),
        ),
        Ok(Err(classification)) => portable_failure(
            FailureDomain::Device,
            fallback_code,
            format!("{error}; error classification failed: {classification}"),
            false,
        ),
        Err(payload) => portable_failure(
            FailureDomain::Device,
            fallback_code,
            format!(
                "{error}; error classification panicked: {}",
                panic_message(payload)
            ),
            false,
        ),
    }
}

fn portable_failure(
    domain: FailureDomain,
    code: impl Into<String>,
    message: impl fmt::Display,
    retryable: bool,
) -> FailureEnvelope {
    let mut code = code.into();
    code.retain(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
    });
    code.truncate(64);
    if code.is_empty() {
        code.push_str("static_initialization");
    }
    let mut message = message
        .to_string()
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .take(1024)
        .collect::<String>();
    if message.trim().is_empty() {
        message.push_str("static initialization failed");
    }
    FailureEnvelope::new(domain, code, message, retryable)
        .expect("static initialization failure metadata is bounded and portable")
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "device runtime panicked during static initialization submission".to_owned()
    }
}
