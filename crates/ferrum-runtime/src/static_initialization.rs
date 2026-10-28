use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::{
    defer_device_cleanup, deferred_device_cleanup_status, maintain_deferred_device_cleanups,
    new_deferred_device_cleanup_domain, BufferUsage, DeferredDeviceCleanupDisposition,
    DeferredDeviceCleanupDomainId, DeferredDeviceCleanupMaintenanceReceipt,
    DeferredDeviceCleanupStatus, DeferredDeviceCleanupTask, DeviceRuntime, ElementType,
    ExecutionPlan, ExecutionWeights, FailureDomain, FailureEnvelope, HostMemoryAccount,
    HostMemoryClass, HostMemoryError, HostMemoryReservation, ManagedHostBytes, ModelWeights,
    PlanRuntimeResources, ResourceId, ResourceTransaction, ResourceTransactionDriver,
    TransactionCommitted, VNextError, MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS,
};
use super::{
    DeviceCommandBatch, DeviceTerminal, HostTransferLayout, PreparedModelFamily,
    StaticWeightTransformDestination, StaticWeightTransformPlan, StaticWeightTransformRequest,
    WeightComponentPayload, WeightComponentSegments, WeightComponentSource, WeightComponentSpec,
    WeightId,
};

static STATIC_INITIALIZATION_CLEANUP_DOMAIN: OnceLock<DeferredDeviceCleanupDomainId> =
    OnceLock::new();

pub(crate) fn static_initialization_cleanup_domain() -> DeferredDeviceCleanupDomainId {
    *STATIC_INITIALIZATION_CLEANUP_DOMAIN.get_or_init(new_deferred_device_cleanup_domain)
}

/// Process-reachable status for abandoned static-provision owners, including
/// failed reservation/allocation and indeterminate initialization retirement.
pub fn static_initialization_cleanup_status() -> DeferredDeviceCleanupStatus {
    deferred_device_cleanup_status(static_initialization_cleanup_domain())
}

/// Runs bounded recovery for abandoned static-provision owners. This may
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

/// Scheduling metadata for one blocking operation. Its streams have retired
/// on success; an error may retain recovery owners and must stop normal work.
/// This declaration is not a capacity grant or permission to create a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct BlockingStreamRequirement {
    maximum_live_streams: usize,
}

impl BlockingStreamRequirement {
    pub const fn new(maximum_live_streams: usize) -> Self {
        Self {
            maximum_live_streams,
        }
    }

    pub const fn maximum_live_streams(self) -> usize {
        self.maximum_live_streams
    }
}

impl StaticInitializationPolicy {
    /// Initialization owns one upload stream, waits every submission batch,
    /// and drops that stream before returning initialized resources.
    pub const fn stream_requirement(self) -> BlockingStreamRequirement {
        BlockingStreamRequirement::new(1)
    }

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
    transformed_component_count: usize,
    transformed_bytes: u64,
    transform_command_count: usize,
    upload_command_count: usize,
    submission_batch_count: usize,
    total_duration_us: u64,
    setup_duration_us: u64,
    source_materialization_duration_us: u64,
    device_encode_duration_us: u64,
    device_import_duration_us: u64,
    device_transform_encode_duration_us: u64,
    submission_wait_duration_us: u64,
    import_seal_duration_us: u64,
    slowest_component_id: Option<WeightId>,
    slowest_component_materialization_duration_us: u64,
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

    pub const fn transformed_component_count(&self) -> usize {
        self.transformed_component_count
    }

    pub const fn transformed_bytes(&self) -> u64 {
        self.transformed_bytes
    }

    pub const fn transform_command_count(&self) -> usize {
        self.transform_command_count
    }

    pub const fn upload_command_count(&self) -> usize {
        self.upload_command_count
    }

    pub const fn submission_batch_count(&self) -> usize {
        self.submission_batch_count
    }

    pub const fn total_duration_us(&self) -> u64 {
        self.total_duration_us
    }

    pub const fn setup_duration_us(&self) -> u64 {
        self.setup_duration_us
    }

    pub const fn source_materialization_duration_us(&self) -> u64 {
        self.source_materialization_duration_us
    }

    pub const fn device_encode_duration_us(&self) -> u64 {
        self.device_encode_duration_us
    }

    pub const fn device_import_duration_us(&self) -> u64 {
        self.device_import_duration_us
    }

    pub const fn device_transform_encode_duration_us(&self) -> u64 {
        self.device_transform_encode_duration_us
    }

    pub const fn submission_wait_duration_us(&self) -> u64 {
        self.submission_wait_duration_us
    }

    pub const fn import_seal_duration_us(&self) -> u64 {
        self.import_seal_duration_us
    }

    pub fn slowest_component_id(&self) -> Option<&WeightId> {
        self.slowest_component_id.as_ref()
    }

    pub const fn slowest_component_materialization_duration_us(&self) -> u64 {
        self.slowest_component_materialization_duration_us
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

    /// Activation is still cold: failed handoff retains the initialized,
    /// quiescent transaction and retires it through its original release ledger.
    pub fn into_plan_runtime(
        self,
    ) -> Result<Arc<PlanRuntimeResources<D::Runtime>>, StaticInitializationFailure<D>>
    where
        D: 'static,
    {
        self.transaction.into_plan_runtime().map_err(|error| {
            let failure = contract_failure(error.error().clone());
            StaticInitializationFailure::new(error.into_transaction(), failure)
        })
    }

    /// Publish exactly this successfully initialized installation for immutable
    /// reuse by compatible fixed-tensor plans. Original allocation, transaction,
    /// generation and capacity ownership remain intact.
    pub fn into_shared_plan_runtime(
        self,
        plan: &ExecutionPlan,
    ) -> Result<
        (
            Arc<PlanRuntimeResources<D::Runtime>>,
            super::SharedComponentWeights<D::Runtime>,
        ),
        StaticInitializationFailure<D>,
    >
    where
        D: 'static,
    {
        let contract = match super::resource::SharedWeightContract::from_plan(plan) {
            Ok(contract) => contract,
            Err(error) => return Err(self.abort_activation(error)),
        };
        if self.transaction.admission().plan_id() != plan.payload().plan_id()
            || self.transaction.admission().plan_hash() != plan.plan_hash()
        {
            return Err(self.abort_activation(VNextError::InvalidExecutionPlan {
                reason: "shared weight publication differs from the initialized plan".to_owned(),
            }));
        }
        self.into_plan_runtime()
            .map(|resources| PlanRuntimeResources::publish_initialized_weights(resources, contract))
    }

    /// Stop after successful initialization when a cold activation check fails.
    /// Only this initialized owner can supply the quiescence proof; callers do
    /// not manufacture a stream, fence or a permission to release a raw lease.
    pub fn abort_activation(self, error: VNextError) -> StaticInitializationFailure<D>
    where
        D: 'static,
    {
        StaticInitializationFailure::new(self.transaction, contract_failure(error))
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
    retirement_started: bool,
    retirement_poison: Option<FailureEnvelope>,
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
                retirement_started: false,
                retirement_poison: None,
            },
            InitializationStepFailure::Indeterminate { failure, recovery } => Self {
                transaction: Some(transaction),
                failure,
                recovery: Some(recovery),
                retirement_started: false,
                retirement_poison: None,
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
        if self.recovery.is_some() || self.retirement_started {
            return Err(self);
        }
        Ok(self
            .transaction
            .take()
            .expect("static initialization failure owns its transaction"))
    }

    /// Retire the original failed initialization without replaying any upload.
    ///
    /// A call makes at most one synchronization attempt. Failure retains this
    /// same owner in place for an explicit retry; success proves quiescence and
    /// releases the original transaction through its actual release ledger.
    /// Once retirement starts, this object cannot return executable resources.
    /// Backend release panics leave the remaining owner poisoned and retained:
    /// their unknown side effects are not a permission to replay release hooks.
    pub fn retire_in_place(&mut self) -> Result<(), FailureEnvelope> {
        self.retirement_started = true;
        if let Some(failure) = &self.retirement_poison {
            return Err(failure.clone());
        }
        let Some(transaction) = self.transaction.as_mut() else {
            return Ok(());
        };
        if let Some(recovery) = self.recovery.as_mut() {
            let runtime = Arc::clone(transaction.lease().runtime());
            let synchronized = catch_unwind(AssertUnwindSafe(|| {
                runtime.synchronize(&mut recovery.stream)
            }));
            match synchronized {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    return Err(device_failure(&runtime, &error, "static_retirement"))
                }
                Err(payload) => {
                    return Err(portable_failure(
                        FailureDomain::Device,
                        "static_retirement_sync_panic",
                        panic_message(payload),
                        false,
                    ))
                }
            }
            // The successful synchronization precedes releasing fence, stream,
            // buffers or capacity. The original failure's cause is unchanged.
            self.recovery.take();
        }
        match catch_unwind(AssertUnwindSafe(|| {
            transaction.release_quiescent_in_place()
        })) {
            Ok(Ok(())) => {}
            Ok(Err(failure)) => return Err(failure.failure().clone()),
            Err(payload) => {
                let failure = portable_failure(
                    FailureDomain::Resource,
                    "static_retirement_release_panic",
                    panic_message(payload),
                    false,
                );
                self.retirement_poison = Some(failure.clone());
                return Err(failure);
            }
        }
        // Every ledger entry is already Released. This typestate transition
        // runs no backend callback and cannot repeat a release prefix.
        let transaction = self
            .transaction
            .take()
            .expect("retirement owns its transaction");
        let released = transaction
            .release()
            .expect("checked release completed every entry");
        drop(released);
        Ok(())
    }

    pub fn is_retired(&self) -> bool {
        self.retirement_started && self.transaction.is_none()
    }

    /// Blocks until the failed initialization stream is quiescent. Success
    /// returns the committed transaction for a complete retry from byte zero.
    pub fn recover(mut self) -> Result<ResourceTransaction<D, TransactionCommitted>, Self> {
        if self.retirement_started {
            return Err(self);
        }
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
        // A fresh quiescent failure needs no background task when its original
        // checked release succeeds. Err/panic preserves the full remaining
        // owner below; never fall through to raw transaction abandonment.
        if !self.retirement_started && self.recovery.is_none() {
            let _ = self.retire_in_place();
        }
        if let Some(transaction) = self.transaction.take() {
            // Preserve the actual state even after quiescence was proved but a
            // release failed/panicked. No fabricated stream/fence is needed.
            let original = StaticInitializationFailure {
                transaction: Some(transaction),
                failure: self.failure.clone(),
                recovery: self.recovery.take(),
                retirement_started: self.retirement_started,
                retirement_poison: self.retirement_poison.take(),
            };
            defer_device_cleanup(
                static_initialization_cleanup_domain(),
                DeferredStaticInitializationCleanup { original },
            );
        }
    }
}

struct DeferredStaticInitializationCleanup<D>
where
    D: ResourceTransactionDriver + 'static,
{
    original: StaticInitializationFailure<D>,
}

impl<D> DeferredDeviceCleanupTask for DeferredStaticInitializationCleanup<D>
where
    D: ResourceTransactionDriver + 'static,
{
    fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
        match self.original.retire_in_place() {
            Ok(()) => DeferredDeviceCleanupDisposition::Completed,
            Err(_) if self.original.retirement_poison.is_some() => {
                DeferredDeviceCleanupDisposition::Quarantined
            }
            Err(_) => DeferredDeviceCleanupDisposition::Retryable,
        }
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
        self.initialize_static_with_input(
            family,
            plan,
            StaticInitializationInput::Stored(source, None),
            policy,
        )
    }

    /// Initializes weights through bounded staging or an explicit managed import.
    ///
    /// Each chunk is admitted as [`HostMemoryClass::TransferStaging`] before
    /// allocating or copying bytes. The backend must retain that exact owner
    /// until its upload is quiescent; unsupported managed uploads never fall
    /// back to a raw-byte upload. A backend may instead provide a managed
    /// import transaction with no temporary host staging. The ordinary import
    /// hook and device transforms are not used. Existing source/materializer
    /// backing, including mmap pages or format conversion allocations, is outside
    /// this staging budget; this is not a bound on total host RSS.
    pub fn initialize_static_managed(
        self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        source: &dyn WeightComponentSource,
        host: &HostMemoryAccount,
        policy: StaticInitializationPolicy,
    ) -> Result<InitializedResourceTransaction<D>, StaticInitializationFailure<D>> {
        self.initialize_static_with_input(
            family,
            plan,
            StaticInitializationInput::Stored(source, Some(host)),
            policy,
        )
    }

    /// Initializes a committed transaction for a family with no stored weights.
    ///
    /// This still zeroes every admitted static allocation and waits for its
    /// initialization fence. In particular, a plan-scoped persistent workspace
    /// is not exempt from allocation, capacity, or failed-fence retirement.
    /// Plans without any static allocations use the provisioning `NoStatic`
    /// handoff instead and cannot produce this committed transaction.
    pub fn initialize_static_without_weights(
        self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        policy: StaticInitializationPolicy,
    ) -> Result<InitializedResourceTransaction<D>, StaticInitializationFailure<D>> {
        self.initialize_static_with_input(family, plan, StaticInitializationInput::None, policy)
    }

    fn initialize_static_with_input(
        self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        input: StaticInitializationInput<'_>,
        policy: StaticInitializationPolicy,
    ) -> Result<InitializedResourceTransaction<D>, StaticInitializationFailure<D>> {
        match initialize_static_inner(&self, family, plan, input, policy) {
            Ok(receipt) => Ok(InitializedResourceTransaction {
                transaction: self,
                receipt,
            }),
            Err(step) => Err(StaticInitializationFailure::new(self, step)),
        }
    }
}

enum StaticInitializationInput<'source> {
    None,
    Stored(
        &'source dyn WeightComponentSource,
        Option<&'source HostMemoryAccount>,
    ),
}

fn initialize_static_inner<D>(
    transaction: &ResourceTransaction<D, TransactionCommitted>,
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    input: StaticInitializationInput<'_>,
    policy: StaticInitializationPolicy,
) -> Result<StaticInitializationReceipt, InitializationStepFailure<D::Runtime>>
where
    D: ResourceTransactionDriver,
{
    let initialization_started = Instant::now();
    let setup_started = Instant::now();
    preflight_transaction(transaction, family, plan).map_err(contract_failure)?;
    let (stored, managed_host) = match (input, family.weights(), plan.payload().execution_weights())
    {
        (StaticInitializationInput::None, ModelWeights::None, ExecutionWeights::None) => {
            validate_without_weights(plan).map_err(contract_failure)?;
            (None, None)
        }
        (
            StaticInitializationInput::Stored(source, host),
            ModelWeights::Stored(_),
            ExecutionWeights::Stored(weights),
        ) => (Some((source, weights)), host),
        _ => {
            return Err(contract_failure(VNextError::InvalidExecutionPlan {
                reason: "static initialization source mode differs from family or plan weights"
                    .to_owned(),
            }))
        }
    };
    let placements = if stored.is_some() {
        weight_placements(family, plan).map_err(contract_failure)?
    } else {
        BTreeMap::new()
    };
    let runtime = Arc::clone(transaction.lease().runtime());
    let has_required_weight_transforms = !plan
        .payload()
        .execution_weights()
        .static_weight_transforms()
        .is_empty();
    let mut weight_import = if placements.is_empty() || has_required_weight_transforms {
        None
    } else {
        let import = if managed_host.is_some() {
            runtime.begin_managed_static_weight_import()
        } else {
            runtime.begin_static_weight_import()
        };
        match import {
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
    if managed_host.is_some()
        && !placements.is_empty()
        && !has_required_weight_transforms
        && !runtime.requires_managed_static_staging()
        && weight_import.is_none()
    {
        return Err(contract_failure(VNextError::InvalidExecutionPlan {
            reason: "runtime declared managed static import without staging but declined import"
                .to_owned(),
        }));
    }
    if managed_host.is_some()
        && (has_required_weight_transforms
            || (weight_import.is_none() && !runtime.supports_managed_tensor_io()))
    {
        return Err(contract_failure(VNextError::InvalidExecutionPlan {
            reason: "managed static initialization requires managed host upload support or managed import, and no device weight transforms".to_owned(),
        }));
    }
    // Admit the first real chunk before even a zero batch can be submitted.
    // Holding this reservation (rather than merely observing free capacity)
    // prevents a concurrent host user from taking its admission in between.
    let mut first_staging = match (managed_host, stored) {
        (Some(host), Some((_, weights))) if weight_import.is_none() => {
            let first = weights
                .schema()
                .components
                .iter()
                .find(|component| placements.contains_key(&component.id));
            match first {
                Some(component) => {
                    let element_bytes = component.physical_element_type().size_bytes();
                    let maximum = policy.maximum_staging_bytes()
                        - policy.maximum_staging_bytes() % element_bytes;
                    let bytes = component
                        .physical_bytes()
                        .map_err(contract_failure)?
                        .min(maximum);
                    let bytes = usize::try_from(bytes).map_err(|_| {
                        contract_failure(VNextError::InvalidExecutionPlan {
                            reason: "managed static staging exceeds host address space".to_owned(),
                        })
                    })?;
                    Some(
                        host.reserve(HostMemoryClass::TransferStaging, bytes)
                            .map_err(|error| {
                                InitializationStepFailure::Quiescent(resource_failure(
                                    "static_host_staging",
                                    error,
                                ))
                            })?,
                    )
                }
                None => None,
            }
        }
        _ => None,
    };
    let created_stream = runtime.create_stream().map_err(|error| {
        InitializationStepFailure::Quiescent(device_failure(
            &runtime,
            &error,
            "static_stream_create",
        ))
    })?;
    let mut stream = Some(created_stream);
    let setup_duration = setup_started.elapsed();
    let mut pending = Vec::<<D::Runtime as DeviceRuntime>::Command>::new();
    let mut pending_staging_bytes = 0_u64;
    let mut submission_batch_count = 0_usize;
    let mut upload_command_count = 0_usize;
    let mut uploaded_component_count = 0_usize;
    let mut uploaded_bytes = 0_u64;
    let mut imported_component_count = 0_usize;
    let mut imported_bytes = 0_u64;
    let mut transformed_component_count = 0_usize;
    let mut transformed_bytes = 0_u64;
    let mut transform_command_count = 0_usize;
    let mut source_materialization_duration = Duration::ZERO;
    let mut device_encode_duration = Duration::ZERO;
    let mut device_import_duration = Duration::ZERO;
    let mut device_transform_encode_duration = Duration::ZERO;
    let mut submission_wait_duration = Duration::ZERO;
    let mut import_seal_duration = Duration::ZERO;
    let mut slowest_component_id = None;
    let mut slowest_component_materialization_duration = Duration::ZERO;
    let mut source_files = BTreeSet::new();

    for allocation in plan.payload().memory().static_allocations() {
        if allocation.usage() == BufferUsage::Weights && weight_import.is_some() {
            continue;
        }
        let encode_started = Instant::now();
        let command = with_static_buffer(transaction, allocation.resource_id(), |buffer| {
            runtime.encode_zero(buffer, 0, allocation.size_bytes())
        })
        .map_err(|error| runtime_or_contract_failure(&runtime, error, "static_zero_encode"))?;
        device_encode_duration += encode_started.elapsed();
        pending.push(command);
        if pending.len() == policy.maximum_commands_per_batch() {
            submission_wait_duration += submit_pending(
                &runtime,
                &mut stream,
                &mut pending,
                &mut pending_staging_bytes,
            )?;
            submission_batch_count += 1;
        }
    }

    if let Some((source, execution_weights)) = stored {
        let execution_weight_schema = execution_weights.schema();
        let mut materialization_groups = Vec::<Vec<&WeightComponentSpec>>::new();
        let mut materialization_group_indices = BTreeMap::<Vec<WeightId>, usize>::new();
        for component in &execution_weight_schema.components {
            if !placements.contains_key(&component.id) {
                continue;
            }
            let source_ids = execution_weights
                .component_sources()
                .get(&component.id)
                .ok_or_else(|| {
                    contract_failure(VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "execution weight component `{}` has no source mapping",
                            component.id
                        ),
                    })
                })?;
            if let Some(group_index) = materialization_group_indices.get(source_ids) {
                materialization_groups[*group_index].push(component);
            } else {
                let group_index = materialization_groups.len();
                materialization_group_indices.insert(source_ids.clone(), group_index);
                materialization_groups.push(vec![component]);
            }
        }

        for components in materialization_groups {
            if let Some(transform) = plan
                .static_weight_transform_for_components(&components)
                .map_err(contract_failure)?
            {
                let materialization_started = Instant::now();
                let sources = prepare_transform_sources(family, source, transform)
                    .map_err(contract_failure)?;
                let materialization_duration = materialization_started.elapsed();
                source_materialization_duration += materialization_duration;
                if slowest_component_id.is_none()
                    || materialization_duration > slowest_component_materialization_duration
                {
                    slowest_component_materialization_duration = materialization_duration;
                    slowest_component_id = Some(components[0].id.clone());
                }
                for source_segments in &sources {
                    source_files.extend(source_segments.source_files().iter().cloned());
                }
                let scratch_resource_id =
                    plan.payload()
                        .execution_weights()
                        .static_weight_transform_scratch_resource_id()
                        .map_err(contract_failure)?
                        .ok_or_else(|| {
                            contract_failure(VNextError::InvalidExecutionPlan {
                        reason: "required static weight transform has no admitted scratch resource"
                            .to_owned(),
                    })
                        })?;
                let encode_started = Instant::now();
                let command = encode_required_weight_transform(
                    transaction,
                    &runtime,
                    transform,
                    &sources,
                    &components,
                    &placements,
                    &scratch_resource_id,
                )
                .map_err(|error| {
                    runtime_or_contract_failure(&runtime, error, "static_weight_transform_encode")
                })?;
                device_transform_encode_duration += encode_started.elapsed();
                pending.push(command);
                transform_command_count += 1;
                transformed_component_count += components.len();
                transformed_bytes = components
                    .iter()
                    .try_fold(transformed_bytes, |total, component| {
                        total.checked_add(
                            placements
                                .get(&component.id)
                                .expect("transform components have selected placements")
                                .length_bytes,
                        )
                    })
                    .ok_or_else(|| {
                        contract_failure(VNextError::InvalidExecutionPlan {
                            reason: "static transformed bytes overflow u64".to_owned(),
                        })
                    })?;
                if pending.len() == policy.maximum_commands_per_batch() {
                    submission_wait_duration += submit_pending(
                        &runtime,
                        &mut stream,
                        &mut pending,
                        &mut pending_staging_bytes,
                    )?;
                    submission_batch_count += 1;
                }
                continue;
            }
            // Retain only outputs derived from one exact source set. This permits
            // multi-output transforms to read and convert a large source matrix
            // once without retaining converted payloads for the whole model.
            let materialization_started = Instant::now();
            let uploads =
                prepare_uploads(family, plan, source, &components, &placements, managed_host)
                    .map_err(contract_failure)?;
            let materialization_duration = materialization_started.elapsed();
            source_materialization_duration += materialization_duration;
            if slowest_component_id.is_none()
                || materialization_duration > slowest_component_materialization_duration
            {
                slowest_component_materialization_duration = materialization_duration;
                slowest_component_id = Some(components[0].id.clone());
            }
            for (component, upload) in components.into_iter().zip(uploads) {
                let placement = placements
                    .get(&component.id)
                    .expect("materialization groups contain only placed components");
                source_files.extend(upload.source_files().iter().cloned());
                if let Some(import) = weight_import.as_mut() {
                    let import_started = Instant::now();
                    with_static_buffer(transaction, &placement.resource_id, |buffer| {
                        import.import_component(&upload, buffer, placement.offset_bytes)
                    })
                    .map_err(|error| {
                        runtime_or_contract_failure(
                            &runtime,
                            error,
                            "static_weight_component_import",
                        )
                    })?;
                    device_import_duration += import_started.elapsed();
                    imported_component_count += 1;
                    imported_bytes = imported_bytes
                        .checked_add(placement.length_bytes)
                        .ok_or_else(|| {
                            contract_failure(VNextError::InvalidExecutionPlan {
                                reason: "static initialization imported bytes overflow u64"
                                    .to_owned(),
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
                                reason: "static staging budget exceeds host address space"
                                    .to_owned(),
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
                        submission_wait_duration += submit_pending(
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
                    let layout = HostTransferLayout::new(
                        upload.element_type(),
                        chunk_bytes_u64 / element_bytes,
                    )
                    .map_err(contract_failure)?;
                    // If prior chunks own the remaining host quota, retire
                    // that bounded batch once before retrying admission. Never
                    // wait while retaining the very staging capacity needed
                    // by this chunk, and never bypass a genuine capacity error.
                    let managed_bytes = if let Some(host) = managed_host {
                        let source_bytes = &bytes[source_offset..source_end];
                        let mut staged = match first_staging.take() {
                            Some(reservation) => copy_staged_upload(reservation, source_bytes),
                            None => stage_upload(host, source_bytes),
                        };
                        if matches!(
                            staged,
                            Err(HostMemoryError::Capacity { .. }
                                | HostMemoryError::ScopeCapacity { .. })
                        ) && pending_staging_bytes != 0
                        {
                            submission_wait_duration += submit_pending(
                                &runtime,
                                &mut stream,
                                &mut pending,
                                &mut pending_staging_bytes,
                            )?;
                            submission_batch_count += 1;
                            staged = stage_upload(host, source_bytes);
                        }
                        Some(staged.map_err(|error| {
                            InitializationStepFailure::Quiescent(resource_failure(
                                "static_host_staging",
                                error,
                            ))
                        })?)
                    } else {
                        None
                    };
                    let encode_started = Instant::now();
                    let command = if let Some(managed_bytes) = managed_bytes {
                        with_static_buffer(transaction, &placement.resource_id, |buffer| {
                            runtime.encode_managed_upload(
                                managed_bytes,
                                layout,
                                buffer,
                                destination_offset,
                            )
                        })
                        .map_err(|error| {
                            runtime_or_contract_failure(&runtime, error, "static_managed_upload_encode")
                        })?
                        .ok_or_else(|| contract_failure(VNextError::InvalidExecutionPlan {
                            reason: "runtime declined managed static upload; raw upload fallback is forbidden".to_owned(),
                        }))?
                    } else {
                        with_static_buffer(transaction, &placement.resource_id, |buffer| {
                            runtime.encode_upload(
                                &bytes[source_offset..source_end],
                                layout,
                                buffer,
                                destination_offset,
                            )
                        })
                        .map_err(|error| {
                            runtime_or_contract_failure(&runtime, error, "static_upload_encode")
                        })?
                    };
                    device_encode_duration += encode_started.elapsed();
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
        }
    }

    if !pending.is_empty() {
        submission_wait_duration += submit_pending(
            &runtime,
            &mut stream,
            &mut pending,
            &mut pending_staging_bytes,
        )?;
        submission_batch_count += 1;
    }
    if let Some(import) = weight_import {
        let seal_started = Instant::now();
        import.seal().map_err(|error| {
            InitializationStepFailure::Quiescent(device_failure(
                &runtime,
                &error,
                "static_weight_import_seal",
            ))
        })?;
        import_seal_duration += seal_started.elapsed();
    }
    Ok(StaticInitializationReceipt {
        initialized_resource_count: plan.payload().memory().static_allocations().len(),
        uploaded_component_count,
        uploaded_bytes,
        imported_component_count,
        imported_bytes,
        transformed_component_count,
        transformed_bytes,
        transform_command_count,
        upload_command_count,
        submission_batch_count,
        total_duration_us: duration_us(initialization_started.elapsed()),
        setup_duration_us: duration_us(setup_duration),
        source_materialization_duration_us: duration_us(source_materialization_duration),
        device_encode_duration_us: duration_us(device_encode_duration),
        device_import_duration_us: duration_us(device_import_duration),
        device_transform_encode_duration_us: duration_us(device_transform_encode_duration),
        submission_wait_duration_us: duration_us(submission_wait_duration),
        import_seal_duration_us: duration_us(import_seal_duration),
        slowest_component_id,
        slowest_component_materialization_duration_us: duration_us(
            slowest_component_materialization_duration,
        ),
        source_files,
    })
}

fn stage_upload(
    host: &HostMemoryAccount,
    bytes: &[u8],
) -> Result<ManagedHostBytes, HostMemoryError> {
    copy_staged_upload(
        host.reserve(HostMemoryClass::TransferStaging, bytes.len())?,
        bytes,
    )
}

fn copy_staged_upload(
    reservation: HostMemoryReservation,
    bytes: &[u8],
) -> Result<ManagedHostBytes, HostMemoryError> {
    if reservation.bytes() != bytes.len() || reservation.class() != HostMemoryClass::TransferStaging
    {
        return Err(HostMemoryError::InvalidReservation);
    }
    Ok(reservation.allocate_copy(bytes)?.freeze())
}

fn submit_pending<R>(
    runtime: &Arc<R>,
    stream: &mut Option<R::Stream>,
    pending: &mut Vec<R::Command>,
    pending_staging_bytes: &mut u64,
) -> Result<Duration, InitializationStepFailure<R>>
where
    R: DeviceRuntime,
{
    let started = Instant::now();
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
            DeviceTerminal::Succeeded => Ok(started.elapsed()),
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

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
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
    payload
        .execution_weights()
        .validate_against_family(family)?;
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

fn validate_without_weights(plan: &ExecutionPlan) -> Result<(), VNextError> {
    if plan
        .payload()
        .memory()
        .static_allocations()
        .iter()
        .any(|allocation| allocation.usage() == BufferUsage::Weights)
        || plan
            .payload()
            .memory()
            .dynamic_descriptors()
            .iter()
            .any(|allocation| allocation.usage() == BufferUsage::Weights)
        || plan.payload().nodes().iter().any(|node| {
            node.values()
                .iter()
                .any(|binding| binding.usage() == BufferUsage::Weights)
        })
        || !plan
            .payload()
            .execution_weights()
            .static_weight_transforms()
            .is_empty()
    {
        return Err(VNextError::InvalidExecutionPlan {
            reason: "weightless static initialization cannot contain weight resources, bindings, or transforms".to_owned(),
        });
    }
    Ok(())
}

fn weight_placements(
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
) -> Result<BTreeMap<WeightId, WeightPlacement>, VNextError> {
    plan.payload()
        .execution_weights()
        .validate_against_family(family)?;
    let execution_weight_schema = plan.payload().execution_weights().stored()?.schema();
    let schema = execution_weight_schema
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
    for component in &execution_weight_schema.components {
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

fn prepare_uploads<'source>(
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    source: &'source dyn WeightComponentSource,
    components: &[&WeightComponentSpec],
    placements: &BTreeMap<WeightId, WeightPlacement>,
    host: Option<&HostMemoryAccount>,
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    let payloads = plan.materialize_weight_components(family, source, components, host)?;
    for (component, payload) in components.iter().zip(&payloads) {
        let placement =
            placements
                .get(&component.id)
                .ok_or_else(|| VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "execution weight component `{}` has no selected placement",
                        component.id
                    ),
                })?;
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
    }
    Ok(payloads)
}

fn prepare_transform_sources<'source>(
    family: &PreparedModelFamily,
    source: &'source dyn WeightComponentSource,
    transform: &StaticWeightTransformPlan,
) -> Result<Vec<WeightComponentSegments<'source>>, VNextError> {
    transform
        .source_component_ids()
        .into_iter()
        .map(|source_id| {
            let component_index = family
                .stored_weight_schema()?
                .components
                .binary_search_by(|component| component.id.cmp(source_id))
                .map_err(|_| VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "static weight transform references unknown source component `{source_id}`"
                    ),
                })?;
            let component = &family.stored_weight_schema()?.components[component_index];
            let segments = source.component_segments(component)?;
            if segments.component_id() != &component.id
                || segments.external_names() != component.external_names.as_slice()
                || segments.dimensions() != component.dimensions.as_slice()
                || segments.element_type() != component.physical_element_type()
                || segments.total_bytes() != component.physical_bytes()?
            {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "static weight transform source segments for `{}` differ from the trusted source schema",
                        component.id
                    ),
                });
            }
            Ok(segments)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn encode_required_weight_transform<'source, D>(
    transaction: &ResourceTransaction<D, TransactionCommitted>,
    runtime: &Arc<D::Runtime>,
    transform: &StaticWeightTransformPlan,
    sources: &[WeightComponentSegments<'source>],
    components: &[&WeightComponentSpec],
    placements: &BTreeMap<WeightId, WeightPlacement>,
    scratch_resource_id: &ResourceId,
) -> Result<
    <D::Runtime as DeviceRuntime>::Command,
    StaticBufferAccessError<<D::Runtime as DeviceRuntime>::Error>,
>
where
    D: ResourceTransactionDriver,
{
    let [packed_values_id, scales_id] = transform.execution_component_ids();
    let packed_component = components
        .iter()
        .copied()
        .find(|component| &component.id == packed_values_id)
        .ok_or_else(|| {
            StaticBufferAccessError::Contract(VNextError::InvalidExecutionPlan {
                reason: "static weight transform packed output is absent from its component group"
                    .to_owned(),
            })
        })?;
    let scales_component = components
        .iter()
        .copied()
        .find(|component| &component.id == scales_id)
        .ok_or_else(|| {
            StaticBufferAccessError::Contract(VNextError::InvalidExecutionPlan {
                reason: "static weight transform scale output is absent from its component group"
                    .to_owned(),
            })
        })?;
    let packed_placement = placements.get(packed_values_id).ok_or_else(|| {
        StaticBufferAccessError::Contract(VNextError::InvalidExecutionPlan {
            reason: "static weight transform packed output has no placement".to_owned(),
        })
    })?;
    let scales_placement = placements.get(scales_id).ok_or_else(|| {
        StaticBufferAccessError::Contract(VNextError::InvalidExecutionPlan {
            reason: "static weight transform scale output has no placement".to_owned(),
        })
    })?;
    let lease = transaction.lease();
    let packed_entry = lease
        .plan_static_entries()
        .find(|entry| entry.resource_id() == &packed_placement.resource_id)
        .ok_or_else(|| {
            StaticBufferAccessError::Contract(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "static lease lacks transform destination `{}`",
                    packed_placement.resource_id
                ),
            })
        })?;
    let scales_entry = lease
        .plan_static_entries()
        .find(|entry| entry.resource_id() == &scales_placement.resource_id)
        .ok_or_else(|| {
            StaticBufferAccessError::Contract(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "static lease lacks transform destination `{}`",
                    scales_placement.resource_id
                ),
            })
        })?;
    let scratch_entry = lease
        .plan_static_entries()
        .find(|entry| entry.resource_id() == scratch_resource_id)
        .ok_or_else(|| {
            StaticBufferAccessError::Contract(VNextError::InvalidExecutionPlan {
                reason: format!("static lease lacks transform scratch `{scratch_resource_id}`"),
            })
        })?;
    let packed_view = lease
        .view(&packed_placement.resource_id, packed_entry.generation())
        .map_err(StaticBufferAccessError::Contract)?;
    let scales_view = lease
        .view(&scales_placement.resource_id, scales_entry.generation())
        .map_err(StaticBufferAccessError::Contract)?;
    let scratch_view = lease
        .view(scratch_resource_id, scratch_entry.generation())
        .map_err(StaticBufferAccessError::Contract)?;
    let destinations = [
        StaticWeightTransformDestination::new(
            packed_component,
            packed_view.buffer(),
            packed_placement.offset_bytes,
        ),
        StaticWeightTransformDestination::new(
            scales_component,
            scales_view.buffer(),
            scales_placement.offset_bytes,
        ),
    ];
    let request =
        StaticWeightTransformRequest::new(transform, sources, &destinations, scratch_view.buffer());
    match runtime.encode_static_weight_transform(request) {
        Some(Ok(command)) => Ok(command),
        Some(Err(error)) => Err(StaticBufferAccessError::Runtime(error)),
        None => Err(StaticBufferAccessError::Contract(
            VNextError::InvalidExecutionPlan {
                reason: "device runtime does not support the required static weight transform"
                    .to_owned(),
            },
        )),
    }
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

#[cfg(test)]
pub(crate) static STATIC_PROVISION_CLEANUP_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
#[path = "static_initialization/retirement_tests.rs"]
mod retirement_tests;

#[cfg(test)]
#[path = "static_initialization/managed_staging_tests.rs"]
mod managed_staging_tests;
