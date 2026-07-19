use super::{
    core_resource_failure, deferred_device_cleanup_status, invalid_resource,
    maintain_deferred_device_cleanups, new_deferred_device_cleanup_domain,
    retire_deferred_device_cleanup_domain, watch, AdmissionDeferred, AdmissionDemand,
    AdmissionFitPolicy, AdmissionPressureAction, AllocationLifetime, Arc, AtomicU8,
    BackingPrepareDecision, CapacityAvailabilityEpoch, CapacityEntry, CapacityEpochs,
    CapacityUnits, CapacityVector, CapacityWaitCondition, CapacityWaitRecheck,
    DeferredDeviceCleanupDomainId, DeferredDeviceCleanupMaintenanceReceipt,
    DeferredDeviceCleanupStatus, DeviceCapacityClaim, DeviceCapacitySignal, DeviceId,
    DeviceRuntime, DynamicBackingDeferred, DynamicDeferredMaintenanceOutcome,
    DynamicPoolMaintenanceController, DynamicPoolMaintenanceStatus, DynamicPoolSet,
    DynamicResourceShape, EvaluatedBackingProjection, EvaluatedBackingRequest, ExecutionLane,
    ExecutionLaneCreationError, FailureEnvelope, InvocationLivenessMode,
    LaneBackingPrepareDecision, LogicalAdmissionCoordinator, LogicalAdmissionCoordinatorId, Mutex,
    NoStatic, NodeId, Ordering, PhysicalBackingClaimIdentity, PlanHash, PlanId, PlanNode,
    ResourceAbandonSignal, ResourceActionCursor, ResourceDriverFailure,
    ResourceLedgerEntrySnapshot, ResourceOwnershipReason, ResourceOwnershipTransferFailure,
    ResourcePoolIdentity, ResourcePoolOwnership, ResourceReservation, ResourceReservationBatch,
    ResourceTransactionAction, ResourceTransactionContext, ResourceTransactionDriver,
    ResourceTransactionIdentity, ResourceTransactionState, RwLock, RwLockReadGuard, Serialize,
    StaticProvisioningBinding, StaticProvisioningLease, VNextError,
};
use crate::vnext::{
    ResolvedReusableExecutionBucket, ReusableExecutionBucketId, ReusableExecutionBucketSpec,
};

pub(super) const PLAN_RUNTIME_OPEN: u8 = 0;
const PLAN_RUNTIME_CLOSING: u8 = 1;

pub(super) trait ErasedPlanStaticDriver<R>: Send
where
    R: DeviceRuntime,
{
    fn release_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, R>,
        reservation: &ResourceReservation,
        buffer: &R::Buffer,
    ) -> Result<(), ResourceDriverFailure>;

    fn quarantine_transaction(
        &mut self,
        context: &ResourceTransactionContext<'_, R>,
        ownership: ResourcePoolOwnership<R>,
    ) -> Result<(), ResourceOwnershipTransferFailure<R>>;

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<R>);
}

impl<D> ErasedPlanStaticDriver<D::Runtime> for D
where
    D: ResourceTransactionDriver,
{
    fn release_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, D::Runtime>,
        reservation: &ResourceReservation,
        buffer: &D::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        ResourceTransactionDriver::release_resource(self, context, reservation, buffer)
    }

    fn quarantine_transaction(
        &mut self,
        context: &ResourceTransactionContext<'_, D::Runtime>,
        ownership: ResourcePoolOwnership<D::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<D::Runtime>> {
        ResourceTransactionDriver::quarantine_transaction(self, context, ownership)
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<D::Runtime>) {
        ResourceTransactionDriver::abandon_transaction(self, ownership);
    }
}

pub(super) struct PlanStaticResources<R>
where
    R: DeviceRuntime,
{
    pub(super) driver: Mutex<Option<Box<dyn ErasedPlanStaticDriver<R>>>>,
    pub(super) identity: ResourceTransactionIdentity,
    pub(super) admission: StaticProvisioningBinding,
    pub(super) reservations: ResourceReservationBatch,
    pub(super) states: Vec<ResourceTransactionState>,
    pub(super) capacity_claim: Option<DeviceCapacityClaim>,
    pub(super) lease: Option<StaticProvisioningLease<R>>,
    pub(super) finalized: bool,
}

impl<R> PlanStaticResources<R>
where
    R: DeviceRuntime,
{
    fn ledger_snapshot_entries(&self) -> Vec<ResourceLedgerEntrySnapshot> {
        self.lease
            .as_ref()
            .expect("open plan runtime owns its static lease")
            .slots
            .iter()
            .zip(&self.states)
            .map(|(slot, &transaction_state)| ResourceLedgerEntrySnapshot {
                entry: slot.entry.clone(),
                transaction_state,
                buffer_present: slot.buffer.is_some(),
                actual_resource_id: slot.actual_resource_id.clone(),
                actual_generation: slot.actual_generation,
                actual_descriptor: slot.descriptor.clone(),
            })
            .collect()
    }

    fn release_all(&mut self) -> Result<usize, ResourceDriverFailure> {
        let mut released = 0;
        for order in 0..self.states.len() {
            if self.states[order] == ResourceTransactionState::Released {
                continue;
            }
            if self.states[order] != ResourceTransactionState::Committed {
                return Err(ResourceDriverFailure::new(core_resource_failure(
                    "plan_runtime_close_ledger_diverged",
                    "plan runtime close found a non-committed static resource",
                    false,
                ))
                .expect("core failure has resource domain"));
            }
            let lease = self
                .lease
                .as_ref()
                .expect("open plan runtime owns its static lease");
            let reservation = self.reservations.reservations[order].clone();
            let buffer = lease
                .buffer(order)
                .expect("committed plan runtime owns its static buffer");
            let context = ResourceTransactionContext {
                runtime: &lease.runtime,
                identity: &self.identity,
                binding: &self.admission,
                reservations: &self.reservations,
                cursor: Some(ResourceActionCursor {
                    order,
                    action: ResourceTransactionAction::Release,
                    before: self.states[order],
                    allocation_authorized: false,
                }),
                allocation_authority: None,
                pending_allocation: None,
            };
            let driver = match self.driver.get_mut() {
                Ok(driver) => driver,
                Err(poisoned) => poisoned.into_inner(),
            };
            driver
                .as_mut()
                .expect("open plan runtime owns its static driver")
                .release_resource(&context, &reservation, buffer)?;
            self.lease
                .as_mut()
                .expect("open plan runtime owns its static lease")
                .clear(order);
            self.states[order] = ResourceTransactionState::Released;
            self.capacity_claim
                .as_mut()
                .expect("open plan runtime owns its static capacity claim")
                .release_bytes(reservation.size_bytes());
            released += 1;
        }
        if let Some(mut claim) = self.capacity_claim.take() {
            claim.release();
        }
        self.finalized = true;
        Ok(released)
    }

    fn quarantine_remaining(&mut self) -> Result<usize, ResourceDriverFailure> {
        let quarantined = self
            .states
            .iter()
            .filter(|state| **state == ResourceTransactionState::Committed)
            .count();
        if self.states.iter().any(|state| {
            !matches!(
                state,
                ResourceTransactionState::Committed | ResourceTransactionState::Released
            )
        }) {
            return Err(ResourceDriverFailure::new(core_resource_failure(
                "plan_runtime_quarantine_ledger_diverged",
                "plan runtime quarantine found an invalid static resource state",
                false,
            ))
            .expect("core failure has resource domain"));
        }
        if quarantined == 0 {
            self.finalized = true;
            return Ok(0);
        }
        let lease = self
            .lease
            .as_mut()
            .expect("failed plan runtime close owns its static lease");
        let buffers = lease.take_owned_buffers(&self.reservations);
        let ownership = ResourcePoolOwnership {
            runtime: Arc::clone(&lease.runtime),
            pool_identity: self.admission.pool_identity.clone(),
            reason: ResourceOwnershipReason::Quarantine,
            signal: None,
            buffers,
            capacity_claim: self.capacity_claim.take(),
        };
        let result = {
            let context = ResourceTransactionContext {
                runtime: &lease.runtime,
                identity: &self.identity,
                binding: &self.admission,
                reservations: &self.reservations,
                cursor: None,
                allocation_authority: None,
                pending_allocation: None,
            };
            let driver = match self.driver.get_mut() {
                Ok(driver) => driver,
                Err(poisoned) => poisoned.into_inner(),
            };
            driver
                .as_mut()
                .expect("failed plan runtime close owns its static driver")
                .quarantine_transaction(&context, ownership)
        };
        if let Err(failure) = result {
            let (failure, mut ownership) = failure.into_parts();
            let expected_claimed_bytes = self
                .states
                .iter()
                .zip(self.reservations.reservations())
                .filter(|(state, _)| state.is_live())
                .map(|(_, reservation)| reservation.size_bytes())
                .sum::<u64>();
            if ownership.pool_identity != self.admission.pool_identity
                || ownership.claimed_bytes() != expected_claimed_bytes
            {
                std::mem::forget(ownership);
                return Err(ResourceDriverFailure::new(core_resource_failure(
                    "ownership_transfer_identity_mismatch",
                    "plan runtime quarantine failure returned foreign ownership",
                    false,
                ))
                .expect("core failure has resource domain"));
            }
            self.lease
                .as_mut()
                .expect("failed plan runtime close owns its static lease")
                .restore_owned_buffers(std::mem::take(&mut ownership.buffers));
            self.capacity_claim = ownership.capacity_claim.take();
            return Err(failure);
        }
        for state in &mut self.states {
            if *state == ResourceTransactionState::Committed {
                *state = ResourceTransactionState::Quarantined;
            }
        }
        self.finalized = true;
        Ok(quarantined)
    }
}

impl<R> Drop for PlanStaticResources<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        let signal = ResourceAbandonSignal {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            state: if self
                .states
                .iter()
                .all(|state| *state == ResourceTransactionState::Committed)
            {
                ResourceTransactionState::Committed
            } else {
                ResourceTransactionState::Released
            },
            pending_action: None,
            ledger: self.ledger_snapshot_entries(),
            active_sequence_slots: Vec::new(),
            poisoned_sequence_slots: Vec::new(),
            undrained_sequence_slots: Vec::new(),
            failure: None,
        };
        let mut lease = self
            .lease
            .take()
            .expect("open plan runtime owns its static lease");
        let buffers = lease.take_owned_buffers(&self.reservations);
        let StaticProvisioningLease {
            slots: _,
            identity: _,
            admission: _,
            runtime,
        } = lease;
        let ownership = ResourcePoolOwnership {
            runtime,
            pool_identity: self.admission.pool_identity.clone(),
            reason: ResourceOwnershipReason::Abandon,
            signal: Some(signal),
            buffers,
            capacity_claim: self.capacity_claim.take(),
        };
        let driver = match self.driver.get_mut() {
            Ok(driver) => driver,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(driver) = driver.as_mut() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                driver.abandon_transaction(ownership);
            }));
        } else {
            std::mem::forget(ownership);
        }
        self.finalized = true;
    }
}

pub(super) enum PlanRuntimeStatic<R>
where
    R: DeviceRuntime,
{
    NoStatic { binding: StaticProvisioningBinding },
    Static(PlanStaticResources<R>),
}

/// Unique plan-lifetime owner of the runtime, dynamic pools, maintenance
/// authority, static buffers, capacity claims, and cleanup authority.
#[must_use = "plan runtime resources must be explicitly closed or safely abandoned"]
pub struct PlanRuntimeResources<R>
where
    R: DeviceRuntime,
{
    pub(super) lifecycle: RwLock<()>,
    pub(super) phase: AtomicU8,
    pub(super) lifecycle_tx: watch::Sender<u8>,
    pub(super) maintenance_controller: DynamicPoolMaintenanceController<R>,
    pub(super) dynamic_pools: Arc<DynamicPoolSet<R>>,
    pub(super) static_resources: PlanRuntimeStatic<R>,
    pub(super) runtime: Arc<R>,
    pub(super) deferred_cleanup_domain: DeferredDeviceCleanupDomainId,
}

/// Sealed owning proof that one exact plan, runtime instance, provisioning
/// outcome, and admission coordinator belong together. Every durable child
/// authority holds the same root `Arc`.
#[must_use = "a trusted plan/runtime binding must be consumed by logical admission"]
pub struct TrustedPlanRuntimeBinding<R>
where
    R: DeviceRuntime,
{
    pub(super) resources: Arc<PlanRuntimeResources<R>>,
}

/// Internal owning capability that binds non-authoritative backing evidence to
/// the one plan runtime allowed to revalidate it. Public scope-specific handles
/// embed this value and retain any additional request/session/step parent.
pub(super) struct PlanBackingDeferral<R>
where
    R: DeviceRuntime,
{
    evidence: DynamicBackingDeferred,
    resources: Arc<PlanRuntimeResources<R>>,
}

impl<R> PlanBackingDeferral<R>
where
    R: DeviceRuntime,
{
    pub(super) fn new(
        resources: Arc<PlanRuntimeResources<R>>,
        evidence: DynamicBackingDeferred,
    ) -> Result<Self, VNextError> {
        {
            let _lifecycle = resources.read_lifecycle("bind deferred backing to its plan")?;
            if evidence.wait_condition().coordinator_id()
                != resources.dynamic_pools.logical_admission.id()
            {
                return Err(invalid_resource(
                    "deferred backing belongs to another plan coordinator",
                ));
            }
        }
        Ok(Self {
            evidence,
            resources,
        })
    }

    pub(super) fn evidence(&self) -> &DynamicBackingDeferred {
        &self.evidence
    }

    pub(super) fn maintain(&self) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let _lifecycle = self
            .resources
            .read_lifecycle("maintain plan-owned deferred backing")?;
        // Stable lane slots own real residency. Reclaim a provably idle slot
        // before asking the pool to grow, otherwise a reclaimable cache entry
        // can make growth look like a terminal resident-ceiling violation.
        if self
            .resources
            .dynamic_pools
            .try_reclaim_expired_lane_slots()?
        {
            return self.retry_admission();
        }
        let outcome = self
            .resources
            .maintenance_controller
            .maintain_for_live_deferred(&self.evidence)?;
        if matches!(
            &outcome,
            DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
        ) && self
            .resources
            .dynamic_pools
            .try_reclaim_one_idle_lane_slot()?
        {
            return self.retry_admission();
        }
        Ok(outcome)
    }

    pub(super) fn retry_admission(&self) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let mut availability = Vec::with_capacity(self.resources.dynamic_pools.domains.len() + 3);
        let current_epochs = self
            .resources
            .dynamic_pools
            .write_capacity_availability(&mut availability)?;
        Ok(DynamicDeferredMaintenanceOutcome::RetryAdmission { current_epochs })
    }

    pub(super) fn register_waiter(&self) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.resources
            .register_capacity_waiter(self.evidence.wait_condition())
    }
}

/// A capacity wait registration that keeps its exact plan runtime alive until
/// the waiter either observes a retry epoch or is cancelled by being dropped.
#[must_use = "capacity wait registrations must be awaited, rechecked, or dropped"]
pub struct PlanCapacityWaitRegistration<R>
where
    R: DeviceRuntime,
{
    observed: CapacityWaitCondition,
    registered: CapacityWaitCondition,
    logical_rx: watch::Receiver<CapacityEpochs>,
    plan_capacity_rx: watch::Receiver<DeviceCapacitySignal>,
    process_capacity_rx: watch::Receiver<DeviceCapacitySignal>,
    lifecycle_rx: watch::Receiver<u8>,
    resources: Arc<PlanRuntimeResources<R>>,
}

impl<R> PlanCapacityWaitRegistration<R>
where
    R: DeviceRuntime,
{
    pub fn recheck(&self) -> Result<CapacityWaitRecheck, VNextError> {
        let _lifecycle = self.resources.read_lifecycle("recheck a capacity waiter")?;
        let mut availability = Vec::with_capacity(self.resources.dynamic_pools.domains.len() + 3);
        let current = self
            .resources
            .dynamic_pools
            .write_capacity_availability(&mut availability)?;
        Ok(CapacityWaitRecheck::new(
            current,
            self.observed.changed_since(&availability)?,
            self.registered.changed_since(&availability)?,
        ))
    }

    pub async fn wait_for_change(self) -> Result<CapacityEpochs, VNextError> {
        let Self {
            observed,
            registered,
            mut logical_rx,
            mut plan_capacity_rx,
            mut process_capacity_rx,
            mut lifecycle_rx,
            resources,
        } = self;
        loop {
            let recheck = {
                let _lifecycle = match resources.read_lifecycle("recheck a capacity waiter") {
                    Ok(lifecycle) => lifecycle,
                    Err(_) if resources.is_closing() => {
                        return Err(invalid_resource(
                            "closing plan runtime cancelled its capacity waiter",
                        ));
                    }
                    Err(error) => return Err(error),
                };
                let mut availability =
                    Vec::with_capacity(resources.dynamic_pools.domains.len() + 3);
                let current = resources
                    .dynamic_pools
                    .write_capacity_availability(&mut availability)?;
                CapacityWaitRecheck::new(
                    current,
                    observed.changed_since(&availability)?,
                    registered.changed_since(&availability)?,
                )
            };
            if recheck.should_retry() {
                return Ok(recheck.current());
            }
            tokio::select! {
                biased;
                changed = lifecycle_rx.changed() => {
                    changed.map_err(|_| invalid_resource(
                        "plan runtime lifecycle signal closed while a capacity waiter was live",
                    ))?;
                    if *lifecycle_rx.borrow_and_update() == PLAN_RUNTIME_CLOSING {
                        return Err(invalid_resource(
                            "closing plan runtime cancelled its capacity waiter",
                        ));
                    }
                    return Err(invalid_resource(
                        "capacity waiter observed an invalid plan runtime lifecycle transition",
                    ));
                }
                changed = logical_rx.changed() => {
                    changed.map_err(|_| invalid_resource(
                        "logical capacity signal closed while a plan waiter was live",
                    ))?;
                    logical_rx.borrow_and_update();
                }
                changed = plan_capacity_rx.changed() => {
                    changed.map_err(|_| invalid_resource(
                        "plan device-capacity signal closed while a waiter was live",
                    ))?;
                    plan_capacity_rx.borrow_and_update();
                }
                changed = process_capacity_rx.changed() => {
                    changed.map_err(|_| invalid_resource(
                        "process device-capacity signal closed while a waiter was live",
                    ))?;
                    process_capacity_rx.borrow_and_update();
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlanRuntimeCloseReceipt {
    evidence: TrustedPlanRuntimeEvidence,
    released_static_resources: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlanRuntimeQuarantineReceipt {
    evidence: TrustedPlanRuntimeEvidence,
    released_static_resources: usize,
    quarantined_static_resources: usize,
}

impl PlanRuntimeQuarantineReceipt {
    pub fn evidence(&self) -> &TrustedPlanRuntimeEvidence {
        &self.evidence
    }

    pub const fn released_static_resources(&self) -> usize {
        self.released_static_resources
    }

    pub const fn quarantined_static_resources(&self) -> usize {
        self.quarantined_static_resources
    }
}

impl PlanRuntimeCloseReceipt {
    pub fn evidence(&self) -> &TrustedPlanRuntimeEvidence {
        &self.evidence
    }

    pub const fn released_static_resources(&self) -> usize {
        self.released_static_resources
    }
}

pub enum PlanRuntimeCloseOutcome<R>
where
    R: DeviceRuntime,
{
    Closed(PlanRuntimeCloseReceipt),
    Referenced {
        resources: Arc<PlanRuntimeResources<R>>,
        strong_count: usize,
        deferred_cleanup: DeferredDeviceCleanupStatus,
    },
}

#[must_use = "failed plan runtime close retains static cleanup ownership"]
pub struct PlanRuntimeCloseFailure<R>
where
    R: DeviceRuntime,
{
    failure: FailureEnvelope,
    evidence: TrustedPlanRuntimeEvidence,
    static_resources: Option<PlanStaticResources<R>>,
}

impl<R> PlanRuntimeCloseFailure<R>
where
    R: DeviceRuntime,
{
    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub fn retry(mut self) -> Result<PlanRuntimeCloseReceipt, Self> {
        let static_resources = self
            .static_resources
            .as_mut()
            .expect("plan runtime close failure owns static cleanup authority");
        let total_static_resources = static_resources.states.len();
        match static_resources.release_all() {
            Ok(_) => {
                self.static_resources.take();
                Ok(PlanRuntimeCloseReceipt {
                    evidence: self.evidence,
                    released_static_resources: total_static_resources,
                })
            }
            Err(failure) => {
                self.failure = failure.into_failure();
                Err(self)
            }
        }
    }

    pub fn quarantine(mut self) -> Result<PlanRuntimeQuarantineReceipt, Self> {
        let static_resources = self
            .static_resources
            .as_mut()
            .expect("plan runtime close failure owns static cleanup authority");
        let released_static_resources = static_resources
            .states
            .iter()
            .filter(|state| **state == ResourceTransactionState::Released)
            .count();
        match static_resources.quarantine_remaining() {
            Ok(quarantined_static_resources) => {
                self.static_resources.take();
                Ok(PlanRuntimeQuarantineReceipt {
                    evidence: self.evidence,
                    released_static_resources,
                    quarantined_static_resources,
                })
            }
            Err(failure) => {
                self.failure = failure.into_failure();
                Err(self)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedPlanRuntimeEvidence {
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    runtime_implementation_fingerprint: String,
    coordinator_id: LogicalAdmissionCoordinatorId,
    static_provisioning_binding: Option<StaticProvisioningBinding>,
    static_pool_identity: Option<ResourcePoolIdentity>,
    static_provisioning_identity: Option<ResourceTransactionIdentity>,
}

impl TrustedPlanRuntimeEvidence {
    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub const fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub fn static_pool_identity(&self) -> Option<&ResourcePoolIdentity> {
        self.static_pool_identity.as_ref()
    }

    pub fn static_provisioning_binding(&self) -> Option<&StaticProvisioningBinding> {
        self.static_provisioning_binding.as_ref()
    }

    pub fn static_provisioning_identity(&self) -> Option<&ResourceTransactionIdentity> {
        self.static_provisioning_identity.as_ref()
    }
}

impl<R> NoStatic<R>
where
    R: DeviceRuntime,
{
    pub fn into_plan_runtime(self) -> Arc<PlanRuntimeResources<R>> {
        let Self {
            maintenance_controller,
            dynamic_pools,
            binding,
            runtime,
        } = self;
        let (lifecycle_tx, _) = watch::channel(PLAN_RUNTIME_OPEN);
        Arc::new(PlanRuntimeResources {
            lifecycle: RwLock::new(()),
            phase: AtomicU8::new(PLAN_RUNTIME_OPEN),
            lifecycle_tx,
            maintenance_controller,
            dynamic_pools,
            static_resources: PlanRuntimeStatic::NoStatic { binding },
            runtime,
            deferred_cleanup_domain: new_deferred_device_cleanup_domain(),
        })
    }
}

impl<R> PlanRuntimeResources<R>
where
    R: DeviceRuntime,
{
    fn evidence(&self) -> TrustedPlanRuntimeEvidence {
        let (binding, identity) = match &self.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => (binding, None),
            PlanRuntimeStatic::Static(source) => (&source.admission, Some(&source.identity)),
        };
        let has_static = matches!(&self.static_resources, PlanRuntimeStatic::Static(_));
        TrustedPlanRuntimeEvidence {
            plan_id: binding.plan_id().clone(),
            plan_hash: binding.plan_hash().clone(),
            device_id: binding.device_id().clone(),
            runtime_implementation_fingerprint: binding
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            coordinator_id: self.dynamic_pools.logical_admission.id(),
            static_provisioning_binding: has_static.then(|| binding.clone()),
            static_pool_identity: has_static.then(|| binding.pool_identity().clone()),
            static_provisioning_identity: identity.cloned(),
        }
    }

    pub(super) fn read_lifecycle(
        &self,
        action: &'static str,
    ) -> Result<RwLockReadGuard<'_, ()>, VNextError> {
        let lifecycle = self
            .lifecycle
            .read()
            .map_err(|_| invalid_resource("plan runtime lifecycle gate is poisoned"))?;
        if self.phase.load(Ordering::Acquire) != PLAN_RUNTIME_OPEN {
            return Err(invalid_resource(format!(
                "closing plan runtime cannot {action}"
            )));
        }
        let cleanup = self.deferred_cleanup_status();
        if cleanup.is_saturated() {
            return Err(invalid_resource(format!(
                "plan runtime cannot {action} while {} deferred device cleanup owners await recovery",
                cleanup.pending()
            )));
        }
        Ok(lifecycle)
    }

    pub fn deferred_cleanup_status(&self) -> DeferredDeviceCleanupStatus {
        deferred_device_cleanup_status(self.deferred_cleanup_domain)
    }

    pub fn create_execution_lane(
        &self,
    ) -> Result<Arc<ExecutionLane<R>>, ExecutionLaneCreationError<R::Error>> {
        let _lifecycle = self
            .read_lifecycle("create an execution lane")
            .map_err(ExecutionLaneCreationError::Contract)?;
        ExecutionLane::create(Arc::clone(&self.runtime))
    }

    /// Attempts each selected cleanup owner at most once. This may block in a
    /// backend recovery call and must therefore run on a scheduler recovery
    /// thread, never on a request, admission, or destructor path.
    pub fn maintain_deferred_cleanups(
        &self,
        maximum_tasks: usize,
    ) -> Result<DeferredDeviceCleanupMaintenanceReceipt, VNextError> {
        if maximum_tasks == 0
            || maximum_tasks > super::MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS
        {
            return Err(invalid_resource(format!(
                "deferred device cleanup maintenance size must be in 1..={}",
                super::MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS
            )));
        }
        Ok(maintain_deferred_device_cleanups(
            self.deferred_cleanup_domain,
            maximum_tasks,
        ))
    }

    pub fn trusted_runtime_binding(
        self: &Arc<Self>,
    ) -> Result<TrustedPlanRuntimeBinding<R>, VNextError> {
        let _lifecycle = self.read_lifecycle("mint a trusted runtime binding")?;
        Ok(TrustedPlanRuntimeBinding {
            resources: Arc::clone(self),
        })
    }

    pub fn maintain_for_admission_deferred(
        self: &Arc<Self>,
        deferred: &AdmissionDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let _lifecycle = self.read_lifecycle("maintain deferred logical backing growth")?;
        self.maintenance_controller
            .maintain_for_admission_deferred(deferred)
    }

    /// Returns a point-in-time view of the exact dynamic pools owned by this
    /// plan. Product telemetry consumes this instead of maintaining a second
    /// allocator ledger that can drift from admission decisions.
    pub fn dynamic_pool_status(&self) -> Result<DynamicPoolMaintenanceStatus, VNextError> {
        let _lifecycle = self.read_lifecycle("observe dynamic pool status")?;
        self.maintenance_controller.status()
    }

    pub fn write_dynamic_capacity_availability(
        &self,
        out: &mut Vec<CapacityAvailabilityEpoch>,
    ) -> Result<CapacityEpochs, VNextError> {
        let _lifecycle = self.read_lifecycle("observe dynamic capacity availability")?;
        self.dynamic_pools.write_capacity_availability(out)
    }

    pub fn register_capacity_waiter(
        self: &Arc<Self>,
        observed: &CapacityWaitCondition,
    ) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        let _lifecycle = self.read_lifecycle("register a capacity waiter")?;
        if observed.coordinator_id() != self.dynamic_pools.logical_admission.id() {
            return Err(invalid_resource(
                "capacity wait condition belongs to another plan coordinator",
            ));
        }
        let logical_rx = self.dynamic_pools.logical_admission.subscribe_epochs();
        let plan_capacity_rx = self.dynamic_pools.budget.subscribe_plan_availability();
        let process_capacity_rx = self.dynamic_pools.budget.subscribe_process_availability();
        let lifecycle_rx = self.lifecycle_tx.subscribe();
        let mut availability = Vec::with_capacity(self.dynamic_pools.domains.len() + 3);
        self.dynamic_pools
            .write_capacity_availability(&mut availability)?;
        let registered = observed.refreshed_from(&availability)?;
        Ok(PlanCapacityWaitRegistration {
            observed: observed.clone(),
            registered,
            logical_rx,
            plan_capacity_rx,
            process_capacity_rx,
            lifecycle_rx,
            resources: Arc::clone(self),
        })
    }

    pub fn is_closing(&self) -> bool {
        self.phase.load(Ordering::Acquire) == PLAN_RUNTIME_CLOSING
    }

    pub fn close(
        resources: Arc<Self>,
    ) -> Result<PlanRuntimeCloseOutcome<R>, PlanRuntimeCloseFailure<R>> {
        {
            let _lifecycle = resources
                .lifecycle
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match resources.phase.compare_exchange(
                PLAN_RUNTIME_OPEN,
                PLAN_RUNTIME_CLOSING,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    resources.lifecycle_tx.send_replace(PLAN_RUNTIME_CLOSING);
                }
                Err(PLAN_RUNTIME_CLOSING) => {}
                Err(_) => unreachable!("plan runtime phase is privately bounded"),
            }
        }
        let resources = match Arc::try_unwrap(resources) {
            Ok(resources) => resources,
            Err(resources) => {
                let strong_count = Arc::strong_count(&resources);
                let deferred_cleanup = resources.deferred_cleanup_status();
                return Ok(PlanRuntimeCloseOutcome::Referenced {
                    resources,
                    strong_count,
                    deferred_cleanup,
                });
            }
        };
        if resources.deferred_cleanup_status().pending() != 0 {
            let resources = Arc::new(resources);
            let deferred_cleanup = resources.deferred_cleanup_status();
            return Ok(PlanRuntimeCloseOutcome::Referenced {
                resources,
                strong_count: 1,
                deferred_cleanup,
            });
        }
        let evidence = resources.evidence();
        let Self {
            lifecycle: _,
            phase: _,
            lifecycle_tx,
            maintenance_controller,
            dynamic_pools,
            static_resources,
            runtime,
            deferred_cleanup_domain,
        } = resources;
        debug_assert!(retire_deferred_device_cleanup_domain(
            deferred_cleanup_domain
        ));
        drop(lifecycle_tx);
        drop(maintenance_controller);
        drop(dynamic_pools);
        match static_resources {
            PlanRuntimeStatic::NoStatic { .. } => {
                drop(runtime);
                Ok(PlanRuntimeCloseOutcome::Closed(PlanRuntimeCloseReceipt {
                    evidence,
                    released_static_resources: 0,
                }))
            }
            PlanRuntimeStatic::Static(mut static_resources) => {
                drop(runtime);
                let total_static_resources = static_resources.states.len();
                match static_resources.release_all() {
                    Ok(_) => {
                        drop(static_resources);
                        Ok(PlanRuntimeCloseOutcome::Closed(PlanRuntimeCloseReceipt {
                            evidence,
                            released_static_resources: total_static_resources,
                        }))
                    }
                    Err(failure) => Err(PlanRuntimeCloseFailure {
                        failure: failure.into_failure(),
                        evidence,
                        static_resources: Some(static_resources),
                    }),
                }
            }
        }
    }
}

impl<R> TrustedPlanRuntimeBinding<R>
where
    R: DeviceRuntime,
{
    pub(super) fn runtime(&self) -> &Arc<R> {
        &self.resources.runtime
    }

    pub(super) fn logical_admission(&self) -> &LogicalAdmissionCoordinator {
        &self.dynamic_pools().logical_admission
    }

    pub(super) fn dynamic_pools(&self) -> &Arc<DynamicPoolSet<R>> {
        &self.resources.dynamic_pools
    }

    pub(super) fn nodes(&self) -> &[PlanNode] {
        &self.dynamic_pools().nodes
    }

    pub(super) fn reusable_execution_bucket(
        &self,
        bucket_id: &ReusableExecutionBucketId,
    ) -> Option<&ResolvedReusableExecutionBucket> {
        self.dynamic_pools()
            .reusable_execution
            .as_ref()
            .and_then(|plan| plan.bucket(bucket_id))
    }

    pub fn plan_id(&self) -> &PlanId {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => binding.plan_id(),
            PlanRuntimeStatic::Static(source) => source.admission.plan_id(),
        }
    }

    pub fn plan_hash(&self) -> &PlanHash {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => binding.plan_hash(),
            PlanRuntimeStatic::Static(source) => source.admission.plan_hash(),
        }
    }

    pub fn device_id(&self) -> &DeviceId {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => binding.device_id(),
            PlanRuntimeStatic::Static(source) => source.admission.device_id(),
        }
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => {
                binding.device_runtime_implementation_fingerprint()
            }
            PlanRuntimeStatic::Static(source) => {
                source.admission.device_runtime_implementation_fingerprint()
            }
        }
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.logical_admission().id()
    }

    pub fn static_provisioning(&self) -> Option<&StaticProvisioningLease<R>> {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { .. } => None,
            PlanRuntimeStatic::Static(source) => source.lease.as_ref(),
        }
    }

    pub fn evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.resources.evidence()
    }

    pub(super) fn scoped_demand(
        &self,
        lifetime: AllocationLifetime,
        node_id: Option<&NodeId>,
        immediate_shape: DynamicResourceShape,
        fit_shape: DynamicResourceShape,
        reusable_execution_bucket: Option<&ReusableExecutionBucketSpec>,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<(AdmissionDemand, Vec<EvaluatedBackingRequest<'_>>), VNextError> {
        if (lifetime == AllocationLifetime::Invocation) != node_id.is_some() {
            return Err(invalid_resource(
                "invocation resource demand requires one exact node identity",
            ));
        }
        let capacity_shape = match reusable_execution_bucket {
            Some(bucket)
                if matches!(
                    lifetime,
                    AllocationLifetime::Step | AllocationLifetime::Invocation
                ) && bucket.capacity().covers(
                    immediate_shape.sequences(),
                    immediate_shape.tokens(),
                    immediate_shape.pages(),
                ) && bucket.capacity().covers(
                    fit_shape.sequences(),
                    fit_shape.tokens(),
                    fit_shape.pages(),
                ) =>
            {
                DynamicResourceShape::from_validated(
                    bucket.capacity().maximum_sequences(),
                    bucket.capacity().maximum_tokens(),
                    bucket.capacity().maximum_pages(),
                )
            }
            Some(_) => {
                return Err(invalid_resource(
                    "reusable execution bucket does not cover this Step or Invocation demand",
                ));
            }
            None => immediate_shape,
        };
        let node_resources = node_id
            .map(|node_id| {
                self.nodes()
                    .iter()
                    .find(|node| node.id() == node_id)
                    .map(PlanNode::resources)
                    .ok_or_else(|| {
                        invalid_resource("resource admission references an unknown node")
                    })
            })
            .transpose()?;
        let mut immediate_entries = Vec::new();
        let mut fit_entries = Vec::new();
        let mut requested_slices = Vec::new();
        for domain in &self.dynamic_pools().domains {
            let mut immediate_pool_bytes = 0_u64;
            let mut fit_pool_bytes = 0_u64;
            let mut matched = false;
            if lifetime == AllocationLifetime::Step {
                for slot in domain.pool.step_resource_slots() {
                    let mut projections = Vec::with_capacity(slot.resource_ids().len());
                    let mut immediate_slot_bytes = 0_u64;
                    let mut fit_slot_bytes = 0_u64;
                    let mut capacity_slot_bytes = 0_u64;
                    for resource_id in slot.resource_ids() {
                        let descriptor = domain
                            .descriptors
                            .iter()
                            .find(|descriptor| descriptor.base_resource_id() == resource_id)
                            .ok_or_else(|| {
                                invalid_resource(
                                    "step physical slot references a descriptor outside its pool",
                                )
                            })?;
                        if descriptor.lifetime() != AllocationLifetime::Step {
                            return Err(invalid_resource(
                                "step physical slot references a non-Step descriptor",
                            ));
                        }
                        let logical_size_bytes =
                            descriptor.evaluate_request_bytes_for_shape(immediate_shape)?;
                        let fit_bytes = descriptor.evaluate_request_bytes_for_shape(fit_shape)?;
                        let capacity_size_bytes =
                            descriptor.evaluate_request_bytes_for_shape(capacity_shape)?;
                        immediate_slot_bytes = immediate_slot_bytes.max(logical_size_bytes);
                        fit_slot_bytes = fit_slot_bytes.max(fit_bytes);
                        capacity_slot_bytes = capacity_slot_bytes.max(capacity_size_bytes);
                        projections.push(EvaluatedBackingProjection {
                            descriptor,
                            physical_offset_bytes: 0,
                            logical_size_bytes,
                            capacity_size_bytes,
                        });
                    }
                    immediate_pool_bytes = immediate_pool_bytes
                        .checked_add(immediate_slot_bytes)
                        .ok_or_else(|| {
                        invalid_resource("dynamic pool immediate demand overflows u64")
                    })?;
                    fit_pool_bytes = fit_pool_bytes
                        .checked_add(fit_slot_bytes)
                        .ok_or_else(|| invalid_resource("dynamic pool fit demand overflows u64"))?;
                    requested_slices.push(EvaluatedBackingRequest {
                        domain,
                        claim_identity: PhysicalBackingClaimIdentity::new(
                            domain.pool_id().clone(),
                            slot.resource_ids().to_vec(),
                        )?,
                        capacity_size_bytes: capacity_slot_bytes,
                        reusable_execution_bucket_id: reusable_execution_bucket
                            .map(|bucket| bucket.bucket_id().clone()),
                        projections,
                    });
                    matched = true;
                }
            } else {
                for descriptor in &domain.descriptors {
                    if descriptor.lifetime() != lifetime
                        || node_resources.is_some_and(|resources| {
                            !resources.contains(descriptor.base_resource_id())
                        })
                    {
                        continue;
                    }
                    matched = true;
                    let logical_size_bytes =
                        descriptor.evaluate_request_bytes_for_shape(immediate_shape)?;
                    let fit_bytes = descriptor.evaluate_request_bytes_for_shape(fit_shape)?;
                    let capacity_size_bytes =
                        descriptor.evaluate_request_bytes_for_shape(capacity_shape)?;
                    immediate_pool_bytes = immediate_pool_bytes
                        .checked_add(logical_size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("dynamic pool immediate demand overflows u64")
                        })?;
                    fit_pool_bytes = fit_pool_bytes
                        .checked_add(fit_bytes)
                        .ok_or_else(|| invalid_resource("dynamic pool fit demand overflows u64"))?;
                    requested_slices.push(EvaluatedBackingRequest {
                        domain,
                        claim_identity: PhysicalBackingClaimIdentity::new(
                            domain.pool_id().clone(),
                            vec![descriptor.base_resource_id().clone()],
                        )?,
                        capacity_size_bytes,
                        reusable_execution_bucket_id: reusable_execution_bucket
                            .map(|bucket| bucket.bucket_id().clone()),
                        projections: vec![EvaluatedBackingProjection {
                            descriptor,
                            physical_offset_bytes: 0,
                            logical_size_bytes,
                            capacity_size_bytes,
                        }],
                    });
                }
            }
            if matched {
                immediate_entries.push(CapacityEntry::new(
                    domain.domain_id(),
                    CapacityUnits::new(immediate_pool_bytes),
                )?);
                fit_entries.push(CapacityEntry::new(
                    domain.domain_id(),
                    CapacityUnits::new(fit_pool_bytes),
                )?);
            }
        }
        let immediate = if immediate_entries.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(immediate_entries)?
        };
        let fit = if fit_entries.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(fit_entries)?
        };
        Ok((
            AdmissionDemand::from_plan(immediate, fit, fit_policy, pressure_action)?,
            requested_slices,
        ))
    }

    /// Derives the exact additional physical/logical claim needed to advance
    /// one sequence's committed frontier. Existing extents remain owned by the
    /// prior snapshot; only paged storage can append disjoint extents.
    pub(super) fn sequence_extension_demand(
        &self,
        committed: DynamicResourceShape,
        target: DynamicResourceShape,
        pressure_action: AdmissionPressureAction,
    ) -> Result<(AdmissionDemand, Vec<EvaluatedBackingRequest<'_>>), VNextError> {
        if committed.sequences() != 1
            || target.sequences() != 1
            || target.tokens() < committed.tokens()
            || target.pages() < committed.pages()
        {
            return Err(invalid_resource(
                "sequence extension target must monotonically advance one committed sequence",
            ));
        }

        let mut entries = Vec::new();
        let mut requested_slices = Vec::new();
        for domain in &self.dynamic_pools().domains {
            let mut pool_delta = 0_u64;
            for descriptor in &domain.descriptors {
                if descriptor.lifetime() != AllocationLifetime::Sequence {
                    continue;
                }
                let committed_bytes = descriptor.evaluate_request_bytes_for_shape(committed)?;
                let target_bytes = descriptor.evaluate_request_bytes_for_shape(target)?;
                let delta_bytes = target_bytes.checked_sub(committed_bytes).ok_or_else(|| {
                    invalid_resource("sequence extension descriptor capacity regressed")
                })?;
                if delta_bytes == 0 {
                    continue;
                }
                if !matches!(
                    descriptor.storage().profile().view(),
                    super::DynamicStorageView::PagedRegions { .. }
                ) {
                    return Err(invalid_resource(
                        "sequence backing extension requires a paged storage profile",
                    ));
                }
                pool_delta = pool_delta.checked_add(delta_bytes).ok_or_else(|| {
                    invalid_resource("sequence extension pool demand overflows u64")
                })?;
                requested_slices.push(EvaluatedBackingRequest {
                    domain,
                    claim_identity: PhysicalBackingClaimIdentity::new(
                        domain.pool_id().clone(),
                        vec![descriptor.base_resource_id().clone()],
                    )?,
                    capacity_size_bytes: delta_bytes,
                    reusable_execution_bucket_id: None,
                    projections: vec![EvaluatedBackingProjection {
                        descriptor,
                        physical_offset_bytes: 0,
                        logical_size_bytes: delta_bytes,
                        capacity_size_bytes: delta_bytes,
                    }],
                });
            }
            if pool_delta != 0 {
                entries.push(CapacityEntry::new(
                    domain.domain_id(),
                    CapacityUnits::new(pool_delta),
                )?);
            }
        }
        let delta = if entries.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(entries)?
        };
        Ok((
            AdmissionDemand::from_plan(
                delta.clone(),
                delta,
                AdmissionFitPolicy::ImmediateOnly,
                pressure_action,
            )?,
            requested_slices,
        ))
    }

    /// Evaluates all Invocation-scoped resources for one immutable-plan
    /// submission wave. A total-order pool reuses one physical extent across
    /// node rows, while a conservative pool retains disjoint row ranges.
    pub(super) fn submission_wave_demand(
        &self,
        immediate_shape: DynamicResourceShape,
        fit_shape: DynamicResourceShape,
        reusable_execution_bucket: Option<&ReusableExecutionBucketSpec>,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<(AdmissionDemand, Vec<EvaluatedBackingRequest<'_>>), VNextError> {
        let capacity_shape = match reusable_execution_bucket {
            Some(bucket)
                if bucket.capacity().covers(
                    immediate_shape.sequences(),
                    immediate_shape.tokens(),
                    immediate_shape.pages(),
                ) && bucket.capacity().covers(
                    fit_shape.sequences(),
                    fit_shape.tokens(),
                    fit_shape.pages(),
                ) =>
            {
                DynamicResourceShape::from_validated(
                    bucket.capacity().maximum_sequences(),
                    bucket.capacity().maximum_tokens(),
                    bucket.capacity().maximum_pages(),
                )
            }
            Some(_) => {
                return Err(invalid_resource(
                    "reusable execution bucket does not cover the submission-wave work shape",
                ));
            }
            None => immediate_shape,
        };

        let mut immediate_entries = Vec::new();
        let mut fit_entries = Vec::new();
        let mut requested_slices = Vec::new();
        let pools = self.dynamic_pools();
        if pools.domains.len() != pools.submission_wave_layouts.len() {
            return Err(invalid_resource(
                "submission wave layout count differs from immutable plan domains",
            ));
        }
        for (domain, layout) in pools.domains.iter().zip(&pools.submission_wave_layouts) {
            let Some(layout) = layout else {
                continue;
            };
            let mode = domain.pool.invocation_liveness_mode();

            let mut projections = vec![None; layout.projection_count];
            let mut immediate_pool_bytes = 0_u64;
            let mut fit_pool_bytes = 0_u64;
            let mut capacity_pool_bytes = 0_u64;
            for row in &layout.rows {
                let row_base = match mode {
                    InvocationLivenessMode::TotalOrderReuse => 0,
                    InvocationLivenessMode::ConservativeConcurrent => capacity_pool_bytes,
                    InvocationLivenessMode::NoInvocationResources => unreachable!(),
                };
                let mut immediate_row_bytes = 0_u64;
                let mut fit_row_bytes = 0_u64;
                let mut capacity_row_bytes = 0_u64;
                for projection_layout in &row.projections {
                    let descriptor = domain
                        .descriptors
                        .get(projection_layout.descriptor_index)
                        .ok_or_else(|| {
                            invalid_resource(
                                "submission wave layout references a descriptor outside its pool",
                            )
                        })?;
                    if descriptor.lifetime() != AllocationLifetime::Invocation {
                        return Err(invalid_resource(
                            "invocation liveness row references a non-Invocation descriptor",
                        ));
                    }
                    let logical_size_bytes =
                        descriptor.evaluate_request_bytes_for_shape(immediate_shape)?;
                    let fit_bytes = if fit_shape == immediate_shape {
                        logical_size_bytes
                    } else {
                        descriptor.evaluate_request_bytes_for_shape(fit_shape)?
                    };
                    let capacity_size_bytes = if capacity_shape == immediate_shape {
                        logical_size_bytes
                    } else if capacity_shape == fit_shape {
                        fit_bytes
                    } else {
                        descriptor.evaluate_request_bytes_for_shape(capacity_shape)?
                    };
                    let physical_offset_bytes =
                        row_base.checked_add(capacity_row_bytes).ok_or_else(|| {
                            invalid_resource("invocation wave projection offset overflows u64")
                        })?;
                    immediate_row_bytes = immediate_row_bytes
                        .checked_add(logical_size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("invocation wave row demand overflows u64")
                        })?;
                    fit_row_bytes = fit_row_bytes.checked_add(fit_bytes).ok_or_else(|| {
                        invalid_resource("invocation wave fit row demand overflows u64")
                    })?;
                    capacity_row_bytes = capacity_row_bytes
                        .checked_add(capacity_size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("invocation wave capacity row overflows u64")
                        })?;
                    if projections[projection_layout.projection_index]
                        .replace(EvaluatedBackingProjection {
                            descriptor,
                            physical_offset_bytes,
                            logical_size_bytes,
                            capacity_size_bytes,
                        })
                        .is_some()
                    {
                        return Err(invalid_resource(
                            "submission wave layout repeated a canonical projection",
                        ));
                    }
                }
                match mode {
                    InvocationLivenessMode::TotalOrderReuse => {
                        immediate_pool_bytes = immediate_pool_bytes.max(immediate_row_bytes);
                        fit_pool_bytes = fit_pool_bytes.max(fit_row_bytes);
                        capacity_pool_bytes = capacity_pool_bytes.max(capacity_row_bytes);
                    }
                    InvocationLivenessMode::ConservativeConcurrent => {
                        immediate_pool_bytes = immediate_pool_bytes
                            .checked_add(immediate_row_bytes)
                            .ok_or_else(|| {
                                invalid_resource("invocation wave pool demand overflows u64")
                            })?;
                        fit_pool_bytes =
                            fit_pool_bytes.checked_add(fit_row_bytes).ok_or_else(|| {
                                invalid_resource("invocation wave pool fit demand overflows u64")
                            })?;
                        capacity_pool_bytes = capacity_pool_bytes
                            .checked_add(capacity_row_bytes)
                            .ok_or_else(|| {
                            invalid_resource("invocation wave capacity pool demand overflows u64")
                        })?;
                    }
                    InvocationLivenessMode::NoInvocationResources => unreachable!(),
                }
            }
            let projections = projections
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    invalid_resource("submission wave layout left a projection unevaluated")
                })?;
            if projections.is_empty() || immediate_pool_bytes == 0 {
                return Err(invalid_resource(
                    "submission wave layout evaluated to empty invocation demand",
                ));
            }

            requested_slices.push(EvaluatedBackingRequest {
                domain,
                claim_identity: layout.claim_identity.clone(),
                capacity_size_bytes: capacity_pool_bytes,
                reusable_execution_bucket_id: reusable_execution_bucket
                    .map(|bucket| bucket.bucket_id().clone()),
                projections,
            });
            immediate_entries.push(CapacityEntry::new(
                domain.domain_id(),
                CapacityUnits::new(immediate_pool_bytes),
            )?);
            fit_entries.push(CapacityEntry::new(
                domain.domain_id(),
                CapacityUnits::new(fit_pool_bytes),
            )?);
        }

        let immediate = if immediate_entries.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(immediate_entries)?
        };
        let fit = if fit_entries.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(fit_entries)?
        };
        Ok((
            AdmissionDemand::from_plan(immediate, fit, fit_policy, pressure_action)?,
            requested_slices,
        ))
    }

    pub(super) fn prepare_backing_slices(
        &self,
        requested_slices: Vec<EvaluatedBackingRequest<'_>>,
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        self.dynamic_pools().prepare_claim(&requested_slices)
    }

    pub(super) fn prepare_lane_stable_backing_slices(
        &self,
        lane: &Arc<ExecutionLane<R>>,
        requested_slices: Vec<EvaluatedBackingRequest<'_>>,
    ) -> Result<LaneBackingPrepareDecision, VNextError> {
        self.dynamic_pools()
            .prepare_lane_stable_claim(lane, &requested_slices)
    }

    pub(super) fn prepare_initial_sequence_backing_slices(
        &self,
        requested_slices: &[EvaluatedBackingRequest<'_>],
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        self.dynamic_pools()
            .prepare_initial_sequence_claim(requested_slices)
    }

    pub(super) fn register_backing_waiter(
        &self,
        deferred: &DynamicBackingDeferred,
    ) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.resources
            .register_capacity_waiter(deferred.wait_condition())
    }

    pub fn register_admission_waiter(
        &self,
        deferred: &AdmissionDeferred,
    ) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.resources
            .register_capacity_waiter(deferred.wait_condition())
    }
}
