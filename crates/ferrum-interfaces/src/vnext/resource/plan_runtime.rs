use super::{
    core_resource_failure, deferred_device_cleanup_status, invalid_resource,
    maintain_deferred_device_cleanups, new_deferred_device_cleanup_domain,
    retire_deferred_device_cleanup_domain, watch, AdmissionDeferred, AdmissionDemand,
    AdmissionFitPolicy, AdmissionPressureAction, AllocationLifetime, Arc, AtomicU8,
    BackingPrepareDecision, CapacityEntry, CapacityEpochs, CapacityUnits, CapacityVector,
    CapacityWaitRecheck, CapacityWaitRegistration, DeferredDeviceCleanupDomainId,
    DeferredDeviceCleanupMaintenanceReceipt, DeferredDeviceCleanupStatus, DeviceCapacityClaim,
    DeviceId, DeviceRuntime, DynamicBackingDeferred, DynamicDeferredMaintenanceOutcome,
    DynamicPoolMaintenanceController, DynamicPoolSet, DynamicResourceShape,
    EvaluatedBackingProjection, EvaluatedBackingRequest, FailureEnvelope, InvocationLivenessMode,
    LogicalAdmissionCoordinator, LogicalAdmissionCoordinatorId, Mutex, NoStatic, NodeId, Ordering,
    PhysicalBackingClaimIdentity, PlanHash, PlanId, PlanNode, ResourceAbandonSignal,
    ResourceActionCursor, ResourceDriverFailure, ResourceLedgerEntrySnapshot,
    ResourceOwnershipReason, ResourceOwnershipTransferFailure, ResourcePoolIdentity,
    ResourcePoolOwnership, ResourceReservation, ResourceReservationBatch,
    ResourceTransactionAction, ResourceTransactionContext, ResourceTransactionDriver,
    ResourceTransactionIdentity, ResourceTransactionState, RwLock, RwLockReadGuard, Serialize,
    StaticProvisioningBinding, StaticProvisioningLease, VNextError,
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

/// A capacity wait registration that keeps its exact plan runtime alive until
/// the waiter either observes a retry epoch or is cancelled by being dropped.
#[must_use = "capacity wait registrations must be awaited, rechecked, or dropped"]
pub struct PlanCapacityWaitRegistration<R>
where
    R: DeviceRuntime,
{
    registration: CapacityWaitRegistration,
    lifecycle_rx: watch::Receiver<u8>,
    resources: Arc<PlanRuntimeResources<R>>,
}

impl<R> PlanCapacityWaitRegistration<R>
where
    R: DeviceRuntime,
{
    pub fn recheck(&self) -> Result<CapacityWaitRecheck, VNextError> {
        let _lifecycle = self.resources.read_lifecycle("recheck a capacity waiter")?;
        self.registration.recheck()
    }

    pub async fn wait_for_change(self) -> Result<CapacityEpochs, VNextError> {
        let Self {
            registration,
            mut lifecycle_rx,
            resources,
        } = self;
        let admission_wait = registration.wait_for_change();
        tokio::pin!(admission_wait);
        tokio::select! {
            result = &mut admission_wait => result,
            changed = lifecycle_rx.changed() => {
                changed.map_err(|_| invalid_resource(
                    "plan runtime lifecycle signal closed while a capacity waiter was live",
                ))?;
                if *lifecycle_rx.borrow_and_update() == PLAN_RUNTIME_CLOSING {
                    Err(invalid_resource(
                        "closing plan runtime cancelled its capacity waiter",
                    ))
                } else {
                    drop(resources);
                    Err(invalid_resource(
                        "capacity waiter observed an invalid plan runtime lifecycle transition",
                    ))
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

    /// Resolves physical backing pressure for this plan while retaining the
    /// owning root for the full maintenance operation. A close racing before
    /// the second phase check is rejected; a close racing after it observes
    /// this operation's `Arc` and cannot release the root underneath it.
    pub fn maintain_for_deferred(
        self: &Arc<Self>,
        deferred: &DynamicBackingDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let _lifecycle = self.read_lifecycle("maintain deferred dynamic backing")?;
        self.maintenance_controller.maintain_for_deferred(deferred)
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
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<(AdmissionDemand, Vec<EvaluatedBackingRequest<'_>>), VNextError> {
        if (lifetime == AllocationLifetime::Invocation) != node_id.is_some() {
            return Err(invalid_resource(
                "invocation resource demand requires one exact node identity",
            ));
        }
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
                        let size_bytes =
                            descriptor.evaluate_request_bytes_for_shape(immediate_shape)?;
                        let fit_bytes = descriptor.evaluate_request_bytes_for_shape(fit_shape)?;
                        immediate_slot_bytes = immediate_slot_bytes.max(size_bytes);
                        fit_slot_bytes = fit_slot_bytes.max(fit_bytes);
                        projections.push(EvaluatedBackingProjection {
                            descriptor,
                            physical_offset_bytes: 0,
                            size_bytes,
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
                        size_bytes: immediate_slot_bytes,
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
                    let size_bytes =
                        descriptor.evaluate_request_bytes_for_shape(immediate_shape)?;
                    let fit_bytes = descriptor.evaluate_request_bytes_for_shape(fit_shape)?;
                    immediate_pool_bytes = immediate_pool_bytes
                        .checked_add(size_bytes)
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
                        size_bytes,
                        projections: vec![EvaluatedBackingProjection {
                            descriptor,
                            physical_offset_bytes: 0,
                            size_bytes,
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
                    size_bytes: delta_bytes,
                    projections: vec![EvaluatedBackingProjection {
                        descriptor,
                        physical_offset_bytes: 0,
                        size_bytes: delta_bytes,
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
        node_shapes: &[(NodeId, DynamicResourceShape, DynamicResourceShape)],
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<(AdmissionDemand, Vec<EvaluatedBackingRequest<'_>>), VNextError> {
        if node_shapes.is_empty()
            || node_shapes.len() != self.nodes().len()
            || node_shapes
                .iter()
                .zip(self.nodes())
                .any(|((node_id, _, _), node)| node_id != node.id())
        {
            return Err(invalid_resource(
                "submission wave demand must cover every plan node in immutable plan order",
            ));
        }

        let mut immediate_entries = Vec::new();
        let mut fit_entries = Vec::new();
        let mut requested_slices = Vec::new();
        for domain in &self.dynamic_pools().domains {
            let mode = domain.pool.invocation_liveness_mode();
            if mode == InvocationLivenessMode::NoInvocationResources {
                continue;
            }

            let mut projections = Vec::new();
            let mut immediate_pool_bytes = 0_u64;
            let mut fit_pool_bytes = 0_u64;
            let mut matched_rows = 0_usize;
            for (node_id, immediate_shape, fit_shape) in node_shapes {
                let Some(row) = domain
                    .pool
                    .invocation_liveness()
                    .iter()
                    .find(|row| row.node_id() == node_id)
                else {
                    continue;
                };
                matched_rows += 1;
                let row_base = match mode {
                    InvocationLivenessMode::TotalOrderReuse => 0,
                    InvocationLivenessMode::ConservativeConcurrent => immediate_pool_bytes,
                    InvocationLivenessMode::NoInvocationResources => unreachable!(),
                };
                let mut immediate_row_bytes = 0_u64;
                let mut fit_row_bytes = 0_u64;
                for resource_id in row.resource_ids() {
                    let descriptor = domain
                        .descriptors
                        .iter()
                        .find(|descriptor| descriptor.base_resource_id() == resource_id)
                        .ok_or_else(|| {
                            invalid_resource(
                                "invocation liveness row references a descriptor outside its pool",
                            )
                        })?;
                    if descriptor.lifetime() != AllocationLifetime::Invocation {
                        return Err(invalid_resource(
                            "invocation liveness row references a non-Invocation descriptor",
                        ));
                    }
                    let size_bytes =
                        descriptor.evaluate_request_bytes_for_shape(*immediate_shape)?;
                    let fit_bytes = descriptor.evaluate_request_bytes_for_shape(*fit_shape)?;
                    let physical_offset_bytes =
                        row_base.checked_add(immediate_row_bytes).ok_or_else(|| {
                            invalid_resource("invocation wave projection offset overflows u64")
                        })?;
                    immediate_row_bytes =
                        immediate_row_bytes.checked_add(size_bytes).ok_or_else(|| {
                            invalid_resource("invocation wave row demand overflows u64")
                        })?;
                    fit_row_bytes = fit_row_bytes.checked_add(fit_bytes).ok_or_else(|| {
                        invalid_resource("invocation wave fit row demand overflows u64")
                    })?;
                    projections.push(EvaluatedBackingProjection {
                        descriptor,
                        physical_offset_bytes,
                        size_bytes,
                    });
                }
                match mode {
                    InvocationLivenessMode::TotalOrderReuse => {
                        immediate_pool_bytes = immediate_pool_bytes.max(immediate_row_bytes);
                        fit_pool_bytes = fit_pool_bytes.max(fit_row_bytes);
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
                    }
                    InvocationLivenessMode::NoInvocationResources => unreachable!(),
                }
            }
            if matched_rows != domain.pool.invocation_liveness().len()
                || projections.is_empty()
                || immediate_pool_bytes == 0
            {
                return Err(invalid_resource(
                    "invocation liveness rows do not map exactly onto the submission wave",
                ));
            }

            projections.sort_by(|left, right| {
                left.descriptor
                    .base_resource_id()
                    .cmp(right.descriptor.base_resource_id())
            });
            let resource_ids = projections
                .iter()
                .map(|projection| projection.descriptor.base_resource_id().clone())
                .collect::<Vec<_>>();
            requested_slices.push(EvaluatedBackingRequest {
                domain,
                claim_identity: PhysicalBackingClaimIdentity::new(
                    domain.pool_id().clone(),
                    resource_ids,
                )?,
                size_bytes: immediate_pool_bytes,
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

    pub fn register_backing_waiter(
        &self,
        deferred: &DynamicBackingDeferred,
    ) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        let _lifecycle = self.resources.read_lifecycle("register a backing waiter")?;
        let lifecycle_rx = self.resources.lifecycle_tx.subscribe();
        let registration = self
            .logical_admission()
            .register_waiter(deferred.epochs())?;
        Ok(PlanCapacityWaitRegistration {
            registration,
            lifecycle_rx,
            resources: Arc::clone(&self.resources),
        })
    }

    pub fn register_admission_waiter(
        &self,
        deferred: &AdmissionDeferred,
    ) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        let _lifecycle = self
            .resources
            .read_lifecycle("register an admission waiter")?;
        let lifecycle_rx = self.resources.lifecycle_tx.subscribe();
        let registration = self
            .logical_admission()
            .register_waiter(deferred.epochs())?;
        Ok(PlanCapacityWaitRegistration {
            registration,
            lifecycle_rx,
            resources: Arc::clone(&self.resources),
        })
    }
}
