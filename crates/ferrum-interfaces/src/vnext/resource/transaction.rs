use super::{
    expected_lease_transition, expected_transition, fmt, invalid_resource,
    new_deferred_device_cleanup_domain, watch, Arc, AtomicBool, AtomicU8, BTreeMap, BTreeSet,
    BufferDescriptor, CoreOwnedAllocation, DeviceCapacityClaim, DriverCommitAcknowledgement,
    DynamicPoolMaintenanceController, DynamicPoolSet, ErasedPlanStaticDriver, FailureDomain,
    FailureEnvelope, Mutex, Ordering, PhantomData, PlanRuntimeResources, PlanRuntimeStatic,
    PlanStaticResources, RefCell, ResourceAbandonSignal, ResourceActionCursor, ResourceCommitView,
    ResourceCompensationRecord, ResourceDriverFailure, ResourceFailureId, ResourceFailurePoint,
    ResourceFailureReceipt, ResourceId, ResourceLeaseAction, ResourceLeaseEntry,
    ResourceLeaseState, ResourceLeaseTransitionReceipt, ResourceLeaseValidationContext,
    ResourceLedgerEntrySnapshot, ResourceLedgerSnapshot, ResourceOwnedBuffer,
    ResourceOwnershipReason, ResourcePoolOwnership, ResourceRecoveryStrategy,
    ResourceReservationBatch, ResourceTransactionAction, ResourceTransactionContext,
    ResourceTransactionDriver, ResourceTransactionIdentity, ResourceTransactionState,
    ResourceTransitionReceipt, ResourceTransitionRecord, ResourceTransitionValidationContext,
    RwLock, StaticProvisioningBinding, StaticProvisioningLease, StaticProvisioningPermit,
    VNextError, PLAN_RUNTIME_OPEN,
};

mod sealed {
    pub trait Sealed {}
}

pub trait TransactionStage: sealed::Sealed {
    const STATE: ResourceTransactionState;
    const TERMINAL: bool;
}

pub struct TransactionNew;
pub struct TransactionReserved;
pub struct TransactionCommitted;
pub struct TransactionRolledBack;
pub struct TransactionReleased;
pub struct TransactionQuarantined;

impl sealed::Sealed for TransactionNew {}
impl sealed::Sealed for TransactionReserved {}
impl sealed::Sealed for TransactionCommitted {}
impl sealed::Sealed for TransactionRolledBack {}
impl sealed::Sealed for TransactionReleased {}
impl sealed::Sealed for TransactionQuarantined {}

impl TransactionStage for TransactionNew {
    const STATE: ResourceTransactionState = ResourceTransactionState::New;
    const TERMINAL: bool = false;
}
impl TransactionStage for TransactionReserved {
    const STATE: ResourceTransactionState = ResourceTransactionState::Reserved;
    const TERMINAL: bool = false;
}
impl TransactionStage for TransactionCommitted {
    const STATE: ResourceTransactionState = ResourceTransactionState::Committed;
    const TERMINAL: bool = false;
}
impl TransactionStage for TransactionRolledBack {
    const STATE: ResourceTransactionState = ResourceTransactionState::RolledBack;
    const TERMINAL: bool = true;
}
impl TransactionStage for TransactionReleased {
    const STATE: ResourceTransactionState = ResourceTransactionState::Released;
    const TERMINAL: bool = true;
}
impl TransactionStage for TransactionQuarantined {
    const STATE: ResourceTransactionState = ResourceTransactionState::Quarantined;
    const TERMINAL: bool = true;
}

struct PendingSubsetRelease {
    target_orders: Vec<usize>,
    failure: ResourceFailureReceipt,
}

#[must_use = "dropping a nonterminal resource transaction emits an abandon signal"]
pub struct ResourceTransaction<D: ResourceTransactionDriver, S: TransactionStage> {
    driver: Option<D>,
    maintenance_controller: Option<DynamicPoolMaintenanceController<D::Runtime>>,
    dynamic_pools: Option<Arc<DynamicPoolSet<D::Runtime>>>,
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    reservations: ResourceReservationBatch,
    capacity_claim: Option<DeviceCapacityClaim>,
    allocation_issued: Vec<AtomicBool>,
    pending_allocations: Vec<RefCell<Option<CoreOwnedAllocation<D::Buffer>>>>,
    states: Vec<ResourceTransactionState>,
    lease: Option<StaticProvisioningLease<D::Runtime>>,
    receipts: Vec<ResourceTransitionReceipt>,
    lease_receipts: Vec<ResourceLeaseTransitionReceipt>,
    recovery_history: Vec<ResourceFailureReceipt>,
    latest_transition_context: Option<ResourceTransitionValidationContext>,
    latest_lease_context: Option<ResourceLeaseValidationContext>,
    pending_failure: Option<ResourceFailureReceipt>,
    pending_subset_release: Option<PendingSubsetRelease>,
    next_failure_id: u64,
    finalized: bool,
    stage: PhantomData<S>,
}

impl<D: ResourceTransactionDriver, S: TransactionStage> ResourceTransaction<D, S> {
    pub fn maintenance_controller(&self) -> &DynamicPoolMaintenanceController<D::Runtime> {
        self.maintenance_controller
            .as_ref()
            .expect("live resource transaction owns dynamic-pool maintenance authority")
    }

    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub fn reservations(&self) -> &ResourceReservationBatch {
        &self.reservations
    }

    pub fn receipts(&self) -> &[ResourceTransitionReceipt] {
        &self.receipts
    }

    pub fn lease_receipts(&self) -> &[ResourceLeaseTransitionReceipt] {
        &self.lease_receipts
    }

    pub fn recovery_history(&self) -> &[ResourceFailureReceipt] {
        &self.recovery_history
    }

    pub const fn state(&self) -> ResourceTransactionState {
        S::STATE
    }

    pub fn actual_states(&self) -> &[ResourceTransactionState] {
        &self.states
    }

    pub fn ledger_snapshot(&self) -> ResourceLedgerSnapshot {
        ResourceLedgerSnapshot {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            entries: self.ledger_snapshot_entries(),
        }
    }

    pub fn latest_transition_validation_context(
        &self,
    ) -> Option<&ResourceTransitionValidationContext> {
        self.latest_transition_context.as_ref()
    }

    pub fn latest_lease_validation_context(&self) -> Option<&ResourceLeaseValidationContext> {
        self.latest_lease_context.as_ref()
    }

    fn ledger_snapshot_entries(&self) -> Vec<ResourceLedgerEntrySnapshot> {
        let lease = self
            .lease
            .as_ref()
            .expect("every resource transaction owns its batch lease ledger");
        lease
            .slots
            .iter()
            .zip(&self.pending_allocations)
            .zip(&self.states)
            .map(|((slot, pending), &transaction_state)| {
                let pending = pending.borrow();
                ResourceLedgerEntrySnapshot {
                    entry: slot.entry.clone(),
                    transaction_state,
                    buffer_present: slot.buffer.is_some() || pending.is_some(),
                    actual_resource_id: slot
                        .actual_resource_id
                        .clone()
                        .or_else(|| pending.as_ref().map(|value| value.resource_id.clone())),
                    actual_generation: slot
                        .actual_generation
                        .or_else(|| pending.as_ref().map(|value| value.generation)),
                    actual_descriptor: slot
                        .descriptor
                        .clone()
                        .or_else(|| pending.as_ref().map(|value| value.descriptor.clone())),
                }
            })
            .collect()
    }

    fn driver_and_context(
        &mut self,
        order: usize,
        action: ResourceTransactionAction,
    ) -> (&mut D, ResourceTransactionContext<'_, D::Runtime>) {
        let before = self.states[order];
        let driver = self
            .driver
            .as_mut()
            .expect("live transaction owns its resource driver");
        let context = ResourceTransactionContext {
            runtime: &self
                .lease
                .as_ref()
                .expect("transaction owns lease ledger")
                .runtime,
            identity: &self.identity,
            binding: &self.admission,
            reservations: &self.reservations,
            cursor: Some(ResourceActionCursor {
                order,
                action,
                before,
                allocation_authorized: action == ResourceTransactionAction::Commit,
            }),
            allocation_authority: (action == ResourceTransactionAction::Commit)
                .then_some(&self.allocation_issued[order]),
            pending_allocation: (action == ResourceTransactionAction::Commit)
                .then_some(&self.pending_allocations[order]),
        };
        (driver, context)
    }

    fn make_record(
        &self,
        order: usize,
        action: ResourceTransactionAction,
        before: ResourceTransactionState,
        after: ResourceTransactionState,
    ) -> ResourceTransitionRecord {
        ResourceTransitionRecord::from_reservation(
            &self.identity,
            &self.admission,
            &self.reservations.reservations[order],
            action,
            before,
            after,
            order,
        )
    }

    fn transition_receipt(
        &mut self,
        action: ResourceTransactionAction,
        before: Vec<ResourceLedgerEntrySnapshot>,
        records: Vec<ResourceTransitionRecord>,
    ) -> Result<ResourceTransitionReceipt, VNextError> {
        let context = ResourceTransitionValidationContext {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            action,
            before,
            after: self.ledger_snapshot_entries(),
        };
        let receipt = ResourceTransitionReceipt::from_context(&context, records)?;
        self.latest_transition_context = Some(context);
        Ok(receipt)
    }

    fn lease_transition(
        &mut self,
        resource_ids: &[ResourceId],
        action: ResourceLeaseAction,
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        if self.pending_subset_release.is_some() {
            return Err(invalid_resource(
                "lease policy cannot change during pending release recovery",
            ));
        }
        let orders = self.orders_for_ids(resource_ids)?;
        if orders
            .iter()
            .any(|&order| self.states[order] != ResourceTransactionState::Committed)
        {
            return Err(invalid_resource(
                "lease policy targets a resource that is not actually committed",
            ));
        }
        let before_ledger = self.ledger_snapshot_entries();
        let (before, after, entries) = self
            .lease
            .as_mut()
            .expect("transaction owns lease ledger")
            .transition_subset(&orders, action)?;
        let context = ResourceLeaseValidationContext {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            action,
            before: before_ledger,
            after: self.ledger_snapshot_entries(),
        };
        let receipt =
            ResourceLeaseTransitionReceipt::from_context(&context, before, after, entries)?;
        self.latest_lease_context = Some(context);
        self.lease_receipts.push(receipt.clone());
        Ok(receipt)
    }

    fn orders_for_ids(&self, resource_ids: &[ResourceId]) -> Result<Vec<usize>, VNextError> {
        if resource_ids.is_empty() {
            return Err(invalid_resource("resource subset must not be empty"));
        }
        let mut unique = BTreeSet::new();
        let by_id = self
            .reservations
            .reservations()
            .iter()
            .enumerate()
            .map(|(order, reservation)| (reservation.resource_id(), order))
            .collect::<BTreeMap<_, _>>();
        let mut orders = Vec::with_capacity(resource_ids.len());
        for resource_id in resource_ids {
            if !unique.insert(resource_id) {
                return Err(invalid_resource("resource subset contains duplicates"));
            }
            orders.push(
                *by_id
                    .get(resource_id)
                    .ok_or_else(|| invalid_resource("resource subset is not admitted"))?,
            );
        }
        orders.sort_unstable();
        Ok(orders)
    }

    fn set_pending_failure(&mut self, failure: &ResourceFailureReceipt) {
        self.pending_failure = Some(failure.clone());
    }

    fn clear_pending_failure(&mut self) {
        self.pending_failure = None;
    }

    fn issue_failure_id(&mut self) -> ResourceFailureId {
        let current = self.next_failure_id;
        self.next_failure_id = current
            .checked_add(1)
            .expect("resource transaction failure id space is exhausted");
        ResourceFailureId::try_from(current).expect("transaction failure ids start at one")
    }

    fn advance<T: TransactionStage>(
        mut self,
        receipt: Option<ResourceTransitionReceipt>,
    ) -> ResourceTransaction<D, T> {
        self.finalized = true;
        if let Some(receipt) = receipt {
            self.receipts.push(receipt);
        }
        ResourceTransaction {
            driver: self.driver.take(),
            maintenance_controller: self.maintenance_controller.take(),
            dynamic_pools: self.dynamic_pools.take(),
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            reservations: self.reservations.clone(),
            capacity_claim: if T::TERMINAL {
                if let Some(mut claim) = self.capacity_claim.take() {
                    claim.release();
                }
                None
            } else {
                self.capacity_claim.take()
            },
            allocation_issued: std::mem::take(&mut self.allocation_issued),
            pending_allocations: std::mem::take(&mut self.pending_allocations),
            states: std::mem::take(&mut self.states),
            lease: self.lease.take(),
            receipts: std::mem::take(&mut self.receipts),
            lease_receipts: std::mem::take(&mut self.lease_receipts),
            recovery_history: std::mem::take(&mut self.recovery_history),
            latest_transition_context: self.latest_transition_context.take(),
            latest_lease_context: self.latest_lease_context.take(),
            pending_failure: None,
            pending_subset_release: None,
            next_failure_id: self.next_failure_id,
            finalized: T::TERMINAL,
            stage: PhantomData,
        }
    }

    fn abandon_signal(&self) -> ResourceAbandonSignal {
        ResourceAbandonSignal {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            state: S::STATE,
            pending_action: self
                .pending_failure
                .as_ref()
                .map(ResourceFailureReceipt::action),
            ledger: self.ledger_snapshot_entries(),
            active_sequence_slots: Vec::new(),
            poisoned_sequence_slots: Vec::new(),
            undrained_sequence_slots: Vec::new(),
            failure: self
                .pending_failure
                .as_ref()
                .map(|receipt| receipt.failure.clone()),
        }
    }

    fn take_all_owned_buffers(&mut self) -> Vec<ResourceOwnedBuffer<D::Buffer>> {
        let mut buffers = self
            .lease
            .as_mut()
            .expect("transaction owns lease ledger")
            .take_owned_buffers(&self.reservations);
        for (order, pending) in self.pending_allocations.iter_mut().enumerate() {
            let Some(allocation) = pending.get_mut().take() else {
                continue;
            };
            let reservation = &self.reservations.reservations[order];
            buffers.push(ResourceOwnedBuffer {
                order,
                expected_resource_id: reservation.resource_id.clone(),
                actual_resource_id: allocation.resource_id,
                expected_generation: reservation.generation,
                actual_generation: allocation.generation,
                expected_descriptor: BufferDescriptor {
                    resource_id: reservation.resource_id.clone(),
                    size_bytes: reservation.size_bytes,
                    alignment_bytes: reservation.alignment_bytes,
                    usage: reservation.usage,
                    element_type: reservation.element_type,
                },
                actual_descriptor: allocation.descriptor,
                buffer: allocation.buffer,
            });
        }
        buffers.sort_by_key(|buffer| buffer.order);
        buffers
    }

    fn quarantine_live(&mut self) -> Result<Vec<ResourceTransitionRecord>, ResourceDriverFailure> {
        let buffers = self.take_all_owned_buffers();
        let ownership = ResourcePoolOwnership {
            runtime: Arc::clone(
                &self
                    .lease
                    .as_ref()
                    .expect("transaction owns lease ledger")
                    .runtime,
            ),
            pool_identity: self.admission.pool_identity.clone(),
            reason: ResourceOwnershipReason::Quarantine,
            signal: None,
            buffers,
            capacity_claim: self.capacity_claim.take(),
        };
        let result = {
            let driver = self
                .driver
                .as_mut()
                .expect("live transaction owns its resource driver");
            let context = ResourceTransactionContext {
                runtime: &self
                    .lease
                    .as_ref()
                    .expect("transaction owns lease ledger")
                    .runtime,
                identity: &self.identity,
                binding: &self.admission,
                reservations: &self.reservations,
                cursor: None,
                allocation_authority: None,
                pending_allocation: None,
            };
            driver.quarantine_transaction(&context, ownership)
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
                    "quarantine failure returned ownership for a different resource pool",
                    false,
                ))
                .expect("core failure has resource domain"));
            }
            self.lease
                .as_mut()
                .expect("transaction owns lease ledger")
                .restore_owned_buffers(std::mem::take(&mut ownership.buffers));
            self.capacity_claim = ownership.capacity_claim.take();
            return Err(failure);
        }

        let mut records = Vec::new();
        for order in 0..self.states.len() {
            let before = self.states[order];
            if before.is_live() {
                self.states[order] = ResourceTransactionState::Quarantined;
                records.push(self.make_record(
                    order,
                    ResourceTransactionAction::Quarantine,
                    before,
                    ResourceTransactionState::Quarantined,
                ));
            }
        }
        Ok(records)
    }
}

impl<D: ResourceTransactionDriver, S: TransactionStage> Drop for ResourceTransaction<D, S> {
    fn drop(&mut self) {
        if self.finalized || S::TERMINAL {
            return;
        }
        let signal = self.abandon_signal();
        drop(self.maintenance_controller.take());
        drop(self.dynamic_pools.take());
        let buffers = self.take_all_owned_buffers();
        let ownership = ResourcePoolOwnership {
            runtime: Arc::clone(
                &self
                    .lease
                    .as_ref()
                    .expect("transaction owns lease ledger")
                    .runtime,
            ),
            pool_identity: self.admission.pool_identity.clone(),
            reason: ResourceOwnershipReason::Abandon,
            signal: Some(signal),
            buffers,
            capacity_claim: self.capacity_claim.take(),
        };
        if let Some(driver) = self.driver.as_mut() {
            // Driver cleanup is an observation/transfer hook, not permission
            // to turn an existing unwind into a process abort. Ownership that
            // drops inside a panicking callback retains its backend lifetimes.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                driver.abandon_transaction(ownership);
            }));
        } else {
            // A live transaction always owns its driver. Leaking is safer than
            // returning capacity while backend ownership is unknown.
            std::mem::forget(ownership);
        }
        self.finalized = true;
    }
}

pub(super) fn core_resource_failure(
    code: &'static str,
    message: impl Into<String>,
    retryable: bool,
) -> FailureEnvelope {
    FailureEnvelope::new(FailureDomain::Resource, code, message, retryable)
        .expect("core-generated resource failure must be valid")
}

impl<D: ResourceTransactionDriver> ResourceTransaction<D, TransactionNew> {
    pub fn begin(
        driver: D,
        identity: ResourceTransactionIdentity,
        permit: StaticProvisioningPermit<D::Runtime>,
    ) -> Result<Self, VNextError> {
        if identity.pool_id != permit.binding.pool_id()
            || identity.request_id != permit.binding.request_id
            || permit.binding.pool_identity.plan_id != permit.binding.plan_id
            || permit.binding.pool_identity.plan_hash != permit.binding.plan_hash
            || permit.binding.pool_identity.device_id != permit.binding.device_id
            || permit
                .binding
                .pool_identity
                .device_runtime_implementation_fingerprint
                != permit.binding.device_runtime_implementation_fingerprint
            || permit.binding.pool_identity.admission_generation
                != permit.binding.admission_generation
            || permit.binding.request_id != *permit.reservations.request_id()
            || permit.binding.admitted_bytes != permit.reservations.total_size_bytes()
            || permit.binding.plan_static_bytes != permit.reservations.plan_static_size_bytes()
            || permit.binding.admitted_bytes != permit.binding.plan_static_bytes
            || permit.binding.plan_static_bytes > permit.binding.usable_capacity_bytes
            || permit.binding.usable_capacity_bytes > permit.binding.device_capacity_bytes
            || driver.device_id() != &permit.binding.device_id
            || driver.device_runtime_implementation_fingerprint()
                != permit.binding.device_runtime_implementation_fingerprint
            || driver.device_capacity_bytes() != permit.binding.device_capacity_bytes
            || !Arc::ptr_eq(driver.runtime(), &permit.runtime)
        {
            return Err(invalid_resource(
                "transaction identity, driver device, or admitted capacity does not match its permit",
            ));
        }
        let _ = permit.seal;
        if permit.capacity_claim.bytes() != permit.binding.admitted_bytes {
            return Err(invalid_resource(
                "device capacity claim differs from admitted bytes",
            ));
        }
        let states = vec![ResourceTransactionState::New; permit.reservations.reservations.len()];
        let allocation_issued = (0..states.len()).map(|_| AtomicBool::new(false)).collect();
        let pending_allocations = (0..states.len()).map(|_| RefCell::new(None)).collect();
        let lease = StaticProvisioningLease::new(
            Arc::clone(&permit.runtime),
            &identity,
            &permit.binding,
            &permit.reservations,
        );
        Ok(Self {
            driver: Some(driver),
            maintenance_controller: Some(permit.maintenance_controller),
            dynamic_pools: Some(permit.dynamic_pools),
            identity,
            admission: permit.binding,
            reservations: permit.reservations,
            capacity_claim: Some(permit.capacity_claim),
            allocation_issued,
            pending_allocations,
            states,
            lease: Some(lease),
            receipts: Vec::new(),
            lease_receipts: Vec::new(),
            recovery_history: Vec::new(),
            latest_transition_context: None,
            latest_lease_context: None,
            pending_failure: None,
            pending_subset_release: None,
            next_failure_id: 1,
            finalized: false,
            stage: PhantomData,
        })
    }

    pub fn reserve(
        mut self,
    ) -> Result<
        ResourceTransaction<D, TransactionReserved>,
        ResourcePrepareTransitionError<D, TransactionNew>,
    > {
        debug_assert!(
            self.states
                .iter()
                .all(|state| *state == ResourceTransactionState::New),
            "new transaction ledger must be uniformly new"
        );
        let ledger_before = self.ledger_snapshot_entries();
        let mut completed = Vec::new();
        for order in 0..self.states.len() {
            let result = {
                let reservation = self.reservations.reservations[order].clone();
                let (driver, context) =
                    self.driver_and_context(order, ResourceTransactionAction::Reserve);
                driver.reserve_resource(&context, &reservation)
            };
            match result {
                Ok(()) => {
                    let before = self.states[order];
                    let after = before
                        .transition(
                            &self.reservations.reservations[order].resource_id,
                            ResourceTransactionAction::Reserve,
                        )
                        .expect("reserve was preflight validated");
                    self.states[order] = after;
                    completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Reserve,
                        before,
                        after,
                    ));
                }
                Err(failure) => {
                    let failure_id = self.issue_failure_id();
                    let receipt = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Reserve,
                        failure.into_failure(),
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ReverseCompensation,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&receipt);
                    return Err(ResourcePrepareTransitionError {
                        transaction: Some(self),
                        failure: receipt,
                    });
                }
            }
        }
        let receipt = self
            .transition_receipt(ResourceTransactionAction::Reserve, ledger_before, completed)
            .expect("core reserve journal must validate");
        Ok(self.advance(Some(receipt)))
    }
}

#[derive(Debug)]
pub enum ResourceCommitTransitionError<D: ResourceTransactionDriver> {
    Recoverable(ResourcePrepareTransitionError<D, TransactionReserved>),
    Poisoned(ResourcePoisonedTransaction<D>),
}

impl<D: ResourceTransactionDriver> ResourceTransaction<D, TransactionReserved> {
    pub fn commit(
        mut self,
    ) -> Result<ResourceTransaction<D, TransactionCommitted>, ResourceCommitTransitionError<D>>
    {
        debug_assert!(
            self.states
                .iter()
                .all(|state| *state == ResourceTransactionState::Reserved),
            "reserved transaction must be uniformly reserved before commit"
        );
        let ledger_before = self.ledger_snapshot_entries();
        let mut completed = Vec::new();
        for order in 0..self.states.len() {
            let driver_result = {
                let reservation = self.reservations.reservations[order].clone();
                let (driver, context) =
                    self.driver_and_context(order, ResourceTransactionAction::Commit);
                driver
                    .commit_resource(&context, &reservation)
                    .map(|receipt| DriverCommitAcknowledgement::from_receipt(&receipt))
            };
            let allocation = self.pending_allocations[order].get_mut().take();
            match (driver_result, allocation) {
                (Ok(acknowledgement), Some(allocation))
                    if acknowledgement.matches(&allocation)
                        && allocation.matches(&self.reservations.reservations[order]) =>
                {
                    self.lease
                        .as_mut()
                        .expect("transaction owns lease ledger")
                        .install(order, allocation);
                    let before = self.states[order];
                    let after = before
                        .transition(
                            &self.reservations.reservations[order].resource_id,
                            ResourceTransactionAction::Commit,
                        )
                        .expect("commit was preflight validated");
                    self.states[order] = after;
                    completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Commit,
                        before,
                        after,
                    ));
                }
                (driver_result, Some(poisoned)) => {
                    let actual_resource_id = poisoned.resource_id.clone();
                    let actual_generation = poisoned.generation;
                    let failure_envelope = match driver_result {
                        Ok(_) => core_resource_failure(
                            "invalid_commit_outcome",
                            format!(
                                "core-owned allocation `{}` generation {} does not match allocation `{}` generation {}",
                                actual_resource_id,
                                actual_generation,
                                self.reservations.reservations[order].resource_id(),
                                self.reservations.reservations[order].generation(),
                            ),
                            false,
                        ),
                        Err(failure) => failure.into_failure(),
                    };
                    self.lease
                        .as_mut()
                        .expect("transaction owns lease ledger")
                        .install(order, poisoned);
                    let failure_id = self.issue_failure_id();
                    let failure = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Commit,
                        failure_envelope,
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ReconcileOrQuarantine,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&failure);
                    return Err(ResourceCommitTransitionError::Poisoned(
                        ResourcePoisonedTransaction {
                            transaction: Some(self),
                            failure,
                            poisoned_order: order,
                        },
                    ));
                }
                (driver_result, None) => {
                    let failure = match driver_result {
                        Ok(_) => core_resource_failure(
                            "commit_acknowledged_without_allocation",
                            "driver acknowledged commit without a core-owned allocation",
                            false,
                        ),
                        Err(failure) => failure.into_failure(),
                    };
                    let failure_id = self.issue_failure_id();
                    let receipt = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Commit,
                        failure,
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ReverseCompensation,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&receipt);
                    return Err(ResourceCommitTransitionError::Recoverable(
                        ResourcePrepareTransitionError {
                            transaction: Some(self),
                            failure: receipt,
                        },
                    ));
                }
            }
        }
        let receipt = self
            .transition_receipt(ResourceTransactionAction::Commit, ledger_before, completed)
            .expect("core commit journal must validate");
        Ok(self.advance(Some(receipt)))
    }

    pub fn rollback(
        mut self,
    ) -> Result<ResourceTransaction<D, TransactionRolledBack>, ResourceRollbackTransitionError<D>>
    {
        debug_assert!(
            self.states
                .iter()
                .all(|state| *state == ResourceTransactionState::Reserved),
            "rollback starts from a uniformly reserved transaction"
        );
        let ledger_before = self.ledger_snapshot_entries();
        let mut completed = Vec::new();
        for order in 0..self.states.len() {
            let result = {
                let reservation = self.reservations.reservations[order].clone();
                let (driver, context) =
                    self.driver_and_context(order, ResourceTransactionAction::Rollback);
                driver.rollback_resource(&context, &reservation)
            };
            match result {
                Ok(()) => {
                    let before = self.states[order];
                    self.states[order] = ResourceTransactionState::RolledBack;
                    completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Rollback,
                        before,
                        ResourceTransactionState::RolledBack,
                    ));
                }
                Err(failure) => {
                    let failure_id = self.issue_failure_id();
                    let receipt = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Rollback,
                        failure.into_failure(),
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ForwardCompletion,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&receipt);
                    return Err(ResourceRollbackTransitionError {
                        transaction: Some(self),
                        failure: receipt,
                    });
                }
            }
        }
        let receipt = self
            .transition_receipt(
                ResourceTransactionAction::Rollback,
                ledger_before,
                completed,
            )
            .expect("core rollback journal must validate");
        Ok(self.advance(Some(receipt)))
    }
}

impl<D: ResourceTransactionDriver> ResourceTransaction<D, TransactionCommitted> {
    pub fn into_plan_runtime(
        mut self,
    ) -> Result<Arc<PlanRuntimeResources<D::Runtime>>, PlanRuntimeHandoffError<D>>
    where
        D: 'static,
    {
        let valid = self
            .states
            .iter()
            .all(|state| *state == ResourceTransactionState::Committed)
            && self.pending_failure.is_none()
            && self.pending_subset_release.is_none()
            && self
                .pending_allocations
                .iter()
                .all(|allocation| allocation.borrow().is_none())
            && self.lease.as_ref().is_some_and(|lease| {
                !lease.slots.is_empty()
                    && lease.slots.iter().all(|slot| {
                        slot.buffer.is_some()
                            && slot.descriptor.is_some()
                            && slot.entry.state == ResourceLeaseState::Active
                    })
            })
            && self.driver.is_some()
            && self.maintenance_controller.is_some()
            && self.dynamic_pools.is_some()
            && self.capacity_claim.is_some();
        if !valid {
            return Err(PlanRuntimeHandoffError {
                error: invalid_resource(
                    "plan runtime handoff requires one complete active committed transaction",
                ),
                transaction: Some(self),
            });
        }
        let lease = self
            .lease
            .take()
            .expect("validated plan runtime handoff owns its static lease");
        let runtime = Arc::clone(&lease.runtime);
        let dynamic_pools = self
            .dynamic_pools
            .take()
            .expect("validated plan runtime handoff owns its dynamic pools");
        let driver: Box<dyn ErasedPlanStaticDriver<D::Runtime>> = Box::new(
            self.driver
                .take()
                .expect("validated plan runtime handoff owns its driver"),
        );
        let static_resources = PlanStaticResources {
            driver: Mutex::new(Some(driver)),
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            reservations: self.reservations.clone(),
            states: std::mem::take(&mut self.states),
            capacity_claim: self.capacity_claim.take(),
            lease: Some(lease),
            finalized: false,
        };
        let maintenance_controller = self
            .maintenance_controller
            .take()
            .expect("validated plan runtime handoff owns maintenance authority");
        let (lifecycle_tx, _) = watch::channel(PLAN_RUNTIME_OPEN);
        self.finalized = true;
        Ok(Arc::new(PlanRuntimeResources {
            lifecycle: RwLock::new(()),
            phase: AtomicU8::new(PLAN_RUNTIME_OPEN),
            lifecycle_tx,
            maintenance_controller,
            dynamic_pools,
            static_resources: PlanRuntimeStatic::Static(static_resources),
            runtime,
            deferred_cleanup_domain: new_deferred_device_cleanup_domain(),
        }))
    }

    pub fn lease(&self) -> &StaticProvisioningLease<D::Runtime> {
        self.lease
            .as_ref()
            .expect("committed transaction owns its batch lease")
    }

    fn defer_lease(
        &mut self,
        resource_ids: &[ResourceId],
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        self.lease_transition(resource_ids, ResourceLeaseAction::Defer)
    }

    fn resume_lease(
        &mut self,
        resource_ids: &[ResourceId],
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        self.lease_transition(resource_ids, ResourceLeaseAction::Resume)
    }

    fn cancel_lease(
        &mut self,
        resource_ids: &[ResourceId],
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        self.lease_transition(resource_ids, ResourceLeaseAction::Cancel)
    }

    pub fn defer_all(&mut self) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        let ids = self
            .reservations
            .resource_ids()
            .cloned()
            .collect::<Vec<_>>();
        self.defer_lease(&ids)
    }

    pub fn resume_all(&mut self) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        let ids = self
            .reservations
            .resource_ids()
            .cloned()
            .collect::<Vec<_>>();
        self.resume_lease(&ids)
    }

    pub fn cancel_all(&mut self) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        let ids = self
            .reservations
            .resource_ids()
            .cloned()
            .collect::<Vec<_>>();
        self.cancel_lease(&ids)
    }

    /// Terminal release always targets the complete pool. On partial backend
    /// failure, the actual prefix remains released in the ledger and
    /// `complete_pending_release` resumes the outstanding suffix.
    fn release_all_resources(
        &mut self,
    ) -> Result<ResourceTransitionReceipt, ResourceFailureReceipt> {
        if let Some(pending) = &self.pending_subset_release {
            return Err(pending.failure.clone());
        }
        let orders = (0..self.states.len()).collect::<Vec<_>>();
        if let Some(&order) = orders
            .iter()
            .find(|&&order| self.states[order] != ResourceTransactionState::Committed)
        {
            return Err(self.local_release_failure(format!(
                "resource `{}` is not actually committed",
                self.reservations.reservations[order].resource_id()
            )));
        }
        let ledger_before = self.ledger_snapshot_entries();
        let mut completed = Vec::new();
        for &order in &orders {
            let result = {
                let reservation = self.reservations.reservations[order].clone();
                let buffer = self
                    .lease
                    .as_ref()
                    .expect("committed transaction owns lease")
                    .buffer(order)
                    .expect("committed ledger entry owns its buffer");
                let driver = self.driver.as_mut().expect("live transaction owns driver");
                let context = ResourceTransactionContext {
                    runtime: &self.lease.as_ref().expect("transaction owns lease").runtime,
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
                driver.release_resource(&context, &reservation, buffer)
            };
            match result {
                Ok(()) => {
                    let before = self.states[order];
                    self.lease
                        .as_mut()
                        .expect("transaction owns lease ledger")
                        .clear(order);
                    self.states[order] = ResourceTransactionState::Released;
                    self.capacity_claim
                        .as_mut()
                        .expect("live transaction owns its static capacity claim")
                        .release_bytes(self.reservations.reservations[order].size_bytes());
                    completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Release,
                        before,
                        ResourceTransactionState::Released,
                    ));
                }
                Err(failure) => {
                    let failure_id = self.issue_failure_id();
                    let receipt = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Release,
                        failure.into_failure(),
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ForwardCompletion,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&receipt);
                    self.pending_subset_release = Some(PendingSubsetRelease {
                        target_orders: orders,
                        failure: receipt.clone(),
                    });
                    return Err(receipt);
                }
            }
        }
        let receipt = self
            .transition_receipt(ResourceTransactionAction::Release, ledger_before, completed)
            .expect("core subset release journal must validate");
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    pub fn complete_pending_release(
        &mut self,
    ) -> Result<ResourceTransitionReceipt, ResourceFailureReceipt> {
        let Some(mut pending) = self.pending_subset_release.take() else {
            return Err(self.local_release_failure(
                "there is no pending subset release to complete".to_owned(),
            ));
        };
        for &order in &pending.target_orders {
            match self.states[order] {
                ResourceTransactionState::Released => continue,
                ResourceTransactionState::Committed => {}
                actual => {
                    let failure = ResourceDriverFailure::new(core_resource_failure(
                        "release_ledger_diverged",
                        format!(
                            "resource `{}` has unexpected actual state {}",
                            self.reservations.reservations[order].resource_id(),
                            actual.as_str()
                        ),
                        false,
                    ))
                    .expect("core failure has resource domain");
                    pending.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            actual,
                        )),
                    );
                    pending.failure.ledger_after = self.ledger_snapshot_entries();
                    self.set_pending_failure(&pending.failure);
                    self.pending_subset_release = Some(pending);
                    return Err(self
                        .pending_subset_release
                        .as_ref()
                        .expect("pending release was restored")
                        .failure
                        .clone());
                }
            }
            let result = {
                let reservation = self.reservations.reservations[order].clone();
                let buffer = self
                    .lease
                    .as_ref()
                    .expect("committed transaction owns lease")
                    .buffer(order)
                    .expect("outstanding release owns its buffer");
                let driver = self.driver.as_mut().expect("live transaction owns driver");
                let context = ResourceTransactionContext {
                    runtime: &self.lease.as_ref().expect("transaction owns lease").runtime,
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
                driver.release_resource(&context, &reservation, buffer)
            };
            match result {
                Ok(()) => {
                    let before = self.states[order];
                    self.lease
                        .as_mut()
                        .expect("transaction owns lease ledger")
                        .clear(order);
                    self.states[order] = ResourceTransactionState::Released;
                    self.capacity_claim
                        .as_mut()
                        .expect("live transaction owns its static capacity claim")
                        .release_bytes(self.reservations.reservations[order].size_bytes());
                    pending.failure.completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Release,
                        before,
                        ResourceTransactionState::Released,
                    ));
                }
                Err(failure) => {
                    pending.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                    );
                    pending.failure.ledger_after = self.ledger_snapshot_entries();
                    self.set_pending_failure(&pending.failure);
                    let snapshot = pending.failure.clone();
                    self.pending_subset_release = Some(pending);
                    return Err(snapshot);
                }
            }
        }
        pending.failure.recovery_complete = true;
        pending.failure.ledger_after = self.ledger_snapshot_entries();
        let receipt = self
            .transition_receipt(
                ResourceTransactionAction::Release,
                pending.failure.ledger_before.clone(),
                pending.failure.completed.clone(),
            )
            .expect("forward-completed release journal must validate");
        self.clear_pending_failure();
        self.recovery_history.push(pending.failure);
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    fn local_release_failure(&mut self, message: String) -> ResourceFailureReceipt {
        let ledger = self.ledger_snapshot_entries();
        let failure_id = self.issue_failure_id();
        ResourceFailureReceipt::new(
            failure_id,
            &self.identity,
            &self.admission,
            ResourceTransactionAction::Release,
            core_resource_failure("invalid_release_request", message, false),
            None,
            Vec::new(),
            ResourceRecoveryStrategy::ForwardCompletion,
            ledger.clone(),
            ledger,
        )
    }

    pub fn release(
        mut self,
    ) -> Result<ResourceTransaction<D, TransactionReleased>, ResourceReleaseTransitionError<D>>
    {
        if self.pending_subset_release.is_some() {
            let failure = self
                .pending_subset_release
                .as_ref()
                .expect("pending release exists")
                .failure
                .clone();
            return Err(ResourceReleaseTransitionError {
                transaction: Some(self),
                failure,
            });
        }
        if self
            .states
            .iter()
            .all(|state| *state == ResourceTransactionState::Released)
        {
            return Ok(self.advance(None));
        }
        if !self
            .states
            .iter()
            .all(|state| *state == ResourceTransactionState::Committed)
        {
            let failure = self.local_release_failure(
                "transaction contains non-releasable actual states".to_owned(),
            );
            return Err(ResourceReleaseTransitionError {
                transaction: Some(self),
                failure,
            });
        }
        match self.release_all_resources() {
            Ok(_) => Ok(self.advance(None)),
            Err(failure) => Err(ResourceReleaseTransitionError {
                transaction: Some(self),
                failure,
            }),
        }
    }
}

#[must_use = "failed plan runtime handoff retains the committed transaction"]
pub struct PlanRuntimeHandoffError<D>
where
    D: ResourceTransactionDriver,
{
    error: VNextError,
    transaction: Option<ResourceTransaction<D, TransactionCommitted>>,
}

impl<D> PlanRuntimeHandoffError<D>
where
    D: ResourceTransactionDriver,
{
    pub fn error(&self) -> &VNextError {
        &self.error
    }

    pub fn into_transaction(mut self) -> ResourceTransaction<D, TransactionCommitted> {
        self.transaction
            .take()
            .expect("plan runtime handoff error owns its transaction")
    }
}

#[must_use = "reverse recovery must complete or ownership must be quarantined"]
pub struct ResourcePrepareTransitionError<D: ResourceTransactionDriver, S: TransactionStage> {
    transaction: Option<ResourceTransaction<D, S>>,
    failure: ResourceFailureReceipt,
}

impl<D: ResourceTransactionDriver, S: TransactionStage> fmt::Debug
    for ResourcePrepareTransitionError<D, S>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourcePrepareTransitionError")
            .field("stage", &S::STATE)
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver, S: TransactionStage> ResourcePrepareTransitionError<D, S> {
    pub fn failure(&self) -> &ResourceFailureReceipt {
        &self.failure
    }

    pub fn recover(mut self) -> Result<ResourceTransaction<D, S>, Self> {
        while self.failure.compensation.len() < self.failure.completed.len() {
            let record_index = self.failure.completed.len() - 1 - self.failure.compensation.len();
            let attempted = self.failure.completed[record_index].clone();
            let order = attempted.order as usize;
            let transaction = self
                .transaction
                .as_mut()
                .expect("prepare recovery owns transaction");
            if transaction.states[order] != attempted.after {
                let failure = ResourceDriverFailure::new(core_resource_failure(
                    "compensation_ledger_diverged",
                    format!(
                        "resource `{}` expected actual state {}, found {}",
                        attempted.resource_id,
                        attempted.after.as_str(),
                        transaction.states[order].as_str()
                    ),
                    false,
                ))
                .expect("core failure has resource domain");
                self.failure.record_recovery_failure(
                    failure,
                    Some(ResourceFailurePoint::new(
                        &transaction.reservations.reservations[order],
                        order,
                        transaction.states[order],
                    )),
                );
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                return Err(self);
            }
            let result = match self.failure.action {
                ResourceTransactionAction::Reserve => {
                    let reservation = transaction.reservations.reservations[order].clone();
                    let driver = transaction
                        .driver
                        .as_mut()
                        .expect("live transaction owns driver");
                    let context = ResourceTransactionContext {
                        runtime: &transaction
                            .lease
                            .as_ref()
                            .expect("transaction owns lease")
                            .runtime,
                        identity: &transaction.identity,
                        binding: &transaction.admission,
                        reservations: &transaction.reservations,
                        cursor: Some(ResourceActionCursor {
                            order,
                            action: ResourceTransactionAction::Reserve,
                            before: transaction.states[order],
                            allocation_authorized: false,
                        }),
                        allocation_authority: None,
                        pending_allocation: None,
                    };
                    driver.compensate_reserve_resource(&context, &reservation)
                }
                ResourceTransactionAction::Commit => {
                    let reservation = transaction.reservations.reservations[order].clone();
                    let buffer = transaction
                        .lease
                        .as_ref()
                        .expect("transaction owns lease ledger")
                        .buffer(order)
                        .expect("uncompensated commit owns its buffer");
                    let driver = transaction
                        .driver
                        .as_mut()
                        .expect("live transaction owns driver");
                    let context = ResourceTransactionContext {
                        runtime: &transaction
                            .lease
                            .as_ref()
                            .expect("transaction owns lease")
                            .runtime,
                        identity: &transaction.identity,
                        binding: &transaction.admission,
                        reservations: &transaction.reservations,
                        cursor: Some(ResourceActionCursor {
                            order,
                            action: ResourceTransactionAction::Commit,
                            before: transaction.states[order],
                            allocation_authorized: false,
                        }),
                        allocation_authority: None,
                        pending_allocation: None,
                    };
                    driver.compensate_commit_resource(&context, &reservation, buffer)
                }
                _ => unreachable!("prepare recovery handles only reserve and commit"),
            };
            match result {
                Ok(()) => {
                    transaction.states[order] = attempted.before;
                    if self.failure.action == ResourceTransactionAction::Commit {
                        transaction
                            .lease
                            .as_mut()
                            .expect("transaction owns lease ledger")
                            .clear(order);
                    }
                    self.failure
                        .compensation
                        .push(ResourceCompensationRecord::from_transition(
                            &attempted,
                            self.failure.compensation.len(),
                        ));
                    self.failure.ledger_after = transaction.ledger_snapshot_entries();
                }
                Err(failure) => {
                    self.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &transaction.reservations.reservations[order],
                            order,
                            transaction.states[order],
                        )),
                    );
                    self.failure.ledger_after = transaction.ledger_snapshot_entries();
                    transaction.set_pending_failure(&self.failure);
                    return Err(self);
                }
            }
        }
        self.failure.recovery_complete = true;
        let mut transaction = self
            .transaction
            .take()
            .expect("prepare recovery owns transaction");
        self.failure.ledger_after = transaction.ledger_snapshot_entries();
        transaction.clear_pending_failure();
        if self.failure.action == ResourceTransactionAction::Commit {
            for authority in &transaction.allocation_issued {
                authority.store(false, Ordering::Release);
            }
        }
        transaction.recovery_history.push(self.failure);
        Ok(transaction)
    }

    pub fn quarantine(mut self) -> Result<ResourceTransaction<D, TransactionQuarantined>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("prepare recovery owns transaction");
        let ledger_before = transaction.ledger_snapshot_entries();
        match transaction.quarantine_live() {
            Ok(records) => {
                self.failure.recovery_complete = true;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.clear_pending_failure();
                transaction.recovery_history.push(self.failure.clone());
                let receipt = transaction
                    .transition_receipt(
                        ResourceTransactionAction::Quarantine,
                        ledger_before,
                        records,
                    )
                    .expect("core quarantine journal must validate");
                let transaction = self
                    .transaction
                    .take()
                    .expect("prepare recovery owns transaction");
                Ok(transaction.advance(Some(receipt)))
            }
            Err(failure) => {
                self.failure.record_recovery_failure(failure, None);
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }
}

#[must_use = "poisoned commit ownership must be reconciled or quarantined"]
pub struct ResourcePoisonedTransaction<D: ResourceTransactionDriver> {
    transaction: Option<ResourceTransaction<D, TransactionReserved>>,
    failure: ResourceFailureReceipt,
    poisoned_order: usize,
}

impl<D: ResourceTransactionDriver> fmt::Debug for ResourcePoisonedTransaction<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourcePoisonedTransaction")
            .field("failure", &self.failure)
            .field("poisoned_order", &self.poisoned_order)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver> ResourcePoisonedTransaction<D> {
    pub fn failure(&self) -> &ResourceFailureReceipt {
        &self.failure
    }

    pub fn reconcile(
        mut self,
    ) -> Result<ResourcePrepareTransitionError<D, TransactionReserved>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("poisoned owner owns transaction");
        let reservation = transaction.reservations.reservations[self.poisoned_order].clone();
        let result = {
            let lease = transaction
                .lease
                .as_ref()
                .expect("poisoned transaction owns lease ledger");
            let slot = &lease.slots[self.poisoned_order];
            let actual = ResourceCommitView {
                resource_id: slot
                    .actual_resource_id
                    .as_ref()
                    .expect("poisoned allocation records actual resource identity"),
                generation: slot
                    .actual_generation
                    .expect("poisoned allocation records actual generation"),
                descriptor: slot
                    .descriptor
                    .as_ref()
                    .expect("poisoned allocation records actual descriptor"),
                buffer: slot
                    .buffer
                    .as_ref()
                    .expect("poisoned allocation remains core-owned"),
            };
            let driver = transaction
                .driver
                .as_mut()
                .expect("live transaction owns driver");
            let context = ResourceTransactionContext {
                runtime: &lease.runtime,
                identity: &transaction.identity,
                binding: &transaction.admission,
                reservations: &transaction.reservations,
                cursor: Some(ResourceActionCursor {
                    order: self.poisoned_order,
                    action: ResourceTransactionAction::Commit,
                    before: transaction.states[self.poisoned_order],
                    allocation_authorized: false,
                }),
                allocation_authority: None,
                pending_allocation: None,
            };
            driver.reconcile_commit_outcome(&context, &reservation, actual)
        };
        match result {
            Ok(()) => {
                transaction
                    .lease
                    .as_mut()
                    .expect("poisoned transaction owns lease ledger")
                    .clear(self.poisoned_order);
                self.failure.recovery_strategy = ResourceRecoveryStrategy::ReverseCompensation;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                let transaction = self
                    .transaction
                    .take()
                    .expect("poisoned owner owns transaction");
                Ok(ResourcePrepareTransitionError {
                    transaction: Some(transaction),
                    failure: self.failure,
                })
            }
            Err(failure) => {
                self.failure.record_recovery_failure(
                    failure,
                    Some(ResourceFailurePoint::new(
                        &reservation,
                        self.poisoned_order,
                        transaction.states[self.poisoned_order],
                    )),
                );
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }

    pub fn quarantine(mut self) -> Result<ResourceTransaction<D, TransactionQuarantined>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("poisoned owner owns transaction");
        let ledger_before = transaction.ledger_snapshot_entries();
        match transaction.quarantine_live() {
            Ok(records) => {
                self.failure.recovery_complete = true;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.clear_pending_failure();
                transaction.recovery_history.push(self.failure.clone());
                let receipt = transaction
                    .transition_receipt(
                        ResourceTransactionAction::Quarantine,
                        ledger_before,
                        records,
                    )
                    .expect("poison quarantine journal must validate");
                let transaction = self
                    .transaction
                    .take()
                    .expect("poisoned owner owns transaction");
                Ok(transaction.advance(Some(receipt)))
            }
            Err(failure) => {
                self.failure.record_recovery_failure(failure, None);
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }
}

#[must_use = "rollback forward completion owns the transaction"]
pub struct ResourceRollbackTransitionError<D: ResourceTransactionDriver> {
    transaction: Option<ResourceTransaction<D, TransactionReserved>>,
    failure: ResourceFailureReceipt,
}

impl<D: ResourceTransactionDriver> fmt::Debug for ResourceRollbackTransitionError<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourceRollbackTransitionError")
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver> ResourceRollbackTransitionError<D> {
    pub fn failure(&self) -> &ResourceFailureReceipt {
        &self.failure
    }

    pub fn complete(mut self) -> Result<ResourceTransaction<D, TransactionRolledBack>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("rollback recovery owns transaction");
        for order in 0..transaction.states.len() {
            match transaction.states[order] {
                ResourceTransactionState::RolledBack => continue,
                ResourceTransactionState::Reserved => {}
                actual => {
                    let failure = ResourceDriverFailure::new(core_resource_failure(
                        "rollback_ledger_diverged",
                        format!("rollback found actual state {}", actual.as_str()),
                        false,
                    ))
                    .expect("core failure has resource domain");
                    self.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &transaction.reservations.reservations[order],
                            order,
                            actual,
                        )),
                    );
                    self.failure.ledger_after = transaction.ledger_snapshot_entries();
                    transaction.set_pending_failure(&self.failure);
                    return Err(self);
                }
            }
            let result = {
                let reservation = transaction.reservations.reservations[order].clone();
                let driver = transaction
                    .driver
                    .as_mut()
                    .expect("live transaction owns driver");
                let context = ResourceTransactionContext {
                    runtime: &transaction
                        .lease
                        .as_ref()
                        .expect("transaction owns lease")
                        .runtime,
                    identity: &transaction.identity,
                    binding: &transaction.admission,
                    reservations: &transaction.reservations,
                    cursor: Some(ResourceActionCursor {
                        order,
                        action: ResourceTransactionAction::Rollback,
                        before: transaction.states[order],
                        allocation_authorized: false,
                    }),
                    allocation_authority: None,
                    pending_allocation: None,
                };
                driver.rollback_resource(&context, &reservation)
            };
            match result {
                Ok(()) => {
                    let before = transaction.states[order];
                    transaction.states[order] = ResourceTransactionState::RolledBack;
                    self.failure.completed.push(transaction.make_record(
                        order,
                        ResourceTransactionAction::Rollback,
                        before,
                        ResourceTransactionState::RolledBack,
                    ));
                }
                Err(failure) => {
                    self.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &transaction.reservations.reservations[order],
                            order,
                            transaction.states[order],
                        )),
                    );
                    self.failure.ledger_after = transaction.ledger_snapshot_entries();
                    transaction.set_pending_failure(&self.failure);
                    return Err(self);
                }
            }
        }
        self.failure.recovery_complete = true;
        self.failure.ledger_after = transaction.ledger_snapshot_entries();
        let receipt = transaction
            .transition_receipt(
                ResourceTransactionAction::Rollback,
                self.failure.ledger_before.clone(),
                self.failure.completed.clone(),
            )
            .expect("forward-completed rollback journal must validate");
        transaction.clear_pending_failure();
        transaction.recovery_history.push(self.failure);
        let transaction = self
            .transaction
            .take()
            .expect("rollback recovery owns transaction");
        Ok(transaction.advance(Some(receipt)))
    }

    pub fn quarantine(mut self) -> Result<ResourceTransaction<D, TransactionQuarantined>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("rollback recovery owns transaction");
        let ledger_before = transaction.ledger_snapshot_entries();
        match transaction.quarantine_live() {
            Ok(records) => {
                self.failure.recovery_complete = true;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.clear_pending_failure();
                transaction.recovery_history.push(self.failure.clone());
                let receipt = transaction
                    .transition_receipt(
                        ResourceTransactionAction::Quarantine,
                        ledger_before,
                        records,
                    )
                    .expect("rollback quarantine journal must validate");
                let transaction = self
                    .transaction
                    .take()
                    .expect("rollback recovery owns transaction");
                Ok(transaction.advance(Some(receipt)))
            }
            Err(failure) => {
                self.failure.record_recovery_failure(failure, None);
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }
}

#[must_use = "release forward completion owns the transaction"]
pub struct ResourceReleaseTransitionError<D: ResourceTransactionDriver> {
    transaction: Option<ResourceTransaction<D, TransactionCommitted>>,
    failure: ResourceFailureReceipt,
}

impl<D: ResourceTransactionDriver> fmt::Debug for ResourceReleaseTransitionError<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourceReleaseTransitionError")
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver> ResourceReleaseTransitionError<D> {
    pub fn failure(&self) -> &ResourceFailureReceipt {
        &self.failure
    }

    pub fn complete(mut self) -> Result<ResourceTransaction<D, TransactionReleased>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("release recovery owns transaction");
        match transaction.complete_pending_release() {
            Ok(_) => {
                self.failure = transaction
                    .recovery_history
                    .last()
                    .cloned()
                    .unwrap_or_else(|| self.failure.clone());
                let transaction = self
                    .transaction
                    .take()
                    .expect("release recovery owns transaction");
                transaction.release().map_err(|next| Self {
                    transaction: next.transaction,
                    failure: next.failure,
                })
            }
            Err(failure) => {
                self.failure = failure;
                Err(self)
            }
        }
    }

    pub fn quarantine(mut self) -> Result<ResourceTransaction<D, TransactionQuarantined>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("release recovery owns transaction");
        let ledger_before = transaction.ledger_snapshot_entries();
        match transaction.quarantine_live() {
            Ok(records) => {
                self.failure.recovery_complete = true;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.clear_pending_failure();
                transaction.pending_subset_release = None;
                transaction.recovery_history.push(self.failure.clone());
                let receipt = transaction
                    .transition_receipt(
                        ResourceTransactionAction::Quarantine,
                        ledger_before,
                        records,
                    )
                    .expect("release quarantine journal must validate");
                let transaction = self
                    .transaction
                    .take()
                    .expect("release recovery owns transaction");
                Ok(transaction.advance(Some(receipt)))
            }
            Err(failure) => {
                self.failure.record_recovery_failure(failure, None);
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }
}

fn same_allocation_entry(left: &ResourceLeaseEntry, right: &ResourceLeaseEntry) -> bool {
    left.owner_node_id == right.owner_node_id
        && left.resource_id == right.resource_id
        && left.size_bytes == right.size_bytes
        && left.alignment_bytes == right.alignment_bytes
        && left.usage == right.usage
        && left.element_type == right.element_type
        && left.retention_policy == right.retention_policy
        && left.generation == right.generation
}

fn validate_context_envelope(
    identity: &ResourceTransactionIdentity,
    admission: &StaticProvisioningBinding,
    before: &[ResourceLedgerEntrySnapshot],
    after: &[ResourceLedgerEntrySnapshot],
) -> Result<(), VNextError> {
    if identity.pool_id != admission.pool_id()
        || identity.request_id != admission.request_id
        || admission.pool_identity.plan_id != admission.plan_id
        || admission.pool_identity.plan_hash != admission.plan_hash
        || admission.pool_identity.device_id != admission.device_id
        || admission
            .pool_identity
            .device_runtime_implementation_fingerprint
            != admission.device_runtime_implementation_fingerprint
        || admission.pool_identity.admission_generation != admission.admission_generation
        || admission.admission_generation == 0
        || admission.maximum_active_sequences == 0
        || admission.device_capacity_bytes == 0
        || admission.usable_capacity_bytes == 0
        || admission.usable_capacity_bytes > admission.device_capacity_bytes
        || admission.plan_static_bytes > admission.usable_capacity_bytes
        || admission.admitted_bytes == 0
        || admission.admitted_bytes != admission.plan_static_bytes
        || admission.admitted_bytes > admission.usable_capacity_bytes
        || before.is_empty()
        || before.len() != after.len()
    {
        return Err(invalid_resource(
            "trusted resource validation context has an invalid envelope",
        ));
    }
    let mut resources = BTreeSet::new();
    let mut total = 0_u64;
    for (before_entry, after_entry) in before.iter().zip(after) {
        if !same_allocation_entry(&before_entry.entry, &after_entry.entry)
            || !resources.insert(before_entry.entry.resource_id.clone())
            || before_entry.entry.generation != admission.admission_generation
        {
            return Err(invalid_resource(
                "trusted resource validation context changes allocation identity",
            ));
        }
        total = total
            .checked_add(before_entry.entry.size_bytes)
            .ok_or_else(|| invalid_resource("validation context bytes overflow u64"))?;
    }
    if total != admission.admitted_bytes {
        return Err(invalid_resource(
            "validation context does not cover the complete admitted allocation set",
        ));
    }
    Ok(())
}

pub(super) fn validate_transition_records_against_context(
    records: &[ResourceTransitionRecord],
    context: &ResourceTransitionValidationContext,
) -> Result<(), VNextError> {
    validate_context_envelope(
        &context.identity,
        &context.admission,
        &context.before,
        &context.after,
    )?;
    let changed = context
        .before
        .iter()
        .zip(&context.after)
        .enumerate()
        .filter(|(_, (before, after))| {
            before.transaction_state != after.transaction_state
                || before.buffer_present != after.buffer_present
        })
        .collect::<Vec<_>>();
    if changed.is_empty() || changed.len() != records.len() {
        return Err(invalid_resource(
            "resource receipt does not cover the exact ledger delta",
        ));
    }
    for (record, (order, (before, after))) in records.iter().zip(changed) {
        record.validate()?;
        if !record.matches_identity_and_admission(&context.identity, &context.admission)
            || record.action != context.action
            || record.order as usize != order
            || !record.matches_snapshot(before)
            || !record.matches_snapshot(after)
            || record.before != before.transaction_state
            || record.after != after.transaction_state
            || expected_transition(context.action, record.before) != Some(record.after)
        {
            return Err(invalid_resource(
                "resource receipt differs from trusted allocation or ledger state",
            ));
        }
        match context.action {
            ResourceTransactionAction::Commit => {
                if before.buffer_present || !after.buffer_present {
                    return Err(invalid_resource(
                        "commit receipt does not prove buffer acquisition",
                    ));
                }
            }
            ResourceTransactionAction::Release | ResourceTransactionAction::Quarantine => {
                if before.transaction_state == ResourceTransactionState::Committed
                    && (!before.buffer_present || after.buffer_present)
                {
                    return Err(invalid_resource(
                        "cleanup receipt does not prove committed buffer return",
                    ));
                }
            }
            ResourceTransactionAction::Reserve | ResourceTransactionAction::Rollback => {
                if before.buffer_present != after.buffer_present {
                    return Err(invalid_resource(
                        "non-buffer transition changed buffer ownership",
                    ));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn validate_lease_entries_against_context(
    entries: &[ResourceLeaseEntry],
    before_state: ResourceLeaseState,
    after_state: ResourceLeaseState,
    context: &ResourceLeaseValidationContext,
) -> Result<(), VNextError> {
    validate_context_envelope(
        &context.identity,
        &context.admission,
        &context.before,
        &context.after,
    )?;
    if expected_lease_transition(context.action, before_state) != Some(after_state) {
        return Err(VNextError::InvalidLeaseTransition {
            lease_id: context.identity.transaction_id.to_string(),
            from: before_state.as_str(),
            action: context.action.as_str(),
        });
    }
    let changed = context
        .before
        .iter()
        .zip(&context.after)
        .filter(|(before, after)| before.entry.state != after.entry.state)
        .collect::<Vec<_>>();
    if changed.is_empty() || changed.len() != entries.len() {
        return Err(invalid_resource(
            "lease receipt does not cover the exact lease-state delta",
        ));
    }
    for (entry, (before, after)) in entries.iter().zip(changed) {
        if !same_allocation_entry(entry, &before.entry)
            || !same_allocation_entry(entry, &after.entry)
            || before.entry.state != before_state
            || after.entry.state != after_state
            || entry.state != after_state
            || before.transaction_state != after.transaction_state
            || before.transaction_state != ResourceTransactionState::Committed
            || before.buffer_present != after.buffer_present
            || !before.buffer_present
        {
            return Err(invalid_resource(
                "lease receipt differs from trusted allocation or ledger state",
            ));
        }
    }
    Ok(())
}
