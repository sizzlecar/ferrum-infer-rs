use super::{
    invalid_resource, watch, Arc, AtomicU64, BTreeMap, BatchInvocationId, BatchStepId,
    CapacityAvailabilityEpoch, DeviceId, Mutex, OnceLock, Ordering, VNextError, Weak,
};
use crate::vnext::{
    CapacityAvailabilitySource, DeviceCapacityPressure, DeviceCapacityPressureScope,
    DynamicAdmissionFaultKind,
};

static NEXT_ADMISSION_GENERATION: AtomicU64 = AtomicU64::new(1);
static NEXT_BATCH_STEP_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_BATCH_INVOCATION_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn issue_batch_step_id() -> Result<BatchStepId, VNextError> {
    let value = NEXT_BATCH_STEP_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| invalid_resource("batch step id space is exhausted"))?;
    BatchStepId::try_from(value)
}

pub(super) fn issue_batch_invocation_id() -> Result<BatchInvocationId, VNextError> {
    let value = NEXT_BATCH_INVOCATION_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| invalid_resource("batch invocation id space is exhausted"))?;
    BatchInvocationId::try_from(value)
}

#[derive(Debug)]
pub(super) struct DeviceCapacityAccount {
    pub(super) device_id: DeviceId,
    pub(super) device_runtime_implementation_fingerprint: String,
    pub(super) device_capacity_bytes: u64,
    pub(super) state: Mutex<DeviceCapacityState>,
    process_availability_tx: watch::Sender<DeviceCapacitySignal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeviceCapacitySignal {
    Live(u64),
    FailClosed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeviceCapacityAvailabilitySnapshot {
    plan_epoch: u64,
    process_epoch: u64,
}

impl DeviceCapacityAvailabilitySnapshot {
    pub(super) const fn plan_epoch(self) -> u64 {
        self.plan_epoch
    }

    pub(super) const fn process_epoch(self) -> u64 {
        self.process_epoch
    }

    pub(super) fn epoch_for_pressure(
        self,
        pressure: &DeviceCapacityPressure,
    ) -> CapacityAvailabilityEpoch {
        let (source, epoch) = match pressure.scope() {
            DeviceCapacityPressureScope::PlanBudget => (
                CapacityAvailabilitySource::PlanDeviceBudget,
                self.plan_epoch,
            ),
            DeviceCapacityPressureScope::ProcessWide => (
                CapacityAvailabilitySource::ProcessDeviceCapacity,
                self.process_epoch,
            ),
        };
        CapacityAvailabilityEpoch::new(source, epoch)
            .expect("live device capacity generations are non-zero")
    }
}

#[derive(Debug)]
pub(super) struct DeviceCapacityBudgetRecord {
    /// A conservative device-wide ceiling contributed by this live plan. It is
    /// not an additive plan share.
    pub(super) device_wide_usable_ceiling_bytes: u64,
    pub(super) claimed_bytes: u64,
    availability_epoch: u64,
    availability_tx: watch::Sender<DeviceCapacitySignal>,
}

#[derive(Debug)]
pub(super) struct DeviceCapacityState {
    pub(super) claimed_bytes: u64,
    pub(super) next_budget_id: u64,
    pub(super) budgets: BTreeMap<u64, DeviceCapacityBudgetRecord>,
    process_availability_epoch: u64,
    poisoned: bool,
}

fn capacity_fault(kind: DynamicAdmissionFaultKind, reason: impl Into<String>) -> VNextError {
    VNextError::DynamicAdmissionContract {
        kind,
        reason: reason.into(),
    }
}

fn poison_capacity_account(account: &DeviceCapacityAccount, state: &mut DeviceCapacityState) {
    state.poisoned = true;
    for record in state.budgets.values() {
        record
            .availability_tx
            .send_replace(DeviceCapacitySignal::FailClosed);
    }
    account
        .process_availability_tx
        .send_replace(DeviceCapacitySignal::FailClosed);
}

fn publish_capacity_release(
    account: &DeviceCapacityAccount,
    state: &mut DeviceCapacityState,
    budget_id: u64,
) {
    let next_plan_epoch = state
        .budgets
        .get(&budget_id)
        .and_then(|record| record.availability_epoch.checked_add(1));
    let next_process_epoch = state.process_availability_epoch.checked_add(1);
    let (Some(next_plan_epoch), Some(next_process_epoch)) = (next_plan_epoch, next_process_epoch)
    else {
        poison_capacity_account(account, state);
        return;
    };
    let record = state
        .budgets
        .get_mut(&budget_id)
        .expect("live device claim retains its budget record");
    record.availability_epoch = next_plan_epoch;
    state.process_availability_epoch = next_process_epoch;
    record
        .availability_tx
        .send_replace(DeviceCapacitySignal::Live(next_plan_epoch));
    account
        .process_availability_tx
        .send_replace(DeviceCapacitySignal::Live(next_process_epoch));
}

fn publish_process_capacity_increase(
    account: &DeviceCapacityAccount,
    state: &mut DeviceCapacityState,
) {
    let Some(next_process_epoch) = state.process_availability_epoch.checked_add(1) else {
        poison_capacity_account(account, state);
        return;
    };
    state.process_availability_epoch = next_process_epoch;
    account
        .process_availability_tx
        .send_replace(DeviceCapacitySignal::Live(next_process_epoch));
}

impl DeviceCapacityAccount {
    pub(super) fn register_budget(
        self: &Arc<Self>,
        device_wide_usable_ceiling_bytes: u64,
    ) -> Result<Arc<DeviceCapacityBudget>, VNextError> {
        if device_wide_usable_ceiling_bytes == 0
            || device_wide_usable_ceiling_bytes > self.device_capacity_bytes
        {
            return Err(invalid_resource(
                "device capacity budget is zero or exceeds raw device capacity",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("device capacity account is poisoned"))?;
        if state.poisoned {
            return Err(capacity_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "device capacity account is fail-closed",
            ));
        }
        let effective = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .map_or(device_wide_usable_ceiling_bytes, |existing| {
                existing.min(device_wide_usable_ceiling_bytes)
            });
        if state.claimed_bytes > effective {
            return Err(invalid_resource(format!(
                "device `{}` cannot register usable capacity {} below live physical claims {}",
                self.device_id, device_wide_usable_ceiling_bytes, state.claimed_bytes
            )));
        }
        let budget_id = state
            .next_budget_id
            .checked_add(1)
            .ok_or_else(|| invalid_resource("device capacity budget id space is exhausted"))?;
        state.next_budget_id = budget_id;
        let (availability_tx, _) = watch::channel(DeviceCapacitySignal::Live(1));
        state.budgets.insert(
            budget_id,
            DeviceCapacityBudgetRecord {
                device_wide_usable_ceiling_bytes,
                claimed_bytes: 0,
                availability_epoch: 1,
                availability_tx: availability_tx.clone(),
            },
        );
        Ok(Arc::new(DeviceCapacityBudget {
            account: Arc::clone(self),
            budget_id,
            device_wide_usable_ceiling_bytes,
            availability_tx,
        }))
    }

    pub(super) fn claim(
        self: &Arc<Self>,
        budget: &Arc<DeviceCapacityBudget>,
        bytes: u64,
    ) -> Result<DeviceCapacityClaim, VNextError> {
        if bytes == 0 || !Arc::ptr_eq(self, &budget.account) {
            return Err(invalid_resource(
                "device capacity claim is empty or belongs to another account",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("device capacity account is poisoned"))?;
        if state.poisoned {
            return Err(capacity_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "device capacity account is fail-closed",
            ));
        }
        let effective_usable_capacity = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .ok_or_else(|| invalid_resource("device capacity account has no live budget"))?;
        let budget_record = state
            .budgets
            .get(&budget.budget_id)
            .ok_or_else(|| invalid_resource("device capacity budget is stale"))?;
        let plan_claimed_bytes = budget_record.claimed_bytes;
        let process_claimed_bytes = state.claimed_bytes;
        let next_budget_claimed = plan_claimed_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid_resource("device plan capacity claim overflows u64"))?;
        let next = process_claimed_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid_resource("device process capacity claim overflows u64"))?;
        if next_budget_claimed > budget.device_wide_usable_ceiling_bytes
            || next > effective_usable_capacity
        {
            return Err(VNextError::DeviceCapacityUnavailable(
                DeviceCapacityPressure::new(
                    if next_budget_claimed > budget.device_wide_usable_ceiling_bytes {
                        DeviceCapacityPressureScope::PlanBudget
                    } else {
                        DeviceCapacityPressureScope::ProcessWide
                    },
                    self.device_id.to_string(),
                    bytes,
                    plan_claimed_bytes,
                    budget.device_wide_usable_ceiling_bytes,
                    process_claimed_bytes,
                    effective_usable_capacity,
                )?,
            ));
        }
        state.claimed_bytes = next;
        state
            .budgets
            .get_mut(&budget.budget_id)
            .expect("validated device budget remains registered")
            .claimed_bytes = next_budget_claimed;
        Ok(DeviceCapacityClaim {
            budget: Some(Arc::clone(budget)),
            bytes,
        })
    }
}

pub(super) struct DeviceCapacityBudget {
    pub(super) account: Arc<DeviceCapacityAccount>,
    pub(super) budget_id: u64,
    pub(super) device_wide_usable_ceiling_bytes: u64,
    availability_tx: watch::Sender<DeviceCapacitySignal>,
}

impl DeviceCapacityBudget {
    pub(super) fn availability_snapshot(
        &self,
    ) -> Result<DeviceCapacityAvailabilitySnapshot, VNextError> {
        let state = self
            .account
            .state
            .lock()
            .map_err(|_| invalid_resource("device capacity account is poisoned"))?;
        if state.poisoned {
            return Err(capacity_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "device capacity account is fail-closed",
            ));
        }
        let record = state
            .budgets
            .get(&self.budget_id)
            .ok_or_else(|| invalid_resource("device capacity budget is stale"))?;
        Ok(DeviceCapacityAvailabilitySnapshot {
            plan_epoch: record.availability_epoch,
            process_epoch: state.process_availability_epoch,
        })
    }

    pub(super) fn write_availability_epochs(
        &self,
        out: &mut Vec<CapacityAvailabilityEpoch>,
    ) -> Result<DeviceCapacityAvailabilitySnapshot, VNextError> {
        let snapshot = self.availability_snapshot()?;
        out.push(CapacityAvailabilityEpoch::new(
            CapacityAvailabilitySource::PlanDeviceBudget,
            snapshot.plan_epoch,
        )?);
        out.push(CapacityAvailabilityEpoch::new(
            CapacityAvailabilitySource::ProcessDeviceCapacity,
            snapshot.process_epoch,
        )?);
        Ok(snapshot)
    }

    pub(super) fn subscribe_plan_availability(&self) -> watch::Receiver<DeviceCapacitySignal> {
        self.availability_tx.subscribe()
    }

    pub(super) fn subscribe_process_availability(&self) -> watch::Receiver<DeviceCapacitySignal> {
        self.account.process_availability_tx.subscribe()
    }
}

impl Drop for DeviceCapacityBudget {
    fn drop(&mut self) {
        let mut state = match self.account.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let previous_effective = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .unwrap_or(self.account.device_capacity_bytes);
        let record = state
            .budgets
            .remove(&self.budget_id)
            .expect("live device capacity budget remains registered");
        assert_eq!(
            record.claimed_bytes, 0,
            "device capacity budget dropped while physical grants remain live"
        );
        let current_effective = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .unwrap_or(self.account.device_capacity_bytes);
        if !state.poisoned && !state.budgets.is_empty() && current_effective > previous_effective {
            publish_process_capacity_increase(&self.account, &mut state);
        }
    }
}

static DEVICE_CAPACITY_ACCOUNTS: OnceLock<Mutex<BTreeMap<DeviceId, Weak<DeviceCapacityAccount>>>> =
    OnceLock::new();

pub(super) fn device_capacity_account(
    device_id: &DeviceId,
    device_runtime_implementation_fingerprint: &str,
    device_capacity_bytes: u64,
) -> Result<Arc<DeviceCapacityAccount>, VNextError> {
    let registry = DEVICE_CAPACITY_ACCOUNTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut registry = registry
        .lock()
        .map_err(|_| invalid_resource("device capacity registry is poisoned"))?;
    let account = match registry.get(device_id).and_then(Weak::upgrade) {
        Some(account) => {
            if account.device_runtime_implementation_fingerprint
                != device_runtime_implementation_fingerprint
                || account.device_capacity_bytes != device_capacity_bytes
            {
                return Err(invalid_resource(format!(
                    "device `{device_id}` has conflicting live runtime or capacity metadata"
                )));
            }
            account
        }
        None => {
            let (process_availability_tx, _) = watch::channel(DeviceCapacitySignal::Live(1));
            let account = Arc::new(DeviceCapacityAccount {
                device_id: device_id.clone(),
                device_runtime_implementation_fingerprint:
                    device_runtime_implementation_fingerprint.to_owned(),
                device_capacity_bytes,
                state: Mutex::new(DeviceCapacityState {
                    claimed_bytes: 0,
                    next_budget_id: 0,
                    budgets: BTreeMap::new(),
                    process_availability_epoch: 1,
                    poisoned: false,
                }),
                process_availability_tx,
            });
            registry.insert(device_id.clone(), Arc::downgrade(&account));
            account
        }
    };
    drop(registry);
    Ok(account)
}

/// Non-cloneable ownership of bytes claimed against the process-wide device
/// account. It follows the device resources, including quarantine ownership.
pub(super) struct DeviceCapacityClaim {
    budget: Option<Arc<DeviceCapacityBudget>>,
    bytes: u64,
}

impl DeviceCapacityClaim {
    pub(super) fn release(&mut self) {
        let bytes = self.bytes;
        self.release_bytes(bytes);
    }

    pub(super) fn release_bytes(&mut self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        assert!(
            self.bytes >= bytes,
            "device capacity claim was returned beyond its live bytes"
        );
        let budget = self
            .budget
            .as_ref()
            .expect("live device capacity bytes retain their plan budget");
        {
            let account = &budget.account;
            let mut state = account
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                state.claimed_bytes >= bytes,
                "device capacity claim was returned more than once"
            );
            state.claimed_bytes -= bytes;
            let record = state
                .budgets
                .get_mut(&budget.budget_id)
                .expect("device capacity claim retains its plan budget");
            assert!(
                record.claimed_bytes >= bytes,
                "device plan capacity claim was returned more than once"
            );
            record.claimed_bytes -= bytes;
            publish_capacity_release(account, &mut state, budget.budget_id);
        }
        self.bytes -= bytes;
        if self.bytes == 0 {
            self.budget.take();
        }
    }

    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }
}

pub(super) struct DeviceCapacityReservation {
    claim: Option<DeviceCapacityClaim>,
}

impl DeviceCapacityReservation {
    pub(super) fn reserve(
        budget: &Arc<DeviceCapacityBudget>,
        bytes: u64,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            claim: Some(budget.account.claim(budget, bytes)?),
        })
    }

    pub(super) fn commit_split(
        mut self,
        parts: &[u64],
    ) -> Result<Vec<DeviceCapacityGrant>, VNextError> {
        let expected = parts.iter().try_fold(0_u64, |total, &bytes| {
            if bytes == 0 {
                return Err(invalid_resource(
                    "device capacity grant part must be non-zero",
                ));
            }
            total
                .checked_add(bytes)
                .ok_or_else(|| invalid_resource("device capacity grant parts overflow u64"))
        })?;
        let mut claim = self
            .claim
            .take()
            .ok_or_else(|| invalid_resource("device capacity reservation was already committed"))?;
        if claim.bytes != expected {
            return Err(invalid_resource(
                "device capacity grant parts differ from the atomic reservation",
            ));
        }
        let budget = claim
            .budget
            .take()
            .ok_or_else(|| invalid_resource("device capacity reservation has no live budget"))?;
        claim.bytes = 0;
        Ok(parts
            .iter()
            .map(|&bytes| DeviceCapacityGrant {
                claim: Some(DeviceCapacityClaim {
                    budget: Some(Arc::clone(&budget)),
                    bytes,
                }),
            })
            .collect())
    }
}

pub(super) struct DeviceCapacityGrant {
    claim: Option<DeviceCapacityClaim>,
}

impl DeviceCapacityGrant {
    pub(super) fn bytes(&self) -> u64 {
        self.claim.as_ref().map_or(0, DeviceCapacityClaim::bytes)
    }
}

impl Drop for DeviceCapacityClaim {
    fn drop(&mut self) {
        self.release();
    }
}

pub(super) fn issue_generation() -> Result<u64, VNextError> {
    NEXT_ADMISSION_GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| invalid_resource("resource admission generation space is exhausted"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn account(capacity_bytes: u64) -> Arc<DeviceCapacityAccount> {
        let generation = issue_generation().unwrap();
        device_capacity_account(
            &DeviceId::new(format!("device.capacity-signal-test-{generation}")).unwrap(),
            "capacity-signal-test-runtime",
            capacity_bytes,
        )
        .unwrap()
    }

    #[test]
    fn releasing_one_plan_advances_its_plan_and_process_generations() {
        let account = account(128);
        let first = account.register_budget(128).unwrap();
        let second = account.register_budget(128).unwrap();
        let first_before = first.availability_snapshot().unwrap();
        let second_before = second.availability_snapshot().unwrap();
        let reservation = DeviceCapacityReservation::reserve(&first, 64).unwrap();

        drop(reservation);

        let first_after = first.availability_snapshot().unwrap();
        let second_after = second.availability_snapshot().unwrap();
        assert_eq!(first_after.plan_epoch(), first_before.plan_epoch() + 1);
        assert_eq!(
            first_after.process_epoch(),
            first_before.process_epoch() + 1
        );
        assert_eq!(second_after.plan_epoch(), second_before.plan_epoch());
        assert_eq!(
            second_after.process_epoch(),
            second_before.process_epoch() + 1
        );
    }

    #[test]
    fn pressure_scope_selects_the_exact_capacity_generation() {
        let account = account(128);
        let first = account.register_budget(128).unwrap();
        let second = account.register_budget(96).unwrap();
        let held = DeviceCapacityReservation::reserve(&first, 96).unwrap();
        let second_snapshot = second.availability_snapshot().unwrap();

        let pressure = match DeviceCapacityReservation::reserve(&second, 64) {
            Err(VNextError::DeviceCapacityUnavailable(pressure)) => pressure,
            _ => panic!("shared device pressure must remain typed"),
        };
        assert_eq!(pressure.scope(), &DeviceCapacityPressureScope::ProcessWide);
        assert_eq!(
            second_snapshot.epoch_for_pressure(&pressure),
            CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::ProcessDeviceCapacity,
                second_snapshot.process_epoch(),
            )
            .unwrap()
        );

        drop(held);
        let after = second.availability_snapshot().unwrap();
        assert_eq!(after.plan_epoch(), second_snapshot.plan_epoch());
        assert!(after.process_epoch() > second_snapshot.process_epoch());
    }
}
