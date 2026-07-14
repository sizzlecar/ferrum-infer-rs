use super::{
    invalid_resource, Arc, AtomicU64, BTreeMap, BatchInvocationId, BatchStepId, DeviceId, Mutex,
    OnceLock, Ordering, VNextError, Weak,
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
}

#[derive(Debug)]
pub(super) struct DeviceCapacityBudgetRecord {
    /// A conservative device-wide ceiling contributed by this live plan. It is
    /// not an additive plan share.
    pub(super) device_wide_usable_ceiling_bytes: u64,
    pub(super) claimed_bytes: u64,
}

#[derive(Debug, Default)]
pub(super) struct DeviceCapacityState {
    pub(super) claimed_bytes: u64,
    pub(super) next_budget_id: u64,
    pub(super) budgets: BTreeMap<u64, DeviceCapacityBudgetRecord>,
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
        state.budgets.insert(
            budget_id,
            DeviceCapacityBudgetRecord {
                device_wide_usable_ceiling_bytes,
                claimed_bytes: 0,
            },
        );
        Ok(Arc::new(DeviceCapacityBudget {
            account: Arc::clone(self),
            budget_id,
            device_wide_usable_ceiling_bytes,
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
        let next_budget_claimed = budget_record
            .claimed_bytes
            .checked_add(bytes)
            .filter(|next| *next <= budget.device_wide_usable_ceiling_bytes)
            .ok_or_else(|| {
                invalid_resource(format!(
                    "device `{}` plan budget exceeds usable capacity: claimed {}, requested {}, usable {}",
                    self.device_id,
                    budget_record.claimed_bytes,
                    bytes,
                    budget.device_wide_usable_ceiling_bytes
                ))
            })?;
        let next = state
            .claimed_bytes
            .checked_add(bytes)
            .filter(|next| *next <= effective_usable_capacity)
            .ok_or_else(|| {
                invalid_resource(format!(
                    "device `{}` resource admission exceeds live usable capacity: claimed {}, requested {}, effective usable {}, raw capacity {}",
                    self.device_id,
                    state.claimed_bytes,
                    bytes,
                    effective_usable_capacity,
                    self.device_capacity_bytes
                ))
            })?;
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
}

impl Drop for DeviceCapacityBudget {
    fn drop(&mut self) {
        let mut state = match self.account.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let record = state
            .budgets
            .remove(&self.budget_id)
            .expect("live device capacity budget remains registered");
        assert_eq!(
            record.claimed_bytes, 0,
            "device capacity budget dropped while physical grants remain live"
        );
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
            let account = Arc::new(DeviceCapacityAccount {
                device_id: device_id.clone(),
                device_runtime_implementation_fingerprint:
                    device_runtime_implementation_fingerprint.to_owned(),
                device_capacity_bytes,
                state: Mutex::new(DeviceCapacityState::default()),
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
