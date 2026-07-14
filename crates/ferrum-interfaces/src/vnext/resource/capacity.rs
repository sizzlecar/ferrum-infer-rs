use super::{
    invalid_resource, validate_runtime_descriptor_for_admission, Arc, AtomicU64, BTreeMap,
    BTreeSet, BatchInvocationId, BatchStepId, CapacityDomainId, CapacityDomainSpec, CapacityUnits,
    DeviceId, DeviceRuntime, DynamicBackingPoolId, DynamicBackingPoolSpec,
    DynamicPoolMaintenanceController, DynamicPoolSet, DynamicResourceDescriptor, ExecutionPlan,
    LogicalAdmissionCoordinator, Mutex, OnceLock, Ordering, PlanHash, PlanId, PlanNode,
    RequestIdentity, ResourcePoolId, ResourcePoolIdentity, ResourceReservationBatch, Serialize,
    StaticProvisioningBinding, VNextError, Weak,
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

    fn claim(
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

/// One-shot plan/admission authority. It cannot be constructed, cloned, or
/// deserialized by product or backend code. `ResourceTransaction::begin`
/// consumes it, closing the old caller-built reservation bypass.
#[must_use = "an admission permit must be consumed by ResourceTransaction::begin"]
pub struct StaticProvisioningPermit<R>
where
    R: DeviceRuntime,
{
    pub(super) maintenance_controller: DynamicPoolMaintenanceController<R>,
    pub(super) dynamic_pools: Arc<DynamicPoolSet<R>>,
    pub(super) reservations: ResourceReservationBatch,
    pub(super) capacity_claim: DeviceCapacityClaim,
    pub(super) binding: StaticProvisioningBinding,
    pub(super) runtime: Arc<R>,
    pub(super) seal: AdmissionSeal,
}

pub(super) struct AdmissionSeal;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct DynamicPoolDomainSpec {
    pub(super) domain_id: CapacityDomainId,
    pub(super) pool: DynamicBackingPoolSpec,
    pub(super) descriptors: Vec<DynamicResourceDescriptor>,
}

impl DynamicPoolDomainSpec {
    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub(super) fn pool_id(&self) -> &DynamicBackingPoolId {
        self.pool.pool_id()
    }
}

pub(super) fn plan_dynamic_pool_admission(
    maximum_active_sequences: u32,
    pools: &[DynamicBackingPoolSpec],
    descriptors: &[DynamicResourceDescriptor],
) -> Result<(LogicalAdmissionCoordinator, Vec<DynamicPoolDomainSpec>), VNextError> {
    let mut descriptors_by_id = descriptors
        .iter()
        .map(|descriptor| (descriptor.base_resource_id().clone(), descriptor.clone()))
        .collect::<BTreeMap<_, _>>();
    if descriptors_by_id.len() != descriptors.len() || pools.is_empty() != descriptors.is_empty() {
        return Err(invalid_resource(
            "dynamic pool catalog and descriptor membership are inconsistent",
        ));
    }
    let mut seen_pools = BTreeSet::new();
    let mut domains = Vec::with_capacity(pools.len());
    for (index, pool) in pools.iter().enumerate() {
        if !seen_pools.insert(pool.pool_id().clone()) {
            return Err(invalid_resource(
                "dynamic pool catalog contains a duplicate pool",
            ));
        }
        let mut members = Vec::with_capacity(pool.resource_ids().len());
        for resource_id in pool.resource_ids() {
            let descriptor = descriptors_by_id.remove(resource_id).ok_or_else(|| {
                invalid_resource("dynamic pool references an unknown or duplicate descriptor")
            })?;
            if descriptor.pool_id() != pool.pool_id() {
                return Err(invalid_resource(
                    "dynamic descriptor belongs to another core-derived pool",
                ));
            }
            members.push(descriptor);
        }
        members.sort_by(|left, right| left.base_resource_id().cmp(right.base_resource_id()));
        let domain_id = CapacityDomainId::new(
            u32::try_from(index + 1)
                .map_err(|_| invalid_resource("dynamic pool domain id exceeds u32"))?,
        )?;
        domains.push(DynamicPoolDomainSpec {
            domain_id,
            pool: pool.clone(),
            descriptors: members,
        });
    }
    if !descriptors_by_id.is_empty() {
        return Err(invalid_resource(
            "dynamic descriptors are missing from the core-derived pool catalog",
        ));
    }
    let coordinator_domains = domains
        .iter()
        .map(|domain| {
            Ok((
                domain.domain_id,
                CapacityDomainSpec::new(
                    CapacityUnits::ZERO,
                    CapacityUnits::new(domain.pool.provisioning().maximum_resident_bytes()),
                )?,
            ))
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    Ok((
        LogicalAdmissionCoordinator::new(coordinator_domains, maximum_active_sequences)?,
        domains,
    ))
}

impl<R> StaticProvisioningPermit<R>
where
    R: DeviceRuntime,
{
    pub fn maintenance_controller(&self) -> &DynamicPoolMaintenanceController<R> {
        &self.maintenance_controller
    }

    pub fn binding(&self) -> &StaticProvisioningBinding {
        &self.binding
    }

    pub fn reservations(&self) -> &ResourceReservationBatch {
        &self.reservations
    }
}

/// Explicit no-op result for plans that have no plan-lifetime buffers. It
/// binds the validated plan to one exact runtime without manufacturing an
/// empty reservation ledger or a zero-byte device-capacity claim.
#[must_use = "no-static provisioning must be retained while the plan runtime is live"]
pub struct NoStatic<R>
where
    R: DeviceRuntime,
{
    pub(super) maintenance_controller: DynamicPoolMaintenanceController<R>,
    pub(super) dynamic_pools: Arc<DynamicPoolSet<R>>,
    pub(super) binding: StaticProvisioningBinding,
    pub(super) runtime: Arc<R>,
}

impl<R> NoStatic<R>
where
    R: DeviceRuntime,
{
    pub fn maintenance_controller(&self) -> &DynamicPoolMaintenanceController<R> {
        &self.maintenance_controller
    }

    pub fn plan_id(&self) -> &PlanId {
        self.binding.plan_id()
    }

    pub fn plan_hash(&self) -> &PlanHash {
        self.binding.plan_hash()
    }

    pub fn device_id(&self) -> &DeviceId {
        self.binding.device_id()
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        self.binding.device_runtime_implementation_fingerprint()
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.binding.device_capacity_bytes()
    }

    pub const fn usable_capacity_bytes(&self) -> u64 {
        self.binding.usable_capacity_bytes()
    }

    pub const fn maximum_active_sequences(&self) -> u32 {
        self.binding.maximum_active_sequences()
    }
}

/// Static provisioning has two physically distinct outcomes. Only `Required`
/// carries transaction authority; `NoStatic` cannot be passed to
/// `ResourceTransaction::begin`.
#[must_use = "static provisioning must be retained or committed"]
pub enum StaticProvisioning<R>
where
    R: DeviceRuntime,
{
    NoStatic(NoStatic<R>),
    Required(StaticProvisioningPermit<R>),
}

/// The indivisible result of plan provisioning. Product code must consume
/// this owner through [`Self::into_parts`], which hands out the plan runtime
/// outcome and its unique maintenance controller together. There is no
/// controller-less extraction path.
#[must_use = "provisioned plan resources must be split into their runtime and maintenance owners"]
pub struct ProvisionedPlanResources<R>
where
    R: DeviceRuntime,
{
    provisioning: StaticProvisioning<R>,
}

/// Named result of consuming [`ProvisionedPlanResources`]. Keeping both
/// fields in product ownership prevents maintenance authority from being
/// silently discarded while request admission remains live.
#[must_use = "both plan provisioning and maintenance ownership must be retained"]
pub struct ProvisionedPlanParts<R>
where
    R: DeviceRuntime,
{
    pub provisioning: StaticProvisioning<R>,
}

impl<R> StaticProvisioning<R>
where
    R: DeviceRuntime,
{
    pub const fn has_static_resources(&self) -> bool {
        matches!(self, Self::Required(_))
    }
}

impl<R> ProvisionedPlanResources<R>
where
    R: DeviceRuntime,
{
    pub(super) fn new(provisioning: StaticProvisioning<R>) -> Self {
        Self { provisioning }
    }

    pub fn provisioning(&self) -> &StaticProvisioning<R> {
        &self.provisioning
    }

    pub fn into_parts(self) -> ProvisionedPlanParts<R> {
        ProvisionedPlanParts {
            provisioning: self.provisioning,
        }
    }

    pub fn into_provisioning(self) -> StaticProvisioning<R> {
        self.provisioning
    }
}

impl ExecutionPlan {
    /// Provisions only plan-lifetime buffers. Dynamic sequence admission is a
    /// separate logical authority and is never implied by this result.
    pub fn provision_static<R>(
        &self,
        runtime: Arc<R>,
        request_id: RequestIdentity,
    ) -> Result<ProvisionedPlanResources<R>, VNextError>
    where
        R: DeviceRuntime,
    {
        let payload = self.payload();
        let memory = payload.memory();
        if runtime.descriptor().id != *payload.device_id()
            || runtime.descriptor().runtime_implementation_fingerprint
                != payload.device_runtime_implementation_fingerprint()
            || runtime.descriptor().total_memory_bytes != memory.device_capacity_bytes()
        {
            return Err(invalid_resource(
                "static provisioning runtime differs from the execution plan",
            ));
        }
        let (logical_admission, domains) = plan_dynamic_pool_admission(
            memory.maximum_active_sequences(),
            memory.dynamic_pools(),
            memory.dynamic_descriptors(),
        )?;
        let generation = issue_generation()?;
        let reservations = ResourceReservationBatch::from_allocations(
            &request_id,
            memory.static_allocations(),
            generation,
        )?;
        if memory.device_capacity_bytes() == 0
            || memory.usable_capacity_bytes() == 0
            || memory.usable_capacity_bytes() > memory.device_capacity_bytes()
            || memory.static_bytes() > memory.usable_capacity_bytes()
            || memory.maximum_active_sequences() == 0
            || (memory.static_bytes() == 0) != memory.static_allocations().is_empty()
            || reservations.plan_static_size_bytes() != memory.static_bytes()
            || reservations.total_size_bytes() != memory.static_bytes()
        {
            return Err(invalid_resource(
                "plan-static reservation or capacity evidence is invalid",
            ));
        }
        let pool_identity = ResourcePoolIdentity {
            pool_id: ResourcePoolId::issue(generation)?,
            plan_id: payload.plan_id().clone(),
            plan_hash: self.plan_hash().clone(),
            device_id: payload.device_id().clone(),
            device_runtime_implementation_fingerprint: payload
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            admission_generation: generation,
        };
        let binding = StaticProvisioningBinding {
            pool_identity,
            plan_id: payload.plan_id().clone(),
            plan_hash: self.plan_hash().clone(),
            request_id,
            device_id: payload.device_id().clone(),
            device_runtime_implementation_fingerprint: payload
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            device_capacity_bytes: memory.device_capacity_bytes(),
            usable_capacity_bytes: memory.usable_capacity_bytes(),
            plan_static_bytes: memory.static_bytes(),
            admitted_bytes: memory.static_bytes(),
            maximum_active_sequences: memory.maximum_active_sequences(),
            admission_generation: generation,
        };
        validate_runtime_descriptor_for_admission(
            runtime.descriptor(),
            &binding,
            "resource admission preflight",
        )?;
        let account = device_capacity_account(
            binding.device_id(),
            binding.device_runtime_implementation_fingerprint(),
            binding.device_capacity_bytes(),
        )?;
        let budget = account.register_budget(binding.usable_capacity_bytes())?;
        let nodes: Arc<[PlanNode]> = Arc::from(payload.nodes().to_vec());
        let dynamic_pools = Arc::new(DynamicPoolSet::new(
            Arc::clone(&runtime),
            binding.clone(),
            Arc::clone(&budget),
            logical_admission,
            domains,
            nodes,
        )?);
        validate_runtime_descriptor_for_admission(
            runtime.descriptor(),
            &binding,
            "resource admission completion",
        )?;
        let maintenance_controller =
            DynamicPoolMaintenanceController::new(Arc::clone(&dynamic_pools));
        if memory.static_allocations().is_empty() {
            return Ok(ProvisionedPlanResources::new(StaticProvisioning::NoStatic(
                NoStatic {
                    maintenance_controller,
                    dynamic_pools,
                    binding,
                    runtime,
                },
            )));
        }
        let capacity_claim = account.claim(&budget, reservations.total_size_bytes())?;
        Ok(ProvisionedPlanResources::new(StaticProvisioning::Required(
            StaticProvisioningPermit {
                maintenance_controller,
                dynamic_pools,
                reservations,
                capacity_claim,
                binding,
                runtime,
                seal: AdmissionSeal,
            },
        )))
    }
}
