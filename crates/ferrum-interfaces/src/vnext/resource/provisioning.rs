use super::{
    device_capacity_account, invalid_resource, issue_generation,
    validate_runtime_descriptor_for_admission, Arc, BTreeMap, BTreeSet, CapacityDomainId,
    CapacityDomainSpec, CapacityUnits, DeviceCapacityClaim, DeviceId, DeviceRuntime,
    DynamicBackingPoolSpec, DynamicPoolDomainSpec, DynamicPoolMaintenanceController,
    DynamicPoolSet, DynamicResourceDescriptor, ExecutionPlan, LogicalAdmissionCoordinator,
    PlanHash, PlanId, PlanNode, RequestIdentity, ResourcePoolId, ResourcePoolIdentity,
    ResourceReservationBatch, StaticProvisioningBinding, VNextError,
};

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
