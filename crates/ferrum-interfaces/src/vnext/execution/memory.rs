use super::{
    invalid_plan, minimum_for_lifetime, node_completion_precedes, quantize_storage_bytes,
    static_contiguous_storage_profile, validate_active_sequence_ceiling,
    validate_pool_liveness_rows, workspace_storage_layout_fingerprint, AllocationKind,
    AllocationLifetime, BTreeMap, BTreeSet, BufferRequest, BufferUsage, CanonicalU128, Deserialize,
    Deserializer, DynamicBackingPoolId, DynamicBackingPoolSpec, DynamicResourceDemand,
    DynamicResourceDescriptor, DynamicResourceShape, ElementType, InvocationLivenessMode,
    InvocationResourceLiveness, NodeId, PlanNode, PoolAggregateEvidence, PoolCompatibilityKey,
    ResolvedReusableExecutionBucket, ResourceAllocation, ResourceId, ReusableExecutionMemoryPlan,
    ReusableExecutionPolicy, ReusablePoolWorkspaceBudget, Serialize, StepResourceSlot,
    StepResourceSlotKind, VNextError, MAX_EXECUTION_PLAN_RESOURCE_ROWS,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryPlan {
    pub(super) device_capacity_bytes: u64,
    pub(super) policy_capacity_bytes: u64,
    pub(super) reserve_bytes: u64,
    pub(super) usable_capacity_bytes: u64,
    pub(super) maximum_active_sequences: u32,
    pub(super) static_bytes: u64,
    pub(super) minimum_request_bytes: u64,
    pub(super) minimum_sequence_bytes: u64,
    pub(super) minimum_step_bytes: u64,
    pub(super) minimum_invocation_peak_bytes: u64,
    pub(super) minimum_runnable_request_bytes: u64,
    pub(super) theoretical_ceiling_bytes: CanonicalU128,
    pub(super) static_allocations: Vec<ResourceAllocation>,
    pub(super) dynamic_descriptors: Vec<DynamicResourceDescriptor>,
    pub(super) dynamic_pools: Vec<DynamicBackingPoolSpec>,
    pub(super) reusable_execution: Option<ReusableExecutionMemoryPlan>,
    pub(super) invocation_liveness_mode: InvocationLivenessMode,
    pub(super) invocation_liveness: Vec<InvocationResourceLiveness>,
}

impl MemoryPlan {
    #[cfg(test)]
    pub(super) fn from_core(
        device_capacity_bytes: u64,
        policy_capacity_bytes: u64,
        reserve_bytes: u64,
        maximum_active_sequences: u32,
        static_allocations: Vec<ResourceAllocation>,
        dynamic_descriptors: Vec<DynamicResourceDescriptor>,
        nodes: &[PlanNode],
        reusable_execution_policy: Option<&ReusableExecutionPolicy>,
    ) -> Result<Self, VNextError> {
        Self::from_core_with_completion_retention(
            device_capacity_bytes,
            policy_capacity_bytes,
            reserve_bytes,
            maximum_active_sequences,
            static_allocations,
            dynamic_descriptors,
            nodes,
            reusable_execution_policy,
            &BTreeSet::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_core_with_completion_retention(
        device_capacity_bytes: u64,
        policy_capacity_bytes: u64,
        reserve_bytes: u64,
        maximum_active_sequences: u32,
        mut static_allocations: Vec<ResourceAllocation>,
        mut dynamic_descriptors: Vec<DynamicResourceDescriptor>,
        nodes: &[PlanNode],
        reusable_execution_policy: Option<&ReusableExecutionPolicy>,
        retained_completion_resources: &BTreeSet<ResourceId>,
    ) -> Result<Self, VNextError> {
        if static_allocations.len() + dynamic_descriptors.len() > MAX_EXECUTION_PLAN_RESOURCE_ROWS {
            return Err(invalid_plan(format!(
                "execution plan resource rows exceed {MAX_EXECUTION_PLAN_RESOURCE_ROWS}"
            )));
        }
        let usable_capacity_bytes = policy_capacity_bytes
            .checked_sub(reserve_bytes)
            .ok_or_else(|| invalid_plan("memory reserve exceeds the policy capacity"))?;
        static_allocations.sort_by(|left, right| left.resource_id.cmp(&right.resource_id));
        dynamic_descriptors
            .sort_by(|left, right| left.base_resource_id.cmp(&right.base_resource_id));
        for resource_id in retained_completion_resources {
            let descriptor = dynamic_descriptors
                .iter()
                .find(|descriptor| &descriptor.base_resource_id == resource_id)
                .ok_or_else(|| {
                    invalid_plan(format!(
                        "retained completion resource `{resource_id}` has no dynamic descriptor"
                    ))
                })?;
            if descriptor.usage != BufferUsage::Activations
                || !matches!(
                    descriptor.lifetime,
                    AllocationLifetime::Step | AllocationLifetime::Request
                )
            {
                return Err(invalid_plan(format!(
                    "retained completion resource `{resource_id}` is not a readable activation"
                )));
            }
        }
        let static_bytes = static_allocations
            .iter()
            .try_fold(0_u64, |total, allocation| {
                total
                    .checked_add(allocation.size_bytes)
                    .ok_or_else(|| invalid_plan("static memory total overflows u64"))
            })?;
        let dynamic_capacity_bytes = usable_capacity_bytes
            .checked_sub(static_bytes)
            .ok_or_else(|| invalid_plan("static memory exceeds usable capacity"))?;
        let base_dynamic_pools = Self::derive_dynamic_pools_with_completion_retention(
            &dynamic_descriptors,
            nodes,
            dynamic_capacity_bytes,
            retained_completion_resources,
        )?;
        let reusable_execution = reusable_execution_policy
            .map(|policy| {
                Self::derive_reusable_execution(
                    policy,
                    nodes.len(),
                    &dynamic_descriptors,
                    &base_dynamic_pools,
                )
            })
            .transpose()?;
        let reusable_workspace_ceilings = reusable_execution
            .as_ref()
            .map(ReusableExecutionMemoryPlan::pool_workspace_ceilings)
            .transpose()?
            .unwrap_or_default();
        let dynamic_pools = Self::derive_dynamic_pools_with_reusable(
            &dynamic_descriptors,
            nodes,
            dynamic_capacity_bytes,
            &reusable_workspace_ceilings,
            retained_completion_resources,
        )?;
        let (invocation_liveness_mode, invocation_liveness) =
            Self::summarize_pool_invocation_liveness(&dynamic_pools)?;
        let minimum_request_bytes = dynamic_pools.iter().try_fold(0_u64, |total, pool| {
            total
                .checked_add(pool.minimum_request_bytes)
                .ok_or_else(|| invalid_plan("minimum request bytes overflow u64"))
        })?;
        let minimum_sequence_bytes = dynamic_pools.iter().try_fold(0_u64, |total, pool| {
            total
                .checked_add(pool.minimum_sequence_bytes)
                .ok_or_else(|| invalid_plan("minimum sequence bytes overflow u64"))
        })?;
        let minimum_step_bytes = dynamic_pools.iter().try_fold(0_u64, |total, pool| {
            total
                .checked_add(pool.minimum_step_bytes)
                .ok_or_else(|| invalid_plan("minimum step bytes overflow u64"))
        })?;
        let minimum_invocation_peak_bytes =
            dynamic_pools.iter().try_fold(0_u64, |total, pool| {
                total
                    .checked_add(pool.minimum_invocation_peak_bytes)
                    .ok_or_else(|| invalid_plan("invocation pool minimum bytes overflow u64"))
            })?;
        let minimum_runnable_request_bytes = minimum_request_bytes
            .checked_add(minimum_sequence_bytes)
            .and_then(|bytes| bytes.checked_add(minimum_step_bytes))
            .and_then(|bytes| bytes.checked_add(minimum_invocation_peak_bytes))
            .ok_or_else(|| invalid_plan("minimum runnable request bytes overflow u64"))?;
        let theoretical_dynamic_ceiling =
            dynamic_pools.iter().try_fold(0_u128, |total, pool| {
                total
                    .checked_add(pool.theoretical_ceiling_bytes.get())
                    .and_then(|bytes| {
                        bytes.checked_add(u128::from(pool.reusable_workspace_ceiling_bytes))
                    })
                    .ok_or_else(|| invalid_plan("dynamic theoretical total overflows u128"))
            })?;
        let theoretical_ceiling_bytes = u128::from(static_bytes)
            .checked_add(theoretical_dynamic_ceiling)
            .ok_or_else(|| invalid_plan("plan theoretical ceiling overflows u128"))?;
        let minimum_runnable_bytes = static_bytes
            .checked_add(minimum_runnable_request_bytes)
            .ok_or_else(|| invalid_plan("minimum runnable plan bytes overflow u64"))?;
        if minimum_runnable_bytes > usable_capacity_bytes {
            return Err(invalid_plan(format!(
                "minimum runnable plan requires {minimum_runnable_bytes} bytes, exceeding usable capacity {usable_capacity_bytes}"
            )));
        }
        let plan = Self {
            device_capacity_bytes,
            policy_capacity_bytes,
            reserve_bytes,
            usable_capacity_bytes,
            maximum_active_sequences,
            static_bytes,
            minimum_request_bytes,
            minimum_sequence_bytes,
            minimum_step_bytes,
            minimum_invocation_peak_bytes,
            minimum_runnable_request_bytes,
            theoretical_ceiling_bytes: CanonicalU128::new(theoretical_ceiling_bytes),
            static_allocations,
            dynamic_descriptors,
            dynamic_pools,
            reusable_execution,
            invocation_liveness_mode,
            invocation_liveness,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub(super) fn validate(&self) -> Result<(), VNextError> {
        validate_active_sequence_ceiling(self.maximum_active_sequences)?;
        if self.static_allocations.len() + self.dynamic_descriptors.len()
            > MAX_EXECUTION_PLAN_RESOURCE_ROWS
        {
            return Err(invalid_plan(format!(
                "execution plan resource rows exceed {MAX_EXECUTION_PLAN_RESOURCE_ROWS}"
            )));
        }
        if self
            .static_allocations
            .windows(2)
            .any(|pair| pair[0].resource_id >= pair[1].resource_id)
            || self
                .dynamic_descriptors
                .windows(2)
                .any(|pair| pair[0].base_resource_id >= pair[1].base_resource_id)
            || self
                .dynamic_pools
                .windows(2)
                .any(|pair| pair[0].pool_id >= pair[1].pool_id)
        {
            return Err(invalid_plan("memory plan resource rows are not canonical"));
        }
        if self.device_capacity_bytes == 0
            || self.policy_capacity_bytes == 0
            || self.policy_capacity_bytes > self.device_capacity_bytes
            || self.reserve_bytes >= self.policy_capacity_bytes
            || self.usable_capacity_bytes
                != self
                    .policy_capacity_bytes
                    .checked_sub(self.reserve_bytes)
                    .ok_or_else(|| invalid_plan("memory reserve underflows policy capacity"))?
        {
            return Err(invalid_plan("memory plan capacity or reserve is invalid"));
        }
        let mut resources = BTreeSet::new();
        let workspace_layout_fingerprint = workspace_storage_layout_fingerprint()?;
        let static_contiguous_profile = static_contiguous_storage_profile()?;
        let actual_static =
            self.static_allocations
                .iter()
                .try_fold(0_u64, |total, allocation| {
                    if !resources.insert(allocation.resource_id.clone())
                        || allocation.per_instance_bytes == 0
                        || allocation.instance_stride_bytes
                            != quantize_storage_bytes(
                                allocation.per_instance_bytes,
                                allocation.alignment_bytes,
                                allocation.storage.profile(),
                            )?
                        || allocation.instance_count != 1
                        || allocation.size_bytes != allocation.instance_stride_bytes
                        || allocation.lifetime != AllocationLifetime::Plan
                        || match &allocation.kind {
                            AllocationKind::Value => {
                                allocation.storage.profile() != static_contiguous_profile
                            }
                            AllocationKind::Persistent { .. } => {
                                allocation.usage != BufferUsage::Persistent
                                    || allocation.element_type != ElementType::U8
                                    || allocation.storage.logical_layout_fingerprint()
                                        != workspace_layout_fingerprint
                            }
                            AllocationKind::Scratch { .. } | AllocationKind::Binding { .. } => true,
                        }
                    {
                        return Err(invalid_plan(format!(
                            "static resource `{}` is duplicate or invalid",
                            allocation.resource_id
                        )));
                    }
                    total
                        .checked_add(allocation.size_bytes)
                        .ok_or_else(|| invalid_plan("static allocation total overflows u64"))
                })?;
        if actual_static != self.static_bytes {
            return Err(invalid_plan("static byte total is not core-derived"));
        }
        let mut actual_theoretical_dynamic = 0_u128;
        let dynamic_by_id = self
            .dynamic_descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), descriptor))
            .collect::<BTreeMap<_, _>>();
        for descriptor in &self.dynamic_descriptors {
            if !resources.insert(descriptor.base_resource_id.clone()) {
                return Err(invalid_plan(format!(
                    "dynamic resource `{}` is duplicated",
                    descriptor.base_resource_id
                )));
            }
            validate_active_sequence_ceiling(descriptor.theoretical_maximum_instances)?;
            descriptor.demand.validate()?;
            if descriptor.lifetime == AllocationLifetime::Plan {
                return Err(invalid_plan(
                    "dynamic descriptor cannot have plan-static lifetime",
                ));
            }
            actual_theoretical_dynamic = actual_theoretical_dynamic
                .checked_add(
                    u128::from(descriptor.theoretical_maximum_request_bytes()?)
                        * u128::from(descriptor.theoretical_maximum_instances),
                )
                .ok_or_else(|| invalid_plan("dynamic ceiling total overflows u128"))?;
        }
        let reusable_workspace_ceilings = self
            .reusable_execution
            .as_ref()
            .map(|plan| {
                plan.validate_local()?;
                plan.pool_workspace_ceilings()
            })
            .transpose()?
            .unwrap_or_default();
        let actual_reusable_workspace =
            reusable_workspace_ceilings
                .values()
                .try_fold(0_u128, |total, bytes| {
                    total
                        .checked_add(u128::from(*bytes))
                        .ok_or_else(|| invalid_plan("reusable workspace total overflows u128"))
                })?;
        let actual_theoretical = u128::from(actual_static)
            .checked_add(actual_theoretical_dynamic)
            .and_then(|bytes| bytes.checked_add(actual_reusable_workspace))
            .ok_or_else(|| invalid_plan("theoretical plan ceiling overflows u128"))?;
        let dynamic_capacity_bytes = self
            .usable_capacity_bytes
            .checked_sub(actual_static)
            .ok_or_else(|| invalid_plan("static memory exceeds usable capacity"))?;
        let aggregate = Self::validate_dynamic_pools_structure(
            &self.dynamic_pools,
            &dynamic_by_id,
            dynamic_capacity_bytes,
            &reusable_workspace_ceilings,
        )?;
        let (actual_liveness_mode, actual_liveness) =
            Self::summarize_pool_invocation_liveness(&self.dynamic_pools)?;
        if actual_liveness_mode != self.invocation_liveness_mode
            || actual_liveness != self.invocation_liveness
        {
            return Err(invalid_plan(
                "global invocation liveness is not derived from per-pool evidence",
            ));
        }
        let actual_request_minimum = aggregate.minimum_request_bytes;
        let actual_sequence_minimum = aggregate.minimum_sequence_bytes;
        let actual_step_minimum = aggregate.minimum_step_bytes;
        let actual_invocation_peak = aggregate.minimum_invocation_peak_bytes;
        let actual_minimum = actual_request_minimum
            .checked_add(actual_sequence_minimum)
            .and_then(|bytes| bytes.checked_add(actual_step_minimum))
            .and_then(|bytes| bytes.checked_add(actual_invocation_peak))
            .ok_or_else(|| invalid_plan("minimum runnable request bytes overflow u64"))?;
        if aggregate.theoretical_ceiling_bytes != actual_theoretical_dynamic
            || aggregate.reusable_workspace_ceiling_bytes != actual_reusable_workspace
            || actual_request_minimum != self.minimum_request_bytes
            || actual_sequence_minimum != self.minimum_sequence_bytes
            || actual_step_minimum != self.minimum_step_bytes
            || actual_invocation_peak != self.minimum_invocation_peak_bytes
            || actual_minimum != self.minimum_runnable_request_bytes
            || actual_theoretical != self.theoretical_ceiling_bytes.get()
            || actual_static
                .checked_add(actual_minimum)
                .is_none_or(|minimum| minimum > self.usable_capacity_bytes)
        {
            return Err(invalid_plan(
                "dynamic minimum or theoretical ceiling is not core-derived",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn derive_dynamic_pools(
        dynamic_descriptors: &[DynamicResourceDescriptor],
        nodes: &[PlanNode],
        dynamic_capacity_bytes: u64,
    ) -> Result<Vec<DynamicBackingPoolSpec>, VNextError> {
        Self::derive_dynamic_pools_with_reusable(
            dynamic_descriptors,
            nodes,
            dynamic_capacity_bytes,
            &BTreeMap::new(),
            &BTreeSet::new(),
        )
    }

    pub(super) fn derive_dynamic_pools_with_completion_retention(
        dynamic_descriptors: &[DynamicResourceDescriptor],
        nodes: &[PlanNode],
        dynamic_capacity_bytes: u64,
        retained_completion_resources: &BTreeSet<ResourceId>,
    ) -> Result<Vec<DynamicBackingPoolSpec>, VNextError> {
        Self::derive_dynamic_pools_with_reusable(
            dynamic_descriptors,
            nodes,
            dynamic_capacity_bytes,
            &BTreeMap::new(),
            retained_completion_resources,
        )
    }

    pub(super) fn derive_dynamic_pools_with_reusable(
        dynamic_descriptors: &[DynamicResourceDescriptor],
        nodes: &[PlanNode],
        dynamic_capacity_bytes: u64,
        reusable_workspace_ceilings: &BTreeMap<DynamicBackingPoolId, u64>,
        retained_completion_resources: &BTreeSet<ResourceId>,
    ) -> Result<Vec<DynamicBackingPoolSpec>, VNextError> {
        let mut groups = BTreeMap::<
            DynamicBackingPoolId,
            (PoolCompatibilityKey, Vec<&DynamicResourceDescriptor>),
        >::new();
        for descriptor in dynamic_descriptors {
            let compatibility = PoolCompatibilityKey::new(
                &descriptor.storage,
                descriptor.usage,
                descriptor.element_type,
                descriptor.alignment_bytes,
            )?;
            let expected_pool_id = DynamicBackingPoolId::from_compatibility(&compatibility)?;
            if descriptor.pool_id != expected_pool_id {
                return Err(invalid_plan(format!(
                    "dynamic resource `{}` has a non-derived backing pool id",
                    descriptor.base_resource_id
                )));
            }
            match groups.entry(expected_pool_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((compatibility, vec![descriptor]));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if entry.get().0 != compatibility {
                        return Err(invalid_plan(
                            "dynamic backing pool hash collision has incompatible contracts",
                        ));
                    }
                    entry.get_mut().1.push(descriptor);
                }
            }
        }

        let pools = groups
            .into_values()
            .map(|(compatibility, mut descriptors)| {
                descriptors
                    .sort_by(|left, right| left.base_resource_id.cmp(&right.base_resource_id));
                let resource_ids = descriptors
                    .iter()
                    .map(|descriptor| descriptor.base_resource_id.clone())
                    .collect::<Vec<_>>();
                let minimum_request_bytes = minimum_for_lifetime(
                    &descriptors,
                    AllocationLifetime::Request,
                    "pool request minimum",
                )?;
                let minimum_sequence_bytes = minimum_for_lifetime(
                    &descriptors,
                    AllocationLifetime::Sequence,
                    "pool sequence minimum",
                )?;
                let step_resource_slots = Self::derive_pool_step_slots(
                    nodes,
                    &descriptors,
                    retained_completion_resources,
                )?;
                let minimum_step_bytes = Self::step_slot_bytes(
                    &step_resource_slots,
                    &descriptors,
                    false,
                    "pool step minimum",
                )?;
                let theoretical_ceiling_bytes =
                    descriptors.iter().try_fold(0_u128, |total, descriptor| {
                        total
                            .checked_add(
                                u128::from(descriptor.theoretical_maximum_request_bytes()?)
                                    * u128::from(descriptor.theoretical_maximum_instances),
                            )
                            .ok_or_else(|| {
                                invalid_plan("dynamic pool theoretical ceiling overflows u128")
                            })
                    })?;
                let (invocation_liveness_mode, invocation_liveness, invocation_peak) =
                    Self::derive_pool_invocation_liveness(nodes, &descriptors)?;
                let pool_id = DynamicBackingPoolId::from_compatibility(&compatibility)?;
                let reusable_workspace_ceiling_bytes = reusable_workspace_ceilings
                    .get(&pool_id)
                    .copied()
                    .unwrap_or(0);
                DynamicBackingPoolSpec::from_core(
                    compatibility,
                    resource_ids,
                    minimum_request_bytes,
                    minimum_sequence_bytes,
                    minimum_step_bytes,
                    invocation_peak,
                    step_resource_slots,
                    theoretical_ceiling_bytes,
                    reusable_workspace_ceiling_bytes,
                    dynamic_capacity_bytes,
                    invocation_liveness_mode,
                    invocation_liveness,
                )
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        if reusable_workspace_ceilings
            .keys()
            .any(|pool_id| !pools.iter().any(|pool| &pool.pool_id == pool_id))
        {
            return Err(invalid_plan(
                "reusable workspace ceiling references an unknown dynamic pool",
            ));
        }
        Ok(pools)
    }

    pub(super) fn derive_reusable_execution(
        policy: &ReusableExecutionPolicy,
        node_count: usize,
        dynamic_descriptors: &[DynamicResourceDescriptor],
        base_pools: &[DynamicBackingPoolSpec],
    ) -> Result<ReusableExecutionMemoryPlan, VNextError> {
        policy.validate()?;
        let node_count = u64::try_from(node_count)
            .ok()
            .filter(|count| *count > 0)
            .ok_or_else(|| invalid_plan("reusable execution requires at least one plan node"))?;
        let bucket_count = u64::try_from(policy.buckets().len())
            .map_err(|_| invalid_plan("reusable execution bucket count exceeds u64"))?;
        let maximum_device_executables = node_count
            .checked_mul(bucket_count)
            .and_then(|count| count.checked_mul(u64::from(policy.maximum_reusable_lanes())))
            .ok_or_else(|| invalid_plan("reusable device executable count overflows u64"))?;
        let descriptors = dynamic_descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), descriptor))
            .collect::<BTreeMap<_, _>>();
        let mut buckets = Vec::with_capacity(policy.buckets().len());
        for bucket in policy.buckets() {
            let capacity = bucket.capacity();
            let shape = DynamicResourceShape::from_validated(
                capacity.maximum_sequences(),
                capacity.maximum_tokens(),
                capacity.maximum_pages(),
            );
            let mut pool_budgets = Vec::new();
            for pool in base_pools {
                let step_bytes = Self::reusable_step_bytes_for_shape(pool, &descriptors, shape)?;
                let invocation_bytes =
                    Self::reusable_invocation_bytes_for_shape(pool, &descriptors, shape)?;
                if step_bytes != 0 || invocation_bytes != 0 {
                    pool_budgets.push(ReusablePoolWorkspaceBudget::new(
                        pool.pool_id.clone(),
                        step_bytes,
                        invocation_bytes,
                    )?);
                }
            }
            buckets.push(ResolvedReusableExecutionBucket::new(
                bucket.clone(),
                pool_budgets,
            )?);
        }
        ReusableExecutionMemoryPlan::new(
            policy.maximum_reusable_lanes(),
            maximum_device_executables,
            buckets,
        )
    }

    fn reusable_step_bytes_for_shape(
        pool: &DynamicBackingPoolSpec,
        descriptors: &BTreeMap<ResourceId, &DynamicResourceDescriptor>,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        pool.step_resource_slots
            .iter()
            .try_fold(0_u64, |total, slot| {
                let slot_bytes =
                    slot.resource_ids
                        .iter()
                        .try_fold(0_u64, |maximum, resource_id| {
                            let descriptor = descriptors.get(resource_id).ok_or_else(|| {
                                invalid_plan(
                                    "reusable Step slot references a missing dynamic descriptor",
                                )
                            })?;
                            Ok::<u64, VNextError>(
                                maximum.max(descriptor.evaluate_request_bytes_for_shape(shape)?),
                            )
                        })?;
                total
                    .checked_add(slot_bytes)
                    .ok_or_else(|| invalid_plan("reusable Step workspace budget overflows u64"))
            })
    }

    fn reusable_invocation_bytes_for_shape(
        pool: &DynamicBackingPoolSpec,
        descriptors: &BTreeMap<ResourceId, &DynamicResourceDescriptor>,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        let row_bytes = |row: &InvocationResourceLiveness| {
            row.resource_ids
                .iter()
                .try_fold(0_u64, |total, resource_id| {
                    total
                        .checked_add(
                            descriptors
                                .get(resource_id)
                                .ok_or_else(|| {
                                    invalid_plan(
                                        "reusable invocation row references a missing descriptor",
                                    )
                                })?
                                .evaluate_request_bytes_for_shape(shape)?,
                        )
                        .ok_or_else(|| invalid_plan("reusable invocation row budget overflows u64"))
                })
        };
        match pool.invocation_liveness_mode {
            InvocationLivenessMode::NoInvocationResources => Ok(0),
            InvocationLivenessMode::TotalOrderReuse => pool
                .invocation_liveness
                .iter()
                .try_fold(0_u64, |maximum, row| {
                    Ok::<u64, VNextError>(maximum.max(row_bytes(row)?))
                }),
            InvocationLivenessMode::ConservativeConcurrent => pool
                .invocation_liveness
                .iter()
                .try_fold(0_u64, |total, row| {
                    total.checked_add(row_bytes(row)?).ok_or_else(|| {
                        invalid_plan("reusable concurrent invocation budget overflows u64")
                    })
                }),
        }
    }

    pub(super) fn derive_pool_step_slots(
        nodes: &[PlanNode],
        descriptors: &[&DynamicResourceDescriptor],
        retained_completion_resources: &BTreeSet<ResourceId>,
    ) -> Result<Vec<StepResourceSlot>, VNextError> {
        struct Interval<'a> {
            resource_id: &'a ResourceId,
            first_user: usize,
            last_user: usize,
            reusable: bool,
        }

        let nodes_by_id = nodes
            .iter()
            .map(|node| (node.id.clone(), node))
            .collect::<BTreeMap<_, _>>();
        let mut intervals = descriptors
            .iter()
            .filter(|descriptor| descriptor.lifetime == AllocationLifetime::Step)
            .map(|descriptor| {
                let users = nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(index, node)| {
                        node.resources
                            .contains(&descriptor.base_resource_id)
                            .then_some(index)
                    })
                    .collect::<Vec<_>>();
                let first_user = users.first().copied().ok_or_else(|| {
                    invalid_plan(format!(
                        "step resource `{}` is not referenced by a plan node",
                        descriptor.base_resource_id
                    ))
                })?;
                let last_user = users.last().copied().ok_or_else(|| {
                    invalid_plan(format!(
                        "step resource `{}` is not referenced by a plan node",
                        descriptor.base_resource_id
                    ))
                })?;
                Ok(Interval {
                    resource_id: &descriptor.base_resource_id,
                    first_user,
                    last_user,
                    // Completion diagnostics read after the terminal fence, so
                    // a retained activation cannot share a liveness slot.
                    reusable: Self::is_reusable_step_activation(descriptor)
                        && !retained_completion_resources.contains(&descriptor.base_resource_id),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        intervals.sort_by(|left, right| {
            (left.first_user, left.last_user, left.resource_id).cmp(&(
                right.first_user,
                right.last_user,
                right.resource_id,
            ))
        });

        let ordered_without_overlap = |left: &Interval<'_>, right: &Interval<'_>| {
            let left_before = left.last_user < right.first_user
                && node_completion_precedes(
                    &nodes_by_id,
                    &nodes[left.last_user].id,
                    &nodes[right.first_user].id,
                )?;
            let right_before = right.last_user < left.first_user
                && node_completion_precedes(
                    &nodes_by_id,
                    &nodes[right.last_user].id,
                    &nodes[left.first_user].id,
                )?;
            Ok::<bool, VNextError>(left_before || right_before)
        };

        let mut slots = Vec::<Vec<Interval<'_>>>::new();
        for interval in intervals {
            if !interval.reusable {
                slots.push(vec![interval]);
                continue;
            }
            let mut reusable_slot = None;
            for (index, slot) in slots.iter().enumerate() {
                let mut reusable = slot.iter().all(|member| member.reusable);
                if reusable {
                    for member in slot {
                        reusable = ordered_without_overlap(member, &interval)?;
                        if !reusable {
                            break;
                        }
                    }
                }
                if reusable {
                    reusable_slot = Some(index);
                    break;
                }
            }
            if let Some(index) = reusable_slot {
                slots[index].push(interval);
            } else {
                slots.push(vec![interval]);
            }
        }
        let mut slots = slots
            .into_iter()
            .map(|slot| {
                let resource_ids = slot
                    .into_iter()
                    .map(|interval| interval.resource_id.clone())
                    .collect::<Vec<_>>();
                if resource_ids.len() == 1 {
                    Ok(StepResourceSlot::dedicated(
                        resource_ids
                            .into_iter()
                            .next()
                            .expect("single resource slot"),
                    ))
                } else {
                    StepResourceSlot::ordered_single_fence_wave(resource_ids)
                }
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        slots.sort_by(|left, right| left.resource_ids.cmp(&right.resource_ids));
        Ok(slots)
    }

    fn is_reusable_step_activation(descriptor: &DynamicResourceDescriptor) -> bool {
        descriptor.lifetime == AllocationLifetime::Step
            && descriptor.usage == BufferUsage::Activations
            && matches!(descriptor.kind, AllocationKind::Value)
            && matches!(descriptor.demand, DynamicResourceDemand::Tokens { .. })
    }

    fn step_slot_bytes(
        slots: &[StepResourceSlot],
        descriptors: &[&DynamicResourceDescriptor],
        theoretical: bool,
        overflow_context: &str,
    ) -> Result<u64, VNextError> {
        let descriptors = descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), *descriptor))
            .collect::<BTreeMap<_, _>>();
        slots.iter().try_fold(0_u64, |total, slot| {
            let slot_bytes = slot
                .resource_ids
                .iter()
                .try_fold(0_u64, |maximum, resource_id| {
                    let descriptor = descriptors.get(resource_id).ok_or_else(|| {
                        invalid_plan("step slot references a missing dynamic descriptor")
                    })?;
                    let bytes = if theoretical {
                        descriptor.theoretical_maximum_request_bytes()?
                    } else {
                        descriptor.minimum_request_bytes()?
                    };
                    Ok::<u64, VNextError>(maximum.max(bytes))
                })?;
            total
                .checked_add(slot_bytes)
                .ok_or_else(|| invalid_plan(format!("{overflow_context} overflows u64")))
        })
    }

    pub(super) fn derive_pool_invocation_liveness(
        nodes: &[PlanNode],
        descriptors: &[&DynamicResourceDescriptor],
    ) -> Result<(InvocationLivenessMode, Vec<InvocationResourceLiveness>, u64), VNextError> {
        let invocation_ids = descriptors
            .iter()
            .filter(|descriptor| descriptor.lifetime == AllocationLifetime::Invocation)
            .map(|descriptor| descriptor.base_resource_id.clone())
            .collect::<BTreeSet<_>>();
        if invocation_ids.is_empty() {
            return Ok((InvocationLivenessMode::NoInvocationResources, Vec::new(), 0));
        }
        let descriptor_by_id = descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), *descriptor))
            .collect::<BTreeMap<_, _>>();
        let mut covered = BTreeSet::new();
        let liveness_in_execution_order = nodes
            .iter()
            .filter_map(|node| {
                let resource_ids = node
                    .resources
                    .iter()
                    .filter(|resource_id| invocation_ids.contains(*resource_id))
                    .cloned()
                    .collect::<Vec<_>>();
                covered.extend(resource_ids.iter().cloned());
                (!resource_ids.is_empty()).then(|| InvocationResourceLiveness {
                    node_id: node.id.clone(),
                    resource_ids,
                })
            })
            .collect::<Vec<_>>();
        if covered != invocation_ids {
            return Err(invalid_plan(
                "pool node liveness does not cover every invocation resource",
            ));
        }
        let nodes_by_id = nodes
            .iter()
            .map(|node| (node.id.clone(), node))
            .collect::<BTreeMap<_, _>>();
        let contains_binding = descriptors.iter().any(|descriptor| {
            descriptor.lifetime == AllocationLifetime::Invocation
                && matches!(descriptor.kind, AllocationKind::Binding { .. })
        });
        let total_ordered = !contains_binding
            && liveness_in_execution_order
                .windows(2)
                .try_fold(true, |ordered, pair| {
                    Ok::<bool, VNextError>(
                        ordered
                            && node_completion_precedes(
                                &nodes_by_id,
                                &pair[0].node_id,
                                &pair[1].node_id,
                            )?,
                    )
                })?;
        let row_bytes = |row: &InvocationResourceLiveness| {
            row.resource_ids
                .iter()
                .try_fold(0_u64, |total, resource_id| {
                    total
                        .checked_add(
                            descriptor_by_id
                                .get(resource_id)
                                .ok_or_else(|| {
                                    invalid_plan("pool liveness references an unknown descriptor")
                                })?
                                .minimum_request_bytes()?,
                        )
                        .ok_or_else(|| invalid_plan("pool invocation row bytes overflow u64"))
                })
        };
        let invocation_peak = if total_ordered {
            liveness_in_execution_order
                .iter()
                .try_fold(0_u64, |peak, row| {
                    Ok::<u64, VNextError>(peak.max(row_bytes(row)?))
                })?
        } else {
            liveness_in_execution_order
                .iter()
                .try_fold(0_u64, |total, row| {
                    total.checked_add(row_bytes(row)?).ok_or_else(|| {
                        invalid_plan("pool concurrent invocation bytes overflow u64")
                    })
                })?
        };
        let mut invocation_liveness = liveness_in_execution_order;
        invocation_liveness.sort_by(|left, right| left.node_id.cmp(&right.node_id));
        Ok((
            if total_ordered {
                InvocationLivenessMode::TotalOrderReuse
            } else {
                InvocationLivenessMode::ConservativeConcurrent
            },
            invocation_liveness,
            invocation_peak,
        ))
    }

    pub(super) fn validate_dynamic_pools_structure(
        pools: &[DynamicBackingPoolSpec],
        descriptors: &BTreeMap<ResourceId, &DynamicResourceDescriptor>,
        dynamic_capacity_bytes: u64,
        reusable_workspace_ceilings: &BTreeMap<DynamicBackingPoolId, u64>,
    ) -> Result<PoolAggregateEvidence, VNextError> {
        let mut expected_members =
            BTreeMap::<DynamicBackingPoolId, (PoolCompatibilityKey, Vec<ResourceId>)>::new();
        for descriptor in descriptors.values() {
            let compatibility = PoolCompatibilityKey::new(
                &descriptor.storage,
                descriptor.usage,
                descriptor.element_type,
                descriptor.alignment_bytes,
            )?;
            let expected_id = DynamicBackingPoolId::from_compatibility(&compatibility)?;
            if descriptor.pool_id != expected_id {
                return Err(invalid_plan(
                    "dynamic descriptor has a non-derived pool identity",
                ));
            }
            match expected_members.entry(expected_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((compatibility, vec![descriptor.base_resource_id.clone()]));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if entry.get().0 != compatibility {
                        return Err(invalid_plan(
                            "dynamic pool hash collision has incompatible descriptors",
                        ));
                    }
                    entry.get_mut().1.push(descriptor.base_resource_id.clone());
                }
            }
        }
        if pools.len() != expected_members.len() {
            return Err(invalid_plan(
                "dynamic backing pool count is not derived from descriptors",
            ));
        }
        if reusable_workspace_ceilings
            .keys()
            .any(|pool_id| !expected_members.contains_key(pool_id))
        {
            return Err(invalid_plan(
                "reusable workspace ceiling references an unknown dynamic pool",
            ));
        }
        let mut aggregate = PoolAggregateEvidence::default();
        for pool in pools {
            pool.validate_local()?;
            let (compatibility, resource_ids) = expected_members
                .remove(&pool.pool_id)
                .ok_or_else(|| invalid_plan("dynamic backing pool has no descriptor members"))?;
            if pool.compatibility != compatibility || pool.resource_ids != resource_ids {
                return Err(invalid_plan(
                    "dynamic backing pool compatibility or membership is not core-derived",
                ));
            }
            let members =
                pool.resource_ids
                    .iter()
                    .map(|resource_id| {
                        descriptors.get(resource_id).copied().ok_or_else(|| {
                            invalid_plan("dynamic pool member descriptor is missing")
                        })
                    })
                    .collect::<Result<Vec<_>, VNextError>>()?;
            let request = minimum_for_lifetime(
                &members,
                AllocationLifetime::Request,
                "pool request minimum",
            )?;
            let sequence = minimum_for_lifetime(
                &members,
                AllocationLifetime::Sequence,
                "pool sequence minimum",
            )?;
            let expected_step_resources = members
                .iter()
                .filter(|descriptor| descriptor.lifetime == AllocationLifetime::Step)
                .map(|descriptor| descriptor.base_resource_id.clone())
                .collect::<BTreeSet<_>>();
            let actual_step_resources = pool
                .step_resource_slots
                .iter()
                .flat_map(|slot| slot.resource_ids.iter().cloned())
                .collect::<BTreeSet<_>>();
            if actual_step_resources != expected_step_resources {
                return Err(invalid_plan(
                    "step resource slots do not cover exactly the pool's Step descriptors",
                ));
            }
            for slot in &pool.step_resource_slots {
                slot.validate()?;
                if slot.kind == StepResourceSlotKind::OrderedSingleFenceStepWave
                    && slot.resource_ids.iter().any(|resource_id| {
                        descriptors
                            .get(resource_id)
                            .is_none_or(|descriptor| !Self::is_reusable_step_activation(descriptor))
                    })
                {
                    return Err(invalid_plan(
                        "shared Step slot contains a non-transient activation resource",
                    ));
                }
            }
            let step = Self::step_slot_bytes(
                &pool.step_resource_slots,
                &members,
                false,
                "pool step minimum",
            )?;
            let theoretical = members.iter().try_fold(0_u128, |total, descriptor| {
                total
                    .checked_add(
                        u128::from(descriptor.theoretical_maximum_request_bytes()?)
                            * u128::from(descriptor.theoretical_maximum_instances),
                    )
                    .ok_or_else(|| invalid_plan("pool theoretical ceiling overflows u128"))
            })?;
            let invocation_ids = members
                .iter()
                .filter(|descriptor| descriptor.lifetime == AllocationLifetime::Invocation)
                .map(|descriptor| descriptor.base_resource_id.clone())
                .collect::<BTreeSet<_>>();
            let invocation_peak = validate_pool_liveness_rows(
                pool.invocation_liveness_mode,
                &pool.invocation_liveness,
                &invocation_ids,
                descriptors,
            )?;
            let reusable_workspace_ceiling_bytes = reusable_workspace_ceilings
                .get(&pool.pool_id)
                .copied()
                .unwrap_or(0);
            let combined_ceiling = theoretical
                .checked_add(u128::from(reusable_workspace_ceiling_bytes))
                .ok_or_else(|| invalid_plan("pool combined ceiling overflows u128"))?;
            let maximum_resident =
                u64::try_from(combined_ceiling.min(u128::from(dynamic_capacity_bytes)))
                    .map_err(|_| invalid_plan("pool resident ceiling exceeds u64"))?;
            if pool.minimum_request_bytes != request
                || pool.minimum_sequence_bytes != sequence
                || pool.minimum_step_bytes != step
                || pool.minimum_invocation_peak_bytes != invocation_peak
                || pool.theoretical_ceiling_bytes.get() != theoretical
                || pool.reusable_workspace_ceiling_bytes != reusable_workspace_ceiling_bytes
                || pool.provisioning.maximum_resident_bytes != maximum_resident
            {
                return Err(invalid_plan(
                    "dynamic backing pool bounds or liveness are not core-derived",
                ));
            }
            aggregate.add(pool)?;
        }
        if !expected_members.is_empty() {
            return Err(invalid_plan(
                "dynamic descriptors are missing canonical backing pools",
            ));
        }
        Ok(aggregate)
    }

    pub(super) fn summarize_pool_invocation_liveness(
        pools: &[DynamicBackingPoolSpec],
    ) -> Result<(InvocationLivenessMode, Vec<InvocationResourceLiveness>), VNextError> {
        let mut by_node = BTreeMap::<NodeId, BTreeSet<ResourceId>>::new();
        let mut has_invocation = false;
        let mut all_total_ordered = true;
        for pool in pools {
            match pool.invocation_liveness_mode {
                InvocationLivenessMode::NoInvocationResources => {}
                InvocationLivenessMode::TotalOrderReuse => has_invocation = true,
                InvocationLivenessMode::ConservativeConcurrent => {
                    has_invocation = true;
                    all_total_ordered = false;
                }
            }
            for row in &pool.invocation_liveness {
                by_node
                    .entry(row.node_id.clone())
                    .or_default()
                    .extend(row.resource_ids.iter().cloned());
            }
        }
        let liveness = by_node
            .into_iter()
            .map(|(node_id, resource_ids)| InvocationResourceLiveness {
                node_id,
                resource_ids: resource_ids.into_iter().collect(),
            })
            .collect::<Vec<_>>();
        let reference_count = liveness.iter().try_fold(0_usize, |total, row| {
            total.checked_add(row.resource_ids.len())
        });
        if reference_count.is_none_or(|count| count > MAX_EXECUTION_PLAN_RESOURCE_ROWS) {
            return Err(invalid_plan(
                "invocation liveness reference count is invalid",
            ));
        }
        Ok((
            if !has_invocation {
                InvocationLivenessMode::NoInvocationResources
            } else if all_total_ordered {
                InvocationLivenessMode::TotalOrderReuse
            } else {
                InvocationLivenessMode::ConservativeConcurrent
            },
            liveness,
        ))
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    pub const fn policy_capacity_bytes(&self) -> u64 {
        self.policy_capacity_bytes
    }

    pub const fn reserve_bytes(&self) -> u64 {
        self.reserve_bytes
    }

    pub const fn usable_capacity_bytes(&self) -> u64 {
        self.usable_capacity_bytes
    }

    pub const fn maximum_active_sequences(&self) -> u32 {
        self.maximum_active_sequences
    }

    pub const fn capacity_bytes(&self) -> u64 {
        self.usable_capacity_bytes
    }

    pub const fn static_bytes(&self) -> u64 {
        self.static_bytes
    }

    pub const fn minimum_request_bytes(&self) -> u64 {
        self.minimum_request_bytes
    }

    pub const fn minimum_sequence_bytes(&self) -> u64 {
        self.minimum_sequence_bytes
    }

    pub const fn minimum_step_bytes(&self) -> u64 {
        self.minimum_step_bytes
    }

    pub const fn minimum_invocation_peak_bytes(&self) -> u64 {
        self.minimum_invocation_peak_bytes
    }

    pub const fn minimum_runnable_request_bytes(&self) -> u64 {
        self.minimum_runnable_request_bytes
    }

    pub const fn invocation_liveness_mode(&self) -> InvocationLivenessMode {
        self.invocation_liveness_mode
    }

    pub fn invocation_liveness(&self) -> &[InvocationResourceLiveness] {
        &self.invocation_liveness
    }

    /// Conservative checked evidence across static allocations, live provider
    /// formula maxima, reusable execution workspace, and the protocol ceiling.
    /// It is never itself a reservation, admission target, or performance claim.
    pub fn theoretical_ceiling_bytes(&self) -> u128 {
        self.theoretical_ceiling_bytes.get()
    }

    pub fn static_allocations(&self) -> &[ResourceAllocation] {
        &self.static_allocations
    }

    pub fn dynamic_descriptors(&self) -> &[DynamicResourceDescriptor] {
        &self.dynamic_descriptors
    }

    pub fn dynamic_pools(&self) -> &[DynamicBackingPoolSpec] {
        &self.dynamic_pools
    }

    pub fn reusable_execution(&self) -> Option<&ReusableExecutionMemoryPlan> {
        self.reusable_execution.as_ref()
    }

    pub fn static_buffer_requests(&self) -> Result<Vec<BufferRequest>, VNextError> {
        self.static_allocations
            .iter()
            .map(ResourceAllocation::buffer_request)
            .collect()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MemoryPlanWire {
    pub(super) device_capacity_bytes: u64,
    pub(super) policy_capacity_bytes: u64,
    pub(super) reserve_bytes: u64,
    pub(super) usable_capacity_bytes: u64,
    pub(super) maximum_active_sequences: u32,
    pub(super) static_bytes: u64,
    pub(super) minimum_request_bytes: u64,
    pub(super) minimum_sequence_bytes: u64,
    pub(super) minimum_step_bytes: u64,
    pub(super) minimum_invocation_peak_bytes: u64,
    pub(super) minimum_runnable_request_bytes: u64,
    pub(super) theoretical_ceiling_bytes: CanonicalU128,
    pub(super) static_allocations: Vec<ResourceAllocation>,
    pub(super) dynamic_descriptors: Vec<DynamicResourceDescriptor>,
    pub(super) dynamic_pools: Vec<DynamicBackingPoolSpec>,
    pub(super) reusable_execution: Option<ReusableExecutionMemoryPlan>,
    pub(super) invocation_liveness_mode: InvocationLivenessMode,
    pub(super) invocation_liveness: Vec<InvocationResourceLiveness>,
}

impl<'de> Deserialize<'de> for MemoryPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MemoryPlanWire::deserialize(deserializer)?;
        let plan = Self {
            device_capacity_bytes: wire.device_capacity_bytes,
            policy_capacity_bytes: wire.policy_capacity_bytes,
            reserve_bytes: wire.reserve_bytes,
            usable_capacity_bytes: wire.usable_capacity_bytes,
            maximum_active_sequences: wire.maximum_active_sequences,
            static_bytes: wire.static_bytes,
            minimum_request_bytes: wire.minimum_request_bytes,
            minimum_sequence_bytes: wire.minimum_sequence_bytes,
            minimum_step_bytes: wire.minimum_step_bytes,
            minimum_invocation_peak_bytes: wire.minimum_invocation_peak_bytes,
            minimum_runnable_request_bytes: wire.minimum_runnable_request_bytes,
            theoretical_ceiling_bytes: wire.theoretical_ceiling_bytes,
            static_allocations: wire.static_allocations,
            dynamic_descriptors: wire.dynamic_descriptors,
            dynamic_pools: wire.dynamic_pools,
            reusable_execution: wire.reusable_execution,
            invocation_liveness_mode: wire.invocation_liveness_mode,
            invocation_liveness: wire.invocation_liveness,
        };
        plan.validate().map_err(serde::de::Error::custom)?;
        Ok(plan)
    }
}
