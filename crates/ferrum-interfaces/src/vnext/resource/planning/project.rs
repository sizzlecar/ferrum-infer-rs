use super::*;

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    /// Replays one whole eager wave against a private numeric state. Persistent
    /// sequence growth survives into the next state; Step and Invocation claims
    /// overlap until completion, then only those transient claims are released.
    /// No new live allocation or release is performed, including on failure.
    pub fn project_resource_wave(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        state: &ResourcePlanningState,
        rows: &[ResourcePlanningRow],
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningProjection> {
        if self.dynamic_pools.reusable_execution.is_some() {
            return ResourcePlanningAvailability::Unknown(
                ResourcePlanningUnknown::ReusableExecution,
            );
        }
        self.project_resource_wave_with_bucket(view, state, rows, None, budget)
    }

    /// Explicit bucket selection is only a numerical query, not permission to
    /// submit. The execution owner must select it from the actual phase policy
    /// and revalidate that selection at the final physical boundary.
    pub fn project_resource_wave_with_bucket(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        state: &ResourcePlanningState,
        rows: &[ResourcePlanningRow],
        bucket: Option<&ReusableExecutionBucketId>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningProjection> {
        match self.project_resource_wave_inner(view, state, rows, bucket, budget) {
            Ok(value) => ResourcePlanningAvailability::Known(value),
            Err(reason) => ResourcePlanningAvailability::Unknown(reason),
        }
    }

    fn project_resource_wave_inner(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        state: &ResourcePlanningState,
        rows: &[ResourcePlanningRow],
        bucket_id: Option<&ReusableExecutionBucketId>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<ResourcePlanningProjection, ResourcePlanningUnknown> {
        use ResourcePlanningUnknown as U;
        poll(budget)?;
        if !Arc::ptr_eq(&view.fence, &state.fence)
            || view.logical.coordinator_id() != self.dynamic_pools.logical_admission.id()
        {
            return Err(U::StaleIdentity);
        }
        if rows.is_empty() || rows.len() > view.limits.maximum_participants {
            return Err(U::InvalidInput);
        }
        if state.waves >= view.limits.maximum_projected_waves {
            return Err(U::LimitExceeded);
        }
        let mut next = state.clone();
        let binding = TrustedPlanRuntimeBinding {
            resources: Arc::clone(self),
        };
        let mut domains: BTreeMap<CapacityDomainId, ResourcePlanningDomainDemand> = BTreeMap::new();
        let mut seen = BTreeSet::new();
        let mut immediate_tokens = 0_u64;
        // VNext extends participant backing in ExecutionBatchParticipants'
        // canonical authority order, independently of product input order.
        // Preserve public indices while replaying that exact allocator order.
        let mut ordered = Vec::with_capacity(rows.len());
        for row in rows {
            poll(budget)?;
            let participant = view
                .participants
                .get(row.participant_index)
                .ok_or(U::InvalidInput)?;
            if row.token_count == 0 || !seen.insert(row.participant_index) {
                return Err(U::InvalidInput);
            }
            ordered.push((participant.authority, row));
        }
        ordered.sort_unstable_by_key(|(authority, _)| *authority);
        for (_, row) in ordered {
            poll(budget)?;
            let participant = &view.participants[row.participant_index];
            let end = row
                .start_token
                .checked_add(row.token_count)
                .ok_or(U::InvalidInput)?;
            if end > participant.maximum_tokens {
                return Err(U::InvalidInput);
            }
            immediate_tokens = immediate_tokens
                .checked_add(row.token_count)
                .ok_or(U::InvalidInput)?;
            let committed = next.covered[row.participant_index];
            if end > committed.tokens() {
                let target = DynamicResourceShape::from_validated(1, end, 0);
                let (demand, requests) = binding
                    .sequence_extension_demand(
                        committed,
                        target,
                        AdmissionPressureAction::WaitForRelease,
                    )
                    .map_err(|_| U::InvalidDemand)?;
                charge_logical(&mut next, &demand, &mut domains, true)?;
                let allocated = reserve(&mut next, &requests, view.limits, budget)?;
                if view.physical_ranges.is_some() {
                    sequence_ranges::extend(
                        &mut next.sequence_ranges[row.participant_index],
                        &requests,
                        &allocated,
                        view.limits.maximum_free_extents,
                        budget,
                    )?;
                    sequence_ranges::check_bound(
                        &next.sequence_ranges,
                        view.limits.maximum_free_extents,
                    )?;
                }
                next.covered[row.participant_index] = target;
            }
        }
        // Keep both transient lifetimes allocated at once. Their independent
        // availability checks must not double spend a shared physical pool.
        let shape = DynamicResourceShape::from_validated(
            u32::try_from(rows.len()).map_err(|_| U::InvalidInput)?,
            immediate_tokens,
            0,
        );
        let bucket = match bucket_id {
            Some(id) => {
                let plan = self
                    .dynamic_pools
                    .reusable_execution
                    .as_ref()
                    .ok_or(U::InvalidInput)?;
                if plan.buckets().len() > view.limits.maximum_descriptors {
                    return Err(U::LimitExceeded);
                }
                let mut selected = None;
                for candidate in plan.buckets() {
                    poll(budget)?;
                    if candidate.bucket().bucket_id() == id {
                        selected = Some(candidate.bucket());
                        break;
                    }
                }
                Some(selected.ok_or(U::InvalidInput)?)
            }
            None => None,
        };
        if bucket.is_some() && next.workspace.is_none() {
            return Err(U::ReusableExecution);
        }
        if bucket.is_some_and(|bucket| {
            !bucket
                .capacity()
                .covers(shape.sequences(), shape.tokens(), shape.pages())
        }) {
            return Err(U::InvalidInput);
        }
        let (step_demand, step_requests) = binding
            .scoped_demand(
                AllocationLifetime::Step,
                None,
                shape,
                shape,
                bucket,
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .map_err(|_| U::InvalidDemand)?;
        poll(budget)?;
        charge_logical(&mut next, &step_demand, &mut domains, false)?;
        let mut step_slot = None;
        let mut invocation_slot = None;
        let mut physical_proof = view
            .physical_ranges
            .as_ref()
            .map(|physical| physical.proof());
        let mut transient = if bucket.is_some() {
            step_slot = workspace::reserve(
                &mut next,
                &step_requests,
                AllocationLifetime::Step,
                view.limits,
                budget,
            )?;
            Vec::new()
        } else {
            reserve(&mut next, &step_requests, view.limits, budget)?
        };
        if let (Some(physical), Some(proof)) = (&view.physical_ranges, physical_proof.as_mut()) {
            if let Some(slot) = &step_slot {
                workspace::project_ranges(&next, slot, physical, proof, budget)?;
            } else {
                physical.insert_transient(proof, &step_requests, &transient, budget)?;
            }
        }
        let (wave_demand, wave_requests) = binding
            .submission_wave_demand(
                shape,
                shape,
                bucket,
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .map_err(|_| U::InvalidDemand)?;
        charge_logical(&mut next, &wave_demand, &mut domains, false)?;
        if bucket.is_some() {
            invocation_slot = workspace::reserve(
                &mut next,
                &wave_requests,
                AllocationLifetime::Invocation,
                view.limits,
                budget,
            )?;
            if let (Some(physical), Some(proof), Some(slot)) = (
                &view.physical_ranges,
                physical_proof.as_mut(),
                invocation_slot.as_ref(),
            ) {
                workspace::project_ranges(&next, slot, physical, proof, budget)?;
            }
        } else {
            let wave_allocations = reserve(&mut next, &wave_requests, view.limits, budget)?;
            if let (Some(physical), Some(proof)) = (&view.physical_ranges, physical_proof.as_mut())
            {
                physical.insert_transient(proof, &wave_requests, &wave_allocations, budget)?;
            }
            transient.extend(wave_allocations);
        }
        for (pool_index, segments) in transient.into_iter().rev() {
            poll(budget)?;
            for segment in segments.iter().rev() {
                next.pools[pool_index]
                    .allocator
                    .release(segment)
                    .map_err(|_| U::InvalidDemand)?;
            }
        }
        for demand in domains.values() {
            let available = next
                .logical_available
                .get_mut(&demand.domain)
                .ok_or(U::InvalidDemand)?;
            *available = available
                .checked_add(demand.transient_peak_bytes)
                .ok_or(U::InvalidDemand)?;
        }
        check_extent_bound(&next, view.limits)?;
        if let (Some(physical), Some(proof)) = (&view.physical_ranges, physical_proof.as_mut()) {
            physical.insert_sequence_ranges(proof, &next.sequence_ranges, rows, budget)?;
        }
        poll(budget)?;
        next.waves += 1;
        Ok(ResourcePlanningProjection {
            state: next,
            domains: domains.into_values().collect(),
            step_slot,
            invocation_slot,
            physical_ranges: physical_proof,
        })
    }
}

fn charge_logical(
    state: &mut ResourcePlanningState,
    demand: &AdmissionDemand,
    evidence: &mut BTreeMap<CapacityDomainId, ResourcePlanningDomainDemand>,
    persistent: bool,
) -> Result<(), ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    for entry in demand.immediate_claim().entries() {
        let available = state
            .logical_available
            .get_mut(&entry.domain())
            .ok_or(U::InvalidDemand)?;
        *available = available
            .checked_sub(entry.units().get())
            .ok_or(U::LogicalCapacity)?;
        let evidence = evidence
            .entry(entry.domain())
            .or_insert(ResourcePlanningDomainDemand {
                domain: entry.domain(),
                persistent_bytes: 0,
                transient_peak_bytes: 0,
            });
        let total = if persistent {
            &mut evidence.persistent_bytes
        } else {
            &mut evidence.transient_peak_bytes
        };
        *total = total
            .checked_add(entry.units().get())
            .ok_or(U::InvalidDemand)?;
    }
    Ok(())
}

fn check_extent_bound(
    state: &ResourcePlanningState,
    limits: ResourcePlanningLimits,
) -> Result<(), ResourcePlanningUnknown> {
    let count = state.pools.iter().try_fold(0_usize, |sum, pool| {
        sum.checked_add(pool.allocator.by_offset.len())
    });
    if count.is_none_or(|count| count > limits.maximum_free_extents) {
        Err(ResourcePlanningUnknown::LimitExceeded)
    } else {
        Ok(())
    }
}

/// Uses the same pool order, descending claim size and identity tie-break as
/// DynamicPoolSet::prepare_claim_scoped, and its exact free-extent allocator.
pub(super) fn reserve(
    state: &mut ResourcePlanningState,
    requests: &[EvaluatedBackingRequest<'_>],
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Vec<(usize, Vec<BackingSegment>)>, ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    if requests.len() > limits.maximum_descriptors {
        return Err(U::LimitExceeded);
    }
    let mut ordered: Vec<_> = requests.iter().collect();
    ordered.sort_by(|left, right| {
        left.domain
            .pool_id()
            .cmp(right.domain.pool_id())
            .then_with(|| right.capacity_size_bytes.cmp(&left.capacity_size_bytes))
            .then_with(|| left.claim_identity.cmp(&right.claim_identity))
    });
    let mut allocations = Vec::with_capacity(ordered.len());
    let mut segment_count = 0_usize;
    for request in ordered {
        poll(budget)?;
        let index = state
            .pools
            .binary_search_by(|pool| pool.id.cmp(request.domain.pool_id()))
            .map_err(|_| U::InvalidDemand)?;
        let pool = &mut state.pools[index];
        let segments = match request.domain.pool.compatibility().profile().view() {
            DynamicStorageView::Contiguous => pool
                .allocator
                .allocate_contiguous(&pool.id, request.capacity_size_bytes)
                .map(|value| value.map(|segment| vec![segment])),
            DynamicStorageView::PagedRegions { block_bytes } => {
                pool.allocator
                    .allocate_paged(&pool.id, request.capacity_size_bytes, block_bytes)
            }
        }
        .map_err(|_| U::InvalidDemand)?
        .ok_or(U::PhysicalCapacity)?;
        segment_count = segment_count
            .checked_add(segments.len())
            .ok_or(U::LimitExceeded)?;
        if segment_count > limits.maximum_free_extents {
            return Err(U::LimitExceeded);
        }
        allocations.push((index, segments));
        check_extent_bound(state, limits)?;
    }
    Ok(allocations)
}
