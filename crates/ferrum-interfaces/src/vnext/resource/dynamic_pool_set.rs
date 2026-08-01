//! Dynamic pool-set orchestration over backing owned by `dynamic_pool`.

use super::{
    align_up_resource, backing_segment_range, bind_lane_stable_slot_projections,
    compile_program_binding_layouts, compile_submission_wave_domain_layout,
    compile_submission_wave_reusable_capacity_layouts, contiguous_packing_growth_bytes,
    free_extent_layout_fingerprint, invalid_resource, lane_stable_layout_key,
    rollback_free_extent_journal, validate_runtime_descriptor_for_admission,
    AllocatedDynamicGrowth, AllocationLifetime, AllocationSeal, Arc, AtomicU64, BTreeMap,
    BackingChunkIdentity, BackingClaimCertificate, BackingPrepareDecision, BackingSegment,
    BufferRequest, CapacityAvailabilityEpoch, CapacityDomainId, CapacityEntry, CapacityEpochs,
    CapacityUnits, CapacityVector, DeviceAllocationPermit, DeviceBufferRetention,
    DeviceCapacityBudget, DeviceCapacityReservation, DeviceRuntime, Digest, DynamicBackingBlocker,
    DynamicBackingClaimOccupancy, DynamicBackingClaimResidency, DynamicBackingClaimScope,
    DynamicBackingDeferralReason, DynamicBackingDeferred, DynamicBackingPool, DynamicBackingPoolId,
    DynamicBackingPoolState, DynamicChunkQuarantineReason, DynamicDeviceCapacityBlocked,
    DynamicPoolDomainSpec, DynamicPoolGrowthBatchReceipt, DynamicPoolGrowthIntent,
    DynamicPoolGrowthReceipt, DynamicPoolIdleReclaim, DynamicPoolLiveOccupancyStatus,
    DynamicPoolRebalanceReceipt, DynamicResourceShape, DynamicStorageView, EvaluatedBackingRequest,
    ExecutionLane, FreeExtentIndex, IdleChunkReclaimCandidate, InvocationLivenessMode,
    LaneBackingPrepareDecision, LaneStableArenaEntry, LaneStableArenaEvictionCandidate,
    LaneStableArenaLane, LaneStableArenaSlot, LaneStableArenaSlotLease, LaneStableArenaState,
    LogicalAdmissionCoordinator, LogicalBackingBufferView, LogicalBackingSegmentBinding,
    LogicalBackingSliceAllocationEvidence, LogicalBackingSliceAuthority,
    LogicalBackingSliceEvidence, Mutex, Ordering, PendingGrowthGuard, PlanNode,
    PlannedDynamicGrowth, PreparedBackingClaim, PreparedBackingExtent, PreparedLaneBackingClaim,
    ProgramBindingLayout, QuarantinedDynamicChunk, ResidentChunkBacking, ResidentChunkState,
    ResourceId, ResourceReservation, ResourceRetentionPolicy, ResourceTransactionIdentity, RunId,
    Sha256, StateInitialization, StaticProvisioningBinding, StepResourceSlotKind,
    SubmissionWaveDomainCapacityLayout, SubmissionWaveDomainLayout, TransactionId, VNextError,
    NEXT_DYNAMIC_POOL_INSTANCE_ID,
};
use crate::vnext::{
    DeviceCapacityPressure, DynamicPoolResidentPressure, ReusableExecutionBucketId,
    ReusableExecutionMemoryPlan,
};

pub(in crate::vnext::resource) struct DynamicPoolSet<R>
where
    R: DeviceRuntime,
{
    pub(in crate::vnext::resource) pools:
        BTreeMap<DynamicBackingPoolId, Arc<DynamicBackingPool<R>>>,
    pub(in crate::vnext::resource) domains: Vec<DynamicPoolDomainSpec>,
    pub(in crate::vnext::resource) nodes: Arc<[PlanNode]>,
    pub(in crate::vnext::resource) submission_wave_layouts: Vec<Option<SubmissionWaveDomainLayout>>,
    pub(in crate::vnext::resource) submission_wave_reusable_capacity_layouts:
        BTreeMap<ReusableExecutionBucketId, Vec<Option<SubmissionWaveDomainCapacityLayout>>>,
    pub(in crate::vnext::resource) program_binding_layouts:
        BTreeMap<ReusableExecutionBucketId, Arc<ProgramBindingLayout>>,
    pub(in crate::vnext::resource) reusable_execution: Option<ReusableExecutionMemoryPlan>,
    pub(in crate::vnext::resource) logical_admission: LogicalAdmissionCoordinator,
    pub(in crate::vnext::resource) budget: Arc<DeviceCapacityBudget>,
    lane_stable_arenas: Arc<Mutex<LaneStableArenaState>>,
    binding: StaticProvisioningBinding,
    // Backend context must outlive every resident/quarantined buffer above.
    runtime: Arc<R>,
}

impl<R> DynamicPoolSet<R>
where
    R: DeviceRuntime,
{
    pub(in crate::vnext::resource) fn new(
        runtime: Arc<R>,
        binding: StaticProvisioningBinding,
        budget: Arc<DeviceCapacityBudget>,
        logical_admission: LogicalAdmissionCoordinator,
        domains: Vec<DynamicPoolDomainSpec>,
        nodes: Arc<[PlanNode]>,
        reusable_execution: Option<ReusableExecutionMemoryPlan>,
    ) -> Result<Self, VNextError> {
        let submission_wave_layouts = domains
            .iter()
            .map(|domain| compile_submission_wave_domain_layout(domain, &nodes))
            .collect::<Result<Vec<_>, _>>()?;
        let submission_wave_reusable_capacity_layouts =
            compile_submission_wave_reusable_capacity_layouts(
                &domains,
                &submission_wave_layouts,
                reusable_execution.as_ref(),
            )?;
        let program_binding_layouts = compile_program_binding_layouts(
            &domains,
            &nodes,
            &submission_wave_layouts,
            &submission_wave_reusable_capacity_layouts,
        )?
        .into_iter()
        .map(|(bucket_id, layout)| (bucket_id, Arc::new(layout)))
        .collect();
        let mut pools = BTreeMap::new();
        for domain in &domains {
            let instance_id = NEXT_DYNAMIC_POOL_INSTANCE_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_add(1)
                })
                .map_err(|_| invalid_resource("dynamic pool instance id space is exhausted"))?;
            let pool = Arc::new(DynamicBackingPool {
                instance_id,
                domain: domain.clone(),
                logical_admission: logical_admission.clone(),
                maintenance: Mutex::new(()),
                next_extent_generation: AtomicU64::new(1),
                state: Mutex::new(DynamicBackingPoolState {
                    resident_bytes: 0,
                    pending_growth_bytes: 0,
                    next_chunk_ordinal: 1,
                    next_chunk_generation: 1,
                    chunks: BTreeMap::new(),
                    allocator: FreeExtentIndex::default(),
                    live_occupancy: DynamicPoolLiveOccupancyStatus::default(),
                    quarantined: Vec::new(),
                    poisoned: false,
                }),
            });
            if pools.insert(domain.pool_id().clone(), pool).is_some() {
                return Err(invalid_resource(
                    "dynamic pool set contains a duplicate pool",
                ));
            }
        }
        Ok(Self {
            runtime,
            binding,
            budget,
            logical_admission,
            domains,
            pools,
            nodes,
            submission_wave_layouts,
            submission_wave_reusable_capacity_layouts,
            program_binding_layouts,
            reusable_execution,
            lane_stable_arenas: Arc::new(Mutex::new(LaneStableArenaState::default())),
        })
    }

    pub(in crate::vnext::resource) fn program_binding_layout(
        &self,
        bucket_id: &ReusableExecutionBucketId,
    ) -> Option<&Arc<ProgramBindingLayout>> {
        self.program_binding_layouts.get(bucket_id)
    }

    pub(in crate::vnext::resource) const fn maximum_active_sequences(&self) -> u32 {
        self.binding.maximum_active_sequences()
    }

    pub(in crate::vnext::resource) fn write_capacity_availability(
        &self,
        out: &mut Vec<CapacityAvailabilityEpoch>,
    ) -> Result<CapacityEpochs, VNextError> {
        let epochs = self.logical_admission.write_availability_epochs(out)?;
        self.budget.write_availability_epochs(out)?;
        debug_assert!(out
            .windows(2)
            .all(|pair| pair[0].source() < pair[1].source()));
        Ok(epochs)
    }

    /// Rebalances only whole, unreferenced chunks from non-target pools. The
    /// batch is selected before mutation, logical totals publish atomically,
    /// and physical grants are returned only after every pool lock is dropped.
    pub(in crate::vnext::resource) fn reclaim_idle_chunks_for_pressure(
        &self,
        pressure: &DeviceCapacityPressure,
        excluded_domains: &[CapacityDomainId],
        protected_immediate: &CapacityVector,
    ) -> Result<Option<DynamicPoolRebalanceReceipt>, VNextError> {
        if pressure.device_id() != self.runtime.descriptor().id.to_string() {
            return Err(invalid_resource(
                "dynamic pool rebalance received pressure for another device",
            ));
        }
        let deficit = pressure
            .requested_bytes()
            .checked_sub(pressure.available_bytes())
            .ok_or_else(|| invalid_resource("dynamic pool pressure has no reclaimable deficit"))?;
        if deficit == 0 {
            return Err(invalid_resource(
                "dynamic pool pressure has an empty reclaimable deficit",
            ));
        }

        let excluded_domains = excluded_domains
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let pools = self.pools.values().cloned().collect::<Vec<_>>();
        let maintenance = pools
            .iter()
            .map(|pool| {
                pool.maintenance
                    .lock()
                    .map_err(|_| invalid_resource("dynamic pool maintenance authority is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut states = pools
            .iter()
            .map(|pool| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let logical = self.logical_admission.snapshot()?;
        let used_by_domain = logical
            .domains()
            .iter()
            .map(|domain| (domain.domain(), domain.used().get()))
            .collect::<BTreeMap<_, _>>();
        let protected_by_domain = protected_immediate
            .entries()
            .iter()
            .map(|entry| (entry.domain(), entry.units().get()))
            .collect::<BTreeMap<_, _>>();

        let mut candidates = Vec::new();
        let mut reclaimable_by_pool = vec![0_u64; pools.len()];
        for (pool_index, (pool, state)) in pools.iter().zip(states.iter()).enumerate() {
            if state.poisoned {
                return Err(invalid_resource("dynamic backing pool is fail-closed"));
            }
            if excluded_domains.contains(&pool.domain.domain_id) || state.pending_growth_bytes != 0
            {
                continue;
            }
            let used = used_by_domain
                .get(&pool.domain.domain_id)
                .copied()
                .ok_or_else(|| invalid_resource("dynamic pool domain is absent from admission"))?;
            let protected = protected_by_domain
                .get(&pool.domain.domain_id)
                .copied()
                .unwrap_or(0);
            let coherent_runnable_floor = used.checked_add(protected).ok_or_else(|| {
                invalid_resource("dynamic pool protected runnable floor overflows u64")
            })?;
            let resident_floor = pool
                .domain
                .pool
                .provisioning()
                .minimum_resident_bytes()
                .max(coherent_runnable_floor);
            let reclaimable = state.resident_bytes.saturating_sub(resident_floor);
            reclaimable_by_pool[pool_index] = reclaimable;
            if reclaimable == 0 {
                continue;
            }
            for (&ordinal, chunk) in &state.chunks {
                let chunk_bytes = chunk.backing._grant.bytes();
                let full_extent = state.allocator.by_offset.get(&(ordinal, 0));
                if chunk.live_segments != 0
                    || Arc::strong_count(&chunk.backing) != 1
                    || chunk_bytes > reclaimable
                    || chunk.backing.descriptor.size_bytes != chunk_bytes
                    || full_extent.is_none_or(|extent| {
                        extent.chunk_generation != chunk.backing.identity.generation()
                            || extent.length_bytes != chunk_bytes
                    })
                {
                    continue;
                }
                candidates.push(IdleChunkReclaimCandidate {
                    pool_index,
                    chunk: chunk.backing.identity.clone(),
                    chunk_bytes,
                });
            }
        }

        let mut selected = Vec::<IdleChunkReclaimCandidate>::new();
        let mut selected_by_pool = vec![0_u64; pools.len()];
        let mut reclaimed_bytes = 0_u64;
        let best_single = candidates
            .iter()
            .filter(|candidate| candidate.chunk_bytes >= deficit)
            .min_by(|left, right| {
                left.chunk_bytes
                    .cmp(&right.chunk_bytes)
                    .then_with(|| left.pool_index.cmp(&right.pool_index))
                    .then_with(|| right.chunk.ordinal().cmp(&left.chunk.ordinal()))
            })
            .cloned();
        if let Some(candidate) = best_single {
            selected_by_pool[candidate.pool_index] = candidate.chunk_bytes;
            reclaimed_bytes = candidate.chunk_bytes;
            selected.push(candidate);
        } else {
            candidates.sort_by(|left, right| {
                right
                    .chunk_bytes
                    .cmp(&left.chunk_bytes)
                    .then_with(|| left.pool_index.cmp(&right.pool_index))
                    .then_with(|| right.chunk.ordinal().cmp(&left.chunk.ordinal()))
            });
            for candidate in candidates {
                let next_pool_total = selected_by_pool[candidate.pool_index]
                    .checked_add(candidate.chunk_bytes)
                    .ok_or_else(|| invalid_resource("dynamic reclaim bytes overflow u64"))?;
                if next_pool_total > reclaimable_by_pool[candidate.pool_index] {
                    continue;
                }
                selected_by_pool[candidate.pool_index] = next_pool_total;
                reclaimed_bytes = reclaimed_bytes
                    .checked_add(candidate.chunk_bytes)
                    .ok_or_else(|| invalid_resource("dynamic reclaim bytes overflow u64"))?;
                selected.push(candidate);
                if reclaimed_bytes >= deficit {
                    break;
                }
            }
        }
        if reclaimed_bytes < deficit {
            return Ok(None);
        }
        selected.sort_by(|left, right| {
            left.pool_index
                .cmp(&right.pool_index)
                .then_with(|| left.chunk.ordinal().cmp(&right.chunk.ordinal()))
        });

        for candidate in &selected {
            let state = &states[candidate.pool_index];
            let chunk = state
                .chunks
                .get(&candidate.chunk.ordinal())
                .ok_or_else(|| invalid_resource("selected dynamic reclaim chunk disappeared"))?;
            let extent = state
                .allocator
                .by_offset
                .get(&(candidate.chunk.ordinal(), 0))
                .ok_or_else(|| invalid_resource("selected dynamic reclaim extent disappeared"))?;
            if chunk.live_segments != 0
                || Arc::strong_count(&chunk.backing) != 1
                || chunk.backing.identity != candidate.chunk
                || extent.chunk_generation != candidate.chunk.generation()
                || extent.length_bytes != candidate.chunk_bytes
            {
                return Err(invalid_resource(
                    "selected dynamic reclaim chunk changed before publication",
                ));
            }
        }

        let published_totals = states
            .iter()
            .zip(&selected_by_pool)
            .map(|(state, &selected_bytes)| {
                state
                    .resident_bytes
                    .checked_sub(selected_bytes)
                    .ok_or_else(|| invalid_resource("dynamic reclaim resident bytes underflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut removed = Vec::with_capacity(selected.len());
        for candidate in &selected {
            let state = &mut states[candidate.pool_index];
            let extent = state
                .allocator
                .remove_extent(candidate.chunk.ordinal(), 0)
                .expect("validated idle chunk retains its exact full extent");
            debug_assert_eq!(extent.chunk_generation, candidate.chunk.generation());
            debug_assert_eq!(extent.length_bytes, candidate.chunk_bytes);
            let chunk = state
                .chunks
                .remove(&candidate.chunk.ordinal())
                .expect("validated idle chunk remains resident");
            removed.push((candidate.clone(), chunk));
        }
        let updates = selected_by_pool
            .iter()
            .enumerate()
            .filter(|(_, selected_bytes)| **selected_bytes != 0)
            .map(|(pool_index, _)| {
                (
                    pools[pool_index].domain.domain_id,
                    CapacityUnits::new(published_totals[pool_index]),
                )
            })
            .collect::<Vec<_>>();
        let epochs = match self.logical_admission.set_domain_totals(&updates) {
            Ok(epochs) => epochs,
            Err(error) => {
                for (candidate, chunk) in removed.drain(..).rev() {
                    let state = &mut states[candidate.pool_index];
                    state
                        .allocator
                        .insert_extent(
                            candidate.chunk.ordinal(),
                            candidate.chunk.generation(),
                            0,
                            candidate.chunk_bytes,
                        )
                        .expect("unpublished idle chunk extent can be restored");
                    assert!(state
                        .chunks
                        .insert(candidate.chunk.ordinal(), chunk)
                        .is_none());
                }
                return Err(error);
            }
        };
        for (state, &published_total) in states.iter_mut().zip(&published_totals) {
            state.resident_bytes = published_total;
        }

        let mut pool_receipts = Vec::new();
        for (pool_index, &pool_reclaimed_bytes) in selected_by_pool.iter().enumerate() {
            if pool_reclaimed_bytes == 0 {
                continue;
            }
            pool_receipts.push(DynamicPoolIdleReclaim {
                pool_id: pools[pool_index].domain.pool_id().clone(),
                chunks: selected
                    .iter()
                    .filter(|candidate| candidate.pool_index == pool_index)
                    .map(|candidate| candidate.chunk.clone())
                    .collect(),
                reclaimed_bytes: pool_reclaimed_bytes,
                published_capacity_bytes: published_totals[pool_index],
            });
        }
        let reclaimed_chunks = selected.len();
        drop(states);
        drop(maintenance);
        drop(removed);
        let availability = self.budget.availability_snapshot()?;
        Ok(Some(DynamicPoolRebalanceReceipt {
            pools: pool_receipts,
            reclaimed_chunks,
            reclaimed_bytes,
            logical_capacity_epoch: epochs.capacity_epoch(),
            plan_device_capacity_epoch: availability.plan_epoch(),
            process_device_capacity_epoch: availability.process_epoch(),
        }))
    }

    pub(in crate::vnext::resource) fn maintain_pools(
        &self,
        intents: Vec<DynamicPoolGrowthIntent>,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        let mut ignored_capacity_block = None;
        self.maintain_pools_observed(intents, &mut ignored_capacity_block)
    }

    pub(in crate::vnext::resource) fn maintain_pools_observed(
        &self,
        mut intents: Vec<DynamicPoolGrowthIntent>,
        capacity_blocked: &mut Option<DynamicDeviceCapacityBlocked>,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        intents.sort_by(|left, right| left.pool_id().cmp(right.pool_id()));
        if intents
            .windows(2)
            .any(|pair| pair[0].pool_id() == pair[1].pool_id())
        {
            return Err(invalid_resource(
                "dynamic maintenance batch contains a duplicate pool",
            ));
        }
        let pools = intents
            .iter()
            .map(|intent| {
                self.pools.get(intent.pool_id()).cloned().ok_or_else(|| {
                    invalid_resource("dynamic maintenance references an unknown pool")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let _maintenance = pools
            .iter()
            .map(|pool| {
                pool.maintenance
                    .lock()
                    .map_err(|_| invalid_resource("dynamic pool maintenance authority is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut planned = Vec::with_capacity(intents.len());
        let mut pending = Vec::with_capacity(intents.len());
        for (intent, pool) in intents.iter().zip(&pools) {
            let requested_bytes = {
                let state = pool
                    .state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                match intent {
                    DynamicPoolGrowthIntent::Additional(request) => request.requested_bytes(),
                    DynamicPoolGrowthIntent::Minimum(_) => {
                        let current = state
                            .resident_bytes
                            .checked_add(state.pending_growth_bytes)
                            .ok_or_else(|| {
                                invalid_resource("dynamic pool residency overflows u64")
                            })?;
                        let minimum = pool.domain.pool.provisioning().minimum_resident_bytes();
                        if current >= minimum {
                            continue;
                        }
                        minimum - current
                    }
                    DynamicPoolGrowthIntent::RevalidatedDeferral(blocker) => {
                        if let Some(claim_bytes) = blocker.contiguous_claim_bytes_descending() {
                            let required_growth = contiguous_packing_growth_bytes(
                                &state.allocator,
                                pool.domain.pool_id(),
                                claim_bytes,
                            )?;
                            if required_growth == 0 {
                                continue;
                            }
                            required_growth
                        } else {
                            match blocker.reason() {
                                DynamicBackingDeferralReason::GrowthRequired => {
                                    let required_free = blocker
                                        .free_bytes()
                                        .checked_add(blocker.requested_bytes())
                                        .ok_or_else(|| {
                                            invalid_resource(
                                            "dynamic backing deferred requirement overflows u64",
                                        )
                                        })?;
                                    if state.allocator.free_bytes >= required_free {
                                        continue;
                                    }
                                    required_free - state.allocator.free_bytes
                                }
                                DynamicBackingDeferralReason::FragmentedContiguous => {
                                    return Err(invalid_resource(
                                        "fragmented contiguous blocker lost its transaction demand",
                                    ));
                                }
                            }
                        }
                    }
                }
            };
            let chunk_bytes = align_up_resource(requested_bytes, pool.allocation_quantum())?;
            let chunk = {
                let mut state = pool
                    .state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                let ordinal = state.next_chunk_ordinal;
                let generation = state.next_chunk_generation;
                let next_ordinal = ordinal
                    .checked_add(1)
                    .ok_or_else(|| invalid_resource("dynamic chunk ordinal space is exhausted"))?;
                let next_generation = generation.checked_add(1).ok_or_else(|| {
                    invalid_resource("dynamic chunk generation space is exhausted")
                })?;
                let next_pending = state
                    .pending_growth_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| invalid_resource("pending dynamic growth bytes overflow u64"))?;
                state.next_chunk_ordinal = next_ordinal;
                state.next_chunk_generation = next_generation;
                state.pending_growth_bytes = next_pending;
                BackingChunkIdentity::from_parts(
                    pool.domain.pool_id().clone(),
                    ordinal,
                    generation,
                )?
            };
            pending.push(PendingGrowthGuard {
                pool: Arc::clone(pool),
                bytes: chunk_bytes,
                armed: true,
            });
            let expected_resource_id = ResourceId::new(format!(
                "{}/chunk/{}/{}",
                pool.domain.pool_id().as_str(),
                chunk.ordinal(),
                chunk.generation()
            ))?;
            planned.push(PlannedDynamicGrowth {
                pool: Arc::clone(pool),
                chunk,
                expected_resource_id,
                chunk_bytes,
            });
        }
        if planned.is_empty() {
            return Ok(DynamicPoolGrowthBatchReceipt {
                coordinator_id: self.logical_admission.id(),
                growths: Vec::new(),
                capacity_epoch: self.logical_admission.epochs()?.capacity_epoch(),
                rebalance: None,
            });
        }

        let total_bytes = planned.iter().try_fold(0_u64, |total, growth| {
            total
                .checked_add(growth.chunk_bytes)
                .ok_or_else(|| invalid_resource("dynamic maintenance batch bytes overflow u64"))
        })?;
        let capacity_availability = self.budget.availability_snapshot()?;
        let reservation = match DeviceCapacityReservation::reserve(&self.budget, total_bytes) {
            Ok(reservation) => reservation,
            Err(VNextError::DeviceCapacityUnavailable(pressure)) => {
                *capacity_blocked = Some(DynamicDeviceCapacityBlocked {
                    pressure: pressure.clone(),
                    availability: capacity_availability,
                    planned_domains: planned
                        .iter()
                        .map(|growth| growth.pool.domain.domain_id)
                        .collect(),
                });
                return Err(VNextError::DeviceCapacityUnavailable(pressure));
            }
            Err(error) => return Err(error),
        };
        // Device-budget saturation is recoverable pressure even when the same
        // growth also crosses a pool's device-derived resident ceiling. Only a
        // growth that the authoritative device budget accepted can prove that
        // the remaining pool ceiling is a terminal theoretical-plan violation.
        for growth in &planned {
            let state = growth
                .pool
                .state
                .lock()
                .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
            let next_residency = state
                .resident_bytes
                .checked_add(state.pending_growth_bytes)
                .ok_or_else(|| invalid_resource("dynamic pool resident bytes overflow u64"))?;
            if next_residency
                > growth
                    .pool
                    .domain
                    .pool
                    .provisioning()
                    .maximum_resident_bytes()
            {
                return Err(VNextError::DynamicPoolResidentUnavailable(
                    DynamicPoolResidentPressure::new(
                        growth.pool.domain.pool_id().clone(),
                        growth.chunk_bytes,
                        state.resident_bytes,
                        growth
                            .pool
                            .domain
                            .pool
                            .provisioning()
                            .maximum_resident_bytes(),
                    )?,
                ));
            }
        }
        let grants = reservation.commit_split(
            &planned
                .iter()
                .map(|growth| growth.chunk_bytes)
                .collect::<Vec<_>>(),
        )?;
        let mut allocated = Vec::with_capacity(planned.len());
        for (growth, grant) in planned.iter().zip(grants) {
            let transaction_identity = ResourceTransactionIdentity {
                pool_id: self.binding.pool_id(),
                run_id: RunId::new(format!("dynamic-grow-{}", growth.chunk.generation()))?,
                transaction_id: TransactionId::new(format!(
                    "dynamic-grow-{}-{}",
                    growth.chunk.ordinal(),
                    growth.chunk.generation()
                ))?,
                request_id: self.binding.request_id().clone(),
            };
            let reservation_evidence = ResourceReservation {
                resource_id: growth.expected_resource_id.clone(),
                request_id: self.binding.request_id().clone(),
                owner_node_id: None,
                size_bytes: growth.chunk_bytes,
                alignment_bytes: growth.pool.domain.pool.compatibility().alignment_bytes(),
                usage: growth.pool.domain.pool.compatibility().usage(),
                element_type: growth.pool.domain.pool.compatibility().element_type(),
                retention_policy: ResourceRetentionPolicy::Plan,
                backing_domain_id: Some(growth.pool.domain.domain_id),
                generation: growth.chunk.generation(),
            };
            let request = BufferRequest::new(
                growth.expected_resource_id.clone(),
                growth.chunk_bytes,
                reservation_evidence.alignment_bytes,
                reservation_evidence.usage,
                reservation_evidence.element_type,
            )?;
            validate_runtime_descriptor_for_admission(
                self.runtime.descriptor(),
                &self.binding,
                "dynamic pool batch growth preflight",
            )?;
            let buffer = self
                .runtime
                .allocate(DeviceAllocationPermit {
                    identity: &transaction_identity,
                    binding: &self.binding,
                    reservation: &reservation_evidence,
                    request: &request,
                    seal: AllocationSeal,
                })
                .map_err(|error| {
                    invalid_resource(format!("dynamic pool device allocation failed: {error}"))
                })?;
            let actual_descriptor = self.runtime.buffer_descriptor(&buffer);
            validate_runtime_descriptor_for_admission(
                self.runtime.descriptor(),
                &self.binding,
                "dynamic pool batch growth completion",
            )?;
            if grant.bytes() != growth.chunk_bytes {
                return Err(invalid_resource(
                    "dynamic pool capacity grant differs from its chunk",
                ));
            }
            let backing = Arc::new(ResidentChunkBacking {
                buffer,
                _grant: grant,
                identity: growth.chunk.clone(),
                descriptor: actual_descriptor.clone(),
            });
            if !reservation_evidence.matches_descriptor(&actual_descriptor) {
                let mut state = growth
                    .pool
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.quarantined.push(QuarantinedDynamicChunk {
                    backing,
                    reason: DynamicChunkQuarantineReason::DescriptorMismatch,
                });
                return Err(invalid_resource(
                    "dynamic chunk descriptor mismatch was quarantined without capacity publication",
                ));
            }
            allocated.push(AllocatedDynamicGrowth { backing });
        }

        let mut states = planned
            .iter()
            .map(|growth| {
                growth.pool.state.lock().map_err(|_| {
                    invalid_resource("dynamic backing pool is poisoned after allocation")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut published_totals = Vec::with_capacity(planned.len());
        for ((growth, allocation), state) in planned.iter().zip(&allocated).zip(&states) {
            if state.poisoned
                || state.pending_growth_bytes < growth.chunk_bytes
                || state.chunks.contains_key(&growth.chunk.ordinal())
                || state
                    .allocator
                    .by_offset
                    .range((growth.chunk.ordinal(), 0)..=(growth.chunk.ordinal(), u64::MAX))
                    .next()
                    .is_some()
                || state
                    .allocator
                    .free_bytes
                    .checked_add(growth.chunk_bytes)
                    .is_none()
                || allocation.backing.identity != growth.chunk
            {
                return Err(invalid_resource(
                    "dynamic batch installation preconditions changed before publication",
                ));
            }
            published_totals.push(
                state
                    .resident_bytes
                    .checked_add(growth.chunk_bytes)
                    .ok_or_else(|| invalid_resource("published dynamic capacity overflows u64"))?,
            );
        }
        for index in 0..planned.len() {
            let growth = &planned[index];
            let state = &mut states[index];
            state.pending_growth_bytes -= growth.chunk_bytes;
            pending[index].disarm();
            state
                .allocator
                .insert_extent(
                    growth.chunk.ordinal(),
                    growth.chunk.generation(),
                    0,
                    growth.chunk_bytes,
                )
                .expect("validated new chunk has one disjoint free extent");
            state.chunks.insert(
                growth.chunk.ordinal(),
                ResidentChunkState {
                    backing: Arc::clone(&allocated[index].backing),
                    live_segments: 0,
                },
            );
        }
        let updates = planned
            .iter()
            .zip(&published_totals)
            .map(|(growth, &total)| (growth.pool.domain.domain_id, CapacityUnits::new(total)))
            .collect::<Vec<_>>();
        let epochs = match self.logical_admission.set_domain_totals(&updates) {
            Ok(epochs) => epochs,
            Err(error) => {
                for index in 0..planned.len() {
                    states[index]
                        .allocator
                        .remove_extent(planned[index].chunk.ordinal(), 0)
                        .expect("unpublished dynamic chunk free extent remains installed");
                    let removed = states[index]
                        .chunks
                        .remove(&planned[index].chunk.ordinal())
                        .expect("unpublished dynamic chunk remains installed");
                    states[index].quarantined.push(QuarantinedDynamicChunk {
                        backing: removed.backing,
                        reason: DynamicChunkQuarantineReason::PublicationRejected,
                    });
                }
                return Err(error);
            }
        };
        for (state, &published_total) in states.iter_mut().zip(&published_totals) {
            state.resident_bytes = published_total;
        }
        Ok(DynamicPoolGrowthBatchReceipt {
            coordinator_id: self.logical_admission.id(),
            growths: planned
                .iter()
                .zip(published_totals)
                .map(
                    |(growth, published_capacity_bytes)| DynamicPoolGrowthReceipt {
                        pool_id: growth.pool.domain.pool_id().clone(),
                        chunk: growth.chunk.clone(),
                        chunk_bytes: growth.chunk_bytes,
                        published_capacity_bytes,
                        capacity_epoch: epochs.capacity_epoch(),
                    },
                )
                .collect(),
            capacity_epoch: epochs.capacity_epoch(),
            rebalance: None,
        })
    }

    pub(in crate::vnext::resource) fn prepare_claim(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        if requests.is_empty() {
            return Ok(BackingPrepareDecision::Prepared(
                PreparedBackingClaim::empty(),
            ));
        }
        let lifetime = requests
            .first()
            .and_then(|request| request.projections.first())
            .map(|projection| projection.descriptor.lifetime())
            .ok_or_else(|| invalid_resource("dynamic backing request has no projection"))?;
        self.prepare_claim_scoped(
            requests,
            DynamicBackingClaimScope::from(lifetime),
            DynamicBackingClaimResidency::Transient,
        )
    }

    fn reusable_capacity_shape_for_requests(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<Option<DynamicResourceShape>, VNextError> {
        let reusable_execution_bucket_id = requests
            .first()
            .and_then(|request| request.reusable_execution_bucket_id.as_ref());
        if requests.iter().any(|request| {
            request.reusable_execution_bucket_id.as_ref() != reusable_execution_bucket_id
        }) {
            return Err(invalid_resource(
                "one dynamic backing claim cannot mix reusable execution buckets",
            ));
        }
        reusable_execution_bucket_id
            .map(|bucket_id| {
                let bucket = self
                    .reusable_execution
                    .as_ref()
                    .and_then(|plan| plan.bucket(bucket_id))
                    .map(|resolved| resolved.bucket())
                    .ok_or_else(|| {
                        invalid_resource(
                            "dynamic backing claim references a reusable bucket outside its immutable plan",
                        )
                    })?;
                Ok(DynamicResourceShape::from_validated(
                    bucket.capacity().maximum_sequences(),
                    bucket.capacity().maximum_tokens(),
                    bucket.capacity().maximum_pages(),
                ))
            })
            .transpose()
    }

    pub(in crate::vnext::resource) fn prepare_lane_stable_claim(
        self: &Arc<Self>,
        lane: &Arc<ExecutionLane<R>>,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<LaneBackingPrepareDecision, VNextError> {
        if !Arc::ptr_eq(&self.runtime, lane.runtime_arc())
            || lane.descriptor() != self.runtime.descriptor()
            || !lane.is_reusable()
        {
            return Err(invalid_resource(
                "lane-stable backing requires the reusable execution lane bound to this plan runtime",
            ));
        }
        if requests.is_empty() {
            return Ok(LaneBackingPrepareDecision::Prepared(
                PreparedLaneBackingClaim::new(Vec::new(), None)?,
            ));
        }
        let reusable_capacity_shape = self.reusable_capacity_shape_for_requests(requests)?;
        if reusable_capacity_shape.is_none() {
            return match self.prepare_claim(requests)? {
                BackingPrepareDecision::Prepared(prepared) => {
                    Ok(LaneBackingPrepareDecision::Prepared(
                        PreparedLaneBackingClaim::new(prepared.commit(), None)?,
                    ))
                }
                BackingPrepareDecision::Deferred(deferred) => {
                    Ok(LaneBackingPrepareDecision::Deferred(deferred))
                }
            };
        }
        let lifetime = requests
            .first()
            .and_then(|request| request.projections.first())
            .map(|projection| projection.descriptor.lifetime())
            .ok_or_else(|| invalid_resource("dynamic backing request has no projection"))?;
        if !matches!(
            lifetime,
            AllocationLifetime::Step | AllocationLifetime::Invocation
        ) || requests.iter().any(|request| {
            request.projections.is_empty()
                || request.projections.iter().any(|projection| {
                    projection.descriptor.lifetime() != lifetime
                        || projection.descriptor.initialization() != StateInitialization::None
                })
        }) {
            return Err(invalid_resource(
                "lane-stable backing accepts only non-initialized Step or Invocation resources",
            ));
        }

        let mut canonical_requests = requests.iter().collect::<Vec<_>>();
        canonical_requests
            .sort_unstable_by(|left, right| left.claim_identity.cmp(&right.claim_identity));
        let key = lane_stable_layout_key(lane.id(), lifetime, &canonical_requests)?;
        let lane_owner: Arc<dyn LaneStableArenaLane> =
            Arc::clone(lane) as Arc<dyn LaneStableArenaLane>;

        loop {
            {
                let mut arenas = self
                    .lane_stable_arenas
                    .lock()
                    .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
                if arenas.poisoned {
                    return Err(invalid_resource(
                        "lane-stable arena registry is fail-closed",
                    ));
                }
                let now = arenas.tick();
                if let Some(entry) = arenas.entries.get_mut(&key) {
                    let owner = entry.lane.upgrade().ok_or_else(|| {
                        invalid_resource("lane-stable arena retained an expired execution lane")
                    })?;
                    if !Arc::ptr_eq(&owner, &lane_owner) {
                        return Err(invalid_resource(
                            "lane-stable arena identity aliases another execution lane",
                        ));
                    }
                    if let Some((slot_id, stable, certificate, slot_domains)) =
                        entry.claim_idle_slot(lane.id(), now, &canonical_requests)?
                    {
                        return Ok(LaneBackingPrepareDecision::Prepared(
                            PreparedLaneBackingClaim::certified(
                                stable,
                                certificate,
                                LaneStableArenaSlotLease {
                                    arenas: Arc::clone(&self.lane_stable_arenas),
                                    logical_admission: self.logical_admission.clone(),
                                    availability_domains: slot_domains,
                                    key: key.clone(),
                                    slot_id,
                                },
                            ),
                        ));
                    }
                }
            }

            match self.prepare_claim_scoped(
                requests,
                DynamicBackingClaimScope::from(lifetime),
                DynamicBackingClaimResidency::LaneStable,
            )? {
                BackingPrepareDecision::Prepared(prepared) => {
                    let mut arenas = self
                        .lane_stable_arenas
                        .lock()
                        .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
                    if arenas.poisoned {
                        return Err(invalid_resource(
                            "lane-stable arena registry is fail-closed",
                        ));
                    }
                    let now = arenas.tick();
                    if let Some(entry) = arenas.entries.get_mut(&key) {
                        let owner = entry.lane.upgrade().ok_or_else(|| {
                            invalid_resource("lane-stable arena retained an expired execution lane")
                        })?;
                        if !Arc::ptr_eq(&owner, &lane_owner) {
                            return Err(invalid_resource(
                                "lane-stable arena identity aliases another execution lane",
                            ));
                        }
                        if let Some((slot_id, stable, certificate, slot_domains)) =
                            entry.claim_idle_slot(lane.id(), now, &canonical_requests)?
                        {
                            drop(arenas);
                            drop(prepared);
                            return Ok(LaneBackingPrepareDecision::Prepared(
                                PreparedLaneBackingClaim::certified(
                                    stable,
                                    certificate,
                                    LaneStableArenaSlotLease {
                                        arenas: Arc::clone(&self.lane_stable_arenas),
                                        logical_admission: self.logical_admission.clone(),
                                        availability_domains: slot_domains,
                                        key: key.clone(),
                                        slot_id,
                                    },
                                ),
                            ));
                        }
                    }
                    let authorities = prepared.commit();
                    let projection_bindings =
                        bind_lane_stable_slot_projections(&authorities, &canonical_requests)?;
                    let certificate = Arc::new(BackingClaimCertificate::from_slices(&authorities)?);
                    let stable = authorities
                        .iter()
                        .map(|authority| authority.retained_for_lane(lane.id()))
                        .collect();
                    let availability_domains = requests
                        .iter()
                        .map(|request| request.domain.domain_id())
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>();
                    let slot_id = arenas.issue_slot_id()?;
                    let entry =
                        arenas
                            .entries
                            .entry(key.clone())
                            .or_insert_with(|| LaneStableArenaEntry {
                                lane: Arc::downgrade(&lane_owner),
                                slots: BTreeMap::new(),
                            });
                    if entry
                        .slots
                        .insert(
                            slot_id,
                            LaneStableArenaSlot {
                                slot_id,
                                authorities,
                                certificate: Arc::clone(&certificate),
                                projection_bindings,
                                availability_domains: availability_domains.clone(),
                                in_use: true,
                                last_used: now,
                            },
                        )
                        .is_some()
                    {
                        arenas.poisoned = true;
                        return Err(invalid_resource(
                            "lane-stable arena slot publication replaced an existing slot",
                        ));
                    }
                    return Ok(LaneBackingPrepareDecision::Prepared(
                        PreparedLaneBackingClaim::certified(
                            stable,
                            certificate,
                            LaneStableArenaSlotLease {
                                arenas: Arc::clone(&self.lane_stable_arenas),
                                logical_admission: self.logical_admission.clone(),
                                availability_domains: availability_domains.clone(),
                                key: key.clone(),
                                slot_id,
                            },
                        ),
                    ));
                }
                BackingPrepareDecision::Deferred(deferred) => {
                    let mut arenas = self
                        .lane_stable_arenas
                        .lock()
                        .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
                    if arenas.poisoned {
                        return Err(invalid_resource(
                            "lane-stable arena registry is fail-closed",
                        ));
                    }
                    let now = arenas.tick();
                    if let Some(entry) = arenas.entries.get_mut(&key) {
                        let owner = entry.lane.upgrade().ok_or_else(|| {
                            invalid_resource("lane-stable arena retained an expired execution lane")
                        })?;
                        if !Arc::ptr_eq(&owner, &lane_owner) {
                            return Err(invalid_resource(
                                "lane-stable arena identity aliases another execution lane",
                            ));
                        }
                        if let Some((slot_id, stable, certificate, slot_domains)) =
                            entry.claim_idle_slot(lane.id(), now, &canonical_requests)?
                        {
                            drop(arenas);
                            drop(deferred);
                            return Ok(LaneBackingPrepareDecision::Prepared(
                                PreparedLaneBackingClaim::certified(
                                    stable,
                                    certificate,
                                    LaneStableArenaSlotLease {
                                        arenas: Arc::clone(&self.lane_stable_arenas),
                                        logical_admission: self.logical_admission.clone(),
                                        availability_domains: slot_domains,
                                        key: key.clone(),
                                        slot_id,
                                    },
                                ),
                            ));
                        }
                    }
                    return Ok(LaneBackingPrepareDecision::Deferred(deferred));
                }
            }
        }
    }

    pub(in crate::vnext::resource) fn try_reclaim_expired_lane_slots(
        &self,
    ) -> Result<bool, VNextError> {
        let expired_entries = {
            let mut arenas = self
                .lane_stable_arenas
                .lock()
                .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
            if arenas.poisoned {
                return Err(invalid_resource(
                    "lane-stable arena registry is fail-closed",
                ));
            }
            arenas.take_expired_lanes()?
        };
        let reclaimed = !expired_entries.is_empty();
        // Releasing backing owners can enter backend/pool destruction paths.
        // Keep that work outside the arena registry's hot mutex.
        drop(expired_entries);
        Ok(reclaimed)
    }

    pub(in crate::vnext::resource) fn try_reclaim_one_idle_lane_slot(
        &self,
    ) -> Result<bool, VNextError> {
        if self.try_reclaim_expired_lane_slots()? {
            return Ok(true);
        }
        let mut candidates = {
            let arenas = self
                .lane_stable_arenas
                .lock()
                .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
            if arenas.poisoned {
                return Err(invalid_resource(
                    "lane-stable arena registry is fail-closed",
                ));
            }
            arenas
                .entries
                .iter()
                .filter_map(|(key, entry)| entry.lane.upgrade().map(|lane| (key, entry, lane)))
                .flat_map(|(key, entry, lane)| {
                    entry
                        .slots
                        .values()
                        .filter(|slot| !slot.in_use)
                        .map(move |slot| LaneStableArenaEvictionCandidate {
                            key: key.clone(),
                            slot_id: slot.slot_id,
                            last_used: slot.last_used,
                            lane: Arc::clone(&lane),
                        })
                })
                .collect::<Vec<_>>()
        };
        candidates.sort_by_key(|candidate| candidate.last_used);

        for candidate in candidates {
            if !candidate.lane.try_trim_reusable_executables()? {
                continue;
            }
            let victim = {
                let mut arenas = self
                    .lane_stable_arenas
                    .lock()
                    .map_err(|_| invalid_resource("lane-stable arena registry is poisoned"))?;
                if arenas.poisoned {
                    return Err(invalid_resource(
                        "lane-stable arena registry is fail-closed",
                    ));
                }
                let removable = arenas
                    .entries
                    .get(&candidate.key)
                    .and_then(|entry| entry.slots.get(&candidate.slot_id))
                    .is_some_and(|slot| !slot.in_use && !slot.has_external_address_pins());
                if !removable {
                    None
                } else {
                    let (victim, remove_entry) = {
                        let entry = arenas.entries.get_mut(&candidate.key).ok_or_else(|| {
                            invalid_resource("lane-stable arena eviction lost its entry")
                        })?;
                        let victim = entry.slots.remove(&candidate.slot_id).ok_or_else(|| {
                            invalid_resource("lane-stable arena eviction lost its idle slot")
                        })?;
                        (victim, entry.slots.is_empty())
                    };
                    if remove_entry {
                        arenas.entries.remove(&candidate.key);
                    }
                    Some(victim)
                }
            };
            if let Some(victim) = victim {
                drop(victim);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(in crate::vnext::resource) fn prepare_initial_sequence_claim(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        self.prepare_claim_scoped(
            requests,
            DynamicBackingClaimScope::InitialSequenceBundle,
            DynamicBackingClaimResidency::Transient,
        )
    }

    fn prepare_claim_scoped(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
        scope: DynamicBackingClaimScope,
        residency: DynamicBackingClaimResidency,
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        if requests.is_empty() {
            return Ok(BackingPrepareDecision::Prepared(
                PreparedBackingClaim::empty(),
            ));
        }
        let reusable_capacity_shape = self.reusable_capacity_shape_for_requests(requests)?;
        let mut grouped =
            BTreeMap::<DynamicBackingPoolId, Vec<&EvaluatedBackingRequest<'_>>>::new();
        for request in requests {
            grouped
                .entry(request.domain.pool_id().clone())
                .or_default()
                .push(request);
        }
        let mut groups = Vec::with_capacity(grouped.len());
        for (pool_id, mut requests) in grouped {
            requests.sort_by(|left, right| left.claim_identity.cmp(&right.claim_identity));
            if requests
                .windows(2)
                .any(|pair| pair[0].claim_identity == pair[1].claim_identity)
            {
                return Err(invalid_resource(
                    "dynamic backing reservation contains a duplicate physical claim",
                ));
            }
            let pool = self.pools.get(&pool_id).cloned().ok_or_else(|| {
                invalid_resource("dynamic backing reservation references an unknown pool")
            })?;
            groups.push((pool, requests));
        }
        let protected_immediate = CapacityVector::new(
            groups
                .iter()
                .map(|(pool, requests)| {
                    let bytes = requests.iter().try_fold(0_u64, |total, request| {
                        total
                            .checked_add(request.capacity_size_bytes)
                            .ok_or_else(|| {
                                invalid_resource("dynamic backing protection bytes overflow u64")
                            })
                    })?;
                    CapacityEntry::new(pool.domain.domain_id, CapacityUnits::new(bytes))
                })
                .collect::<Result<Vec<_>, VNextError>>()?,
        )?;
        'prepare: loop {
            let mut states = groups
                .iter()
                .map(|(pool, _)| {
                    pool.state
                        .lock()
                        .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (group_index, (pool, pool_requests)) in groups.iter().enumerate() {
                if states[group_index].poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                let quantum = pool.allocation_quantum();
                for request in pool_requests {
                    let projection_ids = request
                        .projections
                        .iter()
                        .map(|projection| projection.descriptor.base_resource_id().clone())
                        .collect::<Vec<_>>();
                    let single_projection = request.projections.len() == 1
                        && request.projections[0].physical_offset_bytes == 0
                        && request.projections[0].capacity_size_bytes
                            == request.capacity_size_bytes;
                    let shared_step_slot = request.projections.len() > 1
                        && request
                            .projections
                            .iter()
                            .all(|projection| projection.physical_offset_bytes == 0)
                        && request
                            .projections
                            .iter()
                            .map(|projection| projection.capacity_size_bytes)
                            .max()
                            == Some(request.capacity_size_bytes)
                        && pool.domain.pool.step_resource_slots().iter().any(|slot| {
                            slot.kind() == StepResourceSlotKind::OrderedSingleFenceStepWave
                                && slot.resource_ids() == request.claim_identity.resource_ids()
                        });
                    let invocation_wave =
                        self.validate_invocation_wave_projection(pool, request)?;
                    if request.domain.pool_id() != pool.domain.pool_id()
                        || request.claim_identity.pool_id() != pool.domain.pool_id()
                        || request.claim_identity.resource_ids() != projection_ids
                        || request.projections.is_empty()
                        || request.projections.windows(2).any(|pair| {
                            pair[0].descriptor.base_resource_id()
                                >= pair[1].descriptor.base_resource_id()
                        })
                        || request.projections.iter().any(|projection| {
                            let capacity_matches_plan = match reusable_capacity_shape {
                                Some(shape) => {
                                    matches!(
                                        projection.descriptor.lifetime(),
                                        AllocationLifetime::Step | AllocationLifetime::Invocation
                                    ) && projection
                                        .descriptor
                                        .evaluate_request_bytes_for_shape(shape)
                                        .is_ok_and(|bytes| bytes == projection.capacity_size_bytes)
                                }
                                None => {
                                    projection.logical_size_bytes == projection.capacity_size_bytes
                                }
                            };
                            projection.descriptor.pool_id() != pool.domain.pool_id()
                                || !scope.accepts(projection.descriptor.lifetime())
                                || !capacity_matches_plan
                                || projection.logical_size_bytes == 0
                                || projection.logical_size_bytes > projection.capacity_size_bytes
                                || projection.capacity_size_bytes == 0
                                || projection.capacity_size_bytes % quantum != 0
                                || projection.physical_offset_bytes % quantum != 0
                                || projection
                                    .physical_offset_bytes
                                    .checked_add(projection.capacity_size_bytes)
                                    .is_none_or(|end| end > request.capacity_size_bytes)
                                || !request
                                    .domain
                                    .descriptors
                                    .iter()
                                    .any(|descriptor| descriptor == projection.descriptor)
                        })
                        || request.capacity_size_bytes == 0
                        || request.capacity_size_bytes % quantum != 0
                        || !(single_projection || shared_step_slot || invocation_wave)
                    {
                        return Err(invalid_resource(
                        "dynamic backing request violates its physical claim, projection, pool, or allocation quantum",
                    ));
                    }
                }
            }
            let blockers = groups
                .iter()
                .enumerate()
                .map(|(group_index, (pool, pool_requests))| {
                    let requested_group_bytes =
                        pool_requests.iter().try_fold(0_u64, |total, request| {
                            total
                                .checked_add(request.capacity_size_bytes)
                                .ok_or_else(|| {
                                    invalid_resource("dynamic backing batch bytes overflow u64")
                                })
                        })?;
                    let state = &states[group_index];
                    if state.allocator.free_bytes >= requested_group_bytes {
                        return Ok(None);
                    }
                    let (reason, requested_bytes, contiguous_claim_bytes_descending) =
                        match pool.domain.pool.compatibility().profile().view() {
                            DynamicStorageView::Contiguous => {
                                let mut claim_bytes = pool_requests
                                    .iter()
                                    .map(|request| request.capacity_size_bytes)
                                    .collect::<Vec<_>>();
                                claim_bytes.sort_unstable_by(|left, right| right.cmp(left));
                                let growth = contiguous_packing_growth_bytes(
                                    &state.allocator,
                                    pool.domain.pool_id(),
                                    &claim_bytes,
                                )?;
                                if growth == 0 {
                                    return Err(invalid_resource(
                                    "insufficient contiguous capacity produced zero packing growth",
                                ));
                                }
                                (
                                    DynamicBackingDeferralReason::GrowthRequired,
                                    growth,
                                    Some(claim_bytes),
                                )
                            }
                            DynamicStorageView::PagedRegions { .. } => (
                                DynamicBackingDeferralReason::GrowthRequired,
                                requested_group_bytes - state.allocator.free_bytes,
                                None,
                            ),
                        };
                    Ok(Some(DynamicBackingBlocker {
                        pool_id: pool.domain.pool_id().clone(),
                        domain_id: pool.domain.domain_id,
                        reason,
                        requested_bytes,
                        free_bytes: state.allocator.free_bytes,
                        largest_contiguous_bytes: state.allocator.largest_contiguous_bytes(),
                        free_extent_layout_fingerprint: free_extent_layout_fingerprint(
                            &state.allocator,
                        ),
                        contiguous_claim_bytes_descending,
                    }))
                })
                .collect::<Result<Vec<_>, VNextError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            if !blockers.is_empty() {
                drop(states);
                if let Some(deferred) =
                    self.confirm_backing_deferral(blockers, scope, protected_immediate.clone())?
                {
                    return Ok(BackingPrepareDecision::Deferred(deferred));
                }
                continue 'prepare;
            }
            let segment_generations = groups
                .iter()
                .map(|(pool, requests)| {
                    (0..requests.len())
                        .map(|_| {
                            pool.next_extent_generation
                                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                                    current.checked_add(1)
                                })
                                .map_err(|_| {
                                    invalid_resource("dynamic extent generation space is exhausted")
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut selections = groups
                .iter()
                .map(|_| {
                    Vec::<(
                        &EvaluatedBackingRequest<'_>,
                        u64,
                        Vec<BackingSegment>,
                        DynamicBackingClaimOccupancy,
                    )>::new()
                })
                .collect::<Vec<_>>();
            let mut journals = groups
                .iter()
                .map(|_| Vec::<Vec<BackingSegment>>::new())
                .collect::<Vec<_>>();
            for group_index in 0..groups.len() {
                let (pool, pool_requests) = &groups[group_index];
                let profile = pool.domain.pool.compatibility().profile();
                let mut allocation_requests = pool_requests.clone();
                allocation_requests.sort_by(|left, right| {
                    right
                        .capacity_size_bytes
                        .cmp(&left.capacity_size_bytes)
                        .then_with(|| left.claim_identity.cmp(&right.claim_identity))
                });
                for request in allocation_requests {
                    let reserved = match match profile.view() {
                        DynamicStorageView::Contiguous => states[group_index]
                            .allocator
                            .allocate_contiguous(pool.domain.pool_id(), request.capacity_size_bytes)
                            .map(|segment| segment.map(|segment| vec![segment])),
                        DynamicStorageView::PagedRegions { block_bytes } => {
                            states[group_index].allocator.allocate_paged(
                                pool.domain.pool_id(),
                                request.capacity_size_bytes,
                                block_bytes,
                            )
                        }
                    } {
                        Ok(reserved) => reserved,
                        Err(error) => {
                            states[group_index].poisoned = true;
                            rollback_free_extent_journal(&mut states, &journals)?;
                            return Err(error);
                        }
                    };
                    let Some(segments) = reserved else {
                        rollback_free_extent_journal(&mut states, &journals)?;
                        if !matches!(profile.view(), DynamicStorageView::Contiguous) {
                            states[group_index].poisoned = true;
                            return Err(invalid_resource(
                                "paged backing allocation failed after its aggregate fit check",
                            ));
                        }
                        let free_bytes = states[group_index].allocator.free_bytes;
                        let reason = DynamicBackingDeferralReason::FragmentedContiguous;
                        let mut claim_bytes_descending = pool_requests
                            .iter()
                            .map(|request| request.capacity_size_bytes)
                            .collect::<Vec<_>>();
                        claim_bytes_descending.sort_unstable_by(|left, right| right.cmp(left));
                        let requested_bytes = contiguous_packing_growth_bytes(
                            &states[group_index].allocator,
                            pool.domain.pool_id(),
                            &claim_bytes_descending,
                        )?;
                        if requested_bytes == 0 {
                            return Err(invalid_resource(
                                "contiguous packing failed without a progress-producing growth",
                            ));
                        }
                        let blocker = DynamicBackingBlocker {
                            pool_id: pool.domain.pool_id().clone(),
                            domain_id: pool.domain.domain_id,
                            reason,
                            requested_bytes,
                            free_bytes,
                            largest_contiguous_bytes: states[group_index]
                                .allocator
                                .largest_contiguous_bytes(),
                            free_extent_layout_fingerprint: free_extent_layout_fingerprint(
                                &states[group_index].allocator,
                            ),
                            contiguous_claim_bytes_descending: Some(claim_bytes_descending),
                        };
                        drop(states);
                        if let Some(deferred) = self.confirm_backing_deferral(
                            vec![blocker],
                            scope,
                            protected_immediate.clone(),
                        )? {
                            return Ok(BackingPrepareDecision::Deferred(deferred));
                        }
                        continue 'prepare;
                    };
                    journals[group_index].push(segments.clone());
                    let extent_bytes = match segments.iter().try_fold(0_u64, |total, segment| {
                        total.checked_add(segment.length_bytes()).ok_or_else(|| {
                            invalid_resource("dynamic backing extent bytes overflow u64")
                        })
                    }) {
                        Ok(bytes) => bytes,
                        Err(error) => {
                            rollback_free_extent_journal(&mut states, &journals)?;
                            return Err(error);
                        }
                    };
                    if extent_bytes != request.capacity_size_bytes {
                        rollback_free_extent_journal(&mut states, &journals)?;
                        return Err(invalid_resource(
                            "dynamic backing extents differ from their physical capacity claim",
                        ));
                    }
                    let generation =
                        segment_generations[group_index][selections[group_index].len()];
                    let segment_count = match u64::try_from(segments.len()) {
                        Ok(count) => count,
                        Err(_) => {
                            rollback_free_extent_journal(&mut states, &journals)?;
                            return Err(invalid_resource(
                                "dynamic backing segment count exceeds portable range",
                            ));
                        }
                    };
                    selections[group_index].push((
                        request,
                        generation,
                        segments,
                        DynamicBackingClaimOccupancy {
                            scope,
                            residency,
                            physical_bytes: request.capacity_size_bytes,
                            segment_count,
                        },
                    ));
                }
            }

            for group_selections in &selections {
                for (request, _, segments, _) in group_selections {
                    for projection in &request.projections {
                        if let Err(error) = backing_segment_range(
                            segments,
                            projection.physical_offset_bytes,
                            projection.capacity_size_bytes,
                        ) {
                            rollback_free_extent_journal(&mut states, &journals)?;
                            return Err(error);
                        }
                    }
                }
            }

            let accounting_updates = match (0..groups.len())
                .map(|group_index| {
                    let mut increments = BTreeMap::<u32, u64>::new();
                    for (_, _, segments, _) in &selections[group_index] {
                        for segment in segments {
                            let count = increments.entry(segment.chunk_ordinal()).or_default();
                            *count = count.checked_add(1).ok_or_else(|| {
                                invalid_resource(
                                    "dynamic chunk live extent increment overflows u64",
                                )
                            })?;
                        }
                    }
                    for (&ordinal, &increment) in &increments {
                        states[group_index]
                            .chunks
                            .get(&ordinal)
                            .ok_or_else(|| invalid_resource("reserved dynamic chunk disappeared"))?
                            .live_segments
                            .checked_add(increment)
                            .ok_or_else(|| {
                                invalid_resource("dynamic chunk live extent count overflowed")
                            })?;
                    }
                    let occupancy = selections[group_index].iter().try_fold(
                        states[group_index].live_occupancy,
                        |occupancy, (_, _, _, claim)| occupancy.checked_with_claim(*claim),
                    )?;
                    Ok((increments, occupancy))
                })
                .collect::<Result<Vec<_>, VNextError>>()
            {
                Ok(updates) => updates,
                Err(error) => {
                    rollback_free_extent_journal(&mut states, &journals)?;
                    return Err(error);
                }
            };
            for (group_index, (increments, occupancy)) in accounting_updates.into_iter().enumerate()
            {
                for (ordinal, increment) in increments {
                    states[group_index]
                        .chunks
                        .get_mut(&ordinal)
                        .expect("validated reserved dynamic chunk remains installed")
                        .live_segments += increment;
                }
                states[group_index].live_occupancy = occupancy;
            }
            drop(states);
            let mut extents = Vec::new();
            for ((pool, _), selections) in groups.into_iter().zip(selections) {
                for (request, segment_generation, segments, occupancy) in selections {
                    let projections = request
                        .projections
                        .iter()
                        .map(|projection| {
                            let mut allocation = LogicalBackingSliceAllocationEvidence {
                                domain_id: pool.domain.domain_id,
                                pool_id: pool.domain.pool_id().clone(),
                                resource_id: projection.descriptor.base_resource_id().clone(),
                                pool_instance_id: pool.instance_id,
                                physical_claim_identity: request.claim_identity.clone(),
                                reusable_execution_bucket_id: request
                                    .reusable_execution_bucket_id
                                    .clone(),
                                segment_generation,
                                segments: backing_segment_range(
                                    &segments,
                                    projection.physical_offset_bytes,
                                    projection.capacity_size_bytes,
                                )?,
                                physical_offset_bytes: projection.physical_offset_bytes,
                                capacity_size_bytes: projection.capacity_size_bytes,
                                physical_size_bytes: request.capacity_size_bytes,
                                alignment_bytes: projection.descriptor.alignment_bytes(),
                                usage: projection.descriptor.usage(),
                                element_type: projection.descriptor.element_type(),
                                storage_profile: pool.domain.pool.compatibility().profile(),
                                initialization: projection.descriptor.initialization(),
                                fingerprint: String::new(),
                            };
                            let bytes = serde_json::to_vec(&allocation).map_err(|error| {
                                invalid_resource(format!(
                                    "logical backing allocation evidence encode failed: {error}"
                                ))
                            })?;
                            allocation.fingerprint = format!("sha256/{:x}", Sha256::digest(bytes));
                            Ok(LogicalBackingSliceEvidence {
                                allocation: Arc::new(allocation),
                                logical_size_bytes: projection.logical_size_bytes,
                            })
                        })
                        .collect::<Result<Vec<_>, VNextError>>()?;
                    extents.push(PreparedBackingExtent {
                        pool: Arc::clone(&pool),
                        claim_identity: request.claim_identity.clone(),
                        segment_generation,
                        occupancy,
                        segments,
                        capacity_size_bytes: request.capacity_size_bytes,
                        projections,
                    });
                }
            }
            return Ok(BackingPrepareDecision::Prepared(PreparedBackingClaim {
                extents,
                committed: false,
            }));
        }
    }

    /// Publishes a physical deferral with the event-subscription ordering
    /// required to avoid a lost release:
    ///
    /// 1. observe the exact coordinator generations after the failed check;
    /// 2. recheck the physical allocator observations;
    /// 3. publish only if those observations are still current.
    ///
    /// A release before step 1 is visible to step 2. A release after step 1
    /// advances the returned predicate (or is visible to step 2 before its
    /// coordinator notification), so the scheduler cannot sleep forever on a
    /// stale blocker.
    fn confirm_backing_deferral(
        &self,
        blockers: Vec<DynamicBackingBlocker>,
        scope: DynamicBackingClaimScope,
        protected_immediate: CapacityVector,
    ) -> Result<Option<DynamicBackingDeferred>, VNextError> {
        let wait_snapshot = self
            .logical_admission
            .wait_snapshot_for_domains(blockers.iter().map(DynamicBackingBlocker::domain_id))?;
        for blocker in &blockers {
            let pool = self.pools.get(blocker.pool_id()).ok_or_else(|| {
                invalid_resource("dynamic backing blocker references an unknown pool")
            })?;
            let state = pool
                .state
                .lock()
                .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
            if state.poisoned {
                return Err(invalid_resource("dynamic backing pool is fail-closed"));
            }
            if state.allocator.free_bytes != blocker.free_bytes()
                || state.allocator.largest_contiguous_bytes() != blocker.largest_contiguous_bytes()
                || free_extent_layout_fingerprint(&state.allocator)
                    != blocker.free_extent_layout_fingerprint()
            {
                return Ok(None);
            }
        }
        Ok(Some(DynamicBackingDeferred {
            blockers,
            epochs: wait_snapshot.epochs(),
            wait_condition: wait_snapshot.wait_condition().clone(),
            scope,
            protected_immediate,
        }))
    }

    fn validate_invocation_wave_projection(
        &self,
        pool: &DynamicBackingPool<R>,
        request: &EvaluatedBackingRequest<'_>,
    ) -> Result<bool, VNextError> {
        let mode = pool.domain.pool.invocation_liveness_mode();
        if mode == InvocationLivenessMode::NoInvocationResources
            || request.projections.len() < 2
            || request.projections.iter().any(|projection| {
                projection.descriptor.lifetime() != super::AllocationLifetime::Invocation
            })
        {
            return Ok(false);
        }
        let mut expected_resources = pool
            .domain
            .pool
            .invocation_liveness()
            .iter()
            .flat_map(|row| row.resource_ids().iter().cloned())
            .collect::<Vec<_>>();
        expected_resources.sort();
        if expected_resources.windows(2).any(|pair| pair[0] == pair[1])
            || expected_resources != request.claim_identity.resource_ids()
        {
            return Ok(false);
        }
        let projections = request
            .projections
            .iter()
            .map(|projection| (projection.descriptor.base_resource_id().clone(), projection))
            .collect::<BTreeMap<_, _>>();
        if projections.len() != request.projections.len() {
            return Ok(false);
        }
        let rows_by_node = pool
            .domain
            .pool
            .invocation_liveness()
            .iter()
            .map(|row| (row.node_id(), row))
            .collect::<BTreeMap<_, _>>();
        let rows = self
            .nodes
            .iter()
            .filter_map(|node| rows_by_node.get(node.id()).copied())
            .collect::<Vec<_>>();
        if rows.len() != rows_by_node.len() {
            return Ok(false);
        }

        let mut concurrent_cursor = 0_u64;
        let mut peak = 0_u64;
        for row in rows {
            let row_base = match mode {
                InvocationLivenessMode::TotalOrderReuse => 0,
                InvocationLivenessMode::ConservativeConcurrent => concurrent_cursor,
                InvocationLivenessMode::NoInvocationResources => unreachable!(),
            };
            let mut row_cursor = 0_u64;
            for resource_id in row.resource_ids() {
                let Some(projection) = projections.get(resource_id) else {
                    return Ok(false);
                };
                let expected_offset = row_base.checked_add(row_cursor).ok_or_else(|| {
                    invalid_resource("invocation wave projection offset overflows u64")
                })?;
                if projection.physical_offset_bytes != expected_offset {
                    return Ok(false);
                }
                row_cursor = row_cursor
                    .checked_add(projection.capacity_size_bytes)
                    .ok_or_else(|| invalid_resource("invocation wave row size overflows u64"))?;
            }
            peak = peak.max(row_cursor);
            if mode == InvocationLivenessMode::ConservativeConcurrent {
                concurrent_cursor = concurrent_cursor.checked_add(row_cursor).ok_or_else(|| {
                    invalid_resource("concurrent invocation wave size overflows u64")
                })?;
            }
        }
        Ok(request.capacity_size_bytes
            == match mode {
                InvocationLivenessMode::TotalOrderReuse => peak,
                InvocationLivenessMode::ConservativeConcurrent => concurrent_cursor,
                InvocationLivenessMode::NoInvocationResources => 0,
            })
    }

    pub(in crate::vnext::resource) fn view<'lease>(
        &'lease self,
        authority: &'lease LogicalBackingSliceAuthority,
    ) -> Result<LogicalBackingBufferView<'lease, R::Buffer>, VNextError> {
        self.view_many(std::slice::from_ref(authority))
    }

    pub(in crate::vnext::resource) fn view_many<'lease>(
        &'lease self,
        authorities: &'lease [LogicalBackingSliceAuthority],
    ) -> Result<LogicalBackingBufferView<'lease, R::Buffer>, VNextError> {
        let first = authorities
            .first()
            .ok_or_else(|| invalid_resource("logical backing view requires an authority"))?;
        let pool = self
            .pools
            .get(&first.evidence.pool_id)
            .ok_or_else(|| invalid_resource("logical backing authority has no dynamic pool"))?;
        let mut logical_size_bytes = 0_u64;
        let mut capacity_size_bytes = 0_u64;
        let mut segment_count = 0_usize;
        for (index, authority) in authorities.iter().enumerate() {
            if authority.evidence.pool_id != first.evidence.pool_id
                || authority.evidence.resource_id != first.evidence.resource_id
                || authority.evidence.storage_profile != first.evidence.storage_profile
                || authority.evidence.alignment_bytes != first.evidence.alignment_bytes
                || authority.evidence.usage != first.evidence.usage
                || authority.evidence.element_type != first.evidence.element_type
                || authority.evidence.initialization != first.evidence.initialization
            {
                return Err(invalid_resource(
                    "logical backing authorities have incompatible resource metadata",
                ));
            }
            Self::validate_authority(pool, authority)?;
            if index + 1 < authorities.len()
                && authority.evidence.logical_size_bytes != authority.evidence.capacity_size_bytes
            {
                return Err(invalid_resource(
                    "multi-extent logical backing cannot contain interior capacity slack",
                ));
            }
            logical_size_bytes = logical_size_bytes
                .checked_add(authority.evidence.logical_size_bytes)
                .ok_or_else(|| invalid_resource("logical backing view size overflows u64"))?;
            capacity_size_bytes = capacity_size_bytes
                .checked_add(authority.evidence.capacity_size_bytes)
                .ok_or_else(|| invalid_resource("logical backing capacity overflows u64"))?;
            segment_count = segment_count
                .checked_add(authority.evidence.segments.len())
                .ok_or_else(|| invalid_resource("logical backing segment count overflows usize"))?;
        }
        let state = pool
            .state
            .lock()
            .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
        if state.poisoned {
            return Err(invalid_resource("dynamic backing pool is fail-closed"));
        }
        let mut bindings = Vec::with_capacity(segment_count);
        for authority in authorities {
            for segment in &authority.evidence.segments {
                let chunk = state.chunks.get(&segment.chunk_ordinal()).ok_or_else(|| {
                    invalid_resource("logical backing references a missing chunk")
                })?;
                if segment.pool_id() != &authority.evidence.pool_id
                    || chunk.backing.identity != *segment.chunk()
                    || segment
                        .offset_bytes()
                        .checked_add(segment.length_bytes())
                        .is_none_or(|end| end > chunk.backing.descriptor.size_bytes)
                {
                    return Err(invalid_resource(
                        "logical backing references a stale or out-of-bounds chunk region",
                    ));
                }
                let retention = match authority.reusable_lane {
                    Some(lane_id) => DeviceBufferRetention::lane_pair(
                        lane_id,
                        Arc::clone(&authority.segment_lease),
                        Arc::clone(&chunk.backing),
                    ),
                    None => DeviceBufferRetention::pair(
                        Arc::clone(&authority.segment_lease),
                        Arc::clone(&chunk.backing),
                    ),
                };
                bindings.push(LogicalBackingSegmentBinding {
                    segment: segment.clone(),
                    chunk: Arc::clone(&chunk.backing),
                    retention,
                });
            }
        }
        drop(state);
        Ok(LogicalBackingBufferView {
            bindings,
            authorities,
            logical_size_bytes,
            capacity_size_bytes,
            alignment_bytes: first.evidence.alignment_bytes,
            usage: first.evidence.usage,
            element_type: first.evidence.element_type,
            storage_profile: first.evidence.storage_profile,
        })
    }

    fn validate_authority(
        pool: &DynamicBackingPool<R>,
        authority: &LogicalBackingSliceAuthority,
    ) -> Result<(), VNextError> {
        if pool.instance_id != authority.evidence.pool_instance_id
            || authority.segment_lease.owner_instance_id != pool.instance_id
            || authority.segment_lease.owner.instance_id() != pool.instance_id
            || authority.segment_lease.claim_identity != authority.evidence.physical_claim_identity
            || authority.segment_lease.segment_generation != authority.evidence.segment_generation
            || authority.segment_lease.size_bytes != authority.evidence.physical_size_bytes
            || authority.evidence.domain_id != pool.domain.domain_id
            || authority.evidence.physical_claim_identity.pool_id() != pool.domain.pool_id()
            || authority
                .evidence
                .physical_claim_identity
                .resource_ids()
                .binary_search(&authority.evidence.resource_id)
                .is_err()
            || authority.evidence.logical_size_bytes == 0
            || authority.evidence.logical_size_bytes > authority.evidence.capacity_size_bytes
            || authority
                .evidence
                .physical_offset_bytes
                .checked_add(authority.evidence.capacity_size_bytes)
                .is_none_or(|end| end > authority.evidence.physical_size_bytes)
            || authority.evidence.storage_profile != pool.domain.pool.compatibility().profile()
        {
            return Err(invalid_resource(
                "logical backing authority belongs to another dynamic pool instance",
            ));
        }
        let expected_projection = backing_segment_range(
            &authority.segment_lease.segments,
            authority.evidence.physical_offset_bytes,
            authority.evidence.capacity_size_bytes,
        )?;
        if expected_projection != authority.evidence.segments {
            return Err(invalid_resource(
                "logical backing projection differs from its shared physical extent",
            ));
        }
        Ok(())
    }
}
