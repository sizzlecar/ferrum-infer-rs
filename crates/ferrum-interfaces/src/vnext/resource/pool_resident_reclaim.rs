//! Replace redundant idle chunks at a pool ceiling without moving live backing.

use super::*;
use crate::vnext::DynamicPoolResidentPressure;

impl<R: DeviceRuntime> DynamicPoolSet<R> {
    /// Retire only whole idle chunks whose removal preserves the complete
    /// pending transaction's fresh packing need. Smaller companion claims
    /// remain protected while lane or graph-cache owners retain other chunks.
    pub(super) fn reclaim_idle_chunks_for_pool_resident_pressure(
        &self,
        pressure: &DynamicPoolResidentPressure,
        protected_immediate: &CapacityVector,
        protected_packing: &[DynamicBackingPackingEnvelope],
    ) -> Result<
        Option<(
            DynamicPoolRebalanceReceipt,
            DynamicPoolMaintenanceBoundaryReceipt,
        )>,
        VNextError,
    > {
        let pool = self
            .pools
            .get(pressure.pool_id())
            .ok_or_else(|| invalid_resource("pool-resident reclaim references an unknown pool"))?;
        let maximum = pool.domain.pool.provisioning().maximum_resident_bytes();
        if maximum != pressure.maximum_resident_bytes() {
            return Err(invalid_resource(
                "pool-resident reclaim pressure differs from its compiled ceiling",
            ));
        }
        if !matches!(
            pool.domain.pool.compatibility().profile().view(),
            DynamicStorageView::Contiguous
        ) {
            return Ok(None);
        }
        let mut envelopes = protected_packing
            .iter()
            .filter(|envelope| envelope.domain_id() == pool.domain.domain_id);
        let Some(envelope) = envelopes.next() else {
            return Ok(None);
        };
        if envelopes.next().is_some() || envelope.pool_id() != pool.domain.pool_id() {
            return Err(invalid_resource(
                "pool-resident reclaim has ambiguous packing protection",
            ));
        }
        let claims = envelope.claim_bytes_descending();
        let protected = protected_immediate
            .entries()
            .iter()
            .find(|entry| entry.domain() == pool.domain.domain_id)
            .map(|entry| entry.units().get())
            .unwrap_or(0);
        if protected != envelope.total_bytes()?
            || claims
                .iter()
                .any(|bytes| bytes % pool.allocation_quantum() != 0)
            || pressure.requested_bytes() > align_up_resource(protected, pool.allocation_quantum())?
        {
            return Err(invalid_resource(
                "pool-resident reclaim byte and packing protection diverged",
            ));
        }

        let capacity_availability = self.budget.availability_snapshot()?;
        let maintenance = pool
            .maintenance
            .lock()
            .map_err(|_| invalid_resource("dynamic pool maintenance authority is poisoned"))?;
        let mut state = pool
            .state
            .lock()
            .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
        let logical = self.logical_admission.snapshot()?;
        if state.poisoned || logical.poisoned() {
            return Err(invalid_resource("pool-resident reclaim is fail-closed"));
        }
        // A receipt from an earlier observation is not reclamation authority.
        // The caller will re-probe on the changed availability epoch.
        if state.resident_bytes != pressure.resident_bytes() || state.pending_growth_bytes != 0 {
            return Ok(None);
        }
        let growth =
            contiguous_packing_growth_bytes(&state.allocator, pool.domain.pool_id(), claims)?;
        if growth == 0
            || align_up_resource(growth, pool.allocation_quantum())? != pressure.requested_bytes()
        {
            return Ok(None);
        }
        let used = logical
            .domains()
            .iter()
            .find(|domain| domain.domain() == pool.domain.domain_id)
            .ok_or_else(|| invalid_resource("pool-resident reclaim has no logical domain"))?
            .used()
            .get();
        let occupied = state
            .resident_bytes
            .checked_sub(state.allocator.free_bytes)
            .ok_or_else(|| invalid_resource("dynamic pool free bytes exceed residency"))?;
        if occupied != state.live_occupancy.total().physical_bytes() {
            return Err(invalid_resource(
                "dynamic pool occupancy differs from its typed live ownership",
            ));
        }
        let coherent_floor = used
            .max(occupied)
            .checked_add(protected)
            .ok_or_else(|| invalid_resource("pool-resident runnable floor overflows"))?;
        let floor = coherent_floor.max(pool.domain.pool.provisioning().minimum_resident_bytes());
        let reclaimable = state.resident_bytes.saturating_sub(floor);
        let deficit = pressure.requested_bytes() - pressure.available_bytes();
        if reclaimable < deficit {
            return Ok(None);
        }
        let mut candidates = Vec::new();
        let mut boundary_chunks = Vec::with_capacity(state.chunks.len());
        for chunk in state.chunks.values() {
            let bytes = chunk.backing._grant.bytes();
            let identity = &chunk.backing.identity;
            let external_references = Arc::strong_count(&chunk.backing).saturating_sub(1);
            let full_extent_available = chunk.backing.descriptor.size_bytes == bytes
                && state
                    .allocator
                    .by_offset
                    .get(&(identity.ordinal(), 0))
                    .is_some_and(|extent| {
                        extent.chunk_generation == identity.generation()
                            && extent.length_bytes == bytes
                    });
            let resident_floor_allows_reclaim = bytes <= reclaimable;
            let physically_reclaimable = chunk.live_segments == 0
                && external_references == 0
                && full_extent_available
                && resident_floor_allows_reclaim;
            let protected_packing = physically_reclaimable
                && allocator_without_chunk_preserving_packing(
                    &state.allocator,
                    pool.domain.pool_id(),
                    claims,
                    growth,
                    identity,
                )?
                .is_none();
            let reclaim_candidate = physically_reclaimable && !protected_packing;
            boundary_chunks.push(DynamicPoolMaintenanceBoundaryChunk {
                identity: identity.clone(),
                bytes,
                live_segments: chunk.live_segments,
                external_references,
                protected_packing,
                full_extent_available,
                resident_floor_allows_reclaim,
                reclaim_candidate,
            });
            if reclaim_candidate {
                candidates.push((identity.clone(), bytes));
            }
        }
        let reclaim_candidate_chunks = candidates.len();
        let reclaim_candidate_bytes = candidates.iter().try_fold(0_u64, |sum, (_, bytes)| {
            sum.checked_add(*bytes)
                .ok_or_else(|| invalid_resource("reclaim candidate bytes overflow"))
        })?;
        candidates.sort_by_key(|(identity, bytes)| (*bytes, identity.ordinal()));
        let mut selected = Vec::new();
        let mut reclaimed = 0_u64;
        let mut after_allocator = state.allocator.clone();
        if let Some(candidate) = candidates.iter().find(|(_, bytes)| *bytes >= deficit) {
            selected.push(candidate.clone());
            reclaimed = candidate.1;
            after_allocator.remove_extent(candidate.0.ordinal(), 0)?;
        } else {
            for candidate in candidates.into_iter().rev() {
                if candidate.1 <= reclaimable - reclaimed {
                    let Some(next_allocator) = allocator_without_chunk_preserving_packing(
                        &after_allocator,
                        pool.domain.pool_id(),
                        claims,
                        growth,
                        &candidate.0,
                    )?
                    else {
                        continue;
                    };
                    after_allocator = next_allocator;
                    reclaimed += candidate.1;
                    selected.push(candidate);
                    if reclaimed >= deficit {
                        break;
                    }
                }
            }
        }
        if reclaimed < deficit {
            return Ok(None);
        }

        let boundary = DynamicPoolMaintenanceBoundaryReceipt {
            schema_version: DYNAMIC_POOL_MAINTENANCE_BOUNDARY_SCHEMA_VERSION,
            coordinator_id: logical.coordinator_id(),
            logical_release_epoch: logical.release_epoch(),
            logical_capacity_epoch: logical.capacity_epoch(),
            plan_device_capacity_epoch: capacity_availability.plan_epoch(),
            process_device_capacity_epoch: capacity_availability.process_epoch(),
            pressure: pressure.clone().into(),
            reclaim_attempted: true,
            planned_domains: vec![pool.domain.domain_id],
            protected_immediate: protected_immediate.clone(),
            protected_packing_envelopes: protected_packing.to_vec(),
            pools: vec![DynamicPoolMaintenanceBoundaryPool {
                pool_id: pool.domain.pool_id().clone(),
                domain_id: pool.domain.domain_id,
                excluded_from_reclaim: false,
                resident_bytes: state.resident_bytes,
                pending_growth_bytes: state.pending_growth_bytes,
                free_bytes: state.allocator.free_bytes,
                largest_contiguous_bytes: state.allocator.largest_contiguous_bytes(),
                free_extent_layout_fingerprint: free_extent_layout_fingerprint(&state.allocator),
                logical_used_bytes: used,
                live_occupancy: state.live_occupancy,
                minimum_resident_bytes: pool.domain.pool.provisioning().minimum_resident_bytes(),
                maximum_resident_bytes: maximum,
                protected_immediate_bytes: protected,
                protected_packing_satisfied: false,
                idle_target_replacement_kept_chunk: None,
                coherent_runnable_floor_bytes: coherent_floor,
                resident_floor_bytes: floor,
                reclaimable_bytes: reclaimable,
                chunks: boundary_chunks,
            }],
            reclaim_candidate_chunks,
            reclaim_candidate_bytes,
            selected_chunks: selected
                .iter()
                .map(|(identity, _)| identity.clone())
                .collect(),
            selected_bytes: reclaimed,
            reclaim_sufficient: true,
        };

        // Prove the selected subset cannot increase the fresh packing need
        // before changing either ledger. Live segments and externally pinned
        // chunk owners were excluded under the same state lock.
        if contiguous_packing_growth_bytes(&after_allocator, pool.domain.pool_id(), claims)?
            != growth
        {
            return Err(invalid_resource(
                "pool-resident reclaim changed its protected packing need",
            ));
        }
        let published = state.resident_bytes - reclaimed;
        let before_allocator = std::mem::replace(&mut state.allocator, after_allocator);
        let mut removed = Vec::with_capacity(selected.len());
        for (identity, _) in &selected {
            removed.push(
                state
                    .chunks
                    .remove(&identity.ordinal())
                    .expect("selected idle chunk remains locked"),
            );
        }
        let epochs = match self
            .logical_admission
            .set_domain_totals(&[(pool.domain.domain_id, CapacityUnits::new(published))])
        {
            Ok(epochs) => epochs,
            Err(error) => {
                state.allocator = before_allocator;
                for chunk in removed {
                    let ordinal = chunk.backing.identity.ordinal();
                    assert!(state.chunks.insert(ordinal, chunk).is_none());
                }
                return Err(error);
            }
        };
        state.resident_bytes = published;
        drop(state);
        drop(maintenance);
        // Backend destruction and grant release must run outside pool locks.
        drop(removed);
        let availability = self.budget.availability_snapshot()?;
        Ok(Some((
            DynamicPoolRebalanceReceipt {
                pools: vec![DynamicPoolIdleReclaim {
                    pool_id: pool.domain.pool_id().clone(),
                    chunks: selected
                        .iter()
                        .map(|(identity, _)| identity.clone())
                        .collect(),
                    reclaimed_bytes: reclaimed,
                    published_capacity_bytes: published,
                }],
                reclaimed_chunks: selected.len(),
                reclaimed_bytes: reclaimed,
                logical_capacity_epoch: epochs.capacity_epoch(),
                plan_device_capacity_epoch: availability.plan_epoch(),
                process_device_capacity_epoch: availability.process_epoch(),
            },
            boundary,
        )))
    }
}

/// Individual eligibility is insufficient: two redundant extents can each
/// hold a companion claim, so removing both may increase the next allocation.
/// Apply the same canonical transaction packing to every cumulative selection.
fn allocator_without_chunk_preserving_packing(
    allocator: &FreeExtentIndex,
    pool_id: &DynamicBackingPoolId,
    claims: &[u64],
    growth: u64,
    chunk: &BackingChunkIdentity,
) -> Result<Option<FreeExtentIndex>, VNextError> {
    let mut remaining = allocator.clone();
    remaining.remove_extent(chunk.ordinal(), 0)?;
    Ok(
        (contiguous_packing_growth_bytes(&remaining, pool_id, claims)? == growth)
            .then_some(remaining),
    )
}
