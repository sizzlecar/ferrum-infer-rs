//! Explicit plan-owner retention or release of fully idle dynamic residency.
use super::*;
use crate::CapacitySnapshot;

/// A point-in-time reason no dynamic chunks were released. This is neither a
/// capacity permit nor permission to discard the owners that made trim busy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IdleDynamicReleaseBusy {
    LifecycleInUse,
    PoolMaintenance {
        pool_id: DynamicBackingPoolId,
    },
    DeferredCleanup {
        pending: usize,
    },
    LogicalAuthorities {
        snapshot: CapacitySnapshot,
    },
    Pool {
        pool_id: DynamicBackingPoolId,
        pending_growth_bytes: u64,
        live_segments: u64,
        externally_retained_chunks: usize,
        quarantined_chunks: usize,
    },
}

/// `Released` can contain zero bytes. The plan remains open and its static
/// weights are unchanged; the next execution uses ordinary backing admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "receipt", rename_all = "snake_case")]
pub enum IdleDynamicReleaseOutcome {
    Released(DynamicPoolRebalanceReceipt),
    Busy(IdleDynamicReleaseBusy),
}

/// An observation of backing kept by its original plan and capacity grants.
/// This receipt grants no execution authority and does not retain any owners.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IdleDynamicRetentionReceipt {
    retained_bytes: u64,
    retained_chunks: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    released: Option<DynamicPoolRebalanceReceipt>,
}

impl IdleDynamicRetentionReceipt {
    pub const fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    pub const fn retained_chunks(&self) -> usize {
        self.retained_chunks
    }

    pub fn released(&self) -> Option<&DynamicPoolRebalanceReceipt> {
        self.released.as_ref()
    }
}

/// An atomic idle observation followed by bounded retention or release.
/// Retention may include a receipt for other chunks released in the same call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "receipt", rename_all = "snake_case")]
pub enum IdleDynamicBackingOutcome {
    Released(DynamicPoolRebalanceReceipt),
    Retained(IdleDynamicRetentionReceipt),
    Busy(IdleDynamicReleaseBusy),
}

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    /// Release all dynamic backing only when this entire plan is idle.
    ///
    /// Unlike foreground pressure maintenance, this explicit owner operation
    /// may go below the compiled *runnable* minimum, down to zero. It does not
    /// close the root, mutate static storage, synchronize unknown work, or
    /// release quarantined owners. Busy is an all-or-nothing result.
    ///
    /// Lock order is lifecycle -> ordered pool maintenance -> ordered pool
    /// state -> logical coordinator. Native buffers and their existing device
    /// grants are dropped after all these locks have been released, as in the
    /// existing pressure-reclaim path. Backend buffers use that same Drop
    /// interface; this does not add or suppress a fallible native-free call.
    pub fn try_release_idle_dynamic_backing(
        &self,
    ) -> Result<IdleDynamicReleaseOutcome, VNextError> {
        match self.try_manage_idle_dynamic_backing(0, false)? {
            IdleDynamicBackingOutcome::Released(receipt) => {
                Ok(IdleDynamicReleaseOutcome::Released(receipt))
            }
            IdleDynamicBackingOutcome::Busy(reason) => Ok(IdleDynamicReleaseOutcome::Busy(reason)),
            IdleDynamicBackingOutcome::Retained(_) => Err(invalid_resource(
                "zero idle retention limit must release backing",
            )),
        }
    }

    /// Keep the original chunks and capacity grants only when the entire plan
    /// is idle and their total bytes fit a nonzero bound. Zero or an exceeded
    /// bound uses the same full release as `try_release_idle_dynamic_backing`.
    /// All idle/ownership checks and the decision share one exclusive lock
    /// scope. Retention changes neither capacity nor execution authority; the
    /// next execution must still pass ordinary admission.
    pub fn try_retain_idle_dynamic_backing(
        &self,
        maximum_bytes: u64,
    ) -> Result<IdleDynamicBackingOutcome, VNextError> {
        self.try_manage_idle_dynamic_backing(maximum_bytes, false)
    }

    /// Retain whole idle chunks within a byte cap, largest first. Equal sizes
    /// are ordered by pool ID and chunk ordinal; each chunk is kept if it fits
    /// the remaining cap. This is deterministic greedy selection, not packing
    /// or physical storage splitting. Original buffers and grants stay paired.
    /// Every pool must pass the same complete idle checks before any release.
    pub fn try_retain_largest_idle_dynamic_chunks(
        &self,
        maximum_bytes: u64,
    ) -> Result<IdleDynamicBackingOutcome, VNextError> {
        self.try_manage_idle_dynamic_backing(maximum_bytes, true)
    }

    fn try_manage_idle_dynamic_backing(
        &self,
        maximum_retained_bytes: u64,
        retain_partial: bool,
    ) -> Result<IdleDynamicBackingOutcome, VNextError> {
        let lifecycle = match self.lifecycle.try_write() {
            Ok(gate) => gate,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Ok(IdleDynamicBackingOutcome::Busy(
                    IdleDynamicReleaseBusy::LifecycleInUse,
                ));
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid_resource("plan runtime lifecycle gate is poisoned"));
            }
        };
        if self.phase.load(Ordering::Acquire) != super::plan_runtime::PLAN_RUNTIME_OPEN {
            return Err(invalid_resource(
                "closing plan runtime cannot release idle backing",
            ));
        }
        let cleanup = self.deferred_cleanup_status();
        if cleanup.pending() != 0 {
            return Ok(IdleDynamicBackingOutcome::Busy(
                IdleDynamicReleaseBusy::DeferredCleanup {
                    pending: cleanup.pending(),
                },
            ));
        }

        let pools = self.dynamic_pools.pools.values().collect::<Vec<_>>();
        let mut maintenance = Vec::with_capacity(pools.len());
        for pool in &pools {
            match pool.maintenance.try_lock() {
                Ok(gate) => maintenance.push(gate),
                Err(std::sync::TryLockError::WouldBlock) => {
                    return Ok(IdleDynamicBackingOutcome::Busy(
                        IdleDynamicReleaseBusy::PoolMaintenance {
                            pool_id: pool.domain.pool_id().clone(),
                        },
                    ));
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(invalid_resource(
                        "dynamic pool maintenance authority is poisoned",
                    ));
                }
            }
        }
        let mut states = pools
            .iter()
            .map(|pool| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let logical = self.dynamic_pools.logical_admission.snapshot()?;
        if logical.poisoned() || states.iter().any(|state| state.poisoned) {
            return Err(invalid_resource("idle backing release is fail-closed"));
        }
        if logical.active_requests() != 0
            || logical.active_sequences() != 0
            || logical.active_child_claims() != 0
            || logical.active_checkpoint_claims() != 0
            || logical.live_request_records() != 0
            || logical.live_sequence_records() != 0
            || logical
                .domains()
                .iter()
                .any(|domain| domain.used().get() != 0)
        {
            return Ok(IdleDynamicBackingOutcome::Busy(
                IdleDynamicReleaseBusy::LogicalAuthorities { snapshot: logical },
            ));
        }

        let mut candidates = Vec::new();
        let mut total_bytes = 0_u64;
        for (pool_index, (pool, state)) in pools.iter().zip(&states).enumerate() {
            let live_segments = state.chunks.values().try_fold(0_u64, |sum, chunk| {
                sum.checked_add(chunk.live_segments)
                    .ok_or_else(|| invalid_resource("idle backing segment count overflow"))
            })?;
            let externally_retained_chunks = state
                .chunks
                .values()
                .filter(|chunk| Arc::strong_count(&chunk.backing) != 1)
                .count();
            if state.pending_growth_bytes != 0
                || live_segments != 0
                || externally_retained_chunks != 0
                || !state.quarantined.is_empty()
            {
                return Ok(IdleDynamicBackingOutcome::Busy(
                    IdleDynamicReleaseBusy::Pool {
                        pool_id: pool.domain.pool_id().clone(),
                        pending_growth_bytes: state.pending_growth_bytes,
                        live_segments,
                        externally_retained_chunks,
                        quarantined_chunks: state.quarantined.len(),
                    },
                ));
            }
            let domain = logical
                .domains()
                .iter()
                .find(|domain| domain.domain() == pool.domain.domain_id)
                .ok_or_else(|| invalid_resource("idle pool has no logical domain"))?;
            if domain.total().get() != state.resident_bytes
                || state.allocator.free_bytes != state.resident_bytes
                || state.live_occupancy.total().physical_bytes() != 0
                || state.live_occupancy.total().segment_count() != 0
                || state.allocator.by_offset.len() != state.chunks.len()
                || state.allocator.by_size.len() != state.chunks.len()
            {
                return Err(invalid_resource(
                    "idle pool capacity/ownership disagrees with its allocator",
                ));
            }
            let mut bytes = 0_u64;
            for (&ordinal, chunk) in &state.chunks {
                let extent = state
                    .allocator
                    .by_offset
                    .get(&(ordinal, 0))
                    .ok_or_else(|| invalid_resource("idle chunk lacks its whole free extent"))?;
                let size = chunk.backing._grant.bytes();
                if chunk.backing.descriptor.size_bytes != size
                    || chunk.backing.identity.ordinal() != ordinal
                    || extent.chunk_generation != chunk.backing.identity.generation()
                    || extent.length_bytes != size
                    || !state.allocator.by_size.contains(&(
                        size,
                        ordinal,
                        extent.chunk_generation,
                        0,
                    ))
                {
                    return Err(invalid_resource(
                        "idle chunk extent differs from its owned backing",
                    ));
                }
                bytes = bytes
                    .checked_add(size)
                    .ok_or_else(|| invalid_resource("idle pool byte count overflow"))?;
                candidates.push((pool_index, ordinal, size));
            }
            if bytes != state.resident_bytes {
                return Err(invalid_resource(
                    "idle pool byte count differs from residency",
                ));
            }
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid_resource("idle release byte count overflow"))?;
        }

        if maximum_retained_bytes != 0 && total_bytes <= maximum_retained_bytes {
            return Ok(IdleDynamicBackingOutcome::Retained(
                IdleDynamicRetentionReceipt {
                    retained_bytes: total_bytes,
                    retained_chunks: candidates.len(),
                    released: None,
                },
            ));
        }

        let mut retained = BTreeSet::new();
        let mut retained_bytes = 0_u64;
        if retain_partial && maximum_retained_bytes != 0 {
            candidates.sort_unstable_by(|left, right| {
                right
                    .2
                    .cmp(&left.2)
                    .then(left.0.cmp(&right.0))
                    .then(left.1.cmp(&right.1))
            });
            for &(pool_index, ordinal, bytes) in &candidates {
                if bytes <= maximum_retained_bytes - retained_bytes {
                    retained.insert((pool_index, ordinal));
                    retained_bytes += bytes;
                }
            }
        }
        let released_bytes = total_bytes - retained_bytes;
        let released_chunks = candidates.len() - retained.len();
        let mut receipts = Vec::with_capacity(pools.len());
        let mut updates = Vec::with_capacity(pools.len());
        let mut release_ordinals = Vec::with_capacity(pools.len());
        for (pool_index, (pool, state)) in pools.iter().zip(&states).enumerate() {
            let mut chunks = Vec::new();
            let mut ordinals = Vec::new();
            let mut bytes = 0_u64;
            for (&ordinal, chunk) in &state.chunks {
                if !retained.contains(&(pool_index, ordinal)) {
                    bytes += chunk.backing._grant.bytes();
                    chunks.push(chunk.backing.identity.clone());
                    ordinals.push(ordinal);
                }
            }
            let remaining = state.resident_bytes - bytes;
            updates.push((pool.domain.domain_id, CapacityUnits::new(remaining)));
            release_ordinals.push(ordinals);
            if !chunks.is_empty() {
                receipts.push(DynamicPoolIdleReclaim {
                    pool_id: pool.domain.pool_id().clone(),
                    chunks,
                    reclaimed_bytes: bytes,
                    published_capacity_bytes: remaining,
                });
            }
        }

        // All validation/allocation precedes publication. A publication error
        // leaves every physical chunk and allocator unchanged. No fallible
        // pool mutation follows successful atomic publication.
        let mut removed = Vec::with_capacity(released_chunks);
        let epochs = self
            .dynamic_pools
            .logical_admission
            .set_domain_totals(&updates)?;
        for ((state, ordinals), (_, remaining)) in
            states.iter_mut().zip(release_ordinals).zip(&updates)
        {
            // Use directory mutation so its monotonic proof epoch survives;
            // replacing it with an empty directory would reset that identity.
            for ordinal in ordinals {
                state
                    .allocator
                    .remove_extent(ordinal, 0)
                    .expect("validated idle chunk retains its entire free extent");
                removed.push(
                    state
                        .chunks
                        .remove(&ordinal)
                        .expect("validated idle chunk remains under the exclusive state lock"),
                );
            }
            state.resident_bytes = remaining.get();
            // Directory epoch and ordinal/generation counters survive trim.
        }
        drop(states);
        drop(maintenance);
        drop(lifecycle);
        // Existing ResidentChunkBacking drops the native buffer before its
        // original DeviceCapacityGrant; no new accounting authority is made.
        drop(removed);
        let availability = self.dynamic_pools.budget.availability_snapshot()?;
        let released = DynamicPoolRebalanceReceipt {
            pools: receipts,
            reclaimed_chunks: released_chunks,
            reclaimed_bytes: released_bytes,
            logical_capacity_epoch: epochs.capacity_epoch(),
            plan_device_capacity_epoch: availability.plan_epoch(),
            process_device_capacity_epoch: availability.process_epoch(),
        };
        Ok(if retained.is_empty() {
            IdleDynamicBackingOutcome::Released(released)
        } else {
            IdleDynamicBackingOutcome::Retained(IdleDynamicRetentionReceipt {
                retained_bytes,
                retained_chunks: retained.len(),
                released: Some(released),
            })
        })
    }
}
