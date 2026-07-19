use super::dynamic_pool::{DynamicDeviceCapacityBlocked, DynamicPoolGrowthIntent, DynamicPoolSet};
use super::{
    invalid_resource, AdmissionDeferred, CapacityEpochs, CapacityVector, CapacityWaitCondition,
    DeviceRuntime, DynamicBackingBlocker, DynamicBackingDeferred, DynamicBackingPoolId,
    DynamicChunkQuarantineReason, DynamicPoolGrowthBatchReceipt, DynamicPoolGrowthReceipt,
    DynamicPoolGrowthRequest, DynamicPoolStatus, VNextError,
};
use crate::vnext::{
    CapacityShortfallKind, CapacityWaitSnapshot, DeferredAction, DynamicBackingPressure,
    DynamicPoolResidentPressure,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolMaintenanceStatus {
    epochs: CapacityEpochs,
    device_capacity_bytes: u64,
    effective_device_usable_ceiling_bytes: u64,
    process_claimed_bytes: u64,
    budget_device_wide_usable_ceiling_bytes: u64,
    budget_claimed_bytes: u64,
    pools: Vec<DynamicPoolStatus>,
}

impl DynamicPoolMaintenanceStatus {
    pub const fn epochs(&self) -> CapacityEpochs {
        self.epochs
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    pub const fn effective_device_usable_ceiling_bytes(&self) -> u64 {
        self.effective_device_usable_ceiling_bytes
    }

    pub const fn process_claimed_bytes(&self) -> u64 {
        self.process_claimed_bytes
    }

    pub const fn budget_device_wide_usable_ceiling_bytes(&self) -> u64 {
        self.budget_device_wide_usable_ceiling_bytes
    }

    pub const fn budget_claimed_bytes(&self) -> u64 {
        self.budget_claimed_bytes
    }

    pub fn pools(&self) -> &[DynamicPoolStatus] {
        &self.pools
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DynamicDeferredMaintenanceOutcome {
    RetryAdmission {
        current_epochs: CapacityEpochs,
    },
    WaitForRelease {
        current_epochs: CapacityEpochs,
        wait_condition: CapacityWaitCondition,
        pressure: DynamicBackingPressure,
    },
    Maintained(DynamicPoolGrowthBatchReceipt),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolQuarantineRelease {
    pool_id: DynamicBackingPoolId,
    released_chunks: usize,
    released_bytes: u64,
}

impl DynamicPoolQuarantineRelease {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn released_chunks(&self) -> usize {
        self.released_chunks
    }

    pub const fn released_bytes(&self) -> u64 {
        self.released_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolQuarantineReleaseReceipt {
    pools: Vec<DynamicPoolQuarantineRelease>,
    released_chunks: usize,
    released_bytes: u64,
}

impl DynamicPoolQuarantineReleaseReceipt {
    pub fn pools(&self) -> &[DynamicPoolQuarantineRelease] {
        &self.pools
    }

    pub const fn released_chunks(&self) -> usize {
        self.released_chunks
    }

    pub const fn released_bytes(&self) -> u64 {
        self.released_bytes
    }
}

/// Plan-owner capability for changing physical dynamic-pool residency. It is
/// created once during provisioning, is intentionally not `Clone`, and cannot
/// be derived from any request, sequence, step, invocation, or static lease.
///
/// ```compile_fail
/// use ferrum_interfaces::vnext::{AdmittedRequestResources, DeviceRuntime};
/// fn request_cannot_mint<R: DeviceRuntime>(request: &AdmittedRequestResources<R>) {
///     let _ = request.dynamic_pool_maintenance_controller();
/// }
/// ```
///
/// ```compile_fail
/// use ferrum_interfaces::vnext::{DeviceRuntime, StaticProvisioningLease};
/// fn lease_cannot_mint<R: DeviceRuntime>(lease: &StaticProvisioningLease<R>) {
///     let _ = lease.dynamic_pool_maintenance_controller();
/// }
/// ```
#[must_use = "dynamic pool maintenance controller must be retained by the plan owner"]
pub struct DynamicPoolMaintenanceController<R>
where
    R: DeviceRuntime,
{
    pools: Arc<DynamicPoolSet<R>>,
}

impl<R> DynamicPoolMaintenanceController<R>
where
    R: DeviceRuntime,
{
    pub(in crate::vnext::resource) fn new(pools: Arc<DynamicPoolSet<R>>) -> Self {
        Self { pools }
    }

    pub fn pool_ids(&self) -> impl ExactSizeIterator<Item = &DynamicBackingPoolId> {
        self.pools.pools.keys()
    }

    /// Returns one exact domain-to-pool and physical-capacity snapshot. The
    /// process-wide usable ceiling is the conservative minimum across all live
    /// plan budgets; plan claims remain separately visible.
    pub fn status(&self) -> Result<DynamicPoolMaintenanceStatus, VNextError> {
        let mut pools = Vec::with_capacity(self.pools.pools.len());
        for pool in self.pools.pools.values() {
            let state = pool
                .state
                .lock()
                .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
            let live_segments = state.chunks.values().try_fold(0_u64, |total, chunk| {
                total
                    .checked_add(chunk.live_segments)
                    .ok_or_else(|| invalid_resource("dynamic live segment count overflows u64"))
            })?;
            let quarantined_bytes = state.quarantined.iter().try_fold(0_u64, |total, chunk| {
                total
                    .checked_add(chunk.backing._grant.bytes())
                    .ok_or_else(|| invalid_resource("dynamic quarantine bytes overflow u64"))
            })?;
            pools.push(DynamicPoolStatus {
                pool_id: pool.domain.pool_id().clone(),
                domain_id: pool.domain.domain_id,
                storage_profile: pool.domain.pool.compatibility().profile(),
                resident_bytes: state.resident_bytes,
                pending_growth_bytes: state.pending_growth_bytes,
                free_bytes: state.allocator.free_bytes,
                largest_contiguous_bytes: state.allocator.largest_contiguous_bytes(),
                resident_chunks: state.chunks.len(),
                live_segments,
                quarantined_chunks: state.quarantined.len(),
                quarantined_bytes,
                descriptor_mismatch_chunks: state
                    .quarantined
                    .iter()
                    .filter(|chunk| {
                        chunk.reason == DynamicChunkQuarantineReason::DescriptorMismatch
                    })
                    .count(),
                publication_rejected_chunks: state
                    .quarantined
                    .iter()
                    .filter(|chunk| {
                        chunk.reason == DynamicChunkQuarantineReason::PublicationRejected
                    })
                    .count(),
                poisoned: state.poisoned,
            });
        }
        let account = &self.pools.budget.account;
        let state = account
            .state
            .lock()
            .map_err(|_| invalid_resource("device capacity account is poisoned"))?;
        let effective_device_usable_ceiling_bytes = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .ok_or_else(|| invalid_resource("device capacity account has no live budget"))?;
        let budget_claimed_bytes = state
            .budgets
            .get(&self.pools.budget.budget_id)
            .ok_or_else(|| invalid_resource("dynamic pool plan budget is stale"))?
            .claimed_bytes;
        Ok(DynamicPoolMaintenanceStatus {
            epochs: self.pools.logical_admission.epochs()?,
            device_capacity_bytes: account.device_capacity_bytes,
            effective_device_usable_ceiling_bytes,
            process_claimed_bytes: state.claimed_bytes,
            budget_device_wide_usable_ceiling_bytes: self
                .pools
                .budget
                .device_wide_usable_ceiling_bytes,
            budget_claimed_bytes,
            pools,
        })
    }

    /// Makes the core-derived runnable minimum resident. A second call is a
    /// no-op and returns `None`; it never creates a duplicate initial chunk.
    pub fn initialize_pool(
        &self,
        pool_id: &DynamicBackingPoolId,
    ) -> Result<Option<DynamicPoolGrowthReceipt>, VNextError> {
        let mut receipt = self
            .pools
            .maintain_pools(vec![DynamicPoolGrowthIntent::Minimum(pool_id.clone())])?;
        Ok(receipt.growths.pop())
    }

    /// Initializes several pools in one canonical all-or-nothing publication.
    pub fn initialize_pools(
        &self,
        pool_ids: &[DynamicBackingPoolId],
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        self.pools.maintain_pools(
            pool_ids
                .iter()
                .cloned()
                .map(DynamicPoolGrowthIntent::Minimum)
                .collect(),
        )
    }

    /// Adds one explicitly-sized resident chunk. The physical allocation is
    /// globally reserved before the runtime is called and published only after
    /// allocation, descriptor validation, and installation all succeed.
    pub fn grow_pool(
        &self,
        pool_id: &DynamicBackingPoolId,
        requested_bytes: u64,
    ) -> Result<DynamicPoolGrowthReceipt, VNextError> {
        let request = DynamicPoolGrowthRequest::new(pool_id.clone(), requested_bytes)?;
        let mut receipt = self.grow_pools(vec![request])?;
        receipt
            .growths
            .pop()
            .ok_or_else(|| invalid_resource("single-pool growth produced no receipt"))
    }

    /// Grows distinct pools under one global reservation and one capacity
    /// epoch publication. Input order is ignored; duplicate pools are rejected.
    pub fn grow_pools(
        &self,
        requests: Vec<DynamicPoolGrowthRequest>,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        self.pools.maintain_pools(
            requests
                .into_iter()
                .map(DynamicPoolGrowthIntent::Additional)
                .collect(),
        )
    }

    fn wait_snapshot_for_pool_ids<'a>(
        &self,
        pool_ids: impl IntoIterator<Item = &'a DynamicBackingPoolId>,
    ) -> Result<CapacityWaitSnapshot, VNextError> {
        self.pools.logical_admission.wait_snapshot_for_domains(
            pool_ids
                .into_iter()
                .map(|pool_id| {
                    self.pools
                        .pools
                        .get(pool_id)
                        .map(|pool| pool.domain.domain_id)
                        .ok_or_else(|| {
                            invalid_resource("dynamic maintenance references an unknown pool")
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    fn capacity_wait_outcome(
        &self,
        logical_snapshot: CapacityWaitSnapshot,
        blocked: DynamicDeviceCapacityBlocked,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let logical_snapshot = logical_snapshot.narrow_to_domains(blocked.planned_domains)?;
        let mut observed = logical_snapshot.wait_condition().observed().to_vec();
        observed.push(blocked.availability.epoch_for_pressure(&blocked.pressure));
        let wait_condition = CapacityWaitCondition::new(
            logical_snapshot.wait_condition().coordinator_id(),
            observed,
        )?;
        Ok(DynamicDeferredMaintenanceOutcome::WaitForRelease {
            current_epochs: logical_snapshot.epochs(),
            wait_condition,
            pressure: blocked.pressure.into(),
        })
    }

    fn pool_resident_wait_outcome(
        &self,
        logical_snapshot: CapacityWaitSnapshot,
        pressure: DynamicPoolResidentPressure,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let domain = self
            .pools
            .pools
            .get(pressure.pool_id())
            .map(|pool| pool.domain.domain_id)
            .ok_or_else(|| {
                invalid_resource("dynamic pool resident pressure references an unknown pool")
            })?;
        let logical_snapshot = logical_snapshot.narrow_to_domains(vec![domain])?;
        Ok(DynamicDeferredMaintenanceOutcome::WaitForRelease {
            current_epochs: logical_snapshot.epochs(),
            wait_condition: logical_snapshot.wait_condition().clone(),
            pressure: pressure.into(),
        })
    }

    fn maintain_deferred_pools(
        &self,
        intents: Vec<DynamicPoolGrowthIntent>,
        capacity_blocked: &mut Option<DynamicDeviceCapacityBlocked>,
        protected_immediate: &CapacityVector,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        let retry_intents = intents.clone();
        match self
            .pools
            .maintain_pools_observed(intents, capacity_blocked)
        {
            Err(VNextError::DeviceCapacityUnavailable(pressure)) => {
                let planned_domains = capacity_blocked
                    .as_ref()
                    .expect("typed capacity failure retains its exact observation")
                    .planned_domains
                    .clone();
                let Some(rebalance) = self.pools.reclaim_idle_chunks_for_pressure(
                    &pressure,
                    &planned_domains,
                    protected_immediate,
                )?
                else {
                    return Err(VNextError::DeviceCapacityUnavailable(pressure));
                };
                *capacity_blocked = None;
                let mut receipt = self
                    .pools
                    .maintain_pools_observed(retry_intents, capacity_blocked)?;
                receipt.rebalance = Some(rebalance);
                Ok(receipt)
            }
            outcome => outcome,
        }
    }

    /// Revalidates a physical deferral while its exact typed owner remains
    /// live. Unrelated capacity epochs may advance between admission and this
    /// bounded maintenance attempt, so growth is recomputed from current pool
    /// state instead of trusting the stale byte snapshot.
    pub(super) fn maintain_for_live_deferred(
        &self,
        deferred: &DynamicBackingDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        self.maintain_for_live_deferred_protecting(deferred, deferred.protected_immediate())
    }

    /// Maintains one uncommitted multi-scope bundle without reclaiming another
    /// pool below the exact immediate envelope required by the same bundle.
    pub(super) fn maintain_for_live_deferred_protecting(
        &self,
        deferred: &DynamicBackingDeferred,
        protected_immediate: &CapacityVector,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let coordinator_id = self.pools.logical_admission.id();
        if coordinator_id != deferred.epochs().coordinator_id()
            || coordinator_id != deferred.wait_condition().coordinator_id()
        {
            return Err(invalid_resource(
                "dynamic backing deferral belongs to another admission coordinator",
            ));
        }
        if deferred.blockers().is_empty() {
            return Err(invalid_resource(
                "dynamic backing deferral contains no blocking pool",
            ));
        }
        for required in deferred.protected_immediate().entries() {
            let protected = protected_immediate
                .entries()
                .iter()
                .find(|entry| entry.domain() == required.domain())
                .map(|entry| entry.units().get())
                .unwrap_or(0);
            if protected < required.units().get() {
                return Err(invalid_resource(
                    "dynamic bundle protection is smaller than its physical deferral",
                ));
            }
        }
        let logical_snapshot = self.wait_snapshot_for_pool_ids(
            deferred
                .blockers()
                .iter()
                .map(DynamicBackingBlocker::pool_id),
        )?;
        let mut capacity_blocked = None;
        let growth = self.maintain_deferred_pools(
            deferred
                .blockers()
                .iter()
                .cloned()
                .map(DynamicPoolGrowthIntent::RevalidatedDeferral)
                .collect(),
            &mut capacity_blocked,
            protected_immediate,
        );
        match growth {
            Ok(receipt) if receipt.growths().is_empty() => {
                let current_epochs = self.pools.logical_admission.epochs()?;
                if current_epochs == deferred.epochs() {
                    return Err(invalid_resource(
                        "dynamic backing maintenance made no progress on an unchanged deferral",
                    ));
                }
                Ok(DynamicDeferredMaintenanceOutcome::RetryAdmission { current_epochs })
            }
            Ok(receipt) => Ok(DynamicDeferredMaintenanceOutcome::Maintained(receipt)),
            Err(VNextError::DeviceCapacityUnavailable(_)) => self.capacity_wait_outcome(
                logical_snapshot,
                capacity_blocked.expect("typed capacity failure retains its exact observation"),
            ),
            Err(VNextError::DynamicPoolResidentUnavailable(pressure)) => {
                self.pool_resident_wait_outcome(logical_snapshot, pressure)
            }
            Err(error) => Err(error),
        }
    }

    /// Materializes fit capacity that logical admission identified as
    /// growable but that immediate backing preparation does not yet claim.
    pub fn maintain_for_admission_deferred(
        &self,
        deferred: &AdmissionDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let coordinator_id = self.pools.logical_admission.id();
        if coordinator_id != deferred.epochs().coordinator_id()
            || coordinator_id != deferred.wait_condition().coordinator_id()
        {
            return Err(invalid_resource(
                "logical admission deferral belongs to another coordinator",
            ));
        }
        if deferred.action() != DeferredAction::AwaitBackingGrowth {
            return Err(invalid_resource(
                "logical admission deferral does not request backing growth",
            ));
        }
        let pools_by_domain = self
            .pools
            .pools
            .values()
            .map(|pool| (pool.domain.domain_id, pool.domain.pool_id().clone()))
            .collect::<BTreeMap<_, _>>();
        let current = self.pools.logical_admission.snapshot()?;
        let mut requested_by_pool = BTreeMap::<DynamicBackingPoolId, u64>::new();
        for blocker in deferred
            .blockers()
            .iter()
            .filter(|blocker| blocker.kind() == CapacityShortfallKind::BackingGrowthRequired)
        {
            let domain = blocker.domain().ok_or_else(|| {
                invalid_resource("backing-growth blocker contains no capacity domain")
            })?;
            let pool_id = pools_by_domain.get(&domain).ok_or_else(|| {
                invalid_resource("backing-growth blocker references a non-pool domain")
            })?;
            let current_total = current
                .domains()
                .iter()
                .find(|snapshot| snapshot.domain() == domain)
                .ok_or_else(|| {
                    invalid_resource("backing-growth blocker references an unknown domain")
                })?
                .total()
                .get();
            let missing = blocker.requested().get().saturating_sub(current_total);
            if missing == 0 {
                continue;
            }
            requested_by_pool
                .entry(pool_id.clone())
                .and_modify(|bytes| *bytes = (*bytes).max(missing))
                .or_insert(missing);
        }
        if requested_by_pool.is_empty() {
            return Ok(DynamicDeferredMaintenanceOutcome::RetryAdmission {
                current_epochs: self.pools.logical_admission.epochs()?,
            });
        }
        let requests = requested_by_pool
            .into_iter()
            .map(|(pool_id, bytes)| DynamicPoolGrowthRequest::new(pool_id, bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let logical_snapshot =
            self.wait_snapshot_for_pool_ids(requests.iter().map(|request| request.pool_id()))?;
        let mut capacity_blocked = None;
        let growth = self.maintain_deferred_pools(
            requests
                .into_iter()
                .map(DynamicPoolGrowthIntent::Additional)
                .collect(),
            &mut capacity_blocked,
            deferred.immediate_requested(),
        );
        match growth {
            Ok(receipt) => Ok(DynamicDeferredMaintenanceOutcome::Maintained(receipt)),
            Err(VNextError::DeviceCapacityUnavailable(_)) => self.capacity_wait_outcome(
                logical_snapshot,
                capacity_blocked.expect("typed capacity failure retains its exact observation"),
            ),
            Err(VNextError::DynamicPoolResidentUnavailable(pressure)) => {
                self.pool_resident_wait_outcome(logical_snapshot, pressure)
            }
            Err(error) => Err(error),
        }
    }

    /// Explicitly releases only unclaimable quarantined chunks. Resident
    /// chunks and logical totals are unchanged; the returned grants are
    /// dropped after all pool locks have been released.
    pub fn release_quarantined_chunks(
        &self,
    ) -> Result<DynamicPoolQuarantineReleaseReceipt, VNextError> {
        let pools = self.pools.pools.values().cloned().collect::<Vec<_>>();
        let _maintenance = pools
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
        let pool_totals = states
            .iter()
            .map(|state| {
                let bytes = state.quarantined.iter().try_fold(0_u64, |total, chunk| {
                    total
                        .checked_add(chunk.backing._grant.bytes())
                        .ok_or_else(|| invalid_resource("released quarantine bytes overflow u64"))
                })?;
                Ok((state.quarantined.len(), bytes))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let released_chunks = pool_totals.iter().try_fold(0_usize, |total, (count, _)| {
            total
                .checked_add(*count)
                .ok_or_else(|| invalid_resource("released quarantine count overflows usize"))
        })?;
        let released_bytes = pool_totals.iter().try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| invalid_resource("released quarantine bytes overflow u64"))
        })?;
        let mut released = Vec::with_capacity(pools.len());
        let mut receipts = Vec::new();
        for ((pool, state), (pool_chunks, pool_bytes)) in
            pools.iter().zip(states.iter_mut()).zip(pool_totals)
        {
            if pool_chunks == 0 {
                continue;
            }
            let chunks = std::mem::take(&mut state.quarantined);
            debug_assert_eq!(chunks.len(), pool_chunks);
            receipts.push(DynamicPoolQuarantineRelease {
                pool_id: pool.domain.pool_id().clone(),
                released_chunks: pool_chunks,
                released_bytes: pool_bytes,
            });
            released.push(chunks);
        }
        drop(states);
        drop(released);
        Ok(DynamicPoolQuarantineReleaseReceipt {
            pools: receipts,
            released_chunks,
            released_bytes,
        })
    }
}
