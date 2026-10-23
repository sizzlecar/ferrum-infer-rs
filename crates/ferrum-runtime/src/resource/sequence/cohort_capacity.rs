//! Bounded observations of original persistent-pool owners, never admission permits.

use super::*;
use crate::{DynamicResourceDemand, DynamicStorageView, PlanRuntimeResources};

/// Work limits for a synchronous, try-lock-only capacity observation. These
/// resource guards bound inspection; they are not performance acceptance gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistentCapacityInspectionBudget {
    pub max_pools: usize,
    pub max_descriptors: usize,
    pub max_extents: usize,
    pub max_scratch_bytes: usize,
}

impl Default for PersistentCapacityInspectionBudget {
    fn default() -> Self {
        Self {
            max_pools: 64,
            max_descriptors: 4096,
            max_extents: 65_536,
            max_scratch_bytes: 1024 * 1024,
        }
    }
}

impl PersistentCapacityInspectionBudget {
    pub fn validate(self) -> Result<(), PersistentCapacityUnknown> {
        let limit = Self::default();
        if self.max_pools == 0
            || self.max_pools > limit.max_pools
            || self.max_descriptors == 0
            || self.max_descriptors > limit.max_descriptors
            || self.max_extents == 0
            || self.max_extents > limit.max_extents
            || self.max_scratch_bytes == 0
            || self.max_scratch_bytes > limit.max_scratch_bytes
        {
            return Err(PersistentCapacityUnknown::InspectionBudget);
        }
        Ok(())
    }
}

/// The observed reason support could not be established. None establishes a
/// permanent fit rejection or changes the original owner's admission policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistentCapacityUnknown {
    Unsupported,
    InvalidQuery,
    InspectionBudget,
    Contended,
    MembershipChanged,
    PendingOwner,
    UnsupportedPacking,
    InsufficientBacking,
    InvalidOwner,
    Arithmetic,
}

fn reserve<T>(
    out: &mut Vec<T>,
    count: usize,
    scratch: &mut usize,
) -> Result<(), PersistentCapacityUnknown> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(PersistentCapacityUnknown::Arithmetic)?;
    *scratch = scratch
        .checked_sub(bytes)
        .ok_or(PersistentCapacityUnknown::InspectionBudget)?;
    out.try_reserve_exact(count)
        .map_err(|_| PersistentCapacityUnknown::InspectionBudget)
}

fn visit(remaining: &mut usize, count: usize) -> Result<(), PersistentCapacityUnknown> {
    *remaining = remaining
        .checked_sub(count)
        .ok_or(PersistentCapacityUnknown::InspectionBudget)?;
    Ok(())
}

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    /// Prove a bounded-concurrency persistent envelope from actual original
    /// owners. Waiting members contribute full ceilings, but consume no claims.
    /// The fixed sequence-slot ceiling and real release/admission path remain
    /// authoritative. Release time and transient workspace are not predicted.
    ///
    /// This initial domain supports Sequence-only persistent pools, paged
    /// monotone token demand and homogeneous fixed contiguous claim cells.
    /// Foreign, prepared, retiring or checkpoint owners remain Unknown.
    pub fn persistent_capacity_for_cohort(
        self: &Arc<Self>,
        residents: &[(Arc<SequenceSession<R>>, u64)],
        ceilings: &[u64],
        max_live_sequences: usize,
        budget: PersistentCapacityInspectionBudget,
    ) -> Result<(), PersistentCapacityUnknown> {
        use PersistentCapacityUnknown as U;
        budget.validate()?;
        if ceilings.is_empty()
            || ceilings.len() > MAX_RESIDENT_CAPACITY_PARTICIPANTS
            || ceilings.contains(&0)
            || residents.len() > ceilings.len()
            || max_live_sequences == 0
            || max_live_sequences > MAX_RESIDENT_CAPACITY_PARTICIPANTS
            || residents.len() > max_live_sequences
        {
            return Err(U::InvalidQuery);
        }
        let _lifecycle = self.try_read_lifecycle().ok_or(U::Contended)?;
        if self.dynamic_pools.pools.len() > budget.max_pools {
            return Err(U::InspectionBudget);
        }
        let mut descriptors_left = budget.max_descriptors;
        for pool in self.dynamic_pools.pools.values() {
            visit(&mut descriptors_left, pool.domain.descriptors.len())?;
        }
        let mut scratch = budget
            .max_scratch_bytes
            .checked_sub(64 * std::mem::size_of::<u64>())
            .ok_or(U::InspectionBudget)?;
        let mut slots = Vec::new();
        let mut backings = Vec::new();
        reserve(&mut slots, residents.len(), &mut scratch)?;
        reserve(&mut backings, residents.len(), &mut scratch)?;
        let mut matched = [false; MAX_RESIDENT_CAPACITY_PARTICIPANTS];
        for (index, (session, ceiling)) in residents.iter().enumerate() {
            if !Arc::ptr_eq(self, &session.resources.request.plan.resources)
                || residents[..index]
                    .iter()
                    .any(|(other, _)| Arc::ptr_eq(&other.resources, &session.resources))
            {
                return Err(U::MembershipChanged);
            }
            let Some(position) = ceilings
                .iter()
                .enumerate()
                .position(|(i, value)| !matched[i] && value == ceiling)
            else {
                return Err(U::InvalidQuery);
            };
            matched[position] = true;
            let slot = session.slot.state.try_lock().map_err(|_| U::Contended)?;
            let SequenceSessionSlotState::Active(active) = &*slot else {
                return Err(U::PendingOwner);
            };
            if active.epoch != session.epoch
                || active.fingerprint != session.fingerprint
                || active.phase != SequenceSessionPhase::Open
                || active.active_frame.is_some()
                || active.has_participant_flights()
                || active.state_transfer.is_reserved()
                || sequence_dispatch_is_poisoned(&session.resources.sequence_dispatch_gate)
            {
                return Err(U::PendingOwner);
            }
            let backing = session
                .resources
                .backing_state
                .try_lock()
                .map_err(|_| U::Contended)?;
            let target = session.resources.request.work_shape().fit_shape();
            if target.sequences() != 1
                || target.tokens() != *ceiling
                || target.pages() != 0
                || backing.current.committed_tokens() > *ceiling
                || backing.current.committed_pages() != 0
            {
                return Err(U::MembershipChanged);
            }
            slots.push(slot);
            backings.push(backing);
        }

        // Deduplicate original physical leases, not logical projections of a
        // shared claim. Any outside owner remains visible in global occupancy.
        let mut lease_count = 0usize;
        for ((session, _), backing) in residents.iter().zip(&backings) {
            lease_count = lease_count
                .checked_add(backing.current.physical.backing_slices.len())
                .and_then(|n| n.checked_add(session.resources.request.backing_slices.len()))
                .ok_or(U::Arithmetic)?;
        }
        let mut visits_left = budget.max_extents;
        visit(&mut visits_left, lease_count)?;
        let mut leases = Vec::new();
        reserve(&mut leases, lease_count, &mut scratch)?;
        for ((session, _), backing) in residents.iter().zip(&backings) {
            for slice in backing
                .current
                .physical
                .backing_slices
                .iter()
                .chain(&session.resources.request.backing_slices)
            {
                leases.push(&slice.segment_lease);
            }
        }
        leases.sort_unstable_by_key(|lease| Arc::as_ptr(lease) as usize);
        leases.dedup_by(|a, b| Arc::ptr_eq(a, b));

        let mut states = Vec::new();
        reserve(&mut states, self.dynamic_pools.pools.len(), &mut scratch)?;
        for pool in self.dynamic_pools.pools.values() {
            let state = pool.state.try_lock().map_err(|_| U::Contended)?;
            if state.poisoned {
                return Err(U::InvalidOwner);
            }
            states.push(state);
        }
        // The snapshot allocation is bounded by the already checked domain count.
        let domain_bytes = self
            .dynamic_pools
            .pools
            .len()
            .checked_mul(std::mem::size_of::<crate::DomainCapacitySnapshot>())
            .ok_or(U::Arithmetic)?;
        if domain_bytes > scratch {
            return Err(U::InspectionBudget);
        }
        let logical = self
            .dynamic_pools
            .logical_admission
            .try_snapshot()
            .ok_or(U::Contended)?;
        if logical.poisoned() {
            return Err(U::InvalidOwner);
        }
        if logical.maximum_active_sequences() as usize != max_live_sequences
            || logical.active_sequences() as usize != residents.len()
            || logical.active_requests() as usize != residents.len()
            || logical.active_checkpoint_claims() != 0
        {
            return Err(U::PendingOwner);
        }

        for ((pool_id, pool), state) in self.dynamic_pools.pools.iter().zip(&states) {
            let has_persistent = pool.domain.descriptors.iter().any(|d| {
                matches!(
                    d.lifetime(),
                    AllocationLifetime::Request | AllocationLifetime::Sequence
                )
            });
            if !has_persistent {
                continue;
            }
            if pool
                .domain
                .descriptors
                .iter()
                .any(|d| d.lifetime() != AllocationLifetime::Sequence)
            {
                return Err(U::Unsupported);
            }
            if state.pending_growth_bytes != 0 || !state.quarantined.is_empty() {
                return Err(U::PendingOwner);
            }
            let mut demands = [0_u64; MAX_RESIDENT_CAPACITY_PARTICIPANTS];
            let mut fixed_cell = None;
            let paged = matches!(
                pool.domain.pool.compatibility().profile().view(),
                DynamicStorageView::PagedRegions { .. }
            );
            for descriptor in &pool.domain.descriptors {
                if matches!(
                    descriptor.demand(),
                    DynamicResourceDemand::Pages { .. }
                        | DynamicResourceDemand::BoundedShapeBuckets { .. }
                ) {
                    return Err(U::Unsupported);
                }
                if !paged {
                    if !matches!(descriptor.demand(), DynamicResourceDemand::Fixed { .. }) {
                        return Err(U::UnsupportedPacking);
                    }
                    let cell = descriptor
                        .minimum_request_bytes()
                        .map_err(|_| U::UnsupportedPacking)?;
                    if fixed_cell.is_some_and(|prior| prior != cell) {
                        return Err(U::UnsupportedPacking);
                    }
                    fixed_cell = Some(cell);
                }
                for (index, ceiling) in ceilings.iter().enumerate() {
                    let bytes = descriptor
                        .evaluate_request_bytes_for_shape(DynamicResourceShape::from_validated(
                            1, *ceiling, 0,
                        ))
                        .map_err(|_| U::Unsupported)?;
                    demands[index] = demands[index].checked_add(bytes).ok_or(U::Arithmetic)?;
                }
            }
            let quantum = fixed_cell.unwrap_or_else(|| pool.allocation_quantum());
            if quantum == 0 {
                return Err(U::InvalidOwner);
            }
            let mut actual_resident = 0_u64;
            visit(&mut visits_left, state.chunks.len())?;
            for chunk in state.chunks.values() {
                let bytes = chunk.backing.descriptor.size_bytes;
                if bytes == 0 || bytes % quantum != 0 {
                    return Err(U::UnsupportedPacking);
                }
                actual_resident = actual_resident.checked_add(bytes).ok_or(U::Arithmetic)?;
            }
            let mut actual_free = 0_u64;
            visit(&mut visits_left, state.allocator.by_offset.len())?;
            for ((_, offset), extent) in &state.allocator.by_offset {
                if offset % quantum != 0 || extent.length_bytes % quantum != 0 {
                    return Err(U::UnsupportedPacking);
                }
                actual_free = actual_free
                    .checked_add(extent.length_bytes)
                    .ok_or(U::Arithmetic)?;
            }
            let mut held = 0_u64;
            let mut held_segments = 0_u64;
            let mut held_claims = 0_u64;
            visit(&mut visits_left, leases.len())?;
            for lease in &leases {
                if lease.claim_identity.pool_id() != pool_id {
                    continue;
                }
                if lease.released
                    || lease.occupancy.residency
                        != crate::resource::DynamicBackingClaimResidency::Transient
                    || !matches!(
                        lease.occupancy.scope,
                        DynamicBackingClaimScope::Sequence
                            | DynamicBackingClaimScope::InitialSequenceBundle
                    )
                {
                    return Err(U::PendingOwner);
                }
                visit(&mut visits_left, lease.segments.len())?;
                for segment in &lease.segments {
                    if segment.offset_bytes() % quantum != 0
                        || segment.length_bytes() % quantum != 0
                    {
                        return Err(U::UnsupportedPacking);
                    }
                }
                held = held.checked_add(lease.size_bytes).ok_or(U::Arithmetic)?;
                held_segments = held_segments
                    .checked_add(lease.segments.len() as u64)
                    .ok_or(U::Arithmetic)?;
                held_claims = held_claims.checked_add(1).ok_or(U::Arithmetic)?;
            }
            let occupancy = state.live_occupancy.total();
            if actual_resident != state.resident_bytes
                || actual_free != state.allocator.free_bytes
                || actual_free.checked_add(held) != Some(actual_resident)
                || occupancy.physical_bytes() != held
                || occupancy.segment_count() != held_segments
                || occupancy.claim_count() != held_claims
            {
                return Err(U::PendingOwner);
            }
            let domain = logical
                .domains()
                .iter()
                .find(|d| d.domain() == pool.domain.domain_id())
                .ok_or(U::InvalidOwner)?;
            if domain.total().get() != actual_resident
                || domain.used().get() != held
                || domain.available().get() != actual_free
            {
                return Err(U::PendingOwner);
            }
            demands[..ceilings.len()].sort_unstable_by(|a, b| b.cmp(a));
            let required = demands[..ceilings.len().min(max_live_sequences)]
                .iter()
                .try_fold(0_u64, |sum, value| {
                    sum.checked_add(*value).ok_or(U::Arithmetic)
                })?;
            if required > actual_resident {
                return Err(U::InsufficientBacking);
            }
        }
        Ok(())
    }
}
