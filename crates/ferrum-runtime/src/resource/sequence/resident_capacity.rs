//! Read-only persistent fit of a complete cohort, using original live owners.

use super::*;
use crate::DynamicStorageView;

/// Bounds the synchronous owner inspection before allocating query scratch.
pub const MAX_RESIDENT_CAPACITY_PARTICIPANTS: usize = 64;

impl<R: DeviceRuntime> SequenceSession<R> {
    /// Checks all live sequences of this plan through their admitted request
    /// ceilings. Existing backing is counted once; only remaining paged growth
    /// consumes current free capacity. Request state is already held through the
    /// same ceiling. This neither reserves capacity nor grows a pool.
    ///
    /// Unknown/contended/incomplete observations return false. The result is a
    /// planning observation; actual extension retains its original checks.
    pub fn resident_cohort_persistent_capacity(sessions: &[Arc<Self>]) -> Result<bool, VNextError> {
        if sessions.is_empty() || sessions.len() > MAX_RESIDENT_CAPACITY_PARTICIPANTS {
            return Ok(false);
        }
        let plan = &sessions[0].resources.request.plan;
        let Some(_lifecycle) = plan.resources.try_read_lifecycle() else {
            return Ok(false);
        };
        for (index, session) in sessions.iter().enumerate() {
            if !Arc::ptr_eq(&plan.resources, &session.resources.request.plan.resources)
                || sessions[..index]
                    .iter()
                    .any(|other| Arc::ptr_eq(&other.resources, &session.resources))
            {
                return Ok(false);
            }
        }

        // Try-lock the original session/backing cells. Holding them prevents a
        // hybrid observation of one incarnation before and after extension.
        let mut slots = Vec::with_capacity(sessions.len());
        let mut backings = Vec::with_capacity(sessions.len());
        for session in sessions {
            let Ok(slot) = session.slot.state.try_lock() else {
                return Ok(false);
            };
            let SequenceSessionSlotState::Active(active) = &*slot else {
                return Ok(false);
            };
            if active.epoch != session.epoch
                || active.fingerprint != session.fingerprint
                || active.phase != SequenceSessionPhase::Open
                || active.active_frame.is_some()
                || active.has_participant_flights()
                || active.state_transfer.is_reserved()
                || sequence_dispatch_is_poisoned(&session.resources.sequence_dispatch_gate)
            {
                return Ok(false);
            }
            let Ok(backing) = session.resources.backing_state.try_lock() else {
                return Ok(false);
            };
            slots.push(slot);
            backings.push(backing);
        }

        let mut required = BTreeMap::new();
        for (session, backing) in sessions.iter().zip(&backings) {
            let current = &backing.current;
            let target = session.resources.request.work_shape().fit_shape();
            if target.sequences() != 1
                || current.committed_tokens() > target.tokens()
                || current.committed_pages() > target.pages()
            {
                return Ok(false);
            }
            let (_, requests) = match plan.sequence_extension_demand_with_coverage(
                current.committed_shape(),
                target,
                target,
                &current.physical.owned_bytes,
                SequenceBackingGrowthPolicy::Exact,
                AdmissionPressureAction::WaitForRelease,
            ) {
                Ok(result) => result,
                // Nonpaged growth and unsupported descriptor coverage must not
                // be reinterpreted as an empty/zero-cost extension.
                Err(_) => return Ok(false),
            };
            for request in requests {
                let bytes = required
                    .entry(request.domain.pool_id().clone())
                    .or_insert(0_u64);
                *bytes = bytes
                    .checked_add(request.capacity_size_bytes)
                    .ok_or_else(|| {
                        invalid_resource("resident cohort persistent demand overflows u64")
                    })?;
            }
        }

        let pools = plan.dynamic_pools();
        // Lock all pools, including fixed recurrent/request pools, in canonical
        // order. This observes existing allocations and remaining free extents
        // together; pending device growth is never credited.
        let mut pool_states = Vec::with_capacity(pools.pools.len());
        for pool in pools.pools.values() {
            let Ok(state) = pool.state.try_lock() else {
                return Ok(false);
            };
            if state.poisoned {
                return Ok(false);
            }
            pool_states.push(state);
        }
        let Some(logical) = plan.logical_admission().try_snapshot() else {
            return Ok(false);
        };
        if logical.poisoned()
            || logical.active_sequences() as usize != sessions.len()
            || logical.active_requests() as usize != sessions.len()
        {
            return Ok(false);
        }
        for ((pool_id, pool), state) in pools.pools.iter().zip(&pool_states) {
            let bytes = required.remove(pool_id).unwrap_or(0);
            if bytes == 0 {
                continue;
            }
            // The actual paged allocator guarantees aligned aggregate free
            // capacity suffices; contiguous allocators require a packing proof
            // and are deliberately unsupported for remaining growth here.
            let DynamicStorageView::PagedRegions { block_bytes } =
                pool.domain.pool.compatibility().profile().view()
            else {
                return Ok(false);
            };
            if bytes % block_bytes != 0
                || bytes % pool.allocation_quantum() != 0
                || bytes > state.allocator.free_bytes
                || !logical.domains().iter().any(|domain| {
                    domain.domain() == pool.domain.domain_id() && domain.available().get() >= bytes
                })
            {
                return Ok(false);
            }
        }
        Ok(required.is_empty())
    }
}
