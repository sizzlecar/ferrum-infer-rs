//! Numeric copies of actual lane-stable slots. No authority or buffer is held.
use super::*;
use crate::vnext::resource::lane_stable_arena::{
    lane_stable_layout_key, LaneStableArenaKey, LaneStableArenaState,
};

mod slot_geometry;
mod state_equivalence;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClaimReadView {
    identity: PhysicalBackingClaimIdentity,
    physical_size: u64,
    pool_instance: u64,
    segment_generation: Option<u64>,
    segments: Vec<BackingSegment>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProjectionReadView {
    // Slot-local index into a numeric claim table, never resource authority.
    // Coalesced workspaces can project many values from the same allocation.
    claim: usize,
    resource: ResourceId,
    capacity: u64,
    physical_offset: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct SlotReadView {
    key: LaneStableArenaKey,
    slot_id: u64,
    in_use: bool,
    last_used: u64,
    geometry: Arc<SlotGeometry>,
}
impl std::ops::Deref for SlotReadView {
    type Target = SlotGeometry;
    fn deref(&self) -> &Self::Target {
        &self.geometry
    }
}

/// Numeric-only immutable slot shape. No leases, buffers, sessions or permits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::vnext::resource) struct SlotGeometry {
    bucket: ReusableExecutionBucketId,
    claims: Vec<ClaimReadView>,
    projections: Vec<ProjectionReadView>,
    segment_count: usize,
    identity_count: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WorkspaceReadView {
    lane_id: ExecutionLaneId,
    lane_epoch: u64,
    arena_clock: u64,
    next_slot_id: u64,
    slots: Vec<SlotReadView>,
}

pub(super) fn capture(
    arenas: &LaneStableArenaState,
    lane_id: ExecutionLaneId,
    lane_epoch: u64,
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<WorkspaceReadView, ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    if arenas.poisoned {
        return Err(U::BusyOrUnavailable);
    }
    if arenas.entries.len() > limits.maximum_descriptors {
        return Err(U::LimitExceeded);
    }
    let mut slots = Vec::new();
    let mut projections = 0_usize;
    let mut segments = 0_usize;
    let mut claims = 0_usize;
    for (key, entry) in &arenas.entries {
        poll(budget)?;
        if key.lane_id != lane_id {
            continue;
        }
        if entry.lane.strong_count() == 0 {
            return Err(U::StaleIdentity);
        }
        if slots
            .len()
            .checked_add(entry.slots.len())
            .is_none_or(|count| count > limits.maximum_descriptors)
        {
            return Err(U::LimitExceeded);
        }
        for slot in entry.slots.values() {
            poll(budget)?;
            let geometry = slot_geometry::capture(key, slot, limits, budget)?;
            projections = projections
                .checked_add(geometry.projections.len())
                .ok_or(U::LimitExceeded)?;
            segments = segments
                .checked_add(geometry.segment_count)
                .ok_or(U::LimitExceeded)?;
            claims = claims
                .checked_add(geometry.identity_count)
                .ok_or(U::LimitExceeded)?;
            if projections > limits.maximum_descriptors
                || claims > limits.maximum_descriptors
                || segments > limits.maximum_free_extents
            {
                return Err(U::LimitExceeded);
            }
            slots.push(SlotReadView {
                key: key.clone(),
                slot_id: slot.slot_id,
                in_use: slot.in_use,
                last_used: slot.last_used,
                geometry,
            });
        }
    }
    Ok(WorkspaceReadView {
        lane_id,
        lane_epoch,
        arena_clock: arenas.clock,
        next_slot_id: arenas.next_slot_id,
        slots,
    })
}

fn check_same_resource_ids(
    left: &PhysicalBackingClaimIdentity,
    right: &PhysicalBackingClaimIdentity,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    if left.shares_resource_id_storage(right) {
        return Ok(());
    }
    check_same_items(left.resource_ids(), right.resource_ids(), budget)
}

fn check_same_items<T: PartialEq>(
    left: &[T],
    right: &[T],
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    if left.len() != right.len() {
        return Err(ResourcePlanningUnknown::InvalidDemand);
    }
    for (left, right) in left.iter().zip(right) {
        poll(budget)?;
        if left != right {
            return Err(ResourcePlanningUnknown::InvalidDemand);
        }
    }
    Ok(())
}

// The table is private to one immutable slot and one canonical request slice.
// A match is always established by complete claim identity, never an index or
// hash alone. Coalesced projections reuse that match while their own resource,
// range, capacity and logical demand are still checked separately below.
fn validate_idle_projections(
    claims: &[ClaimReadView],
    projections: &[ProjectionReadView],
    canonical: &[&EvaluatedBackingRequest<'_>],
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    // Capture/project already bound the number of claims by maximum_descriptors.
    // No reference escapes this call or pins a physical resource/owner/epoch.
    let mut request_indices = vec![None; claims.len()];
    for projection in projections {
        poll(budget)?;
        let claim = claims.get(projection.claim).ok_or(U::InvalidDemand)?;
        if request_indices[projection.claim].is_none() {
            let index = canonical
                .binary_search_by(|request| request.claim_identity.cmp(&claim.identity))
                .map_err(|_| U::InvalidDemand)?;
            let projection_indices =
                projection_lookup_indices(&canonical[index].projections, budget)?;
            request_indices[projection.claim] = Some((index, projection_indices));
        }
        let (index, projection_indices) = request_indices[projection.claim]
            .as_ref()
            .ok_or(U::InvalidDemand)?;
        let request = canonical[*index];
        let matched_index = match projection_indices {
            None => request.projections.binary_search_by(|candidate| {
                candidate
                    .descriptor
                    .base_resource_id()
                    .cmp(&projection.resource)
            }),
            Some(indices) => indices
                .binary_search_by(|&index| {
                    request.projections[index]
                        .descriptor
                        .base_resource_id()
                        .cmp(&projection.resource)
                })
                .map(|index| indices[index]),
        }
        .map_err(|_| U::InvalidDemand)?;
        let matched = &request.projections[matched_index];
        if request.capacity_size_bytes != claim.physical_size
            || matched.physical_offset_bytes != projection.physical_offset
            || matched.capacity_size_bytes != projection.capacity
            || matched.logical_size_bytes == 0
            || matched.logical_size_bytes > matched.capacity_size_bytes
        {
            return Err(U::InvalidDemand);
        }
    }
    Ok(())
}

// Production Step/Invocation projections preserve the immutable plan's strict
// ResourceId order. Verify it once per matched claim, then borrow that order.
// Other private callers may provide an equivalent permutation: build only a
// call-local numeric index for that case. No ID, descriptor or authority is
// copied, and the per-projection range/capacity checks remain above.
fn projection_lookup_indices(
    projections: &[crate::vnext::resource::dynamic_pool::EvaluatedBackingProjection<'_>],
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Option<Vec<usize>>, ResourcePlanningUnknown> {
    // Even empty or singleton inputs cannot bypass an expired budget.
    poll(budget)?;
    let mut ordered = true;
    for pair in projections.windows(2) {
        poll(budget)?;
        if pair[0].descriptor.base_resource_id() >= pair[1].descriptor.base_resource_id() {
            ordered = false;
            break;
        }
    }
    if ordered {
        return Ok(None);
    }
    let mut by_resource = BTreeMap::new();
    for (index, projection) in projections.iter().enumerate() {
        poll(budget)?;
        if by_resource
            .insert(projection.descriptor.base_resource_id(), index)
            .is_some()
        {
            return Err(ResourcePlanningUnknown::InvalidDemand);
        }
    }
    Ok(Some(by_resource.into_values().collect()))
}

/// Match the same first idle slot as LaneStableArenaEntry::claim_idle_slot.
/// A new slot uses real resident free extents and is retained after this wave.
pub(super) fn reserve(
    state: &mut ResourcePlanningState,
    requests: &[EvaluatedBackingRequest<'_>],
    lifetime: AllocationLifetime,
    compiled: Option<&crate::vnext::resource::lane_stable_arena::CompiledLaneStableLayout>,
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Option<LaneStableArenaSlotIdentity>, ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    if requests.is_empty() {
        return Ok(None);
    }
    if requests.len() > limits.maximum_descriptors {
        return Err(U::LimitExceeded);
    }
    let mut new_projection_count = 0_usize;
    let mut new_identity_count = 0_usize;
    for request in requests {
        poll(budget)?;
        new_projection_count = new_projection_count
            .checked_add(request.projections.len())
            .ok_or(U::LimitExceeded)?;
        new_identity_count = new_identity_count
            .checked_add(request.claim_identity.resource_ids().len())
            .ok_or(U::LimitExceeded)?;
        if new_projection_count > limits.maximum_descriptors
            || new_identity_count > limits.maximum_descriptors
        {
            return Err(U::LimitExceeded);
        }
        for projection in &request.projections {
            poll(budget)?;
            if projection.descriptor.lifetime() != lifetime
                || projection.descriptor.initialization() != StateInitialization::None
            {
                return Err(U::InvalidDemand);
            }
        }
    }
    let mut canonical = requests.iter().collect::<Vec<_>>();
    canonical.sort_unstable_by(|a, b| a.claim_identity.cmp(&b.claim_identity));
    let workspace = state.workspace.as_ref().ok_or(U::ReusableExecution)?;
    poll(budget)?;
    let key = lane_stable_layout_key(workspace.lane_id, lifetime, &canonical, compiled)
        .map_err(|_| U::InvalidDemand)?;
    poll(budget)?;
    let mut idle: Option<&SlotReadView> = None;
    for slot in &workspace.slots {
        poll(budget)?;
        if slot.key == key
            && !slot.in_use
            && idle.is_none_or(|current| slot.slot_id < current.slot_id)
        {
            idle = Some(slot);
        }
    }
    if let Some(slot) = idle {
        if slot.projections.len() != new_projection_count {
            return Err(U::InvalidDemand);
        }
        validate_idle_projections(&slot.claims, &slot.projections, &canonical, budget)?;
        return Ok(Some(slot_identity(&slot.key, slot.slot_id)));
    }
    if workspace.slots.len() >= limits.maximum_descriptors {
        return Err(U::LimitExceeded);
    }
    let mut retained_projections = 0_usize;
    let mut retained_segments = 0_usize;
    let mut retained_identities = 0_usize;
    for slot in &workspace.slots {
        poll(budget)?;
        retained_projections = retained_projections
            .checked_add(slot.projections.len())
            .ok_or(U::LimitExceeded)?;
        for claim in &slot.claims {
            poll(budget)?;
            retained_segments = retained_segments
                .checked_add(claim.segments.len())
                .ok_or(U::LimitExceeded)?;
            retained_identities = retained_identities
                .checked_add(claim.identity.resource_ids().len())
                .ok_or(U::LimitExceeded)?;
        }
    }
    if retained_projections
        .checked_add(new_projection_count)
        .is_none_or(|count| count > limits.maximum_descriptors)
        || retained_identities
            .checked_add(new_identity_count)
            .is_none_or(|count| count > limits.maximum_descriptors)
    {
        return Err(U::LimitExceeded);
    }

    // Allocation order must be identical to the actual pool transaction; the
    // returned extents are owned only by this numeric state and never released
    // as transient Step/Invocation bytes.
    let allocated = super::project::reserve(state, requests, limits, budget)?;
    let mut ordered = requests.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.domain
            .pool_id()
            .cmp(b.domain.pool_id())
            .then_with(|| b.capacity_size_bytes.cmp(&a.capacity_size_bytes))
            .then_with(|| a.claim_identity.cmp(&b.claim_identity))
    });
    let mut projections = Vec::new();
    let mut claims = Vec::new();
    for (request, (pool_index, segments)) in ordered.into_iter().zip(allocated) {
        poll(budget)?;
        let pool = &state.pools[pool_index];
        retained_segments = retained_segments
            .checked_add(segments.len())
            .ok_or(U::LimitExceeded)?;
        if retained_segments > limits.maximum_free_extents {
            return Err(U::LimitExceeded);
        }
        let claim = claims.len();
        claims.push(ClaimReadView {
            identity: request.claim_identity.clone(),
            physical_size: request.capacity_size_bytes,
            pool_instance: pool.instance,
            segment_generation: None,
            segments,
        });
        for projection in &request.projections {
            poll(budget)?;
            if projections.len() >= limits.maximum_descriptors {
                return Err(U::LimitExceeded);
            }
            projections.push(ProjectionReadView {
                claim,
                resource: projection.descriptor.base_resource_id().clone(),
                capacity: projection.capacity_size_bytes,
                physical_offset: projection.physical_offset_bytes,
            });
        }
    }
    let workspace = Arc::make_mut(state.workspace.as_mut().ok_or(U::ReusableExecution)?);
    let slot_id = workspace.next_slot_id;
    workspace.next_slot_id = slot_id.checked_add(1).ok_or(U::LimitExceeded)?;
    let identity = slot_identity(&key, slot_id);
    workspace.slots.push(SlotReadView {
        key,
        slot_id,
        in_use: false,
        last_used: workspace.arena_clock,
        geometry: Arc::new(SlotGeometry {
            bucket: identity.reusable_execution_bucket_id().clone(),
            segment_count: claims.iter().map(|c| c.segments.len()).sum(),
            identity_count: claims.iter().map(|c| c.identity.resource_ids().len()).sum(),
            claims,
            projections,
        }),
    });
    Ok(Some(identity))
}

fn slot_identity(key: &LaneStableArenaKey, slot_id: u64) -> LaneStableArenaSlotIdentity {
    LaneStableArenaSlotIdentity::new(
        key.lane_id,
        key.lifetime,
        key.reusable_execution_bucket_id.clone(),
        key.layout_fingerprint.clone(),
        slot_id,
    )
}

pub(super) fn project_ranges(
    state: &ResourcePlanningState,
    selected: &LaneStableArenaSlotIdentity,
    physical: &physical_ranges::PhysicalRanges,
    proof: &mut ResourceCostRangeProof,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    let workspace = state.workspace.as_ref().ok_or(U::ReusableExecution)?;
    let mut found = None;
    for slot in &workspace.slots {
        poll(budget)?;
        if slot_identity(&slot.key, slot.slot_id) == *selected {
            found = Some(slot);
            break;
        }
    }
    let slot = found.ok_or(U::StaleIdentity)?;
    for projection in &slot.projections {
        poll(budget)?;
        let claim = slot.claims.get(projection.claim).ok_or(U::InvalidDemand)?;
        physical.insert_projection(
            proof,
            &projection.resource,
            &claim.segments,
            projection.physical_offset,
            projection.capacity,
        )?;
        proof.record_workspace_scope(&projection.resource, selected.lane_id())?;
    }
    Ok(())
}
