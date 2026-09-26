//! Numeric copies of actual lane-stable slots. No authority or buffer is held.
use super::*;
use crate::vnext::resource::backing_extent::backing_segment_range_matches_with_poll;
use crate::vnext::resource::lane_stable_arena::{
    lane_stable_layout_key, LaneStableArenaKey, LaneStableArenaState,
};

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
    claims: Vec<ClaimReadView>,
    projections: Vec<ProjectionReadView>,
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
            if slot.authorities.len() != slot.projection_bindings.len() {
                return Err(U::InvalidDemand);
            }
            projections = projections
                .checked_add(slot.authorities.len())
                .ok_or(U::LimitExceeded)?;
            if projections > limits.maximum_descriptors {
                return Err(U::LimitExceeded);
            }
            let mut values = Vec::with_capacity(slot.authorities.len());
            let mut physical: Vec<ClaimReadView> = Vec::new();
            // Borrow only while the arena is locked. No lease reference or Arc
            // escapes into the numeric result.
            let mut claim_indices = BTreeMap::new();
            for (authority, binding) in slot.authorities.iter().zip(&slot.projection_bindings) {
                poll(budget)?;
                let evidence = authority.evidence();
                // Evidence segments describe this logical projection, not the
                // shared physical allocation. Borrow the real lease while the
                // arena is locked; retain only its numeric fields in the view.
                let backing = &authority.segment_lease;
                if evidence.initialization() != StateInitialization::None
                    || evidence.reusable_execution_bucket_id()
                        != Some(&key.reusable_execution_bucket_id)
                    || backing.released
                    || backing.owner_instance_id != evidence.pool_instance_id()
                    || backing.owner.instance_id() != backing.owner_instance_id
                    || backing.claim_identity.pool_id()
                        != evidence.physical_claim_identity().pool_id()
                    || backing.segment_generation != evidence.segment_generation()
                    || backing.size_bytes != evidence.physical_size_bytes()
                    || evidence
                        .physical_offset_bytes()
                        .checked_add(evidence.capacity_size_bytes())
                        .is_none_or(|end| end > backing.size_bytes)
                {
                    return Err(U::InvalidDemand);
                }
                check_same_resource_ids(
                    &backing.claim_identity,
                    evidence.physical_claim_identity(),
                    budget,
                )?;
                match backing_segment_range_matches_with_poll(
                    &backing.segments,
                    evidence.physical_offset_bytes(),
                    evidence.capacity_size_bytes(),
                    evidence.segments(),
                    || budget.has_budget(),
                )
                .map_err(|_| U::InvalidDemand)?
                {
                    Some(true) => {}
                    Some(false) => return Err(U::InvalidDemand),
                    None => return Err(U::BudgetExhausted),
                }
                let claim = if let Some(&(index, prior_backing)) =
                    claim_indices.get(&binding.request_index)
                {
                    let prior: &ClaimReadView = &physical[index];
                    if prior.identity.pool_id() != evidence.physical_claim_identity().pool_id()
                        || prior.physical_size != evidence.physical_size_bytes()
                        || prior.pool_instance != evidence.pool_instance_id()
                        || prior.segment_generation != Some(evidence.segment_generation())
                    {
                        return Err(U::InvalidDemand);
                    }
                    // A request index alone is not evidence. Shared immutable
                    // identity and the exact lease permit constant-time checks;
                    // independently supplied evidence is still checked in full.
                    check_same_resource_ids(
                        &prior.identity,
                        evidence.physical_claim_identity(),
                        budget,
                    )?;
                    if !Arc::ptr_eq(prior_backing, backing) {
                        check_same_items(&prior.segments, &backing.segments, budget)?;
                    }
                    index
                } else {
                    segments = segments
                        .checked_add(backing.segments.len())
                        .ok_or(U::LimitExceeded)?;
                    claims = claims
                        .checked_add(evidence.physical_claim_identity().resource_ids().len())
                        .ok_or(U::LimitExceeded)?;
                    if segments > limits.maximum_free_extents || claims > limits.maximum_descriptors
                    {
                        return Err(U::LimitExceeded);
                    }
                    let index = physical.len();
                    physical.push(ClaimReadView {
                        identity: evidence.physical_claim_identity().clone(),
                        physical_size: evidence.physical_size_bytes(),
                        pool_instance: evidence.pool_instance_id(),
                        segment_generation: Some(evidence.segment_generation()),
                        segments: backing.segments.clone(),
                    });
                    claim_indices.insert(binding.request_index, (index, backing));
                    index
                };
                values.push(ProjectionReadView {
                    claim,
                    resource: evidence.resource_id().clone(),
                    capacity: evidence.capacity_size_bytes(),
                    physical_offset: evidence.physical_offset_bytes(),
                });
            }
            slots.push(SlotReadView {
                key: key.clone(),
                slot_id: slot.slot_id,
                in_use: slot.in_use,
                last_used: slot.last_used,
                claims: physical,
                projections: values,
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
        let index = match request_indices[projection.claim] {
            Some(index) => index,
            None => {
                let index = canonical
                    .binary_search_by(|request| request.claim_identity.cmp(&claim.identity))
                    .map_err(|_| U::InvalidDemand)?;
                request_indices[projection.claim] = Some(index);
                index
            }
        };
        let request = canonical[index];
        let matched = request
            .projections
            .iter()
            .find(|candidate| candidate.descriptor.base_resource_id() == &projection.resource)
            .ok_or(U::InvalidDemand)?;
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

/// Match the same first idle slot as LaneStableArenaEntry::claim_idle_slot.
/// A new slot uses real resident free extents and is retained after this wave.
pub(super) fn reserve(
    state: &mut ResourcePlanningState,
    requests: &[EvaluatedBackingRequest<'_>],
    lifetime: AllocationLifetime,
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
    let key = lane_stable_layout_key(workspace.lane_id, lifetime, &canonical)
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
    let workspace = state.workspace.as_mut().ok_or(U::ReusableExecution)?;
    let slot_id = workspace.next_slot_id;
    workspace.next_slot_id = slot_id.checked_add(1).ok_or(U::LimitExceeded)?;
    let identity = slot_identity(&key, slot_id);
    workspace.slots.push(SlotReadView {
        key,
        slot_id,
        in_use: false,
        last_used: workspace.arena_clock,
        claims,
        projections,
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
