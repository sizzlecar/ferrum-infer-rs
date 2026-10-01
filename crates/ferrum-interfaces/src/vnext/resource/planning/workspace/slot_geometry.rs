//! Verified immutable slot geometry; no device authority is retained.
//! Authority/projection vectors are private and never mutated. Only a fully
//! successful bounded read is retained. Every capture still reads dynamic slot
//! flags, lane liveness, limits and current pool state.
use super::*;
use crate::vnext::resource::lane_stable_arena::LaneStableArenaSlot;

pub(super) fn capture(
    key: &LaneStableArenaKey,
    slot: &LaneStableArenaSlot,
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Arc<SlotGeometry>, ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    poll(budget)?;
    if let Some(geometry) = slot.planning_geometry().get() {
        if geometry.bucket != key.reusable_execution_bucket_id {
            return Err(U::InvalidDemand);
        }
        return Ok(Arc::clone(geometry));
    }
    let mut segments = 0_usize;
    let mut claims = 0_usize;
    if slot.authorities().len() != slot.projection_bindings().len() {
        return Err(U::InvalidDemand);
    }
    let projections = slot.authorities().len();
    if projections > limits.maximum_descriptors {
        return Err(U::LimitExceeded);
    }
    let mut values = Vec::with_capacity(slot.authorities().len());
    let mut physical: Vec<ClaimReadView> = Vec::new();
    // Borrow only while the arena is locked. No lease reference or Arc
    // escapes into the numeric result.
    let mut claim_indices = BTreeMap::new();
    for (authority, binding) in slot.authorities().iter().zip(slot.projection_bindings()) {
        poll(budget)?;
        let evidence = authority.evidence();
        // Evidence segments describe this logical projection, not the
        // shared physical allocation. Borrow the real lease while the
        // arena is locked; retain only its numeric fields in the view.
        let backing = &authority.segment_lease;
        if evidence.initialization() != StateInitialization::None
            || evidence.reusable_execution_bucket_id() != Some(&key.reusable_execution_bucket_id)
            || backing.released
            || backing.owner_instance_id != evidence.pool_instance_id()
            || backing.owner.instance_id() != backing.owner_instance_id
            || backing.claim_identity.pool_id() != evidence.physical_claim_identity().pool_id()
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
        match backing
            .segments
            .range_matches_with_poll(
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
        let claim = if let Some(&(index, prior_backing)) = claim_indices.get(&binding.request_index)
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
            check_same_resource_ids(&prior.identity, evidence.physical_claim_identity(), budget)?;
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
            if segments > limits.maximum_free_extents || claims > limits.maximum_descriptors {
                return Err(U::LimitExceeded);
            }
            let index = physical.len();
            physical.push(ClaimReadView {
                identity: evidence.physical_claim_identity().clone(),
                physical_size: evidence.physical_size_bytes(),
                pool_instance: evidence.pool_instance_id(),
                segment_generation: Some(evidence.segment_generation()),
                segments: backing.segments.to_vec(),
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
    let geometry = Arc::new(SlotGeometry {
        bucket: key.reusable_execution_bucket_id.clone(),
        claims: physical,
        projections: values,
        segment_count: segments,
        identity_count: claims,
    });
    poll(budget)?;
    // The arena lock serializes capture; OnceLock prevents any replacement.
    let _ = slot.planning_geometry().set(Arc::clone(&geometry));
    Ok(geometry)
}
