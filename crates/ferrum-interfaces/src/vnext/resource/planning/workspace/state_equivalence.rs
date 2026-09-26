use super::super::state_equivalence::same_items;
use super::*;

impl WorkspaceReadView {
    pub(in crate::vnext::resource::planning) fn same_future_state(
        &self,
        other: &Self,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<bool, ResourcePlanningUnknown> {
        poll(budget)?;
        if self.lane_id != other.lane_id
            || self.lane_epoch != other.lane_epoch
            || self.arena_clock != other.arena_clock
            || self.next_slot_id != other.next_slot_id
            || self.slots.len() != other.slots.len()
        {
            return Ok(false);
        }
        for (a, b) in self.slots.iter().zip(&other.slots) {
            poll(budget)?;
            if a.key != b.key
                || a.slot_id != b.slot_id
                || a.in_use != b.in_use
                || a.last_used != b.last_used
                || a.claims.len() != b.claims.len()
                || !same_items(a.projections.iter(), b.projections.iter(), budget)?
            {
                return Ok(false);
            }
            for (a, b) in a.claims.iter().zip(&b.claims) {
                poll(budget)?;
                if a.physical_size != b.physical_size
                    || a.pool_instance != b.pool_instance
                    || a.segment_generation != b.segment_generation
                    || a.identity.pool_id() != b.identity.pool_id()
                    || (!a.identity.shares_resource_id_storage(&b.identity)
                        && !same_items(
                            a.identity.resource_ids().iter(),
                            b.identity.resource_ids().iter(),
                            budget,
                        )?)
                    || !same_items(a.segments.iter(), b.segments.iter(), budget)?
                {
                    return Ok(false);
                }
            }
        }
        poll(budget)?;
        Ok(true)
    }
}
