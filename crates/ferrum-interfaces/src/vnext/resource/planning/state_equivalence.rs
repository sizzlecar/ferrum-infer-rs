//! Exact, interruptible equality of private numerical successors. Equality
//! never transfers live resource authority or crosses a capture fence.
use super::*;

pub(super) fn same_items<I>(
    left: I,
    right: I,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<bool, ResourcePlanningUnknown>
where
    I: ExactSizeIterator,
    I::Item: PartialEq,
{
    poll(budget)?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for (a, b) in left.zip(right) {
        poll(budget)?;
        if a != b {
            return Ok(false);
        }
    }
    poll(budget)?;
    Ok(true)
}

impl ResourcePlanningState {
    pub(crate) fn same_future_state(
        &self,
        other: &Self,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<bool, ResourcePlanningUnknown> {
        poll(budget)?;
        if !Arc::ptr_eq(&self.fence, &other.fence)
            || self.waves != other.waves
            || self.pools.len() != other.pools.len()
            || self.sequence_ranges.len() != other.sequence_ranges.len()
        {
            return Ok(false);
        }
        for (a, b) in self.pools.iter().zip(&other.pools) {
            poll(budget)?;
            if a.id != b.id
                || a.instance != b.instance
                || a.next_extent_generation != b.next_extent_generation
                || a.resident_bytes != b.resident_bytes
                || a.allocator.free_bytes != b.allocator.free_bytes
                || a.allocator.search_probes != b.allocator.search_probes
                || !same_items(
                    a.allocator.by_offset.iter(),
                    b.allocator.by_offset.iter(),
                    budget,
                )?
                || !same_items(
                    a.allocator.by_size.iter(),
                    b.allocator.by_size.iter(),
                    budget,
                )?
            {
                return Ok(false);
            }
        }
        match (&self.workspace, &other.workspace) {
            (Some(a), Some(b)) if a.same_future_state(b, budget)? => {}
            (None, None) => {}
            _ => return Ok(false),
        }
        if !same_items(
            self.logical_available.iter(),
            other.logical_available.iter(),
            budget,
        )? || !same_items(self.covered.iter(), other.covered.iter(), budget)?
        {
            return Ok(false);
        }
        for (a, b) in self.sequence_ranges.iter().zip(&other.sequence_ranges) {
            poll(budget)?;
            // These maps and segment arrays are immutable while shared.
            if Arc::ptr_eq(a, b) {
                continue;
            }
            if a.len() != b.len() {
                return Ok(false);
            }
            for ((a_id, a_segments), (b_id, b_segments)) in a.iter().zip(b.iter()) {
                poll(budget)?;
                if a_id != b_id
                    || (!Arc::ptr_eq(a_segments, b_segments)
                        && !same_items(a_segments.iter(), b_segments.iter(), budget)?)
                {
                    return Ok(false);
                }
            }
        }
        poll(budget)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests;
