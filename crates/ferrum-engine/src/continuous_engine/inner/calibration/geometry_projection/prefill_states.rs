//! Reuse only completed sequential prefill within one captured owner group.
//! The caller owns the view, roots, host policies and chunk for this lifetime.
//! Prefix constraints first affect decode, whose trajectories stay independent.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum PrefillReuse {
    Share,
    #[cfg(test)]
    Replay,
}

pub(super) struct PrefillStates {
    entries: Vec<(usize, Arc<ExecutionCostRouteState>)>,
    maximum_entries: usize,
    maximum_active: usize,
}

impl PrefillStates {
    fn capacity(owners: usize, maximum_active: usize, reuse: PrefillReuse) -> usize {
        match reuse {
            PrefillReuse::Share => owners.min(maximum_active.saturating_sub(1)),
            #[cfg(test)]
            PrefillReuse::Replay => 0,
        }
    }

    /// Opaque state payload obeys the existing ResourcePlanningLimits. Cache
    /// plus active states use one original vector's slots; successors use the
    /// other. Thus sharing adds no opaque state beyond the old two-vector peak.
    /// Reserve the new Arc controls/pointers and cache headers separately.
    pub fn overhead_bytes(
        owners: usize,
        maximum_active: usize,
        reuse: PrefillReuse,
    ) -> Option<usize> {
        let states = maximum_active.checked_mul(2)?.checked_add(1)?;
        states
            .checked_mul(
                2 * std::mem::size_of::<usize>()
                    + std::mem::size_of::<Arc<ExecutionCostRouteState>>(),
            )?
            .checked_add(
                Self::capacity(owners, maximum_active, reuse)
                    .checked_mul(std::mem::size_of::<(usize, Arc<ExecutionCostRouteState>)>())?,
            )?
            .checked_add(std::mem::size_of::<Self>())
    }

    pub fn new(owners: usize, maximum_active: usize, reuse: PrefillReuse) -> GeometryResult<Self> {
        let maximum_entries = Self::capacity(owners, maximum_active, reuse);
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(maximum_entries)
            .map_err(|_| GeometryProjectionUnknown::Capacity)?;
        // Never retain an allocator-expanded cache beyond its authorized bound.
        if entries.capacity() > maximum_entries {
            return Err(GeometryProjectionUnknown::Capacity);
        }
        Ok(Self {
            entries,
            maximum_entries,
            maximum_active,
        })
    }

    pub fn get(&self, width: usize) -> Option<Arc<ExecutionCostRouteState>> {
        self.entries
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, state)| Arc::clone(state))
    }

    pub fn insert(&mut self, width: usize, state: &Arc<ExecutionCostRouteState>) {
        if self.maximum_entries == 0 {
            return;
        }
        self.leave_room(1);
        if self.entries.len() == self.maximum_entries {
            self.entries.remove(0);
        }
        self.entries.push((width, Arc::clone(state)));
    }

    /// Eviction affects work only: the next miss replays from the same view.
    /// No branch or prefix result is dropped to retain a cached prefill.
    pub fn leave_room(&mut self, active: usize) {
        let remaining = self.maximum_active.saturating_sub(active);
        while self.entries.len() > remaining {
            self.entries.remove(0);
        }
    }
}
