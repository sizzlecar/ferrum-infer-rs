//! Reuse checked prefill successors within one captured owner group.
//! The caller owns the view, roots, host policies and limits for this lifetime.
//! Joint and serial preparation retain separate paths. Prefix constraints first
//! affect decode, whose trajectories and observed queries stay independent.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum PrefillReuse {
    Share,
    #[cfg(test)]
    Replay,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrefillPath {
    Joint { completed_waves: u32 },
    Sequential,
}

struct PrefillState {
    width: usize,
    path: PrefillPath,
    state: Arc<ExecutionCostRouteState>,
}

pub(super) struct PrefillStates {
    entries: Vec<PrefillState>,
    maximum_entries: usize,
    maximum_active: usize,
}

impl PrefillStates {
    fn capacity(owners: usize, maximum_active: usize, reuse: PrefillReuse) -> usize {
        match reuse {
            PrefillReuse::Share => owners
                .saturating_mul(2)
                .min(maximum_active.saturating_sub(1)),
            #[cfg(test)]
            PrefillReuse::Replay => 0,
        }
    }

    /// Cache plus active states use one original vector's slots; successors
    /// use the other. Thus sharing adds no opaque payload to that two-vector
    /// peak. Reserve every Arc control, pointer and fixed cache record here.
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
                    .checked_mul(std::mem::size_of::<PrefillState>())?,
            )?
            .checked_add(std::mem::size_of::<Self>())
    }

    pub fn new(owners: usize, maximum_active: usize, reuse: PrefillReuse) -> GeometryResult<Self> {
        let maximum_entries = Self::capacity(owners, maximum_active, reuse);
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(maximum_entries)
            .map_err(|_| GeometryProjectionUnknown::Capacity)?;
        if entries.capacity() > maximum_entries {
            return Err(GeometryProjectionUnknown::Capacity);
        }
        Ok(Self {
            entries,
            maximum_entries,
            maximum_active,
        })
    }

    /// A target still executes its own wave. Reuse only an earlier successful
    /// joint target's successor, never an observed query or a later frontier.
    pub fn joint(
        &self,
        width: usize,
        target_wave: u32,
    ) -> Option<(u32, Arc<ExecutionCostRouteState>)> {
        self.entries.iter().find_map(|entry| match entry.path {
            PrefillPath::Joint { completed_waves }
                if entry.width == width && completed_waves <= target_wave =>
            {
                Some((completed_waves, Arc::clone(&entry.state)))
            }
            _ => None,
        })
    }

    /// Reuse a prefix of the original serial owner order only when each old
    /// owner's complete query sequence is identical at the requested width.
    /// Equal final prompt lengths alone do not establish this equivalence.
    pub fn sequential(
        &self,
        roots: &[&HostRoot],
        limits: &GeometryProjectionLimits,
    ) -> Option<(usize, Arc<ExecutionCostRouteState>)> {
        let chunk = prefill_chunk_for_width(
            limits.prefill_chunk,
            limits.prefill_row_ceiling,
            roots.len(),
        )?
        .get();
        self.entries
            .iter()
            .filter(|entry| entry.width <= roots.len())
            .filter(|entry| {
                let Some(old_chunk) = prefill_chunk_for_width(
                    limits.prefill_chunk,
                    limits.prefill_row_ceiling,
                    entry.width,
                ) else {
                    return false;
                };
                let old_chunk = old_chunk.get();
                let identical = roots
                    .iter()
                    .filter(|root| root.participant < entry.width)
                    .all(|root| old_chunk == chunk || root.prompt <= old_chunk.min(chunk));
                if !identical {
                    return false;
                }
                match entry.path {
                    PrefillPath::Sequential => true,
                    // Joint and serial queries are identical at width one;
                    // multi-owner joint states never seed serial preparation.
                    PrefillPath::Joint { completed_waves } if entry.width == 1 => roots
                        .iter()
                        .find(|root| root.participant == 0)
                        .is_some_and(|root| {
                            u64::from(completed_waves) * u64::from(old_chunk)
                                >= u64::from(root.prompt)
                        }),
                    PrefillPath::Joint { .. } => false,
                }
            })
            .max_by_key(|entry| entry.width)
            .map(|entry| (entry.width, Arc::clone(&entry.state)))
    }

    pub fn insert_joint(
        &mut self,
        width: usize,
        completed_waves: u32,
        state: &Arc<ExecutionCostRouteState>,
    ) {
        self.insert(width, PrefillPath::Joint { completed_waves }, state);
    }

    pub fn insert_sequential(&mut self, width: usize, state: &Arc<ExecutionCostRouteState>) {
        self.insert(width, PrefillPath::Sequential, state);
    }

    fn insert(&mut self, width: usize, path: PrefillPath, state: &Arc<ExecutionCostRouteState>) {
        if self.maximum_entries == 0 {
            return;
        }
        self.leave_room(1);
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.width == width
                && matches!(
                    (entry.path, path),
                    (PrefillPath::Joint { .. }, PrefillPath::Joint { .. })
                        | (PrefillPath::Sequential, PrefillPath::Sequential)
                )
        }) {
            self.entries.remove(index);
        }
        if self.entries.len() == self.maximum_entries {
            self.entries.remove(0);
        }
        self.entries.push(PrefillState {
            width,
            path,
            state: Arc::clone(state),
        });
    }

    /// Eviction affects work only. Branches are never dropped to retain cache.
    pub fn leave_room(&mut self, active: usize) {
        let remaining = self.maximum_active.saturating_sub(active);
        while self.entries.len() > remaining {
            self.entries.remove(0);
        }
    }
}
