//! Weak, non-authorizing identity observations across successful permits.
use super::super::BackingSegmentLease;
use super::*;
use crate::vnext::ExecutionLaneId;
use serde::Serialize;
use std::{cmp::Ordering, fmt, sync::Weak};

struct OwnerIdentity {
    lease: Weak<BackingSegmentLease>,
    chunk: Weak<dyn Send + Sync>,
    lane: Option<ExecutionLaneId>,
    lifetime: AllocationLifetime,
}

impl OwnerIdentity {
    fn key(&self) -> (AllocationLifetime, Option<ExecutionLaneId>, usize, usize) {
        // The Weak control blocks remain allocated for the whole comparison,
        // preventing address reuse. These addresses never escape this module.
        (
            self.lifetime,
            self.lane,
            self.lease.as_ptr() as usize,
            self.chunk.as_ptr().cast::<()>() as usize,
        )
    }
}

struct Entry {
    owner: OwnerIdentity,
    offset: u64,
    length: u64,
    logical_size: u64,
    resource_view: usize,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct ResourceView {
    resource: ResourceId,
    committed_logical_size: u64,
    current_view_size: u64,
    windows: Vec<(u64, u64)>,
}

/// Diagnostic comparison values only. Equality does not authorize reuse,
/// validate a future authority, or retain any physical lease or device buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SegmentBackingOwnerContinuity {
    pub lifetime: AllocationLifetime,
    pub previous_owners: u64,
    pub current_owners: u64,
    pub shared_owners: u64,
    pub added_owners: u64,
    pub removed_owners: u64,
    pub owner_set_equal: bool,
    pub previous_ranges: u64,
    pub current_ranges: u64,
    pub shared_ranges: u64,
    pub range_set_equal: bool,
    pub shared_owners_with_changed_ranges: u64,
    pub previous_logical_windows: u64,
    pub current_logical_windows: u64,
    pub shared_logical_windows: u64,
    pub logical_window_set_equal: bool,
    pub shared_owners_with_changed_logical_windows: u64,
}

/// Opaque diagnostic metadata with weak object identities. No upgrade,
/// authority constructor, address formatter or serialized owner is exposed.
/// Weak references keep control blocks allocated but allow the actual lease,
/// buffer and capacity grant to drop normally.
pub struct SegmentBackingOwnerSnapshot {
    entries: Vec<Entry>,
    views: Vec<ResourceView>,
    owners: Vec<usize>,
    ranges: Vec<usize>,
    logical_windows: Vec<usize>,
}

impl fmt::Debug for SegmentBackingOwnerSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentBackingOwnerSnapshot")
            .field("owners", &self.owners.len())
            .field("ranges", &self.ranges.len())
            .field("logical_windows", &self.logical_windows.len())
            .finish()
    }
}

impl SegmentBackingOwnerSnapshot {
    pub fn len(&self) -> usize {
        self.owners.len()
    }
    pub fn is_empty(&self) -> bool {
        self.owners.is_empty()
    }

    /// Compares exact observed control-block identities, ranges and logical
    /// views. The caller separately qualifies program, lane epoch and actual
    /// adjacent successful attempt; this method grants no execution capability.
    pub fn compare(&self, previous: &Self) -> [SegmentBackingOwnerContinuity; 5] {
        [
            AllocationLifetime::Plan,
            AllocationLifetime::Request,
            AllocationLifetime::Sequence,
            AllocationLifetime::Step,
            AllocationLifetime::Invocation,
        ]
        .map(|lifetime| {
            let (po, co, so) =
                self.count_sets(previous, &previous.owners, &self.owners, lifetime, 0);
            let (pr, cr, sr) =
                self.count_sets(previous, &previous.ranges, &self.ranges, lifetime, 1);
            let (pw, cw, sw) = self.count_sets(
                previous,
                &previous.logical_windows,
                &self.logical_windows,
                lifetime,
                2,
            );
            SegmentBackingOwnerContinuity {
                lifetime,
                previous_owners: po,
                current_owners: co,
                shared_owners: so,
                added_owners: co - so,
                removed_owners: po - so,
                owner_set_equal: po == co && co == so,
                previous_ranges: pr,
                current_ranges: cr,
                shared_ranges: sr,
                range_set_equal: pr == cr && cr == sr,
                shared_owners_with_changed_ranges: self
                    .changed_shared_owners(previous, lifetime, 1),
                previous_logical_windows: pw,
                current_logical_windows: cw,
                shared_logical_windows: sw,
                logical_window_set_equal: pw == cw && cw == sw,
                shared_owners_with_changed_logical_windows: self
                    .changed_shared_owners(previous, lifetime, 2),
            }
        })
    }

    fn cmp_entry(&self, a: usize, other: &Self, b: usize, kind: u8) -> Ordering {
        let a = &self.entries[a];
        let b = &other.entries[b];
        a.owner.key().cmp(&b.owner.key()).then_with(|| match kind {
            1 => (a.offset, a.length).cmp(&(b.offset, b.length)),
            2 => a
                .logical_size
                .cmp(&b.logical_size)
                .then_with(|| self.views[a.resource_view].cmp(&other.views[b.resource_view])),
            _ => Ordering::Equal,
        })
    }

    fn count_sets(
        &self,
        previous: &Self,
        p: &[usize],
        c: &[usize],
        lifetime: AllocationLifetime,
        kind: u8,
    ) -> (u64, u64, u64) {
        let mut p = p
            .iter()
            .copied()
            .filter(|&i| previous.entries[i].owner.lifetime == lifetime)
            .peekable();
        let mut c = c
            .iter()
            .copied()
            .filter(|&i| self.entries[i].owner.lifetime == lifetime)
            .peekable();
        let mut counts = (0_u64, 0_u64, 0_u64);
        while let (Some(&pi), Some(&ci)) = (p.peek(), c.peek()) {
            match previous.cmp_entry(pi, self, ci, kind) {
                Ordering::Less => {
                    p.next();
                    counts.0 += 1;
                }
                Ordering::Greater => {
                    c.next();
                    counts.1 += 1;
                }
                Ordering::Equal => {
                    p.next();
                    c.next();
                    counts.0 += 1;
                    counts.1 += 1;
                    counts.2 += 1;
                }
            }
        }
        counts.0 += p.count() as u64;
        counts.1 += c.count() as u64;
        counts
    }

    fn changed_shared_owners(
        &self,
        previous: &Self,
        lifetime: AllocationLifetime,
        kind: u8,
    ) -> u64 {
        let p = if kind == 1 {
            &previous.ranges
        } else {
            &previous.logical_windows
        };
        let c = if kind == 1 {
            &self.ranges
        } else {
            &self.logical_windows
        };
        let mut changed = 0;
        for &owner in &self.owners {
            let key = self.entries[owner].owner.key();
            if key.0 != lifetime {
                continue;
            }
            let pb = p.partition_point(|&i| previous.entries[i].owner.key() < key);
            let pe = p.partition_point(|&i| previous.entries[i].owner.key() <= key);
            if pb == pe {
                continue;
            }
            let cb = c.partition_point(|&i| self.entries[i].owner.key() < key);
            let ce = c.partition_point(|&i| self.entries[i].owner.key() <= key);
            if pe - pb != ce - cb
                || p[pb..pe]
                    .iter()
                    .zip(&c[cb..ce])
                    .any(|(&a, &b)| previous.cmp_entry(a, self, b, kind) != Ordering::Equal)
            {
                changed += 1;
            }
        }
        changed
    }

    pub(super) fn capture<R: DeviceRuntime, S: SubmissionWaveDispatchTimingSink>(
        pools: &DynamicPoolSet<R>,
        groups: &[&[LogicalBackingSliceAuthority]],
        batch: &SegmentBackingBatch<R::Buffer>,
        windows: SegmentBackingWindowInput<'_>,
    ) -> Option<Self> {
        if !S::ENABLED {
            return None;
        }
        let mut snapshot = Self {
            entries: Vec::new(),
            views: Vec::new(),
            owners: Vec::new(),
            ranges: Vec::new(),
            logical_windows: Vec::new(),
        };
        let count = batch
            .resources
            .iter()
            .try_fold(0usize, |n, r| n.checked_add(r.bindings.len()))?;
        snapshot.entries.try_reserve_exact(count).ok()?;
        snapshot.views.try_reserve_exact(groups.len()).ok()?;
        for (group_index, group) in groups.iter().enumerate() {
            let first = group.first()?;
            let resource = batch.resources.get(group_index)?;
            let pool = pools.pools.get(first.evidence.pool_id())?;
            let lifetime = pool
                .domain
                .descriptors
                .iter()
                .find(|d| d.base_resource_id() == first.evidence.resource_id())?
                .lifetime();
            let view = ResourceView {
                resource: first.evidence.resource_id().clone(),
                committed_logical_size: resource.logical_size_bytes,
                current_view_size: resource.view_size_bytes,
                windows: Vec::new(),
            };
            snapshot.views.push(view);
            let mut bindings = batch.bindings[resource.bindings.clone()].iter();
            for authority in *group {
                for segment in &authority.evidence.segments {
                    let binding = bindings.next()?;
                    let erased: Arc<dyn Send + Sync> = binding.chunk.clone();
                    snapshot.entries.push(Entry {
                        owner: OwnerIdentity {
                            lease: Arc::downgrade(&authority.segment_lease),
                            chunk: Arc::downgrade(&erased),
                            lane: authority.reusable_lane,
                            lifetime,
                        },
                        offset: segment.offset_bytes(),
                        length: segment.length_bytes(),
                        logical_size: authority.evidence.logical_size_bytes,
                        resource_view: group_index,
                    });
                }
            }
            if bindings.next().is_some() {
                return None;
            }
        }
        for window in windows.iter() {
            let window = window.ok()?;
            let view = snapshot.views.get_mut(window.resource_index)?;
            view.windows.try_reserve(1).ok()?;
            view.windows
                .push((window.offset_bytes, window.length_bytes));
        }
        for view in &mut snapshot.views {
            view.windows.sort_unstable();
            view.windows.dedup();
        }
        snapshot.owners = snapshot.index(0)?;
        snapshot.ranges = snapshot.index(1)?;
        snapshot.logical_windows = snapshot.index(2)?;
        Some(snapshot)
    }

    fn index(&self, kind: u8) -> Option<Vec<usize>> {
        let mut indices = Vec::new();
        indices.try_reserve_exact(self.entries.len()).ok()?;
        indices.extend(0..self.entries.len());
        indices.sort_unstable_by(|&a, &b| self.cmp_entry(a, self, b, kind));
        indices.dedup_by(|a, b| self.cmp_entry(*a, self, *b, kind) == Ordering::Equal);
        Some(indices)
    }
}
