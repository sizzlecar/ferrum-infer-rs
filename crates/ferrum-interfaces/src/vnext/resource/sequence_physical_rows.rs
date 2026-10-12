//! Private physical rows owned by the exact immutable Sequence snapshot.
//! Cache presence is not authority: the permit still observes current pools
//! and chunk Arcs, and publishes only after every current window succeeds.
//! Sources come only from immutable private snapshot producers. Physical
//! append inheritance checks actual allocation/lease/lane/logical identity;
//! there is no supported mutation of a published snapshot's old authorities.
use super::{
    Arc, BTreeMap, BufferDescriptor, DynamicBackingPoolState, LogicalBackingSegmentBinding,
    LogicalBackingSliceAllocationEvidence, LogicalBackingSliceAuthority, OnceLock,
    ResidentChunkBacking, ResourceId,
};
use std::ops::Range;

pub(super) struct SequencePhysicalRows<B> {
    rows: BTreeMap<ResourceId, PhysicalRowEntry<B>>,
}

struct PhysicalRowEntry<B> {
    range: Range<usize>,
    cache: Arc<PhysicalRowCache<B>>,
}

pub(super) struct PhysicalRowCache<B> {
    pub(super) authority_count: usize,
    segment_count: Option<usize>,
    previous: Option<Arc<ValidatedSequencePhysicalRow<B>>>,
    validated: OnceLock<Arc<ValidatedSequencePhysicalRow<B>>>,
}

/// Fields are private so a cache cannot be paired with caller-chosen authority.
pub(crate) struct SequencePhysicalRowSource<'a, B> {
    authorities: &'a [LogicalBackingSliceAuthority],
    cache: &'a PhysicalRowCache<B>,
}

impl<'a, B> SequencePhysicalRowSource<'a, B> {
    pub(super) fn authorities(&self) -> &'a [LogicalBackingSliceAuthority] {
        self.authorities
    }

    pub(super) fn cached_prefix(&self) -> Option<Arc<ValidatedSequencePhysicalRow<B>>> {
        self.cache
            .validated
            .get()
            .or(self.cache.previous.as_ref())
            .map(Arc::clone)
    }

    pub(super) fn segment_count(&self) -> Option<usize> {
        self.cache.segment_count
    }

    pub(super) fn publish(&self, row: Arc<ValidatedSequencePhysicalRow<B>>) {
        // Losing concurrent publication only drops extra owners outside guards.
        let _ = self.cache.validated.set(row);
    }

    #[cfg(test)]
    pub(crate) fn cached_binding_count(&self) -> Option<usize> {
        self.cache.validated.get().map(|row| row.len())
    }
}

impl<B> SequencePhysicalRows<B> {
    pub(super) fn initial(slices: &[LogicalBackingSliceAuthority]) -> Self {
        Self::build(slices, None)
    }

    pub(super) fn appended(
        prior: &Arc<Self>,
        prior_slices: &[LogicalBackingSliceAuthority],
        slices: &[LogicalBackingSliceAuthority],
    ) -> Self {
        Self::build(slices, Some((prior, prior_slices)))
    }

    fn build(
        slices: &[LogicalBackingSliceAuthority],
        prior: Option<(&Arc<Self>, &[LogicalBackingSliceAuthority])>,
    ) -> Self {
        let mut rows = BTreeMap::new();
        let mut start = 0;
        while start < slices.len() {
            let resource = slices[start].resource_id();
            let end =
                start + slices[start..].partition_point(|slice| slice.resource_id() == resource);
            let previous = prior.and_then(|(frame, old_slices)| {
                let entry = frame.rows.get(resource)?;
                let old = &old_slices[entry.range.clone()];
                let current = &slices[start..end];
                // This cold publication check binds the inherited prefix to
                // the actual retained immutable allocations and exact leases.
                (current.len() >= old.len()
                    && old.iter().zip(current).all(|(a, b)| {
                        Arc::ptr_eq(&a.evidence.allocation, &b.evidence.allocation)
                            && Arc::ptr_eq(&a.segment_lease, &b.segment_lease)
                            && a.reusable_lane == b.reusable_lane
                            && a.evidence.logical_size_bytes == b.evidence.logical_size_bytes
                    }))
                .then(|| Arc::clone(&entry.cache))
            });
            let cache = if let Some(previous) = previous
                .as_ref()
                .filter(|old| old.authority_count == end - start)
            {
                Arc::clone(previous)
            } else {
                Arc::new(PhysicalRowCache {
                    authority_count: end - start,
                    segment_count: slices[start..end].iter().try_fold(0usize, |count, slice| {
                        count.checked_add(slice.evidence.segments.len())
                    }),
                    previous: previous.and_then(|old| {
                        old.validated
                            .get()
                            .or(old.previous.as_ref())
                            .map(Arc::clone)
                    }),
                    validated: OnceLock::new(),
                })
            };
            rows.insert(
                resource.clone(),
                PhysicalRowEntry {
                    range: start..end,
                    cache,
                },
            );
            start = end;
        }
        Self { rows }
    }

    pub(super) fn source<'a>(
        &'a self,
        slices: &'a [LogicalBackingSliceAuthority],
        resource: &ResourceId,
    ) -> Option<SequencePhysicalRowSource<'a, B>> {
        let entry = self.rows.get(resource)?;
        Some(SequencePhysicalRowSource {
            authorities: slices.get(entry.range.clone())?,
            cache: &entry.cache,
        })
    }
}

pub(super) struct SequenceBindingPart<B> {
    pub(super) bindings: Vec<LogicalBackingSegmentBinding<B>>,
    // Keep immutable allocation evidence shared: Arc::get_mut cannot alter
    // certified facts after publication while a row remains cached.
    _allocations: Vec<Arc<LogicalBackingSliceAllocationEvidence>>,
}

pub(super) struct ValidatedSequencePhysicalRow<B> {
    pub(super) authority_count: usize,
    pub(super) logical: u64,
    pub(super) capacity: u64,
    pub(super) last_complete: bool,
    parts: Vec<Arc<SequenceBindingPart<B>>>,
    positions: Vec<(usize, usize)>,
    chunks: Vec<Arc<ResidentChunkBacking<B>>>,
}

impl<B> ValidatedSequencePhysicalRow<B> {
    pub(super) fn new(
        authority_count: usize,
        prefix: Option<&Self>,
        bindings: Vec<LogicalBackingSegmentBinding<B>>,
        allocations: Vec<Arc<LogicalBackingSliceAllocationEvidence>>,
        logical: u64,
        capacity: u64,
        last_complete: bool,
    ) -> Self {
        let mut parts = prefix.map_or_else(Vec::new, |row| row.parts.clone());
        if !bindings.is_empty() {
            parts.push(Arc::new(SequenceBindingPart {
                bindings,
                _allocations: allocations,
            }));
        }
        let mut positions = prefix.map_or_else(Vec::new, |row| row.positions.clone());
        let mut unique = BTreeMap::new();
        if let Some(prefix) = prefix {
            for chunk in &prefix.chunks {
                unique.insert(chunk.identity.ordinal(), Arc::clone(chunk));
            }
        }
        // Only new physical parts are visited; old exact owners are shared.
        if let Some(tail) = parts.get(prefix.map_or(0, |row| row.parts.len())) {
            let part_index = prefix.map_or(0, |row| row.parts.len());
            positions.extend((0..tail.bindings.len()).map(|local| (part_index, local)));
            for binding in &tail.bindings {
                unique
                    .entry(binding.chunk.identity.ordinal())
                    .or_insert_with(|| Arc::clone(&binding.chunk));
            }
        }
        Self {
            authority_count,
            logical,
            capacity,
            last_complete,
            parts,
            positions,
            chunks: unique.into_values().collect(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.positions.len()
    }

    pub(super) fn get(&self, index: usize) -> Option<&LogicalBackingSegmentBinding<B>> {
        let (part, local) = *self.positions.get(index)?;
        self.parts.get(part)?.bindings.get(local)
    }

    pub(super) fn part(&self, index: usize) -> Option<&[LogicalBackingSegmentBinding<B>]> {
        self.parts.get(index).map(|part| part.bindings.as_slice())
    }

    pub(super) fn current(&self, state: &DynamicBackingPoolState<B>) -> bool {
        self.chunks.iter().all(|expected| {
            state
                .chunks
                .get(&expected.identity.ordinal())
                .is_some_and(|current| {
                    Arc::ptr_eq(&current.backing, expected)
                        && current.backing.identity == expected.identity
                        && current.backing.descriptor == expected.descriptor
                })
        })
    }

    pub(super) fn metadata(&self, index: usize) -> Option<(&B, &BufferDescriptor)> {
        self.chunks
            .get(index)
            .map(|chunk| (&chunk.buffer, &chunk.descriptor))
    }
}
