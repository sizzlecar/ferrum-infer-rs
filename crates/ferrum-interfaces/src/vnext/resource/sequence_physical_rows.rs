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

/// A scalar template namespace, never an allocation/lease identity.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SegmentBindingPageAddressLayout {
    Flat,
    Blocked {
        combined_block_bytes: u64,
        blocks_per_page: u64,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SegmentBindingPageTemplateKey {
    pub physical_page_bytes: u64,
    pub element_type: super::ElementType,
    pub address_layout: SegmentBindingPageAddressLayout,
    pub runtime_instance: u64,
    pub provider_fingerprint: [u8; 32],
    pub numerical_profile: [u8; 32],
}

#[derive(Clone, Copy)]
pub struct SegmentBindingTemplatePage {
    pub local_owner: usize,
    pub physical_offset_bytes: u64,
    pub length_bytes: u64,
    pub address: u64,
}

/// Host-only derived scalars. This schema cannot own a buffer, lease, command,
/// snapshot or its enclosing physical part, preventing owner-cache cycles.
pub struct SegmentBindingPageTemplate {
    pub key: SegmentBindingPageTemplateKey,
    pub pages: Vec<SegmentBindingTemplatePage>,
    pub addresses: Vec<u64>,
    pub physical_bytes: u64,
}

/// Real ownership of a Core-validated immutable Sequence row. Retaining this
/// block preserves every independent lease/current allocation; it does not
/// authorize any new logical window or a later wave.
pub struct SharedSequenceBindingOwnerBlock<B> {
    row: Arc<ValidatedSequencePhysicalRow<B>>,
}

impl<B> Clone for SharedSequenceBindingOwnerBlock<B> {
    fn clone(&self) -> Self {
        Self {
            row: Arc::clone(&self.row),
        }
    }
}

impl<B> SharedSequenceBindingOwnerBlock<B> {
    pub(super) fn new(row: Arc<ValidatedSequencePhysicalRow<B>>) -> Self {
        Self { row }
    }
    pub fn len(&self) -> usize {
        self.row.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn same_owner_block(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.row, &other.row)
    }
    pub fn owner(&self, local: usize) -> Option<crate::vnext::SegmentBindingOwnerView<'_, B>> {
        self.row
            .get(local)
            .map(|binding| crate::vnext::SegmentBindingOwnerView {
                buffer: binding.buffer(),
                retention: binding.retention_ref(),
            })
    }
    pub(super) fn get(&self, local: usize) -> Option<&LogicalBackingSegmentBinding<B>> {
        self.row.get(local)
    }
    pub(super) fn part(&self, index: usize) -> Option<&[LogicalBackingSegmentBinding<B>]> {
        self.row.part(index)
    }
    pub(super) fn metadata(&self, index: usize) -> Option<(&B, &BufferDescriptor)> {
        self.row.metadata(index)
    }
    pub(super) fn current_window(
        &self,
        owner_block: usize,
        start: u64,
        length: u64,
        element_type: super::ElementType,
    ) -> CurrentSequenceBindingWindow<'_, B> {
        CurrentSequenceBindingWindow {
            block: self,
            owner_block,
            start,
            length,
            element_type,
        }
    }
}

/// Available only through this wave's successful physical/window permit.
/// The retained owner block alone cannot manufacture this approval.
pub struct CurrentSequenceBindingWindow<'a, B> {
    block: &'a SharedSequenceBindingOwnerBlock<B>,
    owner_block: usize,
    start: u64,
    length: u64,
    element_type: super::ElementType,
}

impl<'a, B> CurrentSequenceBindingWindow<'a, B> {
    pub fn shared_block(&self) -> &'a SharedSequenceBindingOwnerBlock<B> {
        self.block
    }
    pub fn owner_block(&self) -> usize {
        self.owner_block
    }
    pub fn length_bytes(&self) -> u64 {
        self.length
    }
    pub fn logical_offset_bytes(&self) -> u64 {
        self.start
    }
    pub fn element_type(&self) -> super::ElementType {
        self.element_type
    }
    pub fn parts(&self) -> impl Iterator<Item = ApprovedSequenceBindingPart<'a, B>> + 'a {
        let block = self.block;
        let start = self.start;
        let end = start + self.length;
        let owner_block = self.owner_block;
        let element_type = self.element_type;
        block
            .row
            .parts
            .iter()
            .scan((0usize, 0u64), move |(owners, logical), part| {
                let owner_base = *owners;
                let origin = *logical;
                *owners += part.bindings.len();
                *logical += part.physical_bytes;
                Some((part, owner_base, origin))
            })
            .filter_map(move |(part, owner_base, origin)| {
                let begin = start.max(origin);
                let finish = end.min(origin + part.physical_bytes);
                (begin < finish).then(|| ApprovedSequenceBindingPart {
                    part,
                    block,
                    owner_base,
                    owner_block,
                    element_type,
                    logical_start_in_part: begin - origin,
                    length: finish - begin,
                    logical_offset_in_window: begin - start,
                })
            })
    }
}

pub struct ApprovedSequenceBindingPart<'a, B> {
    part: &'a Arc<SequenceBindingPart<B>>,
    block: &'a SharedSequenceBindingOwnerBlock<B>,
    owner_base: usize,
    owner_block: usize,
    element_type: super::ElementType,
    logical_start_in_part: u64,
    length: u64,
    logical_offset_in_window: u64,
}

impl<B> ApprovedSequenceBindingPart<'_, B> {
    pub fn shared_block(&self) -> &SharedSequenceBindingOwnerBlock<B> {
        self.block
    }
    pub fn owner_base(&self) -> usize {
        self.owner_base
    }
    pub fn owner_block(&self) -> usize {
        self.owner_block
    }
    pub fn logical_start_in_part(&self) -> u64 {
        self.logical_start_in_part
    }
    pub fn length_bytes(&self) -> u64 {
        self.length
    }
    pub fn logical_offset_in_window(&self) -> u64 {
        self.logical_offset_in_window
    }
    pub fn physical_bytes(&self) -> u64 {
        self.part.physical_bytes
    }
    pub fn owner_count(&self) -> usize {
        self.part.bindings.len()
    }
    pub fn owner(&self, local: usize) -> Option<crate::vnext::SegmentBindingOwnerView<'_, B>> {
        self.part
            .bindings
            .get(local)
            .map(|binding| crate::vnext::SegmentBindingOwnerView {
                buffer: binding.buffer(),
                retention: binding.retention_ref(),
            })
    }
    pub fn physical_range(&self, local: usize) -> Option<Range<u64>> {
        self.part.bindings.get(local).map(|binding| {
            binding.segment.offset_bytes()
                ..binding.segment.offset_bytes() + binding.segment.length_bytes()
        })
    }
    pub fn page_template(
        &self,
        key: &SegmentBindingPageTemplateKey,
    ) -> Option<Arc<SegmentBindingPageTemplate>> {
        self.part
            .page_template
            .get()
            .filter(|template| template.key == *key)
            .map(Arc::clone)
    }
    pub fn same_page_template(&self, template: &Arc<SegmentBindingPageTemplate>) -> bool {
        self.part
            .page_template
            .get()
            .is_some_and(|stored| Arc::ptr_eq(stored, template))
    }
    pub fn matches_page_template(&self, template: &SegmentBindingPageTemplate) -> bool {
        self.part
            .page_template
            .get()
            .is_some_and(|stored| std::ptr::eq(stored.as_ref(), template))
    }

    /// Checks the scalar projection's complete geometry before publication.
    /// Actual native address derivation is checked by the backend when cold;
    /// memo presence never authorizes its addresses for the current window.
    pub fn publish_page_template(&self, template: Arc<SegmentBindingPageTemplate>) -> bool {
        let key = template.key;
        if key.physical_page_bytes == 0
            || key.element_type != self.element_type
            || template.physical_bytes != self.part.physical_bytes
        {
            return false;
        }
        let per_page = match key.address_layout {
            SegmentBindingPageAddressLayout::Flat => 1,
            SegmentBindingPageAddressLayout::Blocked {
                combined_block_bytes,
                blocks_per_page,
            } => {
                if combined_block_bytes == 0
                    || blocks_per_page == 0
                    || combined_block_bytes
                        .checked_mul(blocks_per_page)
                        .is_none_or(|bytes| bytes > key.physical_page_bytes)
                {
                    return false;
                }
                blocks_per_page
            }
        };
        let Some(per_page) = usize::try_from(per_page).ok() else {
            return false;
        };
        if template.pages.len().checked_mul(per_page) != Some(template.addresses.len()) {
            return false;
        }
        let mut page_index = 0usize;
        for (owner, binding) in self.part.bindings.iter().enumerate() {
            if binding.segment.length_bytes() % key.physical_page_bytes != 0 {
                return false;
            }
            let mut offset = 0u64;
            while offset < binding.segment.length_bytes() {
                let Some(page) = template.pages.get(page_index) else {
                    return false;
                };
                let Some(physical) = binding.segment.offset_bytes().checked_add(offset) else {
                    return false;
                };
                if page.local_owner != owner
                    || page.physical_offset_bytes != physical
                    || page.length_bytes != key.physical_page_bytes
                    || page.address == 0
                    || page.address.checked_add(page.length_bytes).is_none()
                {
                    return false;
                }
                for entry in 0..per_page {
                    let delta = match key.address_layout {
                        SegmentBindingPageAddressLayout::Flat => 0,
                        SegmentBindingPageAddressLayout::Blocked {
                            combined_block_bytes,
                            ..
                        } => {
                            let Some(delta) = (entry as u64).checked_mul(combined_block_bytes)
                            else {
                                return false;
                            };
                            delta
                        }
                    };
                    if page.address.checked_add(delta)
                        != template
                            .addresses
                            .get(page_index * per_page + entry)
                            .copied()
                    {
                        return false;
                    }
                }
                page_index += 1;
                offset += key.physical_page_bytes;
            }
        }
        if page_index != template.pages.len() {
            return false;
        }
        self.part.page_template.set(template).is_ok()
    }
}

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
    physical_bytes: u64,
    page_template: OnceLock<Arc<SegmentBindingPageTemplate>>,
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
            let physical_bytes = bindings.iter().fold(0u64, |bytes, binding| {
                bytes
                    .checked_add(binding.segment.length_bytes())
                    .expect("validated row capacity fits u64")
            });
            parts.push(Arc::new(SequenceBindingPart {
                bindings,
                physical_bytes,
                page_template: OnceLock::new(),
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
