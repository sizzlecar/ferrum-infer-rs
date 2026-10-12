//! Submission-local ownership for fresh segment ranges. Indices never identify
//! an owner outside the current core patch, and a table never grants a range.
use super::*;
use ferrum_interfaces::vnext::{
    ApprovedSequenceBindingPart, SegmentBindingOwnerBlock, SegmentBindingOwnerIndex,
    SegmentBindingOwnerSources, SegmentBindingOwnerView, SegmentBindingPageTemplate,
    SharedSequenceBindingOwnerBlock,
};

#[derive(Clone, Copy)]
pub(crate) struct CudaRegionView<'a> {
    pub(super) allocation: &'a Arc<CudaAllocation>,
    pub(super) retention: Option<&'a DeviceBufferRetention>,
    pub(super) runtime_instance: u64,
    pub(super) device_ptr: u64,
    pub(super) length_bytes: u64,
    pub(super) element_type: ElementType,
    pub(super) reusable_address_scope: Option<DeviceReusableAddressScope>,
}

impl CudaRegionView<'_> {
    pub(crate) fn runtime_instance(self) -> u64 {
        self.runtime_instance
    }

    pub(crate) fn device_ptr(self) -> u64 {
        self.device_ptr
    }
    pub(crate) fn length_bytes(self) -> u64 {
        self.length_bytes
    }
    pub(crate) fn element_type(self) -> ElementType {
        self.element_type
    }

    pub(crate) fn same_region(self, other: Self) -> bool {
        Arc::ptr_eq(self.allocation, other.allocation)
            && self.runtime_instance == other.runtime_instance
            && self.device_ptr == other.device_ptr
            && self.length_bytes == other.length_bytes
            && self.element_type == other.element_type
            && self.reusable_address_scope == other.reusable_address_scope
            && match (self.retention, other.retention) {
                (Some(a), Some(b)) => a.same_owners(b),
                (None, None) => true,
                _ => false,
            }
    }

    pub(crate) fn to_owned(self) -> CudaBufferRegion {
        CudaBufferRegion {
            _allocation: Arc::clone(self.allocation),
            _core_retention: self.retention.cloned(),
            reusable_address_scope: self.reusable_address_scope,
            runtime_instance: self.runtime_instance,
            device_ptr: self.device_ptr,
            length_bytes: self.length_bytes,
            element_type: self.element_type,
        }
    }
}

impl CudaBufferRegion {
    pub(crate) fn borrowed(&self) -> CudaRegionView<'_> {
        CudaRegionView {
            allocation: &self._allocation,
            retention: self._core_retention.as_ref(),
            runtime_instance: self.runtime_instance,
            device_ptr: self.device_ptr,
            length_bytes: self.length_bytes,
            element_type: self.element_type,
            reusable_address_scope: self.reusable_address_scope,
        }
    }
}

impl CudaDeviceBuffer {
    pub(crate) fn borrowed_region<'a>(
        &'a self,
        range: Range<u64>,
        retention: &'a DeviceBufferRetention,
    ) -> Result<CudaRegionView<'a>, CudaDeviceRuntimeError> {
        if range.start >= range.end
            || range.end > self.descriptor.size_bytes
            || range.end > self.allocation.requested_bytes
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA buffer region is empty or outside its admitted allocation",
            ));
        }
        let device_ptr = self
            .allocation
            .aligned_ptr
            .checked_add(range.start)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA buffer pointer overflow"))?;
        Ok(CudaRegionView {
            allocation: &self.allocation,
            retention: Some(retention),
            runtime_instance: self.runtime_instance,
            device_ptr,
            length_bytes: range.end - range.start,
            element_type: self.descriptor.element_type,
            reusable_address_scope: retention.reusable_address_scope(),
        })
    }
}

struct CudaSegmentOwner {
    allocation: Arc<CudaAllocation>,
    retention: DeviceBufferRetention,
    runtime_instance: u64,
    element_type: ElementType,
}

/// Private scalar projection of one already checked core parent range.
#[derive(Clone, Copy)]
pub(crate) struct CudaSegmentRange {
    owner_block: usize,
    owner_local: usize,
    byte_offset: u64,
    length_bytes: u64,
}

impl CudaSegmentRange {
    fn validate_capacity(self, capacity: u64) -> Result<(), CudaDeviceRuntimeError> {
        if self.length_bytes == 0
            || self
                .byte_offset
                .checked_add(self.length_bytes)
                .is_none_or(|end| end > capacity)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "segment range is empty or outside its retained allocation",
            ));
        }
        Ok(())
    }

    pub(crate) fn subrange(self, offset: u64, bytes: u64) -> Result<Self, CudaDeviceRuntimeError> {
        let byte_offset =
            checked_subregion_address(self.byte_offset, self.length_bytes, offset, bytes)?;
        Ok(Self {
            byte_offset,
            length_bytes: bytes,
            ..self
        })
    }
}

enum CudaSegmentOwnerBlock {
    Fresh(Vec<Option<CudaSegmentOwner>>),
    Sequence(SharedSequenceBindingOwnerBlock<CudaDeviceBuffer>),
}

pub(crate) struct CudaSegmentOwnerTable {
    blocks: Vec<CudaSegmentOwnerBlock>,
    ends: Vec<usize>,
}

impl CudaSegmentOwnerTable {
    fn location(&self, index: usize) -> Option<(usize, usize)> {
        let block = self.ends.partition_point(|end| *end <= index);
        let base = block
            .checked_sub(1)
            .map_or(0, |previous| self.ends[previous]);
        self.blocks.get(block)?;
        Some((block, index - base))
    }

    /// Validate against the actual retained owner without deriving an address.
    /// The range was issued from this patch's checked parent/current window.
    pub(crate) fn runtime_instance_for(
        &self,
        range: &CudaSegmentRange,
    ) -> Result<u64, CudaDeviceRuntimeError> {
        match self.blocks.get(range.owner_block) {
            Some(CudaSegmentOwnerBlock::Sequence(block)) => {
                let source = block.owner(range.owner_local).ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("shared segment range owner is absent")
                })?;
                let (buffer, _) = source.buffer_and_retention();
                range.validate_capacity(
                    buffer
                        .descriptor
                        .size_bytes
                        .min(buffer.allocation.requested_bytes),
                )?;
                Ok(buffer.runtime_instance)
            }
            Some(CudaSegmentOwnerBlock::Fresh(owners)) => {
                let owner = owners
                    .get(range.owner_local)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("segment range owner is absent")
                    })?;
                range.validate_capacity(owner.allocation.requested_bytes)?;
                Ok(owner.runtime_instance)
            }
            None => Err(CudaDeviceRuntimeError::contract(
                "segment range owner block is absent",
            )),
        }
    }

    pub(crate) fn region(
        &self,
        range: &CudaSegmentRange,
    ) -> Result<CudaRegionView<'_>, CudaDeviceRuntimeError> {
        let block = self.blocks.get(range.owner_block).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("segment range owner block is absent")
        })?;
        let owner = match block {
            CudaSegmentOwnerBlock::Sequence(block) => {
                let source = block.owner(range.owner_local).ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("shared segment range owner is absent")
                })?;
                let (buffer, retention) = source.buffer_and_retention();
                let end = range
                    .byte_offset
                    .checked_add(range.length_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("shared segment range end overflows")
                    })?;
                return buffer.borrowed_region(range.byte_offset..end, retention);
            }
            CudaSegmentOwnerBlock::Fresh(owners) => owners
                .get(range.owner_local)
                .and_then(Option::as_ref)
                .ok_or_else(|| CudaDeviceRuntimeError::contract("segment range owner is absent"))?,
        };
        let device_ptr = owner
            .allocation
            .aligned_ptr
            .checked_add(range.byte_offset)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("segment range address overflows"))?;
        Ok(CudaRegionView {
            allocation: &owner.allocation,
            retention: Some(&owner.retention),
            runtime_instance: owner.runtime_instance,
            device_ptr,
            length_bytes: range.length_bytes,
            element_type: owner.element_type,
            reusable_address_scope: owner.retention.reusable_address_scope(),
        })
    }
}

pub(crate) struct CudaSegmentOwnerBuilder<'a> {
    sources: Vec<Option<Vec<SegmentBindingOwnerView<'a, CudaDeviceBuffer>>>>,
    table: CudaSegmentOwnerTable,
}

/// Borrowed approval for this exact patch, physical part and current window.
/// The template supplies only sealed scalars; ownership stays in the table.
pub(crate) struct CudaApprovedSequencePart<'p, 'w> {
    _table: &'p CudaSegmentOwnerTable,
    part: &'p ApprovedSequenceBindingPart<'w, CudaDeviceBuffer>,
    template: &'p SegmentBindingPageTemplate,
}

impl CudaApprovedSequencePart<'_, '_> {
    pub(crate) fn range_for_template_page(
        &self,
        page_index: usize,
    ) -> Result<CudaSegmentRange, CudaDeviceRuntimeError> {
        let page =
            self.template.pages.get(page_index).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("shared template page is absent")
            })?;
        let logical_start = u64::try_from(page_index)
            .ok()
            .and_then(|index| index.checked_mul(self.template.key.physical_page_bytes))
            .ok_or_else(|| CudaDeviceRuntimeError::contract("shared page offset overflows"))?;
        let logical_end = logical_start
            .checked_add(page.length_bytes)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("shared page end overflows"))?;
        let window_end = self
            .part
            .logical_start_in_part()
            .checked_add(self.part.length_bytes())
            .ok_or_else(|| CudaDeviceRuntimeError::contract("approved part window overflows"))?;
        if page.length_bytes == 0
            || logical_start < self.part.logical_start_in_part()
            || logical_end > window_end
            || page.local_owner >= self.part.owner_count()
        {
            return Err(CudaDeviceRuntimeError::contract(
                "shared template page is outside the current approved part window",
            ));
        }
        let owner_local = self
            .part
            .owner_base()
            .checked_add(page.local_owner)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("shared page owner offset overflows")
            })?;
        Ok(CudaSegmentRange {
            owner_block: self.part.owner_block(),
            owner_local,
            byte_offset: page.physical_offset_bytes,
            length_bytes: page.length_bytes,
        })
    }
}

impl<'a> CudaSegmentOwnerBuilder<'a> {
    pub(crate) fn new(
        sources: SegmentBindingOwnerSources<'a, CudaDeviceBuffer>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let mut borrowed = Vec::new();
        let mut blocks = Vec::new();
        let mut ends = Vec::new();
        let mut count = 0usize;
        for source in sources.into_blocks() {
            let length = match &source {
                SegmentBindingOwnerBlock::Borrowed(views) => views.len(),
                SegmentBindingOwnerBlock::SharedSequence(block) => block.len(),
            };
            count = count
                .checked_add(length)
                .ok_or_else(|| CudaDeviceRuntimeError::contract("segment owner count overflows"))?;
            match source {
                SegmentBindingOwnerBlock::Borrowed(views) => {
                    let owners = (0..views.len()).map(|_| None).collect();
                    blocks.push(CudaSegmentOwnerBlock::Fresh(owners));
                    borrowed.push(Some(views));
                }
                SegmentBindingOwnerBlock::SharedSequence(block) => {
                    // Move the one real Core row owner into this submission.
                    // Individual buffers and exact lease retentions stay in it.
                    blocks.push(CudaSegmentOwnerBlock::Sequence(block));
                    borrowed.push(None);
                }
            }
            ends.push(count);
        }
        Ok(Self {
            sources: borrowed,
            table: CudaSegmentOwnerTable { blocks, ends },
        })
    }

    pub(crate) fn retain_shared_part<'p, 'w>(
        &'p self,
        part: &'p ApprovedSequenceBindingPart<'w, CudaDeviceBuffer>,
        template: &'p SegmentBindingPageTemplate,
    ) -> Result<CudaApprovedSequencePart<'p, 'w>, CudaDeviceRuntimeError> {
        let Some(CudaSegmentOwnerBlock::Sequence(block)) =
            self.table.blocks.get(part.owner_block())
        else {
            return Err(CudaDeviceRuntimeError::contract(
                "approved shared part is not in the current owner table",
            ));
        };
        if !block.same_owner_block(part.shared_block()) || !part.matches_page_template(template) {
            return Err(CudaDeviceRuntimeError::contract(
                "approved shared part or template differs from its exact current owner",
            ));
        }
        let first = part
            .owner(0)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("approved shared part has no owner"))?;
        let (buffer, _) = first.buffer_and_retention();
        if template.key.runtime_instance != buffer.runtime_instance {
            return Err(CudaDeviceRuntimeError::contract(
                "shared template belongs to another CUDA runtime",
            ));
        }
        Ok(CudaApprovedSequencePart {
            _table: &self.table,
            part,
            template,
        })
    }

    pub(crate) fn retain(
        &mut self,
        index: SegmentBindingOwnerIndex,
        buffer: &CudaDeviceBuffer,
        parent: Range<u64>,
        retention: &DeviceBufferRetention,
    ) -> Result<CudaSegmentRange, CudaDeviceRuntimeError> {
        let (owner_block, owner_local) = self.table.location(index.get()).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("segment owner index is outside current patch")
        })?;
        let source = match &self.table.blocks[owner_block] {
            CudaSegmentOwnerBlock::Sequence(block) => block.owner(owner_local),
            CudaSegmentOwnerBlock::Fresh(_) => self.sources[owner_block]
                .as_ref()
                .and_then(|views| views.get(owner_local))
                .copied(),
        }
        .ok_or_else(|| CudaDeviceRuntimeError::contract("segment owner source is absent"))?;
        let (current_buffer, current_retention) = source.buffer_and_retention();
        if !std::ptr::eq(current_buffer, buffer) || !current_retention.same_owners(retention) {
            return Err(CudaDeviceRuntimeError::contract(
                "segment range differs from its current owner",
            ));
        }
        let view = buffer.borrowed_region(parent.clone(), retention)?;
        if let CudaSegmentOwnerBlock::Fresh(owners) = &mut self.table.blocks[owner_block] {
            let slot = owners
                .get_mut(owner_local)
                .ok_or_else(|| CudaDeviceRuntimeError::contract("segment owner slot is absent"))?;
            if slot.is_none() {
                *slot = Some(CudaSegmentOwner {
                    allocation: Arc::clone(view.allocation),
                    retention: retention.clone(),
                    runtime_instance: view.runtime_instance,
                    element_type: view.element_type,
                });
            }
        }
        Ok(CudaSegmentRange {
            owner_block,
            owner_local,
            byte_offset: parent.start,
            length_bytes: view.length_bytes,
        })
    }

    pub(crate) fn region(
        &self,
        range: &CudaSegmentRange,
    ) -> Result<CudaRegionView<'_>, CudaDeviceRuntimeError> {
        self.table.region(range)
    }

    pub(crate) fn finish(self) -> Arc<CudaSegmentOwnerTable> {
        Arc::new(self.table)
    }
}

pub(super) enum CudaProgramBindingRetention {
    Legacy(Vec<CudaBufferRegion>),
    Indexed {
        owners: Arc<CudaSegmentOwnerTable>,
        ranges: Vec<CudaSegmentRange>,
    },
}

impl CudaProgramBindingRetention {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Legacy(regions) => regions.len(),
            Self::Indexed { ranges, .. } => ranges.len(),
        }
    }

    pub(super) fn region(&self, index: usize) -> Option<CudaRegionView<'_>> {
        match self {
            Self::Legacy(regions) => regions.get(index).map(CudaBufferRegion::borrowed),
            Self::Indexed { owners, ranges } => owners.region(ranges.get(index)?).ok(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_owner_locations_do_not_alias_across_empty_or_adjacent_blocks() {
        let table = CudaSegmentOwnerTable {
            blocks: vec![
                CudaSegmentOwnerBlock::Fresh(vec![None, None]),
                CudaSegmentOwnerBlock::Fresh(vec![]),
                CudaSegmentOwnerBlock::Fresh(vec![None]),
            ],
            ends: vec![2, 2, 3],
        };
        assert_eq!(table.location(0), Some((0, 0)));
        assert_eq!(table.location(1), Some((0, 1)));
        assert_eq!(table.location(2), Some((2, 0)));
        assert_eq!(table.location(3), None);
        assert_eq!(table.location(usize::MAX), None);
        let absent = CudaSegmentRange {
            owner_block: 2,
            owner_local: 0,
            byte_offset: 0,
            length_bytes: 1,
        };
        assert!(table.region(&absent).is_err());
        assert!(table.runtime_instance_for(&absent).is_err());
        assert!(table
            .region(&CudaSegmentRange {
                owner_block: 3,
                ..absent
            })
            .is_err());
        assert!(table
            .runtime_instance_for(&CudaSegmentRange {
                owner_block: 3,
                ..absent
            })
            .is_err());
    }

    #[test]
    fn segment_ranges_remain_inside_the_checked_parent_and_its_tail() {
        let parent = CudaSegmentRange {
            owner_block: 1,
            owner_local: 3,
            byte_offset: 4096,
            length_bytes: 96,
        };
        let tail = parent.subrange(64, 32).unwrap();
        assert!(parent.validate_capacity(4192).is_ok());
        assert!(tail.validate_capacity(4192).is_ok());
        assert!(parent.validate_capacity(4191).is_err());
        assert!(CudaSegmentRange {
            length_bytes: 0,
            ..parent
        }
        .validate_capacity(4192)
        .is_err());
        assert_eq!(
            (
                tail.owner_block,
                tail.owner_local,
                tail.byte_offset,
                tail.length_bytes
            ),
            (1, 3, 4160, 32)
        );
        assert!(parent.subrange(64, 33).is_err());
        assert!(parent.subrange(96, 1).is_err());
        assert!(parent.subrange(0, 0).is_err());
        assert!(tail.subrange(0, 33).is_err());
        assert!(parent.subrange(u64::MAX, 2).is_err());
        let overflowing = CudaSegmentRange {
            byte_offset: u64::MAX - 16,
            ..parent
        };
        assert!(overflowing.subrange(0, 1).is_err());
        assert!(overflowing.validate_capacity(u64::MAX).is_err());
    }
}
