//! Submission-local ownership for fresh segment ranges. Indices never identify
//! an owner outside the current core patch, and a table never grants a range.
use super::*;
use ferrum_interfaces::vnext::{SegmentBindingOwnerIndex, SegmentBindingOwnerView};

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
    owner_index: usize,
    byte_offset: u64,
    length_bytes: u64,
}

impl CudaSegmentRange {
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

pub(crate) struct CudaSegmentOwnerTable {
    owners: Vec<Option<CudaSegmentOwner>>,
}

impl CudaSegmentOwnerTable {
    pub(crate) fn region(
        &self,
        range: &CudaSegmentRange,
    ) -> Result<CudaRegionView<'_>, CudaDeviceRuntimeError> {
        let owner = self
            .owners
            .get(range.owner_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("segment range owner is absent"))?;
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
    sources: Vec<SegmentBindingOwnerView<'a, CudaDeviceBuffer>>,
    table: CudaSegmentOwnerTable,
}

impl<'a> CudaSegmentOwnerBuilder<'a> {
    pub(crate) fn new(sources: Vec<SegmentBindingOwnerView<'a, CudaDeviceBuffer>>) -> Self {
        let owners = (0..sources.len()).map(|_| None).collect();
        Self {
            sources,
            table: CudaSegmentOwnerTable { owners },
        }
    }

    pub(crate) fn retain(
        &mut self,
        index: SegmentBindingOwnerIndex,
        buffer: &CudaDeviceBuffer,
        parent: Range<u64>,
        retention: &DeviceBufferRetention,
    ) -> Result<CudaSegmentRange, CudaDeviceRuntimeError> {
        let source = self.sources.get(index.get()).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("segment owner index is outside current patch")
        })?;
        let (current_buffer, current_retention) = source.buffer_and_retention();
        if !std::ptr::eq(current_buffer, buffer) || !current_retention.same_owners(retention) {
            return Err(CudaDeviceRuntimeError::contract(
                "segment range differs from its current owner",
            ));
        }
        let view = buffer.borrowed_region(parent.clone(), retention)?;
        let slot = self
            .table
            .owners
            .get_mut(index.get())
            .ok_or_else(|| CudaDeviceRuntimeError::contract("segment owner slot is absent"))?;
        if slot.is_none() {
            *slot = Some(CudaSegmentOwner {
                allocation: Arc::clone(view.allocation),
                retention: retention.clone(),
                runtime_instance: view.runtime_instance,
                element_type: view.element_type,
            });
        }
        Ok(CudaSegmentRange {
            owner_index: index.get(),
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
    fn segment_ranges_remain_inside_the_checked_parent_and_its_tail() {
        let parent = CudaSegmentRange {
            owner_index: 3,
            byte_offset: 4096,
            length_bytes: 96,
        };
        let tail = parent.subrange(64, 32).unwrap();
        assert_eq!(
            (tail.owner_index, tail.byte_offset, tail.length_bytes),
            (3, 4160, 32)
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
    }
}
