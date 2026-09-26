//! Borrowed results of fresh logical-backing inspection. These contain no
//! physical buffer, retention or executable authority, and do not cross waves.
use super::{
    BackingSegment, BufferUsage, DynamicStorageProfile, ElementType, LogicalBackingSliceAuthority,
    VNextError,
};

pub(super) struct LogicalBackingMetadata<'a> {
    authorities: &'a [LogicalBackingSliceAuthority],
    logical_bytes: u64,
    capacity_bytes: u64,
    segment_count: usize,
}

impl<'a> LogicalBackingMetadata<'a> {
    pub(super) fn new(
        authorities: &'a [LogicalBackingSliceAuthority],
        logical_bytes: u64,
        capacity_bytes: u64,
        segment_count: usize,
    ) -> Self {
        Self {
            authorities,
            logical_bytes,
            capacity_bytes,
            segment_count,
        }
    }

    pub(crate) fn size_bytes(&self) -> u64 {
        self.logical_bytes
    }
    pub(crate) fn capacity_size_bytes(&self) -> u64 {
        self.capacity_bytes
    }
    pub(crate) fn alignment_bytes(&self) -> u64 {
        self.authorities[0].evidence().alignment_bytes()
    }
    pub(crate) fn usage(&self) -> BufferUsage {
        self.authorities[0].evidence().usage()
    }
    pub(crate) fn element_type(&self) -> ElementType {
        self.authorities[0].evidence().element_type()
    }
    pub(crate) fn storage_profile(&self) -> DynamicStorageProfile {
        self.authorities[0].evidence().storage_profile()
    }
    pub(crate) fn segment_count(&self) -> usize {
        self.segment_count
    }
    pub(crate) fn segments(&self) -> impl Iterator<Item = &BackingSegment> {
        self.authorities
            .iter()
            .flat_map(|authority| authority.evidence().segments())
    }
    pub(crate) fn physical_coverage_bytes(&self) -> Result<u64, VNextError> {
        self.segments().try_fold(0_u64, |sum, segment| {
            sum.checked_add(segment.length_bytes())
                .ok_or_else(|| super::invalid_resource("backing segment coverage overflows u64"))
        })
    }
}

/// Issued only after the shared pool/authority/chunk visitor also rereads each
/// runtime buffer descriptor. The lifetime borrows the original authorities;
/// dropping the caller's prepared wave releases everything as before.
pub(crate) struct ValidatedLogicalBacking<'a> {
    metadata: LogicalBackingMetadata<'a>,
}

impl<'a> ValidatedLogicalBacking<'a> {
    pub(super) fn from_runtime_checked(metadata: LogicalBackingMetadata<'a>) -> Self {
        Self { metadata }
    }
    pub(crate) fn size_bytes(&self) -> u64 {
        self.metadata.size_bytes()
    }
    pub(crate) fn capacity_size_bytes(&self) -> u64 {
        self.metadata.capacity_size_bytes()
    }
    pub(crate) fn alignment_bytes(&self) -> u64 {
        self.metadata.alignment_bytes()
    }
    pub(crate) fn usage(&self) -> BufferUsage {
        self.metadata.usage()
    }
    pub(crate) fn element_type(&self) -> ElementType {
        self.metadata.element_type()
    }
    pub(crate) fn storage_profile(&self) -> DynamicStorageProfile {
        self.metadata.storage_profile()
    }
    pub(crate) fn segment_count(&self) -> usize {
        self.metadata.segment_count()
    }
    pub(crate) fn physical_coverage_bytes(&self) -> Result<u64, VNextError> {
        self.metadata.physical_coverage_bytes()
    }
}
