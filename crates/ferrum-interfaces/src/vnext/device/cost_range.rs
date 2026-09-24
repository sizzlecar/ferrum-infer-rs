//! Read-only numeric addresses. These values neither retain storage nor grant
//! permission to access a device pointer. They expire with their planning view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCostBufferRange {
    allocation: Option<u64>,
    start: u64,
    length: u64,
}

impl DeviceCostBufferRange {
    pub fn new(start: u64, length: u64) -> Option<Self> {
        (start != 0 && length != 0 && start.checked_add(length).is_some()).then_some(Self {
            allocation: None,
            start,
            length,
        })
    }

    /// Opaque retained-allocation identity plus an offset, never a pointer grant.
    /// Zero offset is legal; the identity is valid only inside its live snapshot.
    pub fn in_allocation(allocation: u64, offset: u64, length: u64) -> Option<Self> {
        (allocation != 0 && length != 0 && offset.checked_add(length).is_some()).then_some(Self {
            allocation: Some(allocation),
            start: offset,
            length,
        })
    }
    pub const fn allocation_id(self) -> Option<u64> {
        self.allocation
    }
    pub const fn start(self) -> u64 {
        self.start
    }
    pub const fn length(self) -> u64 {
        self.length
    }
    pub fn end(self) -> u64 {
        self.start + self.length
    }

    pub fn slice(self, offset: u64, length: u64) -> Option<Self> {
        let end = offset.checked_add(length)?;
        if length == 0 || end > self.length {
            return None;
        }
        Some(Self {
            allocation: self.allocation,
            start: self.start.checked_add(offset)?,
            length,
        })
    }

    pub fn overlaps(self, other: Self) -> bool {
        match (self.allocation, other.allocation) {
            (Some(a), Some(b)) => a == b && self.start < other.end() && other.start < self.end(),
            (None, None) => self.start < other.end() && other.start < self.end(),
            // Two different identity domains carry no mutual disjointness proof.
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_ranges_preserve_namespace_zero_offset_and_boundaries() {
        let a = DeviceCostBufferRange::in_allocation(7, 0, 256).unwrap();
        let b = DeviceCostBufferRange::in_allocation(8, 0, 256).unwrap();
        assert!(!a.overlaps(b));
        let tail = a.slice(128, 128).unwrap();
        assert_eq!(tail.allocation_id(), Some(7));
        assert_eq!(tail.start(), 128);
        assert!(!tail.overlaps(a.slice(0, 128).unwrap()));
        assert!(tail.overlaps(a.slice(127, 2).unwrap()));
        assert!(a.overlaps(DeviceCostBufferRange::new(4096, 256).unwrap()));
        assert!(DeviceCostBufferRange::in_allocation(0, 0, 1).is_none());
        assert!(DeviceCostBufferRange::in_allocation(1, u64::MAX, 1).is_none());
        assert!(a.slice(256, 1).is_none());
    }
    #[test]
    fn numeric_range_rejects_zero_overflow_and_escape_without_rounding() {
        assert!(DeviceCostBufferRange::new(0, 8).is_none());
        assert!(DeviceCostBufferRange::new(8, 0).is_none());
        assert!(DeviceCostBufferRange::new(u64::MAX, 1).is_none());
        let range = DeviceCostBufferRange::new(24, 32).unwrap();
        assert_eq!(range.slice(8, 24).unwrap().end(), 56);
        assert!(range.slice(8, 25).is_none());
        assert!(!range.overlaps(DeviceCostBufferRange::new(56, 8).unwrap()));
        assert!(range.overlaps(DeviceCostBufferRange::new(55, 8).unwrap()));
    }
}
