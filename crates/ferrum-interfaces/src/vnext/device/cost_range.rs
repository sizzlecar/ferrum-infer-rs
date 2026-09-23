//! Read-only numeric addresses. These values neither retain storage nor grant
//! permission to access a device pointer. They expire with their planning view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCostBufferRange {
    start: u64,
    length: u64,
}

impl DeviceCostBufferRange {
    pub fn new(start: u64, length: u64) -> Option<Self> {
        (start != 0 && length != 0 && start.checked_add(length).is_some())
            .then_some(Self { start, length })
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
        Self::new(self.start.checked_add(offset)?, length)
    }

    pub fn overlaps(self, other: Self) -> bool {
        self.start < other.end() && other.start < self.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
